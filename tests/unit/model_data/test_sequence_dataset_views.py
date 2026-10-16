import numpy as np
import torch

from model.CL4KT.CL4KT_data import CL4KTTrainDataset
from model.GKT.GKT_data import GKTDataset
from model.GKT.GKT_trainer import GKTTrainer
from utils.model_data.packed_sequences import SequencePartition


def test_augmentation_seed_follows_sequence_identity_when_rows_are_reordered() -> None:
    skills = np.array([[1, 2, 3, 1, 2, 0], [3, 2, 1, 2, 3, 0]])
    responses = np.array([[0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0]])
    masks = np.array([[1, 1, 1, 1, 1, 0]] * 2)
    identities = np.array([41, 9])
    probabilities = {
        "mask_prob": 0.2,
        "crop_prob": 0.8,
        "permute_prob": 0.4,
        "replace_prob": 0.0,
        "negative_prob": 0.5,
    }
    first = CL4KTTrainDataset(
        skills, responses, masks, 6, 4, probabilities, {}, {}, identities
    )
    reordered = CL4KTTrainDataset(
        skills[::-1],
        responses[::-1],
        masks[::-1],
        6,
        4,
        probabilities,
        {},
        {},
        identities[::-1],
    )
    for original, moved in zip(first[0], reordered[1], strict=True):
        for a, b in zip(original, moved, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_transition_graph_uses_only_selected_training_sequences() -> None:
    sequences = np.array([[0, 1, 1], [1, 2, 2], [2, 0, 0]])
    dataset = GKTDataset(sequences, np.zeros_like(sequences), np.ones_like(sequences))
    train, _, _ = SequencePartition.from_folds(np.array([1, 0, -1]), 0, 2).apply(
        dataset
    )
    trainer = GKTTrainer.__new__(GKTTrainer)
    from_view = trainer._build_graph("transition", train, 3)
    from_selected = trainer._build_graph(
        "transition",
        GKTDataset(sequences[:1], np.zeros((1, 3)), np.ones((1, 3))),
        3,
    )
    torch.testing.assert_close(from_view, from_selected, rtol=0, atol=0)
