from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F

from model.DyGKT.DyGKT_data import (
    DyGKTDataset,
    DyGKTModelData,
    DyGKTNeighborConfig,
    build_tensors,
)
from model.DyGKT.DyGKT_model import DyGKT
from model.DyGKT.DyGKT_trainer import DyGKTConfig, DyGKTTrainer
from utils.data_process import DataSource

_ABLATIONS = ("-1", "counter", "dual", "q_qid", "q_kid", "embed", "skill", "time")


class _Source:
    dataset = "dygkt_events"

    def __init__(self, swapped: bool = False) -> None:
        skills = [[0], [1], [0, 2], [2], [1], [0], [2], [2], [1]]
        if swapped:
            skills[0], skills[1] = skills[1], skills[0]
        self.data = pl.DataFrame(
            {
                "sequence_id": [0] * 4 + [1] * 3 + [2] * 2,
                "seq_pos": [0, 1, 2, 3, 0, 1, 2, 0, 1],
                "user": [0] * 4 + [1] * 3 + [2] * 2,
                "question": [0, 0, 0, 1, 0, 0, 1, 0, 1],
                "interaction_id": list(range(100, 109)),
                "label": [1, 0, 1, 0, 1, 0, 1, 0, 1],
                "timestamp": [10_000 * (index + 1) for index in range(9)],
                "fold": [1] * 4 + [0] * 3 + [-1] * 2,
                "skills": pl.Series(skills, dtype=pl.List(pl.Int32)),
            }
        )

    def get_split_question_sequence_data(self, **kwargs: Any) -> pl.DataFrame:
        return self.data

    def get_metadata(self, key: str) -> int:
        return {
            "max_question_seq_len": 5,
            "num_questions": 2,
            "num_skills": 3,
            "num_users": 3,
            "kfold_n_splits": 2,
        }[key]

    def get_relation(self, name: str) -> pl.DataFrame:
        assert name == "question_skill"
        return (
            self.data.select("question", pl.col("skills").alias("skill"))
            .explode("skill", empty_as_null=False)
            .unique()
            .sort("question", "skill")
        )


def _config(**overrides: Any) -> SimpleNamespace:
    model = DyGKTConfig()
    model.num_neighbor = 4
    model.neighbor_sampling_strategy = "recent"
    model.edge_dim = 8
    model.node_dim = 8
    model.dim_time = 3
    model.dropout = 0.0
    model.graph_neg_num_samples = 1
    for name, value in overrides.items():
        setattr(model, name, value)
    return SimpleNamespace(
        data=SimpleNamespace(fold=0),
        general=SimpleNamespace(device="cpu"),
        model=model,
    )


def _prepare(
    swapped: bool = False, **overrides: Any
) -> tuple[DyGKTDataset, DyGKTDataset, DyGKTDataset, dict[str, Any]]:
    return DyGKTModelData(cast(DataSource, _Source(swapped))).prepare_data(
        _config(**overrides)
    )


def _model(metadata: dict[str, Any], ablation: str = "-1") -> DyGKT:
    return DyGKT(
        metadata,
        num_neighbor=4,
        ablation=ablation,
        edge_dim=8,
        node_dim=8,
        dim_time=3,
        dropout=0.0,
    )


def test_repeated_questions_keep_target_and_history_event_skills() -> None:
    original = _prepare()
    swapped = _prepare(swapped=True)
    assert (
        _Source()
        .get_relation("question_skill")
        .equals(_Source(swapped=True).get_relation("question_skill"))
    )
    np.testing.assert_array_equal(
        original[3]["question_features"], swapped[3]["question_features"]
    )
    assert "question_skill_ids" not in original[3]
    assert original[3]["num_skills"] == 3

    expected_skills = ([1, 2, 1, 3], [2, 1, 3], [3, 2])
    for index, (before, after) in enumerate(zip(original[:3], swapped[:3])):
        batch = before.get_batch(list(range(len(before))))
        changed = after.get_batch(list(range(len(after))))
        assert batch["skill"].tolist() == expected_skills[index]
        assert torch.all(batch["skill"] > 0)
        assert "dst_neighbor_skill_ids" not in batch
        torch.testing.assert_close(batch["question"], changed["question"])
        for prefix, history_key in (("src", "user_his_idx"), ("dst", "que_his_idx")):
            assert f"{prefix}_neighbor_event_ids" not in batch
            event_ids = before.tensors[history_key][before.target_indices].long()
            changed_event_ids = after.tensors[history_key][after.target_indices].long()
            torch.testing.assert_close(event_ids, changed_event_ids)
            if prefix == "src":
                torch.testing.assert_close(
                    batch["src_neighbor_skill_ids"],
                    before.tensors["lookup_skill"][event_ids],
                )
                assert torch.all(batch["src_neighbor_skill_ids"][event_ids == 0] == 0)

    train = original[0].get_batch([0, 1, 2])
    changed_train = swapped[0].get_batch([0, 1, 2])
    assert train["question"].tolist() == [1, 1, 1]
    assert train["skill"].tolist() == [1, 2, 1]
    assert changed_train["skill"].tolist() == [2, 1, 1]
    assert original[0].tensors["user_his_idx"][
        original[0].target_indices[2]
    ].tolist() == [1, 2, 0, 0]
    assert train["src_neighbor_skill_ids"][2].tolist() == [1, 2, 0, 0]
    assert changed_train["src_neighbor_skill_ids"][2].tolist() == [2, 1, 0, 0]
    assert original[1].tensors["que_his_idx"][
        original[1].target_indices[0]
    ].tolist() == [1, 2, 3, 0]


@pytest.mark.parametrize("strategy", ["recent", "time_decay"])
def test_sampled_histories_lookup_skills_by_event_identity(strategy: str) -> None:
    records: list[dict[str, Any]] = [
        {
            "user_id": 0,
            "seq_len": 8,
            "question_seq": [0] * 8,
            "skill_seq": [1, 2, 3, 1, 2, 3, 1, 2],
            "correctness_seq": [0, 1] * 4,
            "time_seq": list(range(1, 9)),
        }
    ]
    config = DyGKTNeighborConfig(
        num_question=1,
        num_neighbor=2,
        sampling_strategy=strategy,
        candidate_pool=5,
        seed=7,
    )
    tensors, indices = build_tensors(records, config)
    dataset = DyGKTDataset(tensors, indices, config.num_neighbor)
    batch_indices = [6, 7]
    batch = dataset.get_batch(batch_indices)
    expected = torch.tensor([0, *records[0]["skill_seq"]], dtype=torch.int32)
    for prefix, history_key in (("src", "user_his_idx"), ("dst", "que_his_idx")):
        event_ids = tensors[history_key][dataset.target_indices[batch_indices]].long()
        torch.testing.assert_close(
            tensors["lookup_skill"][event_ids], expected[event_ids]
        )
        if prefix == "src":
            torch.testing.assert_close(
                batch["src_neighbor_skill_ids"], expected[event_ids]
            )
        assert torch.all(event_ids < batch["idx"].unsqueeze(1))
        assert torch.all(
            batch[f"{prefix}_neighbor_node_ids"] == (1 if prefix == "src" else 2)
        )


def test_skill_comparison_uses_event_skills_and_excludes_padding() -> None:
    train, _, _, metadata = _prepare()
    changed_train = _prepare(swapped=True)[0]
    model = _model(metadata)
    captured: list[torch.Tensor] = []
    handle = model.projection_layer["struct"].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        model(train.get_batch([0, 2]))
        assert captured[-1][2, :, :, 0].tolist() == [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
        model(changed_train.get_batch([0, 2]))
        assert captured[-1][2, :, :, 0].tolist() == [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    finally:
        handle.remove()


@pytest.mark.parametrize("ablation", ["embed", "q_kid"])
def test_skill_encoding_uses_history_and_target_annotations(ablation: str) -> None:
    train, _, _, metadata = _prepare()
    model = _model(metadata, ablation)
    embedding = model.projection_layer["feature_Embed"]
    assert embedding.num_embeddings == 4
    assert embedding.padding_idx == 0
    captured: list[torch.Tensor] = []
    handle = embedding.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        model(train.get_batch([0, 2, 3]))
    finally:
        handle.remove()
    assert len(captured) == 2
    expected = [
        [
            [0, 0, 0, 0],
            [1, 2, 0, 0],
            [1, 2, 1, 0],
        ],
        [1, 1, 3],
    ]
    for skills, expected_skills in zip(captured, expected):
        assert skills.tolist() == expected_skills
        features = embedding(skills)
        assert torch.count_nonzero(features[skills == 0]) == 0


@pytest.mark.parametrize("ablation", _ABLATIONS)
def test_cpu_training_and_evaluation_with_graph_negative_sampling(
    ablation: str,
) -> None:
    torch.manual_seed(7)
    trainer = DyGKTTrainer.__new__(DyGKTTrainer)
    trainer.device_ = torch.device("cpu")
    runtime = trainer.build_components(
        _config(ablation=ablation), cast(DataSource, _Source())
    )
    trainer.model = runtime.model
    trainer.loss = runtime.loss_fn

    for dataset, training in (
        (runtime.train_data, True),
        (runtime.val_data, False),
        (runtime.test_data, False),
    ):
        trainer.model.train(training)
        batch = dataset.get_batch(list(range(len(dataset))))
        outputs = trainer.forward_pass(batch)
        assert outputs["y_hat"].shape == (len(dataset),)
        assert torch.isfinite(outputs["y_hat"]).all()
        torch.testing.assert_close(outputs["y_label"], batch["correctness"])
        torch.testing.assert_close(outputs["dst_node_ids"], batch["question"].long())
        torch.testing.assert_close(outputs["y_hat"], trainer.model(batch))

        if training:
            torch.manual_seed(0)
            perm = torch.randperm(len(dataset))
            if torch.equal(perm, torch.arange(len(dataset))):
                perm = torch.roll(perm, shifts=1)
            src = outputs["src_embeddings"]
            dst = outputs["dst_embeddings"]
            positive = (src * dst).sum(-1) / trainer.graph_neg_temperature
            negative = (src * dst[perm]).sum(-1) / trainer.graph_neg_temperature
            same_question = batch["question"] == batch["question"][perm]
            assert same_question.any() and (~same_question).any()
            negative = negative.masked_fill(same_question, -1e9)
            expected = F.cross_entropy(
                torch.stack([positive, negative], dim=1),
                torch.zeros(len(dataset), dtype=torch.long),
            )
            torch.manual_seed(0)
            torch.testing.assert_close(
                trainer._compute_graph_negative_loss(outputs), expected
            )
            loss = trainer._compute_loss(outputs)
        else:
            loss = trainer._compute_eval_loss(outputs)

        assert torch.isfinite(loss)
        runtime.optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        if ablation in ("embed", "q_kid"):
            grad = trainer.model.projection_layer["feature_Embed"].weight.grad
            assert grad is not None
            assert torch.count_nonzero(grad[0]) == 0
            assert torch.count_nonzero(grad[1:]) > 0
        runtime.optimizer.step()


@pytest.mark.parametrize("ablation", ["-1", "embed", "q_kid"])
def test_predictions_respond_to_event_skills_with_unchanged_question_features(
    ablation: str,
) -> None:
    torch.manual_seed(7)
    train, _, _, metadata = _prepare()
    changed = _prepare(swapped=True)[0]
    model = _model(metadata, ablation).eval()
    with torch.no_grad():
        original_logits = model(train.get_batch([0, 1, 2, 3]))
        changed_logits = model(changed.get_batch([0, 1, 2, 3]))
    assert not torch.allclose(original_logits, changed_logits)


@pytest.mark.parametrize("ablation", _ABLATIONS)
def test_question_history_skills_have_no_consumer(ablation: str) -> None:
    torch.manual_seed(7)
    _, val, _, metadata = _prepare()
    changed_val = _prepare(swapped=True)[1]
    history_indices = val.tensors["que_his_idx"][val.target_indices[0]].long()
    assert val.tensors["lookup_skill"][history_indices].tolist() == [1, 2, 1, 0]
    assert changed_val.tensors["lookup_skill"][history_indices].tolist() == [2, 1, 1, 0]
    batch = val.get_batch([0])
    changed_batch = changed_val.get_batch([0])
    assert "dst_neighbor_skill_ids" not in batch
    assert batch.keys() == changed_batch.keys()
    for field in batch:
        torch.testing.assert_close(batch[field], changed_batch[field], rtol=0, atol=0)

    model = _model(metadata, ablation).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(changed_batch), rtol=0, atol=0)


@pytest.mark.parametrize("ablation", _ABLATIONS)
def test_target_skill_changes_preserve_ablation_semantics(ablation: str) -> None:
    torch.manual_seed(7)
    train, _, _, metadata = _prepare()
    batch = train.get_batch([2])
    changed_batch = {**batch, "skill": torch.tensor([2], dtype=batch["skill"].dtype)}
    model = _model(metadata, ablation).eval()
    with torch.no_grad():
        original = model(batch)
        changed = model(changed_batch)
    if ablation == "counter":
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
    else:
        assert not torch.allclose(original, changed)


@pytest.mark.parametrize("ablation", _ABLATIONS)
@pytest.mark.parametrize("field", ("dst_neighbor_edge_feats", "dst_neighbor_times"))
def test_question_history_retains_edge_and_time_paths(
    ablation: str, field: str
) -> None:
    torch.manual_seed(7)
    _, val, _, metadata = _prepare()
    batch = val.get_batch([0])
    changed_batch = {**batch, field: batch[field].clone()}
    valid_history = batch["dst_neighbor_node_ids"] != 0
    if field == "dst_neighbor_edge_feats":
        changed_batch[field][valid_history] = 1 - batch[field][valid_history]
    else:
        changed_batch[field][valid_history] += torch.tensor([1.0, 10.0, 100.0])

    model = _model(metadata, ablation).eval()
    with torch.no_grad():
        original = model(batch)
        changed = model(changed_batch)
    if ablation in ("q_qid", "q_kid") or (
        field == "dst_neighbor_times" and ablation == "time"
    ):
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
    else:
        assert not torch.allclose(original, changed)
