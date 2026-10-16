from types import SimpleNamespace
from typing import Any, cast

import polars as pl
import torch

from model.BDGKT.BDGKT_data import BDGKTModelData, _collate_fn
from model.BDGKT.BDGKT_model import BDGKT
from utils.data_process import DataSource


class _Source:
    dataset = "bdgkt_events"

    def __init__(self, swapped: bool = False) -> None:
        skills = [[1], [0], [0, 1]] if swapped else [[0], [1], [0, 1]]
        self.data = pl.DataFrame(
            {
                "interaction_id": list(range(9)),
                "user": [0] * 3 + [1] * 3 + [2] * 3,
                "question": [0] * 9,
                "label": [1, 0, 1, 0, 1, 0, 1, 0, 1],
                "timestamp": [30, 40, 50, 10, 20, 60, 15, 25, 35],
                "fold": [1] * 3 + [0] * 3 + [-1] * 3,
                "skills": pl.Series(skills * 3, dtype=pl.List(pl.Int32)),
            }
        )

    def get_sequence_data(self) -> pl.DataFrame:
        return self.data

    def get_split_question_sequence_data(self, **kwargs: Any) -> pl.DataFrame:
        raise AssertionError("BDGKT operates on canonical events")

    def get_metadata(self, key: str | None = None) -> Any:
        metadata = {"num_questions": 1, "num_skills": 2, "kfold_n_splits": 2}
        return metadata if key is None else metadata[key]

    def get_relation(self, name: str) -> pl.DataFrame:
        raise AssertionError("Event skill inputs must not use question relations")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        general=SimpleNamespace(cache=False, device="cpu"),
        data=SimpleNamespace(fold=0),
        model=SimpleNamespace(
            hidden_dim=8,
            layer_num=2,
            drop1=0.0,
            drop2=0.0,
            question_max_length=4,
            student_max_length=3,
            learning_rate=0.001,
            weight_decay=0.0,
        ),
    )


def _prepare(source: _Source) -> tuple:
    return BDGKTModelData(cast(DataSource, source)).prepare_data(_config())


def _model(layer_num: int = 1) -> BDGKT:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return BDGKT(
            student_num=3,
            question_num=1,
            skill_num=2,
            hidden_size=8,
            student_max_length=3,
            question_max_length=4,
            drop1=0.0,
            drop2=0.0,
            layer_num=layer_num,
        ).eval()


def _inputs(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in data.items() if key != "labels"}


def test_preparation_and_history_features_use_each_events_own_skills() -> None:
    source = _Source()
    swapped_source = _Source(swapped=True)
    relation = (
        source.data.select("question", "skills")
        .explode("skills", empty_as_null=True)
        .unique()
    )
    swapped_relation = (
        swapped_source.data.select("question", "skills")
        .explode("skills", empty_as_null=True)
        .unique()
    )
    assert relation.sort("skills").equals(swapped_relation.sort("skills"))

    prepared = _prepare(source)
    swapped = _prepare(swapped_source)
    assert prepared[3:] == swapped[3:] == (3, 1)
    for dataset, swapped_dataset in zip(prepared[:3], swapped[:3], strict=True):
        assert len(dataset) == len(swapped_dataset) == 2
        for key in dataset.data:
            if key not in {"hist_skills", "target_skills"}:
                assert torch.equal(dataset.data[key], swapped_dataset.data[key])
        assert not torch.equal(
            dataset.data["target_skills"], swapped_dataset.data["target_skills"]
        )
        assert not torch.equal(
            dataset.data["hist_skills"], swapped_dataset.data["hist_skills"]
        )

    train, swapped_train = prepared[0], swapped[0]
    assert train.data["target_skills"].tolist() == [[1, -1], [0, 1]]
    assert train.data["hist_skills"][1].tolist() == [
        [-1, -1],
        [0, -1],
        [1, -1],
        [-1, -1],
    ]

    model = _model()
    features = []
    hook = model.layers[0].question_abs_layer.register_forward_pre_hook(
        lambda module, args: features.append(args[0].detach())
    )
    try:
        with torch.no_grad():
            prediction = model(**_inputs(train.data))
            swapped_prediction = model(**_inputs(swapped_train.data))
    finally:
        hook.remove()

    torch.testing.assert_close(features[0][1, 1, 8:], model.skill_emb.weight[0])
    torch.testing.assert_close(features[1][1, 1, 8:], model.skill_emb.weight[1])
    assert not torch.equal(features[0], features[1])
    assert not torch.allclose(prediction, swapped_prediction)


def test_target_prediction_averages_only_annotated_skills() -> None:
    train = _prepare(_Source())[0]
    inputs = {
        key: value[:1].expand(3, *value.shape[1:]).clone()
        for key, value in _inputs(train.data).items()
    }
    inputs["target_skills"] = torch.tensor([[0, -1], [1, -1], [0, 1]])
    model = _model(layer_num=0)
    with torch.no_grad():
        model.dynamic_fusion.weight.zero_()
        model.dynamic_fusion.bias.zero_()
        model.qdiff_encoder.weight.zero_()
        model.qdiff_encoder.bias.copy_(torch.tensor([-2.0, 2.0]))
        prediction = model(**inputs)

    skill_probabilities = torch.sigmoid(
        1.702 * 5.0 * (0.5 - torch.sigmoid(torch.tensor([-2.0, 2.0])))
    )
    torch.testing.assert_close(
        prediction,
        torch.stack(
            [
                skill_probabilities[0],
                skill_probabilities[1],
                skill_probabilities.mean(),
            ]
        ),
    )


def test_skill_padding_and_invalid_history_do_not_change_predictions() -> None:
    train = _prepare(_Source())[0]
    inputs = _inputs(train.data)
    padded_inputs = dict(inputs)
    for key in ("target_skills", "hist_skills"):
        skills = inputs[key]
        padding = torch.full((*skills.shape[:-1], 3), -1, dtype=torch.long)
        padded_inputs[key] = torch.cat([skills, padding], dim=-1)
    padded_inputs["hist_skills"][~inputs["hist_mask"]] = 1

    model = _model()
    with torch.no_grad():
        prediction = model(**inputs)
        padded_prediction = model(**padded_inputs)

    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction, padded_prediction)
    assert (train.data["hist_skills"][~train.data["hist_mask"]] == -1).all()


def test_peer_responses_belong_to_selected_interactions() -> None:
    source = _Source()
    source.data = source.data.filter(pl.col("user") < 2)
    train = _prepare(source)[0]
    history_position = train.data["hist_mask"][0].nonzero().item()
    candidate_mask = train.data["q_ans_mask"][0, history_position]
    assert candidate_mask.sum() == 2
    assert train.data["q_ans_s"][0, history_position, candidate_mask].tolist() == [1, 1]
    assert train.data["q_ans_r"][0, history_position, candidate_mask].tolist() == [0, 1]


def test_timestamp_ties_keep_skills_and_labels_in_interaction_order() -> None:
    source = _Source()
    source.data = source.data.with_columns(pl.lit(100).alias("timestamp"))
    original = source.data
    source.data = source.data.reverse()
    model_data = BDGKTModelData(cast(DataSource, source))
    edges = model_data._build_sorted_edges(source.data.to_pandas())
    assert edges["s2q_response"].tolist() == original["label"].to_list()
    assert edges["s2q_skills"].tolist() == [[0, -1], [1, -1], [0, 1]] * 3


def test_empty_folds_keep_skill_tensor_shapes() -> None:
    source = _Source()
    source.data = source.data.with_columns(pl.lit(1).alias("fold"))
    train, val, test, *_ = _prepare(source)
    assert len(train) == 6
    for dataset in (val, test):
        assert len(dataset) == 0
        assert dataset.data["target_skills"].shape == (0, 2)
        assert dataset.data["hist_skills"].shape == (0, 4, 2)


def test_trainer_runs_cpu_forward_backward_with_padded_events(
    registry_snapshot: None,
) -> None:
    from model.BDGKT.BDGKT_trainer import BDGKTTrainer

    trainer = BDGKTTrainer.__new__(BDGKTTrainer)
    trainer.device_ = torch.device("cpu")
    components = trainer.build_components(_config(), cast(DataSource, _Source()))
    trainer.model = components.model
    train = components.train_data
    batch = _collate_fn([train[i] for i in range(len(train))])

    trainer.model.train()
    outputs = trainer.forward_pass(batch)
    torch.testing.assert_close(outputs["y_label"], train.data["labels"].float())
    loss = components.loss_fn(outputs["y_hat"], outputs["y_label"])
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        (name, parameter.grad)
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for _, gradient in gradients), [
        name for name, gradient in gradients if not torch.isfinite(gradient).all()
    ]
    assert trainer.model.skill_emb.weight.grad.abs().sum() > 0
    assert trainer.model.qdiff_encoder.weight.grad.abs().sum() > 0
    components.optimizer.step()

    trainer.model.eval()
    with torch.no_grad():
        assert torch.isfinite(trainer.forward_pass(batch)["y_hat"]).all()
