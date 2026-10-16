from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
import torch

from model.AxisKT.AxisKT_analyzer import AxisKTAnalyzer, _analyzer_collate_fn
from model.AxisKT.AxisKT_data import (
    AxisKTDataset,
    AxisKTModelData,
    axiskt_packed_collate_fn,
    build_axiskt_model,
)
from model.AxisKT.AxisKT_model import AxisKT
from model.AxisKT.AxisKT_trainer import AxisKTTrainer
from utils.data_process import DataSource


class _EventSource:
    dataset = "axiskt_events"

    def __init__(self, allow_relation: bool = False) -> None:
        rows = []
        for sequence, (length, fold) in enumerate([(4, 0), (3, 0), (4, 1), (3, -1)]):
            for position in range(length):
                rows.append(
                    {
                        "sequence_id": sequence,
                        "seq_pos": position,
                        "user": 10 + sequence,
                        "question": int(position == 3),
                        "label": int(position % 2 == 0),
                        "interaction_id": sequence * 4 + position,
                        "skills": [[0], [1], [0, 2], [1]][position],
                        "timestamp": position * 1000,
                        "fold": fold,
                    }
                )
        self.data = pl.DataFrame(rows).with_columns(
            pl.col("skills").cast(pl.List(pl.Int32))
        )
        self.allow_relation = allow_relation
        self.relation = pl.DataFrame(
            {"question": [1, 0, 0, 0, 0], "skill": [1, 2, 1, 0, 1]}
        )
        self.relation_reads = 0

    def get_metadata(self, key: str | None = None) -> Any:
        metadata = {
            "num_questions": 2,
            "num_skills": 4,
            "num_users": 4,
            "max_question_seq_len": 5,
            "kfold_n_splits": 2,
        }
        return metadata if key is None else metadata[key]

    def get_split_question_sequence_data(self, **kwargs: Any) -> pl.DataFrame:
        return self.data

    def get_relation(self, name: str) -> pl.DataFrame:
        self.relation_reads += 1
        if not self.allow_relation:
            raise AssertionError("Training must use event annotations")
        assert name == "question_skill"
        return self.relation


@pytest.fixture
def rc() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(fold=0),
        compile=SimpleNamespace(compile=False),
        model=SimpleNamespace(
            hidden_dim=8,
            n_blocks=1,
            dropout=0.0,
            conv_kernel_size=3,
            conv_dilation_base=2,
            question_embed_dim=-1,
            use_global=True,
            use_local=True,
            use_forgetting=True,
            learning_rate=0.01,
            weight_decay=0.0,
            max_clip_grad_norm=5.0,
            amp=False,
        ),
    )


@pytest.fixture
def event_batch(rc: SimpleNamespace) -> tuple[torch.Tensor, ...]:
    _, val, _, _ = AxisKTModelData(cast(DataSource, _EventSource())).prepare_data(rc)
    return axiskt_packed_collate_fn([val[index] for index in range(len(val))])


def _model(use_local: bool = True, use_global: bool = True) -> AxisKT:
    torch.manual_seed(17)
    model = AxisKT(
        {"num_questions": 2, "num_skills": 4},
        hidden_dim=8,
        n_blocks=1,
        dropout=0.0,
        use_local=use_local,
        use_global=use_global,
    )
    with torch.no_grad():
        model.ability_head.weight.normal_(std=0.2)
        model.local_write.weight.normal_(std=0.2)
    return model


def test_training_preparation_and_model_state_need_no_static_skill_table(
    rc: SimpleNamespace,
) -> None:
    source = _EventSource()
    train, val, test, extra = AxisKTModelData(cast(DataSource, source)).prepare_data(rc)
    assert [len(train), len(val), len(test)] == [1, 2, 1]
    assert set(extra) == {"max_gap_bins", "user_ids"}
    assert source.relation_reads == 0
    model = build_axiskt_model(rc, cast(DataSource, source), extra)
    assert not hasattr(model, "question_skill_ids")
    assert not hasattr(model, "question_skill_mask")
    assert not any("question_skill" in name for name in model.state_dict())
    assert val.dataset.skill_ids[val.indices[0], :3].tolist() == [
        [0, 4],
        [1, 4],
        [0, 2],
    ]


@pytest.mark.parametrize(
    "method",
    [
        "forward",
        "_resolve_event_features",
        "_pack_kc_occurrences",
        "_scan_states",
        "_local_pre_states",
        "_event_embeddings",
    ],
)
def test_event_paths_require_annotations(
    method: str, event_batch: tuple[torch.Tensor, ...]
) -> None:
    args = (
        event_batch[:1]
        if method in {"_resolve_event_features", "_event_embeddings"}
        else event_batch[:4]
    )
    with pytest.raises(TypeError):
        getattr(_model(), method)(*args)


@pytest.mark.parametrize(
    ("column", "shape", "message"),
    [
        (-2, (2, 5), "skill_ids must have shape"),
        (-2, (1, 5, 2), "skill_ids must have shape"),
        (-2, (2, 4, 2), "skill_ids must have shape"),
        (-2, (2, 5, 0), "skill_ids must have shape"),
        (-1, (2, 5, 1), "skill_mask must match"),
        (4, (2, 2, 1), "kc_order must have shape"),
        (4, (1, 2), "kc_order must have shape"),
        (4, (2, 11), "kc_order width exceeds"),
        (5, (2, 2, 5), "kc_inverse must have shape"),
        (5, (1, 10), "kc_inverse must have shape"),
        (5, (2, 9), "kc_inverse must have shape"),
    ],
)
def test_forward_rejects_misaligned_event_inputs(
    column: int,
    shape: tuple[int, ...],
    message: str,
    event_batch: tuple[torch.Tensor, ...],
) -> None:
    batch = list(event_batch)
    batch[column] = torch.zeros(shape, dtype=batch[column].dtype)
    with pytest.raises(ValueError, match=message):
        _model()(
            *batch[:4],
            skill_ids=batch[-2],
            skill_mask=batch[-1],
            kc_order=batch[4],
            kc_inverse=batch[5],
        )


def test_dataset_normalizes_event_tensor_dtypes() -> None:
    questions = np.zeros((1, 3), dtype=np.int32)
    skills = np.array([[[0, 4], [1, 4], [4, 4]]], dtype=np.int16)
    dataset = AxisKTDataset(
        questions,
        np.array([[1, 0, 0]], dtype=np.int8),
        np.array([[0, 1, 0]], dtype=np.float32),
        np.array([[1, 1, 0]], dtype=np.int8),
        skills,
        (skills != 4).astype(np.uint8),
    )
    batch = axiskt_packed_collate_fn([dataset[0]])
    assert batch[0].dtype == batch[1].dtype == torch.long
    assert batch[2].dtype == torch.float64
    assert batch[3].dtype == batch[-1].dtype == torch.bool
    assert batch[4].dtype == batch[5].dtype == batch[-2].dtype == torch.long
    with torch.no_grad():
        logits = _model()(*batch[:4], batch[-2], batch[-1], kc_order=batch[4])
    assert torch.isfinite(logits).all()


def test_repeated_question_predictions_follow_event_skills(
    event_batch: tuple[torch.Tensor, ...],
) -> None:
    model = _model()
    questions, responses, times, mask = event_batch[:4]
    skills, skill_mask = event_batch[-2:]
    changed_skills = skills.clone()
    changed_skills[:, 1, 0] = 2
    with torch.no_grad():
        original = model(questions, responses, times, mask, skills, skill_mask)
        changed = model(questions, responses, times, mask, changed_skills, skill_mask)
    assert questions[:, :3].eq(0).all()
    assert not torch.allclose(original[:, 0], changed[:, 0])


def test_local_history_stays_with_the_annotated_skill() -> None:
    model = _model(use_global=False)
    questions = torch.zeros(1, 4, dtype=torch.long)
    responses = torch.tensor([[0, 1, 1, 0]])
    times = torch.arange(4, dtype=torch.float64).unsqueeze(0)
    mask = torch.ones_like(questions, dtype=torch.bool)
    skills = torch.tensor([[[0], [1], [0], [1]]])
    skill_mask = torch.ones_like(skills, dtype=torch.bool)
    changed_responses = responses.clone()
    changed_responses[0, 0] = 1
    with torch.no_grad():
        original = model(questions, responses, times, mask, skills, skill_mask)
        changed = model(questions, changed_responses, times, mask, skills, skill_mask)
    torch.testing.assert_close(original[:, [0, 2]], changed[:, [0, 2]])
    assert not torch.allclose(original[:, 1], changed[:, 1])


@pytest.mark.parametrize("prepacked", [False, True])
@pytest.mark.parametrize("use_local", [False, True])
def test_cpu_training_and_inference_share_event_inputs(
    prepacked: bool, use_local: bool, event_batch: tuple[torch.Tensor, ...]
) -> None:
    model = _model(use_local=use_local).eval()
    questions, responses, times, mask, order, inverse, _, skills, skill_mask = (
        event_batch
    )
    packed = {"kc_order": order, "kc_inverse": inverse} if prepacked else {}
    logits = model(questions, responses, times, mask, skills, skill_mask, **packed)
    logits.square().sum().backward()
    assert model.skill_embed.weight.grad is not None
    assert torch.isfinite(model.skill_embed.weight.grad).all()
    assert model.skill_embed.weight.grad[:3].abs().sum() > 0
    assert model.skill_embed.weight.grad[model.num_skills].eq(0).all()
    with torch.inference_mode():
        inferred = model(
            questions, responses, times, mask, skills, skill_mask, **packed
        )
    torch.testing.assert_close(logits, inferred)
    assert inferred[~mask].eq(0).all()


def test_autocast_keeps_local_attention_in_fp32() -> None:
    model = _model()
    with torch.no_grad():
        model.local_readout.weight.normal_(std=0.2)
    states = torch.randn(2, 6, model.hidden_dim).bfloat16()
    embeddings = torch.randn_like(states, dtype=torch.float32)
    positions = torch.tensor([[0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 2, 2]])
    valid = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    questions = torch.zeros(2, 3, model.hidden_dim)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model._packed_question_conditioned_readout(
            states, embeddings, positions, valid, questions
        )
    weight = model.local_readout.weight[0]
    expected = torch.zeros_like(questions)
    for row in range(2):
        for position in range(3):
            selected = (positions[row] == position) & valid[row]
            local = states[row, selected].float()
            skills = embeddings[row, selected]
            scores = (
                local * weight[: model.hidden_dim] + skills * weight[model.hidden_dim :]
            ).sum(dim=-1)
            expected[row, position] = (scores.softmax(dim=0).unsqueeze(-1) * local).sum(
                dim=0
            )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    assert model.local_readout.weight.grad is not None
    assert torch.isfinite(model.local_readout.weight.grad).all()


def test_trainer_accepts_event_features_and_runs_a_cpu_optimizer_step(
    rc: SimpleNamespace,
) -> None:
    source = _EventSource()
    trainer = AxisKTTrainer.__new__(AxisKTTrainer)
    trainer.run_config = rc
    trainer.device_ = torch.device("cpu")
    components = trainer.build_components(rc, source)
    trainer.model = components.model
    train = components.train_data
    batch = axiskt_packed_collate_fn([train[0]])
    outputs = trainer.forward_pass(batch)
    loss = components.loss_fn(outputs["y_hat"], outputs["y_label"])
    loss.backward()
    components.optimizer.step()
    assert torch.isfinite(loss)
    assert source.relation_reads == 0
    with torch.inference_mode():
        inferred = trainer.forward_pass(batch)
    assert torch.isfinite(inferred["y_hat"]).all()
    torch.testing.assert_close(inferred["y_label"], torch.tensor([0.0, 1.0, 0.0]))


def test_analyzer_reads_coverage_separately_and_keeps_event_alignment(
    rc: SimpleNamespace,
) -> None:
    source = _EventSource(allow_relation=True)
    analyzer = AxisKTAnalyzer.__new__(AxisKTAnalyzer)
    analyzer.run_config = rc
    analyzer.device_ = torch.device("cpu")
    components = analyzer.build_components(rc, cast(DataSource, source))
    analyzer.model = components.model.eval()
    with torch.no_grad():
        analyzer.model.ability_head.weight.normal_(std=0.2)
        analyzer.model.local_write.weight.normal_(std=0.2)
    assert source.relation_reads == 1
    assert [questions.tolist() for questions in analyzer._skill_questions] == [
        [0],
        [0, 1],
        [0],
        [],
    ]
    assert not hasattr(analyzer, "question_skill_ids_np")
    dataset = components.val_data
    rows = [dataset[index] for index in range(len(dataset))]
    batch = _analyzer_collate_fn(rows)
    trainer = AxisKTTrainer.__new__(AxisKTTrainer)
    trainer.run_config = rc
    trainer.device_ = analyzer.device_
    trainer.model = analyzer.model
    with torch.inference_mode():
        expected = trainer.forward_pass(axiskt_packed_collate_fn(rows))
        outputs = analyzer.forward_pass(batch)
        case = analyzer.extract_case_data(batch, outputs)
    torch.testing.assert_close(outputs["y_hat"], expected["y_hat"])
    assert outputs["knowledge_states"].shape == (2, 5, 4)
    assert torch.isfinite(outputs["knowledge_states"][:, :, :3]).all()
    assert outputs["knowledge_states"][:, :, 3].isnan().all()
    assert case["skills"] == [[1], [0, 2], [1], [1], [0, 2]]
    np.testing.assert_array_equal(case["user_ids"], [10, 10, 10, 11, 11])
    np.testing.assert_array_equal(case["question_ids"], [0, 0, 1, 0, 0])
    assert source.relation_reads == 1
    assert not analyzer.model.local_write._forward_hooks
    assert analyzer.model._scan_states.__func__ is AxisKT._scan_states


@pytest.mark.parametrize(
    ("column", "invalid_id", "message"),
    [
        ("skill", -1, "outside metadata range"),
        ("skill", 4, "outside metadata range"),
        ("question", -1, "out-of-range question id"),
        ("question", 2, "out-of-range question id"),
    ],
)
def test_analyzer_rejects_out_of_range_relation_ids(
    column: str, invalid_id: int, message: str
) -> None:
    source = _EventSource(allow_relation=True)
    source.relation = source.relation.with_columns(pl.lit(invalid_id).alias(column))
    analyzer = AxisKTAnalyzer.__new__(AxisKTAnalyzer)
    analyzer.num_questions = source.get_metadata("num_questions")
    analyzer.num_skills = source.get_metadata("num_skills")
    with pytest.raises(ValueError, match=message):
        analyzer._build_skill_questions(cast(DataSource, source))
