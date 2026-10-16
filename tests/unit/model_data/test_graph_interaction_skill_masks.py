from types import SimpleNamespace
from typing import Any, TypedDict, cast

import polars as pl
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from model.DAGKT.DAGKT_model import DAGKT
from model.HDHKT.HDHKT_model import HDHKT
from model.layers.components import GeneralInteraction
from model.layers.hypergraph import Hypergraph

_NUM_SKILLS = 4
_METADATA = {
    "num_questions": 4,
    "num_skills": _NUM_SKILLS,
    "num_assignments": 2,
    "num_templates": 2,
}


class _ModelOptions(TypedDict):
    hidden_dim: int
    n_hop: int
    heads: int
    lstm_layers: int
    dropout: float
    history_neighbour: int
    att_bound: float


@pytest.fixture
def graphs() -> tuple[HeteroData, Hypergraph]:
    graph = HeteroData()
    for node_type, count in (
        ("question", 4),
        ("skill", _NUM_SKILLS),
        ("assignment", 2),
        ("template", 2),
    ):
        graph[node_type].num_nodes = count
    for edge_type, edges in (
        (
            ("question", "has", "skill"),
            [[0, 0, 1, 1, 2, 3], [0, 1, 1, 2, 2, 3]],
        ),
        (("skill", "related_to", "assignment"), [[0, 1, 2, 3], [0, 0, 1, 1]]),
        (("question", "belongs_to", "template"), [[0, 1, 2, 3], [0, 1, 0, 1]]),
    ):
        edge_index = torch.tensor(edges)
        graph[edge_type].edge_index = edge_index
        src, relation, dst = edge_type
        graph[dst, f"rev_{relation}", src].edge_index = edge_index.flip(0)
    return graph, Hypergraph(num_v=4, e_list=[[0, 1], [1, 2], [2, 3]])


def _build_model(name: str, graph: HeteroData) -> DAGKT | HDHKT:
    torch.manual_seed(17)
    kwargs: _ModelOptions = {
        "hidden_dim": 8,
        "n_hop": 1,
        "heads": 1,
        "lstm_layers": 1,
        "dropout": 0.0,
        "history_neighbour": 1,
        "att_bound": 0.0,
    }
    if name == "DAGKT":
        return DAGKT(_METADATA, embedding_dim=8, ae_hidden_dim=4, **kwargs)
    return HDHKT(_METADATA, graph.metadata(), **kwargs)


@pytest.fixture(params=["DAGKT", "HDHKT"])
def graph_model(
    request: pytest.FixtureRequest, graphs: tuple[HeteroData, Hypergraph]
) -> DAGKT | HDHKT:
    return _build_model(request.param, graphs[0])


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    return {
        "sequence": torch.tensor([[0, 1, 0, 0, 0], [1, 2, 0, 3, 1], [2, 3, 0, 0, 0]]),
        "response": torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 0, 0, 0]]),
        "mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True] * 5,
                [True, True, False, False, False],
            ]
        ),
        "skills": torch.tensor(
            [
                [[0, 4, 4], [1, 4, 4], [2, 4, 4], [4, 4, 4], [4, 4, 4]],
                [[2, 0, 4], [0, 2, 1], [3, 4, 4], [1, 0, 4], [2, 3, 4]],
                [[2, 3, 4], [3, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
            ]
        ),
        "attempts": torch.tensor(
            [[0.5, 0.5, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 1.0], [0.5, 0.5, 0, 0, 0]]
        ),
    }


def _forward(
    model: DAGKT | HDHKT,
    graphs: tuple[HeteroData, Hypergraph],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    args = (batch["sequence"], batch["response"], batch["mask"], graphs[0])
    if isinstance(model, DAGKT):
        return model(*args, batch["skills"], batch["attempts"])
    return (model(*args, graphs[1], batch["skills"]),)


def test_predictions_do_not_depend_on_other_samples_skill_counts(
    graph_model: DAGKT | HDHKT,
    graphs: tuple[HeteroData, Hypergraph],
    batch: dict[str, torch.Tensor],
) -> None:
    graph_model.eval()
    alone = {name: value[:1, :3] for name, value in batch.items()}
    alone["skills"] = alone["skills"][..., :1]
    with torch.no_grad():
        alone_logits = _forward(graph_model, graphs, alone)[0]
        mixed_logits = _forward(graph_model, graphs, batch)[0]
    torch.testing.assert_close(alone_logits[0, :2], mixed_logits[0, :2])


@pytest.mark.parametrize("interleave", [False, True])
def test_unused_skill_slots_do_not_change_valid_predictions(
    graph_model: DAGKT | HDHKT,
    graphs: tuple[HeteroData, Hypergraph],
    batch: dict[str, torch.Tensor],
    interleave: bool,
) -> None:
    graph_model.eval()
    extended = dict(batch)
    extended["skills"] = F.pad(batch["skills"], (0, 2), value=_NUM_SKILLS)
    if interleave:
        extended["skills"] = extended["skills"][..., [3, 0, 4, 1, 2]]
    with torch.no_grad():
        baseline = _forward(graph_model, graphs, batch)[0]
        padded = _forward(graph_model, graphs, extended)[0]
    valid = batch["mask"][:, :-1] & batch["mask"][:, 1:]
    torch.testing.assert_close(baseline[:, :-1][valid], padded[:, :-1][valid])


class _RecordingReadout(nn.Module):
    def __init__(self, readout: nn.Module) -> None:
        super().__init__()
        self.readout = readout
        self.skill_slot_mask: torch.Tensor | None = None
        self.knowledge_status: torch.Tensor | None = None

    def forward(
        self,
        student_status: torch.Tensor,
        knowledge_status: torch.Tensor,
        user_mask: torch.Tensor,
        skill_slot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.skill_slot_mask = skill_slot_mask
        self.knowledge_status = knowledge_status
        return self.readout(
            student_status, knowledge_status, user_mask, skill_slot_mask=skill_slot_mask
        )


def test_readout_receives_shifted_event_skills_and_per_event_slot_masks(
    graph_model: DAGKT | HDHKT,
    graphs: tuple[HeteroData, Hypergraph],
    batch: dict[str, torch.Tensor],
) -> None:
    readout = _RecordingReadout(graph_model.general_interaction)
    graph_model.general_interaction = cast(GeneralInteraction, readout)
    graph_layer = (
        graph_model.conv if isinstance(graph_model, DAGKT) else graph_model.hetero_conv
    )
    captured: dict[str, torch.Tensor] = {}

    def capture_skill_nodes(
        module: nn.Module, args: tuple[Any, ...], output: dict[str, torch.Tensor]
    ) -> None:
        captured["skills"] = output["skill"]

    handle = graph_layer.register_forward_hook(capture_skill_nodes)
    try:
        _forward(graph_model, graphs, batch)
    finally:
        handle.remove()

    shifted_skills = torch.cat([batch["skills"][:, 1:], batch["skills"][:, -1:]], dim=1)
    assert readout.skill_slot_mask is not None
    assert readout.knowledge_status is not None
    torch.testing.assert_close(readout.skill_slot_mask, shifted_skills != _NUM_SKILLS)
    expected_embeddings = F.pad(captured["skills"], (0, 0, 0, 1))[shifted_skills]
    torch.testing.assert_close(readout.knowledge_status[:, :, 1:], expected_embeddings)


def test_variable_length_sequences_and_skill_lists_support_backward(
    graph_model: DAGKT | HDHKT,
    graphs: tuple[HeteroData, Hypergraph],
    batch: dict[str, torch.Tensor],
) -> None:
    graph_model.train()
    logits, *auxiliary_losses = _forward(graph_model, graphs, batch)
    valid = batch["mask"][:, :-1] & batch["mask"][:, 1:]
    loss = F.binary_cross_entropy_with_logits(
        logits[:, :-1][valid], batch["response"][:, 1:][valid].float()
    ) + sum(auxiliary_losses)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(logits[~batch["mask"]]) == 0
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in graph_model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)
    for parameter in (
        graph_model.question_embedding.weight,
        graph_model.skill_embedding.weight,
        graph_model.general_interaction.w1,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


class _Source:
    dataset = "graph_events"

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        rows: list[dict[str, int | list[int]]] = []
        for sequence_id, mask in enumerate(batch["mask"]):
            for seq_pos in mask.nonzero(as_tuple=True)[0].tolist():
                skills = batch["skills"][sequence_id, seq_pos]
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "seq_pos": seq_pos,
                        "interaction_id": len(rows),
                        "user": sequence_id,
                        "question": int(batch["sequence"][sequence_id, seq_pos].item()),
                        "label": int(batch["response"][sequence_id, seq_pos].item()),
                        "attempt_count": 1,
                        "skills": skills[skills != _NUM_SKILLS].tolist(),
                        "fold": [1, 0, -1][sequence_id],
                    }
                )
        self.data = pl.DataFrame(rows).with_columns(
            pl.col("skills").cast(pl.List(pl.Int32))
        )
        self.relation_reads = 0

    def get_metadata(self, key: str | None = None) -> Any:
        metadata = {**_METADATA, "max_question_seq_len": 5, "kfold_n_splits": 2}
        return metadata if key is None else metadata[key]

    def get_split_question_sequence_data(self, **kwargs: Any) -> pl.DataFrame:
        return self.data

    def get_sequence_data(self) -> pl.DataFrame:
        return self.data

    def get_relation(self, name: str) -> pl.DataFrame:
        assert name == "question_skill"
        self.relation_reads += 1
        return (
            self.data.select("question", "skills")
            .explode("skills", empty_as_null=False)
            .rename({"skills": "skill"})
            .unique()
        )


def test_dagkt_trainer_keeps_graph_without_unused_dense_skill_matrix(
    batch: dict[str, torch.Tensor], registry_snapshot: None
) -> None:
    from model.DAGKT.DAGKT_trainer import DAGKTConfig, DAGKTTrainer

    source = _Source(batch)
    model_config = DAGKTConfig()
    model_config.hidden_dim = 8
    model_config.embedding_dim = 8
    model_config.lstm_layers = 1
    model_config.n_hop = 1
    model_config.heads = 1
    model_config.dropout = 0.0
    model_config.ae_hidden_dim = 4
    model_config.history_neighbour = 1
    config = SimpleNamespace(
        general=SimpleNamespace(device="cpu"),
        data=SimpleNamespace(fold=0),
        model=model_config,
    )
    trainer = DAGKTTrainer.__new__(DAGKTTrainer)
    components = trainer.build_components(config, source)
    assert source.relation_reads == 1
    assert not hasattr(trainer, "question_skill_matrix")
    assert ("question", "has", "skill") in trainer.graph.edge_types
    trainer.model = components.model
    trainer.device_ = torch.device("cpu")
    for dataset in (components.train_data, components.val_data, components.test_data):
        inputs = next(iter(DataLoader(dataset, batch_size=1)))
        assert len(inputs) == 5
        outputs = trainer.forward_pass(inputs)
        assert torch.isfinite(outputs["y_hat"]).all()
        assert outputs["y_hat"].numel() == int(inputs[2].sum()) - 1
    assert components.optimizer is not None
    loss = (
        F.binary_cross_entropy_with_logits(outputs["y_hat"], outputs["y_label"])
        + outputs["_ae_loss_diff"]
        + outputs["_ae_loss_attempt"]
    )
    loss.backward()
    components.optimizer.step()


def test_hdhkt_analyzer_and_trainer_keep_event_annotations_aligned(
    graphs: tuple[HeteroData, Hypergraph],
    batch: dict[str, torch.Tensor],
    registry_snapshot: None,
) -> None:
    from model.HDHKT.HDHKT_analyzer import HDHKTAnalyzer
    from model.HDHKT.HDHKT_data import HDHKTDataset
    from model.HDHKT.HDHKT_trainer import HDHKTTrainer

    model = _build_model("HDHKT", graphs[0]).eval()
    users = torch.arange(10, 13)
    dataset = HDHKTDataset(
        batch["sequence"], batch["response"], batch["mask"], batch["skills"], users
    )
    inputs = next(iter(DataLoader(dataset, batch_size=3)))
    assert len(inputs) == 5
    analyzer = HDHKTAnalyzer.__new__(HDHKTAnalyzer)
    analyzer.model = model
    analyzer.device_ = torch.device("cpu")
    analyzer.hetero_graph, analyzer.hypergraph = graphs
    analyzer.num_questions = _METADATA["num_questions"]
    analyzer.num_skills = _NUM_SKILLS
    question_skills = torch.zeros(analyzer.num_questions, _NUM_SKILLS)
    question_ids, skill_ids = graphs[0]["question", "has", "skill"].edge_index
    question_skills[question_ids, skill_ids] = 1
    analyzer.skill_question_mask = question_skills.T
    analyzer.question_to_skill_ids = [
        row.nonzero(as_tuple=True)[0].tolist() for row in question_skills
    ]
    trainer = HDHKTTrainer.__new__(HDHKTTrainer)
    trainer.model = model
    trainer.device_ = torch.device("cpu")
    trainer.hetero_graph, trainer.hypergraph = graphs
    with torch.no_grad():
        outputs = analyzer.forward_pass(inputs)
        trained = trainer.forward_pass(inputs[1:])
        extracted = analyzer.extract_case_data(inputs, outputs)
    torch.testing.assert_close(outputs["y_hat"], trained["y_hat"])
    valid = batch["mask"][:, :-1] & batch["mask"][:, 1:]
    selected_skills = batch["skills"][:, 1:][valid]
    assert extracted["skills"] == [
        ids[ids != _NUM_SKILLS].tolist() for ids in selected_skills
    ]
    assert (
        extracted["question_ids"].tolist() == batch["sequence"][:, 1:][valid].tolist()
    )
    assert (
        extracted["user_ids"].tolist()
        == users[:, None].expand_as(valid)[valid].tolist()
    )
    assert extracted["labels"].tolist() == batch["response"][:, 1:][valid].tolist()
    assert extracted["knowledge_states"].shape == (int(valid.sum()), _NUM_SKILLS)
    torch.testing.assert_close(inputs[3], batch["mask"])
