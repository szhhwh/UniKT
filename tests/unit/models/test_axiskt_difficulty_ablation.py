"""Difficulty ablations must isolate embeddings and prediction heads."""

import pytest
import torch

from model.AxisKT.AxisKT_model import AxisKT


def make_model(**flags):
    torch.manual_seed(42)
    model = AxisKT(
        {"num_questions": 4, "num_skills": 2},
        torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        torch.ones(4, 2, dtype=torch.bool),
        hidden_dim=8,
        n_blocks=1,
        dropout=0,
        **flags,
    )
    with torch.no_grad():
        model.question_diff.weight.copy_(torch.tensor([[0.2], [-0.4], [0.8], [0.1]]))
        model.local_write.weight.normal_(std=0.1)
        model.ability_head.weight.normal_(std=0.1)
        model.irt_disc.fill_(1.7)
    return model


def batch(device="cpu"):
    q = torch.tensor([[0, 1, 2, 3, 0, 2]], device=device)
    r = torch.tensor([[1, 0, 1, 0, 0, 1]], device=device)
    t = torch.arange(6, device=device).float().unsqueeze(0)
    return q, r, t, torch.ones_like(q, dtype=torch.bool)


def test_no_rasch_removes_difficulty_from_events_and_local_writes_only():
    model = make_model(use_rasch_embedding=False)
    q, r, t, mask = batch()
    events = model._event_embeddings(q, mask=mask)
    local = model._local_pre_states(q, r, t, mask)
    features = torch.randn(1, 5, 24)
    logits = model._irt_term(features, q[:, 1:])
    with torch.no_grad():
        model.question_diff.weight.add_(0.5)
    torch.testing.assert_close(events, model._event_embeddings(q, mask=mask))
    torch.testing.assert_close(local, model._local_pre_states(q, r, t, mask))
    torch.testing.assert_close(
        model._irt_term(features, q[:, 1:]), logits - model.irt_disc * 0.5
    )


def test_no_irt_keeps_rasch_but_uses_plain_linear_logits():
    model = make_model(use_irt_head=False)
    q, _, _, mask = batch()
    features = torch.randn(1, 5, 24)
    events = model._event_embeddings(q, mask=mask)
    logits = model._irt_term(features, q[:, 1:])
    torch.testing.assert_close(logits, model.ability_head(features).squeeze(-1))
    with torch.no_grad():
        model.question_diff.weight.add_(0.5)
        model.irt_disc.fill_(3.0)
    torch.testing.assert_close(logits, model._irt_term(features, q[:, 1:]))
    assert not torch.allclose(events, model._event_embeddings(q, mask=mask))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Fused inference requires CUDA"
)
@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"use_rasch_embedding": False},
        {"use_irt_head": False},
        {"use_rasch_embedding": False, "use_irt_head": False},
    ],
)
def test_fused_inference_matches_training_path_and_remaining_difficulty_has_gradient(
    flags,
):
    model = make_model(**flags).cuda().eval()
    args = batch("cuda")
    with torch.enable_grad():
        logits = model(*args)
        logits.sum().backward()
    if model.use_rasch_embedding or model.use_irt_head:
        assert model.question_diff.weight.grad is not None
        assert model.question_diff.weight.grad.abs().sum() > 0
    else:
        assert model.question_diff.weight.grad is None
    if not model.use_rasch_embedding:
        assert (
            model.skill_change.weight.grad is None
            or not model.skill_change.weight.grad.any()
        )
    if not model.use_irt_head:
        assert model.irt_disc.grad is None
    with torch.no_grad():
        fused = model(*args)
    torch.testing.assert_close(fused, logits.detach(), atol=2e-5, rtol=2e-5)


def test_joint_ablation_predictions_are_independent_of_difficulty():
    model = make_model(use_rasch_embedding=False, use_irt_head=False).eval()
    args = batch()
    with torch.enable_grad():
        before = model(*args)
        with torch.no_grad():
            model.question_diff.weight.normal_(mean=10, std=5)
            model.skill_change.weight.normal_(mean=5, std=5)
            model.irt_disc.fill_(100)
        after = model(*args)
    torch.testing.assert_close(before, after)
