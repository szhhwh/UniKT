"""Tests for model/layers/interaction.py.

Mirrors ``model/layers/`` — the cross-model shared components. These are not
per-model implementations: every model composes them, so a defect here shows up
in all of them at once.
"""

from __future__ import annotations

import torch

from model.layers.interaction import skills_to_multihot


def test_skills_to_multihot_keeps_event_boundaries_and_excludes_padding() -> None:
    skills = torch.tensor([[[0, 3], [1, 2], [3, 3]]])
    assert skills_to_multihot(skills, num_skills=3).tolist() == [
        [[True, False, False], [False, True, True], [False, False, False]]
    ]
