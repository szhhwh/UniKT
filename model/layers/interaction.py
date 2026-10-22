"""Conversions for padded interaction annotations."""

import torch


def skills_to_multihot(skills: torch.Tensor, num_skills: int) -> torch.Tensor:
    """Encode each event's skill list, padded with num_skills, as a multi-hot row."""
    skill_indicator = torch.zeros(
        *skills.shape[:-1], num_skills + 1, dtype=torch.bool, device=skills.device
    )
    return skill_indicator.scatter_(-1, skills, True)[..., :num_skills]
