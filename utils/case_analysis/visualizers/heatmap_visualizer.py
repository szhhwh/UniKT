"""Knowledge state heatmap visualization for case analysis.

Generates heatmap visualizations showing per-skill knowledge state changes
over a student's answer sequence, styled after the reference design.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap

from ...core import get_logger, register_case_visualizer
from ..interfaces import CaseVisualizer

logger = get_logger(__name__)


@register_case_visualizer("heatmap")
class HeatmapVisualizer(CaseVisualizer):
    """Generates knowledge state heatmap visualizations for case analysis.

    Creates a visual representation of a student's answer sequence showing:
    - Top row: question label (q0, q1, ...) for each time step
    - Second row: skill label (c0, c1, ...) for each time step
    - Third row: correctness marker (✓/✗) for each time step
    - Main body: heatmap where each row is a skill's knowledge state over time (colour encodes mastery level, green=high, red=low)

    Requires the predictions DataFrame to contain a 'knowledge_state' column
    (list of floats per row, one value per skill).
    """

    def __init__(self) -> None:
        """Initialize the visualizer with default styling."""
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10

    def plot_user(
        self,
        user_data: pd.DataFrame,
        user_id: int,
        *,
        output_path: str | None = None,
    ) -> plt.Figure:
        """Plot a knowledge state heatmap for one user (plugin entrypoint)."""
        return self.plot_user_heatmap(user_data, user_id, output_path=output_path)

    def plot_user_heatmap(
        self,
        user_data: pd.DataFrame,
        user_id: int,
        output_path: str | None = None,
        show_skill_names: bool = False,
    ) -> plt.Figure:
        """Plot a knowledge state heatmap for a single user."""
        required_cols = {"position", "question_id", "skill", "label", "knowledge_state"}
        missing = required_cols - set(user_data.columns)
        if missing:
            raise ValueError(f"Missing required columns in user_data: {missing}")

        if user_data["knowledge_state"].isna().any():
            raise ValueError(
                "knowledge_state column contains None values. Model must return knowledge states for case analysis."
            )

        user_data = user_data.sort_values("position").reset_index(drop=True)
        return self._plot_knowledge_state_heatmap(user_data, user_id, output_path)

    def _plot_knowledge_state_heatmap(
        self,
        user_data: pd.DataFrame,
        user_id: int,
        output_path: str | None,
    ) -> plt.Figure:
        T = len(user_data)

        seen: dict[int, None] = {}
        for skills in user_data["skill"]:
            for skill_id in skills:
                seen.setdefault(skill_id, None)
        unique_skills: list[int] = list(seen.keys())
        num_skills = len(unique_skills)
        skill_to_row = {s: i for i, s in enumerate(unique_skills)}

        ks_matrix = np.full((num_skills, T), np.nan)
        for t, (_, row) in enumerate(user_data.iterrows()):
            ks = row.get("knowledge_state")
            if ks is None:
                continue
            for skill_id, row_idx in skill_to_row.items():
                if skill_id < len(ks):
                    ks_matrix[row_idx, t] = float(ks[skill_id])

        ks_valid = ks_matrix[~np.isnan(ks_matrix)]
        if ks_valid.size > 0:
            ks_min = float(ks_valid.min())
            ks_max = float(ks_valid.max())
            if ks_max - ks_min < 1e-8:
                ks_min = ks_max - 1.0
        else:
            ks_min, ks_max = 0.0, 1.0

        header_h = 0.55
        cell_h = max(0.45, min(0.85, 10.0 / max(num_skills, 1)))
        heatmap_h = num_skills * cell_h
        fig_w = max(14, T * 0.38 + 3.0)
        fig_h = max(4.5, header_h * 3 + heatmap_h + 1.2)

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"User {user_id} - Knowledge State Heatmap",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )

        gs = gridspec.GridSpec(
            4,
            2,
            figure=fig,
            height_ratios=[header_h, header_h, header_h, heatmap_h],
            width_ratios=[1, 0.015],
            hspace=0.0,
            wspace=0.02,
            left=0.07,
            right=0.93,
            top=0.93,
            bottom=0.10,
        )

        ax_question = fig.add_subplot(gs[0, 0])
        ax_skill = fig.add_subplot(gs[1, 0])
        ax_resp = fig.add_subplot(gs[2, 0])
        ax_main = fig.add_subplot(gs[3, 0])
        ax_cbar = fig.add_subplot(gs[3, 1])

        self._draw_question_row(ax_question, user_data, T)
        self._draw_skill_row(ax_skill, user_data, unique_skills, T)
        self._draw_resp_row(ax_resp, user_data, T)

        cmap = plt.get_cmap("RdYlGn")
        im = ax_main.imshow(
            ks_matrix,
            aspect="auto",
            cmap=cmap,
            vmin=ks_min,
            vmax=ks_max,
            interpolation="none",
        )

        ax_main.set_yticks(np.arange(num_skills))
        ax_main.set_yticklabels(
            [self._skill_label(s) for s in unique_skills], fontsize=10
        )
        ax_main.tick_params(axis="y", length=0, pad=4)

        ax_main.set_xticks(np.arange(T))
        ax_main.set_xticklabels([str(t + 1) for t in range(T)], fontsize=8)
        ax_main.tick_params(axis="x", length=0)

        for spine in ax_main.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=8, length=2)

        self._add_value_labels(
            ax_main,
            ks_matrix,
            ks_min,
            ks_max,
            cmap,
            fig_w,
            fig_h,
            T,
            num_skills,
            heatmap_h,
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved user {user_id} heatmap to {output_path}")

        return fig

    def _draw_question_row(self, ax: plt.Axes, user_data: pd.DataFrame, T: int) -> None:
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Question"], fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for t, (_, row) in enumerate(user_data.iterrows()):
            question_id = int(row["question_id"])
            ax.text(
                t,
                0.5,
                self._question_label(question_id),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
            )

    def _draw_skill_row(
        self, ax: plt.Axes, user_data: pd.DataFrame, unique_skills: list[int], T: int
    ) -> None:
        """Draw the 'Skill' header row with multiple skills displayed vertically."""
        cmap20 = plt.get_cmap("tab20")
        skill_color = {s: cmap20(i % 20) for i, s in enumerate(unique_skills)}

        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Skill"], fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for t, (_, row) in enumerate(user_data.iterrows()):
            skills = row["skill"]

            # Display skills vertically, centered
            num_skills = min(len(skills), 3)  # Limit to prevent overflow
            if num_skills == 1:
                # Single skill: center at 0.5
                y_positions = [0.5]
            elif num_skills == 2:
                # Two skills: space evenly
                y_positions = [0.65, 0.35]
            else:
                # Three skills: space evenly
                y_positions = [0.75, 0.5, 0.25]

            for i, skill_id in enumerate(skills[:3]):
                ax.text(
                    t,
                    y_positions[i],
                    self._skill_label(skill_id),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=skill_color.get(skill_id, "black"),
                    fontweight="normal",
                )

            # Show "+N" indicator if more skills than display limit
            if len(skills) > 3:
                ax.text(
                    t,
                    0.1,
                    f"+{len(skills) - 3}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="gray",
                    fontstyle="italic",
                )

    def _draw_resp_row(self, ax: plt.Axes, user_data: pd.DataFrame, T: int) -> None:
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Resp"], fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for t, (_, row) in enumerate(user_data.iterrows()):
            correct = int(row["label"]) == 1
            ax.text(
                t,
                0.5,
                "✓" if correct else "✗",
                ha="center",
                va="center",
                fontsize=11,
                color="#2e7d32" if correct else "#c62828",
                fontweight="bold",
            )

    @staticmethod
    def _get_text_color_for_value(normalized_value: float, cmap: Colormap) -> str:
        rgb = cmap(normalized_value)[:3]
        y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return "black" if y > 0.5 else "white"

    def _add_value_labels(
        self,
        ax: plt.Axes,
        ks_matrix: np.ndarray,
        ks_min: float,
        ks_max: float,
        cmap: Colormap,
        fig_w: float,
        fig_h: float,
        T: int,
        num_skills: int,
        heatmap_h: float,
    ) -> None:
        dpi = 300
        cell_width_px = (fig_w / T) * dpi
        cell_height_px = (heatmap_h / fig_h * fig_w / num_skills) * dpi
        max_cell_size = min(cell_width_px, cell_height_px)
        font_size = max(6, min(10, max_cell_size / 4))

        for i in range(num_skills):
            for j in range(T):
                if not np.isnan(ks_matrix[i, j]):
                    value = ks_matrix[i, j]
                    normalized_value = (
                        (value - ks_min) / (ks_max - ks_min) if ks_max > ks_min else 0.5
                    )
                    text_color = self._get_text_color_for_value(normalized_value, cmap)
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                        color=text_color,
                        fontweight="normal",
                    )

    @staticmethod
    def _skill_label(skill_id: int) -> str:
        return f"c{skill_id}"

    @staticmethod
    def _question_label(question_id: int) -> str:
        return f"q{question_id}"


__all__ = ["HeatmapVisualizer"]
