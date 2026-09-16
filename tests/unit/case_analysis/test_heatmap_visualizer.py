"""Tests for the knowledge-state heatmap visualizer: validation, guards, formatting.

``MPLBACKEND=Agg`` is set globally by this area's ``conftest.py`` (imported
before test modules), so no local os.environ tweak is needed here.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from utils.case_analysis.visualizers.heatmap_visualizer import HeatmapVisualizer


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _row(
    position: int,
    question_id: int,
    skills: list[int],
    label: int,
    ks: list[float] | None,
) -> dict[str, Any]:
    return {
        "position": position,
        "question_id": question_id,
        "skill": skills,
        "label": label,
        "knowledge_state": ks,
    }


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def vis() -> HeatmapVisualizer:
    return HeatmapVisualizer()


def _main_axes(fig: Figure) -> Axes:
    # axes are created in order: question, skill, resp, main, colorbar
    return fig.axes[3]


def _main_image(fig: Figure) -> AxesImage:
    return _main_axes(fig).images[0]


def _is_nan_cell(image: AxesImage, row: int, col: int) -> bool:
    array = image.get_array()
    assert array is not None
    value = array[row, col]
    return np.ma.is_masked(value) or bool(np.isnan(float(value)))


class TestValidation:
    def test_missing_required_column_raises(self, vis: HeatmapVisualizer) -> None:
        df = _frame(
            [
                {
                    "position": 0,
                    "question_id": 1,
                    "skill": [0],
                    "label": 1,
                }
            ]
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            vis.plot_user_heatmap(df, user_id=7)

    def test_none_knowledge_state_raises(self, vis: HeatmapVisualizer) -> None:
        df = _frame(
            [
                _row(0, 1, [0], 1, [0.5]),
                _row(1, 2, [0], 0, None),
            ]
        )
        with pytest.raises(ValueError, match="knowledge_state"):
            vis.plot_user_heatmap(df, user_id=7)


class TestPlotBehaviour:
    def test_rows_sorted_by_position_before_plotting(
        self, vis: HeatmapVisualizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = _frame(
            [
                _row(2, 30, [0], 1, [0.1, 0.2, 0.3]),
                _row(0, 10, [1], 0, [0.4, 0.5, 0.6]),
                _row(1, 20, [2], 1, [0.7, 0.8, 0.9]),
            ]
        )
        captured: dict[str, list[int]] = {}
        dummy = plt.figure()

        def fake_plot(
            sorted_df: pd.DataFrame, user_id: int, output_path: str | None = None
        ) -> Figure:
            captured["positions"] = list(sorted_df["position"])
            return dummy

        monkeypatch.setattr(vis, "_plot_knowledge_state_heatmap", fake_plot)
        fig = vis.plot_user_heatmap(df, user_id=7)
        assert fig is dummy
        assert captured["positions"] == [0, 1, 2]

    def test_skill_dedup_keeps_first_seen_order(self, vis: HeatmapVisualizer) -> None:
        df = _frame(
            [
                _row(0, 1, [1, 2], 1, [0.9, 0.8, 0.7, 0.6]),
                _row(1, 2, [2, 3], 0, [0.4, 0.3, 0.2, 0.1]),
            ]
        )
        fig = vis.plot_user_heatmap(df, user_id=7)
        labels = [t.get_text() for t in _main_axes(fig).get_yticklabels()]
        assert labels == ["c1", "c2", "c3"]

    def test_short_knowledge_state_skips_out_of_range_skills(
        self, vis: HeatmapVisualizer
    ) -> None:
        # skills 0..2 but each ks list has only 2 entries: skill 2 stays NaN
        df = _frame(
            [
                _row(0, 1, [0, 1, 2], 1, [0.8, 0.2]),
            ]
        )
        fig = vis.plot_user_heatmap(df, user_id=7)
        image = _main_image(fig)
        assert not _is_nan_cell(image, 0, 0)
        assert not _is_nan_cell(image, 1, 0)
        assert _is_nan_cell(image, 2, 0)

    def test_degenerate_normalization_widens_range(
        self, vis: HeatmapVisualizer
    ) -> None:
        df = _frame(
            [
                _row(0, 1, [0], 1, [0.5, 0.5, 0.5]),
                _row(1, 2, [0], 0, [0.5, 0.5, 0.5]),
            ]
        )
        fig = vis.plot_user_heatmap(df, user_id=7)
        assert _main_image(fig).get_clim() == (pytest.approx(-0.5), pytest.approx(0.5))

    def test_all_nan_matrix_falls_back_to_unit_range(
        self, vis: HeatmapVisualizer
    ) -> None:
        df = _frame(
            [
                _row(0, 1, [5], 1, [0.4]),  # skill 5 >= len(ks): nothing plotted
            ]
        )
        fig = vis.plot_user_heatmap(df, user_id=7)
        assert _main_image(fig).get_clim() == (pytest.approx(0.0), pytest.approx(1.0))

    def test_output_path_written_under_tmp(
        self, vis: HeatmapVisualizer, tmp_path: Path
    ) -> None:
        df = _frame([_row(0, 1, [0], 1, [0.2, 0.9])])
        out = tmp_path / "sub" / "user_7.png"
        fig = vis.plot_user_heatmap(df, user_id=7, output_path=str(out))
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)

    def test_more_than_three_skills_show_plus_n(self, vis: HeatmapVisualizer) -> None:
        df = _frame([_row(0, 1, [0, 1, 2, 3, 4], 1, [0.5] * 5)])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        vis._draw_skill_row(ax, df, unique_skills=[0, 1, 2, 3, 4], T=1)
        texts = [t.get_text() for t in ax.texts]
        assert "+2" in texts
        # only the first three skills get a visible label
        assert texts.count("c0") == 1 and "c3" not in texts


class TestFormattingHelpers:
    def test_text_color_luminance_boundary(self, vis: HeatmapVisualizer) -> None:
        cmap = plt.get_cmap("RdYlGn")
        # dark red / dark green ends are dark: white text; yellow middle: black
        assert HeatmapVisualizer._get_text_color_for_value(0.0, cmap) == "white"
        assert HeatmapVisualizer._get_text_color_for_value(1.0, cmap) == "white"
        assert HeatmapVisualizer._get_text_color_for_value(0.5, cmap) == "black"

    def test_skill_and_question_labels(self) -> None:
        assert HeatmapVisualizer._skill_label(5) == "c5"
        assert HeatmapVisualizer._question_label(12) == "q12"

    def test_plot_user_delegates_to_plot_user_heatmap(
        self, vis: HeatmapVisualizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = _frame([_row(0, 1, [0], 1, [0.5])])
        seen: dict[str, int] = {}

        def fake(
            user_data: pd.DataFrame, user_id: int, output_path: str | None = None
        ) -> str:
            seen["user_id"] = user_id
            return "FIG"

        monkeypatch.setattr(vis, "plot_user_heatmap", fake)
        assert vis.plot_user(df, 3) == "FIG"
        assert seen["user_id"] == 3
