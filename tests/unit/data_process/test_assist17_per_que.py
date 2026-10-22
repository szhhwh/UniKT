from types import SimpleNamespace

import polars as pl

from utils.data_process.assist17_per_que import Assistments2017PerQueData


def test_encounter_skill_follows_the_action_supplying_its_label() -> None:
    source = Assistments2017PerQueData.__new__(Assistments2017PerQueData)
    source.args = SimpleNamespace(min_seq_len=1)
    source.raw_data = pl.DataFrame(
        {
            "studentId": [0] * 6,
            "problemId": [10, 10, 20, 20, 30, 30],
            "assignmentId": [0] * 6,
            "action_num": [0, 1, 0, 1, 0, 1],
            "skill": [100, 101, 200, 201, 300, 301],
            "correct": [None, 1, None, 0, None, None],
            "scaffold": [0, 0, 0, 1, 0, 0],
            "hint": [1, 0, 1, 0, 1, 1],
            "startTime": [0, 1, 10, 11, 20, 21],
            "timeTaken": [1] * 6,
        }
    ).lazy()

    source.clean_raw_data()

    assert source.cleaned_raw_data.sort("question").select(
        "question", "skill", "label"
    ).rows() == [(10, 101, 1), (20, 201, 0), (30, 300, 0)]
    assert source.cleaned_raw_data["interaction_id"].to_list() == [0, 2, 4]


def test_encounter_order_uses_first_observation_not_label_action() -> None:
    source = Assistments2017PerQueData.__new__(Assistments2017PerQueData)
    source.args = SimpleNamespace(min_seq_len=1)
    source.raw_data = pl.DataFrame(
        {
            "studentId": [0] * 5,
            "problemId": [30, 10, 30, 20, 30],
            "assignmentId": [0] * 5,
            "action_num": [0, 0, 1, 0, 2],
            "skill": [300, 100, 301, 200, 302],
            "correct": [None, 0, 1, 1, 0],
            "scaffold": [0] * 5,
            "hint": [1, 0, 0, 0, 0],
            "startTime": [100, 100, 100, 100, 4001],
            "timeTaken": [3, 2, 1, 0, 4],
        }
    ).lazy()

    source.clean_raw_data()

    assert source.cleaned_raw_data.select(
        "interaction_id", "question", "skill", "label"
    ).rows() == [
        (0, 30, 301, 1),
        (1, 10, 100, 0),
        (3, 20, 200, 1),
        (4, 30, 302, 0),
    ]
