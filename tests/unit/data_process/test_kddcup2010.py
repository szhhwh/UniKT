from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from utils.data_process.kddcup2010 import (
    Algebra2005Data,
    Algebra2006Data,
    Bridge2006Data,
)


@pytest.fixture
def raw_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Anon Student Id": ["student"] * 4,
            "Problem Hierarchy": ["unit1", "unit2", "unit1", "unit1"],
            "Problem Name": ["EG40", "EG40", "EG40", "EG55"],
            "Step Name": ["FinalAnswer", "FinalAnswer", "x = 1", "FinalAnswer"],
            "First Transaction Time": [f"2005-09-01 10:00:0{i}.000" for i in range(4)],
            "Correct First Attempt": [1, 0, 1, 0],
            "KC(Default)": ["skill"] * 4,
            "KC(SubSkills)": ["skill"] * 4,
        }
    )


@pytest.mark.parametrize(
    ("source_class", "expected_questions"),
    [(Algebra2005Data, 3), (Algebra2006Data, 4), (Bridge2006Data, 4)],
)
def test_question_identity_across_curriculum_contexts(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
    expected_questions: int,
) -> None:
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    source.raw_data = raw_frame.lazy()

    source.clean_raw_data()
    source.transform_data()

    sequence = source.sequence_data.sort("timestamp")
    questions = sequence["question"].to_list()
    assert sequence.height == 4
    assert sequence["label"].to_list() == [1, 0, 1, 0]
    assert len(set(questions)) == expected_questions
    assert len({questions[0], questions[2], questions[3]}) == 3
    source._validate_data(source.relation_data, sequence)


@pytest.mark.parametrize("hierarchy_present", [True, False])
def test_algebra2005_accepts_missing_curriculum_context(
    tmp_path: Path, raw_frame: pl.DataFrame, hierarchy_present: bool
) -> None:
    if hierarchy_present:
        raw_frame = raw_frame.with_columns(pl.lit(None).alias("Problem Hierarchy"))
    else:
        raw_frame = raw_frame.drop("Problem Hierarchy")
    source = Algebra2005Data(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    source.raw_data = raw_frame.lazy()

    source.clean_raw_data()
    source.transform_data()

    assert source.sequence_data.height == 4
    assert source.sequence_data["question"].n_unique() == 3
