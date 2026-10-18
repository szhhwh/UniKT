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
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
def test_question_identity_across_curriculum_contexts(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
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
    assert questions[0] == questions[1]
    assert len(set(questions)) == 3
    assert questions[0] != questions[2]
    assert questions[0] != questions[3]
    source._validate_data(source.relation_data, sequence)


@pytest.mark.parametrize("hierarchy_present", [True, False])
@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
def test_accepts_missing_curriculum_context(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    hierarchy_present: bool,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
) -> None:
    if hierarchy_present:
        raw_frame = raw_frame.with_columns(pl.lit(None).alias("Problem Hierarchy"))
    else:
        raw_frame = raw_frame.drop("Problem Hierarchy")
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    source.raw_data = raw_frame.lazy()

    source.clean_raw_data()
    source.transform_data()

    assert source.sequence_data.height == 4
    assert source.sequence_data["question"].n_unique() == 3


@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
def test_question_keys_escape_and_join_names(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
) -> None:
    # Escape both fields before joining them with "----".
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    source.raw_data = raw_frame.with_columns(
        pl.Series("Problem Name", ["P_1,a", "P_1,a", "P_1,a", "P_2,a"]),
        pl.Series("Step Name", ["S_1,b", "S_1,b", "S_2,b", "S_1,b"]),
    ).lazy()

    source.clean_raw_data()
    assert source.cleaned_raw_data["question"].to_list() == [
        "P####1@@@@a----S####1@@@@b",
        "P####1@@@@a----S####1@@@@b",
        "P####1@@@@a----S####2@@@@b",
        "P####2@@@@a----S####1@@@@b",
    ]
    source.transform_data()
    assert source.sequence_data["question"].to_list() == [0, 0, 1, 2]


@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize("column", ["Problem Name", "Step Name"])
def test_requires_question_columns(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
    from_file: bool,
    column: str,
) -> None:
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    if from_file:
        path = Path(source.raw_data_path)
        path.parent.mkdir(parents=True)
        raw_frame.drop(column).write_csv(path, separator="\t")
    else:
        source.raw_data = raw_frame.drop(column).lazy()

    with pytest.raises(ValueError, match=rf"missing columns: \['{column}'\]"):
        source.clean_raw_data()


@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
def test_preserves_numeric_names_and_leading_zeros_from_file(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
) -> None:
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    path = Path(source.raw_data_path)
    path.parent.mkdir(parents=True)
    raw_frame.with_columns(
        pl.Series("Problem Name", ["001", "001", "1", "001"]),
        pl.Series("Step Name", ["01", "1", "01", "01"]),
    ).write_csv(path, separator="\t")

    source.clean_raw_data()
    assert source.cleaned_raw_data["question"].to_list() == [
        "001----01",
        "001----1",
        "1----01",
        "001----01",
    ]
    source.transform_data()
    assert source.sequence_data["question"].to_list() == [0, 1, 2, 0]


@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
@pytest.mark.parametrize("column", ["Problem Name", "Step Name"])
def test_drops_missing_question_components(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
    column: str,
) -> None:
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    source.raw_data = raw_frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
    ).lazy()

    source.clean_raw_data()
    source.transform_data()

    assert source.sequence_data["interaction_id"].to_list() == [0, 1, 3]
    assert source.sequence_data["question"].n_unique() == 2


@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
@pytest.mark.parametrize(
    "problem_names,step_names",
    [
        (["P_1", "P####1"], ["S", "S"]),
        (["P,1", "P@@@@1"], ["S", "S"]),
        (["A----B", "A"], ["C", "B----C"]),
    ],
)
def test_rejects_ambiguous_question_keys(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
    problem_names: list[str],
    step_names: list[str],
) -> None:
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    source.raw_data = (
        raw_frame.head(2)
        .with_columns(
            pl.Series("Problem Name", problem_names),
            pl.Series("Step Name", step_names),
        )
        .lazy()
    )

    with pytest.raises(ValueError, match="same question key"):
        source.clean_raw_data()


@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
def test_question_identity_is_independent_of_student_and_problem_view(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
) -> None:
    source = source_class(
        SimpleNamespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    source.raw_data = raw_frame.with_columns(
        pl.Series("Anon Student Id", ["s1", "s2", "s1", "s2"]),
        pl.Series("Problem View", [1, 2, 3, 4]),
        pl.lit("default_skill").alias("KC(Default)"),
        pl.lit("sub_skill").alias("KC(SubSkills)"),
    ).lazy()
    source.clean_raw_data()
    source.transform_data()

    sequence = source.sequence_data.sort("interaction_id")
    assert sequence["question"].to_list() == [0, 0, 1, 2]
    assert sequence["user"][0] != sequence["user"][1]
    expected_skill = "sub_skill" if source_class is Bridge2006Data else "default_skill"
    assert source._id_mappings["skill"] == {expected_skill: 0}


@pytest.mark.parametrize(
    "source_class", [Algebra2005Data, Algebra2006Data, Bridge2006Data]
)
@pytest.mark.parametrize("stage", ["post_expansion", "pre_expansion"])
def test_preserves_observed_skills_and_interaction_order(
    tmp_path: Path,
    raw_frame: pl.DataFrame,
    source_class: type[Algebra2005Data | Algebra2006Data | Bridge2006Data],
    stage: str,
) -> None:
    source = source_class(
        SimpleNamespace(
            data_base_path=str(tmp_path),
            seed=42,
            min_seq_len=1,
            max_seq_len=3,
            truncation_stage=stage,
        )
    )
    source.raw_data = raw_frame.with_columns(
        pl.Series("KC(Default)", ["z~~a", "b", "c", "a"]),
        pl.Series("KC(SubSkills)", ["z~~a", "b", "c", "a"]),
        pl.Series(
            "First Transaction Time",
            [
                "2005-09-01 10:00:00.000",
                "2005-09-01 10:00:00.000",
                "2005-09-01 10:00:01.000",
                "2005-09-01 10:00:02.000",
            ],
        ),
    ).lazy()
    source.clean_raw_data()
    source.transform_data()
    source._validate_data(source.relation_data, source.sequence_data)
    source.build_split_question_sequence_data()
    source.build_split_skill_sequence_data()

    questions = source.split_question_sequence_data
    assert questions["interaction_id"].to_list() == [0, 1, 2, 3]
    assert questions["label"].to_list() == [1, 0, 1, 0]
    assert questions["skills"].to_list() == [[0, 3], [1], [2], [0]]
    assert questions["question"][0] == questions["question"][1]
    relation = source.relation_data["question_skill"].filter(
        pl.col("question") == questions["question"][0]
    )
    assert relation["skill"].sort().to_list() == [0, 1, 3]
    step_relation = source.relation_data["question_skill"].filter(
        pl.col("question") == questions["question"][2]
    )
    assert step_relation["skill"].to_list() == [2]

    skills = source.split_skill_sequence_data
    assert "skills" not in skills.columns
    assert skills["skill"].to_list() == [0, 3, 1, 2, 0]
    assert skills["label"].to_list() == [1, 1, 0, 1, 0]
    assert skills["interaction_id"].to_list() == [0, 0, 1, 2, 3]
    lengths = skills.group_by("sequence_id").len().sort("sequence_id")["len"]
    assert lengths.to_list() == ([3, 2] if stage == "post_expansion" else [4, 1])

    Path(source.data_folder).mkdir(parents=True)
    source.save_data()
    saved = pl.read_parquet(
        Path(source.data_folder) / f"{source.dataset}_sequence.parquet"
    )
    assert saved["skills"].to_list() == [[0, 3], [1], [2], [0]]
    assert saved["interaction_id"].to_list() == [0, 1, 2, 3]
