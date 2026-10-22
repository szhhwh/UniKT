import json
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from utils.data_process.assist09 import Assistments2009Data
from utils.data_process.assist12 import Assistments2012Data
from utils.data_process.assist15 import Assistments2015Data
from utils.data_process.assist17 import Assistments2017Data
from utils.data_process.assist17_per_que import Assistments2017PerQueData
from utils.data_process.data_source import DataSource
from utils.data_process.ednet_kt1 import EdNetKT1Data
from utils.data_process.junyi2015 import Junyi2015Data
from utils.data_process.kddcup2010 import (
    Algebra2005Data,
    Algebra2006Data,
    Bridge2006Data,
)
from utils.data_process.moocradar import MoocRadarData
from utils.data_process.nips2020_t34 import NIPS2020T34Data
from utils.data_process.practiceanatomy import PracticeAnatomyData
from utils.data_process.slepemapy import SlepemapyData
from utils.data_process.windowlate_processor import WindowlateProcessor
from utils.data_process.xes3g5m import Xes3g5mData

_SOURCE_CLASSES: list[Callable[[Namespace], DataSource]] = [
    Assistments2009Data,
    Assistments2012Data,
    Assistments2015Data,
    Assistments2017Data,
    Assistments2017PerQueData,
    Algebra2005Data,
    Algebra2006Data,
    Bridge2006Data,
    EdNetKT1Data,
    Junyi2015Data,
    MoocRadarData,
    NIPS2020T34Data,
    PracticeAnatomyData,
    SlepemapyData,
    Xes3g5mData,
]

_ASSIST17_UNUSED_COLUMNS = (
    "MiddleSchoolId",
    "InferredGender",
    "SY ASSISTments Usage",
    "AveKnow",
    "AveCarelessness",
    "AveCorrect",
    "NumActions",
    "AveResBored",
    "AveResEngcon",
    "AveResConf",
    "AveResFrust",
    "AveResOfftask",
    "AveResGaming",
    "problemType",
    "assistmentId",
    "endTime",
    "hintTotal",
    "bottomHint",
    "frIsHelpRequest",
    "frPast5HelpRequest",
    "frPast8HelpRequest",
    "stlHintUsed",
    "past8BottomOut",
    "totalFrPercentPastWrong",
    "totalFrPastWrongCount",
    "frPast5WrongCount",
    "frPast8WrongCount",
    "totalFrTimeOnSkill",
    "timeSinceSkill",
    "frWorkingInSchool",
    "totalFrAttempted",
    "totalFrSkillOpportunities",
    "responseIsFillIn",
    "responseIsChosen",
    "endsWithScaffolding",
    "endsWithAutoScaffolding",
    "frTimeTakenOnScaffolding",
    "frTotalSkillOpportunitiesScaffolding",
    "totalFrSkillOpportunitiesByScaffolding",
    "frIsHelpRequestScaffolding",
    "timeGreater5Secprev2wrong",
    "sumRight",
    "helpAccessUnder2Sec",
    "timeGreater10SecAndNextActionRight",
    "consecutiveErrorsInRow",
    "sumTime3SDWhen3RowRight",
    "sumTimePerSkill",
    "totalTimeByPercentCorrectForskill",
    "Prev5count",
    "timeOver80",
    "manywrong",
    "confidence(BORED)",
    "confidence(CONCENTRATING)",
    "confidence(CONFUSED)",
    "confidence(FRUSTRATED)",
    "confidence(OFF TASK)",
    "confidence(GAMING)",
    "RES_BORED",
    "RES_CONCENTRATING",
    "RES_CONFUSED",
    "RES_FRUSTRATED",
    "RES_OFFTASK",
    "RES_GAMING",
    "Ln-1",
    "Ln",
    "MCAS",
    "Enrolled",
    "Selective",
    "isSTEM",
)


def _write_native_fixture(source: DataSource, labels: list[int]) -> list[int | str]:
    raw = Path(source.raw_folder)
    raw.mkdir(parents=True)
    questions: list[int | str] = [30, 10, 20]
    times = ["2020-01-01 00:00:00"] * 3
    if isinstance(source, (Assistments2009Data, Assistments2012Data)):
        frame = pl.DataFrame(
            {
                "user_id": [7] * 3,
                "problem_id": questions,
                "correct": labels,
                "skill_id": [3, 1, 2],
                "assignment_id": [0] * 3,
                "template_id": [0] * 3,
                "attempt_count": [3, 1, 2],
                "hint_count": [0] * 3,
                "ms_first_response": [3, 1, 2],
            }
        )
        if source.dataset == "assistments09":
            frame = frame.with_columns(
                pl.Series("skill_id", ["3_3", "1_1", "2_2"]),
                pl.lit(100).alias("order_id"),
            )
        else:
            frame = frame.with_columns(
                pl.lit("unused").alias("skill"), pl.Series("start_time", times)
            )
        frame.write_csv(source.raw_data_path)
    elif isinstance(source, Assistments2015Data):
        pl.DataFrame(
            {
                "user_id": [7] * 3,
                "sequence_id": questions,
                "log_id": [100] * 3,
                "correct": labels,
            }
        ).write_csv(source.raw_data_path)
    elif isinstance(source, (Assistments2017Data, Assistments2017PerQueData)):
        pl.DataFrame(
            {
                "studentId": [7] * 3,
                "problemId": questions,
                "assignmentId": [0] * 3,
                "action_num": [0] * 3,
                "skill": [3, 1, 2],
                "correct": labels,
                "scaffold": [0] * 3,
                "hint": [0] * 3,
                "startTime": [100] * 3,
                "timeTaken": [3, 1, 2],
                "hintCount": [0] * 3,
                "attemptCount": [3, 1, 2],
            }
        ).with_columns(pl.lit(0).alias(c) for c in _ASSIST17_UNUSED_COLUMNS).write_csv(
            source.raw_data_path
        )
    elif isinstance(source, (Algebra2005Data, Algebra2006Data, Bridge2006Data)):
        questions = ["z", "a", "m"]
        pl.DataFrame(
            {
                "Anon Student Id": ["student"] * 3,
                "Problem Name": questions,
                "Step Name": ["FinalAnswer"] * 3,
                "First Transaction Time": times,
                "Correct First Attempt": labels,
                source.skill_column: ["skill3", "skill1", "skill2"],
            }
        ).write_csv(source.raw_data_path, separator="\t")
        questions = [f"{question}----FinalAnswer" for question in questions]
    elif isinstance(source, EdNetKT1Data):
        contents = raw / "EdNet-Contents"
        logs = raw / "EdNet-KT1" / "KT1"
        contents.mkdir()
        logs.mkdir(parents=True)
        questions = ["q30", "q10", "q20"]
        pl.DataFrame(
            {
                "question_id": questions,
                "correct_answer": ["a"] * 3,
                "tags": ["3", "1", "2"],
                "bundle_id": [0] * 3,
            }
        ).write_csv(contents / "questions.csv")
        pl.DataFrame(
            {
                "question_id": questions,
                "user_answer": ["a" if label else "b" for label in labels],
                "timestamp": [100] * 3,
            }
        ).write_csv(logs / "u7.csv")
    elif isinstance(source, Junyi2015Data):
        questions = ["z", "a", "m"]
        pl.DataFrame(
            {"name": questions, "topic": ["3", "1", "2"], "area": ["area"] * 3}
        ).write_csv(raw / "junyi_Exercise_table.csv")
        pl.DataFrame(
            {
                "user_id": [7] * 3,
                "exercise": questions,
                "correct": ["true" if label else "false" for label in labels],
                "time_done": [100_000] * 3,
                "count_attempts": [3, 1, 2],
                "count_hints": [0] * 3,
                "time_taken_attempts": ["3&1", "1&1", "2&1"],
            }
        ).write_csv(raw / "junyi_ProblemLog_original.csv")
    elif isinstance(source, MoocRadarData):
        questions = ["z", "a", "m"]
        records = [
            {
                "user_id": "student",
                "problem_id": question,
                "is_correct": label,
                "submit_time": time,
                "log_id": f"student_{i}",
            }
            for i, (question, label, time) in enumerate(zip(questions, labels, times))
        ]
        Path(source.fine_path).write_text(json.dumps([{"seq": records}]))
        Path(source.problem_path).write_text(
            "\n".join(
                json.dumps({"problem_id": question, "concepts": [str(i)]})
                for i, question in enumerate(questions)
            )
        )
    elif isinstance(source, NIPS2020T34Data):
        pl.DataFrame(
            {
                "UserId": [7] * 3,
                "QuestionId": questions,
                "AnswerId": [2, 0, 1],
                "IsCorrect": labels,
            }
        ).write_csv(raw / "train_task_3_4.csv")
        pl.DataFrame({"AnswerId": [0, 1, 2], "DateAnswered": times}).write_csv(
            raw / "answer_metadata_task_3_4.csv"
        )
        pl.DataFrame({"QuestionId": questions, "SubjectId": ["[3]"] * 3}).write_csv(
            raw / "question_metadata_task_3_4.csv"
        )
        pl.DataFrame({"SubjectId": [3], "Level": [3]}).write_csv(
            raw / "subject_metadata.csv"
        )
    elif isinstance(source, (PracticeAnatomyData, SlepemapyData)):
        asked = "item_asked" if source.dataset == "practiceanatomy" else "place_asked"
        answered = (
            "item_answered" if source.dataset == "practiceanatomy" else "place_answered"
        )
        time_col = "time" if source.dataset == "practiceanatomy" else "inserted"
        direction = "t2d" if source.dataset == "practiceanatomy" else 1
        pl.DataFrame(
            {
                "user": [7] * 3,
                asked: questions,
                answered: [q if label else 999 for q, label in zip(questions, labels)],
                "type": [direction] * 3,
                time_col: times,
            }
        ).write_csv(
            source.raw_data_path,
            separator="," if source.dataset == "practiceanatomy" else ";",
        )
        questions = [f"{q}_{direction}" for q in questions]
    elif isinstance(source, Xes3g5mData):
        Path(source.train_path).parent.mkdir()
        pl.DataFrame(
            {
                "fold": [0],
                "uid": [7],
                "questions": ["30,10,-1"],
                "concepts": ["3,1,-1"],
                "responses": [f"{labels[0]},{labels[1]},-1"],
                "timestamps": ["100,100,-1"],
                "selectmasks": ["1,1,-1"],
            }
        ).write_csv(source.train_path)
        pl.DataFrame(
            {
                "fold": [-1],
                "uid": [7],
                "questions": ["20,-1"],
                "concepts": ["2,-1"],
                "responses": [f"{labels[2]},-1"],
                "timestamps": ["100,-1"],
            }
        ).write_csv(source.test_path)
    else:
        raise AssertionError(source.dataset)
    return questions


@pytest.fixture(params=_SOURCE_CLASSES, ids=lambda cls: cls.__name__)
def native_source(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[DataSource, list[int | str]]:
    source = request.param(
        Namespace(
            data_base_path=str(tmp_path),
            seed=42,
            min_seq_len=1,
            max_seq_len=2,
            truncation_stage="post_expansion",
        )
    )
    questions = _write_native_fixture(source, [1, 0, 1])
    source.clean_raw_data()
    source.transform_data()
    return source, questions


def test_native_source_preserves_equal_timestamp_observations(
    native_source: tuple[DataSource, list[int | str]],
) -> None:
    source, raw_questions = native_source
    sequence = source.sequence_data
    assert sequence["interaction_id"].to_list() == [0, 1, 2]
    assert sequence["interaction_id"].dtype == pl.Int64
    assert sequence["interaction_id"].n_unique() == sequence.height
    assert sequence["timestamp"].n_unique() == 1
    assert sequence["question"].to_list() == [
        source._id_mappings["question"][q] for q in raw_questions
    ]
    assert sequence["label"].to_list() == [1, 0, 1]
    source._validate_data(source.relation_data, sequence)


@pytest.mark.parametrize("stage", ["pre_expansion", "post_expansion"])
def test_native_source_split_and_windowlate_preserve_order(
    native_source: tuple[DataSource, list[int | str]], tmp_path: Path, stage: str
) -> None:
    source, _ = native_source
    source.args.truncation_stage = stage
    source.build_split_question_sequence_data()
    source.build_split_skill_sequence_data()
    assert source.split_question_sequence_data["interaction_id"].to_list() == [0, 1, 2]
    assert source.split_skill_sequence_data["interaction_id"].to_list() == [0, 1, 2]
    output = tmp_path / "windowlate.parquet"
    previous_columns = WindowlateProcessor.EXTRA_COLUMNS
    previous_dtypes = WindowlateProcessor.EXTRA_DTYPES
    try:
        WindowlateProcessor.build(
            test_data=source.sequence_data.reverse(),
            max_seq_len=10,
            output_path=str(output),
            num_workers=1,
            truncation_stage=stage,
        )
    finally:
        WindowlateProcessor.EXTRA_COLUMNS = previous_columns
        WindowlateProcessor.EXTRA_DTYPES = previous_dtypes
    samples = pl.read_parquet(output)
    assert samples.filter(pl.col("mask") == 1)["interaction_id"].to_list() == [1, 2]
    assert (
        samples.group_by("sample_id", maintain_order=True)
        .agg(pl.col("interaction_id"))["interaction_id"]
        .to_list()
    ) == [[0, 1], [0, 1, 2]]


@pytest.mark.parametrize("source_class", _SOURCE_CLASSES, ids=lambda cls: cls.__name__)
def test_native_order_is_independent_of_labels(
    tmp_path: Path, source_class: Callable[[Namespace], DataSource]
) -> None:
    sequences = []
    for suffix, labels in [("original", [1, 0, 1]), ("changed", [0, 1, 0])]:
        source = source_class(
            Namespace(data_base_path=str(tmp_path / suffix), seed=42, min_seq_len=1)
        )
        _write_native_fixture(source, labels)
        source.clean_raw_data()
        source.transform_data()
        sequences.append(source.sequence_data.select("interaction_id", "question"))
    assert_frame_equal(*sequences)


def test_set_sequence_data_requires_source_identity(
    make_data_source: Callable[..., DataSource],
) -> None:
    source = make_data_source()
    data = pl.DataFrame(
        {
            "user": [0],
            "question": [0],
            "label": [1],
            "timestamp": [0],
            "skills": pl.Series([[0]], dtype=pl.List(pl.Int32)),
        }
    )
    with pytest.raises(ValueError, match="must provide interaction_id"):
        source.set_sequence_data(data)


@pytest.mark.parametrize("ids", [[1, 1], [1, None]])
def test_set_sequence_data_rejects_invalid_identity(
    make_data_source: Callable[..., DataSource], ids: list[int | None]
) -> None:
    source = make_data_source()
    data = pl.DataFrame(
        {
            "user": [0, 0],
            "question": [0, 0],
            "label": [1, 0],
            "timestamp": [0, 0],
            "interaction_id": ids,
            "skills": pl.Series([[0], [0]], dtype=pl.List(pl.Int32)),
        }
    )
    with pytest.raises(AssertionError, match="interaction_id"):
        source.set_sequence_data(data)


@pytest.mark.parametrize("stage", ["pre_expansion", "post_expansion"])
@pytest.mark.parametrize("change", ["label", "optional", "optional_added", "question"])
def test_equal_timestamp_order_ignores_answers_and_features(
    make_data_source: Callable[..., DataSource], stage: str, change: str, tmp_path: Path
) -> None:
    data = pl.DataFrame(
        {
            "user": [0] * 3,
            "question": [0] * 3,
            "label": [1, 0, 1],
            "timestamp": [0] * 3,
            "interaction_id": [22, 17, 19],
            "skills": pl.Series([[0, 1], [0], [1]], dtype=pl.List(pl.Int32)),
            "attempt_count": [8, 5, 2],
        }
    )
    if change == "label":
        data = data.with_columns(pl.Series("label", [0, 1, 0]))
    elif change == "optional":
        data = data.with_columns(pl.Series("attempt_count", [0, 8, 4]))
    elif change == "optional_added":
        data = data.with_columns(pl.Series("arbitrary_feature", [3.5, -1.2, 0.0]))
    else:
        data = data.with_columns(pl.Series("question", [0, 2, 1]))
    source = make_data_source(max_seq_len=10, truncation_stage=stage)
    source.set_sequence_data(data)
    source.build_split_question_sequence_data()
    source.build_split_skill_sequence_data()
    assert source.sequence_data["interaction_id"].to_list() == [17, 19, 22]
    assert source.split_question_sequence_data["interaction_id"].to_list() == [
        17,
        19,
        22,
    ]
    assert source.split_skill_sequence_data["interaction_id"].to_list() == [
        17,
        19,
        22,
        22,
    ]
    previous_columns = WindowlateProcessor.EXTRA_COLUMNS
    previous_dtypes = WindowlateProcessor.EXTRA_DTYPES
    try:
        WindowlateProcessor.build(
            test_data=data,
            max_seq_len=10,
            output_path=str(tmp_path / "windowlate.parquet"),
            num_workers=1,
            truncation_stage=stage,
        )
    finally:
        WindowlateProcessor.EXTRA_COLUMNS = previous_columns
        WindowlateProcessor.EXTRA_DTYPES = previous_dtypes
    targets = pl.read_parquet(tmp_path / "windowlate.parquet").filter(
        pl.col("mask") == 1
    )
    assert targets["interaction_id"].to_list() == [19, 22, 22]


def test_ednet_pipeline_identifies_rows_before_answer_join(tmp_path: Path) -> None:
    source = EdNetKT1Data(Namespace(data_base_path=str(tmp_path), seed=42))
    logs = pl.DataFrame(
        {
            "user_id": [7] * 3,
            "question_id": ["q30", "missing", "q10"],
            "user_answer": ["a", "b", "b"],
            "timestamp": [100] * 3,
        }
    ).lazy()
    result = source._process_lazy_pipeline(logs, {"q10": "a", "q30": "a"}).collect()
    assert result.sort("interaction_id").select("interaction_id", "label").rows() == [
        (0, 1),
        (2, 0),
    ]


def test_moocradar_uses_log_sequence_before_file_order(tmp_path: Path) -> None:
    source = MoocRadarData(
        Namespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    _write_native_fixture(source, [1, 0, 1])
    source.load_src_data()
    source.raw_data = source.raw_data.reverse()
    source.clean_raw_data()
    source.transform_data()
    assert source.sequence_data["label"].to_list() == [1, 0, 1]
    assert source.sequence_data["question"].to_list() == [2, 0, 1]


def test_moocradar_identity_keeps_duplicate_filter(tmp_path: Path) -> None:
    source = MoocRadarData(
        Namespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    _write_native_fixture(source, [1, 0, 1])
    source.load_src_data()
    duplicate = source.raw_data.head(1).with_columns(pl.lit(3).alias("log_seq"))
    source.raw_data = pl.concat([source.raw_data, duplicate], how="vertical_relaxed")
    source.clean_raw_data()
    source.transform_data()
    assert source.sequence_data["interaction_id"].to_list() == [0, 1, 2]
    assert source.sequence_data["label"].to_list() == [1, 0, 1]


def test_nips_duplicate_selection_uses_source_row_order(tmp_path: Path) -> None:
    source = NIPS2020T34Data(
        Namespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    _write_native_fixture(source, [1, 0, 1])
    source.load_src_data()
    rows = source.raw_data["primary"].collect()
    duplicate = rows.head(1).with_columns(
        pl.lit(0, dtype=rows.schema["IsCorrect"]).alias("IsCorrect")
    )
    source.raw_data["primary"] = pl.concat(
        [rows.head(1), duplicate, rows.slice(1)]
    ).lazy()
    source.clean_raw_data()
    source.transform_data()
    assert source.sequence_data["interaction_id"].to_list() == [0, 2, 3]
    assert source.sequence_data["label"].to_list() == [1, 0, 1]


@pytest.mark.parametrize(
    "source_class",
    [
        Assistments2009Data,
        Assistments2012Data,
        Assistments2015Data,
        Assistments2017Data,
        PracticeAnatomyData,
        SlepemapyData,
    ],
    ids=lambda cls: cls.__name__,
)
def test_source_identity_does_not_disable_existing_duplicate_filter(
    tmp_path: Path, source_class: Callable[[Namespace], DataSource]
) -> None:
    source = source_class(
        Namespace(data_base_path=str(tmp_path), seed=42, min_seq_len=1)
    )
    _write_native_fixture(source, [1, 0, 1])
    source.load_src_data()
    rows = source.raw_data.collect()
    source.raw_data = pl.concat([rows.head(1), rows]).lazy()
    source.clean_raw_data()
    source.transform_data()
    assert source.sequence_data["interaction_id"].to_list() == [0, 2, 3]
    assert source.sequence_data["label"].to_list() == [1, 0, 1]
