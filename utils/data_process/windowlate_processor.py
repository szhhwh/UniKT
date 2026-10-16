"""Windowlate data processor for the windowlate AUC metric."""

import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import tqdm

from utils.core import get_logger

if TYPE_CHECKING:
    # polars 1.x 把 DataTypeClass 别名标记为废弃且从存根中移除, 此处按本文件
    # 实际用到的 dtype 定义等价的窄化别名, 保持 schema 传参可类型检查.
    from polars import DataType, Int8, Int32, Int64

    CoreDType = type[Int64] | type[Int32] | type[Int8]
    DataTypeSpec = CoreDType | DataType

logger = get_logger(__name__)


class WindowlateProcessor:
    """Build windowlate evaluation data.

    - Interactions without prior skill history are excluded from evaluation.
      Their skills and labels remain available as history for later targets.
    - Each remaining target KC generates one evaluable window (at the end).
    - History positions serve as context only (mask=0).
    - Target positions are evaluated (mask=1).
    - When a sequence exceeds max_seq_len, only the last window ending at the
      target position is kept: with ``truncation_stage="post_expansion"`` the
      window holds at most ``max_seq_len`` expanded skill rows; with
      ``truncation_stage="pre_expansion"`` it holds the last ``max_seq_len``
      whole interactions, whose skill rows stay together (possibly exceeding
      ``max_seq_len``).
    - All extra columns from the source sequence_data are preserved as-is
      (target position retains its true value).
    """

    # ===== Data structure definitions =====
    # Core output columns: mapped from source data, not preserved as extra columns.
    CORE_DTYPE_MAP: ClassVar[dict[str, "CoreDType"]] = {
        "sample_id": pl.Int64,
        "position": pl.Int32,
        "skill": pl.Int32,
        "question": pl.Int32,
        "response": pl.Int8,
        "mask": pl.Int8,
        "user_id": pl.Int32,
        "group_id": pl.Int64,
        "true_label": pl.Int8,
        "fold": pl.Int32,
    }
    CORE_SAMPLE_COLUMNS: ClassVar[list[str]] = [
        col for col in CORE_DTYPE_MAP if col != "fold"
    ]
    # Source columns already mapped to core columns; remaining columns are preserved as-is.
    RESERVED_COLUMNS: ClassVar[set[str]] = {
        "user",
        "question",
        "label",
        "skills",
        "fold",
    }
    CHUNK_ROW_LIMIT: int = 500_000

    # Configured per build: extra source columns to preserve as-is and their dtypes.
    EXTRA_COLUMNS: ClassVar[list[str]] = []
    EXTRA_DTYPES: ClassVar[dict[str, "DataTypeSpec"]] = {}

    @classmethod
    def _init_worker(
        cls, extra_columns: list[str], extra_dtypes: dict[str, "DataTypeSpec"]
    ) -> None:
        """Initialize worker processes with the extra column configuration."""
        cls.EXTRA_COLUMNS = extra_columns
        cls.EXTRA_DTYPES = extra_dtypes

    # ===== Core algorithm =====

    @staticmethod
    def count_user_samples(skills_list: list[list[int]], max_seq_len: int) -> int:
        """Count the number of samples for a single user without generating rows.

        Args:
            skills_list: Skill list for each interaction.
            max_seq_len: Maximum sequence length.

        Returns:
            Number of target skills after the first interaction with skills.
        """
        _ = max_seq_len  # Window length does not change the target count.
        total_skills = sum(len(q_skills) for q_skills in skills_list)
        first_interaction_skills = next(
            (len(q_skills) for q_skills in skills_list if q_skills), 0
        )
        return total_skills - first_interaction_skills

    @classmethod
    def generate_user_samples(
        cls,
        user_id: int,
        labels: list[int],
        skills_list: list[list[int]],
        questions: list[int],
        sample_id_start: int,
        group_id_start: int,
        max_seq_len: int,
        extras: dict[str, list],
        truncation_stage: str = "post_expansion",
    ) -> Iterator[list[tuple]]:
        """Generate evaluation samples for a user, excluding targets without history.

        Args:
            user_id: User ID.
            labels: Correctness label for each interaction.
            skills_list: Skill list for each interaction.
            questions: Question ID for each interaction.
            sample_id_start: Starting sample ID.
            group_id_start: Starting group ID.
            max_seq_len: Maximum sequence length; what it counts follows
                ``truncation_stage``.
            extras: Extra source columns to preserve as-is, keyed by column name.
            truncation_stage: Stage of the KC expansion at which the window
                is truncated. ``"post_expansion"`` keeps at most
                ``max_seq_len`` expanded skill rows; ``"pre_expansion"``
                keeps the last ``max_seq_len`` whole interactions (target one
                included), whose skill rows stay together.

        Yields:
            list[tuple]: All rows for one complete sample, each row formatted as
                (sample_id, position, skill, question, response, mask, user_id,
                 group_id, true_label, *extra_values)
        """
        extra_columns = cls.EXTRA_COLUMNS
        # Expand skills, labels, and all extra columns (aligned by skill expansion)
        expanded_skills = []
        expanded_questions = []
        expanded_labels = []
        expanded_group_ids = []
        expanded_extras: dict[str, list] = {col: [] for col in extra_columns}
        inter_boundaries = [0]

        for i, (q_skills, label) in enumerate(zip(skills_list, labels)):
            question_id = questions[i]
            for skill in q_skills:
                expanded_skills.append(skill)
                expanded_questions.append(question_id)
                expanded_labels.append(label)
                expanded_group_ids.append(group_id_start + i)
                for col in extra_columns:
                    expanded_extras[col].append(extras[col][i])
            inter_boundaries.append(inter_boundaries[-1] + len(q_skills))

        if not expanded_skills:
            return

        sample_id = sample_id_start
        num_interactions = len(skills_list)

        for inter_idx in range(num_interactions):
            history_end = inter_boundaries[inter_idx]
            if history_end == 0:
                continue

            n_skills = len(skills_list[inter_idx])
            for skill_offset in range(n_skills):
                current_skill_pos = inter_boundaries[inter_idx] + skill_offset
                current_skill = expanded_skills[current_skill_pos]
                current_question = expanded_questions[current_skill_pos]
                current_label = expanded_labels[current_skill_pos]
                current_group_id = expanded_group_ids[current_skill_pos]
                current_extras = {
                    col: expanded_extras[col][current_skill_pos]
                    for col in extra_columns
                }

                # Build prediction sequence: history + current skill (response=0 to prevent leakage).
                # Extra columns at the target position retain their true value (same as true_label).
                full_skills = [*expanded_skills[:history_end], current_skill]
                full_questions = [*expanded_questions[:history_end], current_question]
                full_labels = [*expanded_labels[:history_end], 0]
                full_group_ids = [*expanded_group_ids[:history_end], current_group_id]
                full_true_labels = [*expanded_labels[:history_end], current_label]
                full_extras = {
                    col: [*expanded_extras[col][:history_end], current_extras[col]]
                    for col in extra_columns
                }

                # Keep only the window ending at the target position.
                if truncation_stage == "pre_expansion":
                    # Keep the last max_seq_len whole interactions (the
                    # target's own interaction included): every kept
                    # interaction's skills stay together, so the window's
                    # row count may exceed max_seq_len.
                    win_start = inter_boundaries[max(0, inter_idx - max_seq_len + 1)]
                else:
                    # Keep the last max_seq_len expanded skill rows.
                    win_start = max(0, len(full_skills) - max_seq_len)
                win_skills = full_skills[win_start:]
                win_questions = full_questions[win_start:]
                win_labels = full_labels[win_start:]
                win_group_ids = full_group_ids[win_start:]
                win_true_labels = full_true_labels[win_start:]
                win_extras = {
                    col: full_extras[col][win_start:] for col in extra_columns
                }

                target_pos = len(win_skills) - 1
                rows = []
                for pos in range(len(win_skills)):
                    row = [
                        sample_id,
                        pos,
                        win_skills[pos],
                        win_questions[pos],
                        win_labels[pos],
                        1 if pos == target_pos else 0,
                        user_id,
                        win_group_ids[pos],
                        win_true_labels[pos],
                    ]
                    for col in extra_columns:
                        row.append(win_extras[col][pos])
                    rows.append(tuple(row))
                yield rows
                sample_id += 1

    # ===== Batch processing =====

    @classmethod
    def process_user_batch(
        cls,
        args: tuple,
    ) -> tuple[int, str | None, int, int]:
        """Process a batch of users, streaming into a single parquet file.

        Args:
            args: (batch_idx, batch_users, max_seq_len, chunk_row_limit,
                output_dir, truncation_stage)

        Returns:
            tuple: (batch_idx, output_path | None, total_rows, max_window_len)
        """
        (
            batch_idx,
            batch_users,
            max_seq_len,
            chunk_row_limit,
            output_dir,
            truncation_stage,
        ) = args

        if not batch_users:
            return batch_idx, None, 0, 0

        output_path = os.path.join(
            output_dir, f"windowlate_worker_{batch_idx:05d}.parquet"
        )
        writer = None
        total_rows = 0
        max_window_len = 0

        sample_columns = cls.CORE_SAMPLE_COLUMNS + cls.EXTRA_COLUMNS
        # Initialize buffers (fold is filled with a constant, not buffered)
        buffers: dict[str, list[Any]] = {col: [] for col in sample_columns}

        try:
            for (
                user_id,
                labels,
                skills_list,
                questions,
                sample_id_start,
                group_id_start,
                extras,
            ) in batch_users:
                for sample_rows in cls.generate_user_samples(
                    user_id,
                    labels,
                    skills_list,
                    questions,
                    sample_id_start,
                    group_id_start,
                    max_seq_len,
                    extras,
                    truncation_stage,
                ):
                    for row in sample_rows:
                        for i, col in enumerate(sample_columns):
                            buffers[col].append(row[i])

                    max_window_len = max(max_window_len, len(sample_rows))
                    if len(buffers["sample_id"]) >= chunk_row_limit:
                        writer = cls._flush_buffers(buffers, writer, output_path)
                        total_rows += len(buffers["sample_id"])
                        for col in buffers:
                            buffers[col].clear()

            # Final flush
            if buffers["sample_id"]:
                writer = cls._flush_buffers(buffers, writer, output_path)
                total_rows += len(buffers["sample_id"])

        finally:
            if writer is not None:
                writer.close()

        return (
            batch_idx,
            (output_path if total_rows > 0 else None),
            total_rows,
            max_window_len,
        )

    @classmethod
    def _flush_buffers(
        cls,
        buffers: dict[str, list],
        writer: pq.ParquetWriter | None,
        output_path: str,
    ) -> pq.ParquetWriter:
        """Flush buffered data to a parquet file."""
        data: dict[str, np.ndarray | list[Any]] = {
            "sample_id": np.asarray(buffers["sample_id"], dtype=np.int64),
            "position": np.asarray(buffers["position"], dtype=np.int32),
            "skill": np.asarray(buffers["skill"], dtype=np.int32),
            "question": np.asarray(buffers["question"], dtype=np.int32),
            "response": np.asarray(buffers["response"], dtype=np.int8),
            "mask": np.asarray(buffers["mask"], dtype=np.int8),
            "user_id": np.asarray(buffers["user_id"], dtype=np.int32),
            "group_id": np.asarray(buffers["group_id"], dtype=np.int64),
            "true_label": np.asarray(buffers["true_label"], dtype=np.int8),
        }
        # Extra columns preserved as-is; dtype conversion handled by schema
        for col in cls.EXTRA_COLUMNS:
            data[col] = buffers[col]
        data["fold"] = np.full(len(buffers["sample_id"]), -1, dtype=np.int32)

        chunk_df = pl.DataFrame(data, schema={**cls.CORE_DTYPE_MAP, **cls.EXTRA_DTYPES})
        chunk_table = chunk_df.to_arrow()

        if writer is None:
            writer = pq.ParquetWriter(
                output_path, chunk_table.schema, compression="NONE"
            )
        writer.write_table(chunk_table)
        return writer

    # ===== High-level interface =====

    @classmethod
    def build(
        cls,
        test_data: pl.DataFrame,
        max_seq_len: int,
        output_path: str,
        num_workers: int = 0,
        users_per_batch: int = 64,
        truncation_stage: str = "post_expansion",
    ) -> int:
        """Build windowlate data and write directly to file.

        Args:
            test_data: Test set sequence data.
            max_seq_len: Maximum sequence length; what it counts follows
                ``truncation_stage``.
            output_path: Output file path (streamed write).
            num_workers: Number of parallel workers (0 or negative for auto).
            users_per_batch: Number of users per batch.
            truncation_stage: Stage of the KC expansion at which windows are
                truncated -- ``"post_expansion"`` (expanded skill rows) or
                ``"pre_expansion"`` (whole interactions kept together).

        Returns:
            The largest window row count across all generated samples; this
            is the padded length evaluation must allocate.
        """
        required = {
            "user",
            "question",
            "label",
            "timestamp",
            "skills",
            "interaction_id",
        }
        if missing := required - set(test_data.columns):
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")
        if test_data.filter(
            pl.col("skills").is_null()
            | (pl.col("skills").list.len() == 0)
            | pl.col("skills").list.eval(pl.element().is_null()).list.any()
        ).height:
            raise ValueError("Interaction skills must be nonempty without nulls")
        sorted_test_data = test_data.sort(["user", "timestamp", "interaction_id"])

        # Configure dynamic schema: preserve all source columns not mapped to core columns
        cls.EXTRA_COLUMNS = [
            c for c in test_data.columns if c not in cls.RESERVED_COLUMNS
        ]
        cls.EXTRA_DTYPES = {c: test_data.schema[c] for c in cls.EXTRA_COLUMNS}
        if cls.EXTRA_COLUMNS:
            logger.debug(f"Windowlate preserving extra columns: {cls.EXTRA_COLUMNS}")

        # Determine worker count
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)

        # Write intermediate chunks to the output file's directory to avoid system temp
        tmp_base_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        os.makedirs(tmp_base_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="windowlate_chunks_", dir=tmp_base_dir
        ) as tmp_dir:
            # Preprocess: extract user records and compute offsets
            user_records, global_sample_id, _global_group_id = (
                cls._prepare_user_records(sorted_test_data, max_seq_len)
            )

            if not user_records:
                raise ValueError("No valid windowlate evaluation samples for test set")

            # Build batch inputs
            batch_inputs = cls._build_batch_inputs(
                user_records, max_seq_len, users_per_batch, tmp_dir, truncation_stage
            )

            logger.debug(
                f"Windowlate parallel workers={num_workers}, batches={len(batch_inputs)}, "
                f"users={len(user_records)}"
            )

            # Parallel processing
            worker_results = cls._parallel_process(batch_inputs, num_workers)

            # Merge directly to final output path
            total_rows, max_window_len = cls._merge_results(worker_results, output_path)

        logger.debug(
            f"Built windowlate data: {global_sample_id} samples, {total_rows} rows, "
            f"max_window_len={max_window_len}"
        )
        return max_window_len

    @classmethod
    def _prepare_user_records(
        cls,
        sorted_test_data: pl.DataFrame,
        max_seq_len: int,
    ) -> tuple[list, int, int]:
        """Preprocess user records and compute ID offsets."""
        user_records = []
        global_sample_id = 0
        global_group_id = 0

        def _normalize_group_key(group_key: Any) -> Any:
            # Polars group_by iterator returns tuple keys even for single grouping col.
            if isinstance(group_key, tuple):
                return group_key[0]
            return group_key

        extra_columns = cls.EXTRA_COLUMNS
        user_groups = sorted_test_data.group_by("user", maintain_order=True)
        for group_key, user_df in user_groups:
            user = _normalize_group_key(group_key)
            labels = user_df["label"].to_list()
            skills_list = user_df["skills"].to_list()
            questions = user_df["question"].to_list()
            extras = {col: user_df[col].to_list() for col in extra_columns}

            sample_count = cls.count_user_samples(skills_list, max_seq_len)
            if sample_count:
                user_records.append(
                    (
                        user,
                        labels,
                        skills_list,
                        questions,
                        global_sample_id,
                        global_group_id,
                        extras,
                    )
                )
            global_sample_id += sample_count
            global_group_id += len(skills_list)

        return user_records, global_sample_id, global_group_id

    @classmethod
    def _build_batch_inputs(
        cls,
        user_records: list,
        max_seq_len: int,
        users_per_batch: int,
        tmp_dir: str,
        truncation_stage: str,
    ) -> list:
        """Build batch input parameters."""
        batch_inputs: list[tuple[int, list, int, int, str, str]] = []
        for idx in range(0, len(user_records), users_per_batch):
            batch_idx = len(batch_inputs)
            batch_users = user_records[idx : idx + users_per_batch]
            batch_inputs.append(
                (
                    batch_idx,
                    batch_users,
                    max_seq_len,
                    cls.CHUNK_ROW_LIMIT,
                    tmp_dir,
                    truncation_stage,
                )
            )
        return batch_inputs

    @classmethod
    def _parallel_process(
        cls,
        batch_inputs: list,
        num_workers: int,
    ) -> list:
        """Process batches in parallel."""
        from concurrent.futures import as_completed

        worker_results = [None] * len(batch_inputs)

        if num_workers == 1 or len(batch_inputs) == 1:
            for res in tqdm.tqdm(
                map(cls.process_user_batch, batch_inputs),
                total=len(batch_inputs),
                desc="Processing windowlate batches",
            ):
                worker_results[res[0]] = res
        else:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=partial(
                    cls._init_worker, cls.EXTRA_COLUMNS, cls.EXTRA_DTYPES
                ),
            ) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(cls.process_user_batch, inp): inp[0]
                    for inp in batch_inputs
                }
                # Collect results in completion order
                for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing windowlate batches",
                ):
                    res = future.result()
                    worker_results[res[0]] = res

        return worker_results

    @classmethod
    def _merge_results(
        cls,
        worker_results: list,
        output_path: str,
    ) -> tuple[int, int]:
        """Merge all worker results into the final output file.

        Returns:
            tuple: (total_rows, max_window_len) across all workers.
        """
        tmp_path = output_path + ".tmp"
        final_writer = None
        total_rows = 0
        max_window_len = 0
        has_written_rows = False

        logger.info("Saving windowlate data to output path")

        try:
            for item in worker_results:
                if item is None:
                    continue
                _, worker_path, worker_rows, worker_max_window = item
                if worker_path is None:
                    continue

                pq_file = pq.ParquetFile(worker_path)
                for rg_idx in range(pq_file.num_row_groups):
                    table = pq_file.read_row_group(rg_idx)
                    if final_writer is None:
                        final_writer = pq.ParquetWriter(tmp_path, table.schema)
                    final_writer.write_table(table)
                    has_written_rows = True
                total_rows += worker_rows
                max_window_len = max(max_window_len, worker_max_window)
        finally:
            if final_writer is not None:
                final_writer.close()

        if not has_written_rows:
            raise ValueError("No valid windowlate evaluation samples generated")

        os.replace(tmp_path, output_path)
        return total_rows, max_window_len
