"""Base model data class for knowledge tracing datasets.

Provides the abstract foundation for preparing model-ready data, including
K-fold splitting, difficulty calculation, and relationship matrix building.
"""

import functools
import hashlib
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, overload

import numpy as np
import polars as pl

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data.packed_sequences import PackedSequences, pack_sequences

logger = get_logger(__name__)


class _SplitSequenceGetter(Protocol):
    """Signature shared by the DataSource split-sequence getters."""

    def __call__(
        self, required: Sequence[str] = (), optional: Sequence[str] = ()
    ) -> pl.DataFrame: ...


class BaseModelData(ABC):
    """Abstract base class for model data preparation.

    Args:
        data_src: Data source object.
        cache: Whether to enable disk caching for expensive operations.
    """

    def __init__(self, data_src: DataSource, cache: bool = False):
        """Initialise the base model data object.

        Args:
            data_src: Data source object providing raw dataset access.
            cache: Whether to enable disk caching for prepared data.
        """
        self.data_src = data_src
        self._cache = cache

    # Bound to the chain-specific split getter (question or skill) by
    # QuestionModelData / SkillModelData at construction.
    _split_getter: _SplitSequenceGetter

    def load_split_frame(
        self, required: Sequence[str] = (), optional: Sequence[str] = ()
    ) -> pl.DataFrame:
        """Load split interaction rows projected to the declared columns.

        Returns the structural core (sequence_id/seq_pos/label/user/question/
        interaction_id, plus skills or skill and fold when present) with the
        declared feature columns merged in.

        Args:
            required: Feature columns that must exist in the dataset;
                missing ones raise ValueError naming the dataset and the
                available columns.
            optional: Feature columns loaded only when present in the
                schema; absent ones are simply not in the frame -- callers
                handle the fallback.
        """
        return self._split_getter(required=required, optional=optional)

    _sequence_length_key: str

    def pack_sequences(self, data: pl.DataFrame) -> PackedSequences:
        """Pack loaded rows with their sequence identity and fold membership."""
        return pack_sequences(
            data,
            max_seq_len=self.data_src.get_metadata(self._sequence_length_key),
            num_folds=self.data_src.get_metadata("kfold_n_splits"),
        )

    def load_sequences(
        self, required: Sequence[str] = (), optional: Sequence[str] = ()
    ) -> PackedSequences:
        """Load and pack split sequences using the chain-specific metadata.

        Column requirements follow :meth:`load_split_frame`. Call that method
        separately when the same rows are also needed for model statistics.
        """
        frame = self.load_split_frame(required=required, optional=optional)
        return self.pack_sequences(frame)

    @overload
    @staticmethod
    def disk_cache(
        cache_name: Callable[..., Any],
    ) -> Callable[..., Any]: ...  # bare @disk_cache form: returns wrapped method

    @overload
    @staticmethod
    def disk_cache(
        cache_name: Any = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...  # factory form

    @staticmethod
    def disk_cache(
        cache_name: Any = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator factory providing disk caching for instance methods.

        Usage::

            @BaseModelData.disk_cache()
            def prepare_data(self, args: Any) -> Any:
                ...

            @BaseModelData.disk_cache("my_data")
            def prepare_data(self, args: Any) -> Any:
                ...

        Args:
            cache_name: Cache filename prefix (optional). Defaults to the decorated function name.

        Notes:
            - Caching is only enabled when the instance's ``self._cache`` is True.
            - Cache keys include method arguments and processed-data checksums.
            - Cache directory: ``.cache/<ClassName>/``.
        """

        def _normalize_for_key(obj: Any) -> Any:
            """Convert an object to a stable, serialisable structure for cache keying."""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj

            if isinstance(obj, bytes):
                return ("__bytes__", len(obj), hashlib.md5(obj).hexdigest())

            if isinstance(obj, (list, tuple)):
                return [_normalize_for_key(x) for x in obj]

            if isinstance(obj, set):
                normalized = [_normalize_for_key(x) for x in obj]
                return sorted(normalized, key=lambda x: repr(x))

            if isinstance(obj, dict):
                items = []
                for k, v in obj.items():
                    items.append((str(k), _normalize_for_key(v)))
                items.sort(key=lambda kv: kv[0])
                return items

            # argparse.Namespace or any object with __dict__
            if hasattr(obj, "__dict__"):
                return _normalize_for_key(vars(obj))

            # numpy / torch large objects: keep type info and content hash
            module = getattr(type(obj), "__module__", "")
            name = getattr(type(obj), "__name__", type(obj).__name__)
            if hasattr(obj, "shape"):
                shape = tuple(obj.shape)
                return ("__object__", module, name, shape)
            return ("__object__", module, name)

        def _stable_key(obj: Any) -> bytes:
            """Encode a normalised object as a stable byte string."""
            normalized = _normalize_for_key(obj)
            return repr(normalized).encode("utf-8")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                if not getattr(self, "_cache", False):
                    return func(self, *args, **kwargs)

                # Cache directory relative to project root
                project_root = Path(__file__).resolve().parent.parent.parent
                cache_dir = project_root / ".cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                gitignore_path = cache_dir / ".gitignore"
                if not gitignore_path.exists():
                    gitignore_path.write_text("*\n", encoding="utf-8")

                # Get class name
                class_name = self.__class__.__name__
                target_dir = cache_dir / class_name
                target_dir.mkdir(parents=True, exist_ok=True)

                method_name = cache_name or func.__name__
                data_src = self.data_src
                metadata = data_src.get_metadata()
                data_checksums = tuple(
                    sorted(
                        (key, value)
                        for key, value in metadata.items()
                        if key.endswith("_md5")
                    )
                )
                key_payload = (
                    class_name,
                    func.__name__,
                    data_src.dataset,
                    data_checksums,
                    _normalize_for_key(args),
                    _normalize_for_key(kwargs),
                )
                key_hash = hashlib.md5(_stable_key(key_payload)).hexdigest()  # nosec B324

                # Build the full cache file path to avoid collisions
                file_path = target_dir / f"{method_name}_{key_hash}.pkl"

                # Check for existing cache
                if file_path.exists():
                    try:
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)
                        logger.info(f"Loaded cache from: {file_path}")
                        return data
                    except Exception as e:
                        logger.error(f"Fail to load cache, rebuilding data: {e}")

                # Cache miss or read failure: run original logic
                result = func(self, *args, **kwargs)

                # Write result to cache
                try:
                    with open(file_path, "wb") as f:
                        pickle.dump(result, f)
                    logger.info(f"Saved cache to: {file_path}")
                except Exception as e:
                    logger.error(f"Fail to save cache: {e}")

                return result

            return wrapper

        # Support @BaseModelData.disk_cache without parentheses
        if callable(cache_name):
            func = cache_name
            cache_name = None
            return decorator(func)

        return decorator

    @abstractmethod
    def prepare_data(self, args: Any) -> Any:
        """Prepare data required by the model.

        Args:
            args: Configuration arguments.
        """
        raise NotImplementedError("Subclasses should implement prepare_data method")

    def calculate_question_difficulty(
        self, exclude_fold: int | None = None
    ) -> dict[int, float]:
        """Calculate difficulty metrics for each question.

        Difficulty is computed from:
        1. Correct rate (correct_rate): correct answers / total answers
        2. Average response time (avg_time): if the dataset has a time field
        3. Hint rate (hint_rate): if the dataset has a hint field

        Args:
            exclude_fold: Fold index to exclude (used during cross-validation
                          to avoid data leakage from the validation fold).

        Returns:
            dict: Question ID to difficulty score mapping (0-1, higher = harder).

        Formula:
            difficulty = (1 - correct_rate) * 0.6 + normalized_time * 0.3 + hint_rate * 0.1
        """
        data = self.data_src.get_sequence_data()

        if exclude_fold is not None and "fold" in data.columns:
            data = data.filter(pl.col("fold") != exclude_fold)
            logger.info(f"Excluding fold {exclude_fold} from difficulty calculation.")

        stats = data.group_by("question").agg(
            pl.col("label").mean().alias("correct_rate"),
            pl.col("label").count().alias("count"),
        )
        confidence = (pl.col("count") / 10.0).clip(0.0, 1.0)
        error_rate = 1.0 - pl.col("correct_rate")
        stats = stats.with_columns(
            (error_rate * confidence + 0.5 * (1.0 - confidence)).alias("difficulty")
        )
        if isinstance(stats, pl.LazyFrame):
            stats = stats.collect()

        questions = stats["question"].to_list()
        difficulties = stats["difficulty"].to_list()
        return {int(q): float(d) for q, d in zip(questions, difficulties, strict=True)}

    def build_relationship_matrix(
        self, edge_type: tuple[str, str, str], value_type: str = "binary"
    ) -> np.ndarray:
        """Build a relationship matrix between entity types.

        Args:
            edge_type: Edge type triplet (source_node_type, relation_name, target_node_type).
                Node types correspond to column names in the data (e.g. 'user',
                'question', 'skill', 'template', 'assignment'). Examples:
                ('user', 'answers', 'question'), ('question', 'has', 'skill'),
                ('question', 'belongs_to', 'template'), ('skill', 'related_to', 'assignment').
            value_type: Matrix value type. 'binary' indicates relationship existence
                (default); 'count' indicates relationship frequency.

        Returns:
            data_matrix: numpy array of shape (num_src_nodes, num_dst_nodes).

        Examples:
            >>> # Build user-question binary matrix
            >>> matrix = model_data.build_data_matrix(('user', 'answers', 'question'))
            >>> # Build question-skill matrix
            >>> matrix = model_data.build_data_matrix(('question', 'has', 'skill'))
            >>> # Build question-template matrix
            >>> matrix = model_data.build_data_matrix(('question', 'belongs_to', 'template'))
            >>> # Build skill-assignment matrix
            >>> matrix = model_data.build_data_matrix(('skill', 'related_to', 'assignment'))
            >>> # Build user-question count matrix
            >>> matrix = model_data.build_data_matrix(('user', 'answers', 'question'), value_type='count')
        """
        src_type, _, dst_type = edge_type

        if src_type == "question":
            rel = self.data_src.get_relation(f"question_{dst_type}")
        elif dst_type == "question":
            rel = self.data_src.get_relation(f"{src_type}_question")
        else:
            qs = self.data_src.get_relation("question_skill")
            other = self.data_src.get_relation(f"question_{dst_type}")
            rel = (
                qs.join(other, on="question", how="inner")
                .select([pl.col(src_type), pl.col(dst_type)])
                .unique(subset=[src_type, dst_type])
            )

        if src_type not in rel.columns or dst_type not in rel.columns:
            raise ValueError(
                f"Required columns '{src_type}' or '{dst_type}' not found in data. "
                f"Available columns: {rel.columns}"
            )

        src_meta_key = f"num_{src_type}s"
        dst_meta_key = f"num_{dst_type}s"
        try:
            num_src = self.data_src.get_metadata(src_meta_key)
        except (KeyError, AttributeError):
            num_src = rel.select(pl.col(src_type).n_unique()).item()
            logger.warning(
                f"{src_meta_key} not found in metadata, calculated from data: {num_src}"
            )
        try:
            num_dst = self.data_src.get_metadata(dst_meta_key)
        except (KeyError, AttributeError):
            num_dst = rel.select(pl.col(dst_type).n_unique()).item()
            logger.warning(
                f"{dst_meta_key} not found in metadata, calculated from data: {num_dst}"
            )

        data_matrix = np.zeros((num_src, num_dst), dtype=int)

        pairs = rel.select(
            pl.col(src_type).cast(pl.Int64).alias("src"),
            pl.col(dst_type).cast(pl.Int64).alias("dst"),
        ).drop_nulls()
        src_idx = pairs["src"].to_numpy()
        dst_idx = pairs["dst"].to_numpy()
        valid = (
            (src_idx >= 0) & (src_idx < num_src) & (dst_idx >= 0) & (dst_idx < num_dst)
        )
        src_idx, dst_idx = src_idx[valid], dst_idx[valid]

        if value_type == "binary":
            data_matrix[src_idx, dst_idx] = 1
        elif value_type == "count":
            np.add.at(data_matrix, (src_idx, dst_idx), 1)
        else:
            raise ValueError(
                f"Unsupported value_type: {value_type}. Supported types: 'binary', 'count'"
            )

        return data_matrix
