"""Base model data class for knowledge tracing datasets.

Provides the abstract foundation for preparing model-ready data, including
K-fold splitting, difficulty calculation, and relationship matrix building.
"""

import functools
import hashlib
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

import numpy as np
import polars as pl

from utils.core import get_logger
from utils.data_process import DataSource

logger = get_logger(__name__)


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
            - Cache key is built from the class name, function name, and arguments.
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
                dataset_tag = getattr(getattr(self, "data_src", None), "dataset", None)
                key_payload = (
                    class_name,
                    func.__name__,
                    dataset_tag,
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

    def _get_kfold_data(self) -> pl.DataFrame:
        """Retrieve the K-fold label data source for this model data.

        Subclasses should override this method to return the actual sequence
        data they use (e.g. split_skill_sequence or split_question_sequence).

        Returns:
            polars.LazyFrame: Data containing user and fold columns.
        """
        return self.data_src.get_sequence_data()

    def _build_user_folds(self, num_users: int) -> np.ndarray:
        """Build a user-to-fold mapping array from K-fold data.

        Args:
            num_users: Number of users.

        Returns:
            np.ndarray: user_folds[user_idx] = fold_label.

        Raises:
            ValueError: If the fold column is missing, the fold data is empty,
                or user counts mismatch.
        """
        data = self._get_kfold_data()

        if "fold" not in data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call data_src.add_kfold_labels() first."
            )

        id_col = "sequence_id" if "sequence_id" in data.columns else "user"

        inconsistent = (
            data.group_by(id_col)
            .agg(pl.col("fold").n_unique().alias("fold_nunique"))
            .filter(pl.col("fold_nunique") > 1)
        )
        if inconsistent.height > 0:
            raise ValueError("Found users with inconsistent fold labels")

        user_fold = data.select([id_col, "fold"]).unique(subset=[id_col], keep="first")

        if user_fold.height != num_users:
            raise ValueError(
                f"User count mismatch: fold data has {user_fold.height} users, "
                f"but model data has {num_users} users. "
                f"Ensure K-fold labels are added to the correct data source."
            )

        user_idx = user_fold[id_col].to_numpy()
        fold_label = user_fold["fold"].to_numpy()

        if user_idx.size == 0:
            raise ValueError(
                "Fold data contains no users; add K-fold labels to a non-empty "
                "data source before splitting."
            )

        if user_idx.min() < 0 or user_idx.max() >= num_users:
            raise ValueError(
                f"User index out of range: min={int(user_idx.min())}, max={int(user_idx.max())}, "
                f"num_users={num_users}"
            )

        user_folds = np.full(num_users, -1, dtype=np.int32)
        user_folds[user_idx] = fold_label
        return user_folds

    @abstractmethod
    def prepare_data(self, args: Any) -> Any:
        """Prepare data required by the model.

        Args:
            args: Configuration arguments.
        """
        raise NotImplementedError("Subclasses should implement prepare_data method")

    def split_kfold_data(
        self, *arrays: Any, fold_idx: int
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        """Split data into train, validation, and test sets by K-fold index.

        Args:
            *arrays: Any number of arrays or tensors with users as the first dimension.
            fold_idx: Current fold index (keyword-only, required).

        Returns:
            train_data: Tuple of train slices with the same structure as input.
            val_data: Tuple of validation slices.
            test_data: Tuple of test slices.

        Notes:
            - Validation set uses the specified fold (fold_idx); test set uses
              fold == -1; training set uses remaining folds.
            - Requires user-to-row-index mapping in the data source.
        """
        import numpy as np

        # Validate inputs
        if len(arrays) == 0:
            raise ValueError(
                "split_kfold_data requires at least one input array/tensor"
            )

        # Get valid user indices
        num_users = arrays[0].shape[0]
        # Validate consistent user count across all inputs
        for i, arr in enumerate(arrays):
            if arr.shape[0] != num_users:
                raise ValueError(
                    f"Input array {i} shape is {arr.shape}, but expected shape is (num_users, *)"
                )

        # Build user fold information mapping
        user_folds = self._build_user_folds(num_users)

        # Split users by fold label:
        # Validation: fold == fold_idx
        # Test: fold == -1
        # Training: fold != fold_idx and fold != -1
        train_user_indices = np.where((user_folds != fold_idx) & (user_folds != -1))[0]
        val_user_indices = np.where(user_folds == fold_idx)[0]
        test_user_indices = np.where(user_folds == -1)[0]

        train_idx_list = train_user_indices[train_user_indices < num_users].tolist()
        val_idx_list = val_user_indices[val_user_indices < num_users].tolist()
        test_idx_list = test_user_indices[test_user_indices < num_users].tolist()

        train_slices = []
        val_slices = []
        test_slices = []
        for arr in arrays:
            # Detect torch.Tensor
            is_torch_tensor = False
            try:
                import torch

                is_torch_tensor = hasattr(arr, "dim") and hasattr(arr, "index_select")
            except Exception:
                is_torch_tensor = False

            if is_torch_tensor:
                import torch

                train_idx = torch.tensor(
                    train_idx_list, dtype=torch.long, device=arr.device
                )
                val_idx = torch.tensor(
                    val_idx_list, dtype=torch.long, device=arr.device
                )
                test_idx = torch.tensor(
                    test_idx_list, dtype=torch.long, device=arr.device
                )
                train_slices.append(arr.index_select(0, train_idx))
                val_slices.append(arr.index_select(0, val_idx))
                test_slices.append(arr.index_select(0, test_idx))
            else:
                train_slices.append(arr[train_idx_list])
                val_slices.append(arr[val_idx_list])
                test_slices.append(arr[test_idx_list])

        return tuple(train_slices), tuple(val_slices), tuple(test_slices)

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
