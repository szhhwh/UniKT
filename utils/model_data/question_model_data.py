"""Question-level model data module.

Provides the QuestionModelData class for preparing question-level knowledge
tracing model data, including sequence building and heterogeneous graph
construction.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import BaseModelData

if TYPE_CHECKING:
    import numpy as np
    import polars as pl
    from torch_geometric.data import HeteroData

logger = get_logger(__name__)


class QuestionModelData(BaseModelData):
    """Base class for question-level model data preparation."""

    def __init__(self, data_src: DataSource, cache: bool = False):
        """Initialise the question-level model data object.

        Args:
            data_src: Data source object.
            cache: Whether to enable disk caching.
        """
        super().__init__(data_src, cache=cache)

    def _get_kfold_data(self) -> pl.DataFrame:
        """Override: retrieve K-fold labels from question sequence data."""
        return self.data_src.get_split_question_sequence_data()

    @abstractmethod
    def prepare_data(self, args: Any) -> Any:
        """Prepare data required by question-level models.

        Args:
            args: Configuration arguments.
        """
        raise NotImplementedError("Subclasses should implement prepare_data method")

    def load_sequence_data(
        self,
    ) -> tuple[np.ndarray, ...]:
        """Load user response sequences.

        Loads split question sequence data from disk and builds sequence arrays.

        Returns:
            tuple: (user_sequence, user_response, user_mask, user_id_sequence)
                   as numpy arrays of shape (num_users, max_seq_len).
        """
        import numpy as np

        logger.info("Building response sequences from split data...")

        # Load split sequence data
        data = self.data_src.get_split_question_sequence_data().to_pandas()
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_users = data["sequence_id"].nunique()

        # Build sequence arrays
        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)

        user_indices = data["sequence_id"].values
        seq_positions = data["seq_pos"].values

        user_sequence[user_indices, seq_positions] = data["question"].values
        user_id_sequence[user_indices, seq_positions] = data["user"].values
        user_response[user_indices, seq_positions] = data["label"].values
        user_mask[user_indices, seq_positions] = 1

        return user_sequence, user_response, user_mask, user_id_sequence

    def build_hetero_graph(
        self,
        edge_types: list[tuple[str, str, str]],
        edge_attrs: dict[tuple[str, str, str], list[str]] | None = None,
        directed: bool = False,
        node_features: dict[str, Any] | None = None,
    ) -> HeteroData:
        """Build a heterogeneous graph with flexible node and edge type configuration.

        Args:
            edge_types: List of edge type triplets (source_type, relation, target_type).
                        Example: [('user', 'answers', 'question'), ('question', 'has', 'skill')].
            edge_attrs: Edge attribute dictionary. Keys are edge type triplets,
                        values are lists of attribute column names.
                        Example: {('user', 'answers', 'question'): ['label', 'timestamp']}.
                        Defaults to None (no edge attributes).
            directed: Whether to build a directed graph. Defaults to False (undirected).
            node_features: Node feature dictionary. Keys are node types, values
                           are feature tensors or None.
                           Defaults to using node IDs as features.

        Returns:
            HeteroData: PyTorch Geometric heterogeneous graph object.

        Examples:
            >>> # Build question-skill undirected graph
            >>> graph = model_data.build_hetero_graph(
            ...     edge_types=[('question', 'has', 'skill')],
            ...     directed=False
            ... )
            >>> # Build combined student-question and question-skill graph
            >>> graph = model_data.build_hetero_graph(
            ...     edge_types=[('user', 'answers', 'question'), ('question', 'has', 'skill')],
            ...     directed=False
            ... )
            >>> # Build graph with edge attributes
            >>> graph = model_data.build_hetero_graph(
            ...     edge_types=[('user', 'answers', 'question')],
            ...     edge_attrs={('user', 'answers', 'question'): ['label', 'timestamp']},
            ...     directed=True
            ... )
        """
        import numpy as np
        import torch
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import ToUndirected
        from tqdm import tqdm

        if edge_attrs is None:
            edge_attrs = {}

        graph = HeteroData()

        # Collect all needed node types
        node_types = set()
        for src_type, _, dst_type in edge_types:
            node_types.add(src_type)
            node_types.add(dst_type)

        # Get node counts for each type
        node_counts = {}
        for node_type in node_types:
            meta_key = f"num_{node_type}s"
            try:
                node_counts[node_type] = self.data_src.get_metadata(meta_key)
            except (KeyError, AttributeError):
                # Look up in relation tables
                found = False
                for rel_df in self.data_src.relation_data.values():
                    if node_type in rel_df.columns:
                        node_counts[node_type] = rel_df[node_type].n_unique()
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"Cannot determine node count for type '{node_type}'"
                    )

        # Set node counts and features
        for node_type in node_types:
            graph[node_type].num_nodes = node_counts[node_type]

            # Set node features
            if node_features and node_type in node_features:
                graph[node_type].x = node_features[node_type]
            else:
                # Default to using node IDs as features
                graph[node_type].x = (
                    torch.arange(node_counts[node_type]).view(-1, 1).float()
                )

        # Build edges for each edge type
        for edge_type in edge_types:
            src_type, relation, dst_type = edge_type

            # Build relationship matrix
            logger.info(
                f"Building relationship matrix for {src_type}-{relation}-{dst_type}"
            )
            rel_matrix = self.build_relationship_matrix(edge_type, value_type="binary")

            # Extract edge indices from the matrix
            src_indices, dst_indices = np.nonzero(rel_matrix)

            if len(src_indices) == 0:
                logger.warning(f"No edges found for {edge_type}")
                continue

            # Convert to PyTorch tensors
            edge_index = torch.tensor(
                np.vstack([src_indices, dst_indices]), dtype=torch.long
            ).contiguous()

            # Add edge index to graph
            graph[src_type, relation, dst_type].edge_index = edge_index

            # Handle edge attributes
            attr_cols = edge_attrs.get(edge_type, [])
            if attr_cols:
                # Extract edge attributes from raw data
                data = self.data_src.get_sequence_data().to_pandas()
                src_col = src_type
                dst_col = dst_type

                # Check column existence
                if src_col not in data.columns or dst_col not in data.columns:
                    logger.warning(
                        f"Columns {src_col} or {dst_col} not found. Skipping edge attributes."
                    )
                    continue

                # Build edge-to-attribute mapping
                edge_attr_dict = {}
                cols_to_select = [src_col, dst_col, *attr_cols]

                for row in tqdm(
                    data[cols_to_select].itertuples(index=False),
                    total=len(data),
                    desc=f"Extracting edge attributes for {src_type}-{relation}-{dst_type}",
                ):
                    src_id = int(getattr(row, src_col))
                    dst_id = int(getattr(row, dst_col))
                    edge_key = (src_id, dst_id)

                    # Update edge attributes (last occurrence wins)
                    edge_attr_dict[edge_key] = {
                        attr: getattr(row, attr) for attr in attr_cols
                    }

                # Extract attribute values in edge_index order
                for attr in attr_cols:
                    attr_values = []
                    for i in range(edge_index.shape[1]):
                        src_id = int(edge_index[0, i].item())
                        dst_id = int(edge_index[1, i].item())
                        edge_key = (src_id, dst_id)

                        if edge_key in edge_attr_dict:
                            attr_values.append(edge_attr_dict[edge_key][attr])
                        else:
                            attr_values.append(0.0)

                    attr_tensor = torch.tensor(attr_values, dtype=torch.float32)
                    setattr(
                        graph[src_type, relation, dst_type],
                        f"edge_attr_{attr}",
                        attr_tensor,
                    )

        # Apply ToUndirected if needed
        if not directed:
            graph = ToUndirected()(graph)

        return graph
