import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, NamedTuple, Optional, Tuple, TypeVar

TNodeData = TypeVar("TNodeData")
TEdgeData = TypeVar("TEdgeData")
TTensorizedNodeData = TypeVar("TTensorizedNodeData")
TTensorizedEdgeData = TypeVar("TTensorizedEdgeData")


class GraphData(Generic[TNodeData, TEdgeData]):
    __slots__ = ("node_information", "edges", "edge_features", "reference_nodes")

    def __init__(
        self,
        node_information: List[TNodeData],
        edges: Dict[str, List[Tuple[int, int]]],
        reference_nodes: Dict[str, List[int]],
        edge_features: Optional[Dict[str, List[TEdgeData]]] = None,
    ):
        self.node_information = node_information
        self.edges = edges
        self.edge_features = edge_features
        self.reference_nodes = reference_nodes


class TensorizedGraphData(Generic[TTensorizedNodeData, TTensorizedEdgeData]):
    __slots__ = (
        "num_nodes",
        "node_tensorized_data",
        "adjacency_lists",
        "edge_features",
        "reference_nodes",
    )

    def __init__(
        self,
        num_nodes: int,
        node_tensorized_data: List[TTensorizedNodeData],
        adjacency_lists: List[Tuple[np.ndarray, np.ndarray]],
        edge_features: Optional[List[TTensorizedEdgeData]],
        reference_nodes: Dict[str, np.ndarray],
    ):
        self.num_nodes = num_nodes
        self.node_tensorized_data = node_tensorized_data
        self.adjacency_lists = adjacency_lists
        self.edge_features = edge_features
        self.reference_nodes = reference_nodes


class GnnOutput(NamedTuple):
    input_node_representations: torch.Tensor
    output_node_representations: torch.Tensor
    node_to_graph_idx: torch.Tensor
    # Which are the idxs of the referenced nodes (in the minibatch)?
    node_idx_references: Dict[str, torch.Tensor]
    # Which graph do the referenced nodes belong to?
    node_graph_idx_reference: Dict[str, torch.Tensor]
    num_graphs: int

    @property
    def reference_nodes_idx(self) -> Dict[str, torch.Tensor]:
        """
        Which are the idxs of the referenced nodes (within the minibatch).
        Better alias for `node_idx_references`.
        """
        return self.node_idx_references

    @property
    def reference_nodes_graph_idx(self) -> Dict[str, torch.Tensor]:
        """
        Which graph do the referenced nodes belong to?
        Better alias for `node_graph_idx_reference`.
        """
        return self.node_graph_idx_reference


class AbstractNodeEmbedder(ABC):
    """Abstract node embedder."""

    @abstractmethod
    def representation_size(self) -> int:
        pass
