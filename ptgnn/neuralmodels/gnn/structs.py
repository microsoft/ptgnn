from abc import ABC, abstractmethod
from typing import Dict, Generic, List, NamedTuple, Tuple, TypeVar

import numpy as np
import torch

TNodeData = TypeVar("TNodeData")
TTensorizedNodeData = TypeVar("TTensorizedNodeData")


class GraphData(Generic[TNodeData]):
    __slots__ = ("node_information", "edges", "reference_nodes")

    def __init__(
        self,
        node_information: List[TNodeData],
        edges: Dict[str, List[Tuple[int, int]]],
        reference_nodes: Dict[str, List[int]],
    ):
        self.node_information = node_information
        self.edges = edges
        self.reference_nodes = reference_nodes


class TensorizedGraphData(Generic[TTensorizedNodeData]):
    __slots__ = ("num_nodes", "node_tensorized_data", "adjacency_lists", "reference_nodes")

    def __init__(
        self,
        num_nodes: int,
        node_tensorized_data: List[TTensorizedNodeData],
        adjacency_lists: List[Tuple[np.ndarray, np.ndarray]],
        reference_nodes: Dict[str, np.ndarray],
    ):
        self.num_nodes = num_nodes
        self.node_tensorized_data = node_tensorized_data
        self.adjacency_lists = adjacency_lists
        self.reference_nodes = reference_nodes


class GnnOutput(NamedTuple):
    input_node_representations: torch.Tensor
    output_node_representations: torch.Tensor
    node_to_graph_idx: torch.Tensor
    # Which are the idxs of the referenced nodes (in the minibatch)?
    node_idx_references: Dict[str, torch.Tensor]
    # Which graph do the referenced nodes belong in?
    node_graph_idx_reference: Dict[str, torch.Tensor]
    num_graphs: int


class AbstractNodeEmbedder(ABC):
    """Abstract node embedder."""

    @abstractmethod
    def representation_size(self) -> int:
        pass
