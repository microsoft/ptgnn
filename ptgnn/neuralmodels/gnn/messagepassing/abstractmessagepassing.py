import torch
from abc import abstractmethod
from torch import nn
from torch_scatter import scatter
from typing import Dict, List, Tuple


class AbstractMessagePassingLayer(nn.Module):
    """Interface for message passing layer to be used in graph neural networks with multiple edge types."""

    @abstractmethod
    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        :param node_states: A [num_nodes, D] matrix containing the states of all nodes.
        :param adjacency_lists: A list of [num_edges, 2] adjacency lists, one per edge type.
        :param node_to_graph_idx:
        :param reference_node_ids:
        :param reference_node_graph_idx:
        :param edge_features: A list of [num_edges, H] with edge features.
            Has the size of `adjacency_lists`.
        :return: the next node states in a [num_nodes, D'] matrix.
        """

    def _aggregate_messages(
        self, messages: torch.Tensor, message_targets: torch.Tensor, num_nodes, aggregation_fn: str
    ):
        """Utility function to be used by concrete implementors."""
        # Support AMP
        msg_dtype = messages.dtype
        return scatter(
            messages.to(torch.float32),
            index=message_targets,
            dim=0,
            dim_size=num_nodes,
            reduce=aggregation_fn,
        ).to(msg_dtype)

    @property
    @abstractmethod
    def input_state_dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def output_state_dimension(self) -> int:
        pass
