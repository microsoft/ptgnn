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
        :param adjacency_lists: A list with as many elements as edge types. Each element in the list
            is a pair (tuple) of [num_edges_for_edge_type]-sized tensors indicating the left/right
            hand-side of the elements in the adjacency list.
        :param node_to_graph_idx: a vector of shape [num_nodes] indicating the graph that each node
            belongs to.
        :param reference_node_ids: a dictionary that maps each reference (key) to
            the indices of the reference nodes in `node_states`.
        :param reference_node_graph_idx: a dictionary that maps each reference (key) to
            the graph it belongs to. For each reference `ref_name`,
                len(reference_node_ids[ref_name])==len(reference_node_graph_idx[ref_name])
        :param edge_features: A list of [num_edges, H] with edge features.
            Has the same length as `adjacency_lists`, ie. len(adjacency_lists) == len(edge_features)
        :return: the output node states in a [num_nodes, D'] matrix.
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


class AbstractMessageAggregation(nn.Module):
    @abstractmethod
    def forward(self, messages: torch.Tensor, message_targets: torch.Tensor, num_nodes):
        pass

    @abstractmethod
    def output_state_size(self, message_input_size: int) -> int:
        pass
