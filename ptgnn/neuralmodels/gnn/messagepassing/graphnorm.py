import torch
from torch import nn
from torch_scatter import scatter_mean
from typing import Dict, List, Tuple

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer


class GraphNorm(AbstractMessagePassingLayer):
    """
    A GraphNorm layer implemented as in

    GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training (arXiv:2009.03294v1)
      Tianle Cai, Shengjie Luo, Keyulu Xu, Di He, Tie-yan Liu, Liwei Wang

    """

    def __init__(self, input_state_dimension: int, eps: float = 1e-10):
        super().__init__()
        self.__input_state_dim = input_state_dimension
        self.__eps = eps

        self.gamma = nn.Parameter(torch.ones(1, input_state_dimension))
        self.alpha = nn.Parameter(torch.ones(1, input_state_dimension))
        self.bias = nn.Parameter(torch.zeros(1, input_state_dimension))

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        per_graph_mean = scatter_mean(
            node_states, index=node_to_graph_idx, dim=0
        )  # [num_graphs, D]
        shifted = node_states - self.alpha * per_graph_mean[node_to_graph_idx]  # [num_nodes, D]
        sigma_2 = (
            scatter_mean(torch.pow(shifted, 2), index=node_to_graph_idx, dim=0) + self.__eps
        )  # [num_graphs, D]

        return (
            self.gamma * shifted / torch.sqrt(sigma_2[node_to_graph_idx]) + self.bias
        )  # [num_nodes, D]

    @property
    def input_state_dimension(self) -> int:
        return self.__input_state_dim

    @property
    def output_state_dimension(self) -> int:
        return self.__input_state_dim
