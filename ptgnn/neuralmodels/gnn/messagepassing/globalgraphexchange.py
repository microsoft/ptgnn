import torch
from abc import abstractmethod
from torch import nn
from typing import Dict, List, Tuple

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer
from ptgnn.neuralmodels.reduceops.varsizedsummary import (
    AbstractVarSizedElementReduce,
    ElementsToSummaryRepresentationInput,
)


class AbstractGlobalGraphExchange(AbstractMessagePassingLayer):
    def __init__(
        self,
        global_graph_representation_module: AbstractVarSizedElementReduce,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.__global_graph_representation_module = global_graph_representation_module
        self.__dropout = nn.Dropout(p=dropout_rate)

    @abstractmethod
    def _update_node_states(
        self, node_states: torch.Tensor, global_info_per_node: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        e = ElementsToSummaryRepresentationInput(
            element_embeddings=node_states,
            element_to_sample_map=node_to_graph_idx,
            num_samples=node_to_graph_idx.max() + 1,
        )
        graph_representations = self.__global_graph_representation_module(e)
        graph_representations = self.__dropout(graph_representations)
        return self._update_node_states(node_states, graph_representations[node_to_graph_idx])


class GruGlobalStateUpdate(AbstractGlobalGraphExchange):
    def __init__(
        self,
        global_graph_representation_module: AbstractVarSizedElementReduce,
        input_state_size: int,
        summarized_state_size: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__(global_graph_representation_module, dropout_rate)
        self.__input_dim = input_state_size
        self.__summarized_state_size = summarized_state_size
        self.__gru_cell = nn.GRUCell(input_size=summarized_state_size, hidden_size=input_state_size)

    def _update_node_states(
        self, node_states: torch.Tensor, global_info_per_node: torch.Tensor
    ) -> torch.Tensor:
        return self.__gru_cell(global_info_per_node, node_states)

    @property
    def input_state_dimension(self) -> int:
        return self.__input_dim

    @property
    def output_state_dimension(self) -> int:
        return self.__input_dim
