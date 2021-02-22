import torch
from torch import nn
from typing import Dict, List, Tuple

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer


class _ResidualOriginLayer(AbstractMessagePassingLayer):
    def __init__(self, input_dim: int, target_layer):
        super().__init__()
        self.__target_layer = target_layer
        self.__input_dim = input_dim

    @property
    def input_state_dimension(self) -> int:
        return self.__input_dim

    @property
    def output_state_dimension(self) -> int:
        return self.__input_dim

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        self.__target_layer._original_input = node_states
        return node_states


class MeanResidualLayer(AbstractMessagePassingLayer):
    def __init__(self, input_dim: int):
        super().__init__()
        self._original_input = None
        self.__input_dim = input_dim

    def pass_through_dummy_layer(self) -> _ResidualOriginLayer:
        return _ResidualOriginLayer(self.__input_dim, target_layer=self)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        assert self._original_input is not None, "Initial Pass Through Layer was not used."
        out = torch.stack((self._original_input, node_states), dim=-1).mean(dim=-1)
        self._original_input = None  # Reset
        return out

    @property
    def input_state_dimension(self) -> int:
        return self.__input_dim

    @property
    def output_state_dimension(self) -> int:
        return self.__input_dim


class ConcatResidualLayer(AbstractMessagePassingLayer):
    def __init__(self, input_dim: int):
        super().__init__()
        self._original_input = None
        self.__input_dim = input_dim

    def pass_through_dummy_layer(self) -> _ResidualOriginLayer:
        return _ResidualOriginLayer(self.__input_dim, target_layer=self)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        assert self._original_input is not None, "Initial Pass Through Layer was not used."
        out = torch.cat((self._original_input, node_states), dim=-1)
        self._original_input = None  # Reset
        return out

    @property
    def input_state_dimension(self) -> int:
        return self.__input_dim

    @property
    def output_state_dimension(self) -> int:
        return 2 * self.__input_dim


class LinearResidualLayer(AbstractMessagePassingLayer):
    def __init__(
        self,
        state_dimension1: int,
        state_dimension2: int,
        target_state_size: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.__input_dim1 = state_dimension1
        self.__input_dim2 = state_dimension2

        self.__linear_combination = nn.Linear(
            in_features=state_dimension1 + state_dimension2,
            out_features=target_state_size,
            bias=False,
        )
        self._original_input = None
        self.__dropout = nn.Dropout(p=dropout_rate)

    def pass_through_dummy_layer(self) -> _ResidualOriginLayer:
        return _ResidualOriginLayer(self.__input_dim1, target_layer=self)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        assert self._original_input is not None, "Initial Pass Through Layer was not used."
        out = self.__linear_combination(torch.cat((self._original_input, node_states), axis=-1))
        self._original_input = None  # Reset
        return self.__dropout(out)

    @property
    def input_state_dimension(self) -> int:
        return self.__input_dim2

    @property
    def output_state_dimension(self) -> int:
        return self.__linear_combination.out_features
