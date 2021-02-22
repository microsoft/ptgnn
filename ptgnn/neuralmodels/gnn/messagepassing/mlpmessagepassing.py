import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer
from ptgnn.neuralmodels.mlp import MLP


class MlpMessagePassingLayer(AbstractMessagePassingLayer):
    def __init__(
        self,
        input_state_dimension: int,
        output_state_dimension: int,
        message_dimension: int,
        num_edge_types: int,
        message_aggregation_function: str,
        message_activation: Optional[nn.Module] = nn.GELU(),
        use_target_state_as_message_input: bool = True,
        mlp_hidden_layers: Union[List[int], int] = 0,
        use_layer_norm: bool = True,
        use_dense_layer: bool = True,
        dropout_rate: float = 0.0,
        dense_activation: Optional[nn.Module] = nn.Tanh(),
        features_dimension: int = 0,
    ):
        super().__init__()
        self.__input_state_dim = input_state_dimension
        self.__use_target_state_as_message_input = use_target_state_as_message_input
        self.__output_state_dim = output_state_dimension

        if use_target_state_as_message_input:
            message_input_size = 2 * input_state_dimension
        else:
            message_input_size = input_state_dimension
        self.__edge_message_transformation_layers = nn.ModuleList(
            [
                MLP(
                    input_dimension=message_input_size + features_dimension,
                    output_dimension=message_dimension,
                    hidden_layers=mlp_hidden_layers,
                )
                for _ in range(num_edge_types)
            ]
        )
        self.__aggregation_fn = message_aggregation_function
        self.__message_activation = message_activation

        state_update_layers: List[nn.Module] = []
        if use_layer_norm:
            state_update_layers.append(nn.LayerNorm(message_dimension))
        if use_dense_layer:
            state_update_layers.append(nn.Linear(message_dimension, output_state_dimension))
            nn.init.xavier_uniform_(state_update_layers[-1].weight)
            if dense_activation is not None:
                state_update_layers.append(dense_activation)
        state_update_layers.append(nn.Dropout(p=dropout_rate))

        self.__state_update = nn.Sequential(*state_update_layers)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        assert len(adjacency_lists) == len(self.__edge_message_transformation_layers)

        all_message_targets, all_messages = [], []
        for edge_type_idx, (adj_list, features, edge_transformation_layer) in enumerate(
            zip(adjacency_lists, edge_features, self.__edge_message_transformation_layers)
        ):
            edge_sources_idxs, edge_target_idxs = adj_list
            all_message_targets.append(edge_target_idxs)

            edge_source_states = nn.functional.embedding(edge_sources_idxs, node_states)

            if self.__use_target_state_as_message_input:
                edge_target_states = nn.functional.embedding(edge_target_idxs, node_states)
                message_input = torch.cat([edge_source_states, edge_target_states], dim=-1)
            else:
                message_input = edge_source_states

            all_messages.append(
                edge_transformation_layer(torch.cat([message_input, features], dim=-1))
            )

        aggregated_messages = self._aggregate_messages(
            messages=torch.cat(all_messages, dim=0),
            message_targets=torch.cat(all_message_targets, dim=0),
            num_nodes=node_states.shape[0],
            aggregation_fn=self.__aggregation_fn,
        )

        if self.__message_activation is not None:
            aggregated_messages = self.__message_activation(aggregated_messages)

        return self.__state_update(aggregated_messages)  # num_nodes x H

    @property
    def input_state_dimension(self) -> int:
        return self.__input_state_dim

    @property
    def output_state_dimension(self) -> int:
        return self.__output_state_dim
