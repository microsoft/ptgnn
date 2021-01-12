import torch
from torch import nn
from torch_scatter import scatter
from typing import Dict, List, Optional, Tuple, Union

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer
from ptgnn.neuralmodels.mlp import MLP


class PnaMessagePassingLayer(AbstractMessagePassingLayer):
    """
    Principal Neighbourhood Aggregation for Graph Nets

    https://arxiv.org/abs/2004.05718
    """

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
        delta: float = 1,
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
                    input_dimension=message_input_size,
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
            state_update_layers.append(nn.LayerNorm(message_dimension * 5 * 3))
        if use_dense_layer:
            state_update_layers.append(nn.Linear(message_dimension * 5 * 3, output_state_dimension))
            nn.init.xavier_uniform_(state_update_layers[-1].weight)
            if dense_activation is not None:
                state_update_layers.append(dense_activation)
        state_update_layers.append(nn.Dropout(p=dropout_rate))

        self.__state_update = nn.Sequential(*state_update_layers)
        self._delta = delta  # See Eq 5 of paper

    def __pna_aggregation_and_scaling(
        self, messages: torch.Tensor, message_targets: torch.Tensor, num_nodes
    ):
        degree = scatter(
            torch.ones_like(message_targets),
            index=message_targets,
            dim_size=num_nodes,
            reduce="sum",
        )

        sum_agg = scatter(messages, index=message_targets, dim=0, dim_size=num_nodes, reduce="sum")
        mean_agg = sum_agg / (degree.unsqueeze(-1) + 1e-5)
        max_agg = scatter(messages, index=message_targets, dim=0, dim_size=num_nodes, reduce="max")
        min_agg = scatter(messages, index=message_targets, dim=0, dim_size=num_nodes, reduce="min")

        std_components = torch.relu(messages.pow(2) - mean_agg[message_targets].pow(2)) + 1e-10
        std = torch.sqrt(
            scatter(std_components, index=message_targets, dim=0, dim_size=num_nodes, reduce="sum")
        )

        all_aggregations = torch.cat([sum_agg, mean_agg, max_agg, min_agg, std], dim=-1)

        scaler_p1 = torch.log(degree.float() + 1).unsqueeze(-1) / self._delta
        scaler_m1 = 1 / (scaler_p1 + 1e-3)

        return torch.cat(
            [all_aggregations, all_aggregations * scaler_p1, all_aggregations * scaler_m1], dim=-1
        )

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        assert len(adjacency_lists) == len(self.__edge_message_transformation_layers)

        all_message_targets, all_messages = [], []
        for edge_type_idx, (adj_list, edge_transformation_layer) in enumerate(
            zip(adjacency_lists, self.__edge_message_transformation_layers)
        ):
            edge_sources_idxs, edge_target_idxs = adj_list
            all_message_targets.append(edge_target_idxs)

            edge_source_states = nn.functional.embedding(edge_sources_idxs, node_states)

            if self.__use_target_state_as_message_input:
                edge_target_states = nn.functional.embedding(edge_target_idxs, node_states)
                message_input = torch.cat([edge_source_states, edge_target_states], dim=-1)
            else:
                message_input = edge_source_states

            all_messages.append(edge_transformation_layer(message_input))

        all_messages = torch.cat(all_messages, dim=0)
        if self.__message_activation is not None:
            all_messages = self.__message_activation(all_messages)

        aggregated_messages = self.__pna_aggregation_and_scaling(
            messages=all_messages,
            message_targets=torch.cat(all_message_targets, dim=0),
            num_nodes=node_states.shape[0],
        )

        return self.__state_update(aggregated_messages)  # num_nodes x H

    @property
    def input_state_dimension(self) -> int:
        return self.__input_state_dim

    @property
    def output_state_dimension(self) -> int:
        return self.__output_state_dim
