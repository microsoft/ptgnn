import torch
from torch import nn
from typing import Dict, List, Tuple

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer


class GatedMessagePassingLayer(AbstractMessagePassingLayer):
    def __init__(
        self,
        state_dimension: int,
        message_dimension: int,
        num_edge_types: int,
        message_aggregation_function: str,
        dropout_rate: float = 0.0,
        edge_feature_dimension: int = 0,
    ):
        super().__init__()

        self.__edge_message_transformation_layers = nn.ModuleList(
            [
                nn.Linear(state_dimension + edge_feature_dimension, message_dimension, bias=False)
                for _ in range(num_edge_types)
            ]
        )
        for msg_layer in self.__edge_message_transformation_layers:
            nn.init.xavier_normal_(msg_layer.weight, gain=(1 / num_edge_types) ** 0.5)
        self.__state_update = nn.GRUCell(input_size=message_dimension, hidden_size=state_dimension)
        nn.init.orthogonal_(self.__state_update.weight_hh)
        nn.init.xavier_uniform_(self.__state_update.weight_ih)
        nn.init.normal_(self.__state_update.bias_hh, std=1e-5)
        nn.init.normal_(self.__state_update.bias_ih, std=1e-5)
        self.__state_dimension = state_dimension
        self.__aggregation_fn = message_aggregation_function
        self.__dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        message_targets = torch.cat([adj_list[1] for adj_list in adjacency_lists])  # num_messages
        assert len(adjacency_lists) == len(self.__edge_message_transformation_layers)

        all_messages = []
        for edge_type_idx, (adj_list, features, edge_transformation_layer) in enumerate(
            zip(adjacency_lists, edge_features, self.__edge_message_transformation_layers)
        ):
            edge_sources_idxs = adj_list[0]
            edge_source_states = nn.functional.embedding(
                edge_sources_idxs, node_states
            )  # [num_edges_of_type_edge_type_idx, H]
            all_messages.append(
                edge_transformation_layer(
                    self.__dropout(torch.cat([edge_source_states, features], -1))
                )
            )  # [num_edges_of_type_edge_type_idx, D]

        aggregated_messages = self._aggregate_messages(
            messages=torch.cat(all_messages, dim=0),
            message_targets=message_targets,
            num_nodes=node_states.shape[0],
            aggregation_fn=self.__aggregation_fn,
        )  # [num_nodes, D]
        return self.__state_update(aggregated_messages, node_states)  # [num_nodes, H]

    @property
    def input_state_dimension(self) -> int:
        return self.__state_dimension

    @property
    def output_state_dimension(self) -> int:
        return self.__state_dimension
