import torch
from torch import nn
from torch_scatter import scatter_sum
from typing import Dict, List, Tuple

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer


class MultiHeadSelfAttentionMessagePassing(AbstractMessagePassingLayer):
    """
    A transformer layer among all nodes in a graph.
    """

    def __init__(
        self,
        input_state_dimension: int,
        key_query_dimension: int,
        value_dimension: int,
        output_dimension: int,
        intermediate_dimension: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        target_reference: str = "all",
        max_num_nodes: int = 250,
    ):
        super().__init__()

        self.__num_heads = num_heads
        self.__key_query_dim = key_query_dimension
        self.__value_dim = value_dimension

        self.__selfatt_head_transforms = nn.Linear(
            in_features=input_state_dimension,
            out_features=num_heads * (2 * key_query_dimension + value_dimension),
            bias=False,
        )

        self.__summarization_layer = nn.Linear(
            in_features=num_heads * value_dimension,
            out_features=output_dimension,
            bias=False,
        )

        self.__intermediate_layer = nn.Linear(
            in_features=output_dimension, out_features=intermediate_dimension
        )

        self.__output_layer = nn.Linear(
            in_features=intermediate_dimension, out_features=output_dimension
        )

        self.__layer_norm1 = nn.LayerNorm(output_dimension)
        self.__layer_norm2 = nn.LayerNorm(output_dimension)
        self.__dropout_layer = nn.Dropout(p=dropout_rate)

        self.__target_reference = target_reference
        self.__max_num_nodes = max_num_nodes

    def __iter_idxs_per_graph(self, node_to_graph_idx):
        with torch.no_grad():
            num_nodes_per_graph = scatter_sum(
                src=torch.ones_like(node_to_graph_idx, dtype=torch.int64), index=node_to_graph_idx
            )  # [num_nodes]

            node_offset = 0
            for num_nodes in num_nodes_per_graph:
                for start_idx in range(0, num_nodes, self.__max_num_nodes):
                    end_idx = min(start_idx + self.__max_num_nodes, num_nodes)
                    yield torch.arange(
                        start=start_idx,
                        end=end_idx,
                        dtype=torch.int64,
                        device=num_nodes_per_graph.device,
                    ) + node_offset
                node_offset += num_nodes

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],  # Not used
    ) -> torch.Tensor:
        if self.__target_reference == "all":
            relevant_node_states = node_states
        else:
            relevant_node_states = node_states[reference_node_ids[self.__target_reference]]
            node_to_graph_idx = reference_node_graph_idx[self.__target_reference]

        keys_queries_values = self.__selfatt_head_transforms(relevant_node_states).reshape(
            relevant_node_states.shape[0], self.__num_heads, -1
        )

        keys = keys_queries_values[:, :, : self.__key_query_dim]  # [num_nodes, num_heads, key_dim]
        queries = keys_queries_values[
            :, :, self.__key_query_dim : 2 * self.__key_query_dim
        ]  # [num_nodes, num_heads, key_dim]
        values = keys_queries_values[
            :, :, 2 * self.__key_query_dim :
        ]  # [num_nodes, num_heads, value_dim]

        all_out_values = []
        for graph_nodes in self.__iter_idxs_per_graph(node_to_graph_idx):
            # Loop is necessary due to limited memory and ease of coding.
            graph_keys = keys[graph_nodes]
            graph_queries = queries[graph_nodes]
            scores = torch.einsum("khd,vhd->khv", graph_keys, graph_queries) / (
                self.__key_query_dim ** 0.5
            )
            attention_probs = nn.functional.softmax(scores, dim=-1)
            attention_probs = self.__dropout_layer(attention_probs)
            out_values = torch.einsum("khv,vhd->khd", attention_probs, values[graph_nodes])
            all_out_values.append(out_values)

        values = torch.cat(all_out_values, dim=0)
        output = self.__summarization_layer(values.reshape(values.shape[0], -1))
        attention_output = self.__layer_norm1(self.__dropout_layer(output) + node_states)

        intermediate = nn.functional.relu(self.__intermediate_layer(attention_output))
        output = self.__dropout_layer(self.__output_layer(intermediate))
        output_node_states = self.__layer_norm2(output + attention_output)
        if self.__target_reference == "all":
            return output_node_states
        else:
            node_states[reference_node_ids[self.__target_reference]] = output_node_states
            return node_states

    @property
    def input_state_dimension(self) -> int:
        return self.__selfatt_head_transforms.in_features

    @property
    def output_state_dimension(self) -> int:
        return self.__output_layer.out_features
