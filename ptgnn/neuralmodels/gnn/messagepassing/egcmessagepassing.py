import torch
from torch import nn
from typing import Dict, List, Tuple

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer


class EGCMessagePassingLayer(AbstractMessagePassingLayer):
    """
    An implementation of the Efficient Graph Convolution (EGC-S) layer of

    @misc{tailor2021adaptive,
      title={Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions},
      author={Shyam A. Tailor and Felix L. Opolka and Pietro Liò and Nicholas D. Lane},
      year={2021},
      eprint={2104.01481},
      note={GNNSys Workshop, MLSys '21 and HAET Workshop, ICLR '21}
    }

    Differences compared to the paper:
    * Allow using a single but different aggregator similar to EGC-M.
    * Different bases per edge-type
    """

    def __init__(
        self,
        input_state_dimension: int,
        output_state_dimension: int,
        num_edge_types: int,
        message_aggregation_function: str,
        num_bases: int = 4,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.__input_state_dim = input_state_dimension
        assert output_state_dimension % num_heads == 0
        self.__aggregation_fn = message_aggregation_function

        self.__num_bases = num_bases
        self.__num_heads = num_heads
        self.__output_state_dim = output_state_dimension

        self.__dropout = nn.Dropout(p=dropout_rate)

        self.__bases = nn.ModuleList(
            [
                nn.Linear(input_state_dimension, num_bases * output_state_dimension, bias=False)
                for _ in range(num_edge_types)
            ]
        )
        self.__weight_coeffs = nn.Linear(input_state_dimension, num_heads * num_bases)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        edge_features: List[torch.Tensor],
    ) -> torch.Tensor:
        assert len(adjacency_lists) == len(self.__bases)
        node_weights = self.__weight_coeffs(node_states).reshape(
            -1, self.__num_heads, self.__num_bases, 1
        )

        all_message_targets, all_messages = [], []
        for edge_type_idx, (adj_list, features, edge_bases_layer) in enumerate(
            zip(adjacency_lists, edge_features, self.__bases)
        ):
            edge_sources_idxs, edge_target_idxs = adj_list
            all_message_targets.append(edge_target_idxs)

            edge_source_states = nn.functional.embedding(edge_sources_idxs, node_states)
            edge_source_states = self.__dropout(edge_source_states)
            all_base_inputs = edge_bases_layer(edge_source_states).reshape(
                -1, self.__num_heads, self.__num_bases, self.__output_state_dim // self.__num_heads
            )
            all_messages.append(all_base_inputs)

        aggregated_messages = self._aggregate_messages(
            messages=torch.cat(all_messages, dim=0),
            message_targets=torch.cat(all_message_targets, dim=0),
            num_nodes=node_states.shape[0],
            aggregation_fn=self.__aggregation_fn,
        )  # [num_nodes, num_heads, num_bases, D]

        return (
            (aggregated_messages * node_weights).sum(axis=-2).reshape(-1, self.__output_state_dim)
        )

    @property
    def input_state_dimension(self) -> int:
        return self.__input_state_dim

    @property
    def output_state_dimension(self) -> int:
        return self.__output_state_dim
