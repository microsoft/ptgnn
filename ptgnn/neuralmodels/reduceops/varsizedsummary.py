from abc import abstractmethod
from typing import NamedTuple, Union
from typing_extensions import Literal

import torch
from torch import nn
from torch_scatter import scatter, scatter_log_softmax, scatter_sum


class ElementsToSummaryRepresentationInput(NamedTuple):
    """Input to AbstractVarSizedElementReduce layers."""

    element_embeddings: torch.Tensor  # float tensor of shape [num_elements, D], the representation of each node in all graphs.
    element_to_sample_map: torch.Tensor  # int tensor of shape [num_elements] with values in range [0, num_sampless-1], mapping each node to a sample ID.

    num_samples: Union[torch.Tensor, int]  # scalar, specifying the number of sets.


class AbstractVarSizedElementReduce(nn.Module):
    """Interface for computing summary representations from multiple variable-sized sets of representations."""

    @abstractmethod
    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        """Returns: float tensor of shape [num_samples, D']"""


class SimpleVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(self, summarization_type: Literal["sum", "mean", "max", "min"]):
        super().__init__()
        assert summarization_type in {"sum", "mean", "max", "min"}
        self.__summarization_type = summarization_type

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        return scatter(
            src=inputs.element_embeddings,
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
            reduce=self.__summarization_type,
        )


class NormalizedWeightsVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(self, input_representation_size: int, output_representation_size: int):
        super().__init__()
        self.__attention_layer = nn.Linear(input_representation_size, 1, bias=False)
        self.__output_layer = nn.Linear(
            input_representation_size, output_representation_size, bias=False
        )

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        attention_scores = self.__attention_layer(inputs.element_embeddings).squeeze(
            -1
        )  # [num_vertices]
        attention_probs = torch.exp(
            scatter_log_softmax(attention_scores, index=inputs.element_to_sample_map, dim=0, eps=0)
        )  # [num_vertices]
        return scatter_sum(
            self.__output_layer(inputs.element_embeddings) * attention_probs.unsqueeze(-1),
            index=inputs.num_samples,
            dim=0,
            dim_size=inputs.num_samples,
        )  # [num_graphs, D']


class WeightedSumVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(self, representation_size: int):
        super().__init__()
        self.__weights_layer = nn.Linear(representation_size, 1, bias=False)

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        weights = torch.sigmoid(
            self.__weights_layer(inputs.element_embeddings).squeeze(-1)
        )  # [num_vertices]
        return scatter_sum(
            inputs.element_embeddings * weights.unsqueeze(-1),
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
        )  # [num_graphs, D']


class SelfAttentionVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(
        self,
        input_representation_size: int,
        hidden_size: int,
        output_representation_size: int,
        key_representation_summarizer: AbstractVarSizedElementReduce,
    ):
        super().__init__()
        self.__key_layer = key_representation_summarizer
        self.__value_layer = nn.Linear(input_representation_size, hidden_size, bias=False)
        self.__output_layer = nn.Linear(
            input_representation_size, output_representation_size, bias=False
        )

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        keys = self.__key_layer(inputs)  # [num_graphs, H]
        keys_all = keys[inputs.element_to_sample_map]  # [num_vertices, H]
        values = self.__value_layer(inputs.element_embeddings)  # [num_vertices, H]

        attention_scores = torch.einsum("vh,vh->v", keys_all, values)  # [num_vertices]
        attention_probs = torch.exp(
            scatter_log_softmax(attention_scores, index=inputs.element_to_sample_map, dim=0, eps=0)
        )  # [num_vertices]
        return scatter_sum(
            self.__output_layer(inputs.element_embeddings) * attention_probs.unsqueeze(-1),
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
        )  # [num_graphs, D']


class MultiheadSelfAttentionVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(
        self,
        input_representation_size: int,
        hidden_size: int,
        output_representation_size: int,
        num_heads: int,
        key_representation_summarizer: AbstractVarSizedElementReduce,
    ):
        super().__init__()
        self.__query_layer = key_representation_summarizer
        self.__value_layer = nn.Linear(input_representation_size, hidden_size, bias=False)
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."
        self.__num_heads = num_heads
        self.__output_layer = nn.Linear(
            input_representation_size * num_heads, output_representation_size, bias=False
        )

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        query = self.__query_layer(inputs)  # [num_graphs, H]
        query_per_node = query[inputs.element_to_sample_map]  # [num_vertices, H]
        values = self.__value_layer(inputs.element_embeddings)  # [num_vertices, H]

        query_per_node = values.reshape(
            (query_per_node.shape[0], self.__num_heads, query_per_node.shape[1] // self.__num_heads)
        )
        values = values.reshape(
            (values.shape[0], self.__num_heads, values.shape[1] // self.__num_heads)
        )

        attention_scores = torch.einsum(
            "vkh,vkh->vk", query_per_node, values
        )  # [num_vertices, num_heads]
        attention_probs = torch.exp(
            scatter_log_softmax(attention_scores, index=inputs.element_to_sample_map, dim=0, eps=0)
        )  # [num_vertices, num_heads]

        outputs = attention_probs.unsqueeze(-1) * inputs.element_embeddings.unsqueeze(
            1
        )  # [num_vertices, num_heads, D']
        per_graph_outputs = scatter_sum(
            outputs, index=inputs.element_to_sample_map, dim=0, dim_size=inputs.num_samples
        )  # [num_graphs, num_heads, D']
        per_graph_outputs = per_graph_outputs.reshape(
            (per_graph_outputs.shape[0], -1)
        )  # [num_graphs, num_heads * D']

        return self.__output_layer(per_graph_outputs)  # [num_graphs, D']
