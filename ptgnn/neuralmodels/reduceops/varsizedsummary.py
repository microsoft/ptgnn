from typing_extensions import Literal

import torch
from abc import abstractmethod
from math import sqrt
from torch import nn
from torch_scatter import scatter, scatter_log_softmax, scatter_sum
from typing import NamedTuple, Union


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
        )  # [num_elements]
        attention_probs = torch.exp(
            scatter_log_softmax(attention_scores, index=inputs.element_to_sample_map, dim=0, eps=0)
        )  # [num_elements]
        return scatter_sum(
            self.__output_layer(inputs.element_embeddings) * attention_probs.unsqueeze(-1),
            index=inputs.num_samples,
            dim=0,
            dim_size=inputs.num_samples,
        )  # [num_samples, D']


class WeightedSumVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(self, representation_size: int):
        super().__init__()
        self.__weights_layer = nn.Linear(representation_size, 1, bias=False)

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        weights = torch.sigmoid(
            self.__weights_layer(inputs.element_embeddings).squeeze(-1)
        )  # [num_elements]
        return scatter_sum(
            inputs.element_embeddings * weights.unsqueeze(-1),
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
        )  # [num_samples, D']


class SelfAttentionVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(
        self,
        input_representation_size: int,
        hidden_size: int,
        output_representation_size: int,
        query_representation_summarizer: AbstractVarSizedElementReduce,
    ):
        super().__init__()
        self.__query_layer = query_representation_summarizer
        self.__key_layer = nn.Linear(input_representation_size, hidden_size, bias=False)
        self.__output_layer = nn.Linear(
            input_representation_size, output_representation_size, bias=False
        )

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        queries = self.__query_layer(inputs)  # [num_samples, H]
        queries_all = queries[inputs.element_to_sample_map]  # [num_elements, H]
        keys = self.__key_layer(inputs.element_embeddings)  # [num_elements, H]

        attention_scores = torch.einsum("vh,vh->v", queries_all, keys)  # [num_elements]
        attention_probs = torch.exp(
            scatter_log_softmax(attention_scores, index=inputs.element_to_sample_map, dim=0, eps=0)
        )  # [num_elements]
        return scatter_sum(
            self.__output_layer(inputs.element_embeddings) * attention_probs.unsqueeze(-1),
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
        )  # [num_samples, D']


class MultiheadSelfAttentionVarSizedElementReduce(AbstractVarSizedElementReduce):
    def __init__(
        self,
        input_representation_size: int,
        hidden_size: int,
        output_representation_size: int,
        num_heads: int,
        query_representation_summarizer: AbstractVarSizedElementReduce,
        use_value_layer: bool = False,
    ):
        super().__init__()
        self.__query_layer = query_representation_summarizer
        self.__key_layer = nn.Linear(input_representation_size, hidden_size, bias=False)
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."
        self.__use_value_layer = use_value_layer
        if use_value_layer:
            self.__value_layer = nn.Linear(input_representation_size, hidden_size, bias=False)
            self.__output_layer = nn.Linear(hidden_size, output_representation_size, bias=False)
        else:
            self.__output_layer = nn.Linear(
                input_representation_size * num_heads, output_representation_size, bias=False
            )
        self.__num_heads = num_heads

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        queries = self.__query_layer(inputs)  # [num_samples, H]
        queries_per_element = queries[inputs.element_to_sample_map]  # [num_elements, H]
        queries_per_element = queries_per_element.reshape(
            (
                queries_per_element.shape[0],
                self.__num_heads,
                queries_per_element.shape[1] // self.__num_heads,
            )
        )

        keys = self.__key_layer(inputs.element_embeddings)  # [num_elements, H]
        keys = keys.reshape((keys.shape[0], self.__num_heads, keys.shape[1] // self.__num_heads))

        attention_scores = torch.einsum("bkh,bkh->bk", queries_per_element, keys) / sqrt(
            keys.shape[-1]
        )  # [num_elements, num_heads]
        attention_probs = torch.exp(
            scatter_log_softmax(attention_scores, index=inputs.element_to_sample_map, dim=0, eps=0)
        )  # [num_elements, num_heads]

        if self.__use_value_layer:
            values = self.__value_layer(inputs.element_embeddings)  # [num_elements, hidden_size]
            values = values.reshape(
                (values.shape[0], self.__num_heads, values.shape[1] // self.__num_heads)
            )
            outputs = attention_probs.unsqueeze(-1) * values
        else:
            outputs = attention_probs.unsqueeze(-1) * inputs.element_embeddings.unsqueeze(
                1
            )  # [num_elements, num_heads, D']

        outputs = outputs.reshape((outputs.shape[0], -1))  # [num_elements, num_heads * D']

        per_sample_outputs = scatter_sum(
            outputs, index=inputs.element_to_sample_map, dim=0, dim_size=inputs.num_samples
        )  # [num_samples, num_heads, D']

        return self.__output_layer(per_sample_outputs)  # [num_samples, D']
