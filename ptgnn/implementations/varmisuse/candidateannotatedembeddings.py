import torch
from torch import nn
from typing import Any, Dict, Tuple

from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import (
    StrElementRepresentationModel,
)
from ptgnn.neuralmodels.gnn.structs import AbstractNodeEmbedder


class CandidateNodeAnnotationModule(nn.Module):
    def __init__(self, node_embeddings_module):
        super().__init__()
        self.__node_embedding_module = node_embeddings_module

    def forward(self, node_data, is_candidate):
        embeddings = self.__node_embedding_module(**node_data)
        return torch.cat((embeddings, is_candidate.unsqueeze(-1)), dim=-1)


class CandidateNodeAnnotationModel(
    AbstractNeuralModel[Tuple[str, bool], Any, CandidateNodeAnnotationModule],
    AbstractNodeEmbedder,
):
    def __init__(self, embedding_size: int = 128, **kwargs):
        super().__init__()
        self.__str_node_annotation = StrElementRepresentationModel(
            embedding_size=embedding_size - 1, **kwargs
        )

    def update_metadata_from(self, datapoint: Tuple[str, bool]) -> None:
        self.__str_node_annotation.update_metadata_from(datapoint[0])

    def build_neural_module(self) -> CandidateNodeAnnotationModule:
        return CandidateNodeAnnotationModule(
            node_embeddings_module=self.__str_node_annotation.build_neural_module(),
        )

    def tensorize(self, datapoint: Tuple[str, bool]) -> Tuple[Any, bool]:
        return self.__str_node_annotation.tensorize(datapoint[0]), datapoint[1]

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"node_data": self.__str_node_annotation.initialize_minibatch(), "is_candidate": []}

    def extend_minibatch_with(
        self, tensorized_datapoint: Tuple[Any, bool], partial_minibatch: Dict[str, Any]
    ) -> bool:
        continue_extending = self.__str_node_annotation.extend_minibatch_with(
            tensorized_datapoint[0], partial_minibatch["node_data"]
        )
        partial_minibatch["is_candidate"].append(tensorized_datapoint[1])
        return continue_extending

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Any
    ) -> Dict[str, Any]:
        return {
            "node_data": self.__str_node_annotation.finalize_minibatch(
                accumulated_minibatch_data["node_data"], device=device
            ),
            "is_candidate": torch.tensor(
                accumulated_minibatch_data["is_candidate"], dtype=torch.float32, device=device
            ),
        }

    def representation_size(self) -> int:
        return self.__str_node_annotation.representation_size() + 1
