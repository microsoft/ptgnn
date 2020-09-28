from typing_extensions import Final

import numpy as np
import torch
from torch import nn
from typing import Any, Dict, Optional, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.baseneuralmodel.utils.data import enforce_not_None
from ptgnn.neuralmodels.gnn.structs import AbstractNodeEmbedder


class LinearFeatureEmbedder(nn.Module):
    def __init__(
        self,
        input_element_size: int,
        output_embedding_size: int,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.__linear_map = nn.Linear(input_element_size, output_embedding_size, bias=False)
        torch.nn.init.xavier_uniform(self.__linear_map.weight)
        self.__activation = activation

    def forward(self, features):
        mapped_features = self.__linear_map(features)
        if self.__activation is not None:
            mapped_features = self.__activation(mapped_features)
        return mapped_features


class FeatureRepresentationModel(
    AbstractNeuralModel[np.ndarray, np.ndarray, LinearFeatureEmbedder],
    AbstractNodeEmbedder,
):
    """
    A model that maps a feature array to a D-sized representation (embedding) using a single linear layer.
    """

    def __init__(
        self,
        *,
        embedding_size: int = 64,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embedding_size: Final = embedding_size
        self.__activation: Final = activation

    def representation_size(self) -> int:
        return self.embedding_size

    def initialize_metadata(self) -> None:
        self.__num_input_features = None

    def update_metadata_from(self, datapoint: np.ndarray) -> None:
        if self.__num_input_features is None:
            self.__num_input_features = datapoint.shape[0]
        else:
            assert (
                self.__num_input_features == datapoint.shape[0]
            ), "All samples should have the same number of features."

    def build_neural_module(self) -> LinearFeatureEmbedder:
        return LinearFeatureEmbedder(
            input_element_size=enforce_not_None(self.__num_input_features),
            output_embedding_size=self.embedding_size,
            activation=self.__activation,
        )

    def tensorize(self, datapoint: np.ndarray) -> np.ndarray:
        return datapoint

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"features": []}

    def extend_minibatch_with(
        self, tensorized_datapoint: np.ndarray, partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["features"].append(tensorized_datapoint)
        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "features": torch.tensor(
                accumulated_minibatch_data["features"], dtype=torch.float32, device=device
            )
        }
