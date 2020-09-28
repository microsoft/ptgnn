import numpy as np
import torch
from torch import nn
from typing import Any, Dict, Iterator, NamedTuple, Optional, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.baseneuralmodel.utils.data import enforce_not_None
from ptgnn.tests.simplemodel.data import SampleDatapoint


class TensorizedDatapoint(NamedTuple):
    input_features: np.ndarray
    target_class: np.ndarray


class SimpleRegressionNetwork(ModuleWithMetrics):
    def __init__(self, num_features: int):
        super().__init__()
        self.__layer = nn.Linear(num_features, 1, bias=True)
        self.__loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        predicted = self.__layer(inputs)[:, 0]  # B
        loss = self.__loss(input=predicted, target=targets)
        return loss

    def predict(self, inputs: torch.Tensor, **kwargs):
        predicted = self.__layer(inputs)[:, 0]  # B
        return predicted >= 0


class SimpleRegressionModel(
    AbstractNeuralModel[SampleDatapoint, TensorizedDatapoint, SimpleRegressionNetwork]
):
    """A simple linear regression model used for testing."""

    def initialize_metadata(self) -> None:
        self.__num_features: Optional[int] = None

    def update_metadata_from(self, datapoint: SampleDatapoint) -> None:
        if self.__num_features is None:
            self.__num_features = len(datapoint.input_features)
        else:
            assert self.__num_features == len(datapoint.input_features)

    def build_neural_module(self) -> SimpleRegressionNetwork:
        return SimpleRegressionNetwork(enforce_not_None(self.__num_features))

    def tensorize(self, datapoint: SampleDatapoint) -> TensorizedDatapoint:
        return TensorizedDatapoint(
            input_features=np.array(datapoint.input_features, dtype=np.float32),
            target_class=np.array(1 if datapoint.target_class else 0, dtype=np.float32),
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"inputs": [], "targets": []}

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedDatapoint, partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["inputs"].append(tensorized_datapoint.input_features)
        partial_minibatch["targets"].append(tensorized_datapoint.target_class)
        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "inputs": torch.tensor(
                np.stack(accumulated_minibatch_data["inputs"], axis=0), device=device
            ),
            "targets": torch.tensor(
                np.stack(accumulated_minibatch_data["targets"], axis=0), device=device
            ),
        }

    def compute_accuracy(
        self,
        trained_network: SimpleRegressionNetwork,
        dataset: Iterator[SampleDatapoint],
        parallelize: bool = True,
        use_multiprocessing=True,
    ) -> float:
        trained_network.eval()
        num_samples, num_correct = 0, 0

        for mb_data, initial_points in self.minibatch_iterator(
            self.tensorize_dataset(
                dataset,
                return_input_data=True,
                parallelize=parallelize,
                use_multiprocessing=use_multiprocessing,
            ),
            max_minibatch_size=10,
            parallelize=parallelize,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        ):
            mb_predictions = trained_network.predict(**mb_data).cpu().numpy()
            for point, prediction in zip(initial_points, mb_predictions):
                num_samples += 1
                if point is None:
                    raise Exception("Inital Point Data was not loaded.")
                if point.target_class == prediction:
                    num_correct += 1
        return num_correct / num_samples
