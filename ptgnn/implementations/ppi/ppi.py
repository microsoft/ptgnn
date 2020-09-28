import numpy as np
import torch
from torch import nn
from typing import Any, Dict, Iterable, NamedTuple, Optional, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.baseneuralmodel.utils.data import enforce_not_None
from ptgnn.implementations.ppi.dataloader import PPIGraphSample
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetwork, GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.structs import GnnOutput, GraphData, TensorizedGraphData


class PPIClassification(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork, num_target_classes: int):
        super().__init__()
        self.__gnn = gnn
        self.__output_representation_to_logits = nn.Linear(
            in_features=gnn.output_node_state_dim, out_features=num_target_classes
        )
        torch.nn.init.xavier_uniform(self.__output_representation_to_logits.weight)
        torch.nn.init.zeros_(self.__output_representation_to_logits.bias)
        self.__output_representation_to_logits

    def _reset_module_metrics(self) -> None:
        self.__num_samples = 0
        self.__sum_f1 = 0.0
        self.__sum_pr = 0.0
        self.__sum_re = 0.0

    def _module_metrics(self) -> Dict[str, Any]:
        return {
            "f1_score": self.__sum_f1 / self.__num_samples,
            "pr_score": self.__sum_pr / self.__num_samples,
            "re_score": self.__sum_re / self.__num_samples,
        }

    def forward(self, graph_data, targets):
        gnn_output: GnnOutput = self.__gnn(**graph_data)
        target_logits = self.__output_representation_to_logits(
            gnn_output.output_node_representations
        )

        with torch.no_grad():
            predictions = torch.sigmoid(target_logits) >= 0.5

            true_positives = (predictions & targets).sum().float()
            false_positives = (predictions & (~targets)).sum().float()
            false_negatives = ((~predictions) & targets).sum().float()

            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            fscore = 2 * precision * recall / (precision + recall + 1e-10)
            num_samples = int(predictions.shape[0])
            self.__sum_f1 += float(fscore.cpu()) * num_samples
            self.__sum_pr += float(precision.cpu()) * num_samples
            self.__sum_re += float(recall.cpu()) * num_samples
            self.__num_samples += num_samples

        losses = nn.functional.binary_cross_entropy_with_logits(
            input=target_logits, target=targets.float(), reduction="none"
        )
        return losses.sum(dim=-1).mean()


class TensorizedPPIData(NamedTuple):
    gnn_data: TensorizedGraphData[np.ndarray]
    targets: np.ndarray


class PPIMulticlassClassification(
    AbstractNeuralModel[PPIGraphSample, TensorizedPPIData, PPIClassification]
):
    def __init__(self, gnn_model: GraphNeuralNetworkModel[np.ndarray, np.ndarray]):
        super().__init__()
        self.__gnn_model = gnn_model

    def initialize_metadata(self) -> None:
        self.__num_target_labels: Optional[int] = None

    def update_metadata_from(self, datapoint: PPIGraphSample) -> None:
        self.__gnn_model.update_metadata_from(
            GraphData(
                node_information=datapoint.node_features,
                edges={f"e{i}": a for i, a in enumerate(datapoint.adjacency_lists)},
                reference_nodes={},
            ),
        )
        if self.__num_target_labels is None:
            self.__num_target_labels = datapoint.node_labels.shape[1]
        else:
            assert self.__num_target_labels == datapoint.node_labels.shape[1]

    def build_neural_module(self) -> PPIClassification:
        gnn = self.__gnn_model.build_neural_module()
        return PPIClassification(gnn, enforce_not_None(self.__num_target_labels))

    def tensorize(self, datapoint: PPIGraphSample) -> Optional[TensorizedPPIData]:
        graph_tensors = self.__gnn_model.tensorize(
            GraphData[np.ndarray](
                node_information=datapoint.node_features,
                edges={f"e{i}": a for i, a in enumerate(datapoint.adjacency_lists)},
                reference_nodes={},
            )
        )

        if graph_tensors is None:
            return None

        return TensorizedPPIData(graph_tensors, datapoint.node_labels)

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"graph_data": self.__gnn_model.initialize_minibatch(), "labels": []}

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedPPIData, partial_minibatch: Dict[str, Any]
    ) -> bool:
        continue_adding = self.__gnn_model.extend_minibatch_with(
            tensorized_datapoint.gnn_data, partial_minibatch["graph_data"]
        )
        partial_minibatch["labels"].append(tensorized_datapoint.targets)
        return continue_adding

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "graph_data": self.__gnn_model.finalize_minibatch(
                accumulated_minibatch_data["graph_data"], device
            ),
            "targets": torch.tensor(
                np.concatenate(accumulated_minibatch_data["labels"], axis=0),
                dtype=torch.bool,
                device=device,
            ),
        }

    def report_metrics(
        self,
        dataset: Iterable[PPIGraphSample],
        trained_network: PPIClassification,
        device: Union[str, torch.device],
    ) -> Dict[str, float]:
        trained_network.eval()
        trained_network._reset_module_metrics()
        for mb_data, _ in self.minibatch_iterator(
            self.tensorize_dataset(iter(dataset)), device, max_minibatch_size=50
        ):
            with torch.no_grad():
                trained_network(**mb_data)
        return trained_network.report_metrics()
