from typing_extensions import TypedDict

import numpy as np
import torch
from dpu_utils.mlutils import Vocabulary
from torch import nn
from typing import Any, Counter, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.baseneuralmodel.utils.data import enforce_not_None
from ptgnn.neuralmodels.gnn import GnnOutput, GraphData, TensorizedGraphData
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetwork, GraphNeuralNetworkModel

SuperNodeData = TypedDict(
    "SuperNodeData",
    {
        "name": str,
        "annotation": Optional[str],
    },
    total=False,
)

TypilusGraph = TypedDict(
    "TypilusGraph",
    {
        "nodes": List[str],
        "edges": Dict[str, Dict[str, List[int]]],
        "token-sequence": List[int],
        "supernodes": Dict[str, SuperNodeData],
        "filename": str,
    },
)

Prediction = Tuple[TypilusGraph, Dict[int, Tuple[str, float]]]


class TensorizedGraph2ClassSample(NamedTuple):
    graph: TensorizedGraphData
    supernode_target_classes: List[int]


IGNORED_TYPES = {
    "typing.Any",
    "Any",
    "",
    "typing.NoReturn",
    "NoReturn",
    "nothing",
    "None",
    "T",
    "_T",
    "_T0",
    "_T1",
    "_T2",
    "_T3",
    "_T4",
    "_T5",
    "_T6",
    "_T7",
}


class Graph2ClassModule(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork, num_target_classes: int):
        super().__init__()
        self.__gnn = gnn
        self.__node_to_class = nn.Linear(
            in_features=gnn.output_node_state_dim, out_features=num_target_classes
        )
        nn.init.uniform_(self.__node_to_class.weight)
        nn.init.zeros_(self.__node_to_class.bias)
        self.__loss = nn.CrossEntropyLoss()

    def _reset_module_metrics(self) -> None:
        self.__num_samples = 0
        self.__sum_accuracy = 0

    def _module_metrics(self) -> Dict[str, Any]:
        return {"Accuracy": self.__sum_accuracy / self.__num_samples}

    def _logits(self, graph_mb_data):
        graph_output: GnnOutput = self.__gnn(**graph_mb_data)
        # Gather the output representation of the nodes of interest
        supernode_idxs = graph_output.node_idx_references["supernodes"]
        supernode_graph_idx = graph_output.node_graph_idx_reference["supernodes"]
        supernode_representations = graph_output.output_node_representations[
            supernode_idxs
        ]  # [num_supernodes_in_mb, D]
        return self.__node_to_class(supernode_representations), supernode_graph_idx

    def predict(self, graph_mb_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, supernode_graph_idx = self._logits(graph_mb_data)
            probs = torch.softmax(logits, dim=-1)
            return torch.max(probs, dim=-1) + (supernode_graph_idx,)

    def forward(self, graph_mb_data, target_classes, original_supernode_idxs):
        logits, _ = self._logits(graph_mb_data)
        with torch.no_grad():
            self.__sum_accuracy += int((torch.argmax(logits, dim=-1) == target_classes).sum())
            self.__num_samples += int(target_classes.shape[0])
        return self.__loss(logits, target_classes)


class Graph2Class(
    AbstractNeuralModel[TypilusGraph, TensorizedGraph2ClassSample, Graph2ClassModule]
):
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel,
        max_num_classes: int = 100,
        try_simplify_unks: bool = True,
    ):
        super().__init__()
        self.__gnn_model = gnn_model
        self.max_num_classes = max_num_classes
        self.__try_simplify_unks = try_simplify_unks
        self.__tensorize_samples_with_no_annotation = False
        self.__tensorize_keep_original_supernode_idx = False

    def __convert(self, typilus_graph: TypilusGraph) -> Tuple[GraphData[str, None], List[str]]:
        def get_adj_list(adjacency_dict):
            for from_node_idx, to_node_idxs in adjacency_dict.items():
                from_node_idx = int(from_node_idx)
                for to_idx in to_node_idxs:
                    yield (from_node_idx, to_idx)

        edges = {}
        for edge_type, adj_dict in typilus_graph["edges"].items():
            adj_list: List[Tuple[int, int]] = list(get_adj_list(adj_dict))
            if len(adj_list) > 0:
                edges[edge_type] = np.array(adj_list, dtype=np.int32)
            else:
                edges[edge_type] = np.zeros((0, 2), dtype=np.int32)

        supernode_idxs_with_ground_truth: List[int] = []
        supernode_annotations: List[str] = []
        for supernode_idx, supernode_data in typilus_graph["supernodes"].items():
            if supernode_data["annotation"] in IGNORED_TYPES:
                continue
            if (
                not self.__tensorize_samples_with_no_annotation
                and supernode_data["annotation"] is None
            ):
                continue
            elif supernode_data["annotation"] is None:
                supernode_data["annotation"] = "??"
            supernode_idxs_with_ground_truth.append(int(supernode_idx))
            supernode_annotations.append(enforce_not_None(supernode_data["annotation"]))

        return (
            GraphData[str, None](
                node_information=typilus_graph["nodes"],
                edges=edges,
                reference_nodes={
                    "token-sequence": typilus_graph["token-sequence"],
                    "supernodes": supernode_idxs_with_ground_truth,
                },
            ),
            supernode_annotations,
        )

    # region Metadata Loading
    def initialize_metadata(self) -> None:
        self.__target_class_counter = Counter[str]()

    def update_metadata_from(self, datapoint: TypilusGraph) -> None:
        graph_data, target_classes = self.__convert(datapoint)
        self.__gnn_model.update_metadata_from(graph_data)
        self.__target_class_counter.update(target_classes)

    def finalize_metadata(self) -> None:
        self.__target_vocab = Vocabulary.create_vocabulary(
            self.__target_class_counter,
            max_size=self.max_num_classes + 1,
        )
        del self.__target_class_counter

    # endregion

    def build_neural_module(self) -> Graph2ClassModule:
        return Graph2ClassModule(
            gnn=self.__gnn_model.build_neural_module(), num_target_classes=len(self.__target_vocab)
        )

    def tensorize(self, datapoint: TypilusGraph) -> Optional[TensorizedGraph2ClassSample]:
        graph_data, target_classes = self.__convert(datapoint)
        if len(target_classes) == 0:
            return None  # Sample contains no ground-truth annotations.

        graph_tensorized_data = self.__gnn_model.tensorize(graph_data)

        if graph_tensorized_data is None:
            return None  # Sample rejected by the GNN

        target_class_ids = []
        for target_cls in target_classes:
            if self.__try_simplify_unks and self.__target_vocab.is_unk(target_cls):
                # TODO: Backoff on the type lattice. For now, just erase generics
                generic_start = target_cls.find("[")
                if generic_start != -1:
                    target_cls = target_cls[:generic_start]
            target_class_ids.append(self.__target_vocab.get_id_or_unk(target_cls))

        return TensorizedGraph2ClassSample(
            graph=graph_tensorized_data, supernode_target_classes=target_class_ids
        )

    # region Minibatching
    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "graph_mb_data": self.__gnn_model.initialize_minibatch(),
            "target_classes": [],
            "original_supernode_idxs": [],
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedGraph2ClassSample, partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["target_classes"].extend(tensorized_datapoint.supernode_target_classes)
        if self.__tensorize_keep_original_supernode_idx:
            partial_minibatch["original_supernode_idxs"].extend(
                tensorized_datapoint.graph.reference_nodes["supernodes"]
            )
        return self.__gnn_model.extend_minibatch_with(
            tensorized_datapoint.graph, partial_minibatch["graph_mb_data"]
        )

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "graph_mb_data": self.__gnn_model.finalize_minibatch(
                accumulated_minibatch_data["graph_mb_data"], device
            ),
            "target_classes": torch.tensor(
                accumulated_minibatch_data["target_classes"], dtype=torch.int64, device=device
            ),
            "original_supernode_idxs": accumulated_minibatch_data["original_supernode_idxs"],
        }

    # endregion

    def report_accuracy(
        self,
        dataset: Iterator[TypilusGraph],
        trained_network: Graph2ClassModule,
        device: Union[str, torch.device],
    ) -> float:
        trained_network.eval()
        unk_class_id = self.__target_vocab.get_id_or_unk(self.__target_vocab.get_unk())

        num_correct, num_elements = 0, 0
        for mb_data, _ in self.minibatch_iterator(
            self.tensorize_dataset(dataset), device, max_minibatch_size=50
        ):
            _, predictions, _ = trained_network.predict(mb_data["graph_mb_data"])
            for target_idx, prediction in zip(mb_data["target_classes"], predictions):
                num_elements += 1
                if target_idx == prediction and target_idx != unk_class_id:
                    num_correct += 1
        return num_correct / num_elements

    def predict(
        self,
        data: Iterator[TypilusGraph],
        trained_network: Graph2ClassModule,
        device: Union[str, torch.device],
    ) -> Iterator[Prediction]:
        trained_network.eval()
        with torch.no_grad():
            try:
                self.__tensorize_samples_with_no_annotation = True
                self.__tensorize_keep_original_supernode_idx = True

                for mb_data, original_datapoints in self.minibatch_iterator(
                    self.tensorize_dataset(data, return_input_data=True, parallelize=False),
                    device,
                    max_minibatch_size=50,
                    parallelize=False,
                ):
                    current_graph_idx = 0
                    graph_preds: Dict[int, Tuple[str, float]] = {}

                    probs, predictions, graph_idxs = trained_network.predict(
                        mb_data["graph_mb_data"]
                    )
                    supernode_idxs = mb_data["original_supernode_idxs"]
                    for graph_idx, prediction_prob, prediction_id, supernode_idx in zip(
                        graph_idxs, probs, predictions, supernode_idxs
                    ):
                        if graph_idx != current_graph_idx:
                            yield original_datapoints[current_graph_idx], graph_preds
                            current_graph_idx = graph_idx
                            graph_preds: Dict[int, Tuple[str, float]] = {}

                        predicted_type = self.__target_vocab.get_name_for_id(prediction_id)
                        graph_preds[supernode_idx] = predicted_type, float(prediction_prob)
                    yield original_datapoints[current_graph_idx], graph_preds
            finally:
                self.__tensorize_samples_with_no_annotation = False
                self.__tensorize_keep_original_supernode_idx = False
