from typing_extensions import Final, TypedDict

import re
import torch
from dpu_utils.codeutils import split_identifier_into_parts
from itertools import chain
from torch import nn
from torch_scatter import scatter_log_softmax, scatter_max
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetwork, GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.structs import GnnOutput, GraphData, TensorizedGraphData


class VarMisuseGraph(TypedDict):
    Edges: Dict[str, List[Tuple[int, int]]]
    NodeLabels: Dict[str, str]
    NodeTypes: Dict[str, str]


class SymbolCandidate(TypedDict):
    SymbolDummyNode: int
    SymbolName: str
    IsCorrect: bool


class VarMisuseSample(TypedDict):
    ContextGraph: VarMisuseGraph
    slotTokenIdx: int
    SlotDummyNode: int
    SymbolCandidates: List[SymbolCandidate]


class TensorizedVarMisuseSample(NamedTuple):
    graph: TensorizedGraphData
    target_candidate_node_idx: int
    num_candidate_nodes: int


class VarMisuseGraphModel(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork):
        super().__init__()
        self._gnn = gnn
        self.__candidate_scores = nn.Linear(
            self._gnn.output_node_state_dim + self._gnn.output_node_state_dim,
            1,
            bias=False,
        )

    def _reset_module_metrics(self) -> None:
        self.__sum_acc = 0
        self.__num_samples = 0

    def _module_metrics(self) -> Dict[str, Any]:
        return {"Accuracy": self.__sum_acc / self.__num_samples}

    def forward(self, graph_data, correct_candidate_idxs):
        gnn_output: GnnOutput = self._gnn(**graph_data)

        # Code assumes that there is one slot per-graph, which is true for the original data

        candidate_node_representations = gnn_output.output_node_representations[
            gnn_output.node_idx_references["candidate_nodes"]
        ]  # [num_candidate_nodes, H_out]
        candidate_nodes_slot_idx = gnn_output.node_graph_idx_reference[
            "candidate_nodes"
        ]  # [num_candidate_nodes]

        slot_representations = gnn_output.output_node_representations[
            gnn_output.node_idx_references["slot_node_idx"]
        ]  # [num_slot_nodes, H_out]
        slot_representations_per_candidate = slot_representations[
            candidate_nodes_slot_idx
        ]  # [num_candidate_nodes, H]
        candidate_scores = self.__candidate_scores(
            torch.cat((candidate_node_representations, slot_representations_per_candidate), dim=-1)
        ).squeeze(-1)
        candidate_nodes_logprobs = scatter_log_softmax(
            src=candidate_scores, index=candidate_nodes_slot_idx, eps=0
        )

        with torch.no_grad():
            self.__sum_acc += int(
                (
                    scatter_max(candidate_scores, index=candidate_nodes_slot_idx)[1]
                    == correct_candidate_idxs
                ).sum()
            )
            self.__num_samples += int(slot_representations.shape[0])
        return -candidate_nodes_logprobs[correct_candidate_idxs].mean()


class VarMisuseModel(
    AbstractNeuralModel[VarMisuseSample, TensorizedVarMisuseSample, VarMisuseGraphModel]
):
    def __init__(self, gnn_model: GraphNeuralNetworkModel):
        super().__init__()
        self.__gnn_model = gnn_model

    IDENTIFIER_REGEX: Final = re.compile("[a-zA-Z][a-zA-Z0-9]*")

    @classmethod
    def __add_subtoken_vocab_nodes(cls, graph: GraphData[Tuple[str, bool]]) -> None:
        all_token_nodes = set(chain(*graph.edges["NextToken"]))

        subtoken_edges: List[Tuple[int, int]] = []
        subtoken_node_ids: Dict[str, int] = {}

        for token_node_idx in all_token_nodes:
            token_text = graph.node_information[token_node_idx][0]
            if not cls.IDENTIFIER_REGEX.match(token_text):
                continue
            for subtoken in split_identifier_into_parts(token_text):
                subtoken_node_idx = subtoken_node_ids.get(subtoken)
                if subtoken_node_idx is None:
                    subtoken_node_idx = len(graph.node_information)
                    graph.node_information.append((subtoken, False))
                    subtoken_node_ids[subtoken] = subtoken_node_idx

                subtoken_edges.append((subtoken_node_idx, token_node_idx))

        graph.edges["SubtokenOf"] = subtoken_edges

    def update_metadata_from(self, datapoint: VarMisuseSample) -> None:
        graph = datapoint["ContextGraph"]
        graph_data = GraphData(
            node_information=[
                (graph["NodeLabels"][str(i)], False) for i in range(len(graph["NodeLabels"]))
            ],
            edges=graph["Edges"],
            reference_nodes={},  # This is not needed for metadata loading
        )
        self.__add_subtoken_vocab_nodes(graph_data)
        self.__gnn_model.update_metadata_from(graph_data)

    def build_neural_module(self) -> VarMisuseGraphModel:
        gnn = self.__gnn_model.build_neural_module()
        return VarMisuseGraphModel(gnn)

    def tensorize(self, datapoint: VarMisuseSample) -> Optional[TensorizedVarMisuseSample]:
        graph = datapoint["ContextGraph"]
        all_correct_slots = [
            i
            for i, cand_symbol in enumerate(datapoint["SymbolCandidates"])
            if cand_symbol["IsCorrect"]
        ]
        assert len(all_correct_slots) == 1

        candidate_node_ids = {s["SymbolDummyNode"] for s in datapoint["SymbolCandidates"]}
        graph_data = GraphData(
            node_information=[
                (graph["NodeLabels"][str(i)], i in candidate_node_ids)
                for i in range(len(graph["NodeLabels"]))
            ],
            edges=graph["Edges"],
            reference_nodes={
                "candidate_nodes": [s["SymbolDummyNode"] for s in datapoint["SymbolCandidates"]],
                "slot_node_idx": [datapoint["SlotDummyNode"]],
            },
        )

        if graph_data is None:
            return None

        self.__add_subtoken_vocab_nodes(graph_data)
        tensorized_graph_data = self.__gnn_model.tensorize(graph_data)
        if tensorized_graph_data is None:
            return None

        return TensorizedVarMisuseSample(
            graph=tensorized_graph_data,
            target_candidate_node_idx=all_correct_slots[0],
            num_candidate_nodes=len(datapoint["SymbolCandidates"]),
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "graph_data": self.__gnn_model.initialize_minibatch(),
            "correct_candidate_idxs": [],
            "total_num_candidate_nodes": 0,
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedVarMisuseSample, partial_minibatch: Dict[str, Any]
    ) -> bool:
        continue_adding = self.__gnn_model.extend_minibatch_with(
            tensorized_datapoint.graph, partial_minibatch["graph_data"]
        )
        partial_minibatch["correct_candidate_idxs"].append(
            tensorized_datapoint.target_candidate_node_idx
            + partial_minibatch["total_num_candidate_nodes"]
        )
        partial_minibatch["total_num_candidate_nodes"] += tensorized_datapoint.num_candidate_nodes
        return continue_adding

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "graph_data": self.__gnn_model.finalize_minibatch(
                accumulated_minibatch_data["graph_data"], device=device
            ),
            "correct_candidate_idxs": torch.tensor(
                accumulated_minibatch_data["correct_candidate_idxs"],
                dtype=torch.int64,
                device=device,
            ),
        }

    def report_accuracy(
        self,
        dataset: Iterator[VarMisuseSample],
        trained_network: VarMisuseGraphModel,
        device: Union[str, torch.device],
    ) -> float:
        trained_network.eval()
        trained_network._reset_module_metrics()
        for mb_data, _ in self.minibatch_iterator(
            self.tensorize_dataset(dataset), device, max_minibatch_size=50
        ):
            with torch.no_grad():
                trained_network(**mb_data)
        return trained_network.report_metrics()["Accuracy"]
