from typing_extensions import Final, TypedDict

import torch
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.baseneuralmodel.utils.data import enforce_not_None
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetwork, GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.structs import GnnOutput, GraphData, TensorizedGraphData
from ptgnn.neuralmodels.reduceops import (
    AbstractVarSizedElementReduce,
    ElementsToSummaryRepresentationInput,
    MultiheadSelfAttentionVarSizedElementReduce,
    SimpleVarSizedElementReduce,
)
from ptgnn.neuralmodels.sequence.grucopydecoder import (
    DecoderData,
    GruCopyingDecoder,
    GruCopyingDecoderModel,
    TokenizedOutput,
)


class CodeGraph2Seq(TypedDict):
    backbone_sequence: List[int]
    node_labels: List[str]
    edges: Dict[str, List[Tuple[int, int]]]
    method_name: List[str]


class TensorizedGraph2Seq(NamedTuple):
    encoder_data: TensorizedGraphData
    decoder_data: TokenizedOutput


class Graph2SeqModule(ModuleWithMetrics):
    def __init__(
        self,
        gnn: GraphNeuralNetwork,
        decoder: GruCopyingDecoder,
        node_to_graph_representation: AbstractVarSizedElementReduce,
    ):
        super().__init__()
        self._gnn = gnn
        self._decoder = decoder
        self.__node_to_graph_representation = node_to_graph_representation

    def _reset_module_metrics(self) -> None:
        self.__loss_sum = 0.0
        self.__num_mbs = 0

    def _module_metrics(self) -> Dict[str, Any]:
        return {"loss": self.__loss_sum / self.__num_mbs}

    def _get_initial_decoder_states(self, gnn_output: GnnOutput):
        return self.__node_to_graph_representation(
            ElementsToSummaryRepresentationInput(
                element_embeddings=torch.cat(
                    (gnn_output.input_node_representations, gnn_output.output_node_representations),
                    dim=-1,
                ),
                element_to_sample_map=gnn_output.node_to_graph_idx,
                num_samples=gnn_output.num_graphs,
            )
        )

    def forward(self, *, encoder_mb_data: Dict[str, Any], decoder_mb_data: Dict[str, Any]):
        gnn_output: GnnOutput = self._gnn(**encoder_mb_data)

        loss = self._decoder(
            input_memories=gnn_output.output_node_representations[
                gnn_output.node_idx_references["backbone_nodes"]
            ],
            input_memories_origin_idx=gnn_output.node_graph_idx_reference["backbone_nodes"],
            initial_states=self._get_initial_decoder_states(gnn_output),
            **decoder_mb_data
        )
        with torch.no_grad():
            self.__loss_sum += float(loss.cpu())
            self.__num_mbs += 1
        return loss


class Graph2Seq(AbstractNeuralModel[CodeGraph2Seq, TensorizedGraph2Seq, Graph2SeqModule]):
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel,
        decoder: GruCopyingDecoderModel,
        num_summarization_heads: int = 8,
    ):
        super().__init__()
        self.__gnn_model = gnn_model
        self.__decoder_model = decoder
        self.num_summarization_heads: Final = num_summarization_heads

    def update_metadata_from(self, datapoint: CodeGraph2Seq) -> None:
        graph_nodes = [l.lower() for l in datapoint["node_labels"]]
        self.__gnn_model.update_metadata_from(
            GraphData(
                node_information=graph_nodes,
                edges=datapoint["edges"],
                reference_nodes={"backbone_nodes": datapoint["backbone_sequence"]},
            )
        )

        self.__decoder_model.update_metadata_from(
            DecoderData(
                input_elements=[graph_nodes[k] for k in datapoint["backbone_sequence"]],
                target_data=datapoint["method_name"],
            ),
        )

    def build_neural_module(self) -> Graph2SeqModule:
        gnn = self.__gnn_model.build_neural_module()
        decoder = self.__decoder_model.build_neural_module()
        node_to_graph_representation = MultiheadSelfAttentionVarSizedElementReduce(
            input_representation_size=gnn.input_node_state_dim + gnn.output_node_state_dim,
            hidden_size=gnn.input_node_state_dim + gnn.output_node_state_dim,
            output_representation_size=gnn.output_node_state_dim,
            num_heads=self.num_summarization_heads,
            query_representation_summarizer=SimpleVarSizedElementReduce("max"),
        )
        return Graph2SeqModule(gnn, decoder, node_to_graph_representation)

    def tensorize(self, datapoint: CodeGraph2Seq) -> Optional[TensorizedGraph2Seq]:
        graph_nodes = [l.lower() for l in datapoint["node_labels"]]
        graph_data = self.__gnn_model.tensorize(
            GraphData(
                node_information=graph_nodes,
                edges=datapoint["edges"],
                reference_nodes={"backbone_nodes": datapoint["backbone_sequence"]},
            )
        )
        if graph_data is None:
            return None  # Discard example

        target_data = self.__decoder_model.tensorize(
            DecoderData(
                input_elements=[graph_nodes[k] for k in datapoint["backbone_sequence"]],
                target_data=datapoint["method_name"],
            )
        )

        return TensorizedGraph2Seq(encoder_data=graph_data, decoder_data=target_data)

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "encoder_mb_data": self.__gnn_model.initialize_minibatch(),
            "decoder_mb_data": self.__decoder_model.initialize_minibatch(),
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: TensorizedGraph2Seq, partial_minibatch: Dict[str, Any]
    ) -> bool:
        continue_adding = self.__gnn_model.extend_minibatch_with(
            tensorized_datapoint.encoder_data, partial_minibatch["encoder_mb_data"]
        )
        continue_adding &= self.__decoder_model.extend_minibatch_with(
            tensorized_datapoint.decoder_data, partial_minibatch["decoder_mb_data"]
        )
        return continue_adding

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "encoder_mb_data": self.__gnn_model.finalize_minibatch(
                accumulated_minibatch_data["encoder_mb_data"], device=device
            ),
            "decoder_mb_data": self.__decoder_model.finalize_minibatch(
                accumulated_minibatch_data["decoder_mb_data"], device=device
            ),
        }

    def greedy_decode(
        self, data: List[CodeGraph2Seq], trained_network: Graph2SeqModule, device: Any
    ) -> List[Tuple[List[str], float]]:
        decoded_sequences = []
        for mb_data, input_data in self.minibatch_iterator(
            self.tensorize_dataset(iter(data), return_input_data=True),
            device,
            max_minibatch_size=50,
        ):
            input_concrete_values: List[str] = []
            for sample in input_data:
                sample = enforce_not_None(sample)
                input_concrete_values.extend(
                    sample["node_labels"][k].lower() for k in sample["backbone_sequence"]
                )

            with torch.no_grad():
                gnn_output = trained_network._gnn(**mb_data["encoder_mb_data"])  # type: GnnOutput
                mb_outputs = self.__decoder_model.greedy_decode(
                    input_concrete_values=input_concrete_values,
                    input_memories=gnn_output.output_node_representations[
                        gnn_output.node_idx_references["backbone_nodes"]
                    ],
                    input_memories_origin_idx=gnn_output.node_graph_idx_reference["backbone_nodes"],
                    initial_states=trained_network._get_initial_decoder_states(gnn_output),
                    neural_model=trained_network._decoder,
                )
                decoded_sequences.extend(mb_outputs)

        assert len(decoded_sequences) == len(data)
        return decoded_sequences
