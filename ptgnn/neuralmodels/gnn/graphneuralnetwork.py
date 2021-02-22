from typing_extensions import Final

import logging
import numpy as np
import torch
from collections import defaultdict
from torch import nn
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.baseneuralmodel.utils.data import enforce_not_None
from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import AbstractMessagePassingLayer
from ptgnn.neuralmodels.gnn.structs import GnnOutput, GraphData, TensorizedGraphData


class GraphNeuralNetwork(ModuleWithMetrics):
    """
    A generic message-passing graph neural network with discrete edge types.
    """

    def __init__(
        self,
        message_passing_layers: List[AbstractMessagePassingLayer],
        node_embedder: nn.Module,
        introduce_backwards_edges: bool,
        add_self_edges: bool,
        edge_dropout_rate: float = 0.0,
        edge_feature_embedder: Optional[nn.Module] = None,
    ):
        """
        :param message_passing_layers: A list of message passing layers.
        :param node_embedder: a `nn.Module` that converts node data into a vector representation
        :param introduce_backwards_edges: If `True` special backwards edges should be automatically created.
        :param add_self_edges: If `True` self-edges will be added. These edges connect the same node across
            multiple timesteps.
        :param edge_dropout_rate: remove random pct of edges
        """
        super().__init__()
        self.__message_passing_layers = nn.ModuleList(message_passing_layers)
        self.__node_embedder = node_embedder
        self.__introduce_backwards_edges = introduce_backwards_edges
        self.__add_self_edges = add_self_edges
        assert 0 <= edge_dropout_rate < 1
        self.__edge_dropout_rate = edge_dropout_rate
        self.__edge_feature_embedder = edge_feature_embedder

    @property
    def input_node_state_dim(self) -> int:
        """The dimension of the input GNN node states."""
        return self.__message_passing_layers[0].input_state_dimension

    @property
    def output_node_state_dim(self) -> int:
        """The dimension of the output GNN node states."""
        return self.__message_passing_layers[-1].output_state_dimension

    @property
    def message_passing_layers(self) -> List[AbstractMessagePassingLayer]:
        return self.__message_passing_layers

    def _reset_module_metrics(self) -> None:
        self.__num_graphs, self.__num_edges, self.__num_nodes = 0, 0, 0

    def _module_metrics(self) -> Dict[str, Any]:
        return {
            "num_graphs": self.__num_graphs,
            "num_nodes": self.__num_nodes,
            "num_edges": self.__num_edges,
        }

    def gnn(
        self,
        node_representations: torch.Tensor,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        edge_feature_embeddings: List[torch.Tensor],
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        return_all_states: bool = False,
    ) -> torch.Tensor:
        """
        :param node_representations: A [num_nodes, hidden_dimension] matrix of node representations.
        :param adjacency_lists: a list of [num_edges_per_type, 2] adjacency lists per edge type.
                The order is fixed across runs. Backwards edges and self-edges are included if
                the appropriate hyperparameter is set.
        :param edge_feature_embeddings: a list of the edge features per edge-type.
        :param node_to_graph_idx: A mapping that tells us which graph the node belongs to
        :param reference_node_ids: A dictionary indicating the reference node index
        :param reference_node_graph_idx: A dictionary indicating the graph index for reference node
        :param return_all_states: Whether to return all states
        :return: a [num_nodes, output_hidden_dimension] matrix of the output representations
        """
        if self.__edge_dropout_rate > 0 and self.training:
            dropped_adj_list, dropped_edge_features = [], []
            for (edge_sources_idxs, edge_target_idxs), edge_features in zip(
                adjacency_lists, edge_feature_embeddings
            ):
                mask = (
                    torch.rand_like(edge_sources_idxs, dtype=torch.float32)
                    > self.__edge_dropout_rate
                )
                dropped_adj_list.append(
                    (edge_sources_idxs.masked_select(mask), edge_target_idxs.masked_select(mask))
                )
                dropped_edge_features.append(edge_features[mask])
            adjacency_lists = dropped_adj_list
            edge_feature_embeddings = dropped_edge_features

        all_states = [node_representations]
        for mp_layer_idx, mp_layer in enumerate(self.__message_passing_layers):
            node_representations = mp_layer(
                node_states=node_representations,
                adjacency_lists=adjacency_lists,
                node_to_graph_idx=node_to_graph_idx,
                reference_node_ids=reference_node_ids,
                reference_node_graph_idx=reference_node_graph_idx,
                edge_features=edge_feature_embeddings,
            )
            all_states.append(node_representations)
        if return_all_states:
            node_representations = torch.cat(all_states, dim=-1)
        return node_representations

    def forward(
        self,
        *,
        node_data,
        adjacency_lists: List[Tuple[torch.Tensor, torch.Tensor]],
        edge_feature_data: List,
        node_to_graph_idx: torch.Tensor,
        reference_node_ids: Dict[str, torch.Tensor],
        reference_node_graph_idx: Dict[str, torch.Tensor],
        num_graphs,
        **kwargs
    ) -> GnnOutput:
        """

        :param node_data: The data for the node embedder to compute the initial node representations.
        :param adjacency_lists: A list of [num_edges, 2] matrices for each edge type.
        :param edge_feature_data: A list of the same size as `adjacency_lists` with the data
            to compute the edge features.
        :param node_to_graph_idx: A [num_nodes] vector that contains the index of the graph it belongs to.
        :param reference_node_ids: A dictionary with values the indices of the reference nodes.
        :param reference_node_graph_idx: A dictionary with values the index of the graph that each
            node in the `reference_node_ids` belongs to.
        :param num_graphs: the number of graphs.
        """
        initial_node_representations = self.__node_embedder(**node_data)  # [num_nodes, D]

        if self.__edge_feature_embedder is None:
            edge_feature_embeddings = [
                torch.empty(f.shape[0], 0, device=node_to_graph_idx.device)
                for f, _ in adjacency_lists
            ]
        else:
            edge_feature_embeddings = [
                self.__edge_feature_embedder(**edge_data) for edge_data in edge_feature_data
            ]

        if self.__introduce_backwards_edges:
            adjacency_lists += [(t, f) for f, t in adjacency_lists]
            edge_feature_embeddings += [e for e in edge_feature_embeddings]

        if self.__add_self_edges:
            num_nodes = node_to_graph_idx.shape[0]
            idents = torch.arange(num_nodes, dtype=torch.int64, device=node_to_graph_idx.device)
            adjacency_lists.append((idents, idents))
            edge_feature_embeddings.append(
                torch.zeros(
                    num_nodes,
                    edge_feature_embeddings[-1].shape[-1],
                    device=node_to_graph_idx.device,
                )
            )

        output_representations = self.gnn(
            initial_node_representations,
            adjacency_lists,
            edge_feature_embeddings,
            node_to_graph_idx,
            reference_node_ids,
            reference_node_graph_idx,
            **kwargs
        )  # [num_nodes, H]

        with torch.no_grad():
            self.__num_edges += int(sum(adj[0].shape[0] for adj in adjacency_lists))
            self.__num_graphs += int(num_graphs)
            self.__num_nodes += int(node_to_graph_idx.shape[0])
        return GnnOutput(
            input_node_representations=initial_node_representations,
            output_node_representations=output_representations,
            node_to_graph_idx=node_to_graph_idx,
            node_idx_references=reference_node_ids,
            node_graph_idx_reference=reference_node_graph_idx,
            num_graphs=num_graphs,
        )


TNodeData = TypeVar("TNodeData")
TEdgeData = TypeVar("TEdgeData")
TTensorizedNodeData = TypeVar("TTensorizedNodeData")
TTensorizedEdgeData = TypeVar("TTensorizedEdgeData")


class GraphNeuralNetworkModel(
    AbstractNeuralModel[
        GraphData[TNodeData, TEdgeData],
        TensorizedGraphData[TTensorizedNodeData, TTensorizedEdgeData],
        GraphNeuralNetwork,
    ],
):
    LOGGER: Final = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        node_representation_model: AbstractNeuralModel[TNodeData, TTensorizedNodeData, nn.Module],
        message_passing_layer_creator: Callable[[int], List[AbstractMessagePassingLayer]],
        max_nodes_per_graph: int = 80000,
        max_graph_edges: int = 100000,
        introduce_backwards_edges: bool = True,
        stop_extending_minibatch_after_num_nodes: int = 10000,
        add_self_edges: bool = False,
        edge_dropout_rate: float = 0.0,
        edge_representation_model: Optional[
            AbstractNeuralModel[TEdgeData, TTensorizedEdgeData, nn.Module]
        ] = None
    ):
        """
        :param node_representation_model: A model that can convert the data of each node into their
            tensor representation and create the neural network that computes the node representations.
        :param message_passing_layer_creator: A function that accepts the number of edge types and creates
            a list of `AbstractMessagePassingLayer` layers to be used by the GNN.
        """
        super().__init__()
        self.__message_passing_layers_creator: Final = message_passing_layer_creator
        self.__node_embedding_model: Final = node_representation_model
        self.__edge_embedding_model: Final = edge_representation_model
        self.max_nodes_per_graph: Final = max_nodes_per_graph
        self.max_graph_edges: Final = max_graph_edges
        self.introduce_backwards_edges: Final = introduce_backwards_edges
        self.stop_extending_minibatch_after_num_nodes: Final = (
            stop_extending_minibatch_after_num_nodes
        )
        self.add_self_edges: Final = add_self_edges
        self.__edge_dropout_rate = edge_dropout_rate

    # region Metadata Loading
    def initialize_metadata(self) -> None:
        self.__edge_types_mdata: Set[str] = set()

    def update_metadata_from(self, datapoint: GraphData[TNodeData, TEdgeData]) -> None:
        for node in datapoint.node_information:
            self.__node_embedding_model.update_metadata_from(node)

        for edge_type in datapoint.edges:
            self.__edge_types_mdata.add(edge_type)

        if datapoint.edge_features is not None and self.__edge_embedding_model is not None:
            for edge_features in datapoint.edge_features.values():
                for edge_feature in edge_features:
                    self.__edge_embedding_model.update_metadata_from(edge_feature)

    def finalize_metadata(self) -> None:
        self.LOGGER.info("Found %s edge types in data.", len(self.__edge_types_mdata))
        self.__edge_idx_to_type = tuple(self.__edge_types_mdata)
        self.__edge_types = {e: i for i, e in enumerate(self.__edge_idx_to_type)}
        del self.__edge_types_mdata

    @property
    def _num_edge_types(self) -> int:
        num_types = len(self.__edge_types)
        if self.introduce_backwards_edges:
            num_types *= 2
        if self.add_self_edges:
            num_types += 1
        return num_types

    def build_neural_module(self) -> GraphNeuralNetwork:
        if self.__edge_embedding_model is None:
            edge_feature_embedder = None
        else:
            edge_feature_embedder = self.__edge_embedding_model.build_neural_module()

        gnn = GraphNeuralNetwork(
            self.__message_passing_layers_creator(self._num_edge_types),
            node_embedder=self.__node_embedding_model.build_neural_module(),
            introduce_backwards_edges=self.introduce_backwards_edges,
            add_self_edges=self.add_self_edges,
            edge_dropout_rate=self.__edge_dropout_rate,
            edge_feature_embedder=edge_feature_embedder,
        )
        del self.__message_passing_layers_creator
        return gnn

    # endregion

    def edge_idx_by_name(self, name: str) -> int:
        return self.__edge_types[name]

    def __iterate_edge_types(
        self, data_to_load: GraphData[TNodeData, TEdgeData]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for edge_type in self.__edge_idx_to_type:
            adjacency_list = data_to_load.edges.get(edge_type)
            if adjacency_list is not None and len(adjacency_list) > 0:
                adj = np.array(adjacency_list, dtype=np.int32)
                yield adj[:, 0], adj[:, 1]
            else:
                yield np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    def tensorize(
        self, datapoint: GraphData[TNodeData, TEdgeData]
    ) -> Optional[TensorizedGraphData[TTensorizedNodeData, TTensorizedEdgeData]]:
        if len(datapoint.node_information) > self.max_nodes_per_graph:
            self.LOGGER.warning("Dropping graph with %s nodes." % len(datapoint.node_information))
            return None

        if self.__edge_embedding_model is None:
            tensorized_edge_features = None
        else:
            tensorized_edge_features = []
            for edge_type in self.__edge_idx_to_type:
                edge_features_for_edge_type = datapoint.edge_features.get(edge_type)
                if edge_features_for_edge_type is None:
                    # No edges of type `edge_type`
                    tensorized_edge_features.append([])
                else:
                    tensorized_edge_features.append(
                        [
                            self.__edge_embedding_model.tensorize(e)
                            for e in edge_features_for_edge_type
                        ]
                    )

        tensorized_data = TensorizedGraphData(
            adjacency_lists=list(self.__iterate_edge_types(datapoint)),
            node_tensorized_data=[
                enforce_not_None(self.__node_embedding_model.tensorize(ni))
                for ni in datapoint.node_information
            ],
            edge_features=tensorized_edge_features,
            reference_nodes={
                n: np.array(np.array(refs, dtype=np.int32))
                for n, refs in datapoint.reference_nodes.items()
            },
            num_nodes=len(datapoint.node_information),
        )

        num_edges = sum(len(adj) for adj in tensorized_data.adjacency_lists)
        if num_edges > self.max_graph_edges:
            self.LOGGER.warning("Dropping graph with %s edges." % num_edges)
            return None

        return tensorized_data

    # region Minibatching
    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "node_data_mb": self.__node_embedding_model.initialize_minibatch(),
            "adjacency_lists": [([], []) for _ in range(len(self.__edge_types))],
            "edge_feature_data": [
                self.__edge_embedding_model.initialize_minibatch()
                if self.__edge_embedding_model is not None
                else None
                for _ in range(len(self.__edge_types))
            ],
            "num_nodes_per_graph": [],
            "reference_node_graph_idx": defaultdict(list),
            "reference_node_ids": defaultdict(list),
            "num_nodes_in_mb": 0,
        }

    def extend_minibatch_with(
        self,
        tensorized_datapoint: TensorizedGraphData[TTensorizedNodeData, TTensorizedEdgeData],
        partial_minibatch: Dict[str, Any],
    ) -> bool:
        continue_extending = True
        for node_tensorized_info in tensorized_datapoint.node_tensorized_data:
            continue_extending &= self.__node_embedding_model.extend_minibatch_with(
                node_tensorized_info, partial_minibatch["node_data_mb"]
            )

        graph_idx = len(partial_minibatch["num_nodes_per_graph"])

        adj_list = partial_minibatch["adjacency_lists"]
        tensorized_edge_feature_data = partial_minibatch["edge_feature_data"]
        nodes_in_mb_so_far = partial_minibatch["num_nodes_in_mb"]

        datapoint_edge_features = tensorized_datapoint.edge_features
        if datapoint_edge_features is None:
            datapoint_edge_features = [None for _ in range(len(adj_list))]

        for (
            sample_adj_list_for_edge_type,
            edge_features,
            mb_adj_lists_for_edge_type,
            mb_edge_feature_data,
        ) in zip(
            tensorized_datapoint.adjacency_lists,
            datapoint_edge_features,
            adj_list,
            tensorized_edge_feature_data,
        ):
            mb_adj_lists_for_edge_type[0].append(
                sample_adj_list_for_edge_type[0] + nodes_in_mb_so_far
            )
            mb_adj_lists_for_edge_type[1].append(
                sample_adj_list_for_edge_type[1] + nodes_in_mb_so_far
            )
            if self.__edge_embedding_model is not None:
                for edge_feature in edge_features:
                    self.__edge_embedding_model.extend_minibatch_with(
                        edge_feature, mb_edge_feature_data
                    )

        for ref_name, ref_nodes in tensorized_datapoint.reference_nodes.items():
            partial_minibatch["reference_node_graph_idx"][ref_name].extend(
                graph_idx for _ in range(len(ref_nodes))
            )
            partial_minibatch["reference_node_ids"][ref_name].append(ref_nodes + nodes_in_mb_so_far)

        partial_minibatch["num_nodes_per_graph"].append(tensorized_datapoint.num_nodes)
        partial_minibatch["num_nodes_in_mb"] = nodes_in_mb_so_far + tensorized_datapoint.num_nodes
        return partial_minibatch["num_nodes_in_mb"] < self.stop_extending_minibatch_after_num_nodes

    @staticmethod
    def __create_node_to_graph_idx(num_nodes_per_graph: List[int]) -> Iterable[int]:
        for i, graph_size in enumerate(num_nodes_per_graph):
            yield from (i for _ in range(graph_size))

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:

        if self.__edge_embedding_model is None:
            edge_feature_data = [None for _ in accumulated_minibatch_data["edge_feature_data"]]
        else:
            edge_feature_data = [
                self.__edge_embedding_model.finalize_minibatch(edge_features_for_type, device)
                for edge_features_for_type in accumulated_minibatch_data["edge_feature_data"]
            ]

        return {
            "node_data": self.__node_embedding_model.finalize_minibatch(
                accumulated_minibatch_data["node_data_mb"], device
            ),
            "adjacency_lists": [
                (
                    torch.tensor(np.concatenate(adjFrom), dtype=torch.int64, device=device),
                    torch.tensor(np.concatenate(adjTo), dtype=torch.int64, device=device),
                )
                for adjFrom, adjTo in accumulated_minibatch_data["adjacency_lists"]
            ],
            "edge_feature_data": edge_feature_data,
            "node_to_graph_idx": torch.tensor(
                list(
                    self.__create_node_to_graph_idx(
                        accumulated_minibatch_data["num_nodes_per_graph"]
                    )
                ),
                dtype=torch.int64,
                device=device,
            ),
            "reference_node_graph_idx": {
                ref_name: torch.tensor(ref_node_graph_idx, dtype=torch.int64, device=device)
                for ref_name, ref_node_graph_idx in accumulated_minibatch_data[
                    "reference_node_graph_idx"
                ].items()
            },
            "reference_node_ids": {
                ref_name: torch.tensor(
                    np.concatenate(ref_node_idxs).astype(np.int32), dtype=torch.int64, device=device
                )
                for ref_name, ref_node_idxs in accumulated_minibatch_data[
                    "reference_node_ids"
                ].items()
            },
            "num_graphs": len(accumulated_minibatch_data["num_nodes_per_graph"]),
        }

    # endregion
