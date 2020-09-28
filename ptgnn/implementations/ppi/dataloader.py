import numpy as np
from dpu_utils.utils import RichPath
from typing import Dict, List


class PPIGraphSample:
    """Data structure holding a single PPI graph."""

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        node_features: np.ndarray,
        node_labels: np.ndarray,
    ):
        self._adjacency_lists = adjacency_lists
        self._node_features = node_features
        self._node_labels = node_labels

    @property
    def node_labels(self) -> np.ndarray:
        """Node labels to predict as ndarray of shape [V, C]"""
        return self._node_labels

    @property
    def adjacency_lists(self) -> List[np.ndarray]:
        """Adjacency information by edge type as list of ndarrays of shape [E, 2]"""
        return self._adjacency_lists

    @property
    def node_features(self) -> np.ndarray:
        """Initial node features as ndarray of shape [V, ...]"""
        return self._node_features


class PPIDatasetLoader:
    @classmethod
    def load_data(cls, data_dir: RichPath, data_fold: str) -> List[PPIGraphSample]:

        print(" Loading PPI %s data from %s." % (data_fold, data_dir))

        graph_json_data = data_dir.join("%s_graph.json" % data_fold).read_by_file_suffix()
        node_to_features = data_dir.join("%s_feats.npy" % data_fold).read_by_file_suffix()
        node_to_labels = data_dir.join("%s_labels.npy" % data_fold).read_by_file_suffix()
        node_to_graph_id = data_dir.join("%s_graph_id.npy" % data_fold).read_by_file_suffix()

        # We read in all the data in two steps:
        #  (1) Read features, labels. Implicitly, this gives us the number of nodes per graph.
        #  (2) Read all edges, and shift them so that each graph starts with node 0.
        fwd_edge_type = 0

        graph_id_to_graph_data: Dict[int, PPIGraphSample] = {}
        graph_id_to_node_offset: Dict[int, int] = {}
        num_total_nodes = node_to_features.shape[0]
        for node_id in range(num_total_nodes):
            graph_id = node_to_graph_id[node_id]
            # In case we are entering a new graph, note its ID, so that we can normalise everything to start at 0
            if graph_id not in graph_id_to_graph_data:
                graph_id_to_graph_data[graph_id] = PPIGraphSample(
                    adjacency_lists=[[]],
                    node_features=[],
                    node_labels=[],
                )
                graph_id_to_node_offset[graph_id] = node_id
            cur_graph_data = graph_id_to_graph_data[graph_id]
            cur_graph_data.node_features.append(node_to_features[node_id])
            cur_graph_data.node_labels.append(node_to_labels[node_id])

        for edge_info in graph_json_data["links"]:
            src_node, tgt_node = edge_info["source"], edge_info["target"]
            # First, shift node IDs so that each graph starts at node 0:
            graph_id = node_to_graph_id[src_node]
            graph_node_offset = graph_id_to_node_offset[graph_id]
            src_node, tgt_node = src_node - graph_node_offset, tgt_node - graph_node_offset

            cur_graph_data = graph_id_to_graph_data[graph_id]
            cur_graph_data.adjacency_lists[fwd_edge_type].append((src_node, tgt_node))

        final_graphs = []
        for graph_data in graph_id_to_graph_data.values():
            # numpy-ize:
            adj_lists = [np.array(graph_data.adjacency_lists[fwd_edge_type], dtype=np.int32)]
            final_graphs.append(
                PPIGraphSample(
                    adjacency_lists=adj_lists,
                    node_features=np.array(graph_data.node_features, dtype=np.float32),
                    node_labels=np.array(graph_data.node_labels, dtype=np.bool),
                )
            )

        return final_graphs
