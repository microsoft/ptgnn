## Coding a GNN Model

In this tutorial, we discuss how to define a GNN-based model with `ptgnn`.
Specifically, we discuss adding the Graph2Class model of [Allamanis _et al._ (2020)](https://arxiv.org/abs/2004.10657).

If you are looking for an introductory tutorial to graph neural networks, you may watch the talk [here](https://www.youtube.com/watch?v=zCEYiCxrL_0).

##### Graph2Class Task
The Graph2Class task is a classification task that classifies a subset of nodes in a graph. Each node of
interest is classified into one of a limited set of potential classes.

Specifically, the Graph2Class model of [Typilus](https://arxiv.org/abs/2004.10657) accepts a graph-based representation of
the code and predicts the type of identifiers in the code. For example, for the sample Python code snippet
```python
x = "some_string" + "!"
```
the type of the identifier `x` should be annotated as `str`. The Graph2Class model
treats this problem as a classification problem, among a limited number of possible type
annotations. Here "type annotations" represent the classes in the classification paradigm.

##### Raw Data
We use the Typilus data, which can be extracted as described [here](https://github.com/typilus/typilus/tree/master/src/data_preparation).
Each sample in the data is a JSON dictionary and has the following format:
```json
{
  "nodes": [list-of-nodes-str],
  "edges": {
    "EDGE_TYPE_NAME_1": {
        "from_node_idx": [to_node_idx1, to_node_idx2, ...],
        ...
    },
    "EDGE_TYPE_NAME_2": {...},
    ...
  },
  "token-sequence": [node-idxs-in-token-sequence],
  "supernodes": {
    "supernode1-node-idx": {
      "name": "the-name-of-the-supernode",
      "annotation": null or str,
      ...
    },
    "supernode2-node-idx": { ... },
    ...
  }
 "filename": "provenance-info-str"
}
```
the `"nodes"` key defines the different nodes, whereas the `"edges"` define
the different edge kinds. The task is to accept this graph as input and for
each supernode in `"supernodes"` to predict the correct type annotation (the `"annotation"`).

### Creating the Graph2Class Neural Model
The high-level architecture of the model we are going to create is shown next
```
+--------------------------------------------------------+
|  Graph2Class                                           |
|  +--------------------------------------------------+  |
|  |  GraphNeuralNetworkModel                         |  |
|  | +------------------+   +~~~~~~~~~~~~~~~~~~~~~~+  |  |
|  | |   NodeEmbedder   |   | Message Passing Def  |  |  |
|  | +------------------+   +~~~~~~~~~~~~~~~~~~~~~~+  |  |
|  +--------------------------------------------------+  |
+--------------------------------------------------------+
```
We are going to create a `Graph2Class` class that encapsulates a graph neural network (a `GraphNeuralNetworkModel`).
The `GraphNeuralNetworkModel` requires a `NodeEmbedder`, _i.e._ a neural model that
can convert the arbitrary data in each node into a single vector representation, the initial node embedding. This is used
as the input of the GNN. The node embeddings are refined and learnt using the `MessagePassingDefinition`, which define the GNN layers.
Finally, we classify using the representation of the "supernode" that the GNN computed.

The resulting code, can be seen [here](../ptgnn/implementations/typilus/graph2class.py). Next we go step-by-step
in the process of creating the model.


##### Define Graph2Class

First we define `Graph2Class` which is an `AbstractNeuralModel`. To assist with
the code completion and allow type checking, we first define the three type parameters of
of our model:
  * The raw input data `TypilusGraph` (_i.e._ a concrete `TRawDatapoint`):
    ```python
    SuperNodeData = TypedDict('SuperNodeData', {
        "name": str,
        "annotation": Optional[str],
    }, total=False)

    TypilusGraph = TypedDict('TypilusGraph', {
        "nodes": List[str],
        "edges": Dict[str, Dict[str, List[int]]],
        "token_sequence": List[int],
        "supernodes": Dict[str, SuperNodeData],
        "filename": str
    })
    ```
    This directly reflects the structure of the Typilus raw data, as discussed above.
  * The tensorized format of the data (`TTensorizedDatapoint`).
    Again, this includes the tensorized graph data, along with the the class
    id of each supernode.
    ```python
    class TensorizedGraph2ClassSample(NamedTuple):
        graph: TensorizedGraphData
        supernode_target_classes: List[int]
    ```
    `supernode_target_classes` contain the target class id for all supernodes
    with a ground-truth annotation.
  * Finally, we can define a skeleton neural module (`TNeuralModule`). We will fill in the code later.
    ```python
    class Graph2ClassModule(ModuleWithMetrics):
        ...
    ```
    Note that `ModuleWithMetrics` is a PyTorch `nn.Module` that allows us to report metrics.
    We will revisit this later.

Finally, we can declare the `Graph2Class` model
```python
class Graph2Class(AbstractNeuralModel[
        TypilusGraph, TensorizedGraph2ClassSample, Graph2ClassModule
    ]):
    ...
```
_i.e._ an `AbstractNeuralModel` that accepts as raw input `TypilusGraph`s,
 the tensorized data's type is `TensorizedGraph2ClassSample` and the
  model controls a `Graph2ClassModule` module.

#### Defining the default hyperparameters and child models
The constructor of the model accepts as input any child models
and its hyperparameters. In this case,
```python
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel[str, Any],
        max_num_classes: int = 100
    ):
        super().__init__()
        self.__gnn_model = gnn_model
        self.max_num_classes = max_num_classes
```
The constructor accepts any `GraphNeuralNetworkModel` child model. This is used internally to represent the graph. We will retrieve the output representations
of this model and use it to perform classification.

#### Converting raw data
First, we need to define a function
that converts `TypilusGraph` into `GraphData`, which is the format that
`GraphNeuralNetworkModel`s accept.
```python
def __convert(self, typilus_graph: TypilusGraph) -> Tuple[GraphData[str], List[str]]:
    edges = {}
    for edge_type, adj_dict in typilus_graph['edges'].items():
        adj_list: List[Tuple[int, int]] = []
        for from_node_idx, to_node_idxs in adj_dict.items():
            from_node_idx = int(from_node_idx)
            adj_list.extend((from_node_idx, to_idx) for to_idx in to_node_idxs)
        edges[edge_type] = np.array(adj_list, dtype=np.int32)

    supernode_idxs_with_ground_truth = []
    supernode_annotations = []
    for supernode_idx, supernode_data in typilus_graph["supernodes"].items():
        if supernode_data["annotation"] is None:
            continue
        supernode_idxs_with_ground_truth.append(int(supernode_idx))
        supernode_annotations.append(supernode_data["annotation"])

    return GraphData[str](
        node_information=typilus_graph["nodes"],
        edges=edges,
        reference_nodes={
            "token-sequence": typilus_graph["token-sequence"],
            "supernodes": supernode_idxs_with_ground_truth
        }
    ), supernode_annotations
```
The code essentially wrangles the data into the appropriate format. The created
`GraphData` additionally contains two kinds of references. The `"token-sequence"`
refers to all nodes that belong to the token sequence and `"supernodes"` contains the indices
of all nodes that we wish to classify. We will need to use the last references to retrieve
the nodes which need to be classified.

#### Defining Metadata Loading
To define how `Graph2Class` handles metadata we need to implement `initialize_metadata()`,
`update_metadata_from()` and `finalize_metadata()`. This can be done using
the following pseudocode
```
metadata = initialize_metadata()
for each training_sample:
    update_metadata_from(training_sample)
metadata = finalize_metadata(metadata)
```
* Metadata initialization:
    ```python
    def initialize_metadata(self) -> None:
        self.__target_class_counter = Counter[str]()
    ```
  This creates a counter which we will to count the observed classes we see in the training data.
* Next we define `update_metadata_from` as
    ```python
    def update_metadata_from(self, datapoint: TypilusGraph) -> None:
        graph_data, target_classes = self.__convert(datapoint)
        self.__gnn_model.update_metadata_from(graph_data)
        self.__target_class_counter.update(target_classes)
    ```
  where `datapoint` is the raw input graphs transformed into `GraphData` which can be
  used by the graph model. We also update the `target_class_counter` with
  the classes we observed in this `datapoint`. Note that we need to explicitly convert the
  raw data into the appropriate structure and pass it to the child GNN model.
* Finally, we define `finalize_metadata`
  ```python
  def finalize_metadata(self) -> None:
      self.__target_vocab = Vocabulary.create_vocabulary(self.__target_class_counter,
                                                         max_size=self.max_num_classes)
      del self.__target_class_counter
  ```
  Here, we create a vocabulary of target classes from the `target_class_counter`
  we kept accumulated in `update_metadata_from`. The `self.__target_class_counter`
  can now be deleted since it is not needed.

#### Build the Neural Module
Once all metadata is known, we are ready to build the neural module (a `nn.Module` object).

First, we can define `Graph2ClassModule` neural module:
```python
class Graph2ClassModule(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork, num_target_classes: int):
        super().__init__()
        self.__gnn = gnn
        self.__node_to_class = nn.Linear(
            in_features=gnn.output_node_state_dim,
            out_features=num_target_classes
        )
        self.__loss = nn.CrossEntropyLoss()

    def _logits(self, graph_mb_data):
        graph_output: GnnOutput = self.__gnn(**graph_mb_data)
        # Gather the output representation of the nodes of interest
        super_node_idxs = graph_output.node_idx_references["supernodes"]
        supernode_representations = graph_output.output_node_representations[super_node_idxs]  # [num_supernodes_in_mb, D]
        return self.__node_to_class(supernode_representations)

    def forward(self, graph_mb_data, target_classes):
        return self.__loss(
            self._logits(graph_mb_data),
            target_classes
        )
```
The module gets the indices of the `"supernodes"` reference nodes, gets the output
representations of those nodes as computed by the GNN and then uses a linear layer
to convert there representations into the classification logits.
The `forward()` returns the classification loss.

We can write the code in the `Graph2Class` model that builds the neural module using
the metadata computed by `finalize_metadata`:
```python
def build_neural_module(self) -> Graph2ClassModule:
    return Graph2ClassModule(
        gnn=self.__gnn_model.build_neural_module(),
        num_target_classes=len(self.__target_vocab)
    )
```


#### Defining Tensorization
Tensorization is the process where we convert the raw data into tensors that can be
fed into our neural module. The `tensorize()` will be called for each sample in our
dataset.
```python
def tensorize(self, datapoint: TypilusGraph) -> Optional[TensorizedGraph2ClassSample]:
    graph_data, target_classes = self.__convert(datapoint)
    graph_tensorized_data = self.__gnn_model.tensorize(graph_data)

    if graph_tensorized_data is None or len(target_classes) == 0:
        # The sample either contained no ground-truth annotations or
        # was rejected for some reason by the GNN model
        return None

    target_class_ids = self.__target_vocab.get_id_or_unk_multiple(target_classes)
    return TensorizedGraph2ClassSample(
        graph=graph_tensorized_data,
        supernode_target_classes=target_class_ids
    )
```
The code converts the raw data point into `GraphData` and asks the GNN Model to
tensorize the graph. Additionally, it converts the target classes (a string representing
the type annotation to be predicted) to the target index in the vocabulary or `Unk`.

Note that we can reject a sample by returning `None`. The GNN model, for example,
may reject a graph that is too large to fit in memory.

#### Defining Minibatching Behavior
A minibatch is created using the following pseudocode:
```
mb_data = initalize_minibatch()
for datapoint in some_samples:
    extend_minibatch_with(tensorized_datapoint, mb_data)
mb_data = finalize_minibatch(mb_data)

# Compute the output of a neural module on the minibatch data
neural_module(**mb_data)
```

The minibatch behavior may be defined as follows:
```python
def initialize_minibatch(self) -> Dict[str, Any]:
    return {
        "graph_mb_data": self.__gnn_model.initialize_minibatch(),
        "target_classes": []
    }

def extend_minibatch_with(
    self, tensorized_datapoint: TensorizedGraph2ClassSample, partial_minibatch: Dict[str, Any]
) -> bool:
    continue_extending = self.__gnn_model.extend_minibatch_with(tensorized_datapoint.graph, partial_minibatch["graph_mb_data"])
    partial_minibatch["target_classes"].extend(tensorized_datapoint.supernode_target_classes)
    return continue_extending

def finalize_minibatch(
    self, accumulated_minibatch_data: Dict[str, Any], device: Any
) -> Dict[str, Any]:
    return {
        "graph_mb_data": self.__gnn_model.finalize_minibatch(accumulated_minibatch_data["graph_mb_data"], device),
        "target_classes": torch.tensor(accumulated_minibatch_data["target_classes"], dtype=torch.int64, device=device)
    }
```
* First `initialize_minibatch` creates a dictionary where we accumulate the
    minibatch data. It explicitly invokes the GNN model and asks it to initialize
    its portion of the minibatch.
* Then, `extend_minibatch_with` accepts a single tensorized datapoint (as returned
    by (`tensorize()`) and extends the `partial_minibatch` with that sample.
    * We unpack the `tensorized_datapoint` and pass the graph-related data to the GNN model
        along with the graph-related partial minibatch.
    * We extend `target_classes` by appending all the target class indices. Note
        that this behavior is different from common minibatching where tensors are
        stacked together using a different "batch" dimension. This is necessary, as
        graphs have different numbers of supernodes.
        > Note that `extend_minibatch_with` should return a Boolean value. If
          for some reason the minibatch cannot be further extended (e.g. it contains
          too many nodes), then `False` should be returned.
* Finally, `finalize_minibatch`, unpacks the GNN-related data and invokes
        `finalize_minibatch` for the child GNN model. It also creates a PyTorch
        `Tensor` for the target classes.
        The keys of the returned dictionary are the names of the arguments in
         the `forward()` of `Graph2ClassModule`.

#### Defining the data loading and training steps
We can now define a concrete neural network model. So far, our `Graph2Class` model
was defined in such a way as it could accept any kind of GNN. For example, in
`ptgnn/implementation/typilus/train.py` we define
```python
def create_graph2class_gnn_model(hidden_state_size: int = 64):
    def create_ggnn_mp_layers(num_edges: int):
        ggnn_mp = GatedMessagePassingLayer(
            state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="sum",
            dropout_rate=0.01,
        )
        r1 = MeanResidualLayer(hidden_state_size)
        r2 = MeanResidualLayer(hidden_state_size)
        return [
            r1.pass_through_dummy_layer(),
            r2.pass_through_dummy_layer(),
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            r1,
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            r2,
        ]

    return Graph2Class(
       gnn_model=GraphNeuralNetworkModel(
           node_representation_model=StrElementRepresentationModel(
               embedding_size=hidden_state_size,
               token_splitting="subtoken",
               subtoken_combination="mean",
               vocabulary_size=10000,
               min_freq_threshol=5,
               dropout_rate=0.1
           ),
           message_passing_layer_creator=create_ggnn_mp_layers,
           max_nodes_per_graph=100000,
           max_graph_edges=500000,
           introduce_backwards_edges=True,
           add_self_edges=True,
           stop_extending_minibatch_after_num_nodes=120000,
       ),
       max_num_classes=100
    )
```
where `create_ggnn_mp_layers` creates the message passing layers that define the GNN.
The `GraphNeuralNetworkModel` accepts a `node_representation_model` that is responsible
to retrieve the initial representations of all the nodes. Here we use the `StrElementRepresentationModel`
which convert string node representations into vectors.

> Up to this point we had not defined a concrete node representation model.
> Indeed, our `Graph2Class` is agnostic on how the nodes are represented and can work
> with any node representation model that accepts the the same raw node data (strings in
> the case of Graph2Class).

Finally, we can code the data loading and invoke the `ModelTrainer` to train this model.
The resulting code can be see [here](../ptgnn/implementations/typilus/train.py).

#### [Optional] Define Module Metrics
A module that implements `ModuleWithMetrics` (such as our `Graph2ClassModule`)
can optionally report metrics during training and testing. For `Graph2ClassModule`
we are interested in computing the classification accuracy of the model.

 To achieve this:
* Implement `__reset_module_metrics` which resets all metrics:
    ```python
    def _reset_module_metrics(self) -> None:
        self.__num_samples = 0
        self.__sum_accuracy = 0
    ```
* Implement `report_metrics`
    ```python
    def report_metrics(self) -> Dict[str, Any]:
        return {
            "Accuracy": self.__sum_accuracy / self.__num_samples
        }
    ```
*  Finally, in `forward()` add a snippet that updates the metrics:
    ```python
    with torch.no_grad():
        self.__sum_accuracy += int((torch.argmax(logits, dim=-1) == target_classes).sum())
        self.__num_samples += int(target_classes.shape[0])
    ```
This will report the accuracy at each training/validation epoch and during evaluation.
The `ModelTrainer` can also accept the name of a metric and use this for measuring
the performance on the validation set.
