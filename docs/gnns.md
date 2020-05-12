## How to define Graph Neural Networks (GNN)
The `ptgnn` library enables the definition of a broad range of models.
The base classes are implemented in `ptgnn.neuralmodels.gnn`. As graphs tends to have irregular shape, batching the GNN computation
across multiple graphs requires flattening down multiple graphs into
a single graph with multiple disconnected components. We provide a class to handle this transparently via `GraphNeuralNetworkModel`.

To create a graph neural network `GraphNeuralNetworkModel`, we first
define the _node embedding model_. This is a model that accepts the initial
data of each node (of type `TNodeData`) embeds them into a vector.
This can be an arbitrary model as long
as it can accept the raw node data `TNodeData`. For example,
`ptgnn.neuralmodels.embeddings.StrElementRepresentationModel` accepts `str`
as the initial node data (_i.e._ `TNodeData` is a `str`).

###### Inputs
A `GraphNeuralNetworkModel` accepts as input an instance of `ptgnn.neuralmodels.gnn.GraphData`. This is a named tuple containing the following fields:
* `node_information`: A list of `TNodeData` with the initial data of each node.
    This data should be exactly the format that the node embedding model accepts
    as input.
* `edges`: A dictionary `Dict[str, List[Tuple[int, int]]]`, which contains an edge for each node type in the adjecency list.
* `reference_nodes`: (optional) A dictionary containing references (indices) to nodes of
    interest. For example, to keep track of nodes #5 and #10, `reference_nodes` would
    be `{"ref_name": [5,10]}`.

Once a tensorized format has been defined, `GraphNeuralNetworkModel`
can transparently batch multiple graphs.

###### Defining the Graph Neural Network
Finally, to create a message-passing GNN, one or more message passing layers
need to be defined. The function that creates the list of message passing nodes
is passed directly to the `GraphNeuralNetworkModel` constructor as the
`message_passing_layer_creator` argument.

For illustration purposes, consider the following message-passing layer
definition, _i.e._ the `message_passing_layer_creator=create_sample_gnn_model`
argument of the constructor.
Note that this architecture is _not_ practically useful, but serves as an explanatory example:
```python
def create_sample_gnn_model(hidden_state_size: int = 64):
    def create_mlp_mp_layers(num_edge_types: int):
        mlp_mp_constructor = lambda: MlpMessagePassingLayer(
            input_state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            output_state_dimension=hidden_state_size,
            num_edge_types=num_edge_types,
            message_aggregation_function="sum",
            dropout_rate=0.1,
        )
        ggnn_mp = GatedMessagePassingLayer(
            state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            num_edge_types=num_edge_types,
            message_aggregation_function="sum",
            dropout_rate=0.1,
        )
        r1 = MeanResidualLayer(hidden_state_size)
        global_update = lambda: GruGlobalStateUpdate(
            global_graph_representation_module=SimpleVarSizedElementReduce("max"),
            input_state_size=hidden_state_size,
            summarized_state_size=hidden_state_size,
            dropout_rate=0.1,
        )
        return [
            r1.pass_through_dummy_layer(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            global_update(),
            mlp_mp_constructor(),
            r1,
            global_update(),
            ggnn_mp,
            ggnn_mp,
        ]
```
This creates the following message passing layers
```
   Graph Input
        +-------------------+
        |                   |
        v                   |
+-------+--------+          |
|  GCN-Layer 1   |          |
+-------+--------+          |
        |                   |
        v                   |
+-------+--------+          |
|  GCN-Layer 2   |          |
+-------+--------+          |
        |                   |
        v                   |
+-------+--------+          |
|Global Update 1 |          |
+-------+--------+          |
        |                   |
        v                   |
+-------+--------+          |
|  GCN-Layer 3   |          |
+-------+--------+          |
        |                   |
+-------v--------+          |
| Mean Residual  +<---------+
+-------+--------+
        |
        v
+-------+--------+
|Global Update 2 |
+-------+--------+
        |
        v
+-------+--------+
|  GGNN Layer 1  |
+-------+--------+
        |
        v
+-------+--------+
|  GGNN Layer 1  |
+----------------+
        |
        v
    GNN Output
```
 where the GGNN layers are coupled (_i.e._ have the same parameters).

###### Using the output of the Graph Neural Network
The output of the `GraphNeuralNetwork` module is a named tuple with the
following fields:
* `input_node_representations` a tensor of size `[num_nodes_in_batch, D]`
    with the input representations of each of the nodes in the minibatch.
    Note that this will contain nodes from multiple graphs.
* `output_node_representations` a tensor of size `[num_nodes_in_batch, D']`
    with the output representations of each of the nodes in the minibatch.
    Note that this will contain nodes from multiple graphs.
* `node_to_graph_idx` a vector of size `[num_nodes_in_batch]` containing
    the index of the graph that the respective node belongs in. This allows
    to trace the origin graph of each node.
* `node_idx_references` a dictionary that contains for each named
    reference in the `reference_nodes` of the `ptgnn.neuralmodels.gnn.GraphData`,
    the indices pointing to the original referenced nodes accounting for
    mini-batching.
* `node_graph_idx_reference` a dictionary that contains the origin graph
    indices for each of the nodes in `node_idx_references`. This is similar
    in nature to `node_to_graph_idx` but contains the origin graph indices
    only for the referenced indices.
* `num_graphs` the number of graphs in the current minibatch.

For example, to retrieve the output node representation of the nodes
referenced in `"some_ref_name"`,
```python
ref_node_idxs = output.node_idx_references["some_ref_name"]
some_ref_node_reps = output.output_node_representations[ref_node_idxs]

ref_node_to_graph = output.node_graph_idx_reference["some_ref_name"]
```
Then, one can pool these representation per-graph (_e.g._ max pooling):
```python
per_graph_reps = scatter_max(
    src=some_ref_node_reps,
    index=ref_node_to_graph
)  # A [num_graphs, D'] tensor, containing one vector per graph.
```
