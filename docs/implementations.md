## Implemented GNN Tasks

The `ptgnn` library offers model implementations for four sample tasks. This file
describes these tasks and how to run these models. We welcome external contributions for
other tasks.

### PPI
The protein-protein interaction (PPI) task is a graph-labeling task where
all nodes of the graph need to be labelled. To train and test a model, run:

```commandline
python -m ptgnn.implementations.ppi.train DATA_PATH MODEL_FILENAME
```
where the `DATA_PATH` contains the data extracted from the original work
of [Zitnik and Leskovec, 2017](https://arxiv.org/abs/1707.04638) and
`MODEL_FILENAME` is the filename (of form `filename.pkl.gz`) where the trained model will be stored.


### Variable Misuse
The variable misuse task ([Allamanis _et al._, 2018](https://arxiv.org/abs/1711.00740))
is the problem of detecting variable misuse bugs in source code. The task is formulated
as a classification problem for picking the correct node among a few candidates nodes
for a given location in a program (a sort of _fill in the blank_ task). Each candidate node
represents a single variable that could be
placed at a given location in the program. The decision needs to be made by considering the context
(a graph representation of a program) for a given location.

To train and test a model, run
```commandline
python -m ptgnn.implementations.varmisuse.train TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME
```
where the data paths point to the train/validation/test folders and `MODEL_FILENAME` is the
target filename of the trained model.

###### Data
The data used in [Allamanis _et al._, 2018](https://arxiv.org/abs/1711.00740) can download from [here](https://aka.ms/iclr18-prog-graphs-dataset).

The input data format is documented in the `VarMisuseSample` raw data type [here](/ptgnn/implementations/varmisuse/varmisuse.py).


### Graph2Sequence
The goal of Graph2Sequence model is to predict a sequence given an input graph structure.
To achieve this, a GNN processes a graph and a GRU predicts the output sequence
step-by-step. The GRU includes an attention mechanism and a copying mechanism similar
to standard sequence-to-sequence models.
The `ptgnn` implementation is a variation of the GNN->GRU model of
[Fernandes _et. al._, 2019](https://arxiv.org/abs/1811.01824).

```commandline
python -m ptgnn.implementations.graph2seq.trainandtest TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME
```
where the data paths point to the train/validation/test `.jsonl.gz` files
and `MODEL_FILENAME` is the target filename of the trained model.

###### Data
The input data used in [Fernandes _et. al._, 2019](https://arxiv.org/abs/1811.01824) can be generated using [these scripts](https://github.com/CoderPat/structured-neural-summarization/tree/master/parsers).

The input data format is documented in the `CodeGraph2Seq` raw data type [here](/ptgnn/implementations/graph2seq/graph2seq.py).

### Graph2Class (Typilus)
The goal of graph2class is to classify a subset of graph nodes. Each to-be-classified
node represents a symbol (variable, parameter, function) of a Python program and the goal is
to classify each symbol to its type (e.g. `int`, `str`).

To train and evaluate a model, run
```commandline
python -m ptgnn.implementations.typilus.train TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME
```

###### Data
The data used in [Typilus](https://arxiv.org/abs/2004.10657) can be generated following [these steps](https://github.com/typilus/typilus/tree/master/src/data_preparation).
The data generation process will create folders with `.jsonl.gz` files containing the graphs.

The input data format is documented in the `TypilusGraph` raw data type [here](/ptgnn/implementations/typilus/graph2class.py).
