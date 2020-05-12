## Neural Model Architecture

The `AbstractNeuralModel` class provides the structure necessary
to define the neural network models in a composable
manner by encapsulating all operations necessary for ingesting the raw
data, transforming it into tensors and defining the neural network operations
(PyTorch's `nn.Module`s).


##### Terms
The following terms are used throughout the library:
* **Metadata:** All information about a model that needs to be computed from the
    (training) data. For example, this may include the vocabulary of tokens
    that a model can represent or the number of edge types a GNN needs to
    represent.
* **Tensorization:** The process of converting raw input data into the appropriate
    tensor format to be used by a neural network. For example, representing a sentence
    into a sequence of (sub)word ids.
* **Neural Module:** The definition of the tensor operations that
    accept as input, the tensorized data and outputs the appropriate predictions,
    losses _etc._. These are subclasses of PyTorch's `nn.Module` class.
    A module can be composed by other modules. Note that the neural
    modules can be used independently of the rest of the library should one wish to do so.
* **Neural Model:** A neural model (_i.e._ a class that
    subclasses `AbstractNeuralModel`) contains the
    logic for accepting raw input data, passing it to the neural module and
    generating the output of the model. Neural models can be thought of as the
    controllers or adapters of neural modules that interface between the target domain and
    the neural network.

##### Why `AbstractNeuralModel`?
The `nn.Module` class from PyTorch allows the definition of neural
operations in a set of composable operations (_e.g._ by composing `nn.Module`s),
but does _not_ concern itself with how the data is transformed to/from
a format that is appropriate for the neural models (commonly tensors).
However, decisions about transforming raw data into tensors are _highly coupled_
with the implementation of the `nn.Module`. If these two aspects are treated
independently it commonly leads to tangled code, that cannot be reused. We address this shortcoming using an `AbstractNeuralModel` class.

The `AbstractNeuralModel` class defines a structure for defining composable
models. This is achieved by encapsulating operations for model building,
data transformations, _etc_.

###### Architecture

The base classes can be found in the [`ptgnn.baseneuralmodel`](/ptgnn/baseneuralmodel/) package.
A high-level overview of the `AbstractNeuralModel` architecture can be seen below:
```
+----------------------------------------------------------------------------------------------------------------------------+
|                                               +-----------------------------------------------------------------------+    |
|      AbstractNeuralModel                      |                                                                       |    |
|                                               | Children                                                              |    |
|  * Compute model metadata and create an       |                                                                       |    |
|    nn.Module by invoking related methods      |                                                                       |    |
|    in child models.                           |    +-------------------------------------------------------------+    |    |
|                                               |    |                                                             |    |    |
|                                               |    |                                                             |    |    |
|     initialize_metadata()       <--------------------->  initialize_metadata() +----------+                      |    |    |
|                                               |    |                                      |                      |    |    |
|     For each training sample                  |    |                                      |                      |    |    |
|                                               |    |                                      |      Children        |    |    |
|        update_metadata_from()   <--------------------->  update_metadata_from() +------+  |  +---------------+   |    |    |
|                                               |    |                                   |  |  |               |   |    |    |
|     finalize_metadata()         <--------------------->  finalize_metadata()    +----+ |  +----->            |   |    |    |
|                                               |    |                                 | |     |               |   |    |    |
|     build_neural_module()       <--------------------->  build_neural_module()  +    | +-------->            |   |    |    |
|                                               |    |                            |    |       |               |   |    |    |
|                                               |    |                            |    +---------->            |   |    |    |
|  * Convert a single input datapoint           |    |                            |            |               |   |    |    |
|    to a tensorized format.                    |    |                            +--------------->            |   |    |    |
|                                               |    |                                         |               |   |    |    |
|     For each sample:                          |    |                                         |               |   |    |    |
|         tensorize()             <--------------------->  tensorize()        +------------------->            |   |    |    |
|                                               |    |                                         |               |   |    |    |
|                                               |    |                                         |               |   |    |    |
|  * Create minibatches by combining            |    |                                         |               |   |    |    |
|    multiple tensorized datapoints.            |    |                                         |               |   |    |    |
|                                               |    |                                         |               |   |    |    |
|                                               |    |                                         |               |   |    |    |
|     initialize_minibatch()     <---------------------->  initialize_minibatch()  +------------->             |   |    |    |
|                                               |    |                                         |               |   |    |    |
|                                               |    |                                         |               |   |    |    |
|     For each minibatch sample:                |    |                                         |               |   |    |    |
|        extend_minibatch_with() <---------------------->  extend_minibatch_with()  +------------>             |   |    |    |
|                                               |    |                                         |               |   |    |    |
|     finalize_minibatch()       <---------------------->  finalize_minibatch()     +------------>             |   |    |    |
|                                               |    |                                         |               |   |    |    |
|                                               |    |                                         +---------------+   |    |    |
|                                               |    |                                                             |    |    |
|                                               |    |                                                             |    |    |
|                                               |    +-------------------------------------------------------------+    |    |
|                                               |                                                                       |    |
|                                               +-----------------------------------------------------------------------+    |
|                                                                                                                            |
+----------------------------------------------------------------------------------------------------------------------------+

```
An `AbstractNeuralModel` is a neural network model that has zero or more children of type
`AbstractNeuralModel` which in turn may have zero or more children, _etc._
Each concrete implementation needs to first define three types:
* `TRawDatapoint`: the format of the raw input data,
* `TTensorizedDatapoint`: the format of the tensorized input, and
* `TNeuralModule`: the neural network module (_e.g._ a `nn.Module`) that defines
    the neural operations.

The following methods also need to be implemented. Most of them involve invoking
the relevant function for all child modules and appropriately composing
the results.
* `initialize_metadata()` initializes any data structures necessary for computing
    the model metadata. These data structures are stored as fields within the model.
* `update_metadata_from(datapoint)` accepts a single raw datapoint `TRawDatapoint`
    and updates the metadata with the information received from the new datapoint.
    This is commonly invoked once for each datapoint in the training set.

    The model is also
    responsible to appropriately unpack the `TRawDatapoint` and appropriately
    invoke `update_metadata_from` of each of its child models, with the appropriate input.
* `finalize_metadata()` once all necessary information to compute the model metadata
    has been processed, the metadata can be finalized. All finalized metadata should
    be stored within the `AbstractNeuralModel`.
    For example, if a model accepts words, the vocabulary of words
    that can be represented, is defined at this stage.
* `build_neural_module()` Once the metadata is finalized, a neural model `TNeuralModule` can
    be built. This returns a `nn.Module`. Again, the
    model should invoke `build_neural_module()` for all its child models and use
    the output modules to build and return a single `nn.Module`.
* `tensorize(datapoint)` accepts a single `TRawDatapoint` and converts it into a
    `TTensorizedDatapoint`. This commonly requires one to unpack `TRawDatapoint` and
    pass the appropriate data to the child models' `tensorize()` along with
    computing any additional tensors/data that will be sent to the neural module.
* `initialize_minibatch()` creates an empty minibatch structure that will
    gradually accumulate multiple `TTensorizedDatapoint`s. Again, each model should
    invoke `initialize_minibatch()` for all its child models and store their
    results appropriately.
* `extend_minibatch_with(datapoint, partial_minibatch)` accepts a partial minibatch
    (as defined by `initialize_minibatch()`) and extends it with one tensorized
    datapoint (`TTensorizedDatapoint`). This requires one to unpack `TTensorizedDatapoint`
    and invoke `extend_minibatch_with()` for all child models.

    If the minibatch should _not_ be extended further, then this function
    should return `False`.
* `finalize_minibatch(partial_minibatch)` accepts a minibatch and finalizes it by
    performing any necessary operations (_e.g._ concatenation or stacking of tensors)
    along with invoking `finalize_minibatch` for all child models. Finally, it returns
    a dictionary with the arguments passed to `TNeuralModule`'s `forward()`.

Please refer to the [docstring](/ptgnn/baseneuralmodel/abstractneuralmodel.py) for `AbstractNeuralModel`
    for more information. Conceptually, the following pseudocode shows the order of the
    functions above:
```python
# Compute Metadata
initialize_metadata()
for raw_sample in training_data:
    update_metadata(raw_sample)
finalize_metadata()

# Build Neural Module
neural_module = build_neural_module()

# Tensorize Data
tensorized_data = (tensorize(d) for d in training_data)

# Compute forward() on Minibatches
while still_have_tensorized_data:
    mb_data = initialize_minibatch()
    while size-of(mb_data) < max_minibatch_size:
        continue_extending = extend_minibatch_with(next(tensorized_data), mb_data)
        if not continue_extending:
            break
    minibatch = finalize_minibatch(mb_data)
    yield neural_module(**minibatch)
```
