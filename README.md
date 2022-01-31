# ptgnn: A PyTorch GNN Library ![PyPI](https://img.shields.io/pypi/v/ptgnn?style=flat-square) ![conda](https://img.shields.io/conda/vn/conda-forge/ptgnn)

This is a library containing pyTorch code for creating graph neural network (GNN) models.
The library provides some sample implementations.

If you are interested in using this library, please read about its [architecture](docs/architecture.md)
and [how to define GNN models](docs/gnns.md) or follow [this tutorial](docs/tutorial.md).

Note that `ptgnn` takes care of defining the whole pipeline, including data wrangling tasks, such
as data loading and tensorization. It also defines PyTorch `nn.Module`s for
the neural network operations. These are independent of the
`AbstractNeuralModel`s and can be used as all other PyTorch's `nn.Module`s,
if one wishes to do so.

The library is mainly engineered to be fast for sparse graphs. For example, for the
Graph2Class task (discussed below) on a V100 with the default hyperparameters and architecture
`ptgnn` can process about 82 graphs/sec (209k nodes/sec and 1,129k edges/sec) during training
and about 200 graph/sec (470k nodes/sec and 2,527k edges/sec) during testing.

#### Implemented Tasks
All task implementations can be found in the [`ptgnn.implementations`](ptgnn/implementations) package.
Detailed instructions on the data and the training steps can be found [here](/docs/implementations.md).
We welcome external contributions. The following GNN-based tasks are implemented:

* **PPI** The PPI task as described by
[Zitnik and Leskovec, 2017](https://arxiv.org/abs/1707.04638).
* **VarMisuse** This is a re-implementation of the VariableMisuse task of
    [Allamanis _et al._, 2018](https://arxiv.org/abs/1711.00740).
* **Graph2Sequence** This is re-implementation of the GNN->GRU model of
                     [Fernandes _et. al._, 2019](https://arxiv.org/abs/1811.01824).
* **Graph2Class** Classify (Label) a subset of the input nodes into classes
                    similar to Graph2Class in [Typilus](https://arxiv.org/abs/2004.10657).


The [tutorial](docs/tutorial.md) gives a step-by-step example for coding the Graph2Class model.


### Installation

This code was tested with PyTorch 1.4 and depends
on [`pytorch-scatter`](https://github.com/rusty1s/pytorch_scatter).
Please install the appropriate versions of these libraries based
on your CUDA setup following their instructions. (Note
that the `pytorch-scatter` binaries built for CUDA 10.1 also work for
CUDA 10.2).

1. To install PyTorch 1.7.0 or higher, use the up-to-date command from [PyTorch Get Started](https://pytorch.org/get-started/), selecting the appropriate options, e.g. for Linux, pip, and CUDA 10.1 it's currently:
    ```bash
    pip install torch torchvision
    ```

1. To install `pytorch-scatter`, follow the instructions from the [GitHub repo](https://github.com/rusty1s/pytorch_scatter), choosing the appropriate CUDA option, _e.g._, for PyTorch 1.7.0 and CUDA 10.1:
    ```bash
    pip install torch-scatter==2.0.5+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
    ```

1. To install `ptgnn` from pypi, including all other dependencies:
    ```bash
   pip install ptgnn
    ```
   If you want to use ptgnn sampels with Azure ML (e.g. the `--aml` flag in the implementation CLIs), install with
   ```bash
    pip install ptgnn[aml]
    ```
   or directly from the sources, `cd` into the root directory of the project and run
    ```bash
    pip install -e .
    ```
    To check that installation was successful and run the unit tests:
    ```python
    python setup.py test
    ```

1. To install `ptgnn` from conda, including all other dependencies:
    ```bash
   conda search ptgnn --channel conda-forge
    ```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


### Developing ptgnn
![Unit Tests](https://github.com/microsoft/ptgnn/workflows/Unit%20Tests/badge.svg)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/microsoft/ptgnn.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/microsoft/ptgnn/alerts/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

To contribute to this library, first follow the next steps to setup your
development environment:
* Install the library requirements.
* Install the pre-commit hooks:
    * Run `pip3 install pre-commit`
    * Install the hooks `pre-commit install`

###### Using Conda
If you are using conda, then download the correct torch-scatter wheel.
If using `torch==1.7.0` and Python 3.7, you can use the [environment.yml](environment.yml)
included in the repo, with the following steps:
```
$ conda env create -f environment.yml
$ conda activate ptgnn-env
$ pip install torch_scatter-2.0.5+cu102-cp37-cp37m-linux_x86_64.whl
$ pip install -e .
$ python setup.py test
$ pip install pre-commit
$ pre-commit install
```

###### Releasing to PyPi
To create a PyPi release push a tag in the form `v1.3.4` in this repository (make
sure that you follow [semantic versioning](https://semver.org/)). The
[Publish on PyPi](https://github.com/microsoft/ptgnn/actions?query=workflow%3A%22Publish+on+PyPi%22)
GitHub action will automatically upload a new release.
