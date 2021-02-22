from typing_extensions import final

import gzip
import os
import torch
from abc import ABC, abstractmethod
from concurrent import futures
from dpu_utils.utils.iterators import BufferedIterator, ThreadedIterator, shuffled_iterator
from itertools import islice
from pathlib import Path
from torch import nn
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

__all__ = ["AbstractNeuralModel"]

TRawDatapoint = TypeVar("TRawDatapoint")
TTensorizedDatapoint = TypeVar("TTensorizedDatapoint")
TNeuralModule = TypeVar("TNeuralModule", bound=nn.Module)
T = TypeVar("T")


class AbstractNeuralModel(ABC, Generic[TRawDatapoint, TTensorizedDatapoint, TNeuralModule]):
    """
    This class provides the structure for defining neural networks in a compositional way.
    An `AbstractNeuralModel` accepts as input `TRawDatapoint`, a single example. Each model can have some
    metadata. For example, this may be the vocabulary of words to be represented and their mappings to integer ids.
    The computed metadata is then used to build a `TNeuralModule` (a `nn.Module`) which contains
    the definition of the neural network architecture and its parameters. During training and testing, each `TRawDatapoint`
    is converted to a tensorized format `TTensorizedDatapoint`. By batching multiple `TTensorizedDatapoint` a minibatch
    is created which is the input of the `TNeuralModule`.

    To create an `AbstractNeuralModel`:

    * Compute the metadata.
        * Initialize metadata in `initialize_metadata`. Store any metadata structures within the model.
        * Update this structure for each example in `update_metadata_from`. Explicitly unpack/transform any parts of the
            `TRawDatapoint` and invoke the `update_metadata_from` for all child models.
        * Finally, compute the model metadata in `finalize_metadata` and store them as fields in your implementation
            of the `AbstractNeuralModel`. Optionally, discard any temporary structures within the model.

    * Implement `build_neural_module()` that uses the metadata previously computed to create a `TNeuralModule`, ie.
        a concrete `nn.Module` or `ModuleWithMetrics`. Note that no elements of the neural network should be stored
        in this object but they should be returned.
    * Define `tensorize()` accepting a single `TRawDatapoint` and yielding a `TTensorizedDatapoint`.
        Each `TTensorizedDatapoint` should be considered a single sample for your neural network. Each model should
        call its child models (if any) to delegate tensorization of the relevant parts of the `TRawDatapoint`

    * Finally, to create minibatches:
        * Implement `initialize_minibatch` which creates a dictionary that will be populated with the information from
            multiple `TTensorizedDatapoint`s. Explicitly invoke child models to initialize aspect of the minibatch
            that are their responsibility.
        * Implement `extend_minibatch_with` to extend the minibatch with an additional sample. Return `True`
            if the minibatch can be further extended. Explicitly deconstruct the `TTensorizedDatapoint` and pass its
            parts to the responsible child models.
        * Implement `finalize_minibatch` to finalize the minibatch (invoking this for any child models). This
            should return a dictionary of PyTorch tensors ready to be consumed by the `forward()` of the `TNeuralModule`.
    """

    def __init__(self):
        self.__metadata_initialized = False

    @final
    @property
    def model_definition(self) -> Mapping[str, Any]:
        """A description of this model."""
        description: Dict[str, Any] = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, AbstractNeuralModel):
                description[attr] = value.model_definition
            elif isinstance(value, (int, float, str, bool, NamedTuple)):
                description[attr] = value
        return description

    # region Metadata and Neural Network Building
    def initialize_metadata(self) -> None:
        """
        Optionally initialize any model metadata (children metadata will be initialized separately).
        """

    @abstractmethod
    def update_metadata_from(self, datapoint: TRawDatapoint) -> None:
        """
        Update the model metadata from a single `datapoint`.

        Implementors of this method should:
        * Update all metadata that is needed by the model.
        * Unpack any parts of the parts of `datapoint` that are needed for the child models
            and invoke here their `update_metadata_from`.
        """
        raise NotImplementedError()

    def finalize_metadata(self) -> None:
        """
        Compute the final metadata. The final metadata should be stored within the model.
        Optionally, discard any temporary structures used to compute the finalized metadata.
        """

    def __initialize_metadata_recursive(self) -> None:
        self.initialize_metadata()
        for attr, value in self.__dict__.items():
            if isinstance(value, AbstractNeuralModel):
                value.__initialize_metadata_recursive()

    def __finalize_metadata_recursive(self) -> None:
        self.finalize_metadata()
        for attr, value in self.__dict__.items():
            if isinstance(value, AbstractNeuralModel):
                value.__finalize_metadata_recursive()
        self.__metadata_initialized = True

    @final
    def compute_metadata(
        self, dataset_iterator: Iterator[TRawDatapoint], parallelize: bool = True
    ) -> None:
        """
        Compute the metadata for this model including its children.
        This function should be invoked by the root-level model.
        """
        assert not self.__metadata_initialized, "Metadata has already been initialized."
        self.__initialize_metadata_recursive()
        for element in ThreadedIterator(dataset_iterator, enabled=parallelize):
            self.update_metadata_from(element)
        self.__finalize_metadata_recursive()

    @abstractmethod
    def build_neural_module(self) -> TNeuralModule:
        """
        Create the neural network that corresponds to this model.

        Note to implementors:
            * All model metadata will have been computed by this point and can be used here.
            * The output neural network should *not* be stored in this object but should be returned.
            * Invoke build_neural_module() for all child models and then compose a single neural module
              (ie. a `nn.Module` or `ModuleWithMetrics`) that encapsulates the children modules.
        """
        raise NotImplementedError()

    # endregion

    # region Saving/Loading
    def save(self, path: Path, model: TNeuralModule) -> None:
        os.makedirs(os.path.dirname(str(path.absolute())), exist_ok=True)
        with gzip.open(path, "wb") as f:
            torch.save((self, model), f)

    @classmethod
    def restore_model(cls: Type[T], path: Path, device=None) -> Tuple[T, TNeuralModule]:
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location=device)

    # endregion

    # region Tensor Conversion
    @abstractmethod
    def tensorize(self, datapoint: TRawDatapoint) -> Optional[TTensorizedDatapoint]:
        """
        This is called to tensorize the data of a single input example in a form that can be consumed by the
        neural network. This function yields a `TTensorizedDatapoint` or `None` if this datapoint should be
        discarded.

        Note to implementors: this usually involves unpacking the datapoint and invoking child models'
          `tensorize()` with the portions of the data that each child model is responsible about. Finally,
          this function should encapsulate this data into a `TTensorizedDatapoint` along with any additional
          tensorized information needed by this model.
        """
        raise NotImplementedError()

    @final
    def tensorize_dataset(
        self,
        dataset_iterator: Iterator[TRawDatapoint],
        *,
        parallelize: bool = True,
        use_multiprocessing: bool = True,
        return_input_data: bool = False,
    ) -> Iterator[Tuple[TTensorizedDatapoint, Optional[TRawDatapoint]]]:
        """
        Given a dataset, return an iterator of the tensorized data. If `return_input_data` is True
        then the original data is also returned.

        :param dataset_iterator: An iterator of raw data.
        :param parallelize: If True, the tensorization will be parallelized. This speeds up tensorization but
            makes debugging harder.
        :param use_multiprocessing: If parallelize==True, then use multiprocessing instead of multithreading.
            If parallelize==False this option is ignored.
        :param return_input_data: Should the raw input data also be returned?
        """
        assert self.__metadata_initialized, "Metadata has not been initialized."
        if parallelize:
            base_iterator = (
                (self.tensorize(d), d if return_input_data else None) for d in dataset_iterator
            )
            if use_multiprocessing:
                for tensorized_sample in BufferedIterator(base_iterator):
                    if tensorized_sample[0] is not None:
                        yield tensorized_sample
            else:
                with futures.ThreadPoolExecutor() as pool:
                    for tensorized_sample in pool.map(
                        lambda d: (
                            self.tensorize(d),
                            d if return_input_data else None,
                        ),
                        dataset_iterator,
                        chunksize=20,
                    ):
                        if tensorized_sample[0] is not None:
                            yield tensorized_sample
        else:
            for datapoint in dataset_iterator:
                tensorized_data_sample = self.tensorize(datapoint)
                if tensorized_data_sample is not None:
                    yield (tensorized_data_sample, datapoint if return_input_data else None)

    # endregion

    # region Minibatching Logic
    @abstractmethod
    def initialize_minibatch(self) -> Dict[str, Any]:
        """
        Initialize a dictionary that will be incrementally populated by `extend_minibatch_with()`. Once the minibatch
        is full, `finalize_minibatch()` will be invoked.

        Explicitly invoke `initialize_minibatch` for child models to initialize the parts of minibatch that are
         their responsibility.
        """
        raise NotImplementedError()

    @abstractmethod
    def extend_minibatch_with(
        self, tensorized_datapoint: TTensorizedDatapoint, partial_minibatch: Dict[str, Any]
    ) -> bool:
        """
        Add a datapoint to the partial minibatch. If for some reason the minibatch cannot accumulate
            additional samples after this one, this method should return False. Once the minibatch
            is full, `finalize_minibatch()` will be invoked.

        Note to implementors: Commonly, each model will unpack the `tensorized_datapoint` and invoke `extend_minibatch_with`
            of its child models. Then, if at least one of the child models returns False
            then this function should also return False.

        :param tensorized_datapoint: the datapoint to be added. This parameter is the output of `tensorize()`.
        :param partial_minibatch: the minibatch data to be populated. This is the dictionary that was initialized
            by `initialize_minibatch`.
        :return True if the minibatch can be further extended. False if for some reason the minibatch is full.
        """
        raise NotImplementedError()

    @abstractmethod
    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        """
        Finalize the minibatch data and make sure that the data is in an appropriate format to be consumed by
        the model. Commonly the values of the returned dictionary are `torch.tensor()` and the keys of the
        returned dictionary are the parameters of the neural module's forward().

        :param accumulated_minibatch_data: the data accumulated by `extend_minibatch_with`.
        :param device: the device for the torch tensors.
        :return: a dictionary `mb_data` that will is the input to the `forward(**mb_data)` of the model's neural module.
        """
        raise NotImplementedError()

    @final
    def __iterate_unfinalized_minibatches(
        self,
        tensorized_data: Iterator[Tuple[TTensorizedDatapoint, Optional[TRawDatapoint]]],
        max_minibatch_size: int,
        yield_partial_minibatches: bool = True,
    ) -> Iterator[Tuple[Dict[str, Any], List[Optional[TRawDatapoint]]]]:
        """
        Initialize and accumulate data into minibatches, but does not finalize them (so that this process can
         be offloaded to a different thread/process).
        """
        while True:
            mb_data = self.initialize_minibatch()
            mb_input_data, num_elements_added = [], 0
            for tensorized_sample, input_data in islice(tensorized_data, max_minibatch_size):
                continue_extending = self.extend_minibatch_with(tensorized_sample, mb_data)
                mb_input_data.append(input_data)
                num_elements_added += 1
                if not continue_extending:
                    # The implementation of the model asked to stop extending the minibatch.
                    data_iterator_exhausted = False
                    break
            else:
                # The data is exhausted if we finished iterating through the loop and still don't have max_num_items
                data_iterator_exhausted = num_elements_added < max_minibatch_size

            if num_elements_added == 0 or (
                data_iterator_exhausted and not yield_partial_minibatches
            ):
                return
            yield mb_data, mb_input_data

    @final
    def minibatch_iterator(
        self,
        tensorized_data: Iterator[Tuple[TTensorizedDatapoint, Optional[TRawDatapoint]]],
        device: Union[str, torch.device],
        max_minibatch_size: int,
        yield_partial_minibatches: bool = True,
        shuffle_input: bool = False,
        parallelize: bool = True,
    ) -> Iterator[Tuple[Dict[str, Any], List[Optional[TRawDatapoint]]]]:
        """
        An iterator that yields minibatches to be consumed by a neural module.
        :param tensorized_data: An iterator of tensorized data. Commonly that's the output of tensorize_dataset()
        :param device: The device on which the tensorized data will be stored.
        :param max_minibatch_size: the maximum size of the minibatch.
        :param yield_partial_minibatches: If true, yield partial minibatches, i.e. minibatches that do not
            reach the `max_minibatch_size` and the `extend_minibatch_with` did not consider full.
            Users might want to set this to False, when training.
        :param shuffle_input: Should the `tensorized_data` be shuffled? (e.g. during training)
        :param parallelize: if True, minibatching will be parallelized. This may make debugging harder.
        :return: an iterator that yield tuples, with the minibatch data and the raw data points (if they are present in `tensorized_data`).
        """
        assert self.__metadata_initialized, "Metadata has not been initialized."

        if shuffle_input:
            tensorized_data = shuffled_iterator(tensorized_data)

        unfinalized_minibatches = ThreadedIterator(
            self.__iterate_unfinalized_minibatches(
                tensorized_data, max_minibatch_size, yield_partial_minibatches
            ),
            enabled=parallelize,
        )
        yield from ThreadedIterator(
            ((self.finalize_minibatch(d[0], device), d[1]) for d in unfinalized_minibatches),
            enabled=parallelize,
        )

    # endregion
