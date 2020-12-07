from typing_extensions import Final, Protocol

import json
import logging
import math
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Dict, Generic, Iterable, List, Optional, TypeVar

from ptgnn.baseneuralmodel.abstractneuralmodel import AbstractNeuralModel
from ptgnn.baseneuralmodel.modulewithmetrics import ModuleWithMetrics
from ptgnn.baseneuralmodel.utils.data import MemorizedDataIterable

TRawDatapoint = TypeVar("TRawDatapoint")
TTensorizedDatapoint = TypeVar("TTensorizedDatapoint")
TNeuralModule = TypeVar("TNeuralModule", bound=ModuleWithMetrics)
ModelType = AbstractNeuralModel[TRawDatapoint, TTensorizedDatapoint, TNeuralModule]
EndOfEpochHook = Callable[[ModelType, TNeuralModule, int, Dict], None]

__all__ = ["ModelTrainer", "AbstractScheduler", "EndOfEpochHook"]


class AbstractScheduler(Protocol):
    def step(self, epoch_idx: int, epoch_step: int) -> None:
        ...


class ModelTrainer(Generic[TRawDatapoint, TTensorizedDatapoint, TNeuralModule]):
    """
    A trainer for `AbstractComponent`s. Used mainly for supervised learning.

    Create a `ComponentTrainer` by passing a `AbstractComponent` in the constructor.
    Invoke `train()` to initiate the training loop. The root `TNeuralModule` should return a scalar loss.
    """

    LOGGER: Final = logging.getLogger(__name__)

    def __init__(
        self,
        model: AbstractNeuralModel[TRawDatapoint, TTensorizedDatapoint, TNeuralModule],
        checkpoint_location: Path,
        *,
        max_num_epochs: int = 100,
        minibatch_size: int = 200,
        optimizer_creator: Optional[
            Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
        ] = None,
        scheduler_creator: Optional[Callable[[torch.optim.Optimizer], AbstractScheduler]] = None,
        clip_gradient_norm: Optional[float] = None,
        target_validation_metric: Optional[str] = None,
        target_validation_metric_higher_is_better: bool = False,
        enable_amp: bool = False,
    ):
        """
        :param model: The Component to be built and trained
        :param checkpoint_location: The location where the trained model will be checkpointed and saved.
        :param max_num_epochs: The maximum number of epochs to run training for.
        :param minibatch_size: The maximum size of the minibatch (`BaseComponent`s can override this
            by detecting full minibatches and returning False in `extend_minibatch_by_sample`)
        :param optimizer_creator: An optional function that accepts an iterable of the training parameters
            (pyTorch tensors) and returns a PyTorch optimizer.
        :param scheduler_creator: An optional function that accepts an optimizer and creates a scheduler
            implementing `AbstractScheduler`. This could be a wrapper for existing learning schedulers.
            The scheduler will be invoked at after each training step.
        """
        self.__model = model
        self.__neural_network: Optional[TNeuralModule] = None
        self.__checkpoint_location = checkpoint_location

        self.__max_num_epochs = max_num_epochs
        self.__minibatch_size = minibatch_size
        if optimizer_creator is None:
            self.__create_optimizer = lambda p: torch.optim.Adam(p)
        else:
            self.__create_optimizer = optimizer_creator

        self.__create_scheduler = scheduler_creator

        self.__metadata_finalized_hooks: List[Callable[[ModelType], None]] = []
        self.__training_start_hooks: List[
            Callable[[ModelType, TNeuralModule, torch.optim.Optimizer], None]
        ] = []
        self.__train_epoch_end_hooks: List[EndOfEpochHook] = []
        self.__validation_epoch_end_hooks: List[EndOfEpochHook] = []
        self.__clip_gradient_norm = clip_gradient_norm
        self.__enable_amp = enable_amp

        self.__target_metric = target_validation_metric
        if target_validation_metric is not None:
            self.__target_metric_higher_is_better = target_validation_metric_higher_is_better
        else:
            assert (
                not target_validation_metric_higher_is_better
            ), "When no explicit metric is passed, the validation loss will be used."
            self.__target_metric_higher_is_better = False

    @property
    def model(self) -> AbstractNeuralModel[TRawDatapoint, TTensorizedDatapoint, TNeuralModule]:
        return self.__model

    @property
    def neural_module(self) -> TNeuralModule:
        if self.__neural_network is None:
            raise Exception("Neural Network Module has not been built.")
        return self.__neural_network

    @neural_module.setter
    def neural_module(self, nn: TNeuralModule):
        self.__neural_network = nn

    def load_metadata_and_create_network(
        self, training_data: Iterable[TRawDatapoint], parallelize: bool, show_progress_bar: bool
    ) -> None:
        return self.__load_metadata_and_create_network(
            training_data, parallelize, show_progress_bar
        )

    def __load_metadata_and_create_network(
        self, training_data: Iterable[TRawDatapoint], parallelize: bool, show_progress_bar: bool
    ) -> None:
        """
        Compute model metadata by doing a full pass over the training data.
        """
        self.__model.compute_metadata(
            iter(
                tqdm(
                    training_data,
                    desc="Loading Metadata",
                    leave=False,
                    disable=not show_progress_bar,
                )
            ),
            parallelize,
        )
        self.__neural_network = self.__model.build_neural_module()
        self.LOGGER.info(
            "Model metadata loaded. The following model was created:\n %s", self.__neural_network
        )

        for m in self.__metadata_finalized_hooks:
            m(self.__model)

        self.LOGGER.info(
            "Model Definition:\n %s",
            json.dumps(dict(self.__model.model_definition), indent=2),
        )

        self.LOGGER.info("Saving model with finalized metadata.")
        self.__save_checkpoint()

    def __save_checkpoint(self) -> None:
        self.__model.save(self.__checkpoint_location, self.neural_module)

    def __restore_checkpoint(self, device=None) -> None:
        _, self.neural_module = self.__model.restore_model(
            self.__checkpoint_location, device=device
        )

    def register_model_metadata_finalized_hook(self, hook: Callable[[ModelType], None]) -> None:
        self.__metadata_finalized_hooks.append(hook)

    def register_training_start_hook(
        self, hook: Callable[[ModelType, TNeuralModule, torch.optim.Optimizer], None]
    ) -> None:
        self.__training_start_hooks.append(hook)

    def register_train_epoch_end_hook(self, hook: EndOfEpochHook) -> None:
        self.__train_epoch_end_hooks.append(hook)

    def register_validation_epoch_end_hook(self, hook: EndOfEpochHook) -> None:
        self.__validation_epoch_end_hooks.append(hook)

    def _run_training(
        self,
        training_tensors,
        epoch,
        device,
        exponential_running_average_factor,
        optimizer,
        parallelize,
        scheduler,
        show_progress_bar,
        shuffle_input: bool = True,
    ):
        sum_epoch_loss, running_avg_loss, num_minibatches, num_samples = 0.0, 0.0, 0, 0
        start_time = time.time()
        self.neural_module.train()

        scaler = torch.cuda.amp.GradScaler(enabled=self.__enable_amp)
        with tqdm(desc="Training", disable=not show_progress_bar, leave=False) as progress_bar:
            for step_idx, (mb_data, raw_samples) in enumerate(
                self.__model.minibatch_iterator(
                    training_tensors(),
                    device=device,
                    max_minibatch_size=self.__minibatch_size,
                    yield_partial_minibatches=False,
                    shuffle_input=shuffle_input,
                    parallelize=parallelize,
                )
            ):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.__enable_amp):
                    mb_loss = self.neural_module(**mb_data)
                    if torch.isnan(mb_loss):
                        raise Exception("Loss has a NaN value.")

                    scaler.scale(mb_loss).backward()

                    if self.__clip_gradient_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.neural_module.parameters(recurse=True), self.__clip_gradient_norm
                        )

                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step(epoch_idx=epoch, epoch_step=step_idx)

                num_minibatches += 1
                num_samples += len(raw_samples)
                mb_loss = float(mb_loss.cpu())
                sum_epoch_loss += mb_loss
                if num_minibatches == 1:  # First minibatch
                    running_avg_loss = mb_loss
                else:
                    running_avg_loss = (
                        exponential_running_average_factor * running_avg_loss
                        + (1 - exponential_running_average_factor) * mb_loss
                    )
                progress_bar.update()
                progress_bar.set_postfix(Loss=f"{running_avg_loss:.2f}")

        elapsed_time = time.time() - start_time
        self.LOGGER.info(
            "Training complete in %.1fsec [%.2f samples/sec]",
            elapsed_time,
            num_samples / elapsed_time,
        )
        assert (
            num_minibatches > 0
        ), "No training minibatches were created. The minibatch size may be too large or the training dataset size too small."
        self.LOGGER.info("Epoch %i: Train Loss %.2f", epoch + 1, sum_epoch_loss / num_minibatches)
        train_metrics = self.neural_module.report_metrics()

        for epoch_hook in self.__train_epoch_end_hooks:
            epoch_hook(self.__model, self.neural_module, epoch, train_metrics)

        if len(train_metrics) > 0:
            self.LOGGER.info("Training Metrics: %s", json.dumps(train_metrics, indent=2))

    def _run_validation(
        self, validation_tensors, epoch, best_target_metric, device, parallelize, show_progress_bar
    ):
        self.neural_module.eval()
        sum_epoch_loss, num_minibatches, num_samples = 0.0, 0, 0
        start_time = time.time()
        with tqdm(
            desc="Validation", disable=not show_progress_bar, leave=False
        ) as progress_bar, torch.no_grad():
            for mb_data, raw_samples in self.__model.minibatch_iterator(
                validation_tensors(),
                device=device,
                max_minibatch_size=self.__minibatch_size,
                yield_partial_minibatches=True,
                shuffle_input=False,
                parallelize=parallelize,
            ):
                with torch.cuda.amp.autocast(enabled=self.__enable_amp):
                    mb_loss = self.neural_module(**mb_data)
                num_minibatches += 1
                num_samples += len(raw_samples)
                sum_epoch_loss += float(mb_loss.cpu())
                progress_bar.update()
                progress_bar.set_postfix(Loss=f"{sum_epoch_loss / num_minibatches:.2f}")

        elapsed_time = time.time() - start_time
        assert num_samples > 0, "No validation data was found."

        validation_loss = sum_epoch_loss / num_minibatches
        self.LOGGER.info(
            "Validation complete in %.1fsec [%.2f samples/sec]",
            elapsed_time,
            (num_samples / elapsed_time),
        )
        self.LOGGER.info("Epoch %i: Valid Loss %.2f", epoch + 1, validation_loss)

        validation_metrics = self.neural_module.report_metrics()
        for epoch_hook in self.__validation_epoch_end_hooks:
            epoch_hook(self.__model, self.neural_module, epoch, validation_metrics)
        if len(validation_metrics) > 0:
            self.LOGGER.info("Validation Metrics: %s", json.dumps(validation_metrics, indent=2))

        if self.__target_metric is not None:
            target_metric = validation_metrics[self.__target_metric]
        else:
            target_metric = validation_loss

        if self.__target_metric_higher_is_better:
            target_metric_improved = target_metric > best_target_metric
        else:
            target_metric_improved = target_metric < best_target_metric

        return target_metric, target_metric_improved

    def train(
        self,
        training_data: Iterable[TRawDatapoint],
        validation_data: Iterable[TRawDatapoint],
        *,
        show_progress_bar: bool = True,
        validate_on_start: bool = True,
        patience: int = 5,
        initialize_metadata: bool = True,
        parallelize: bool = True,
        use_multiprocessing: bool = True,
        exponential_running_average_factor: float = 0.97,
        device=None,
        store_tensorized_data_in_memory: bool = False,
        shuffle_training_data: bool = True,
    ) -> None:
        """
        The training-validation loop for `AbstractNeuralModel`s.

        :param training_data: An iterable that each iteration yields the full training data.
        :param validation_data: An iterable that each iteration yields the full validation data.
        :param show_progress_bar: Show a progress bar
        :param validate_on_start: Whether to run a validation loop on start
        :param patience: The number of iterations before early stopping kicks in.
        :param initialize_metadata: If true, initialize the metadata from the training_data. Otherwise,
            assume that the model that is being trained has its metadata already initialized.
        :param parallelize: Bool indicating whether to run in parallel
        :param use_multiprocessing: Whether to use multiprocessing
        :param exponential_running_average_factor: The factor of the running average of the training loss
            displayed in the progress bar.
        :param device: the target PyTorch device for training
        :param store_tensorized_data_in_memory: store all tensorized data in memory instead of computing them on-line.
        :param shuffle_training_data: shuffle the incoming data from `training_data`.
        """
        if initialize_metadata:
            self.load_metadata_and_create_network(training_data, parallelize, show_progress_bar)

        self.LOGGER.info(
            "Model has %s trainable parameters.",
            sum(
                param.numel()
                for param in self.neural_module.parameters(recurse=True)
                if param.requires_grad
            ),
        )

        training_tensors = lambda: self.__model.tensorize_dataset(
            iter(training_data), parallelize=parallelize, use_multiprocessing=use_multiprocessing
        )
        validation_tensors = lambda: self.__model.tensorize_dataset(
            iter(validation_data), parallelize=parallelize, use_multiprocessing=use_multiprocessing
        )
        if store_tensorized_data_in_memory:
            training_tensors = MemorizedDataIterable(training_tensors, shuffle=True)
            validation_tensors = MemorizedDataIterable(validation_tensors)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.LOGGER.info("Using `%s` for training." % device)
        self.neural_module.to(device)

        optimizer = self.__create_optimizer(self.neural_module.parameters())
        scheduler = None if self.__create_scheduler is None else self.__create_scheduler(optimizer)

        for hook in self.__training_start_hooks:
            hook(self.__model, self.neural_module, optimizer)

        if self.__target_metric_higher_is_better and self.__target_metric is not None:
            best_target_metric = -math.inf
        else:
            best_target_metric = math.inf

        if validate_on_start:
            target_metric, improved = self._run_validation(
                validation_tensors, 0, best_target_metric, device, parallelize, show_progress_bar
            )
            assert improved
            self.LOGGER.info(f"Initial {self.__target_metric or 'Loss'}: {target_metric}")
            best_target_metric = target_metric

        num_epochs_not_improved: int = 0
        for epoch in range(self.__max_num_epochs):
            self._run_training(
                training_tensors,
                epoch,
                device,
                exponential_running_average_factor,
                optimizer,
                parallelize,
                scheduler,
                show_progress_bar,
                shuffle_training_data,
            )

            target_metric, target_metric_improved = self._run_validation(
                validation_tensors,
                epoch,
                best_target_metric,
                device,
                parallelize,
                show_progress_bar,
            )
            if target_metric_improved:
                self.LOGGER.info(
                    f"Best performance so far "
                    f"({self.__target_metric or 'Loss'}: {target_metric:.3f} from {best_target_metric:.3f}). "
                    "Saving model checkpoint."
                )
                num_epochs_not_improved = 0
                self.__save_checkpoint()
                best_target_metric = target_metric
            else:
                num_epochs_not_improved += 1
                if num_epochs_not_improved > patience:
                    self.LOGGER.warning(
                        f"The target metric has not improved for {num_epochs_not_improved} epochs . Stopping."
                    )
                    break

        # Restore the best model params that were found.
        self.__restore_checkpoint(device=device)
