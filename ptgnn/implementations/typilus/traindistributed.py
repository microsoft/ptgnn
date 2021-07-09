#!/usr/bin/env python
"""
Usage:
    train.py [options] TRAIN_DATA_PATH VALID_DATA_PATH MODEL_FILENAME

Options:
    --amp                      Enable automatic mixed precision.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --max-num-epochs=<epochs>  The maximum number of epochs to run training for. [default: 100]
    --minibatch-size=<size>    The minibatch size. [default: 300]
    --restore-path=<path>      The path to previous model file for starting from previous checkpoint.
    --autorestore              Automatically restore a checkpoint if one exists.
    --sequential-run           Do not parallelize data loading. Makes debugging easier.
    --quiet                    Do not show progress bar.
    --world-size=<int>         The number of GPUs to use (assumes single-node, multi-GPUs). [default: -1]
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import random
import torch
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from functools import partial
from pathlib import Path

from ptgnn.baseneuralmodel.distributedtrainer import DistributedModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import ShardedLazyDataIterable
from ptgnn.implementations.typilus.graph2class import Graph2Class
from ptgnn.implementations.typilus.train import create_graph2class_gnn_model


def load_from_folder(path: RichPath, shuffle: bool, rank: int, world_size):
    all_files = [
        p
        for i, p in enumerate(path.get_filtered_files_in_dir("*.jsonl.gz"))
        if i % world_size == rank
    ]

    if shuffle:
        random.shuffle(all_files)
    for file in all_files:
        yield from file.read_as_jsonl()


def create_optimizer(parameters):
    from torch.distributed.optim import ZeroRedundancyOptimizer

    # return torch.optim.Adam(parameters, lr=0.01)
    return ZeroRedundancyOptimizer(parameters, optimizer_class=torch.optim.Adam, lr=0.001)


def log_run_lambda(aml_ctx, fold, model, nn, epoch, metrics):
    """A utility function that can be used with partial(), and can be serialized through multiprocessing."""
    log_run(aml_ctx, fold, model, epoch, metrics)


def worker_init(trainer: DistributedModelTrainer, rank, world_isze):
    try:
        from azureml.core.run import Run

        aml_ctx = Run.get_context()
    except:
        aml_ctx = None

    log_path = configure_logging(aml_ctx, rank=rank)

    trainer.register_train_epoch_end_hook(partial(log_run_lambda, aml_ctx, "train-" + str(rank)))
    trainer.register_validation_epoch_end_hook(
        partial(log_run_lambda, aml_ctx, "valid-" + str(rank))
    )

    def upload_hook(model, nn, epoch, metrics):
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(trainer._checkpoint_location))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)

    if rank == 0 and aml_ctx is not None:
        trainer.register_epoch_improved_end_hook(upload_hook)


def run(arguments):
    azure_info_path = arguments.get("--azure-info", None)
    training_data_path = RichPath.create(arguments["TRAIN_DATA_PATH"], azure_info_path)
    training_data = ShardedLazyDataIterable(
        partial(load_from_folder, training_data_path, shuffle=True)
    )

    validation_data_path = RichPath.create(arguments["VALID_DATA_PATH"], azure_info_path)
    validation_data = ShardedLazyDataIterable(
        partial(load_from_folder, validation_data_path, shuffle=False)
    )

    model_path = Path(arguments["MODEL_FILENAME"])
    assert model_path.name.endswith(".pkl.gz"), "MODEL_FILENAME must have a `.pkl.gz` suffix."

    initialize_metadata = True
    restore_path = arguments.get("--restore-path", None)
    if restore_path:
        initialize_metadata = False
        model, nn = Graph2Class.restore_model(Path(restore_path), device="cpu")
    elif arguments["--autorestore"] and model_path.exists():
        initialize_metadata = False
        model, nn = Graph2Class.restore_model(model_path, device="cpu")
    else:
        nn = None
        model = create_graph2class_gnn_model()

    trainer = DistributedModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=int(arguments["--minibatch-size"]),
        optimizer_creator=create_optimizer,
        clip_gradient_norm=1,
        target_validation_metric="Accuracy",
        target_validation_metric_higher_is_better=True,
        enable_amp=arguments["--amp"],
    )
    if nn is not None:
        trainer.neural_module = nn

    world_size = int(arguments["--world-size"])
    if world_size == -1:
        world_size = torch.cuda.device_count()

    trainer.distributed_train(
        world_size,
        training_data,
        validation_data,
        initialize_metadata=initialize_metadata,
        parallelize=not arguments["--sequential-run"],
        validate_on_start=True,
        shuffle_training_data=True,
        patience=10,
        worker_init=worker_init,
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
