#!/usr/bin/env python
"""
Usage:
    train.py [options] TRAIN_DATA_PATH VALID_DATA_PATH MODEL_FILENAME

Options:
    --aml                      Run this in Azure ML
    --amp                      Enable automatic mixed precision.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --max-num-epochs=<epochs>  The maximum number of epochs to run training for. [default: 100]
    --minibatch-size=<size>    The minibatch size. [default: 300]
    --restore-path=<path>      The path to previous model file for starting from previous checkpoint.
    --sequential-run           Do not parallelize data loading. Makes debugging easier.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import random
import torch
import torch.distributed as dist
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from functools import partial
from pathlib import Path

from ptgnn.baseneuralmodel.distributedtrainer import DistributedModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable, ShardedLazyDataIterable
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
    # TODO: Use ZeRo
    return torch.optim.Adam(parameters, lr=0.00025)


def run(arguments):
    if arguments["--aml"]:
        from azureml.core.run import Run

        aml_ctx = Run.get_context()
        assert torch.cuda.is_available(), "No CUDA available. Aborting training."
    else:
        aml_ctx = None

    log_path = configure_logging(aml_ctx)
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
        model, nn = Graph2Class.restore_model(Path(restore_path))
    elif arguments["--aml"] and model_path.exists():
        initialize_metadata = False
        model, nn = Graph2Class.restore_model(model_path)
    else:
        nn = None
        model = create_graph2class_gnn_model()

    trainer = DistributedModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=int(arguments["--minibatch-size"]),
        optimizer_creator=create_optimizer,
        clip_gradient_norm=None,
        # target_validation_metric="Accuracy",
        # target_validation_metric_higher_is_better=True,
        enable_amp=arguments["--amp"],
    )
    if nn is not None:
        trainer.neural_module = nn

    # TODO: Use a serializable form instead of the lambdas (`partial`)
    # trainer.register_train_epoch_end_hook(
    #     lambda model, nn, epoch, metrics: log_run(aml_ctx, "train", model, epoch, metrics)
    # )
    # trainer.register_validation_epoch_end_hook(
    #     lambda model, nn, epoch, metrics: log_run(aml_ctx, "valid", model, epoch, metrics)
    # )

    trainer.distributed_train(
        training_data,
        validation_data,
        initialize_metadata=initialize_metadata,
        parallelize=not arguments["--sequential-run"],
        validate_on_start=True,
        shuffle_training_data=False,  # TODO: Remove!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        patience=10,
    )

    if aml_ctx is not None:
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(model_path))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
