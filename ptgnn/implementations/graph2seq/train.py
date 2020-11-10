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
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from pathlib import Path

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable
from ptgnn.implementations.graph2seq.graph2seq import Graph2Seq
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import (
    StrElementRepresentationModel,
)
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.messagepassing.gatedmessagepassing import GatedMessagePassingLayer
from ptgnn.neuralmodels.gnn.messagepassing.residuallayers import MeanResidualLayer
from ptgnn.neuralmodels.sequence.grucopydecoder import GruCopyingDecoderModel


def run(arguments):
    if arguments["--aml"]:
        import torch
        from azureml.core.run import Run

        aml_ctx = Run.get_context()
        assert torch.cuda.is_available(), "No CUDA available. Aborting training."
    else:
        aml_ctx = None

    log_path = configure_logging(aml_ctx)
    azure_info_path = arguments.get("--azure-info", None)

    training_data_path = RichPath.create(arguments["TRAIN_DATA_PATH"], azure_info_path)
    training_data = LazyDataIterable(lambda: training_data_path.read_as_jsonl())

    validation_data_path = RichPath.create(arguments["VALID_DATA_PATH"], azure_info_path)
    validation_data = LazyDataIterable(lambda: validation_data_path.read_as_jsonl())

    model_path = Path(arguments["MODEL_FILENAME"])
    assert model_path.name.endswith(".pkl.gz"), "MODEL_FILENAME must have a `.pkl.gz` suffix."

    initialize_metadata = True
    restore_path = arguments.get("--restore-path", None)
    if restore_path:
        initialize_metadata = False
        model, nn = AbstractNeuralModel.restore_model(Path(restore_path))
    else:
        embedding_size = 128
        dropout_rate = 0.1
        nn = None

        def create_mp_layers(num_edges: int):
            ggnn_mp = GatedMessagePassingLayer(
                state_dimension=embedding_size,
                message_dimension=embedding_size,
                num_edge_types=num_edges,
                message_aggregation_function="sum",
                dropout_rate=dropout_rate,
            )
            r1 = MeanResidualLayer(embedding_size)
            return [
                r1.pass_through_dummy_layer(),
                ggnn_mp,
                ggnn_mp,
                ggnn_mp,
                ggnn_mp,
                ggnn_mp,
                ggnn_mp,
                ggnn_mp,
                r1,
                GatedMessagePassingLayer(
                    state_dimension=embedding_size,
                    message_dimension=embedding_size,
                    num_edge_types=num_edges,
                    message_aggregation_function="sum",
                    dropout_rate=dropout_rate,
                ),
            ]

        model = Graph2Seq(
            gnn_model=GraphNeuralNetworkModel(
                node_representation_model=StrElementRepresentationModel(
                    token_splitting="token",
                    embedding_size=embedding_size,
                ),
                message_passing_layer_creator=create_mp_layers,
            ),
            decoder=GruCopyingDecoderModel(
                hidden_size=128, embedding_size=256, memories_hidden_dim=embedding_size
            ),
        )

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=int(arguments["--minibatch-size"]),
        enable_amp=arguments["--amp"],
    )
    if nn is not None:
        trainer.neural_module = nn

    trainer.register_train_epoch_end_hook(
        lambda model, nn, epoch, metrics: log_run(aml_ctx, "train", model, epoch, metrics)
    )
    trainer.register_validation_epoch_end_hook(
        lambda model, nn, epoch, metrics: log_run(aml_ctx, "valid", model, epoch, metrics)
    )

    trainer.train(
        training_data,
        validation_data,
        show_progress_bar=not arguments["--quiet"],
        initialize_metadata=initialize_metadata,
        parallelize=not arguments["--sequential-run"],
    )

    if aml_ctx is not None:
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(model_path))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
