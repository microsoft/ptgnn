#!/usr/bin/env python
"""
Usage:
    train.py [options] DATA_PATH MODEL_FILENAME

Options:
    --aml                      Run this in Azure ML
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --max-num-epochs=<epochs>  The maximum number of epochs to run training for. [default: 1000]
    --minibatch-size=<size>    The minibatch size. [default: 20]
    --restore-path=<path>      The path to previous model file for starting from previous checkpoint.
    --sequential-run           Do not parallelize data loading. Makes debugging easier.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import json
import numpy as np
import torch
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from pathlib import Path
from torch import nn

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.implementations.ppi.dataloader import PPIDatasetLoader
from ptgnn.implementations.ppi.ppi import PPIMulticlassClassification
from ptgnn.neuralmodels.embeddings.linearmapembedding import FeatureRepresentationModel
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.messagepassing.mlpmessagepassing import MlpMessagePassingLayer
from ptgnn.neuralmodels.gnn.messagepassing.residuallayers import MeanResidualLayer


def create_ppi_gnn_model(hidden_state_size: int = 256):
    def create_mp_layers(num_edges: int):
        mlp_mp_constructor = lambda: MlpMessagePassingLayer(
            input_state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            output_state_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="sum",
            dropout_rate=0.2,
        )
        r1 = MeanResidualLayer(hidden_state_size)
        r2 = MeanResidualLayer(hidden_state_size)
        return [
            r1.pass_through_dummy_layer(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            r1,
            r2.pass_through_dummy_layer(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            r2,
        ]

    return PPIMulticlassClassification(
        gnn_model=GraphNeuralNetworkModel[np.ndarray, np.ndarray](
            node_representation_model=FeatureRepresentationModel(
                embedding_size=hidden_state_size,
                activation=nn.Tanh(),
            ),
            message_passing_layer_creator=create_mp_layers,
            max_nodes_per_graph=6000,
            max_graph_edges=300000,
            introduce_backwards_edges=True,
            add_self_edges=True,
            stop_extending_minibatch_after_num_nodes=3000,
        ),
    )


def run(arguments):
    if arguments["--aml"]:
        from azureml.core.run import Run

        aml_ctx = Run.get_context()
        assert torch.cuda.is_available(), "No CUDA available. Aborting training."
    else:
        aml_ctx = None

    log_path = configure_logging(aml_ctx)
    azure_info_path = arguments.get("--azure-info", None)

    data_path = RichPath.create(arguments["DATA_PATH"], azure_info_path)
    training_data = PPIDatasetLoader.load_data(data_path, "train")
    validation_data = PPIDatasetLoader.load_data(data_path, "valid")

    model_path = Path(arguments["MODEL_FILENAME"])
    assert model_path.name.endswith(".pkl.gz"), "MODEL_FILENAME must have a `.pkl.gz` suffix."

    initialize_metadata = True
    restore_path = arguments.get("--restore-path", None)

    if restore_path:
        initialize_metadata = False
        model, nn = AbstractNeuralModel.restore_model(Path(restore_path))
    else:
        model = create_ppi_gnn_model()
        nn = None

    def create_optimizer(parameters):
        return torch.optim.Adam(
            parameters,
            lr=1e-3,
        )

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=int(arguments["--minibatch-size"]),
        optimizer_creator=create_optimizer,
        clip_gradient_norm=1,
        target_validation_metric="f1_score",
        target_validation_metric_higher_is_better=True,
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
        patience=20,
    )

    test_data = PPIDatasetLoader.load_data(data_path, "test")
    metrics = model.report_metrics(
        test_data,
        trainer.neural_module,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    print(f"Test metrics: {json.dumps(metrics, indent=3)}")

    if aml_ctx is not None:
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(model_path))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
