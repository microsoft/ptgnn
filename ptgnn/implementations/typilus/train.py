#!/usr/bin/env python
"""
Usage:
    train.py [options] TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME

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
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from pathlib import Path

from ptgnn.baseneuralmodel import ModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable
from ptgnn.implementations.typilus.graph2class import Graph2Class
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import (
    StrElementRepresentationModel,
)
from ptgnn.neuralmodels.gnn import GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.messagepassing import GatedMessagePassingLayer, MlpMessagePassingLayer
from ptgnn.neuralmodels.gnn.messagepassing.residuallayers import ConcatResidualLayer


def create_graph2class_gnn_model(hidden_state_size: int = 64, dropout_rate: float = 0.1):
    def create_ggnn_mp_layers(num_edges: int):
        ggnn_mp = GatedMessagePassingLayer(
            state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="max",
            dropout_rate=dropout_rate,
        )
        r1 = ConcatResidualLayer(hidden_state_size)
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
                state_dimension=2 * hidden_state_size,
                message_dimension=hidden_state_size,
                num_edge_types=num_edges,
                message_aggregation_function="max",
                dropout_rate=dropout_rate,
            ),
        ]

    def create_mlp_mp_layers(num_edges: int):
        mlp_mp_constructor = lambda: MlpMessagePassingLayer(
            input_state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            output_state_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="max",
            dropout_rate=0.1,
        )
        mlp_mp_after_res_constructor = lambda: MlpMessagePassingLayer(
            input_state_dimension=2 * hidden_state_size,
            message_dimension=2 * hidden_state_size,
            output_state_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="max",
            dropout_rate=0.1,
        )
        r1 = ConcatResidualLayer(hidden_state_size)
        r2 = ConcatResidualLayer(hidden_state_size)
        return [
            r1.pass_through_dummy_layer(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            r1,
            mlp_mp_after_res_constructor(),
            r2.pass_through_dummy_layer(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            r2,
            mlp_mp_after_res_constructor(),
        ]

    return Graph2Class(
        gnn_model=GraphNeuralNetworkModel(
            node_representation_model=StrElementRepresentationModel(
                embedding_size=hidden_state_size,
                token_splitting="subtoken",
                subtoken_combination="mean",
                vocabulary_size=10000,
                min_freq_threshold=5,
                dropout_rate=dropout_rate,
            ),
            message_passing_layer_creator=create_mlp_mp_layers,
            max_nodes_per_graph=100000,
            max_graph_edges=500000,
            introduce_backwards_edges=True,
            add_self_edges=True,
            stop_extending_minibatch_after_num_nodes=120000,
            edge_dropout_rate=0.0,
        ),
        max_num_classes=100,
    )


def load_from_folder(path: RichPath, shuffle: bool):
    all_files = path.get_filtered_files_in_dir("*.jsonl.gz")
    if shuffle:
        random.shuffle(all_files)
    for file in all_files:
        yield from file.read_as_jsonl()


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
    training_data = LazyDataIterable(lambda: load_from_folder(training_data_path, shuffle=True))

    validation_data_path = RichPath.create(arguments["VALID_DATA_PATH"], azure_info_path)
    validation_data = LazyDataIterable(
        lambda: load_from_folder(validation_data_path, shuffle=False)
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

    def create_optimizer(parameters):
        return torch.optim.Adam(parameters, lr=0.00025)

    trainer = ModelTrainer(
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
        patience=10,
        store_tensorized_data_in_memory=True,
    )

    test_data_path = RichPath.create(arguments["TEST_DATA_PATH"], azure_info_path)
    test_data = LazyDataIterable(lambda: load_from_folder(test_data_path, shuffle=False))
    acc = model.report_accuracy(
        test_data,
        trainer.neural_module,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    print(f"Test accuracy: {acc:%}")

    if aml_ctx is not None:
        aml_ctx.log("Test Accuracy", acc)
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(model_path))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
