#!/usr/bin/env python
"""
Usage:
    train.py [options] TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME

Options:
    --aml                      Run this in Azure ML
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
    --max-num-epochs=<epochs>  The maximum number of epochs to run training for. [default: 100]
    --minibatch-size=<size>    The minibatch size. [default: 300]
    --sequential-run           Do not parallelize data loading. Makes debugging easier.
    --quiet                    Do not show progress bar.
    --restore-path=<path>      The path to previous model file for starting from previous checkpoint.
    -h --help                  Show this screen.
"""
import random
import torch
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from pathlib import Path

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable
from ptgnn.implementations.varmisuse.candidateannotatedembeddings import (
    CandidateNodeAnnotationModel,
)
from ptgnn.implementations.varmisuse.varmisuse import VarMisuseModel
from ptgnn.neuralmodels.gnn import GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.messagepassing import (
    GatedMessagePassingLayer,
    GruGlobalStateUpdate,
    MeanResidualLayer,
    MlpMessagePassingLayer,
)
from ptgnn.neuralmodels.gnn.messagepassing.residuallayers import ConcatResidualLayer
from ptgnn.neuralmodels.reduceops import WeightedSumVarSizedElementReduce


def create_var_misuse_gnn_model(hidden_state_size: int = 64):
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

    def create_ggnn_mp_layers(num_edges: int):
        ggnn_mp = GatedMessagePassingLayer(
            state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="sum",
            dropout_rate=0.01,
        )
        r1 = MeanResidualLayer(hidden_state_size)
        r2 = MeanResidualLayer(hidden_state_size)
        global_update = lambda: GruGlobalStateUpdate(
            global_graph_representation_module=WeightedSumVarSizedElementReduce(hidden_state_size),
            input_state_size=hidden_state_size,
            summarized_state_size=hidden_state_size,
            dropout_rate=0.1,
        )
        return [
            r1.pass_through_dummy_layer(),
            r2.pass_through_dummy_layer(),
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            global_update(),
            ggnn_mp,
            r1,
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            global_update(),
            ggnn_mp,
            r2,
        ]

    return VarMisuseModel(
        gnn_model=GraphNeuralNetworkModel(
            node_representation_model=CandidateNodeAnnotationModel(
                embedding_size=hidden_state_size, token_splitting="char"
            ),
            message_passing_layer_creator=create_mlp_mp_layers,
            max_nodes_per_graph=50000,
            max_graph_edges=500000,
            introduce_backwards_edges=True,
            add_self_edges=True,
            stop_extending_minibatch_after_num_nodes=80000,
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

    def load_from_folder(path: RichPath, shuffle: bool):
        all_files = path.get_filtered_files_in_dir("*.jsonl.gz")
        if shuffle:
            random.shuffle(all_files)
        for file in all_files:
            yield from file.read_as_jsonl()

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
        model, nn = AbstractNeuralModel.restore_model(Path(restore_path))
    else:
        nn = None
        model = create_var_misuse_gnn_model()

    def create_optimizer(parameters):
        return torch.optim.Adam(parameters, lr=1e-3)

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=int(arguments["--minibatch-size"]),
        optimizer_creator=create_optimizer,
        clip_gradient_norm=1,
        target_validation_metric="Accuracy",
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
