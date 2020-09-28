#!/usr/bin/env python
"""
Usage:
    predict.py [options] MODEL_FILENAME DATA_PATH

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from pathlib import Path

from ptgnn.implementations.typilus.graph2class import Graph2Class
from ptgnn.implementations.typilus.train import load_from_folder


def run(arguments):
    azure_info_path = arguments.get("--azure-info", None)
    data_path = RichPath.create(arguments["DATA_PATH"], azure_info_path)
    data = load_from_folder(data_path, shuffle=False)

    model_path = Path(arguments["MODEL_FILENAME"])
    model, nn = Graph2Class.restore_model(model_path, "cpu")

    predictions = model.predict(data, nn, "cpu")
    for graph, suggestions in predictions:
        for supernode_idx, (target_type, prob) in suggestions.items():
            supernode_info = graph["supernodes"][str(supernode_idx)]
            print(
                f'`{supernode_info["name"]}` Original: `{supernode_info["annotation"]}` Predicted: `{target_type}` ({prob:.2%})'
            )


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
