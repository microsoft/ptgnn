#!/usr/bin/env python
"""
Usage:
    trainandtest.py [options] TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME

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
from dpu_utils.utils import run_and_debug

from ptgnn.implementations.graph2seq import test, train

if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: train.run(args), args.get("--debug", False))
    run_and_debug(lambda: test.run(args), args.get("--debug", False))
