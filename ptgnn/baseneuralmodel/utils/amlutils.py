import logging
import os
import sys
from typing import Any, Dict


def configure_logging(aml_ctx) -> str:
    os.makedirs("logs", exist_ok=True)

    base_logger = logging.getLogger()
    base_logger.setLevel(logging.INFO)

    log_path = os.path.join("logs", "full.log")
    formatter = logging.Formatter(
        "%(asctime)s [%(name)-35.35s @ %(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    base_logger.addHandler(file_handler)

    if aml_ctx is None:
        # AML seems to be adding its own logger behind the scenes, no need to add another one.
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        base_logger.addHandler(stream_handler)

    return log_path


def log_run(aml_ctx, fold_name: str, model, epoch_idx: int, metrics: Dict[str, Any]) -> None:
    if aml_ctx is None:
        return
    for metric_name, metric_value in metrics.items():
        aml_ctx.log(f"{fold_name}-{metric_name}", metric_value)
