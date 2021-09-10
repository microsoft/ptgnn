from typing_extensions import Final

import logging
import torch
from contextlib import contextmanager

LOGGER: Final = logging.getLogger(__name__)


@contextmanager
def catch_cuda_oom(enabled: bool = True):
    if enabled:
        try:
            yield
        except RuntimeError as re:
            if "CUDA out of memory." in repr(re):
                LOGGER.exception("CUDA Out-Of-Memory Caught and Execution Resumed.", exc_info=re)
                torch.cuda.empty_cache()
            else:
                raise re
    else:
        yield
