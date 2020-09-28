from typing_extensions import final

from abc import ABC
from torch import nn
from typing import Any, Dict


class ModuleWithMetrics(nn.Module, ABC):
    """
    A decorator around `nn.Module` that allows a module and its ancestors to collect and report metrics, such
    as accuracy, loss, etc.

    Each Module can collect arbitrary metrics in its fields. The Module implementors should initialize
    and update the metrics as they see fit. Implementors should define
     * `_reset_module_metrics` that resets the module metrics (e.g. sets a counter to zero)
     * `_module_metrics` that reports a dictionary of metrics for this specific module.

    `ModuleWithMetrics` can recursively collect all the metrics from its children and report it
    by invoking `report_metrics()`. Metrics are automatically reset when `train()` or `eval()`
    is invoked.
    """

    def __init__(self):
        super().__init__()
        self._reset_module_metrics()

    @final
    def report_metrics(self) -> Dict[str, Any]:
        """
        Report the collected metrics for this `ModuleWithMetrics` and its descendant modules.

        Each module can internally collect its own metrics as the implementor sees fit. For example,
        a counter may be incremented when the `forward()` function is invoked or a running average may
        by updated when a loss is computed. The metrics counter can be reset outside of the module
        when `reset_metrics` is invoked.

        To add metrics to a Module, implementors need to:
        * Implement `_module_metrics` that computes the reported metrics from any component-internal variables.
        * Implement `_reset_module_metrics` which resets any variables that compute metrics.
        * Store any metric-related variables as fields in the module.
        """
        metrics = self._module_metrics()
        for child_module in self.modules():
            if isinstance(child_module, ModuleWithMetrics):
                child_metrics = child_module._module_metrics()
                if len(child_metrics) > 0:
                    metrics.update(child_metrics)
        return metrics

    @final
    def reset_metrics(self) -> None:
        """Reset any reported metrics. Often called after report_metrics() to reset any counters etc."""
        self._reset_module_metrics()
        for child_module in self.modules():
            if isinstance(child_module, ModuleWithMetrics):
                child_module._reset_module_metrics()

    def train(self, mode: bool = True) -> "ModuleWithMetrics":
        self.reset_metrics()
        return super().train(mode=mode)

    def eval(self) -> "ModuleWithMetrics":
        self.reset_metrics()
        return super().eval()

    def _module_metrics(self) -> Dict[str, Any]:
        """
        Return a dictionary of metrics for the current module.

        The key is the name of the metric as it will be appear reported.
        The value can be anything, but using a formatted string may often be the preferred choice.
        """
        return {}

    def _reset_module_metrics(self) -> None:
        """Reset any metrics related to the module, such as any counters, running sums, averages, etc."""
        pass
