import torch
from torch import nn
from torch_scatter import scatter
from typing import Dict, List, Optional, Tuple, Union

from ptgnn.neuralmodels.gnn.messagepassing.abstractmessagepassing import (
    AbstractMessageAggregation,
    AbstractMessagePassingLayer,
)
from ptgnn.neuralmodels.mlp import MLP


class PnaMessageAggregation(AbstractMessageAggregation):
    """
    Principal Neighbourhood Aggregation for Graph Nets

    https://arxiv.org/abs/2004.05718
    """

    def __init__(
        self,
        delta: float = 1,
    ):
        super().__init__()
        self._delta = delta  # See Eq 5 of paper

    def forward(self, messages: torch.Tensor, message_targets: torch.Tensor, num_nodes):
        degree = scatter(
            torch.ones_like(message_targets),
            index=message_targets,
            dim_size=num_nodes,
            reduce="sum",
        )

        msg_dtype = messages.dtype
        messages = messages.to(torch.float32)
        sum_agg = scatter(messages, index=message_targets, dim=0, dim_size=num_nodes, reduce="sum")
        mean_agg = sum_agg / (degree.unsqueeze(-1) + 1e-5)
        max_agg = scatter(messages, index=message_targets, dim=0, dim_size=num_nodes, reduce="max")
        min_agg = scatter(messages, index=message_targets, dim=0, dim_size=num_nodes, reduce="min")

        std_components = torch.relu(messages.pow(2) - mean_agg[message_targets].pow(2)) + 1e-10
        std = torch.sqrt(
            scatter(std_components, index=message_targets, dim=0, dim_size=num_nodes, reduce="sum")
        )

        all_aggregations = torch.cat([sum_agg, mean_agg, max_agg, min_agg, std], dim=-1).to(
            msg_dtype
        )

        scaler_p1 = torch.log(degree.float() + 1).unsqueeze(-1) / self._delta
        scaler_m1 = 1 / (scaler_p1 + 1e-3)

        return torch.cat(
            [all_aggregations, all_aggregations * scaler_p1, all_aggregations * scaler_m1], dim=-1
        )

    def output_state_size(self, message_input_size: int) -> int:
        return message_input_size * 5 * 3
