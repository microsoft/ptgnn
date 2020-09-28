from typing_extensions import Final

import logging
import torch
from torch import nn
from typing import List, Optional, Union


class MLP(nn.Module):
    LOGGER: Final = logging.getLogger(__name__)

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_layers: Union[List[int], int] = 1,
        use_biases: bool = False,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout_rate: float = 0.0,
    ):
        """
        :param input_dimension: Dimensionality of input.
        :param hidden_layers: Either an integer determining number of hidden layers, which
                will have output_dimension units each; or list of integers whose lengths
                determines the number of hidden layers and whose contents the number of units in
                each layer.
        :param output_dimension: Dimensionality of output.
        :param use_biases: Flag indicating use of bias in fully connected layers.
        :param activation: Activation function applied between hidden layers (NB: the output of the
                MLP is always the direct result of a linear transformation)
        :param dropout_rate: Dropout applied to inputs of each MLP layer.
        """
        super().__init__()
        if isinstance(hidden_layers, int):
            if output_dimension == 1:
                self.LOGGER.warning(
                    f"W: MLP was created with {hidden_layers} layers of size 1, which is most likely wrong."
                    f" Switching to {hidden_layers} layers of size 32; to get hidden layers of size 1,"
                    f" use hidden_layers=[1,...,1] explicitly.",
                )
                hidden_layer_sizes = [32] * hidden_layers
            else:
                hidden_layer_sizes = [output_dimension] * hidden_layers
        else:
            hidden_layer_sizes = hidden_layers

        if len(hidden_layer_sizes) > 1:
            assert activation is not None, "Multiple linear layers without an activation"

        layers: List[nn.Module] = []
        cur_in_dim = input_dimension
        for hidden_layer_size in hidden_layer_sizes:
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(
                nn.Linear(
                    in_features=cur_in_dim,
                    out_features=hidden_layer_size,
                    bias=use_biases,
                )
            )
            nn.init.xavier_uniform_(layers[-1].weight)
            if activation is not None:
                layers.append(activation)
            cur_in_dim = hidden_layer_size

        # Sdd the final layer, but no final activation:
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(
            nn.Linear(
                in_features=cur_in_dim,
                out_features=output_dimension,
                bias=use_biases,
            )
        )
        nn.init.xavier_uniform_(layers[-1].weight)

        self.__mlp_modules = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__mlp_modules(x)
