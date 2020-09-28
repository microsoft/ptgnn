import math
import torch
from torch import nn


class LuongAttentionModule(nn.Module):
    """
    A Luong-style attention that also includes the inner product of targets-lookup.
    """

    def __init__(
        self, memories_hidden_dimension: int, lookup_hidden_dimension: int, output_size: int
    ) -> None:
        super().__init__()
        self.__Whd = nn.Parameter(
            torch.randn(
                memories_hidden_dimension,
                lookup_hidden_dimension,
                dtype=torch.float,
                requires_grad=True,
            )
        )
        self.__Wout = nn.Linear(
            memories_hidden_dimension + lookup_hidden_dimension, output_size, bias=False
        )

    def forward(
        self, *, memories: torch.Tensor, memories_length: torch.Tensor, lookup_vectors: torch.Tensor
    ) -> torch.Tensor:
        # memories: [B, max-inp-len, H]
        # memories_length: [B]
        # look_up_vectors: [B, max-out-len, D]
        return self.forward_with_attention_vec(
            memories=memories, memories_length=memories_length, lookup_vectors=lookup_vectors
        )[0]

    def forward_with_attention_vec(
        self, *, memories: torch.Tensor, memories_length: torch.Tensor, lookup_vectors: torch.Tensor
    ):
        attention = self.get_attention_vector(
            lookup_vectors, memories, memories_length
        )  # [B, max-out-len, max-inp-len]

        contexts = torch.einsum("blq,bqh->blh", attention, memories)  # [B, max-out-len, H]
        hc = torch.cat((contexts, lookup_vectors), dim=-1)  # [B, max-out-len, H]
        return torch.tanh(self.__Wout(hc)), attention

    def get_attention_vector(
        self, lookup_vectors: torch.Tensor, memories: torch.Tensor, memories_length: torch.Tensor
    ) -> torch.Tensor:
        # memories: [B, max-inp-len, H]
        # memories_length: [B]
        # look_up_vectors: [B, max-out-len, D]
        # Output: [B, max-out-len, max-inp-len]
        memories_in_d = torch.einsum("blh,hd->bld", memories, self.__Whd)  # [B, max-inp-len, D]
        logits = torch.einsum(
            "bld,bqd->bql", memories_in_d, lookup_vectors
        )  # [B, max-out-len, max-inp-len]

        mask = (
            torch.arange(memories.shape[1], device=self.device).view(1, -1)
            >= memories_length.view(-1, 1)
        ).unsqueeze(
            1
        )  # [B, 1, max-inp-len]
        logits.masked_fill_(mask, -math.inf)
        attention = nn.functional.softmax(logits, dim=-1)  # [B, max-len]
        return attention
