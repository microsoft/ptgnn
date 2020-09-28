from typing_extensions import Final, Literal

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import BpeVocabulary, CharTensorizer, Vocabulary
from typing import Any, Counter, Dict, List, NamedTuple, Optional, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.neuralmodels.gnn.structs import AbstractNodeEmbedder


class TokenUnitEmbedder(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, dropout_rate: float):
        super().__init__()
        self.__embeddings = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_size
        )
        nn.init.xavier_uniform_(self.__embeddings.weight)  # TODO: Reconsider later?
        self.__dropout_layer = nn.Dropout(p=dropout_rate)

    @property
    def embedding_layer(self) -> nn.Embedding:
        return self.__embeddings

    def forward(self, token_idxs: torch.Tensor) -> torch.Tensor:
        return self.__dropout_layer(self.__embeddings(token_idxs))  # [B, D]


class SubtokenUnitEmbedder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        dropout_rate: float,
        subtoken_combination_kind: str,
        use_dense_output: bool = True,
    ):
        super().__init__()
        assert subtoken_combination_kind in {"mean", "max", "sum"}
        self.__subtoken_combination_kind = subtoken_combination_kind
        self.__embeddings = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_size
        )
        nn.init.uniform_(self.__embeddings.weight)
        if use_dense_output:
            self.__out_layer = nn.Linear(embedding_size, embedding_size, bias=False)
            nn.init.xavier_uniform_(self.__out_layer.weight)
        else:
            self.__out_layer = None

        self.__dropout_layer = nn.Dropout(p=dropout_rate)

    @property
    def embedding_layer(self) -> nn.Embedding:
        return self.__embeddings

    def forward(self, token_idxs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        :param token_idxs: The subtoken ids in a [B, max_num_subtokens] matrix.
        :param lengths: A [B]-sized vector containing the lengths
        :return: a [B, D] matrix of D-sized representations, one per input example.
        """
        embedded = self.__embeddings(token_idxs)  # [B, max_num_subtokens, D]
        mask = torch.arange(embedded.shape[1], device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(
            -1
        )  # [B, max_num_subtokens]

        if self.__subtoken_combination_kind == "mean":
            embedded = embedded * mask.unsqueeze(-1).float()
            embedded = embedded.sum(dim=-2) / (lengths.unsqueeze(-1).float() + 1e-10)  # [B, D]
        elif self.__subtoken_combination_kind == "sum":
            embedded = embedded * mask.unsqueeze(-1).float()
            embedded = embedded.sum(dim=-2)  # [B, D]
        elif self.__subtoken_combination_kind == "max":
            embedded.masked_fill_(mask=~mask.unsqueeze(-1), value=-math.inf)
            embedded, _ = embedded.max(dim=-2)  # [B, D]
        else:
            raise ValueError(
                f'Unrecognized subtoken combination "{self.__subtoken_combination_kind}".'
            )
        if self.__out_layer is not None:
            embedded = self.__out_layer(embedded)
        return self.__dropout_layer(embedded)


class CnnConfig(NamedTuple):
    l1_filters: int
    l1_window_size: int
    l2_filters: int
    l2_window_size: int
    lout_window_size: int


class CharUnitEmbedder(nn.Module):
    def __init__(
        self,
        num_chars: int,
        embedding_size: int,
        config: CnnConfig,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.__num_chars_in_vocabulary = num_chars
        self.__conv_l1 = nn.Conv1d(
            in_channels=num_chars, out_channels=config.l1_filters, kernel_size=config.l1_window_size
        )

        self.__conv_l2 = nn.Conv1d(
            in_channels=config.l1_filters,
            out_channels=config.l2_filters,
            kernel_size=config.l2_window_size,
        )

        self.__conv_l3 = nn.Conv1d(
            in_channels=config.l2_filters,
            out_channels=embedding_size,
            kernel_size=config.lout_window_size,
            bias=False,
        )
        self.__dropout = nn.Dropout(p=dropout_rate)

    def forward(self, chars):
        """
        :param chars: [B, max_num_chars]
        :return: [B, D]
        """
        cnn_input = nn.functional.one_hot(
            chars, self.__num_chars_in_vocabulary
        )  # [B , max_num_chars, char_vocab_size]
        cnn_input = cnn_input.transpose(1, 2).float()
        l1_out = self.__conv_l1(cnn_input)
        l2_out = self.__conv_l2(nn.functional.relu(l1_out))
        l3_out = self.__conv_l3(nn.functional.relu(l2_out))  # [B, Dout, max_num_chars - pad]

        summary, _ = torch.max(l3_out, dim=-1)  # [B, D']
        return self.__dropout(summary)


class StrElementRepresentationModel(
    AbstractNeuralModel[str, Any, Union[TokenUnitEmbedder, SubtokenUnitEmbedder, CharUnitEmbedder]],
    AbstractNodeEmbedder,
):
    """
    A model that accepts strings and returns a single representation (embedding) for each one of them.
    """

    def __init__(
        self,
        *,
        token_splitting: Literal["token", "subtoken", "bpe", "char"],
        embedding_size: int = 128,
        dropout_rate: float = 0.2,
        # Vocabulary Options
        vocabulary_size: int = 10000,
        min_freq_threshold: int = 5,
        # BPE/Subtoken Options
        max_num_subtokens: Optional[int] = 5,
        subtoken_combination: Literal["sum", "mean", "max"] = "sum",
        # Char Models
        cnn_config: CnnConfig = CnnConfig(
            l1_filters=256, l1_window_size=3, l2_filters=128, l2_window_size=3, lout_window_size=3
        ),
        max_num_chars: int = 15,
    ):
        super().__init__()
        self._splitting_kind: Final = token_splitting
        self.embedding_size: Final = embedding_size
        self.dropout_rate: Final = dropout_rate
        self.__vocabulary: Union[Vocabulary, BpeVocabulary, CharTensorizer]
        if token_splitting in {"bpe", "subtoken"}:
            self.max_num_subtokens: Final = max_num_subtokens
            self.subtoken_combination: Final = subtoken_combination
        elif token_splitting == "char":
            self.cnn_config: Final = cnn_config
            self.max_num_chars: Final = max_num_chars
        if token_splitting != "char":
            self.max_vocabulary_size: Final = vocabulary_size
            self.min_freq_threshold: Final = min_freq_threshold

    LOGGER: Final = logging.getLogger(__name__)

    def representation_size(self) -> int:
        return self.embedding_size

    @property
    def splitting_kind(self) -> Literal["token", "subtoken", "bpe", "char"]:
        return self._splitting_kind

    # region Metadata Loading
    def initialize_metadata(self) -> None:
        self.__tok_counter = Counter[str]()

    def update_metadata_from(self, datapoint: str) -> None:
        if self.splitting_kind in {"token", "bpe"}:
            self.__tok_counter[datapoint] += 1
        elif self.splitting_kind == "subtoken":
            self.__tok_counter.update(split_identifier_into_parts(datapoint))
        elif self.splitting_kind == "char":
            pass
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}".')

    def finalize_metadata(self) -> None:
        if self.splitting_kind in {"token", "subtoken"}:
            self.__vocabulary = Vocabulary.create_vocabulary(
                self.__tok_counter,
                max_size=self.max_vocabulary_size,
                count_threshold=self.min_freq_threshold,
            )
        elif self.splitting_kind == "bpe":
            self.__vocabulary = BpeVocabulary(self.max_vocabulary_size)
            self.__vocabulary.create_vocabulary(self.__tok_counter)
        elif self.splitting_kind == "char":
            self.__vocabulary = CharTensorizer(
                max_num_chars=self.max_num_chars, lower_case_all=False, include_space=False
            )
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}"')

        del self.__tok_counter

    def build_neural_module(self) -> Union[TokenUnitEmbedder, SubtokenUnitEmbedder]:
        if self.splitting_kind == "token":
            vocabulary_size = len(self.vocabulary)
            embedding_size = self.embedding_size
            return TokenUnitEmbedder(vocabulary_size, embedding_size, self.dropout_rate)
        elif self.splitting_kind in {"bpe", "subtoken"}:
            vocabulary_size = len(self.vocabulary)
            embedding_size = self.embedding_size
            return SubtokenUnitEmbedder(
                vocabulary_size,
                embedding_size,
                self.dropout_rate,
                self.subtoken_combination,
            )
        elif self.splitting_kind == "char":
            return CharUnitEmbedder(
                num_chars=self.vocabulary.num_chars_in_vocabulary(),
                embedding_size=self.embedding_size,
                config=self.cnn_config,
                dropout_rate=self.dropout_rate,
            )
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}"')

    @property
    def vocabulary(self) -> Union[Vocabulary, BpeVocabulary, CharTensorizer]:
        return self.__vocabulary

    # endregion

    # region Tensorization
    def tensorize(self, datapoint: str, return_str_rep: bool = False):
        if self.splitting_kind == "token":
            token_idxs = self.vocabulary.get_id_or_unk(datapoint)
            str_repr = datapoint
        elif self.splitting_kind == "subtoken":
            subtoks = split_identifier_into_parts(datapoint)
            if len(subtoks) == 0:
                subtoks = [Vocabulary.get_unk()]
            token_idxs = self.vocabulary.get_id_or_unk_multiple(subtoks)
        elif self.splitting_kind == "bpe":
            if len(datapoint) == 0:
                datapoint = "<empty>"
            token_idxs = self.vocabulary.get_id_or_unk_for_text(datapoint)
            if return_str_rep:  # Do _not_ compute for efficiency
                str_repr = self.vocabulary.tokenize(datapoint)
        elif self.splitting_kind == "char":
            token_idxs = self.vocabulary.tensorize_str(datapoint)
            if return_str_rep:
                str_repr = datapoint[: self.vocabulary.max_char_length]
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}".')

        if return_str_rep:
            return token_idxs, str_repr
        return token_idxs

    # endregion

    # region Minibatching
    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"token_idxs": []}

    def extend_minibatch_with(
        self, tensorized_datapoint: Union[int, List[int]], partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["token_idxs"].append(tensorized_datapoint)
        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if self.splitting_kind == "token":
            return {
                "token_idxs": torch.tensor(
                    accumulated_minibatch_data["token_idxs"], dtype=torch.int64, device=device
                ),
            }
        elif self.splitting_kind in {"subtoken", "bpe"}:
            max_num_subtokens = max(len(t) for t in accumulated_minibatch_data["token_idxs"])
            if self.max_num_subtokens is not None:
                max_num_subtokens = min(max_num_subtokens, self.max_num_subtokens)

            subtoken_idxs = np.zeros(
                (len(accumulated_minibatch_data["token_idxs"]), max_num_subtokens), dtype=np.int32
            )
            lengths = np.empty(len(accumulated_minibatch_data["token_idxs"]), dtype=np.int32)
            for i, subtokens in enumerate(accumulated_minibatch_data["token_idxs"]):
                idxs = subtokens[:max_num_subtokens]
                subtoken_idxs[i, : len(idxs)] = idxs
                lengths[i] = len(idxs)

            return {
                "token_idxs": torch.tensor(subtoken_idxs, dtype=torch.int64, device=device),
                "lengths": torch.tensor(lengths, dtype=torch.int64, device=device),
            }
        elif self.splitting_kind == "char":
            return {
                "chars": torch.tensor(
                    np.stack(accumulated_minibatch_data["token_idxs"], axis=0),
                    dtype=torch.int64,
                    device=device,
                )
            }
        else:
            raise Exception("Non-reachable state.")

    # endregion
