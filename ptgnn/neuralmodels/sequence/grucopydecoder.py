from typing_extensions import Final

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from dpu_utils.mlutils import Vocabulary
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_log_softmax, scatter_logsumexp
from typing import Any, Counter, Dict, List, NamedTuple, Tuple, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel


class DecoderData(NamedTuple):
    input_elements: List[str]
    target_data: List[str]


class TokenizedOutput(NamedTuple):
    token_ids: List[int]
    length: int
    num_input_elements: int
    # The indices that can be copied from the input elements at each point, if any.
    copyable_elements: List[np.ndarray]


class GruCopyingDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        memories_hidden_dim: int,
        unk_id: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.__embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_size
        )
        self.__output_gru = nn.GRU(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=1, batch_first=True
        )

        self.__unk_id = unk_id

        self.__memories_to_standard_attention = nn.Linear(
            in_features=memories_hidden_dim, out_features=hidden_size, bias=False
        )
        self.__memories_to_copy_attention = nn.Linear(
            in_features=memories_hidden_dim, out_features=hidden_size, bias=False
        )
        self.__hidden_to_vocab = nn.Parameter(0.01 * torch.randn((2 * hidden_size, embedding_size)))
        self.__vocab_bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.__dropout = nn.Dropout(dropout_rate)

    def _compute_logprobs(
        self, initial_states, input_memories, input_memories_origin_idx, input_token_ids
    ):
        """
        :param input_memories: [num-inputs-flattened, D]
        :param input_memories_origin_idx: [num-inputs-flattened]
        :param initial_states: [num-targets, H]
        :param input_token_ids: [num-targets, max-seq-size-1]

        :return: the logprobs for all copying locations and the logprobs for all elements in the vocabulary
        """
        target_token_embeddings = self.__dropout(
            self.__embedding_layer(input_token_ids)
        )  # [num-targets, max-seq-size - 1, H]
        output_states, output_gru_state = self.__output_gru(
            target_token_embeddings, initial_states.unsqueeze(0)
        )
        output_states = output_states.contiguous()  # [num-targets, max-seq-size, H]

        # Standard and copy attention representations
        standard_attention_reps = self.__memories_to_standard_attention(
            input_memories
        )  # [num-inputs-flattened, H]

        copy_attention_reps = self.__memories_to_copy_attention(
            input_memories
        )  # [num-inputs-flattened, H]
        copy_attention_reps = self.__dropout(copy_attention_reps)

        # Compute attention scores [num-inputs-flattened, max-seq-size]
        output_states_per_input = output_states[
            input_memories_origin_idx
        ]  # [num-inputs-flattened, max-seq-size, H]
        standard_attention_scores = torch.einsum(
            "ilh,ih->il", output_states_per_input, standard_attention_reps
        )  # [num-inputs-flattened, max-seq-size]
        copy_attention_scores = torch.einsum(
            "ilh,ih->il", output_states_per_input, copy_attention_reps
        )  # [num-inputs-flattened, max-seq-size]

        # For some reason scatter_softmax doesn't work here
        standard_attention_logprobs = scatter_log_softmax(
            standard_attention_scores, index=input_memories_origin_idx, dim=0, eps=0
        )  # [num-inputs-flattened, max-seq-size]

        # Compute standard attention output
        standard_attention_mul = torch.einsum(
            "il,ih->ilh", torch.exp(standard_attention_logprobs), standard_attention_reps
        )
        standard_attention_out = scatter_add(
            standard_attention_mul, index=input_memories_origin_idx, dim=0
        )  # [num-targets, max-seq-size, H]
        target_scores = (
            torch.einsum(
                "blh,hd,vd->blv",
                torch.cat((self.__dropout(standard_attention_out), output_states), dim=-1),
                self.__hidden_to_vocab,
                self.__dropout(self.__embedding_layer.weight),
            )
            + self.__vocab_bias
        )  # [num-targets, max-seq-size, vocab-size]

        # A "manual" log_softmax
        total_copy_scores = scatter_logsumexp(
            copy_attention_scores, index=input_memories_origin_idx, dim=0, eps=0
        )  # [num-targets, max-seq-size]
        all_scores = torch.cat(
            (target_scores, total_copy_scores.unsqueeze(-1)), dim=-1
        )  # [num-targets, max-seq-size, vocab-size + 1]
        normalizing_const = torch.logsumexp(all_scores, dim=-1)  # [num-targets, max-seq-size]

        target_logprobs = target_scores - normalizing_const.unsqueeze(
            -1
        )  # [num-targets, max-seq-size, vocab-size]
        copy_logprobs = (
            copy_attention_scores - normalizing_const[input_memories_origin_idx]
        )  # [num-inputs-flattened, max-seq-size]

        # Do probabilities actually sum up to 1?
        # sum_logcopy = scatter_logsumexp(copy_logprobs, index=input_memories_origin_idx, dim=0, eps=0) # [num-targets, max-seq-size]
        # sum_targets = torch.logsumexp(target_logprobs, dim=-1)                          # [num-targets, max-seq-size]
        # torch.allclose(torch.logsumexp(torch.stack([sum_logcopy, sum_targets], dim=-1), dim=-1), torch.zeros_like(sum_logcopy), atol=.0001)

        return copy_logprobs, target_logprobs, output_gru_state

    def forward(
        self,
        *,
        input_memories,
        input_memories_origin_idx,
        initial_states,
        target_token_ids,
        copyable_elements_idxs,
        copyable_elements_sample_idxs,
        target_lengths
    ):
        """
        :param input_memories: [num-inputs-flattened, D]
        :param input_memories_origin_idx: [num-inputs-flattened]
        :param initial_states: [num-targets, H]
        :param target_token_ids: [num-targets, max-seq-size]
        :param target_lengths: [num-targets]
        :param copyable_elements_idxs: [num-copyable-elements]
        :param copyable_elements_sample_idxs: [num-copyable-elements]

        :return: the loss function.
        """
        copy_logprobs, target_logprobs, _ = self._compute_logprobs(
            initial_states, input_memories, input_memories_origin_idx, target_token_ids[:, :-1]
        )

        # Get loss. UNKs are only predicted if we cannot copy.
        num_valid_copy_actions = scatter_add(
            src=torch.ones_like(copyable_elements_sample_idxs),
            index=copyable_elements_sample_idxs,
            dim_size=target_token_ids.shape[0] * (target_token_ids.shape[1] - 1),
        )
        locations_with_valid_copy_actions = (
            num_valid_copy_actions.reshape(target_token_ids.shape[0], target_token_ids.shape[1] - 1)
            > 0
        )
        unk_prediction_locations = target_token_ids[:, 1:] == self.__unk_id
        mask = locations_with_valid_copy_actions & unk_prediction_locations

        correct_generation_logprobs = torch.gather(
            target_logprobs, index=target_token_ids[:, 1:].unsqueeze(-1), dim=-1
        ).squeeze(
            -1
        )  # [num-targets, max-seq-size -1]
        correct_generation_logprobs.masked_fill_(mask, -math.inf)

        correct_copy_logprobs = scatter_logsumexp(
            src=copy_logprobs.flatten()[copyable_elements_idxs],
            index=copyable_elements_sample_idxs,
            dim=0,
            dim_size=target_token_ids.shape[0] * (target_token_ids.shape[1] - 1),
            eps=0,
        )  # [num-targets * (max-seq-size-1)]
        correct_copy_logprobs = correct_copy_logprobs.view(
            target_token_ids.shape[0], target_token_ids.shape[1] - 1
        )

        any_correct_action_logprob = torch.logsumexp(
            torch.stack((correct_generation_logprobs, correct_copy_logprobs)), dim=0
        )  # [num-targets, max-seq-size]

        mask = torch.arange(
            any_correct_action_logprob.shape[1], device=target_lengths.device
        ).unsqueeze(0) < target_lengths.unsqueeze(1)
        per_seq_loss = (any_correct_action_logprob * mask.float()).sum(dim=-1) / mask.float().sum(
            dim=-1
        )

        return -per_seq_loss.mean()


class GruCopyingDecoderModel(AbstractNeuralModel[DecoderData, TokenizedOutput, GruCopyingDecoder]):
    """A GRU copying decoder that accepts an variable-sized set of encodings."""

    LOGGER: Final = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        max_seq_len: int = 8,
        hidden_size=128,
        embedding_size: int = 256,
        memories_hidden_dim: int = 128,
        vocabulary_max_size: int = 20000,
        vocabulary_count_threshold: int = 5,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.max_seq_len: Final = max_seq_len
        self.hidden_size: Final = hidden_size
        self.embedding_size: Final = embedding_size
        self.memories_hidden_dim: Final = memories_hidden_dim
        self.vocabulary_max_size: Final = vocabulary_max_size
        self.vocabulary_count_threshold: Final = vocabulary_count_threshold
        self.dropout_rate: Final = dropout_rate

    @property
    def END(self) -> str:
        return "%END%"

    @property
    def START(self) -> str:
        return "%START%"

    # region Metadata
    def initialize_metadata(self) -> None:
        self.__token_counter = Counter[str]()

    def update_metadata_from(self, datapoint: DecoderData) -> None:
        self.__token_counter.update(datapoint.target_data)

    def finalize_metadata(self) -> None:
        self.__token_counter[self.START] = 1000000
        self.__token_counter[self.END] = 1000000
        self.__output_vocabulary = Vocabulary.create_vocabulary(
            self.__token_counter,
            max_size=self.vocabulary_max_size,
            count_threshold=self.vocabulary_count_threshold,
        )
        self.LOGGER.info("Output vocabulary Size %s", len(self.__output_vocabulary))
        del self.__token_counter

    def build_neural_module(self) -> GruCopyingDecoder:
        return GruCopyingDecoder(
            vocabulary_size=len(self.__output_vocabulary),
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            memories_hidden_dim=self.memories_hidden_dim,
            unk_id=self.__output_vocabulary.get_id_or_unk(self.__output_vocabulary.get_unk()),
            dropout_rate=self.dropout_rate,
        )

    # endregion

    def tensorize(self, datapoint: DecoderData) -> TokenizedOutput:
        max_seq_len = self.max_seq_len
        target_with_start_end = [self.START] + datapoint.target_data + [self.END]
        target_with_start_end = target_with_start_end[:max_seq_len]

        seq_len = min(len(target_with_start_end), max_seq_len)

        return TokenizedOutput(
            token_ids=self.__output_vocabulary.get_id_or_unk_multiple(target_with_start_end),
            length=seq_len,
            num_input_elements=len(datapoint.input_elements),
            # Which elements can be copied at this step?
            copyable_elements=[
                np.array(
                    [
                        i
                        for i, input_element in enumerate(datapoint.input_elements)
                        if input_element == target_token
                    ],
                    dtype=np.int32,
                )
                for target_token in target_with_start_end[1:]
            ],
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "target_token_ids": [],
            "target_seq_lengths": [],
            # The indices of the input elements that can be copied for each timestep in range(max-seq-len-1)
            "copyable_elements_idxs": [],
            "num_input_elements": [],
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: TokenizedOutput, partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["target_token_ids"].append(tensorized_datapoint.token_ids)
        partial_minibatch["target_seq_lengths"].append(tensorized_datapoint.length)
        partial_minibatch["copyable_elements_idxs"].append(tensorized_datapoint.copyable_elements)
        partial_minibatch["num_input_elements"].append(tensorized_datapoint.num_input_elements)
        return True

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        max_seq_length = max(accumulated_minibatch_data["target_seq_lengths"])
        num_elements = len(accumulated_minibatch_data["target_token_ids"])

        target_token_ids = np.zeros((num_elements, max_seq_length), dtype=np.int32)

        # These need to index into index into [num-inputs-flattened, max-seq-size - 1] memories
        copyable_elements_idxs: List[int] = []
        # Each group of elements is one entry of a [num-target, max-seq-size - 1]
        copyable_elements_sample_idxs: List[int] = []

        # Count how many input sequences have been seen up to each point of the iteration
        offset_num_input_elements_so_far = 0
        for sample_idx, (token_ids, copyable_elements, num_input_elements_for_sample) in enumerate(
            zip(
                accumulated_minibatch_data["target_token_ids"],
                accumulated_minibatch_data["copyable_elements_idxs"],
                accumulated_minibatch_data["num_input_elements"],
            )
        ):
            target_token_ids[sample_idx, : len(token_ids)] = token_ids

            sample_row_starting_idx = sample_idx * (max_seq_length - 1)
            for timestep_idx, copyable_elements_at_timestep in enumerate(copyable_elements):
                flattened_index = (
                    sample_row_starting_idx + timestep_idx
                )  # The -1 is here since the flattened output does not have a prediction for the last element
                # Confusingly this is indexed by timestep. So (after the offset) we have a flattened out [num-inputs-flattened, max-seq-size-1]
                copyable_elements_idxs.extend(
                    offset_num_input_elements_so_far
                    + copyable_elements_at_timestep * (max_seq_length - 1)
                    + timestep_idx
                )
                copyable_elements_sample_idxs.extend(
                    flattened_index for _ in range(len(copyable_elements_at_timestep))
                )
                # Add one full sequence of inputs here
            offset_num_input_elements_so_far += num_input_elements_for_sample * (max_seq_length - 1)

        return {
            "target_token_ids": torch.tensor(target_token_ids, dtype=torch.int64, device=device),
            "copyable_elements_idxs": torch.tensor(
                copyable_elements_idxs, dtype=torch.int64, device=device
            ),
            "copyable_elements_sample_idxs": torch.tensor(
                copyable_elements_sample_idxs, dtype=torch.int64, device=device
            ),
            "target_lengths": torch.tensor(
                accumulated_minibatch_data["target_seq_lengths"], dtype=torch.int64, device=device
            ),
        }

    def greedy_decode(
        self,
        *,
        input_concrete_values: List[str],
        input_memories,
        input_memories_origin_idx,
        initial_states,
        neural_model: GruCopyingDecoder
    ) -> List[Tuple[List[str], float]]:
        output_vocab = self.__output_vocabulary
        batch_size = initial_states.shape[0]
        assert len(input_concrete_values) == input_memories.shape[0]

        current_decoder_states = initial_states  # [num-targets, H]
        next_token_sequences = torch.tensor(
            [[output_vocab.get_id_or_unk(self.START)]] * batch_size, device=input_memories.device
        )  # [num-targets x 1]

        predicted_tokens: List[List[str]] = [[] for _ in range(batch_size)]
        predicted_logprobs: List[float] = [0.0 for _ in range(batch_size)]
        sample_id_done = np.zeros(batch_size, dtype=np.bool)

        for i in range(self.max_seq_len):
            copy_logprobs, target_logprobs, output_decoder_state = neural_model._compute_logprobs(
                current_decoder_states,
                input_memories,
                input_memories_origin_idx,
                next_token_sequences,
            )
            current_decoder_states = output_decoder_state.squeeze(0)  # [num-targets, H]

            # To speed things up, look up the top 100 vocab tokens
            topk_vocab_logprobs, topk_vocab_idxs = torch.topk(
                target_logprobs.squeeze(1), min(100, target_logprobs.shape[-1]), dim=-1
            )  # [num-targets, 100], [num-targets, 100]

            # Move to CPU and show as numpy
            topk_vocab_logprobs = topk_vocab_logprobs.cpu().numpy()
            topk_vocab_idxs = topk_vocab_idxs.cpu().numpy()
            copy_logprobs = copy_logprobs.squeeze(1).cpu().numpy()

            target_token_predictions = [
                {
                    output_vocab.get_name_for_id(int(token_idx)): token_logprob
                    for token_idx, token_logprob in zip(
                        topk_vocab_idxs[batch_idx], topk_vocab_logprobs[batch_idx]
                    )
                }
                for batch_idx in range(batch_size)
            ]  # for each element in the batch, the logprobs of the target predictions

            for batch_idx, concrete_value, copy_logprob in zip(
                input_memories_origin_idx.cpu().numpy(), input_concrete_values, copy_logprobs
            ):
                predictions = target_token_predictions[batch_idx]
                predictions[concrete_value] = np.logaddexp(
                    predictions.get(concrete_value, -math.inf), copy_logprob
                )

            predicted_tokens_for_this_step = []
            for batch_idx, predictions in enumerate(target_token_predictions):
                if sample_id_done[batch_idx]:
                    predicted_tokens_for_this_step.append(self.END)
                    continue

                predicted_token, predicted_logprob = max(
                    target_token_predictions[batch_idx].items(), key=lambda x: x[1]
                )

                if predicted_token == self.END:
                    sample_id_done[batch_idx] = True
                else:
                    predicted_tokens[batch_idx].append(predicted_token)

                predicted_tokens_for_this_step.append(predicted_token)
                predicted_logprobs[batch_idx] += predicted_logprob

            next_token_sequences = torch.tensor(
                [[output_vocab.get_id_or_unk(t)] for t in predicted_tokens_for_this_step],
                device=input_memories.device,
            )

        return list(zip(predicted_tokens, predicted_logprobs))
