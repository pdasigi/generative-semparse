from typing import List, Dict

import torch
from overrides import overrides
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.modules import Attention, Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.models import SimpleSeq2Seq
from allennlp.nn import util
from allennlp.semparse.domain_languages import WikiTablesLanguage


@Model.register('wtq-question-generator')
class WikiTablesQuestionGenerator(SimpleSeq2Seq):
    """
    Simple encoder decoder model that encodes a logical form using a tree LSTM and greedily decodes the utterance
    using an LSTM with attention over encoder outputs.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 encode_trees: bool = True,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True) -> None:
        super().__init__(vocab=vocab,
                         source_embedder=source_embedder,
                         encoder=encoder,
                         max_decoding_steps=max_decoding_steps,
                         attention=attention,
                         attention_function=attention_function,
                         beam_size=beam_size,
                         target_namespace=target_namespace,
                         target_embedding_dim=target_embedding_dim,
                         scheduled_sampling_ratio=scheduled_sampling_ratio,
                         use_bleu=use_bleu)
        self._encode_trees = encode_trees
        # We'll encode multiple logical forms per instance. So we're time-distributing the encoder here.
        self._time_distributed_encoder = TimeDistributed(self._encoder)

    @overrides
    def forward(self,  # type: ignore
                action_sequences: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                actions: List[List[List[ProductionRule]]] = None,
                world: List[WikiTablesLanguage] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        if self._encode_trees:
            assert actions is not None and world is not None, \
                    "Need actions and world to be passed to forward to encode trees!"
        # TODO (pradeep): It might be more efficient to also have the decoder process multiple input sequences
        # instead of having ``_encode`` return multiple states, and looping over them below.
        states = self._encode(action_sequences,
                              actions,
                              world)

        candidate_outputs: List[Dict[str, torch.Tensor]] = []
        for state in states:
            if target_tokens:
                state = self._init_decoder_state(state)
                # The `_forward_loop` decodes the input sequence and computes the loss during training
                # and validation.
                candidate_output_dict = self._forward_loop(state, target_tokens)
            else:
                candidate_output_dict = {}

            if not self.training:
                state = self._init_decoder_state(state)
                predictions = self._forward_beam_search(state)
                candidate_output_dict.update(predictions)
            candidate_outputs.append(candidate_output_dict)

        output_dict = self._merge_output_dicts(candidate_outputs)
        if not self.training and target_tokens and self._bleu:
            # shape: (batch_size, beam_size, max_sequence_length)
            top_k_predictions = output_dict["predictions"]
            # shape: (batch_size, max_predicted_sequence_length)
            best_predictions = top_k_predictions[:, 0, :]
            self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict

    @overrides
    def _encode(self,
                action_sequences: Dict[str, torch.Tensor],
                actions: List[List[List[ProductionRule]]] = None,
                world: List[WikiTablesLanguage] = None) -> List[Dict[str, torch.Tensor]]:
        # pylint: disable=arguments-differ
        # shape: (batch_size, max_num_inputs, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(action_sequences)
        # shape: (batch_size, max_num_inputs, max_input_sequence_length)
        source_mask = util.get_text_field_mask(action_sequences, num_wrapping_dims=1)
        if self._encode_trees:
            # ``production_rules`` is just a list of strings, and not a tensor. So we'll "time-distribute" it here
            # itself unlike the embedded input and mask, while will be time-distributed in the forward method of
            # ``TimeDistribute``. We'll pass-through ``production_rules`` below.
            # (batch_size * max_num_inputs, max_input_sequence_length)
            production_rules: List[List[str]] = []
            for instance_actions in actions:
                for instance_actions_candidate in instance_actions:
                    production_rules.append([action.rule for action in instance_actions_candidate])
            # Assuming the "is_nonterminal" logic is the same for all the worlds.
            is_nonterminal = world[0].is_nonterminal
            # shape: (batch_size, max_num_inputs, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = self._time_distributed_encoder(inputs=embedded_input,
                                                             mask=source_mask,
                                                             production_rules=production_rules,
                                                             is_nonterminal=is_nonterminal,
                                                             pass_through=['production_rules', 'is_nonterminal'])
        else:
            # shape: (batch_size, max_num_inputs, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = self._time_distributed_encoder(embedded_input, source_mask)

        # We return one state per action sequence candidate for each instance because the decoder cannot process
        # multiple input sequences per instance yet.
        permuted_source_mask = source_mask.permute(1, 0, 2)  # (max_num_inputs, batch_size, seq_length)
        permuted_outputs = encoder_outputs.permute(1, 0, 2, 3)  # (max_num_inputs, batch_size, seq_len, output_dim)
        # list of length max_num_inputs
        return [{"source_mask": mask, "encoder_outputs": outputs}
                for mask, outputs in zip(permuted_source_mask, permuted_outputs)]

    @overrides
    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Overriding this method only to use our own ``get_final_encoder_states`` that allows entire elements to be
        # masked.
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = self._get_final_encoder_states(
                state["encoder_outputs"],
                state["source_mask"],
                self._encoder.is_bidirectional())
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self._decoder_output_dim)
        return state

    @staticmethod
    def _merge_output_dicts(candidate_output_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # TODO (pradeep): These losses are batch averaged. Is that a problem?
        # (max_num_inputs,)
        losses = torch.stack([output["loss"] for output in candidate_output_dicts])
        # Losses are negative log-likelihoods. The final loss we need is be the negative log of sum of all
        # likelihoods.
        output_dict = {"loss": -util.logsumexp(-losses)}
        if "class_log_probabilities" in candidate_output_dicts[0]:
            # This means we have an k-best list of sequences.
            # (batch_size, max_num_inputs * k)
            log_probabilities = torch.cat([output["class_log_probabilities"]
                                           for output in candidate_output_dicts], dim=-1)
            # (batch_size, max_num_inputs * k, sequence_length)
            predictions = torch.cat([output["predictions"] for output in candidate_output_dicts],
                                    dim=1)
            sorted_log_probabilities, indices = torch.sort(log_probabilities, descending=True)
            _, _, sequence_length = predictions.size()
            # (batch_size, max_num_inputs * k, sequence_length)
            indices_for_selection = indices.unsqueeze(-1).repeat_interleave(sequence_length, dim=2)
            sorted_predictions = predictions.gather(1, indices_for_selection)
            output_dict["class_log_probabilities"] = sorted_log_probabilities
            output_dict["predictions"] = sorted_predictions
        return output_dict

    @staticmethod
    def _get_final_encoder_states(encoder_outputs: torch.Tensor,
                                  mask: torch.Tensor,
                                  bidirectional: bool = False) -> torch.Tensor:
        # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
        # are assuming sequences are right padded.
        # Shape: (batch_size,)
        last_word_indices = mask.sum(1).long() - 1
        # If an entire sequence is masked, one of the values in the above tensor will be -1. In that case, we
        # simply return the output at the first word, assuming that the mask will be used later to ignore this
        # output. This is the only difference between this method and the one in allennlp.nn.util.
        last_word_indices = last_word_indices * (last_word_indices > 0).long()
        batch_size, _, encoder_output_dim = encoder_outputs.size()
        expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
        # Shape: (batch_size, 1, encoder_output_dim)
        final_encoder_output = encoder_outputs.gather(1, expanded_indices)
        final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
        if bidirectional:
            final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
            final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
            final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
        return final_encoder_output
