from typing import List, Dict

import torch
from overrides import overrides
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.modules import Attention, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.models import SimpleSeq2Seq
from allennlp.nn import util
from allennlp.semparse.domain_languages import WikiTablesLanguage

from gensem.modules.tree_lstm import TreeLSTM


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

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                actions: List[List[ProductionRule]] = None,
                world: List[WikiTablesLanguage] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        state = self._encode(source_tokens,
                             actions,
                             world)

        if target_tokens:
            state = self._init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict

    @overrides
    def _encode(self,
                source_tokens: Dict[str, torch.Tensor],
                actions: List[List[ProductionRule]] = None,
                world: List[WikiTablesLanguage] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        if self._encode_trees:
            assert actions is not None and world is not None, \
                    "Need actions and world to be passed to forward to encode trees!"
            assert isinstance(self._encoder, TreeLSTM), "Cannot encode trees without a TreeLSTM!"
            production_rules = [[action.rule for action in instance_actions] for instance_actions in actions]
            # Assuming the "is_nonterminal" logic is the same for all the worlds.
            is_nonterminal = world[0].is_nonterminal
            encoder_outputs = self._encoder(embedded_input,
                                            source_mask,
                                            production_rules,
                                            is_nonterminal)
        else:
            encoder_outputs = self._encoder(embedded_input, source_mask)
        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
                }
