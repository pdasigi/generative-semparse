from typing import List

from overrides import overrides
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.modules import Attention, Embedding, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.models import SimpleSeq2Seq
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
                 encode_trees: bool = False,
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
