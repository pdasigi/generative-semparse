from typing import List, Tuple, Callable

from overrides import overrides
import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from gensem.modules.rnn_grammar_state import RnnGrammarState


@Seq2SeqEncoder.register('tree_lstm')
class TreeLSTM(Seq2SeqEncoder):
    """
    Bottom-up tree encoder that can process a variable number of children per node. We deal with varying number of
    children by aggregaring their representations using a linear GRU.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self._lstm_cell = torch.nn.LSTMCell(input_dim, output_dim)
        # Context for leaf node
        self._leaf_hidden_state = torch.nn.Parameter(torch.Tensor(output_dim))
        self._leaf_memory_cell = torch.nn.Parameter(torch.Tensor(output_dim))
        self._child_representation_aggregator = torch.nn.GRU(output_dim, output_dim)

    @overrides
    def get_input_dim(self) -> int:
        return self._lstm_cell.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self._lstm_cell.hidden_size

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    def _aggregate_child_states(self, child_rnn_states: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
        # (num_children, 1, output_dim) because gru input needs a batch size dimension as well.
        child_hidden_states = torch.stack([rnn_state[0]
                                           for rnn_state in child_rnn_states]).unsqueeze(1)
        # (1, 1, output_dim)
        _, aggregated_hidden_state = self._child_representation_aggregator(child_hidden_states)
        # (1, output_dim)
        aggregated_hidden_state = aggregated_hidden_state.squeeze(1)
        # (num_children, 1, output_dim) because gru input needs a batch size dimension as well.
        child_memory_cells = torch.stack([rnn_state[1]
                                          for rnn_state in child_rnn_states]).unsqueeze(1)
        # (1, 1, output_dim)
        _, aggregated_memory_cell = self._child_representation_aggregator(child_memory_cells)
        # (1, output_dim)
        aggregated_memory_cell = aggregated_memory_cell.squeeze(1)
        return aggregated_hidden_state, aggregated_memory_cell

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor,
                production_rules: List[List[str]],
                is_nonterminal: Callable[[str], bool]) -> torch.Tensor:
        # TODO (pradeep): Make this implementation more efficient.
        grammar_state = RnnGrammarState([], is_nonterminal)
        outputs: List[torch.Tensor] = []
        batch_size, _, _ = inputs.size()
        for i in range(batch_size):
            instance_outputs: List[torch.Tensor] = []
            rule_index = 0
            for input_, input_mask in zip(inputs[i], mask[i]):
                if not input_mask:
                    instance_outputs.append(input_.new(torch.zeros(self.get_output_dim())))
                    continue
                production_rule = production_rules[i][rule_index]
                rule_index += 1
                child_rnn_states = grammar_state.get_child_rnn_states(production_rule)
                if child_rnn_states is None:
                    # (1, output_dim)
                    aggregated_hidden_state = self._leaf_hidden_state.unsqueeze(0)
                    aggregated_memory_cell = self._leaf_memory_cell.unsqueeze(0)
                else:
                    aggregated_hidden_state, aggregated_memory_cell = \
                            self._aggregate_child_states(child_rnn_states)

                hidden_state, memory_cell = self._lstm_cell(input_.unsqueeze(0),
                                                            (aggregated_hidden_state, aggregated_memory_cell))
                hidden_state = hidden_state.squeeze(0)
                memory_cell = memory_cell.squeeze(0)
                grammar_state = grammar_state.update(production_rule, (hidden_state, memory_cell))
                instance_outputs.append(hidden_state)
            # (sequence_length, output_dim)
            outputs.append(torch.stack(instance_outputs))
        # (batch_size, sequence_length, output_dim)
        return torch.stack(outputs)
