# pylint: disable=no-self-use
import torch
from allennlp.common.testing import AllenNlpTestCase

from gensem.modules.tree_lstm import TreeLSTM


class TestTreeLSTM(AllenNlpTestCase):
    def test_forward(self):
        # We're just testing whether forward runs successfully and whether the outputs are of the right size. The
        # actual computation is in the grammar state, which is tested separately, and in the LSTM and GRU torch
        # modules.
        batch_size = 5
        sequence_length = 4
        input_dim = 3
        output_dim = 2
        tree_lstm = TreeLSTM(input_dim, output_dim)
        inputs = torch.FloatTensor(batch_size, sequence_length, input_dim)
        mask = torch.ones(batch_size, sequence_length)
        production_rules = [['A -> a', 'B -> b', 'S -> [A, B]', '@start@ -> S']] * batch_size
        is_nonterminal = lambda x: x.isupper() or x == '@start@'
        outputs = tree_lstm(inputs, mask, production_rules, is_nonterminal)
        assert outputs.size() == (batch_size, sequence_length, output_dim)

    def test_forward_with_masked_input(self):
        # We're just testing whether forward runs successfully and whether the outputs are of the right size. The
        # actual computation is in the grammar state, which is tested separately, and in the LSTM and GRU torch
        # modules.
        batch_size = 2
        sequence_length = 4
        input_dim = 3
        output_dim = 5
        tree_lstm = TreeLSTM(input_dim, output_dim)
        inputs = torch.FloatTensor(batch_size, sequence_length, input_dim)
        mask = torch.LongTensor([[0, 0, 0, 1], [0, 1, 1, 1]])
        production_rules = [['@PADDING@', '@PADDING@', '@PADDING@', 'A -> a'],
                            ['@PADDING@', 'A -> a', 'S -> A', '@start@ -> S']]
        is_nonterminal = lambda x: x.isupper() or x == '@start@'
        outputs = tree_lstm(inputs, mask, production_rules, is_nonterminal)
        assert outputs.size() == (batch_size, sequence_length, output_dim)
