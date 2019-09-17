# pylint: disable=no-self-use
import pytest
from allennlp.common.testing import AllenNlpTestCase

from gensem.modules.rnn_grammar_state import RnnGrammarState


def is_nonterminal(symbol: str) -> bool:
    return symbol.isupper() or symbol == '@start@'


class TestRnnGrammarStatelet(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.action_rnn_state_mapping = {'@start@ -> S': object(),
                                         'S -> [A, B]': object(),
                                         'A -> a': object(),
                                         'B -> b': object()}

    def test_update_works(self):
        # pylint: disable=protected-access
        grammar_state = RnnGrammarState([], is_nonterminal)
        action1 = "A -> a"
        grammar_state = grammar_state.update(action1, self.action_rnn_state_mapping[action1])
        assert grammar_state._nonterminal_stack == [('A', self.action_rnn_state_mapping[action1])]
        action2 = "B -> b"
        grammar_state = grammar_state.update(action2, self.action_rnn_state_mapping[action2])
        assert grammar_state._nonterminal_stack == [('A', self.action_rnn_state_mapping[action1]),
                                                    ('B', self.action_rnn_state_mapping[action2])]
        action3 = "S -> [A, B]"
        grammar_state = grammar_state.update(action3, self.action_rnn_state_mapping[action3])
        assert grammar_state._nonterminal_stack == [('S', self.action_rnn_state_mapping[action3])]

    def test_get_child_rnn_states(self):
        grammar_state = RnnGrammarState([], is_nonterminal)
        assert grammar_state.get_child_rnn_states("A -> a") is None
        with pytest.raises(AssertionError):
            grammar_state.get_child_rnn_states("S -> [A, B]")

        grammar_state = RnnGrammarState([("A", self.action_rnn_state_mapping["A -> a"]),
                                         ("B", self.action_rnn_state_mapping["B -> b"])],
                                        is_nonterminal)
        child_rnn_states = grammar_state.get_child_rnn_states("S -> [A, B]")
        assert child_rnn_states == [self.action_rnn_state_mapping["B -> b"],
                                    self.action_rnn_state_mapping["A -> a"]]
        with pytest.raises(AssertionError):
            grammar_state.get_child_rnn_states("S -> [B, A]")

    def test_update_followed_by_get_states(self):
        grammar_state = RnnGrammarState([], is_nonterminal)
        grammar_state = grammar_state.update('A -> a', self.action_rnn_state_mapping['A -> a'])
        grammar_state = grammar_state.update('B -> b', self.action_rnn_state_mapping['B -> b'])
        child_states = grammar_state.get_child_rnn_states('S -> [A, B]')
        assert child_states == [self.action_rnn_state_mapping['B -> b'],
                                self.action_rnn_state_mapping['A -> a']]
