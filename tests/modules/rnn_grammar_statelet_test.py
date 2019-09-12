# pylint: disable=no-self-use
from allennlp.common.testing import AllenNlpTestCase

from gensem.modules.rnn_grammar_statelet import RnnGrammarStatelet


def is_nonterminal(symbol: str) -> bool:
    return symbol.isupper() or symbol == '@start@'


class TestRnnGrammarStatelet(AllenNlpTestCase):
    def test_take_action_keeps_stacks_in_sync(self):
        initial_rnn_state = object()
        # Passing None for valid actions.
        grammar_state = RnnGrammarStatelet(['@start@'],
                                           [initial_rnn_state],
                                           None,
                                           is_nonterminal)
        action_rnn_state_mapping = {'@start@ -> S': object(),
                                    'S -> [A, B]': object(),
                                    'A -> a': object(),
                                    'B -> [C, D, E]': object(),
                                    'C -> c': object(),
                                    'D -> d': object(),
                                    'E -> e': object()}
        action_sequence = ['@start@ -> S', 'S -> [A, B]', 'A -> a', 'B -> [C, D, E]',
                           'C -> c', 'D -> d', 'E -> e']
        parent_rnn_states = []

        for action in action_sequence:
            parent_rnn_states.append(grammar_state.get_parent_rnn_state())
            grammar_state = grammar_state.take_action(action,
                                                      action_rnn_state_mapping[action])

        assert parent_rnn_states == [initial_rnn_state,
                                     action_rnn_state_mapping['@start@ -> S'],
                                     action_rnn_state_mapping['S -> [A, B]'],
                                     action_rnn_state_mapping['S -> [A, B]'],
                                     action_rnn_state_mapping['B -> [C, D, E]'],
                                     action_rnn_state_mapping['B -> [C, D, E]'],
                                     action_rnn_state_mapping['B -> [C, D, E]']]
