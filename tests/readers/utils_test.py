# pylint: disable=no-self-use
from allennlp.common.testing import AllenNlpTestCase

from gensem.readers import utils


class TestUtils(AllenNlpTestCase):
    def test_make_bottom_up_action_sequence(self):
        is_nonterminal = lambda x: (len(x) == 1 and x.isupper()) or '[' in x

        top_down_sequence = ['@start@ -> S', 'S -> [A, B]', 'A -> a', 'B -> b']
        bottom_up_sequence = utils.make_bottom_up_action_sequence(top_down_sequence, is_nonterminal)
        assert bottom_up_sequence == ['A -> a', 'B -> b', 'S -> [A, B]', '@start@ -> S']

        top_down_sequence = ['@start@ -> S', 'S -> [A, B]', 'A -> [C, D]', 'C -> c', 'D -> d', 'B -> b']
        bottom_up_sequence = utils.make_bottom_up_action_sequence(top_down_sequence, is_nonterminal)
        assert bottom_up_sequence == ['C -> c', 'D -> d', 'A -> [C, D]', 'B -> b', 'S -> [A, B]', '@start@ -> S']

        top_down_sequence = ['@start@ -> S', 'S -> A', 'A -> B', 'B -> C', 'C -> c']
        bottom_up_sequence = utils.make_bottom_up_action_sequence(top_down_sequence, is_nonterminal)
        assert bottom_up_sequence == ['C -> c', 'B -> C', 'A -> B', 'S -> A', '@start@ -> S']
