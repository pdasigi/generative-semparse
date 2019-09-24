# pylint: disable=no-self-use,invalid-name
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


    def test_make_bottom_up_action_sequence_on_wtq_sequence(self):
        is_nonterminal = lambda x: x[0].isupper() or x[0] == '['
        top_down_sequence = ['@start@ -> Date', 'Date -> [<Number,Number,Number:Date>, Number, Number, Number]',
                             '<Number,Number,Number:Date> -> date', 'Number -> [<List[Row]:Number>, List[Row]]',
                             '<List[Row]:Number> -> count',
                             'List[Row] -> [<List[Row],Column:List[Row]>, List[Row], Column]',
                             '<List[Row],Column:List[Row]> -> same_as', 'List[Row] -> all_rows',
                             'Column -> string_column:party', 'Number -> -1', 'Number -> -1']
        bottom_up_sequence = utils.make_bottom_up_action_sequence(top_down_sequence, is_nonterminal)
        assert bottom_up_sequence == ['<Number,Number,Number:Date> -> date', '<List[Row]:Number> -> count',
                                      '<List[Row],Column:List[Row]> -> same_as', 'List[Row] -> all_rows',
                                      'Column -> string_column:party',
                                      'List[Row] -> [<List[Row],Column:List[Row]>, List[Row], Column]',
                                      'Number -> [<List[Row]:Number>, List[Row]]', 'Number -> -1', 'Number -> -1',
                                      'Date -> [<Number,Number,Number:Date>, Number, Number, Number]',
                                      '@start@ -> Date']
