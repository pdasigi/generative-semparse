# pylint: disable=no-self-use

from allennlp.common import Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.domain_languages import WikiTablesLanguage

from gensem.readers.dataset_reader import WikiTablesQuestionGeneratorReader

class TestWikiTablesBackTranslationDatasetReader(AllenNlpTestCase):
    def test_reader_reads_jsonl(self):
        params = {
                'lazy': False,
                'tables_directory': "fixtures/data/tables",
                }
        reader = WikiTablesQuestionGeneratorReader.from_params(Params(params))
        dataset = reader.read("fixtures/data/sample_data.jsonl")
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]

        assert instance.fields.keys() == {
                'action_sequences',
                'target_tokens',
                'world',
                'actions',
                }

        question_tokens = [START_SYMBOL, "who", "was", "appointed", "before", "h.w", ".", "whillock", "?",
                           END_SYMBOL]
        assert [t.text for t in instance.fields["target_tokens"].tokens] == question_tokens

        # The content of this will be tested indirectly by checking the actions; we'll just make
        # sure we get a WikiTablesWorld object in here.
        assert isinstance(instance.fields['world'].as_tensor({}), WikiTablesLanguage)

        all_action_fields = instance.fields['actions'].field_list
        actions = [[action_field.rule for action_field in action_fields] for action_fields in all_action_fields]

        # Actions in bottom-up order.
        expected_sequence = [['<List[Row],StringColumn:List[str]> -> select_string',
                              '<List[Row]:List[Row]> -> previous',
                              '<List[Row],StringColumn,List[str]:List[Row]> -> filter_in',
                              'List[Row] -> all_rows',
                              'StringColumn -> string_column:incumbent',
                              'List[str] -> string:h_w_whillock',
                              'List[Row] -> [<List[Row],StringColumn,List[str]:List[Row]>, List[Row], StringColumn, List[str]]',  # pylint: disable=line-too-long
                              'List[Row] -> [<List[Row]:List[Row]>, List[Row]]',
                              'StringColumn -> string_column:incumbent',
                              'List[str] -> [<List[Row],StringColumn:List[str]>, List[Row], StringColumn]',
                              '@start@ -> List[str]']]
        assert actions == expected_sequence
        assert [[t.text for t in text_field.tokens]
                for text_field in instance.fields["action_sequences"]] == expected_sequence

    def test_reader_reads_examples(self):
        params = {
                'lazy': False,
                'tables_directory': "fixtures/data/tables",
                'offline_logical_forms_directory': "fixtures/data/logical_forms"
                }
        reader = WikiTablesQuestionGeneratorReader.from_params(Params(params))
        dataset = reader.read("fixtures/data/sample_data.examples")
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]

        assert instance.fields.keys() == {
                'action_sequences',
                'target_tokens',
                'world',
                'actions',
                }

        question_tokens = [START_SYMBOL, "who", "was", "appointed", "before", "h.w", ".", "whillock", "?",
                           END_SYMBOL]
        assert [t.text for t in instance.fields["target_tokens"].tokens] == question_tokens

        # The content of this will be tested indirectly by checking the actions; we'll just make
        # sure we get a WikiTablesWorld object in here.
        assert isinstance(instance.fields['world'].as_tensor({}), WikiTablesLanguage)
        all_action_fields = instance.fields['actions'].field_list
        assert len(all_action_fields) == 2
