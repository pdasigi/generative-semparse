# pylint: disable=no-self-use

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.domain_languages import WikiTablesLanguage

from gensem.readers.dataset_reader import WikiTablesBackTranslationDatasetReader

class TestWikiTablesBackTranslationDatasetReader(AllenNlpTestCase):
    def test_reader_reads(self):
        params = {
                'lazy': False,
                'tables_directory': "fixtures/data/tables",
                }
        reader = WikiTablesBackTranslationDatasetReader.from_params(Params(params))
        dataset = reader.read("fixtures/data/sample_data.jsonl")
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]

        assert instance.fields.keys() == {
                'question',
                'world',
                'actions',
                'action_sequence',
                }

        question_tokens = ["who", "was", "appointed", "before", "h.w", ".", "whillock", "?"]
        assert [t.text for t in instance.fields["question"].tokens] == question_tokens

        # The content of this will be tested indirectly by checking the actions; we'll just make
        # sure we get a WikiTablesWorld object in here.
        assert isinstance(instance.fields['world'].as_tensor({}), WikiTablesLanguage)

        action_fields = instance.fields['actions'].field_list
        actions = [action_field.rule for action_field in action_fields]

        action_sequence = instance.fields["action_sequence"]
        action_indices = [l.sequence_index for l in action_sequence.field_list]
        actions = [actions[i] for i in action_indices]
        assert actions == ['@start@ -> List[str]',
                           'List[str] -> [<List[Row],StringColumn:List[str]>, List[Row], StringColumn]',
                           '<List[Row],StringColumn:List[str]> -> select_string',
                           'List[Row] -> [<List[Row]:List[Row]>, List[Row]]',
                           '<List[Row]:List[Row]> -> previous',
                           'List[Row] -> [<List[Row],StringColumn,List[str]:List[Row]>, List[Row], StringColumn, List[str]]',  # pylint: disable=line-too-long
                           '<List[Row],StringColumn,List[str]:List[Row]> -> filter_in',
                           'List[Row] -> all_rows',
                           'StringColumn -> string_column:incumbent',
                           'List[str] -> string:h_w_whillock',
                           'StringColumn -> string_column:incumbent']
