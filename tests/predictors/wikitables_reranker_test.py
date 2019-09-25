#pylint: disable=no-self-use,unused-import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from gensem.models import WikiTablesQuestionGenerator
from gensem.predictors import WikiTablesReranker

class TestWikiTablesReranker(AllenNlpTestCase):
    def test_ranked_logical_forms_present(self):
        archive_path = "fixtures/trained_models/seq2seq_model.tar.gz"
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'wikitables-reranker')

        inputs = {"question": "Who is a good boy?",
                  "table": "Dog\tType\nFido\tgood\nDofi\tbad",
                  "logical_forms":
                  ["(select_string (filter_in all_rows string_column:type string:good) string_column:dog)",
                   "(select_string (first all_rows) string_column:dog)"]}
        result = predictor.predict_json(inputs)
        assert result["ranked_logical_forms"] == \
                ["(select_string (first all_rows) string_column:dog)",
                 "(select_string (filter_in all_rows string_column:type string:good) string_column:dog)"]
