from flaky import flaky
from allennlp.common.testing import ModelTestCase

# pylint: disable=unused-import
from gensem import WikiTablesQuestionGenerator, WikiTablesQuestionGeneratorReader

class TestWikiTablesQuestionGenerator(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model("fixtures/experiment.json",
                          "fixtures/data/sample_data.jsonl")

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
