# pylint: disable=unused-import,invalid-name
from flaky import flaky
from allennlp.common.testing import ModelTestCase

from gensem import WikiTablesQuestionGenerator, WikiTablesQuestionGeneratorReader

class TestWikiTablesQuestionGenerator(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model("fixtures/experiment.json",
                          "fixtures/data/sample_data.jsonl")

    @flaky(max_runs=4, min_passes=1)
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_simple_seq2seq_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load("fixtures/simple-seq2seq-experiment.json")
