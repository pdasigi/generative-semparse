from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('wikitables-reranker')
class WikiTablesReranker(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        logical_forms = json_dict["logical_forms"]
        table_lines = json_dict["table"].split("\n")
        question_text = json_dict["question"]
        instance = self._dataset_reader.text_to_instance(logical_forms=logical_forms,
                                                         table_lines=table_lines,
                                                         question=question_text)
        return instance


    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        outputs = super().predict_json(inputs)
        logical_form_indices = outputs["sorted_logical_form_indices"]
        ranked_logical_forms = [inputs["logical_forms"][index] for index in logical_form_indices]
        outputs_to_return = {"question": inputs["question"],
                             "table": inputs["table"],
                             "ranked_logical_forms": ranked_logical_forms}
        return outputs_to_return
