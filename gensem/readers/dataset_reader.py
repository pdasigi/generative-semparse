import logging
from typing import Dict, List
import os
import json

from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.instance import Instance
from allennlp.data.fields import (Field, TextField, MetadataField, ProductionRuleField,
                                  ListField)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.semparse import ParsingError
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.domain_languages import WikiTablesLanguage

from gensem.readers import utils as reader_utils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wikitables-back-translator")
class WikiTablesBackTranslationDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tables_directory: str = None,
                 offline_logical_forms_directory: str = None,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 rule_indexers: Dict[str, TokenIndexer] = None,
                 output_production_rules: bool = True,
                 output_world: bool = True) -> None:
        super().__init__(lazy=lazy)
        self._tables_directory = tables_directory
        self._offline_logical_forms_directory = offline_logical_forms_directory
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer("tokens")}
        self._rule_indexers = rule_indexers or {"tokens": SingleIdTokenIndexer("rules")}
        self._output_production_rules = output_production_rules
        self._output_world = output_world

    @overrides
    def _read(self, file_path: str):
        # TODO(pradeep): Fix.
        if self._offline_logical_forms_directory is not None:
            logger.warning("Cannot handle multiple logical forms per instance yet!")

        with open(file_path, "r") as data_file:
            for line in data_file:
                if not line:
                    continue
                line_data = json.loads(line)
                question = line_data["question"]
                logical_form = line_data["logical_form"]
                # We want the tagged file, but the ``*.examples`` files typically point to CSV.
                table_filename = os.path.join(self._tables_directory,
                                              line_data["table_filename"].replace("csv", "tagged"))

                table_lines = [line.split("\t") for line in open(table_filename).readlines()]
                instance = self.text_to_instance(logical_form=logical_form,
                                                 table_lines=table_lines,
                                                 question=question)
                if instance is not None:
                    yield instance

    def text_to_instance(self,  # type: ignore
                         logical_form: str,
                         table_lines: List[List[str]],
                         question: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_question = self._tokenizer.tokenize(question.lower())
        tokenized_question.insert(0, Token(START_SYMBOL))
        tokenized_question.append(Token(END_SYMBOL))
        question_field = TextField(tokenized_question, self._question_token_indexers)
        table_context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
        world = WikiTablesLanguage(table_context)

        action_sequence = None
        try:
            action_sequence = world.logical_form_to_action_sequence(logical_form)
            action_sequence = reader_utils.make_bottom_up_action_sequence(action_sequence,
                                                                          world.is_nonterminal)
            action_sequence_field = TextField([Token(rule) for rule in  action_sequence],
                                              self._rule_indexers)
        except ParsingError as error:
            logger.debug(f'Parsing error: {error.message}, skipping logical form')
            logger.debug(f'Question was: {question}')
            logger.debug(f'Logical form was: {logical_form}')
            logger.debug(f'Table info was: {table_lines}')
        except:
            logger.error(logical_form)
            raise

        if not action_sequence:
            return None

        # Not very happy about calling the action sequence field "source_tokens", but doing so to make these
        # instances compatible with SimpleSeq2Seq.
        fields = {'source_tokens': action_sequence_field,
                  'target_tokens': question_field}

        if self._output_world:
            fields['world'] = MetadataField(world)

        if self._output_production_rules:
            production_rule_fields: List[Field] = []
            for production_rule in action_sequence:
                _, rule_right_side = production_rule.split(' -> ')
                is_global_rule = not world.is_instance_specific_entity(rule_right_side)
                field = ProductionRuleField(production_rule, is_global_rule=is_global_rule)
                production_rule_fields.append(field)
            action_field = ListField(production_rule_fields)
            fields['actions'] = action_field

        return Instance(fields)
