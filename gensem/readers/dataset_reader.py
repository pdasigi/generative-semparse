import logging
from typing import Dict, List
import os
import json
import gzip

from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.instance import Instance
from allennlp.data.fields import (Field, TextField, MetadataField, ProductionRuleField,
                                  ListField)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wtq_data_util
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


@DatasetReader.register("wikitables-question-generator")
class WikiTablesQuestionGeneratorReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tables_directory: str = None,
                 offline_logical_forms_directory: str = None,
                 max_num_logical_forms: int = 30,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 rule_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tables_directory = tables_directory
        self._offline_logical_forms_directory = offline_logical_forms_directory
        self._max_num_logical_forms = max_num_logical_forms
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer("tokens")}
        self._rule_indexers = rule_indexers or {"tokens": SingleIdTokenIndexer("rules")}

    @overrides
    def _read(self, file_path: str):
        data: List[Dict[str, str]] = []
        if file_path.endswith(".jsonl"):
            with open(file_path, "r") as data_file:
                for line in data_file:
                    if not line:
                        continue
                    line_data = json.loads(line)
                    line_data["logical_form"] = [line_data["logical_form"]]
                    data.append(line_data)
        elif file_path.endswith(".examples"):
            num_examples = 0
            num_examples_without_lf = 0
            if self._offline_logical_forms_directory is None:
                raise RuntimeError("Logical forms directory required when processing examples files!")
            with open(file_path, "r") as data_file:
                for line in data_file:
                    num_examples += 1
                    line_data = wtq_data_util.parse_example_line(line)
                    example_id = line_data["id"]
                    logical_forms_file = os.path.join(self._offline_logical_forms_directory,
                                                      f"{example_id}.gz")
                    if not os.path.exists(logical_forms_file):
                        num_examples_without_lf += 1
                        continue
                    logical_forms = None
                    with gzip.open(logical_forms_file, "rt") as lf_file:
                        logical_forms = [x.strip() for x in lf_file.readlines()][:self._max_num_logical_forms]
                    line_data["logical_form"] = logical_forms
                    data.append(line_data)
            logger.info(f"Skipped {num_examples_without_lf} out of {num_examples} examples")
        else:
            raise RuntimeError(f"Unknown file type: {file_path}. Was expecting either *.examples or *.jsonl")

        for datum in data:
            # We want the tagged file, but the ``*.examples`` files typically point to CSV.
            table_filename = os.path.join(self._tables_directory,
                                          datum["table_filename"].replace("csv", "tagged"))

            table_lines = [line.split("\t") for line in open(table_filename).readlines()]
            instance = self.text_to_instance(logical_forms=datum["logical_form"],
                                             table_lines=table_lines,
                                             question=datum["question"])
            if instance is not None:
                yield instance

    def text_to_instance(self,  # type: ignore
                         logical_forms: List[str],
                         table_lines: List[List[str]],
                         question: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_question = self._tokenizer.tokenize(question.lower())
        tokenized_question.insert(0, Token(START_SYMBOL))
        tokenized_question.append(Token(END_SYMBOL))
        question_field = TextField(tokenized_question, self._question_token_indexers)
        table_context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
        world = WikiTablesLanguage(table_context)

        action_sequences_list: List[List[str]] = []
        action_sequence_fields_list: List[TextField] = []
        for logical_form in logical_forms:
            try:
                action_sequence = world.logical_form_to_action_sequence(logical_form)
                action_sequence = reader_utils.make_bottom_up_action_sequence(action_sequence,
                                                                              world.is_nonterminal)
                action_sequence_field = TextField([Token(rule) for rule in  action_sequence],
                                                  self._rule_indexers)
                action_sequences_list.append(action_sequence)
                action_sequence_fields_list.append(action_sequence_field)
            except ParsingError as error:
                logger.debug(f'Parsing error: {error.message}, skipping logical form')
                logger.debug(f'Question was: {question}')
                logger.debug(f'Logical form was: {logical_form}')
                logger.debug(f'Table info was: {table_lines}')
            except:
                logger.error(logical_form)
                raise

        if not action_sequences_list:
            return None

        all_production_rule_fields: List[List[Field]] = []
        for action_sequence in action_sequences_list:
            all_production_rule_fields.append([])
            for production_rule in action_sequence:
                _, rule_right_side = production_rule.split(' -> ')
                is_global_rule = not world.is_instance_specific_entity(rule_right_side)
                field = ProductionRuleField(production_rule, is_global_rule=is_global_rule)
                all_production_rule_fields[-1].append(field)
        action_field = ListField([ListField(production_rule_fields) for production_rule_fields in
                                  all_production_rule_fields])

        fields = {'action_sequences': ListField(action_sequence_fields_list),
                  'target_tokens': question_field,
                  'world': MetadataField(world),
                  'actions': action_field}

        return Instance(fields)
