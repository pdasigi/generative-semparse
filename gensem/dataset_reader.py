import logging
from typing import Dict, List
import os
import json

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.fields import (Field, TextField, MetadataField, ProductionRuleField,
                                  ListField, IndexField)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.semparse import ParsingError
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.domain_languages import WikiTablesLanguage


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wikitables-back-translate")
class WikiTablesBackTranslationDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tables_directory: str = None,
                 offline_logical_forms_directory: str = None,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tables_directory = tables_directory
        self._offline_logical_forms_directory = offline_logical_forms_directory
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}

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
                instance = self.text_to_instance(question=question,
                                                 table_lines=table_lines,
                                                 logical_form=logical_form)
                if instance is not None:
                    yield instance

    def text_to_instance(self,  # type: ignore
                         logical_form: str,
                         table_lines: List[List[str]],
                         question: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_question = self._tokenizer.tokenize(question.lower())
        question_field = TextField(tokenized_question, self._question_token_indexers)
        table_context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
        world = WikiTablesLanguage(table_context)
        world_field = MetadataField(world)
        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_productions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not world.is_instance_specific_entity(rule_right_side)
            field = ProductionRuleField(production_rule, is_global_rule=is_global_rule)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        # We'll make each target action sequence a List[IndexField], where the index is into
        # the action list we made above.  We need to ignore the type here because mypy doesn't
        # like `action.rule` - it's hard to tell mypy that the ListField is made up of
        # ProductionRuleFields.
        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
        action_sequence_field: Field = None
        try:
            action_sequence = world.logical_form_to_action_sequence(logical_form)
            index_fields: List[Field] = []
            for production_rule in action_sequence:
                index_fields.append(IndexField(action_map[production_rule], action_field))
            action_sequence_field = ListField(index_fields)
        except ParsingError as error:
            logger.debug(f'Parsing error: {error.message}, skipping logical form')
            logger.debug(f'Question was: {question}')
            logger.debug(f'Logical form was: {logical_form}')
            logger.debug(f'Table info was: {table_lines}')
        except KeyError as error:
            logger.debug(f'Missing production rule: {error.args}, skipping logical form')
            logger.debug(f'Question was: {question}')
            logger.debug(f'Table info was: {table_lines}')
            logger.debug(f'Logical form was: {logical_form}')
        except:
            logger.error(logical_form)
            raise

        if not action_sequence_field:
            return None

        fields = {'question': question_field,
                  'world': world_field,
                  'actions': action_field,
                  'action_sequence': action_sequence_field}

        return Instance(fields)