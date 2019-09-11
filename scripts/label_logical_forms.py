import argparse
from typing import List, Dict
import random
import os
import gzip
import json

from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wtq_util


MAX_NUM_LOGICAL_FORMS_TO_SHOW = 15


def main(examples_file: str,
         tables_directory: str,
         logical_forms_directory: str,
         output_file: str) -> None:
    examples: List[Dict] = []
    with open(examples_file) as input_file:
        for line in input_file:
            examples.append(wtq_util.parse_example_line(line))
    random.shuffle(examples)  # Shuffling to label in random order

    processed_examples: Set[str] = set()
    if os.path.exists(output_file):
        with open(output_file) as output_file_for_reading:
            for line in output_file_for_reading:
                example_id = json.loads(line)["id"]
                processed_examples.add(example_id)

    with open(output_file, "a") as output_file_for_appending:
        for example in examples:
            example_id = example["id"]
            if example_id in processed_examples:
                # We've already labeled this example
                continue
            question = example["question"]
            table_filename = example["table_filename"]
            full_table_filename = os.path.join(tables_directory, table_filename)
            table_lines = []
            with open(full_table_filename.replace(".csv", ".tsv")) as table_file:
                table_lines = table_file.readlines()
            logical_forms_file = os.path.join(logical_forms_directory, f"{example_id}.gz")
            if not os.path.exists(logical_forms_file):
                continue
            print("".join(table_lines))
            print()
            with gzip.open(logical_forms_file, "rt") as lf_file:
                for i, logical_form in enumerate(lf_file):
                    logical_form = logical_form.strip()
                    print(question)
                    print(logical_form)
                    print()
                    user_input = None
                    while user_input not in ["y", "n", "w", "s"]:
                        user_input = input("Correct? ('y'es / 'n'o / 'w'rite correct lf / 's'kip): ")
                        user_input = user_input.lower()
                    if user_input == "s":
                        break
                    elif user_input == "y":
                        instance_output = {"id": example_id,
                                           "question": question,
                                           "table_filename": table_filename,
                                           "logical_form": logical_form}
                        print(json.dumps(instance_output), file=output_file_for_appending)
                        break
                    elif user_input == "w":
                        correct_logical_form = input("Enter correct logical form: ")
                        instance_output = {"id": example_id,
                                           "question": question,
                                           "table_filename": table_filename,
                                           "logical_form": correct_logical_form}
                        print(json.dumps(instance_output), file=output_file_for_appending)
                        break
                    if i >= MAX_NUM_LOGICAL_FORMS_TO_SHOW:
                        break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("""Interface for collecting binary labels for utterance-lf pairs for WTQ""")
    argparser.add_argument("--examples", required=True, type=str, help="Path to the examples file")
    argparser.add_argument("--tables", required=True, type=str, help="Path to the tables directory")
    argparser.add_argument("--lfs", required=True, type=str,
                           help="Path to the directory containing logical forms")
    argparser.add_argument("--output", required=True, type=str, help="Path to the output file")
    args = argparser.parse_args()
    main(args.examples, args.tables, args.lfs, args.output)
