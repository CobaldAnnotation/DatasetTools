import argparse

from datasets import Dataset, DatasetDict

from src.parsing import parse_incr, OPTIONAL_TAGS


def main():
    parser = argparse.ArgumentParser(description="Load and process a CONLLU file")

    parser.add_argument(
        "--train_data_path",
        type=str,
        help="path to the train conllu file"
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
        help="path to the optional validation conllu file"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="path to the optional test conllu file"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="huggingface repo to push built dataset to"
    )
    parser.add_argument(
        "config_name",
        type=str,
        help='dataset configuration name within the repo, e.g. "en"'
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        type=str,
        default=OPTIONAL_TAGS,
        choices=OPTIONAL_TAGS,
        help=(
            "Tags to include in dataset, e.g. [heads, deprels, deps]."
            "By default, all CoBaLD tags are used."
        )
    )

    args = parser.parse_args()

    dataset_dict = DatasetDict()

    if args.train_data_path:
        train_sentences = parse_incr(args.train_data_path, args.tags)
        dataset_dict['train'] = Dataset.from_generator(lambda: train_sentences)

    if args.validation_data_path:
        validation_sentences = parse_incr(args.validation_data_path, args.tags)
        dataset_dict['validation'] = Dataset.from_generator(lambda: validation_sentences)

    if args.test_data_path:
        test_sentences = parse_incr(args.test_data_path, args.tags)
        dataset_dict['test'] = Dataset.from_generator(lambda: test_sentences)
    
    dataset_dict.push_to_hub(args.repo_id, args.config_name)


if __name__ == "__main__":
    main()