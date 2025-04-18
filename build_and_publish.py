import argparse

from datasets import Dataset, DatasetDict

from src.parsing import parse_incr
from src.train_test_split import train_test_split


def main():
    """Main function to load and process dataset."""
    
    parser = argparse.ArgumentParser(description="Load and process a CONLLU file")

    parser.add_argument("data_path", type=str, help="path to the conllu file")
    parser.add_argument("repo_id", type=str,
                        help="huggingface repo to push built dataset to")
    parser.add_argument("config_name", type=str,
                        help='dataset configuration name, e.g. "en"')
    parser.add_argument("--train_fraction", type=float, default=0.8,
                        help="relative size of train set")
    args = parser.parse_args()

    sentences = list(parse_incr(args.data_path))
    train_sentences, validation_sentences = train_test_split(
        sentences,
        train_fraction=args.train_fraction,
        tagsets_names=[
            'upos',
            'xpos',
            'feats',
            'deprels',
            'deps',
            'miscs',
            'deepslots',
            'semclasses'
        ]
    )
    
    train_dataset = Dataset.from_list(train_sentences)
    validation_dataset = Dataset.from_list(validation_sentences)
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })
    dataset_dict.push_to_hub(args.repo_id, args.config_name)


if __name__ == "__main__":
    main()