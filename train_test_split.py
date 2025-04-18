import json
import numpy


def extract_sentence_tagset(sentence: dict, tagset_name: str) -> dict:
    tagset = {}
    if tagset_name == 'deps':
        tagset = {deprel
                  for dep in sentence[tagset_name]
                  if dep is not None
                  for deprel in json.loads(dep).values()}
    else:
        tagset = set(sentence[tagset_name])
    # Remove None from tagset.
    tagset.discard(None)
    return tagset


def build_min_coverage(sentences: list[dict], tagsets_names: list[str]) -> set[int]:
    """
    Build a set of sentences such that every tag from `tagsets` tagsets has at least one occurrence.
    We use this to make sure training set contains all the tags of all tagsets.
    """

    sentences_tagsets = []
    tagsets = {tagset_name: set() for tagset_name in tagsets_names}

    for sentence in sentences:
        sentence_tagsets = dict()
        for tagset_name in tagsets_names:
            sentence_tagsets[tagset_name] = extract_sentence_tagset(sentence, tagset_name)
            tagsets[tagset_name] |= sentence_tagsets[tagset_name]
        sentences_tagsets.append(sentence_tagsets)

    # inv_tagsets is a dict of inverted tagsets.
    # Inverted tagset is a dict of (tag -> indexes of sentences containing this tag).
    inv_tagsets = {tagset_name: {tag: set() for tag in tagsets[tagset_name]} for tagset_name in tagsets_names}
    for sentence_index, sentence_tagsets in enumerate(sentences_tagsets):
        for tagset_name in tagsets_names:
            for tag in sentence_tagsets[tagset_name]:
                inv_tagsets[tagset_name][tag].add(sentence_index)

    cover_sentences_indexes = set()

    # While there is non-empty inverted tagset, i.e. we have "uncovered" tags, do the following:
    while sum(len(inv_tagset) for inv_tagset in inv_tagsets.values()) > 0:
        # Find the rarest tag over all tagsets.
        rarest_tag, rarest_tag_frequency, rarest_tag_tagset_name = None, float('inf'), None
        for tagset_name, inv_tagset in inv_tagsets.items():
            if len(inv_tagset) == 0:
                continue
            rare_tag, rare_tag_sentences = min(inv_tagset.items(), key=lambda item: len(item[1]))
            # Number of sentences with this tag.
            rare_tag_frequency = len(rare_tag_sentences)
            if rare_tag_frequency < rarest_tag_frequency:
                rarest_tag = rare_tag
                rarest_tag_frequency = rare_tag_frequency
                rarest_tag_tagset_name = tagset_name
        assert rarest_tag is not None

        # Remove tag from tagset.
        sentences_with_tag = inv_tagsets[rarest_tag_tagset_name].pop(rarest_tag)
        # Pick any sentence with this tag.
        sentence_with_tag = list(sentences_with_tag)[0]
        # Add sentence to training set
        cover_sentences_indexes.add(sentence_with_tag)
        # ...and also remove all tags present in this sentence from all inverted indexes.
        for tagset_name, inv_tagset in inv_tagsets.items():
            for tag in list(inv_tagset.keys()):
                sentences = inv_tagset[tag]
                if sentence_with_tag in sentences:
                    inv_tagset.pop(tag)

    return cover_sentences_indexes


def train_test_split(
    sentences: list[dict],
    train_fraction: float,
    tagsets_names: list[str]
) -> tuple[list, list]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0.0 and 1.0")

    sentence_count = len(sentences)
    target_train_size = int(train_fraction * sentence_count)
    target_test_size = sentence_count - target_train_size

    train_sentences = []
    test_sentences = []

    while len(train_sentences) < target_train_size and len(test_sentences) < target_test_size:
        # Update train set.
        min_train_coverage = build_min_coverage(sentences, tagsets_names)
        train_sentences += [sentence for i, sentence in enumerate(sentences) if i in min_train_coverage]
        # Remove sentences that have been added to the train set.
        sentences = [sentence for i, sentence in enumerate(sentences) if i not in min_train_coverage]
        
        # Update test set.
        min_test_coverage = build_min_coverage(sentences, tagsets_names)
        test_sentences += [sentence for i, sentence in enumerate(sentences) if i in min_test_coverage]
        # Remove sentences that have been added to the test set.
        sentences = [sentence for i, sentence in enumerate(sentences) if i not in min_test_coverage]

    if len(train_sentences) < target_train_size:
        train_sentences += sentences
    else:
        test_sentences += sentences

    assert len(train_sentences) + len(test_sentences) == sentence_count
    assert numpy.isclose(len(train_sentences) / sentence_count, train_fraction, atol=0.05)

    return train_sentences, test_sentences
