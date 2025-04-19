import json


ROOT_HEAD = 0

# ========================= Field level =========================

def parse_nullable(field: str) -> str | None:
    if field in ["", "_"]:
        return None
    return field


def is_null_index(x: str) -> bool:
    try:
        a, b = x.split('.')
        return a.isdecimal() and b.isdecimal()
    except ValueError:
        return False

def is_range_index(x: str) -> bool:
    try:
        a, b = x.split('-')
        return a.isdecimal() and b.isdecimal()
    except ValueError:
        return False

def parse_id(field: str) -> str:
    if not field.isdecimal() and not is_null_index(field) and not is_range_index(field):
        raise ValueError(f"Incorrect token id: {field}.")
    return field


def parse_word(field: str) -> str:
    if field == "":
        raise ValueError(f"Token form cannot be empty.")
    return field


def parse_joint_field(field: str, inner_sep: str, outer_sep: str) -> dict[str, str]:
    """
    Parse joint field with two kinds of separators (inner and outer) and return a dict.

    E.g.
    >>> parse_joint_field("Mood=Ind|Number=Sing|Person=3", '=', '|')
    {'Mood': 'Ind', 'Number': 'Sing', 'Person': '3'}
    >>> parse_joint_field("26:conj|18.1:advcl:while", ':', '|')
    {'26': 'conj', '18.1': 'advcl:while'}
    """
    if inner_sep not in field:
        raise ValueError(f"Non-empty field {field} must contain {inner_sep} separator.")

    field_dict = {}
    for item in field.split(outer_sep):
        key, value = item.split(inner_sep, 1)
        if key in field_dict:
            raise ValueError(f"Field {field} has duplicate key {key}.")
        field_dict[key] = value
    return field_dict


def parse_feats(field: str) -> str | None:
    if field in ["", "_"]:
        return None

    # validate feats are parsable
    parse_joint_field(field, inner_sep='=', outer_sep='|')
    
    return field


def parse_head(field: str) -> int | None:
    if field in ["", "_"]:
        return None

    if not field.isdecimal():
        raise ValueError(f"Non-empty head must be a decimal number, not {field}.")

    head = int(field)
    if head < 0:
        raise ValueError(f"Non-empty head must be a positive integer, not {head}.")

    return head


def parse_deps(field: str) -> str | None:
    if field in ["", "_"]:
        return None

    token_deps = parse_joint_field(field, inner_sep=':', outer_sep='|')

    # Validate deps.
    if len(token_deps) == 0:
        raise ValueError(f"Empty deps are not allowed: {field}.")
    for head in token_deps:
        if not head.isdecimal() and not is_null_index(head):
            raise ValueError(f"Deps head must be either integer or float (x.1), not {head}.")
    return json.dumps(token_deps)

# ========================= Token level =========================

Token = dict[str, str]

def validate_null_token(token: Token):
    if token['word'] != "#NULL":
        raise ValueError(f"Null token must have #NULL form, not {token['word']}.")
    if token['head'] is not None:
        raise ValueError(f"Null token must have no head, but found {token['head']}.")
    if token['deprel'] is not None:
        raise ValueError(f"Null token must have no deprel, but found {token['deprel']}.")
    if token['misc'] != 'ellipsis':
        raise ValueError(f"Null token must have 'ellipsis' misc, not {token['misc']}.")

def validate_range_token(token: Token):
    if token['lemma'] != '_':
        raise ValueError(f"Range token lemma must be _, but found {token['lemma']}.")
    if token['upos'] is not None:
        raise ValueError(f"Range token upos must be _, but found {token['upos']}.")
    if token['xpos'] is not None:
        raise ValueError(f"Range token xpos must be _, but found {token['xpos']}.")
    if token['feats'] is not None:
        raise ValueError(f"Range token feats must be _, but found {token['feats']}.")
    if token['head'] is not None:
        raise ValueError(f"Range token head must be _, but found {token['head']}.")
    if token['deprel'] is not None:
        raise ValueError(f"Range token deprel must be _, but found {token['deprel']}.")
    if token['misc'] is not None:
        raise ValueError(f"Range token misc must be _, but found {token['misc']}.")
    if token['deepslot'] is not None:
        raise ValueError(f"Range token deepslot must be _, but found {token['deepslot']}.")
    if token['semclass'] is not None:
        raise ValueError(f"Range token semclass must be _, but found {token['semclass']}.")

def validate_regular_token(token: Token):
    if token['head'] == token['id']:
        raise ValueError(f"Self-loop detected in head.")
    for head in json.loads(token['deps']):
        if head == token['id']:
            raise ValueError(f"Self-loop detected in deps. head: {head}, id: {token['id']}")

def validate_token(token: Token):
    idtag = token['id']
    if idtag.isdecimal():
        validate_regular_token(token)
    elif is_range_index(idtag):
        validate_range_token(token)
    elif is_null_index(idtag):
        validate_null_token(token)
    else:
        raise ValueError(f"Incorrect token id: {idtag}.")


def parse_token(line: str) -> Token:
    try:
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) != 12:
            raise SyntaxError(f"line {line} must have 12 fields, not {len(fields)}")

        token = {
            "id": parse_id(fields[0]),
            "word": parse_word(fields[1]),
            "lemma": fields[2] if fields[2] != "" else None,
            "upos": parse_nullable(fields[3]),
            "xpos": parse_nullable(fields[4]),
            "feats": parse_feats(fields[5]),
            "head": parse_head(fields[6]),
            "deprel": parse_nullable(fields[7]),
            "deps": parse_deps(fields[8]),
            "misc": parse_nullable(fields[9]),
            "deepslot": parse_nullable(fields[10]),
            "semclass": parse_nullable(fields[11])
        }
        validate_token(token)

    except Exception as e:
        raise ValueError(f"Validation failed on token {fields[0]}: {e}")

    return token

# ========================= Sentence level =========================

Sentence = dict[str, list[str]]

def validate_sentence(sentence: Sentence):
    # Ensure the sentence is not empty
    num_words = len(sentence["words"])
    if num_words == 0:
        raise ValueError("Empty sentence.")

    # Check that all token lists have the same length
    if not (
        len(sentence["lemmas"]) == num_words and
        len(sentence["upos"]) == num_words and
        len(sentence["xpos"]) == num_words and
        len(sentence["feats"]) == num_words and
        len(sentence["heads"]) == num_words and
        len(sentence["deprels"]) == num_words and
        len(sentence["deps"]) == num_words and
        len(sentence["miscs"]) == num_words and
        len(sentence["deepslots"]) == num_words and
        len(sentence["semclasses"]) == num_words
    ):
        raise ValueError(
            f"Inconsistent token counts in sentence: words({num_words}), "
            f"lemmas({len(sentence['lemmas'])}), upos({len(sentence['upos'])}), "
            f"xpos({len(sentence['xpos'])}), feats({len(sentence['feats'])}), "
            f"heads({len(sentence['heads'])}), deprels({len(sentence['deprels'])}), "
            f"deps({len(sentence['deps'])}), miscs({len(sentence['miscs'])}), "
            f"deepslots({len(sentence['deepslots'])}), semclasses({len(sentence['semclasses'])})"
        )

    # Validate sentence heads and ids agreement.
    ids = set(sentence["ids"])
    int_ids = {int(idtag) for idtag in sentence["ids"] if idtag.isdecimal()}

    contiguous_int_ids = set(range(min(int_ids), max(int_ids) + 1))
    if int_ids != contiguous_int_ids:
        raise ValueError(f"ids are not contiguous: {sorted(int_ids)}.")

    has_labels = any(head is not None for head in sentence["heads"])
    roots_count = sum(head == ROOT_HEAD for head in sentence["heads"])
    if has_labels and roots_count != 1:
        raise ValueError(f"There must be exactly one ROOT in a sentence, but found {roots_count}.")

    sentence_heads = set(head for head in sentence["heads"] if head is not None and head != ROOT_HEAD)
    excess_heads = sentence_heads - int_ids
    if excess_heads:
        raise ValueError(f"Heads are inconsistent with sentence ids. Excessive heads: {excess_heads}.")

    sentence_deps_heads = {
        head
        for deps in sentence["deps"] if deps is not None
        for head in json.loads(deps) if head != str(ROOT_HEAD)
    }
    excess_deps_heads = sentence_deps_heads - ids
    if excess_deps_heads:
        raise ValueError(f"Deps heads are inconsistent with sentence ids. Excessive heads: {excess_deps_heads}.")


def parse_sentence(token_lines: list[str], metadata: dict) -> Sentence:
    try:
        sentence = {
            "ids": [],
            "words": [],
            "lemmas": [],
            "upos": [],
            "xpos": [],
            "feats": [],
            "heads": [],
            "deprels": [],
            "deps": [],
            "miscs": [],
            "deepslots": [],
            "semclasses": []
        }
        for token_line in token_lines:
            token = parse_token(token_line)
            sentence["ids"].append(token["id"])
            sentence["words"].append(token["word"])
            sentence["lemmas"].append(token["lemma"])
            sentence["upos"].append(token["upos"])
            sentence["xpos"].append(token["xpos"])
            sentence["feats"].append(token["feats"])
            sentence["heads"].append(token["head"])
            sentence["deprels"].append(token["deprel"])
            sentence["deps"].append(token["deps"])
            sentence["miscs"].append(token["misc"])
            sentence["deepslots"].append(token["deepslot"])
            sentence["semclasses"].append(token["semclass"])
        validate_sentence(sentence)

    except Exception as e:
        raise ValueError(f"Validation failed on sentence {metadata.get('sent_id', None)}: {e}")

    return sentence | metadata


def parse_incr(file_path: str):
    """
    Generator that parses a CoNLL-U Plus file in CoBaLD format and yields one sentence at a time.

    Each sentence is represented as a dictionary:
    {
        "sent_id": <sent_id from metadata or None>,
        "text": <text from metadata or None>,
        "ids": list of sentence tokens ids,
        "words": list of tokens forms,
        "lemmas",
        "upos",
        "xpos",
        "feats",         .
        "heads",         .
        "deprels",       .
        "deps",
        "miscs",
        "deepslots",
        "semclasses"
    }

    The input file should have metadata lines starting with '#' (e.g., "# sent_id = 1")
    and token lines. Sentence blocks should be separated by blank lines.
    """
    metadata = {}
    token_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                # End of a sentence block.
                if token_lines:
                    yield parse_sentence(token_lines, metadata)
                    metadata = {}
                    token_lines = []
                continue
            if line.startswith("#"):
                # Process metadata lines.
                if "=" in line:
                    # Remove the '#' and split by the first '='.
                    key, value = line[1:].split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ("sent_id", "text"):
                        metadata[key] = value
                continue
            # Accumulate token lines.
            token_lines.append(line)

    # Yield any remaining sentence at the end of the file.
    if token_lines:
        yield parse_sentence(token_lines, metadata)
