# ============================================================
# SECTION 2 — DEPTH-OT V2
# Load PKL Records and Build Train-Only Vocabulary
# ============================================================

import os
import re
import gc
import json
import pickle
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# 2.1 Preconditions
# ============================================================

REQUIRED_SECTION2_GLOBALS = [
    "CONFIG",
    "DIRS",
    "FEATURE_SHARDS",
    "RUN_LOG_DIR",
    "RUN_RESULT_DIR",
]

missing_section2_globals = [
    name
    for name in REQUIRED_SECTION2_GLOBALS
    if name not in globals()
]

if missing_section2_globals:
    raise RuntimeError(
        "Sections 0–1을 먼저 실행하세요. "
        f"Missing: {missing_section2_globals}"
    )


# ============================================================
# 2.2 Fixed record paths
# ============================================================

PROCESSED_ROOT = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed"
)

RECORD_PATHS = {
    "train": (
        PROCESSED_ROOT
        / "train_records.pkl"
    ),
    "dev": (
        PROCESSED_ROOT
        / "dev_records.pkl"
    ),
    "test": (
        PROCESSED_ROOT
        / "test_records.pkl"
    ),
}

EXPECTED_PATENT_COUNTS = {
    "train": 49_599,
    "dev": 9_855,
    "test": 9_881,
}

EXPECTED_CLAIM_COUNTS = {
    "train": 844_636,
    "dev": 160_048,
    "test": 161_661,
}

for split_name, path in (
    RECORD_PATHS.items()
):
    if not path.is_file():
        raise FileNotFoundError(
            f"{split_name} record 파일이 없습니다: "
            f"{path}"
        )

print("=== RECORD FILES ===")

for split_name, path in (
    RECORD_PATHS.items()
):
    print(
        f"{split_name:5s}: "
        f"{path} "
        f"({path.stat().st_size / 2**20:.2f} MiB)"
    )


# ============================================================
# 2.3 Output paths
# ============================================================

VOCABULARY_OUTPUT_DIR = (
    Path(DIRS["depth_ot_v2_processed"])
    / "vocabulary"
    / CONFIG.run_name
)

VOCABULARY_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VOCAB_PATH = (
    VOCABULARY_OUTPUT_DIR
    / "vocabulary.json"
)

TOKEN_TO_ID_PATH = (
    VOCABULARY_OUTPUT_DIR
    / "token_to_id.json"
)

VOCAB_STATS_PATH = (
    VOCABULARY_OUTPUT_DIR
    / "vocabulary_statistics.json"
)

SECTION2_MANIFEST_PATH = (
    Path(RUN_LOG_DIR)
    / "section2_data_manifest.json"
)


# ============================================================
# 2.4 Serialization utilities
# ============================================================

def atomic_json_save_section2(
    data,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.unlink(
        missing_ok=True
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )

    os.replace(
        temporary,
        path,
    )


def extract_record_list(
    payload,
):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, tuple):
        return list(payload)

    if isinstance(payload, dict):
        for key in [
            "records",
            "patents",
            "items",
            "data",
            "examples",
        ]:
            value = payload.get(key)

            if isinstance(
                value,
                (list, tuple),
            ):
                return list(value)

    # Pandas DataFrame 지원
    if hasattr(
        payload,
        "to_dict",
    ):
        try:
            records = payload.to_dict(
                orient="records"
            )

            if isinstance(records, list):
                return records
        except Exception:
            pass

    raise TypeError(
        "지원하지 않는 record payload입니다: "
        f"{type(payload)}"
    )


# ============================================================
# 2.5 Load records
# ============================================================

def load_pickle_records(
    path,
    split_name,
):
    print(
        f"[LOAD] {split_name}: {path}"
    )

    with open(
        path,
        "rb",
    ) as file:
        payload = pickle.load(
            file
        )

    records = extract_record_list(
        payload
    )

    del payload
    gc.collect()

    if not records:
        raise RuntimeError(
            f"{split_name} records가 비어 있습니다."
        )

    if not isinstance(
        records[0],
        dict,
    ):
        raise TypeError(
            f"{split_name} record가 dict가 아닙니다: "
            f"{type(records[0])}"
        )

    print(
        f"[LOADED] {split_name}: "
        f"{len(records):,} patents"
    )

    return records


RECORDS_BY_SPLIT = {}

for split_name in [
    "train",
    "dev",
    "test",
]:
    RECORDS_BY_SPLIT[
        split_name
    ] = load_pickle_records(
        RECORD_PATHS[
            split_name
        ],
        split_name,
    )

train_records = RECORDS_BY_SPLIT[
    "train"
]

dev_records = RECORDS_BY_SPLIT[
    "dev"
]

test_records = RECORDS_BY_SPLIT[
    "test"
]


# ============================================================
# 2.6 Record normalization
# ============================================================

def normalize_patent_id_section2(
    patent_id,
):
    value = str(
        patent_id
    ).strip()

    if not value:
        raise ValueError(
            "Empty patent ID."
        )

    return value


def normalize_claim_id_section2(
    claim_id,
):
    try:
        value = int(
            claim_id
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Invalid claim ID: "
            f"{claim_id}"
        ) from error

    if value < 1:
        raise ValueError(
            f"Claim ID must be positive: "
            f"{value}"
        )

    return value


def normalize_claims_section2(
    record,
):
    claims = record.get(
        "claims"
    )

    if isinstance(claims, dict):
        return {
            normalize_claim_id_section2(
                claim_id
            ): (
                ""
                if text is None
                else str(text)
            )
            for claim_id, text
            in claims.items()
        }

    if isinstance(claims, list):
        normalized = {}

        for index, item in enumerate(
            claims,
            start=1,
        ):
            if isinstance(item, dict):
                claim_id = item.get(
                    "claim_id",
                    item.get(
                        "id",
                        index,
                    ),
                )

                text = item.get(
                    "text",
                    item.get(
                        "claim_text",
                        "",
                    ),
                )
            else:
                claim_id = index
                text = item

            normalized[
                normalize_claim_id_section2(
                    claim_id
                )
            ] = (
                ""
                if text is None
                else str(text)
            )

        return normalized

    raise TypeError(
        "Unsupported claims structure: "
        f"{type(claims)}"
    )


# ============================================================
# 2.7 Validate record structure
# ============================================================

def validate_records(
    split_name,
    records,
):
    expected_patents = (
        EXPECTED_PATENT_COUNTS[
            split_name
        ]
    )

    if len(records) != expected_patents:
        raise RuntimeError(
            f"{split_name} patent count mismatch: "
            f"observed={len(records):,}, "
            f"expected={expected_patents:,}"
        )

    seen_patents = set()

    number_of_claims = 0
    number_of_edges = 0
    number_of_adjacent_edges = 0
    number_of_non_adjacent_edges = 0
    number_of_empty_claims = 0

    maximum_claims = 0
    maximum_depth = 0

    first_record_keys = sorted(
        records[0].keys()
    )

    for record in tqdm(
        records,
        desc=f"Validating {split_name}",
    ):
        if "patent_id" not in record:
            raise KeyError(
                "Record에 patent_id가 없습니다."
            )

        if "claims" not in record:
            raise KeyError(
                "Record에 claims가 없습니다."
            )

        patent_id = (
            normalize_patent_id_section2(
                record["patent_id"]
            )
        )

        if patent_id in seen_patents:
            raise ValueError(
                f"Duplicate patent ID: "
                f"{split_name}/{patent_id}"
            )

        seen_patents.add(
            patent_id
        )

        claims = normalize_claims_section2(
            record
        )

        if not claims:
            raise ValueError(
                f"Claim이 없는 patent: "
                f"{patent_id}"
            )

        claim_ids = set(
            claims.keys()
        )

        number_of_claims += len(
            claims
        )

        maximum_claims = max(
            maximum_claims,
            len(claims),
        )

        number_of_empty_claims += sum(
            not text.strip()
            for text in claims.values()
        )

        raw_depth = record.get(
            "depth",
            {},
        )

        normalized_depth = {}

        if isinstance(raw_depth, dict):
            normalized_depth = {
                normalize_claim_id_section2(
                    claim_id
                ): int(depth)
                for claim_id, depth
                in raw_depth.items()
            }

        if normalized_depth:
            maximum_depth = max(
                maximum_depth,
                max(
                    normalized_depth.values()
                ),
            )

        edges = record.get(
            "edges",
            [],
        )

        if edges is None:
            edges = []

        for edge in edges:
            if (
                not isinstance(
                    edge,
                    (list, tuple),
                )
                or len(edge) != 2
            ):
                raise ValueError(
                    f"Invalid edge: "
                    f"{patent_id}/{edge}"
                )

            parent_id = (
                normalize_claim_id_section2(
                    edge[0]
                )
            )

            child_id = (
                normalize_claim_id_section2(
                    edge[1]
                )
            )

            if (
                parent_id not in claim_ids
                or child_id not in claim_ids
            ):
                raise ValueError(
                    f"Unknown claim in edge: "
                    f"{patent_id}/"
                    f"{parent_id}->{child_id}"
                )

            number_of_edges += 1

            if (
                parent_id in normalized_depth
                and child_id in normalized_depth
            ):
                difference = (
                    normalized_depth[
                        child_id
                    ]
                    - normalized_depth[
                        parent_id
                    ]
                )

                if difference == 1:
                    number_of_adjacent_edges += 1
                elif difference > 1:
                    number_of_non_adjacent_edges += 1
                elif difference <= 0:
                    raise ValueError(
                        "Non-increasing dependency edge: "
                        f"{patent_id}/"
                        f"{parent_id}->{child_id}"
                    )

    expected_claims = (
        EXPECTED_CLAIM_COUNTS[
            split_name
        ]
    )

    if number_of_claims != expected_claims:
        raise RuntimeError(
            f"{split_name} claim count mismatch: "
            f"observed={number_of_claims:,}, "
            f"expected={expected_claims:,}"
        )

    return {
        "split": split_name,
        "record_keys": (
            first_record_keys
        ),
        "number_of_patents": int(
            len(records)
        ),
        "number_of_claims": int(
            number_of_claims
        ),
        "number_of_edges": int(
            number_of_edges
        ),
        "number_of_adjacent_edges": int(
            number_of_adjacent_edges
        ),
        "number_of_non_adjacent_edges": int(
            number_of_non_adjacent_edges
        ),
        "empty_text_claims": int(
            number_of_empty_claims
        ),
        "maximum_claims_per_patent": int(
            maximum_claims
        ),
        "maximum_depth": int(
            maximum_depth
        ),
    }


RECORD_STATISTICS = {}

for split_name, records in (
    RECORDS_BY_SPLIT.items()
):
    RECORD_STATISTICS[
        split_name
    ] = validate_records(
        split_name,
        records,
    )


# ============================================================
# 2.8 Vocabulary tokenizer
# ============================================================

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[-_][a-z0-9]+)*"
)

STANDARD_STOPWORDS = {
    str(word).lower()
    for word in ENGLISH_STOP_WORDS
}

PATENT_BOILERPLATE_STOPWORDS = {
    "according",
    "claim",
    "claims",
    "claimed",
    "comprise",
    "comprises",
    "comprising",
    "consist",
    "consists",
    "consisting",
    "thereof",
    "therein",
    "thereto",
    "whereby",
    "wherein",
    "whereof",
    "whereon",
    "whereupon",
    "said",
    "respective",
    "respectively",
    "plurality",
    "least",
    "one",
    "first",
    "second",
    "third",
    "fourth",
    "configured",
    "adapted",
    "provided",
    "providing",
    "including",
    "includes",
    "include",
    "having",
    "based",
    "associated",
    "corresponding",
    "method",
    "system",
    "apparatus",
    "device",
}

ALL_STOPWORDS = (
    STANDARD_STOPWORDS
    | PATENT_BOILERPLATE_STOPWORDS
)


def tokenize_claim_section2(
    text,
):
    text = (
        ""
        if text is None
        else str(text).lower()
    )

    raw_tokens = TOKEN_PATTERN.findall(
        text
    )

    tokens = []

    for token in raw_tokens:
        if len(token) < 3:
            continue

        if token in ALL_STOPWORDS:
            continue

        if not any(
            character.isalpha()
            for character in token
        ):
            continue

        tokens.append(
            token
        )

    return tokens


# ============================================================
# 2.9 Build train-only vocabulary
# ============================================================

term_frequency = Counter()
document_frequency = Counter()

number_of_train_claims = 0
empty_after_tokenization = 0

for record in tqdm(
    train_records,
    desc="Building train-only vocabulary",
):
    claims = normalize_claims_section2(
        record
    )

    for text in claims.values():
        tokens = tokenize_claim_section2(
            text
        )

        number_of_train_claims += 1

        if not tokens:
            empty_after_tokenization += 1
            continue

        term_frequency.update(
            tokens
        )

        document_frequency.update(
            set(tokens)
        )

if (
    number_of_train_claims
    != EXPECTED_CLAIM_COUNTS["train"]
):
    raise RuntimeError(
        "Vocabulary claim count mismatch."
    )

minimum_df = int(
    CONFIG.vocab_min_document_frequency
)

maximum_df = int(
    CONFIG.vocab_max_document_frequency_ratio
    * number_of_train_claims
)

eligible_tokens = [
    token
    for token, frequency
    in document_frequency.items()
    if (
        frequency >= minimum_df
        and frequency <= maximum_df
    )
]

eligible_tokens.sort(
    key=lambda token: (
        -term_frequency[token],
        -document_frequency[token],
        token,
    )
)

VOCAB = eligible_tokens[
    :int(CONFIG.vocab_size)
]

if len(VOCAB) != int(
    CONFIG.vocab_size
):
    raise RuntimeError(
        "8,000개 vocabulary를 만들 수 없습니다: "
        f"selected={len(VOCAB):,}"
    )

TOKEN_TO_ID = {
    token: index
    for index, token in enumerate(
        VOCAB
    )
}

VOCAB_SET = set(
    VOCAB
)

CONFIG.vocab_size = len(
    VOCAB
)


# ============================================================
# 2.10 Vocabulary coverage
# ============================================================

def vocabulary_coverage(
    split_name,
    records,
):
    total_tokens = 0
    vocabulary_tokens = 0
    number_of_claims = 0
    empty_bow_claims = 0
    empty_bow_patents = 0

    for record in tqdm(
        records,
        desc=f"Vocabulary coverage {split_name}",
    ):
        claims = normalize_claims_section2(
            record
        )

        valid_claims = 0

        for text in claims.values():
            tokens = tokenize_claim_section2(
                text
            )

            number_of_claims += 1
            total_tokens += len(
                tokens
            )

            valid_count = sum(
                token in VOCAB_SET
                for token in tokens
            )

            vocabulary_tokens += (
                valid_count
            )

            if valid_count == 0:
                empty_bow_claims += 1
            else:
                valid_claims += 1

        if valid_claims == 0:
            empty_bow_patents += 1

    return {
        "split": split_name,
        "number_of_claims": int(
            number_of_claims
        ),
        "total_filtered_tokens": int(
            total_tokens
        ),
        "vocabulary_tokens": int(
            vocabulary_tokens
        ),
        "token_coverage": float(
            vocabulary_tokens
            / max(total_tokens, 1)
        ),
        "empty_bow_claims": int(
            empty_bow_claims
        ),
        "empty_bow_ratio": float(
            empty_bow_claims
            / max(number_of_claims, 1)
        ),
        "empty_bow_patents": int(
            empty_bow_patents
        ),
    }


VOCABULARY_COVERAGE = {}

for split_name, records in (
    RECORDS_BY_SPLIT.items()
):
    VOCABULARY_COVERAGE[
        split_name
    ] = vocabulary_coverage(
        split_name,
        records,
    )


# ============================================================
# 2.11 Save outputs
# ============================================================

atomic_json_save_section2(
    {
        "run_name": (
            CONFIG.run_name
        ),
        "created_at": (
            datetime.now().isoformat()
        ),
        "source_split": "train",
        "vocabulary_size": int(
            len(VOCAB)
        ),
        "minimum_document_frequency": int(
            minimum_df
        ),
        "maximum_document_frequency": int(
            maximum_df
        ),
        "tokens": VOCAB,
    },
    VOCAB_PATH,
)

atomic_json_save_section2(
    TOKEN_TO_ID,
    TOKEN_TO_ID_PATH,
)

top_200_tokens = [
    {
        "rank": int(rank),
        "token": token,
        "term_frequency": int(
            term_frequency[token]
        ),
        "document_frequency": int(
            document_frequency[token]
        ),
    }
    for rank, token in enumerate(
        VOCAB[:200],
        start=1,
    )
]

vocabulary_statistics = {
    "run_name": (
        CONFIG.run_name
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "number_of_train_claims": int(
        number_of_train_claims
    ),
    "empty_after_tokenization": int(
        empty_after_tokenization
    ),
    "raw_unique_terms": int(
        len(term_frequency)
    ),
    "eligible_terms": int(
        len(eligible_tokens)
    ),
    "selected_vocabulary_size": int(
        len(VOCAB)
    ),
    "coverage": (
        VOCABULARY_COVERAGE
    ),
    "top_200_tokens": (
        top_200_tokens
    ),
    "stopwords": sorted(
        ALL_STOPWORDS
    ),
}

atomic_json_save_section2(
    vocabulary_statistics,
    VOCAB_STATS_PATH,
)

SECTION2_MANIFEST = {
    "run_name": (
        CONFIG.run_name
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "record_paths": {
        split_name: str(path)
        for split_name, path
        in RECORD_PATHS.items()
    },
    "record_statistics": (
        RECORD_STATISTICS
    ),
    "vocabulary_path": str(
        VOCAB_PATH
    ),
    "token_to_id_path": str(
        TOKEN_TO_ID_PATH
    ),
    "vocabulary_statistics_path": str(
        VOCAB_STATS_PATH
    ),
    "vocabulary_size": int(
        len(VOCAB)
    ),
    "vocabulary_source_split": "train",
    "vocabulary_coverage": (
        VOCABULARY_COVERAGE
    ),
}

atomic_json_save_section2(
    SECTION2_MANIFEST,
    SECTION2_MANIFEST_PATH,
)


# ============================================================
# 2.12 Final status
# ============================================================

print("\n" + "=" * 88)
print("SECTION 2 — RECORDS AND VOCABULARY COMPLETED")
print("=" * 88)

print(
    f"Record keys: "
    f"{RECORD_STATISTICS['train']['record_keys']}"
)

for split_name in [
    "train",
    "dev",
    "test",
]:
    statistics = (
        RECORD_STATISTICS[
            split_name
        ]
    )

    coverage = (
        VOCABULARY_COVERAGE[
            split_name
        ]
    )

    print(
        f"{split_name:5s}: "
        f"patents="
        f"{statistics['number_of_patents']:,}, "
        f"claims="
        f"{statistics['number_of_claims']:,}, "
        f"edges="
        f"{statistics['number_of_edges']:,}, "
        f"adjacent="
        f"{statistics['number_of_adjacent_edges']:,}, "
        f"coverage="
        f"{coverage['token_coverage']:.2%}, "
        f"empty-BoW="
        f"{coverage['empty_bow_ratio']:.4%}"
    )

print(
    f"\nVocabulary size : "
    f"{len(VOCAB):,}"
)

print(
    f"Raw unique terms: "
    f"{len(term_frequency):,}"
)

print(
    f"Eligible terms  : "
    f"{len(eligible_tokens):,}"
)

print("\nTop 30 vocabulary terms:")

for rank, token in enumerate(
    VOCAB[:30],
    start=1,
):
    print(
        f"{rank:2d}. "
        f"{token:24s} "
        f"TF={term_frequency[token]:,} "
        f"DF={document_frequency[token]:,}"
    )

print(
    f"\nVocabulary      : "
    f"{VOCAB_PATH}"
)

print(
    f"Statistics      : "
    f"{VOCAB_STATS_PATH}"
)

print(
    f"Manifest        : "
    f"{SECTION2_MANIFEST_PATH}"
)

print("=" * 88)
print(
    "[PASS] 기존 vocab.pkl은 수정하지 않았습니다."
)
print(
    "[PASS] 새 vocabulary는 train claim만 사용해 생성했습니다."
)
print(
    "[NEXT] 마지막 요약과 Top 30 vocabulary terms를 보내주세요."
)
