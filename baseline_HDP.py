#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Standalone Gensim HDP Baseline
# - Google Drive mount when running in Colab
# - Fixed 50,000-document training subset
# - Seeds: 42, 43, 44
# - Claim-level theta inference
# - Claim theta -> patent theta aggregation
# - Patent-level CPC evaluation: Pur_p, Pur_a, NMI
# - Per-seed and mean ± std saving
# - No invalid theta averaging across seeds
# ============================================================

# ============================================================
# 0. Google Drive mount
# ============================================================
try:
    from google.colab import drive

    drive.mount(
        "/content/drive",
        force_remount=False,
    )
except ImportError:
    print("[INFO] Not running in Colab; Drive mount skipped.")


# ============================================================
# 1. Package installation
# ============================================================
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "gensim": "gensim",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "pandas": "pandas",
    "tqdm": "tqdm",
}

missing_packages = [
    pip_name
    for module_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(module_name) is None
]

if missing_packages:
    print("Installing:", missing_packages)

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        *missing_packages,
    ])
else:
    print("[PASS] Required packages are installed.")


# ============================================================
# 2. Imports
# ============================================================
import os
import gc
import json
import time
import pickle
import random
import logging
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

import gensim
from gensim.models import HdpModel
from gensim.matutils import Sparse2Corpus

from sklearn.metrics import normalized_mutual_info_score
from tqdm.auto import tqdm


# ============================================================
# 3. Configuration
# ============================================================
PROJECT_ROOT = Path(
    os.environ.get(
        "DEPTH_OT_ROOT",
        "/content/drive/MyDrive/depth_ot_patent",
    )
)

# Input
TRAIN_BOW_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bow_train.npz"
)

TEST_BOW_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bow_test.npz"
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vocab.pkl"
)

TEST_RECORDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_records.pkl"
)

REF_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ref_test.pkl"
)

# 다른 baseline과 동일한 CPC reference
CPC_REFERENCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "cpc_alignment"
    / "etm_patent_predictions_by_seed.csv"
)

# HDP
SEEDS = [42, 43, 44]

HDP_SAMPLE_SEED = 42
HDP_TRAIN_SAMPLE_SIZE = 50_000

HDP_T = 150
HDP_K = 15

HDP_CHUNKSIZE = 1000
HDP_KAPPA = 0.9
HDP_TAU = 64.0
HDP_ALPHA = 1.0
HDP_GAMMA = 1.0
HDP_ETA = 0.01

HDP_INFERENCE_BATCH_SIZE = 2000
TOP_WORDS = 20

# Resume
REUSE_EXISTING_MODEL = True
REUSE_EXISTING_THETA = True
SAVE_MODEL = True

# Logging
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger(
    "gensim.models.hdpmodel"
).setLevel(logging.ERROR)


# ============================================================
# 4. Directories
# ============================================================
HDP_ROOT = PROJECT_ROOT / "topic_model" / "hdp"

MODEL_DIR = HDP_ROOT / "models"
THETA_DIR = HDP_ROOT / "theta"
TOPIC_DIR = HDP_ROOT / "topics"
LOG_DIR = HDP_ROOT / "logs"
SAMPLE_DIR = HDP_ROOT / "samples"

CPC_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cpc_alignment"
)

for directory in [
    HDP_ROOT,
    MODEL_DIR,
    THETA_DIR,
    TOPIC_DIR,
    LOG_DIR,
    SAMPLE_DIR,
    CPC_RESULT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

SAMPLE_IDX_PATH = SAMPLE_DIR / (
    f"hdp_train_sample_idx_n{HDP_TRAIN_SAMPLE_SIZE}"
    f"_seed{HDP_SAMPLE_SEED}.npy"
)

METRICS_PATH = (
    CPC_RESULT_DIR
    / "hdp_patent_cpc_alignment_by_seed.csv"
)

METRIC_SUMMARY_CSV_PATH = (
    CPC_RESULT_DIR
    / "hdp_patent_cpc_alignment_summary.csv"
)

SUMMARY_JSON_PATH = (
    CPC_RESULT_DIR
    / "hdp_patent_cpc_alignment_summary.json"
)

PREDICTIONS_PATH = (
    CPC_RESULT_DIR
    / "hdp_patent_predictions_by_seed.csv"
)

ERROR_LOG_PATH = (
    LOG_DIR
    / "hdp_baseline_errors.txt"
)


# ============================================================
# 5. Atomic save
# ============================================================
def atomic_save_json(obj, path):
    path = Path(path)
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            obj,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def atomic_save_npy(array, path):
    path = Path(path)
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(temporary_path, "wb") as file:
        np.save(file, array)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


# ============================================================
# 6. Data loading
# ============================================================
def validate_required_files():
    required_files = {
        "Train BoW": TRAIN_BOW_PATH,
        "Test BoW": TEST_BOW_PATH,
        "Vocabulary": VOCAB_PATH,
        "Test records": TEST_RECORDS_PATH,
        "Test reference": REF_TEST_PATH,
        "CPC reference": CPC_REFERENCE_PATH,
    }

    missing = []

    for name, path in required_files.items():
        if not path.is_file():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(missing)
        )


def load_sparse_matrix(path):
    print(f"Loading sparse matrix: {path}")

    matrix = sp.load_npz(path).tocsr()
    matrix.sort_indices()

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected 2-D matrix: {matrix.shape}"
        )

    if matrix.data.size:
        if not np.isfinite(matrix.data).all():
            raise FloatingPointError(
                f"{path.name} contains NaN/Inf."
            )

        if np.any(matrix.data < 0):
            raise ValueError(
                f"{path.name} contains negative values."
            )

    return matrix


def normalize_vocabulary(obj):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, pd.Series):
        obj = obj.tolist()

    if isinstance(obj, (list, tuple)):
        return [str(word) for word in obj]

    if isinstance(obj, dict):
        # {id: word}
        if all(
            isinstance(key, (int, np.integer))
            for key in obj
        ):
            return [
                str(obj[key])
                for key in sorted(obj)
            ]

        # {word: id}
        if all(
            isinstance(value, (int, np.integer))
            for value in obj.values()
        ):
            return [
                str(word)
                for word, _ in sorted(
                    obj.items(),
                    key=lambda item: int(item[1]),
                )
            ]

        for key in [
            "vocab",
            "vocabulary",
            "words",
            "tokens",
            "id2word",
        ]:
            if key in obj:
                return normalize_vocabulary(obj[key])

    if hasattr(obj, "token2id"):
        return [
            str(word)
            for word, _ in sorted(
                obj.token2id.items(),
                key=lambda item: int(item[1]),
            )
        ]

    raise TypeError(
        f"Unsupported vocabulary type: {type(obj)}"
    )


def load_vocabulary(path):
    print(f"Loading vocabulary: {path}")

    with open(path, "rb") as file:
        obj = pickle.load(file)

    vocabulary = normalize_vocabulary(obj)

    if not vocabulary:
        raise ValueError("Vocabulary is empty.")

    return vocabulary


# ============================================================
# 7. Patent ID/reference handling
# ============================================================
PATENT_ID_KEYS = [
    "patent_id",
    "patent_ids",
    "publication_number",
    "doc_id",
    "document_id",
]


def normalize_patent_ids(values):
    normalized = (
        pd.Series(np.asarray(values).ravel())
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .to_numpy()
    )

    return normalized


def extract_record_patent_ids(records):
    if isinstance(records, pd.DataFrame):
        for key in PATENT_ID_KEYS:
            if key in records.columns:
                return normalize_patent_ids(
                    records[key].to_numpy()
                )

    if isinstance(records, dict):
        for key in PATENT_ID_KEYS:
            if key in records:
                return normalize_patent_ids(
                    records[key]
                )

        for nested_key in [
            "records",
            "data",
            "items",
            "examples",
        ]:
            if nested_key in records:
                try:
                    return extract_record_patent_ids(
                        records[nested_key]
                    )
                except Exception:
                    pass

    if isinstance(records, (list, tuple, np.ndarray)):
        if len(records) == 0:
            raise ValueError(
                "test_records is empty."
            )

        first = records[0]

        if isinstance(first, dict):
            for key in PATENT_ID_KEYS:
                if key in first:
                    return normalize_patent_ids([
                        record[key]
                        for record in records
                    ])

        for key in PATENT_ID_KEYS:
            if hasattr(first, key):
                return normalize_patent_ids([
                    getattr(record, key)
                    for record in records
                ])

    raise TypeError(
        "Could not extract patent_id from test_records."
    )


def reference_candidates(ref_test):
    candidates = []

    if isinstance(ref_test, dict):
        candidate_keys = [
            "patent_index",
            "patent_indices",
            "record_index",
            "record_indices",
            "patent_id",
            "patent_ids",
            "ref",
            "references",
            "indices",
        ]

        for key in candidate_keys:
            if key in ref_test:
                candidates.extend(
                    reference_candidates(ref_test[key])
                )

        return candidates

    values = np.asarray(ref_test)

    if values.ndim == 1:
        candidates.append(values)

    elif values.ndim == 2:
        for column in range(values.shape[1]):
            candidates.append(values[:, column])

    return candidates


def resolve_claim_patent_ids(
    ref_test,
    record_patent_ids,
    expected_claims,
):
    number_of_patents = len(record_patent_ids)
    record_id_set = set(record_patent_ids)

    # Nested list: one list per patent
    if (
        isinstance(ref_test, (list, tuple))
        and len(ref_test) == number_of_patents
        and len(ref_test) > 0
        and isinstance(
            ref_test[0],
            (list, tuple, np.ndarray),
        )
    ):
        lengths = np.asarray([
            len(item) for item in ref_test
        ])

        if lengths.sum() == expected_claims:
            return (
                np.repeat(
                    record_patent_ids,
                    lengths,
                ),
                "nested references by patent",
            )

    candidates = reference_candidates(ref_test)

    for candidate in candidates:
        candidate = np.asarray(candidate).ravel()

        if len(candidate) != expected_claims:
            continue

        # Direct patent IDs
        direct_ids = normalize_patent_ids(candidate)

        direct_overlap = np.mean([
            patent_id in record_id_set
            for patent_id in direct_ids
        ])

        if direct_overlap >= 0.99:
            return direct_ids, "direct patent IDs"

        # Numeric record indices
        try:
            indices = candidate.astype(np.int64)
        except (ValueError, TypeError):
            continue

        if (
            indices.min() >= 0
            and indices.max() < number_of_patents
        ):
            return (
                record_patent_ids[indices],
                "zero-based record indices",
            )

        if (
            indices.min() >= 1
            and indices.max() <= number_of_patents
        ):
            return (
                record_patent_ids[indices - 1],
                "one-based record indices",
            )

    raise ValueError(
        "Could not resolve claim-to-patent mapping from "
        "ref_test.pkl."
    )


# ============================================================
# 8. CPC reference
# ============================================================
def load_cpc_reference(path):
    reference = pd.read_csv(
        path,
        dtype={"patent_id": str},
    )

    required_columns = [
        "patent_id",
        "section",
        "class",
        "subclass",
    ]

    missing = [
        column
        for column in required_columns
        if column not in reference.columns
    ]

    if missing:
        raise KeyError(
            f"Missing CPC columns: {missing}"
        )

    reference = reference[
        required_columns
    ].copy()

    reference["patent_id"] = normalize_patent_ids(
        reference["patent_id"]
    )

    reference = reference.drop_duplicates(
        subset=["patent_id"]
    ).reset_index(drop=True)

    return reference


# ============================================================
# 9. HDP inference
# ============================================================
def infer_hdp_theta(
    model,
    bow_matrix,
    batch_size,
):
    number_of_documents = bow_matrix.shape[0]
    number_of_topics = int(model.m_T)

    theta = np.zeros(
        (number_of_documents, number_of_topics),
        dtype=np.float32,
    )

    progress = tqdm(
        range(0, number_of_documents, batch_size),
        desc="HDP test inference",
    )

    for start in progress:
        end = min(
            start + batch_size,
            number_of_documents,
        )

        batch_matrix = bow_matrix[
            start:end
        ].tocsr()

        batch_nnz = np.asarray(
            batch_matrix.getnnz(axis=1)
        ).ravel()

        nonempty_indices = np.flatnonzero(
            batch_nnz > 0
        )

        if len(nonempty_indices) == 0:
            continue

        nonempty_matrix = batch_matrix[
            nonempty_indices
        ].tocsr()

        batch_corpus = Sparse2Corpus(
            nonempty_matrix,
            documents_columns=False,
        )

        gamma = model.inference(
            list(batch_corpus)
        )

        gamma = np.asarray(
            gamma,
            dtype=np.float64,
        )

        if gamma.shape != (
            len(nonempty_indices),
            number_of_topics,
        ):
            raise ValueError(
                f"Unexpected gamma shape: {gamma.shape}"
            )

        if not np.isfinite(gamma).all():
            raise FloatingPointError(
                f"Non-finite gamma: {start}:{end}"
            )

        row_sums = gamma.sum(
            axis=1,
            keepdims=True,
        )

        normalized_gamma = np.divide(
            gamma,
            row_sums,
            out=np.zeros_like(gamma),
            where=row_sums > 0,
        )

        theta[
            start + nonempty_indices
        ] = normalized_gamma.astype(np.float32)

        progress.set_postfix(
            processed=f"{end:,}/{number_of_documents:,}",
            refresh=False,
        )

    if not np.isfinite(theta).all():
        raise FloatingPointError(
            "Final theta contains NaN/Inf."
        )

    valid = theta.sum(axis=1) > 0

    if np.any(valid):
        row_sums = theta[valid].sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                "Theta rows are not normalized."
            )

    return theta


# ============================================================
# 10. Claim theta -> Patent theta
# ============================================================
def aggregate_claim_theta_to_patent(
    claim_theta,
    claim_patent_ids,
):
    valid_claims = np.asarray(
        claim_theta.sum(axis=1) > 0
    )

    valid_ids = claim_patent_ids[valid_claims]

    valid_theta = np.asarray(
        claim_theta[valid_claims],
        dtype=np.float64,
    )

    patent_ids, inverse_indices = np.unique(
        valid_ids,
        return_inverse=True,
    )

    patent_theta_sum = np.zeros(
        (
            len(patent_ids),
            claim_theta.shape[1],
        ),
        dtype=np.float64,
    )

    patent_claim_counts = np.zeros(
        len(patent_ids),
        dtype=np.int64,
    )

    np.add.at(
        patent_theta_sum,
        inverse_indices,
        valid_theta,
    )

    np.add.at(
        patent_claim_counts,
        inverse_indices,
        1,
    )

    patent_theta = patent_theta_sum / np.maximum(
        patent_claim_counts[:, None],
        1,
    )

    row_sums = patent_theta.sum(
        axis=1,
        keepdims=True,
    )

    patent_theta = np.divide(
        patent_theta,
        row_sums,
        out=np.zeros_like(patent_theta),
        where=row_sums > 0,
    )

    if not np.isfinite(patent_theta).all():
        raise FloatingPointError(
            "Patent theta contains NaN/Inf."
        )

    return (
        normalize_patent_ids(patent_ids),
        patent_theta,
        patent_claim_counts,
    )


# ============================================================
# 11. Evaluation
# ============================================================
def clustering_metrics(
    true_labels,
    predicted_topics,
):
    true_labels = np.asarray(true_labels)
    predicted_topics = np.asarray(predicted_topics)

    valid = (
        ~pd.isna(true_labels)
        & ~pd.isna(predicted_topics)
    )

    true_labels = true_labels[valid].astype(str)
    predicted_topics = predicted_topics[valid]

    if len(true_labels) == 0:
        return {
            "n_documents": 0,
            "Pur_p": np.nan,
            "Pur_a": np.nan,
            "NMI": np.nan,
        }

    contingency = pd.crosstab(
        pd.Series(
            predicted_topics,
            name="predicted_topic",
        ),
        pd.Series(
            true_labels,
            name="actual_label",
        ),
        dropna=False,
    )

    total = contingency.to_numpy().sum()

    pur_p = (
        contingency.max(axis=1).sum()
        / total
    )

    pur_a = (
        contingency.max(axis=0).sum()
        / total
    )

    nmi = normalized_mutual_info_score(
        true_labels,
        predicted_topics.astype(str),
    )

    return {
        "n_documents": int(total),
        "Pur_p": float(pur_p),
        "Pur_a": float(pur_a),
        "NMI": float(nmi),
    }


# ============================================================
# 12. Top words
# ============================================================
def save_top_words(model, seed):
    output_path = (
        TOPIC_DIR
        / f"hdp_top_words_seed{seed}.json"
    )

    topics = []

    for topic_id in range(int(model.m_T)):
        topic_words = model.show_topic(
            topic_id,
            topn=TOP_WORDS,
        )

        topics.append({
            "topic_id": int(topic_id),
            "words": [
                {
                    "word": str(word),
                    "weight": float(weight),
                }
                for word, weight in topic_words
            ],
        })

    atomic_save_json(
        topics,
        output_path,
    )

    return output_path


# ============================================================
# 13. Main
# ============================================================
def main():
    validate_required_files()

    print("\n=== Loading data ===")

    bow_train = load_sparse_matrix(
        TRAIN_BOW_PATH
    )

    bow_test = load_sparse_matrix(
        TEST_BOW_PATH
    )

    vocabulary = load_vocabulary(
        VOCAB_PATH
    )

    if bow_train.shape[1] != len(vocabulary):
        raise ValueError(
            "Train BoW/vocabulary mismatch."
        )

    if bow_test.shape[1] != len(vocabulary):
        raise ValueError(
            "Test BoW/vocabulary mismatch."
        )

    with open(TEST_RECORDS_PATH, "rb") as file:
        test_records = pickle.load(file)

    with open(REF_TEST_PATH, "rb") as file:
        ref_test = pickle.load(file)

    record_patent_ids = extract_record_patent_ids(
        test_records
    )

    claim_patent_ids, reference_type = (
        resolve_claim_patent_ids(
            ref_test=ref_test,
            record_patent_ids=record_patent_ids,
            expected_claims=bow_test.shape[0],
        )
    )

    if len(claim_patent_ids) != bow_test.shape[0]:
        raise ValueError(
            "Claim-patent mapping length mismatch."
        )

    cpc_reference = load_cpc_reference(
        CPC_REFERENCE_PATH
    )

    print("\n=== Data ===")
    print(f"Train BoW       : {bow_train.shape}")
    print(f"Test BoW        : {bow_test.shape}")
    print(f"Vocabulary      : {len(vocabulary):,}")
    print(f"Patent records  : {len(record_patent_ids):,}")
    print(f"Claim references: {len(claim_patent_ids):,}")
    print(f"Unique patents  : {len(np.unique(claim_patent_ids)):,}")
    print(f"Reference type  : {reference_type}")
    print(f"CPC references  : {len(cpc_reference):,}")

    # --------------------------------------------------------
    # Fixed training sample
    # --------------------------------------------------------
    nonempty_train_indices = np.flatnonzero(
        np.asarray(
            bow_train.getnnz(axis=1)
        ).ravel() > 0
    )

    sample_size = min(
        HDP_TRAIN_SAMPLE_SIZE,
        len(nonempty_train_indices),
    )

    if SAMPLE_IDX_PATH.is_file():
        sample_indices = np.load(
            SAMPLE_IDX_PATH
        )

        valid_sample = (
            sample_indices.ndim == 1
            and len(sample_indices) == sample_size
            and sample_indices.min() >= 0
            and sample_indices.max() < bow_train.shape[0]
        )

        if not valid_sample:
            raise ValueError(
                f"Invalid saved sample: {SAMPLE_IDX_PATH}"
            )

        print(
            f"[LOAD] Training sample: {SAMPLE_IDX_PATH}"
        )

    else:
        sample_rng = np.random.RandomState(
            HDP_SAMPLE_SEED
        )

        sample_indices = sample_rng.choice(
            nonempty_train_indices,
            size=sample_size,
            replace=False,
        ).astype(np.int64)

        atomic_save_npy(
            sample_indices,
            SAMPLE_IDX_PATH,
        )

        print(
            f"[SAVE] Training sample: {SAMPLE_IDX_PATH}"
        )

    bow_train_sample = bow_train[
        sample_indices
    ].tocsr()

    id2word = {
        index: str(word)
        for index, word in enumerate(vocabulary)
    }

    print("\n=== HDP configuration ===")
    print(f"Seeds            : {SEEDS}")
    print(f"Training sample  : {sample_size:,}")
    print(f"T / K            : {HDP_T} / {HDP_K}")
    print(f"Chunksize        : {HDP_CHUNKSIZE:,}")
    print(f"Inference batch  : {HDP_INFERENCE_BATCH_SIZE:,}")
    print("Device           : CPU")

    metric_rows = []
    run_records = []
    prediction_table = None
    successful_seeds = []
    failed_seeds = []

    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"HDP SEED {seed}")
        print("=" * 70)

        random.seed(seed)
        np.random.seed(seed)

        model_path = MODEL_DIR / (
            f"hdp_seed{seed}"
            f"_sample{sample_size}"
            f"_T{HDP_T}_K{HDP_K}.model"
        )

        theta_path = THETA_DIR / (
            f"hdp_theta_test_seed{seed}"
            f"_sample{sample_size}"
            f"_T{HDP_T}_K{HDP_K}.npy"
        )

        try:
            # ------------------------------------------------
            # Train/load model
            # ------------------------------------------------
            if (
                REUSE_EXISTING_MODEL
                and model_path.is_file()
            ):
                print(
                    f"[LOAD] HDP model: {model_path}"
                )

                model = HdpModel.load(
                    str(model_path)
                )

                training_seconds = 0.0
                reused_model = True

            else:
                print(f"[TRAIN] HDP seed={seed}")

                train_corpus = Sparse2Corpus(
                    bow_train_sample,
                    documents_columns=False,
                )

                training_start = time.time()

                model = HdpModel(
                    corpus=train_corpus,
                    id2word=id2word,
                    random_state=seed,
                    T=HDP_T,
                    K=HDP_K,
                    chunksize=HDP_CHUNKSIZE,
                    kappa=HDP_KAPPA,
                    tau=HDP_TAU,
                    alpha=HDP_ALPHA,
                    gamma=HDP_GAMMA,
                    eta=HDP_ETA,
                )

                training_seconds = (
                    time.time() - training_start
                )

                reused_model = False

                if SAVE_MODEL:
                    model.save(str(model_path))

                print(
                    f"[DONE] Training: "
                    f"{training_seconds / 60:.2f} min"
                )

            if int(model.m_T) != HDP_T:
                raise ValueError(
                    f"Model T={model.m_T}, expected={HDP_T}"
                )

            # ------------------------------------------------
            # Infer/load theta
            # ------------------------------------------------
            if (
                REUSE_EXISTING_THETA
                and theta_path.is_file()
            ):
                print(
                    f"[LOAD] Claim theta: {theta_path}"
                )

                claim_theta = np.load(
                    theta_path,
                    mmap_mode="r",
                )

                inference_seconds = 0.0
                reused_theta = True

            else:
                print(
                    f"[INFERENCE] HDP seed={seed}"
                )

                inference_start = time.time()

                claim_theta = infer_hdp_theta(
                    model=model,
                    bow_matrix=bow_test,
                    batch_size=HDP_INFERENCE_BATCH_SIZE,
                )

                inference_seconds = (
                    time.time() - inference_start
                )

                atomic_save_npy(
                    claim_theta,
                    theta_path,
                )

                reused_theta = False

                print(
                    f"[DONE] Inference: "
                    f"{inference_seconds / 60:.2f} min"
                )

            expected_shape = (
                bow_test.shape[0],
                HDP_T,
            )

            if claim_theta.shape != expected_shape:
                raise ValueError(
                    f"Theta shape={claim_theta.shape}, "
                    f"expected={expected_shape}"
                )

            # ------------------------------------------------
            # Claim -> patent aggregation
            # ------------------------------------------------
            (
                patent_ids,
                patent_theta,
                patent_claim_counts,
            ) = aggregate_claim_theta_to_patent(
                claim_theta=claim_theta,
                claim_patent_ids=claim_patent_ids,
            )

            predicted_topics = patent_theta.argmax(
                axis=1
            ).astype(np.int32)

            predicted_probabilities = patent_theta.max(
                axis=1
            )

            seed_predictions = pd.DataFrame({
                "patent_id": patent_ids,
                "num_claims": patent_claim_counts,
                f"dominant_topic_seed{seed}":
                    predicted_topics,
                f"dominant_probability_seed{seed}":
                    predicted_probabilities,
            })

            evaluation_data = cpc_reference.merge(
                seed_predictions,
                on="patent_id",
                how="inner",
                validate="one_to_one",
            )

            if len(evaluation_data) == 0:
                raise ValueError(
                    "No patent IDs matched CPC reference."
                )

            match_ratio = (
                len(evaluation_data)
                / len(seed_predictions)
            )

            if match_ratio < 0.95:
                raise ValueError(
                    f"Low CPC matching ratio: "
                    f"{match_ratio:.2%}"
                )

            topic_column = (
                f"dominant_topic_seed{seed}"
            )

            metric_row = {
                "seed": int(seed),
                "num_claims": int(
                    patent_claim_counts.sum()
                ),
                "num_patents": int(
                    len(evaluation_data)
                ),
                "active_topics": int(
                    evaluation_data[
                        topic_column
                    ].nunique()
                ),
                "mean_claims_per_patent": float(
                    evaluation_data[
                        "num_claims"
                    ].mean()
                ),
            }

            print("\n=== CPC evaluation ===")

            for level in [
                "section",
                "class",
                "subclass",
            ]:
                valid = evaluation_data[
                    level
                ].notna()

                result = clustering_metrics(
                    true_labels=evaluation_data.loc[
                        valid,
                        level,
                    ].to_numpy(),
                    predicted_topics=evaluation_data.loc[
                        valid,
                        topic_column,
                    ].to_numpy(),
                )

                metric_row[
                    f"{level}_Pur_p"
                ] = result["Pur_p"]

                metric_row[
                    f"{level}_Pur_a"
                ] = result["Pur_a"]

                metric_row[
                    f"{level}_NMI"
                ] = result["NMI"]

                print(
                    f"{level:8s} | "
                    f"Pur_p={result['Pur_p']:.4f} | "
                    f"Pur_a={result['Pur_a']:.4f} | "
                    f"NMI={result['NMI']:.4f}"
                )

            metric_rows.append(metric_row)

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------
            current_predictions = evaluation_data[[
                "patent_id",
                "section",
                "class",
                "subclass",
                "num_claims",
                topic_column,
                f"dominant_probability_seed{seed}",
            ]].copy()

            if prediction_table is None:
                prediction_table = current_predictions

            else:
                prediction_table = prediction_table.merge(
                    current_predictions[[
                        "patent_id",
                        topic_column,
                        f"dominant_probability_seed{seed}",
                    ]],
                    on="patent_id",
                    how="inner",
                    validate="one_to_one",
                )

            top_words_path = save_top_words(
                model=model,
                seed=seed,
            )

            run_records.append({
                "seed": int(seed),
                "status": "success",
                "training_seconds": float(
                    training_seconds
                ),
                "inference_seconds": float(
                    inference_seconds
                ),
                "reused_model": bool(reused_model),
                "reused_theta": bool(reused_theta),
                "model_path": str(model_path),
                "theta_path": str(theta_path),
                "top_words_path": str(
                    top_words_path
                ),
                "claim_theta_shape": list(
                    claim_theta.shape
                ),
                "evaluated_patents": int(
                    len(evaluation_data)
                ),
                "active_topics": int(
                    metric_row["active_topics"]
                ),
            })

            successful_seeds.append(seed)

            del model
            del claim_theta
            del patent_theta
            del evaluation_data
            gc.collect()

        except Exception as error:
            failed_seeds.append(seed)

            error_traceback = traceback.format_exc()

            print(
                f"\n[FAILED] seed={seed}: {error}"
            )
            print(error_traceback)

            run_records.append({
                "seed": int(seed),
                "status": "failed",
                "error": str(error),
                "traceback": error_traceback,
            })

            with open(
                ERROR_LOG_PATH,
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    f"\nseed={seed}: {error}\n"
                )
                file.write(error_traceback)
                file.write("\n")

            gc.collect()

    # --------------------------------------------------------
    # Save per-seed metrics
    # --------------------------------------------------------
    if not metric_rows:
        raise RuntimeError(
            "No HDP seed was evaluated successfully."
        )

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if prediction_table is not None:
        prediction_table.to_csv(
            PREDICTIONS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Mean ± standard deviation
    # --------------------------------------------------------
    summary_rows = []

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        summary_row = {
            "model": "HDP",
            "level": level,
            "successful_seeds": len(
                successful_seeds
            ),
        }

        for metric in [
            "Pur_p",
            "Pur_a",
            "NMI",
        ]:
            column = f"{level}_{metric}"

            summary_row[
                f"{metric}_mean"
            ] = float(metrics_df[column].mean())

            summary_row[
                f"{metric}_std"
            ] = (
                float(
                    metrics_df[column].std(ddof=1)
                )
                if len(metrics_df) > 1
                else 0.0
            )

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        METRIC_SUMMARY_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    final_summary = {
        "model": "HDP",
        "created_at": datetime.now().isoformat(),
        "configuration": {
            "training_sample_size": int(
                sample_size
            ),
            "sample_seed": int(
                HDP_SAMPLE_SEED
            ),
            "seeds": [
                int(seed) for seed in SEEDS
            ],
            "T": int(HDP_T),
            "K": int(HDP_K),
            "chunksize": int(
                HDP_CHUNKSIZE
            ),
            "kappa": float(
                HDP_KAPPA
            ),
            "tau": float(
                HDP_TAU
            ),
            "alpha": float(
                HDP_ALPHA
            ),
            "gamma": float(
                HDP_GAMMA
            ),
            "eta": float(
                HDP_ETA
            ),
            "aggregation": (
                "mean claim theta per patent"
            ),
            "theta_ensemble": False,
        },
        "data": {
            "train_bow_path": str(
                TRAIN_BOW_PATH
            ),
            "test_bow_path": str(
                TEST_BOW_PATH
            ),
            "test_records_path": str(
                TEST_RECORDS_PATH
            ),
            "ref_test_path": str(
                REF_TEST_PATH
            ),
            "cpc_reference_path": str(
                CPC_REFERENCE_PATH
            ),
            "train_shape": list(
                bow_train.shape
            ),
            "test_shape": list(
                bow_test.shape
            ),
            "patent_records": int(
                len(record_patent_ids)
            ),
            "reference_type": reference_type,
        },
        "successful_seeds": [
            int(seed)
            for seed in successful_seeds
        ],
        "failed_seeds": [
            int(seed)
            for seed in failed_seeds
        ],
        "per_seed_metrics": metric_rows,
        "mean_std": summary_rows,
        "runs": run_records,
        "environment": {
            "python": sys.version,
            "gensim": gensim.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "note": (
            "Models were evaluated independently per seed. "
            "Theta matrices were not averaged because topic "
            "indices are not aligned across independent runs."
        ),
    }

    atomic_save_json(
        final_summary,
        SUMMARY_JSON_PATH,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("HDP BASELINE: MEAN ± STD")
    print("=" * 70)

    for row in summary_rows:
        print(
            f"[{row['level']}] "
            f"Pur_p={row['Pur_p_mean']:.4f}"
            f"±{row['Pur_p_std']:.4f} | "
            f"Pur_a={row['Pur_a_mean']:.4f}"
            f"±{row['Pur_a_std']:.4f} | "
            f"NMI={row['NMI_mean']:.4f}"
            f"±{row['NMI_std']:.4f}"
        )

    print("\nSuccessful seeds:", successful_seeds)
    print("Failed seeds    :", failed_seeds)
    print("Per-seed metrics:", METRICS_PATH)
    print("Summary CSV     :", METRIC_SUMMARY_CSV_PATH)
    print("Summary JSON    :", SUMMARY_JSON_PATH)
    print("Predictions     :", PREDICTIONS_PATH)

    if failed_seeds:
        raise RuntimeError(
            f"Failed HDP seeds: {failed_seeds}"
        )

    print("\n[PASS] HDP baseline completed successfully.")


if __name__ == "__main__":
    main()
