# ============================================================
# Depth-OT V2: Claim Text + Theta Fusion DEV Experiment
# Copy and run this entire cell in a new Google Colab notebook.
# ============================================================

!pip -q install -U sentence-transformers scikit-learn scipy pandas tqdm joblib

from google.colab import drive
drive.mount("/content/drive")

import os
import re
import json
import time
import pickle
import warnings
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# 0. Configuration
# ============================================================

RUN_ROOT = Path(
    "/content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/"
    "depth_ot_v2_patent_semantic_seed42_20260814_055110"
)

THETA_PATH = (
    RUN_ROOT
    / "epoch016_independent_depth_confidence_dev_search"
    / "a08_l010_g000_dev_theta.npy"
)

RECORDS_PATH = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/dev_records.pkl"
)

OUTPUT_DIR = RUN_ROOT / "epoch016_text_theta_fusion_dev_search"
CACHE_DIR = OUTPUT_DIR / "cache"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PATENTS = 9855
N_CLUSTERS = 30

RANDOM_SEEDS = [17, 42, 73]
SPLIT_SEED = 20260816
TUNE_RATIO = 0.70

SEMANTIC_MODELS = [
    "AI-Growth-Lab/PatentSBERTa",
    "sentence-transformers/all-mpnet-base-v2",
]

SEMANTIC_PCA_DIM = 96
LEXICAL_SVD_DIM = 128
MAX_TEXT_CHARS = 14000
BATCH_SIZE = 32

PREVIOUS_STABLE_HYBRID_NMI = 0.310544
TARGET_NMI = 0.312000

# name, theta power, theta weight, semantic weight, lexical weight
FUSION_CONFIGS = [
    ("theta_q050", 0.50, 1.00, 0.00, 0.00),
    ("theta_q075", 0.75, 1.00, 0.00, 0.00),
    ("theta_q100", 1.00, 1.00, 0.00, 0.00),

    ("q050_t80_s20", 0.50, 0.80, 0.20, 0.00),
    ("q075_t80_s20", 0.75, 0.80, 0.20, 0.00),
    ("q100_t80_s20", 1.00, 0.80, 0.20, 0.00),

    ("q050_t70_s30", 0.50, 0.70, 0.30, 0.00),
    ("q075_t70_s30", 0.75, 0.70, 0.30, 0.00),
    ("q100_t70_s30", 1.00, 0.70, 0.30, 0.00),

    ("q050_t65_s25_l10", 0.50, 0.65, 0.25, 0.10),
    ("q075_t65_s25_l10", 0.75, 0.65, 0.25, 0.10),
    ("q100_t65_s25_l10", 1.00, 0.65, 0.25, 0.10),

    ("q050_t60_s30_l10", 0.50, 0.60, 0.30, 0.10),
    ("q075_t60_s30_l10", 0.75, 0.60, 0.30, 0.10),
    ("q100_t60_s30_l10", 1.00, 0.60, 0.30, 0.10),

    ("q050_t50_s35_l15", 0.50, 0.50, 0.35, 0.15),
    ("q075_t50_s35_l15", 0.75, 0.50, 0.35, 0.15),
    ("q100_t50_s35_l15", 1.00, 0.50, 0.35, 0.15),

    ("q050_t40_s40_l20", 0.50, 0.40, 0.40, 0.20),
    ("q075_t40_s40_l20", 0.75, 0.40, 0.40, 0.20),
    ("q100_t40_s40_l20", 1.00, 0.40, 0.40, 0.20),

    ("q050_t30_s50_l20", 0.50, 0.30, 0.50, 0.20),
    ("q075_t30_s50_l20", 0.75, 0.30, 0.50, 0.20),
    ("q100_t30_s50_l20", 1.00, 0.30, 0.50, 0.20),

    ("semantic_lexical", 1.00, 0.00, 0.75, 0.25),
]

print("=" * 80)
print("DEPTH-OT V2: TEXT + THETA FUSION")
print("=" * 80)
print("Theta:", THETA_PATH)
print("Records:", RECORDS_PATH)
print("Output:", OUTPUT_DIR)

assert THETA_PATH.exists(), f"Missing theta: {THETA_PATH}"
assert RECORDS_PATH.exists(), f"Missing records: {RECORDS_PATH}"

start_time = time.time()

# ============================================================
# 1. Remove invalid caches from the previous failed run
# ============================================================

bad_cache_names = [
    "dev_patent_texts.pkl",
    "semantic_embeddings_raw.npy",
    "semantic_embeddings_pca96.npy",
    "semantic_model.json",
    "lexical_tfidf_svd128.npy",
]

print("\n" + "=" * 80)
print("RESETTING PREVIOUS INVALID CACHES")
print("=" * 80)

for name in bad_cache_names:
    path = CACHE_DIR / name

    if path.exists():
        path.unlink()
        print("Deleted:", path)

print("Cache reset complete.")

# ============================================================
# 2. Helpers for loading records and claims
# ============================================================

def to_plain_dict(record):
    if isinstance(record, dict):
        return record

    if hasattr(record, "__dict__"):
        return vars(record)

    return {}


def claim_number(value):
    try:
        return int(value)
    except Exception:
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else 10**9


def get_claim_depth(depth_map, claim_id):
    candidates = [
        claim_id,
        str(claim_id),
        claim_number(claim_id),
    ]

    for candidate in candidates:
        if candidate in depth_map:
            try:
                return int(depth_map[candidate])
            except Exception:
                continue

    return 999


def extract_patent_text(record):
    """
    Extract all claims while placing independent/root claims first.
    Root-first ordering is important because SentenceTransformer models
    truncate documents exceeding their maximum token length.
    """
    record = to_plain_dict(record)

    claims = record.get("claims", {})
    depth_map = record.get("depth", {})

    if not isinstance(claims, dict) or len(claims) == 0:
        return "empty patent document"

    ordered_ids = sorted(
        claims.keys(),
        key=claim_number,
    )

    root_ids = [
        claim_id
        for claim_id in ordered_ids
        if get_claim_depth(depth_map, claim_id) == 0
    ]

    if not root_ids and ordered_ids:
        root_ids = [ordered_ids[0]]

    root_set = set(root_ids)

    dependent_ids = [
        claim_id
        for claim_id in ordered_ids
        if claim_id not in root_set
    ]

    text_parts = []

    for claim_id in root_ids:
        claim_text = re.sub(
            r"\s+",
            " ",
            str(claims[claim_id]),
        ).strip()

        if claim_text:
            text_parts.append(
                f"INDEPENDENT CLAIM {claim_id}: {claim_text}"
            )

    for claim_id in dependent_ids:
        claim_text = re.sub(
            r"\s+",
            " ",
            str(claims[claim_id]),
        ).strip()

        if claim_text:
            depth = get_claim_depth(depth_map, claim_id)

            text_parts.append(
                f"DEPENDENT CLAIM {claim_id} DEPTH {depth}: {claim_text}"
            )

    patent_text = " ".join(text_parts)
    patent_text = re.sub(r"\s+", " ", patent_text).strip()

    if not patent_text:
        return "empty patent document"

    return patent_text[:MAX_TEXT_CHARS]


# ============================================================
# 3. CPC label extraction
# ============================================================

CPC_PATTERN = re.compile(
    r"\b([A-HY])\s*(\d{2})\s*([A-Z])",
    re.I,
)


def extract_cpc_labels(record):
    record = to_plain_dict(record)

    section = str(record.get("section", "")).strip().upper()
    cpc_class = str(record.get("class", "")).strip().upper()
    subclass = str(record.get("subclass", "")).strip().upper()

    if section and cpc_class and subclass:
        return section, cpc_class, subclass

    codes = record.get("cpc_codes", [])

    if isinstance(codes, str):
        codes = [codes]

    for code in codes:
        match = CPC_PATTERN.search(str(code).upper())

        if match:
            sec, digits, letter = match.groups()
            return sec, sec + digits, sec + digits + letter

    return "UNK", "UNK", "UNK"


# ============================================================
# 4. Load theta and records
# ============================================================

theta = np.load(THETA_PATH).astype(np.float32)

with open(RECORDS_PATH, "rb") as file:
    records = pickle.load(file)

if isinstance(records, dict):
    for possible_key in ["records", "data", "items", "patents"]:
        if (
            possible_key in records
            and isinstance(records[possible_key], (list, tuple))
        ):
            records = records[possible_key]
            break

records = list(records)

print("\n" + "=" * 80)
print("INPUT VALIDATION")
print("=" * 80)
print("Theta shape:", theta.shape)
print("Number of records:", len(records))

assert theta.ndim == 2
assert theta.shape[0] == EXPECTED_PATENTS
assert theta.shape[1] == N_CLUSTERS
assert theta.shape[0] == len(records)
assert np.isfinite(theta).all()
assert (theta >= 0).all()

theta = np.maximum(theta, 1e-12)
theta = theta / theta.sum(axis=1, keepdims=True)

labels = [
    extract_cpc_labels(record)
    for record in records
]

section_labels = np.array(
    [label[0] for label in labels],
    dtype=str,
)

class_labels = np.array(
    [label[1] for label in labels],
    dtype=str,
)

subclass_labels = np.array(
    [label[2] for label in labels],
    dtype=str,
)

missing_mask = (
    (section_labels == "UNK")
    | (class_labels == "UNK")
    | (subclass_labels == "UNK")
)

print("Missing CPC labels:", int(missing_mask.sum()))

assert missing_mask.sum() == 0, "Missing CPC labels detected."

# ============================================================
# 5. Evaluation metrics
# ============================================================

def clustering_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = 0

    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        counts = Counter(y_true[mask])
        correct += max(counts.values())

    return correct / len(y_true)


def inverse_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = 0

    for label in np.unique(y_true):
        mask = y_true == label
        counts = Counter(y_pred[mask])
        correct += max(counts.values())

    return correct / len(y_true)


def evaluate_predictions(predictions, indices=None):
    predictions = np.asarray(predictions).astype(int)

    if indices is None:
        indices = np.arange(len(predictions))

    p = predictions[indices]
    sec = section_labels[indices]
    cls = class_labels[indices]
    sub = subclass_labels[indices]

    section_nmi = normalized_mutual_info_score(sec, p)
    class_nmi = normalized_mutual_info_score(cls, p)
    subclass_nmi = normalized_mutual_info_score(sub, p)

    pur_p_values = [
        clustering_purity(sec, p),
        clustering_purity(cls, p),
        clustering_purity(sub, p),
    ]

    pur_a_values = [
        inverse_purity(sec, p),
        inverse_purity(cls, p),
        inverse_purity(sub, p),
    ]

    counts = np.bincount(
        p,
        minlength=N_CLUSTERS,
    )

    return {
        "section_nmi": float(section_nmi),
        "class_nmi": float(class_nmi),
        "subclass_nmi": float(subclass_nmi),
        "mean_nmi": float(
            np.mean([
                section_nmi,
                class_nmi,
                subclass_nmi,
            ])
        ),
        "mean_pur_p": float(np.mean(pur_p_values)),
        "mean_pur_a": float(np.mean(pur_a_values)),
        "active_topics": int(np.count_nonzero(counts)),
        "max_topic_share": float(counts.max() / len(p)),
    }


# ============================================================
# 6. Fixed tune/holdout split
# ============================================================

all_indices = np.arange(len(records))

# Stratify using class when possible. Rare classes are mapped to
# their corresponding section only for splitting.
class_counts = Counter(class_labels)

split_strata = np.array([
    label if class_counts[label] >= 2 else section
    for label, section in zip(class_labels, section_labels)
])

try:
    tune_indices, holdout_indices = train_test_split(
        all_indices,
        train_size=TUNE_RATIO,
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=split_strata,
    )

    split_mode = "class_with_rare_class_section_fallback"

except Exception as error:
    print("Class stratification failed:", repr(error))

    tune_indices, holdout_indices = train_test_split(
        all_indices,
        train_size=TUNE_RATIO,
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=section_labels,
    )

    split_mode = "section_stratified"

np.save(
    OUTPUT_DIR / "dev_tune_indices.npy",
    tune_indices,
)

np.save(
    OUTPUT_DIR / "dev_holdout_indices.npy",
    holdout_indices,
)

base_predictions = theta.argmax(axis=1).astype(np.int32)

base_full = evaluate_predictions(base_predictions)
base_tune = evaluate_predictions(
    base_predictions,
    tune_indices,
)
base_holdout = evaluate_predictions(
    base_predictions,
    holdout_indices,
)

print("\n" + "=" * 80)
print("BASE THETA ARGMAX")
print("=" * 80)
print("Split mode:", split_mode)
print("Full Mean NMI:", f"{base_full['mean_nmi']:.6f}")
print("Tune Mean NMI:", f"{base_tune['mean_nmi']:.6f}")
print("Holdout Mean NMI:", f"{base_holdout['mean_nmi']:.6f}")

# ============================================================
# 7. Extract patent claim text
# ============================================================

TEXT_CACHE = CACHE_DIR / "dev_patent_texts.pkl"

texts = [
    extract_patent_text(record)
    for record in tqdm(
        records,
        desc="Extracting patent claim text",
    )
]

with open(TEXT_CACHE, "wb") as file:
    pickle.dump(texts, file)

text_lengths = np.array([
    len(text)
    for text in texts
])

empty_count = sum(
    text == "empty patent document"
    for text in texts
)

print("\n" + "=" * 80)
print("TEXT VALIDATION")
print("=" * 80)
print("Text length mean:", f"{text_lengths.mean():.2f}")
print("Text length median:", f"{np.median(text_lengths):.2f}")
print("Text length maximum:", int(text_lengths.max()))
print("Empty texts:", int(empty_count))
print("First text preview:", texts[0][:500])

if empty_count > 0:
    raise ValueError(
        f"Empty patent texts detected: {empty_count}. "
        "Do not continue with semantic embedding."
    )

if text_lengths.mean() < 100:
    raise ValueError(
        "Extracted texts are unexpectedly short."
    )

# ============================================================
# 8. PatentSBERT semantic embeddings
# ============================================================

SEMANTIC_RAW_CACHE = CACHE_DIR / "semantic_embeddings_raw.npy"
SEMANTIC_PCA_CACHE = (
    CACHE_DIR
    / f"semantic_embeddings_pca{SEMANTIC_PCA_DIM}.npy"
)
SEMANTIC_META_PATH = CACHE_DIR / "semantic_model.json"

model = None
selected_semantic_model = None
model_errors = {}

for model_name in SEMANTIC_MODELS:
    try:
        print("\nTrying semantic model:", model_name)

        model = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        selected_semantic_model = model_name
        break

    except Exception as error:
        model_errors[model_name] = repr(error)
        print("Model loading failed:", repr(error))

if model is None:
    raise RuntimeError(
        "All semantic models failed:\n"
        + json.dumps(model_errors, indent=2)
    )

semantic_raw = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

np.save(
    SEMANTIC_RAW_CACHE,
    semantic_raw,
)

semantic_pca_dim = min(
    SEMANTIC_PCA_DIM,
    semantic_raw.shape[1],
    semantic_raw.shape[0] - 1,
)

semantic_pca = PCA(
    n_components=semantic_pca_dim,
    whiten=False,
    random_state=SPLIT_SEED,
)

semantic_features = semantic_pca.fit_transform(
    semantic_raw
).astype(np.float32)

semantic_features = normalize(
    semantic_features,
    norm="l2",
).astype(np.float32)

np.save(
    SEMANTIC_PCA_CACHE,
    semantic_features,
)

joblib.dump(
    semantic_pca,
    MODEL_DIR / "semantic_pca.joblib",
)

with open(SEMANTIC_META_PATH, "w") as file:
    json.dump(
        {
            "model": selected_semantic_model,
            "raw_shape": list(semantic_raw.shape),
            "pca_shape": list(semantic_features.shape),
            "max_text_chars": MAX_TEXT_CHARS,
        },
        file,
        indent=2,
    )

del model
del semantic_raw

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Semantic model:", selected_semantic_model)
print("Semantic feature shape:", semantic_features.shape)

# ============================================================
# 9. TF-IDF + SVD lexical embeddings
# ============================================================

LEXICAL_CACHE = (
    CACHE_DIR
    / f"lexical_tfidf_svd{LEXICAL_SVD_DIM}.npy"
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.995,
    max_features=120000,
    sublinear_tf=True,
    dtype=np.float32,
)

lexical_sparse = vectorizer.fit_transform(texts)

print("TF-IDF shape:", lexical_sparse.shape)

if lexical_sparse.shape[1] == 0:
    raise ValueError("TF-IDF vocabulary is empty.")

lexical_dim = min(
    LEXICAL_SVD_DIM,
    lexical_sparse.shape[0] - 1,
    lexical_sparse.shape[1] - 1,
)

lexical_svd = TruncatedSVD(
    n_components=lexical_dim,
    n_iter=10,
    random_state=SPLIT_SEED,
)

lexical_features = lexical_svd.fit_transform(
    lexical_sparse
).astype(np.float32)

lexical_features = normalize(
    lexical_features,
    norm="l2",
).astype(np.float32)

np.save(
    LEXICAL_CACHE,
    lexical_features,
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib",
)

joblib.dump(
    lexical_svd,
    MODEL_DIR / "lexical_svd.joblib",
)

print("Lexical feature shape:", lexical_features.shape)

del lexical_sparse

# ============================================================
# 10. Fusion helpers
# ============================================================

def make_theta_features(theta_matrix, power):
    features = np.power(
        np.maximum(theta_matrix, 1e-12),
        power,
    )

    features = normalize(
        features,
        norm="l2",
    )

    return features.astype(np.float32)


def make_fusion_features(
    theta_matrix,
    theta_power,
    theta_weight,
    semantic_weight,
    lexical_weight,
):
    blocks = []

    if theta_weight > 0:
        theta_features = make_theta_features(
            theta_matrix,
            theta_power,
        )

        blocks.append(
            np.sqrt(theta_weight) * theta_features
        )

    if semantic_weight > 0:
        blocks.append(
            np.sqrt(semantic_weight)
            * semantic_features
        )

    if lexical_weight > 0:
        blocks.append(
            np.sqrt(lexical_weight)
            * lexical_features
        )

    if not blocks:
        raise ValueError("No active feature block.")

    fused = np.concatenate(
        blocks,
        axis=1,
    )

    fused = normalize(
        fused,
        norm="l2",
    )

    return fused.astype(np.float32)


def align_predictions(reference, candidate):
    contingency = np.zeros(
        (N_CLUSTERS, N_CLUSTERS),
        dtype=np.int64,
    )

    np.add.at(
        contingency,
        (
            reference.astype(int),
            candidate.astype(int),
        ),
        1,
    )

    reference_labels, candidate_labels = (
        linear_sum_assignment(-contingency)
    )

    candidate_to_reference = {
        int(candidate_label): int(reference_label)
        for reference_label, candidate_label
        in zip(reference_labels, candidate_labels)
    }

    return np.array(
        [
            candidate_to_reference.get(
                int(label),
                int(label),
            )
            for label in candidate
        ],
        dtype=np.int32,
    )


def make_consensus(seed_predictions):
    reference = seed_predictions[0]

    aligned_predictions = [reference]

    for candidate in seed_predictions[1:]:
        aligned_predictions.append(
            align_predictions(
                reference,
                candidate,
            )
        )

    stacked = np.stack(
        aligned_predictions,
        axis=1,
    )

    consensus = np.zeros(
        len(reference),
        dtype=np.int32,
    )

    for index, row in enumerate(stacked):
        counts = np.bincount(
            row,
            minlength=N_CLUSTERS,
        )

        consensus[index] = int(
            np.argmax(counts)
        )

    agreement = np.mean(
        stacked == consensus[:, None],
        axis=1,
    )

    return consensus, agreement


# ============================================================
# 11. Run clustering search
# ============================================================

ranking_rows = []
prediction_store = {}
agreement_store = {}

print("\n" + "=" * 80)
print("RUNNING FUSION SEARCH")
print("=" * 80)

for (
    config_name,
    theta_power,
    theta_weight,
    semantic_weight,
    lexical_weight,
) in tqdm(
    FUSION_CONFIGS,
    desc="Fusion configurations",
):
    print("\nConfiguration:", config_name)

    fused_features = make_fusion_features(
        theta_matrix=theta,
        theta_power=theta_power,
        theta_weight=theta_weight,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
    )

    seed_predictions = []

    for seed in RANDOM_SEEDS:
        clustering = MiniBatchKMeans(
            n_clusters=N_CLUSTERS,
            init="k-means++",
            n_init=20,
            max_iter=500,
            batch_size=2048,
            reassignment_ratio=0.005,
            random_state=seed,
            verbose=0,
        )

        predictions = clustering.fit_predict(
            fused_features
        ).astype(np.int32)

        seed_predictions.append(predictions)

        variant_name = (
            f"{config_name}_seed{seed}"
        )

        prediction_store[variant_name] = predictions

        tune_metrics = evaluate_predictions(
            predictions,
            tune_indices,
        )

        ranking_rows.append({
            "variant": variant_name,
            "config": config_name,
            "mode": "single_seed",
            "seed": int(seed),
            "theta_power": float(theta_power),
            "theta_weight": float(theta_weight),
            "semantic_weight": float(semantic_weight),
            "lexical_weight": float(lexical_weight),
            "mean_seed_agreement": np.nan,
            "low_agreement_share": np.nan,
            **{
                f"tune_{key}": value
                for key, value in tune_metrics.items()
            },
        })

    consensus, agreement = make_consensus(
        seed_predictions
    )

    consensus_name = (
        f"{config_name}_consensus"
    )

    prediction_store[consensus_name] = consensus
    agreement_store[consensus_name] = agreement

    tune_metrics = evaluate_predictions(
        consensus,
        tune_indices,
    )

    ranking_rows.append({
        "variant": consensus_name,
        "config": config_name,
        "mode": "consensus",
        "seed": -1,
        "theta_power": float(theta_power),
        "theta_weight": float(theta_weight),
        "semantic_weight": float(semantic_weight),
        "lexical_weight": float(lexical_weight),
        "mean_seed_agreement": float(
            agreement.mean()
        ),
        "low_agreement_share": float(
            (agreement < 1.0).mean()
        ),
        **{
            f"tune_{key}": value
            for key, value in tune_metrics.items()
        },
    })

    del fused_features

ranking = pd.DataFrame(ranking_rows)

# ============================================================
# 12. Select candidate using tune labels only
# ============================================================

stable_mask = (
    (
        ranking["tune_mean_pur_p"]
        >= base_tune["mean_pur_p"] - 0.002
    )
    & (
        ranking["tune_mean_pur_a"]
        >= base_tune["mean_pur_a"] - 0.010
    )
    & (
        ranking["tune_section_nmi"]
        >= base_tune["section_nmi"] - 0.003
    )
    & (
        ranking["tune_active_topics"] >= 28
    )
    & (
        ranking["tune_max_topic_share"] <= 0.25
    )
)

ranking["is_stable"] = stable_mask

ranking["tune_selection_score"] = (
    ranking["tune_mean_nmi"]
    + 0.020 * ranking["tune_mean_pur_p"]
    + 0.010 * ranking["tune_mean_pur_a"]
    - 0.010 * np.maximum(
        ranking["tune_max_topic_share"] - 0.20,
        0,
    )
)

ranking = ranking.sort_values(
    [
        "is_stable",
        "tune_mean_nmi",
        "tune_selection_score",
        "tune_mean_pur_p",
        "tune_mean_pur_a",
    ],
    ascending=[
        False,
        False,
        False,
        False,
        False,
    ],
).reset_index(drop=True)

stable_ranking = ranking[
    ranking["is_stable"]
].copy()

if len(stable_ranking) > 0:
    selected_row = stable_ranking.iloc[0]
    selection_type = "BEST_STABLE_TUNE_CANDIDATE"
else:
    selected_row = ranking.iloc[0]
    selection_type = "NO_STABLE_CANDIDATE_USE_BEST_RAW"

selected_variant = str(
    selected_row["variant"]
)

selected_predictions = prediction_store[
    selected_variant
]

# Holdout labels are evaluated only after selection.
selected_tune = evaluate_predictions(
    selected_predictions,
    tune_indices,
)

selected_holdout = evaluate_predictions(
    selected_predictions,
    holdout_indices,
)

selected_full = evaluate_predictions(
    selected_predictions,
)

holdout_delta = (
    selected_holdout["mean_nmi"]
    - base_holdout["mean_nmi"]
)

full_delta = (
    selected_full["mean_nmi"]
    - base_full["mean_nmi"]
)

delta_vs_previous_stable = (
    selected_full["mean_nmi"]
    - PREVIOUS_STABLE_HYBRID_NMI
)

# ============================================================
# 13. Final decision
# ============================================================

if (
    holdout_delta >= 0.002
    and selected_full["mean_nmi"] >= TARGET_NMI
    and selected_holdout["mean_pur_p"]
        >= base_holdout["mean_pur_p"] - 0.002
    and selected_holdout["mean_pur_a"]
        >= base_holdout["mean_pur_a"] - 0.010
):
    decision = (
        "STRONG_FUSION_CANDIDATE_"
        "PROCEED_TO_CHECKPOINT_ENSEMBLE"
    )

elif (
    holdout_delta > 0
    and selected_full["mean_nmi"]
        > PREVIOUS_STABLE_HYBRID_NMI
):
    decision = (
        "FUSION_IMPROVES_HOLDOUT_"
        "BUT_NEEDS_ENSEMBLE_CONFIRMATION"
    )

else:
    decision = (
        "FUSION_NOT_CONFIRMED_"
        "KEEP_PREVIOUS_STABLE_HYBRID"
    )

run_test = False

# ============================================================
# 14. Save outputs
# ============================================================

ranking_path = (
    OUTPUT_DIR
    / "text_theta_fusion_tune_ranking.csv"
)

summary_path = (
    OUTPUT_DIR
    / "text_theta_fusion_summary.json"
)

predictions_path = (
    OUTPUT_DIR
    / "selected_fusion_dev_predictions.npy"
)

assignments_path = (
    OUTPUT_DIR
    / "selected_fusion_dev_assignments.csv"
)

ranking.to_csv(
    ranking_path,
    index=False,
)

np.save(
    predictions_path,
    selected_predictions,
)

if selected_variant in agreement_store:
    np.save(
        OUTPUT_DIR
        / "selected_fusion_dev_agreement.npy",
        agreement_store[selected_variant],
    )

split_array = np.full(
    len(records),
    "holdout",
    dtype=object,
)

split_array[tune_indices] = "tune"

assignment_frame = pd.DataFrame({
    "index": np.arange(len(records)),
    "patent_id": [
        to_plain_dict(record).get(
            "patent_id",
            index,
        )
        for index, record in enumerate(records)
    ],
    "split": split_array,
    "section": section_labels,
    "class": class_labels,
    "subclass": subclass_labels,
    "base_topic": base_predictions,
    "selected_topic": selected_predictions,
})

assignment_frame.to_csv(
    assignments_path,
    index=False,
)

runtime_minutes = (
    time.time() - start_time
) / 60

summary = {
    "selected_variant": selected_variant,
    "selection_type": selection_type,
    "semantic_model": selected_semantic_model,
    "split_mode": split_mode,
    "split_seed": SPLIT_SEED,
    "random_seeds": RANDOM_SEEDS,
    "base_full": base_full,
    "base_tune": base_tune,
    "base_holdout": base_holdout,
    "selected_full": selected_full,
    "selected_tune": selected_tune,
    "selected_holdout": selected_holdout,
    "full_mean_nmi_delta": float(full_delta),
    "holdout_mean_nmi_delta": float(holdout_delta),
    "delta_vs_previous_stable_hybrid": float(
        delta_vs_previous_stable
    ),
    "previous_stable_hybrid_nmi": (
        PREVIOUS_STABLE_HYBRID_NMI
    ),
    "target_nmi": TARGET_NMI,
    "number_of_variants": int(len(ranking)),
    "number_of_stable_variants": int(
        ranking["is_stable"].sum()
    ),
    "run_test": run_test,
    "decision": decision,
    "runtime_minutes": float(runtime_minutes),
}

with open(summary_path, "w") as file:
    json.dump(
        summary,
        file,
        indent=2,
    )

# ============================================================
# 15. Print final results
# ============================================================

print("\n" + "=" * 80)
print("FINAL DECISION")
print("=" * 80)

print("Selected variant:", selected_variant)
print("Selection type:", selection_type)
print("Semantic model:", selected_semantic_model)
print("Stable candidates:", int(ranking["is_stable"].sum()))
print("Total candidates:", len(ranking))

print("\nBASE FULL")
print("Mean NMI:", f"{base_full['mean_nmi']:.6f}")
print("Mean Pur_p:", f"{base_full['mean_pur_p']:.6f}")
print("Mean Pur_a:", f"{base_full['mean_pur_a']:.6f}")

print("\nSELECTED TUNE")
print("Mean NMI:", f"{selected_tune['mean_nmi']:.6f}")
print("Mean Pur_p:", f"{selected_tune['mean_pur_p']:.6f}")
print("Mean Pur_a:", f"{selected_tune['mean_pur_a']:.6f}")

print("\nSELECTED HOLDOUT")
print(
    "Base Holdout Mean NMI:",
    f"{base_holdout['mean_nmi']:.6f}",
)
print(
    "Selected Holdout Mean NMI:",
    f"{selected_holdout['mean_nmi']:.6f}",
)
print(
    "Holdout NMI Delta:",
    f"{holdout_delta:+.6f}",
)
print(
    "Mean Pur_p:",
    f"{selected_holdout['mean_pur_p']:.6f}",
)
print(
    "Mean Pur_a:",
    f"{selected_holdout['mean_pur_a']:.6f}",
)

print("\nSELECTED FULL DEV")
print(
    "Mean NMI:",
    f"{selected_full['mean_nmi']:.6f}",
)
print(
    "Delta vs theta input:",
    f"{full_delta:+.6f}",
)
print(
    "Delta vs previous stable hybrid:",
    f"{delta_vs_previous_stable:+.6f}",
)
print(
    "Mean Pur_p:",
    f"{selected_full['mean_pur_p']:.6f}",
)
print(
    "Mean Pur_a:",
    f"{selected_full['mean_pur_a']:.6f}",
)
print(
    "Section NMI:",
    f"{selected_full['section_nmi']:.6f}",
)
print(
    "Class NMI:",
    f"{selected_full['class_nmi']:.6f}",
)
print(
    "Subclass NMI:",
    f"{selected_full['subclass_nmi']:.6f}",
)
print(
    "Active topics:",
    selected_full["active_topics"],
)
print(
    "Maximum topic share:",
    f"{selected_full['max_topic_share']:.2%}",
)

print("\nRun TEST now:", run_test)
print("Decision:", decision)
print("Runtime:", f"{runtime_minutes:.2f} minutes")

print("\nOUTPUT FILES")
print("Ranking:", ranking_path)
print("Summary:", summary_path)
print("Predictions:", predictions_path)
print("Assignments:", assignments_path)

print("\nTOP 10 TUNE CANDIDATES")
display_columns = [
    "variant",
    "is_stable",
    "tune_mean_nmi",
    "tune_mean_pur_p",
    "tune_mean_pur_a",
    "tune_section_nmi",
    "tune_class_nmi",
    "tune_subclass_nmi",
]

print(
    ranking[display_columns]
    .head(10)
    .to_string(index=False)
)
