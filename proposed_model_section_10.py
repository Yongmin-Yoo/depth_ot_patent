# ============================================================
# Resumable claim-wise pooling extractor
# Run after interrupting the previous cell.
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

import gc
import json
import shutil
import time
import pickle
from pathlib import Path

import numpy as np
import torch

# ============================================================
# 0. Paths
# ============================================================

RUN_ROOT = Path(
    "/content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/"
    "depth_ot_v2_patent_semantic_seed42_20260814_055110"
)

FEATURE_DIR = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/"
    "token_features/full/dev"
)

RECORDS_PATH = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/"
    "dev_records.pkl"
)

OUTPUT_DIR = (
    RUN_ROOT
    / "epoch016_gpu_claimwise_semantic_fusion_dev_search"
)

CACHE_DIR = OUTPUT_DIR / "cache"
PARTIAL_DIR = CACHE_DIR / "shard_pool_partials"
LOCAL_DIR = Path("/content/depth_ot_shard_cache")

POOL_CACHE_PATH = (
    CACHE_DIR
    / "claimwise_raw_patent_pools.npz"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

N_PATENTS = 9855
HIDDEN_DIM = 768
EPS = 1e-12

POOLING_CONFIGS = [
    ("uniform", 1.0, 0.00),
    ("root4_d000", 4.0, 0.00),
    ("root8_d000", 8.0, 0.00),
    ("root8_d005", 8.0, 0.05),
    ("root8_d010", 8.0, 0.10),
    ("root8_d020", 8.0, 0.20),
    ("root12_d010", 12.0, 0.10),
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError("Select a T4 GPU runtime.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("=" * 80)
print("RESUMABLE CLAIM-WISE POOLING")
print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))
print("Partial cache:", PARTIAL_DIR)
print("Final cache:", POOL_CACHE_PATH)

# ============================================================
# 1. Load patent records and depth maps
# ============================================================

with open(RECORDS_PATH, "rb") as file:
    records = pickle.load(file)

if isinstance(records, dict):
    for key in ["records", "data", "items", "patents"]:
        if key in records:
            records = records[key]
            break

records = list(records)

assert len(records) == N_PATENTS

patent_ids = np.asarray(
    [
        int(record["patent_id"])
        for record in records
    ],
    dtype=np.int64,
)

patent_to_global_index = {
    int(patent_id): index
    for index, patent_id in enumerate(patent_ids)
}

depth_lookup = {}

for record in records:
    patent_id = int(record["patent_id"])

    for claim_id, depth in record.get(
        "depth",
        {},
    ).items():
        depth_lookup[
            (patent_id, int(claim_id))
        ] = int(depth)

shard_paths = sorted(
    FEATURE_DIR.glob("shard_*.pt")
)

assert len(shard_paths) == 99, (
    f"Expected 99 shards, found {len(shard_paths)}"
)

print("Records:", len(records))
print("Depth entries:", len(depth_lookup))
print("Shards:", len(shard_paths))

# ============================================================
# 2. Process one shard
# ============================================================

def process_one_shard(
    drive_shard_path,
    partial_path,
):
    shard_start = time.time()

    local_path = (
        LOCAL_DIR
        / drive_shard_path.name
    )

    # Remove an incomplete local copy from an interrupted run.
    if local_path.exists():
        local_path.unlink()

    print("  Copying to local disk...")

    copy_start = time.time()

    shutil.copyfile(
        drive_shard_path,
        local_path,
    )

    copy_seconds = time.time() - copy_start

    drive_size = drive_shard_path.stat().st_size
    local_size = local_path.stat().st_size

    if drive_size != local_size:
        local_path.unlink(missing_ok=True)

        raise IOError(
            f"Incomplete copy: {drive_shard_path.name}, "
            f"expected {drive_size}, got {local_size}"
        )

    print(
        f"  Copied {local_size / 1024**2:.1f} MiB "
        f"in {copy_seconds:.1f}s"
    )

    print("  Loading local shard...")

    shard = torch.load(
        local_path,
        map_location="cpu",
        weights_only=False,
    )

    shard_patent_ids = np.asarray(
        [
            int(patent_id)
            for patent_id in shard["patent_ids"]
        ],
        dtype=np.int64,
    )

    local_patent_index = {
        int(patent_id): index
        for index, patent_id
        in enumerate(shard_patent_ids)
    }

    entries = shard["entries"]
    n_local_patents = len(shard_patent_ids)
    n_poolings = len(POOLING_CONFIGS)

    token_tensors = []
    token_lengths = []
    local_patent_indices = []
    claim_depths = []

    missing_depth_count = 0
    truncated_count = 0

    for entry in entries:
        patent_id = int(entry["patent_id"])
        claim_id = int(entry["claim_id"])

        token_embeddings = entry[
            "token_embeddings"
        ]

        num_tokens = min(
            int(
                entry.get(
                    "num_tokens",
                    token_embeddings.shape[0],
                )
            ),
            int(token_embeddings.shape[0]),
        )

        if num_tokens <= 0:
            continue

        depth = depth_lookup.get(
            (patent_id, claim_id),
            None,
        )

        if depth is None:
            missing_depth_count += 1
            depth = 0

        token_tensors.append(
            token_embeddings[:num_tokens]
        )

        token_lengths.append(num_tokens)

        local_patent_indices.append(
            local_patent_index[patent_id]
        )

        claim_depths.append(depth)

        if bool(entry.get("truncated", False)):
            truncated_count += 1

    if not token_tensors:
        raise ValueError(
            f"No usable claims in {drive_shard_path}"
        )

    print(
        f"  Pooling {len(token_tensors):,} claims on GPU..."
    )

    concatenated_tokens = torch.cat(
        token_tensors,
        dim=0,
    )

    lengths_gpu = torch.tensor(
        token_lengths,
        dtype=torch.long,
        device=DEVICE,
    )

    token_to_claim_gpu = torch.repeat_interleave(
        torch.arange(
            len(token_lengths),
            dtype=torch.long,
            device=DEVICE,
        ),
        lengths_gpu,
    )

    tokens_gpu = concatenated_tokens.to(
        DEVICE,
        dtype=torch.float32,
        non_blocking=True,
    )

    claim_sums_gpu = torch.zeros(
        (
            len(token_lengths),
            HIDDEN_DIM,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    claim_sums_gpu.index_add_(
        0,
        token_to_claim_gpu,
        tokens_gpu,
    )

    claim_means_gpu = (
        claim_sums_gpu
        / lengths_gpu.to(
            torch.float32
        ).unsqueeze(1)
    )

    claim_means_gpu = (
        torch.nn.functional.normalize(
            claim_means_gpu,
            p=2,
            dim=1,
        )
    )

    local_patent_indices_gpu = torch.tensor(
        local_patent_indices,
        dtype=torch.long,
        device=DEVICE,
    )

    depths_gpu = torch.tensor(
        claim_depths,
        dtype=torch.float32,
        device=DEVICE,
    )

    root_indicator_gpu = (
        depths_gpu == 0
    ).to(torch.float32)

    partial_output = {
        "patent_ids": shard_patent_ids,
        "claim_count": np.asarray(
            [len(token_tensors)],
            dtype=np.int64,
        ),
        "truncated_count": np.asarray(
            [truncated_count],
            dtype=np.int64,
        ),
        "missing_depth_count": np.asarray(
            [missing_depth_count],
            dtype=np.int64,
        ),
    }

    for (
        pooling_name,
        root_weight,
        depth_decay,
    ) in POOLING_CONFIGS:
        claim_weights_gpu = torch.pow(
            torch.tensor(
                root_weight,
                dtype=torch.float32,
                device=DEVICE,
            ),
            root_indicator_gpu,
        )

        if depth_decay > 0:
            claim_weights_gpu = (
                claim_weights_gpu
                * torch.exp(
                    -depth_decay * depths_gpu
                )
            )

        patent_sums_gpu = torch.zeros(
            (
                n_local_patents,
                HIDDEN_DIM,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        patent_weights_gpu = torch.zeros(
            n_local_patents,
            dtype=torch.float32,
            device=DEVICE,
        )

        patent_sums_gpu.index_add_(
            0,
            local_patent_indices_gpu,
            claim_means_gpu
            * claim_weights_gpu.unsqueeze(1),
        )

        patent_weights_gpu.index_add_(
            0,
            local_patent_indices_gpu,
            claim_weights_gpu,
        )

        if (
            patent_weights_gpu <= 0
        ).any():
            raise ValueError(
                f"Patent without claims in "
                f"{drive_shard_path.name}"
            )

        pooled_gpu = (
            patent_sums_gpu
            / patent_weights_gpu.unsqueeze(1)
        )

        pooled_gpu = (
            torch.nn.functional.normalize(
                pooled_gpu,
                p=2,
                dim=1,
            )
        )

        partial_output[pooling_name] = (
            pooled_gpu
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # Write to a temporary path first.
    temporary_partial = Path(
        str(partial_path) + ".tmp.npz"
    )

    np.savez(
        temporary_partial,
        **partial_output,
    )

    temporary_partial.replace(
        partial_path
    )

    del shard
    del entries
    del token_tensors
    del concatenated_tokens
    del tokens_gpu
    del lengths_gpu
    del token_to_claim_gpu
    del claim_sums_gpu
    del claim_means_gpu
    del local_patent_indices_gpu
    del depths_gpu
    del root_indicator_gpu

    gc.collect()
    torch.cuda.empty_cache()

    local_path.unlink(missing_ok=True)

    elapsed = time.time() - shard_start

    return {
        "claims": len(token_lengths),
        "patents": n_local_patents,
        "truncated": truncated_count,
        "missing_depth": missing_depth_count,
        "copy_seconds": copy_seconds,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# 3. Process all shards with resume support
# ============================================================

overall_start = time.time()
completed_before_start = 0

for shard_index, shard_path in enumerate(
    shard_paths
):
    partial_path = (
        PARTIAL_DIR
        / f"{shard_path.stem}_pools.npz"
    )

    if partial_path.exists():
        try:
            cached = np.load(
                partial_path,
                allow_pickle=False,
            )

            valid = (
                "patent_ids" in cached.files
                and all(
                    name in cached.files
                    for name, _, _
                    in POOLING_CONFIGS
                )
            )

            if valid:
                completed_before_start += 1

                print(
                    f"[{shard_index + 1:03d}/"
                    f"{len(shard_paths):03d}] "
                    f"{shard_path.name}: cached"
                )

                continue

        except Exception:
            pass

        partial_path.unlink(missing_ok=True)

    print(
        f"\n[{shard_index + 1:03d}/"
        f"{len(shard_paths):03d}] "
        f"{shard_path.name}"
    )

    try:
        result = process_one_shard(
            shard_path,
            partial_path,
        )

        print(
            f"  Completed: "
            f"{result['patents']} patents, "
            f"{result['claims']:,} claims, "
            f"{result['elapsed_seconds']:.1f}s"
        )

    except Exception as error:
        print(
            f"  FAILED: {type(error).__name__}: "
            f"{error}"
        )

        print(
            "  Completed partial caches are safe. "
            "Rerun this cell to resume."
        )

        raise

print("\n" + "=" * 80)
print("ASSEMBLING FINAL PATENT POOLS")
print("=" * 80)

raw_patent_pools = {
    name: np.zeros(
        (N_PATENTS, HIDDEN_DIM),
        dtype=np.float32,
    )
    for name, _, _
    in POOLING_CONFIGS
}

observed_patents = np.zeros(
    N_PATENTS,
    dtype=bool,
)

total_claims = 0
total_truncated = 0
total_missing_depth = 0

for shard_path in shard_paths:
    partial_path = (
        PARTIAL_DIR
        / f"{shard_path.stem}_pools.npz"
    )

    if not partial_path.exists():
        raise FileNotFoundError(
            f"Missing partial cache: {partial_path}"
        )

    partial = np.load(
        partial_path,
        allow_pickle=False,
    )

    partial_patent_ids = partial[
        "patent_ids"
    ].astype(np.int64)

    global_indices = np.asarray(
        [
            patent_to_global_index[
                int(patent_id)
            ]
            for patent_id
            in partial_patent_ids
        ],
        dtype=np.int64,
    )

    if observed_patents[
        global_indices
    ].any():
        raise ValueError(
            f"Duplicate patents found in {partial_path}"
        )

    for name, _, _ in POOLING_CONFIGS:
        raw_patent_pools[name][
            global_indices
        ] = partial[name].astype(np.float32)

    observed_patents[
        global_indices
    ] = True

    total_claims += int(
        partial["claim_count"][0]
    )

    total_truncated += int(
        partial["truncated_count"][0]
    )

    total_missing_depth += int(
        partial["missing_depth_count"][0]
    )

if not observed_patents.all():
    missing_indices = np.flatnonzero(
        ~observed_patents
    )

    raise ValueError(
        f"Missing {len(missing_indices)} patents."
    )

temporary_final_path = Path(
    str(POOL_CACHE_PATH) + ".tmp.npz"
)

np.savez(
    temporary_final_path,
    **raw_patent_pools,
)

temporary_final_path.replace(
    POOL_CACHE_PATH
)

runtime_minutes = (
    time.time() - overall_start
) / 60

print("\n" + "=" * 80)
print("POOLING COMPLETE")
print("=" * 80)
print("Patents:", int(observed_patents.sum()))
print("Claims:", f"{total_claims:,}")
print("Truncated claims:", total_truncated)
print("Missing depths:", total_missing_depth)
print(
    "Previously cached shards:",
    completed_before_start,
)
print(
    "Final cache:",
    POOL_CACHE_PATH,
)
print(
    "Final cache size:",
    f"{POOL_CACHE_PATH.stat().st_size / 1024**2:.2f} MiB",
)
print("Runtime:", f"{runtime_minutes:.2f} minutes")
print(
    "\nNow rerun the original claim-wise fusion code. "
    "It will detect this final cache and skip all 99 shards."
)


# ============================================================
# Depth-OT: GPU Claim-Wise Semantic Fusion DEV Search
# Google Colab T4, single-cell runnable version
# TEST DATA IS NOT USED
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

import os
import re
import gc
import csv
import json
import time
import pickle
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import joblib

# ------------------------------------------------------------
# 1. Reproducibility and GPU
# ------------------------------------------------------------
GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")

print("=" * 80)
print("DEPTH-OT GPU CLAIM-WISE SEMANTIC FUSION")
print("=" * 80)
print("Device:", DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "GPU가 감지되지 않았습니다. Colab 메뉴에서 "
        "'런타임 > 런타임 유형 변경 > T4 GPU'를 선택하세요."
    )

print("GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
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

TEXT_FUSION_DIR = (
    RUN_ROOT / "epoch016_text_theta_fusion_dev_search"
)

TEXT_CACHE_DIR = TEXT_FUSION_DIR / "cache"

SEMANTIC_PATH = TEXT_CACHE_DIR / "semantic_embeddings_pca96.npy"
LEXICAL_PATH = TEXT_CACHE_DIR / "lexical_tfidf_svd128.npy"

OUTPUT_DIR = (
    RUN_ROOT / "epoch016_gpu_claimwise_semantic_fusion_dev_search"
)

CACHE_DIR = OUTPUT_DIR / "cache"
MODEL_DIR = OUTPUT_DIR / "models"

POOL_CACHE_PATH = CACHE_DIR / "claimwise_raw_patent_pools.npz"
PCA_MODEL_PATH = MODEL_DIR / "claimwise_semantic_pca128.joblib"
PCA_CACHE_PATH = CACHE_DIR / "claimwise_semantic_pca128_all_pools.npz"

RANKING_PATH = OUTPUT_DIR / "claimwise_fusion_tune_ranking.csv"
SUMMARY_PATH = OUTPUT_DIR / "claimwise_fusion_summary.json"
PREDICTIONS_PATH = OUTPUT_DIR / "selected_claimwise_fusion_dev_predictions.npy"
ASSIGNMENTS_PATH = OUTPUT_DIR / "selected_claimwise_fusion_dev_assignments.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Stable reference path candidates
STABLE_REFERENCE_CANDIDATES = [
    RUN_ROOT
    / "epoch016_uncertainty_gated_hybrid_dev_search"
    / "best_stable_hybrid_dev_predictions.npy",

    RUN_ROOT
    / "epoch016_gpu_confidence_gated_text_fusion_dev_search"
    / "best_stable_hybrid_dev_predictions.npy",

    RUN_ROOT
    / "epoch016_confidence_gated_text_fusion_dev_search"
    / "best_stable_hybrid_dev_predictions.npy",
]

# ------------------------------------------------------------
# 3. Constants
# ------------------------------------------------------------
N_PATENTS = 9855
N_CLUSTERS = 30
CLAIM_PCA_DIM = 128

TUNE_RATIO = 0.70
SPLIT_SEED = 42

KMEANS_SEEDS = [17, 42, 73]
KMEANS_MAX_ITER = 100
KMEANS_TOL = 1e-5

# Stable selection constraints
MIN_ACTIVE_TOPICS = 28
MAX_TOPIC_SHARE = 0.12
MAX_PURA_DROP = 0.010
MIN_TUNE_NMI_GAIN = 0.0

# 18 configurations × 7 pooling variants × 3 seeds = 378 runs.
# Weights: theta, document semantic, lexical, claim-wise semantic
FUSION_CONFIGS = [
    {"name": "claim_only",
     "q": 0.50, "t": 0.00, "s": 0.00, "l": 0.00, "c": 1.00},

    {"name": "theta_claim_30_70",
     "q": 0.50, "t": 0.30, "s": 0.00, "l": 0.00, "c": 0.70},

    {"name": "theta_claim_40_60",
     "q": 0.50, "t": 0.40, "s": 0.00, "l": 0.00, "c": 0.60},

    {"name": "theta_claim_50_50",
     "q": 0.50, "t": 0.50, "s": 0.00, "l": 0.00, "c": 0.50},

    {"name": "theta_doc_claim_30_20_50",
     "q": 0.50, "t": 0.30, "s": 0.20, "l": 0.00, "c": 0.50},

    {"name": "theta_doc_claim_30_30_40",
     "q": 0.50, "t": 0.30, "s": 0.30, "l": 0.00, "c": 0.40},

    {"name": "theta_lex_claim_30_20_50",
     "q": 0.50, "t": 0.30, "s": 0.00, "l": 0.20, "c": 0.50},

    {"name": "theta_lex_claim_30_30_40",
     "q": 0.50, "t": 0.30, "s": 0.00, "l": 0.30, "c": 0.40},

    {"name": "all_30_20_10_40",
     "q": 0.50, "t": 0.30, "s": 0.20, "l": 0.10, "c": 0.40},

    {"name": "all_30_10_20_40",
     "q": 0.50, "t": 0.30, "s": 0.10, "l": 0.20, "c": 0.40},

    {"name": "all_30_20_20_30",
     "q": 0.50, "t": 0.30, "s": 0.20, "l": 0.20, "c": 0.30},

    {"name": "all_40_20_10_30",
     "q": 0.50, "t": 0.40, "s": 0.20, "l": 0.10, "c": 0.30},

    {"name": "all_20_20_10_50",
     "q": 0.50, "t": 0.20, "s": 0.20, "l": 0.10, "c": 0.50},

    {"name": "all_20_10_20_50",
     "q": 0.50, "t": 0.20, "s": 0.10, "l": 0.20, "c": 0.50},

    {"name": "q075_all_30_20_10_40",
     "q": 0.75, "t": 0.30, "s": 0.20, "l": 0.10, "c": 0.40},

    {"name": "q075_all_30_10_20_40",
     "q": 0.75, "t": 0.30, "s": 0.10, "l": 0.20, "c": 0.40},

    {"name": "q075_theta_claim_30_70",
     "q": 0.75, "t": 0.30, "s": 0.00, "l": 0.00, "c": 0.70},

    {"name": "q075_theta_claim_40_60",
     "q": 0.75, "t": 0.40, "s": 0.00, "l": 0.00, "c": 0.60},
]

# ------------------------------------------------------------
# 4. Utility functions
# ------------------------------------------------------------
def require_file(path, description):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{description} 파일을 찾지 못했습니다:\n{path}"
        )
    print(f"{description}: {path}")


def safe_string(x):
    if x is None:
        return "UNKNOWN"
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    x = str(x).strip()
    return x if x else "UNKNOWN"


def get_record_value(record, keys, default="UNKNOWN"):
    if not isinstance(record, dict):
        return default

    for key in keys:
        if key in record and record[key] is not None:
            value = record[key]

            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) > 0:
                    return value[0]
            else:
                return value

    return default


def extract_cpc_labels(record):
    section = safe_string(
        get_record_value(
            record,
            ["section", "cpc_section", "section_label"]
        )
    )

    class_label = safe_string(
        get_record_value(
            record,
            ["class", "cpc_class", "class_label"]
        )
    )

    subclass = safe_string(
        get_record_value(
            record,
            ["subclass", "cpc_subclass", "subclass_label"]
        )
    )

    # Fallback from CPC code
    codes = record.get("cpc_codes", []) if isinstance(record, dict) else []

    if isinstance(codes, str):
        codes = [codes]

    if codes:
        code = re.sub(r"[^A-Z0-9]", "", safe_string(codes[0]).upper())

        if section == "UNKNOWN" and len(code) >= 1:
            section = code[:1]

        if class_label == "UNKNOWN" and len(code) >= 3:
            class_label = code[:3]

        if subclass == "UNKNOWN" and len(code) >= 4:
            subclass = code[:4]

    return section, class_label, subclass


def encode_labels(labels):
    unique = sorted(set(map(str, labels)))
    mapping = {value: i for i, value in enumerate(unique)}
    return np.asarray([mapping[str(x)] for x in labels], dtype=np.int64)


def cluster_purity(y_true, y_pred):
    """
    Pur_p: predicted-cluster purity.
    For every predicted cluster, count its majority true category.
    """
    total = 0

    for cluster_id in np.unique(y_pred):
        values = y_true[y_pred == cluster_id]

        if len(values) > 0:
            total += np.bincount(values).max()

    return float(total / len(y_true))


def inverse_purity(y_true, y_pred):
    """
    Pur_a: category-side inverse purity.
    For every true category, count its dominant predicted cluster.
    """
    total = 0

    for category_id in np.unique(y_true):
        values = y_pred[y_true == category_id]

        if len(values) > 0:
            total += np.bincount(values).max()

    return float(total / len(y_true))


def evaluate_predictions(predictions, label_sets, indices=None):
    predictions = np.asarray(predictions, dtype=np.int64)

    if indices is None:
        indices = np.arange(len(predictions), dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)

    pred = predictions[indices]

    result = {}
    nmis = []
    pur_ps = []
    pur_as = []

    for level_name, labels in label_sets.items():
        true = labels[indices]

        nmi = normalized_mutual_info_score(
            true,
            pred,
            average_method="arithmetic"
        )
        pur_p = cluster_purity(true, pred)
        pur_a = inverse_purity(true, pred)

        result[f"{level_name}_nmi"] = float(nmi)
        result[f"{level_name}_pur_p"] = float(pur_p)
        result[f"{level_name}_pur_a"] = float(pur_a)

        nmis.append(nmi)
        pur_ps.append(pur_p)
        pur_as.append(pur_a)

    counts = np.bincount(pred, minlength=N_CLUSTERS)

    result["mean_nmi"] = float(np.mean(nmis))
    result["mean_pur_p"] = float(np.mean(pur_ps))
    result["mean_pur_a"] = float(np.mean(pur_as))
    result["active_topics"] = int(np.sum(counts > 0))
    result["max_topic_share"] = float(counts.max() / len(pred))

    return result


def l2_normalize_array(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def sanitize_name(name):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return name.strip("_")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj


# ------------------------------------------------------------
# 5. GPU spherical K-means
# ------------------------------------------------------------
@torch.no_grad()
def gpu_spherical_kmeans(
    features,
    n_clusters=30,
    seed=42,
    max_iter=100,
    tol=1e-5
):
    x = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    )

    x = torch.nn.functional.normalize(x, p=2, dim=1)

    n_samples, n_features = x.shape

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))

    # K-means++ initialization
    first_index = torch.randint(
        0,
        n_samples,
        (1,),
        generator=generator,
        device=DEVICE
    ).item()

    centroid_indices = [first_index]
    closest_distance = 1.0 - torch.matmul(
        x,
        x[first_index].unsqueeze(1)
    ).squeeze(1)

    for _ in range(1, n_clusters):
        probabilities = torch.clamp(closest_distance, min=1e-8)
        probabilities = probabilities / probabilities.sum()

        next_index = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator
        ).item()

        centroid_indices.append(next_index)

        new_distance = 1.0 - torch.matmul(
            x,
            x[next_index].unsqueeze(1)
        ).squeeze(1)

        closest_distance = torch.minimum(
            closest_distance,
            new_distance
        )

    centroids = x[
        torch.tensor(
            centroid_indices,
            dtype=torch.long,
            device=DEVICE
        )
    ].clone()

    centroids = torch.nn.functional.normalize(
        centroids,
        p=2,
        dim=1
    )

    previous_objective = None
    labels = None

    for iteration in range(max_iter):
        similarities = torch.matmul(x, centroids.T)
        max_similarities, new_labels = similarities.max(dim=1)

        new_centroids = torch.zeros(
            (n_clusters, n_features),
            dtype=torch.float32,
            device=DEVICE
        )

        new_centroids.index_add_(0, new_labels, x)

        counts = torch.bincount(
            new_labels,
            minlength=n_clusters
        ).float()

        empty_clusters = torch.where(counts == 0)[0]

        if len(empty_clusters) > 0:
            difficult_points = torch.argsort(
                max_similarities
            )[:len(empty_clusters)]

            new_centroids[empty_clusters] = x[difficult_points]
            counts[empty_clusters] = 1.0

        new_centroids = new_centroids / counts.unsqueeze(1)
        new_centroids = torch.nn.functional.normalize(
            new_centroids,
            p=2,
            dim=1
        )

        objective = float(max_similarities.mean().item())

        centroids = new_centroids
        labels = new_labels

        if previous_objective is not None:
            if abs(objective - previous_objective) < tol:
                break

        previous_objective = objective

    final_similarities = torch.matmul(x, centroids.T)
    final_values, final_labels = final_similarities.max(dim=1)

    predictions = final_labels.detach().cpu().numpy().astype(np.int64)
    mean_similarity = float(final_values.mean().item())

    del x
    del centroids
    del final_similarities
    del final_values
    del final_labels

    torch.cuda.empty_cache()

    return predictions, mean_similarity, iteration + 1


# ------------------------------------------------------------
# 6. Validate paths
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("PATH VALIDATION")
print("=" * 80)

require_file(THETA_PATH, "Theta")
require_file(RECORDS_PATH, "DEV records")
require_file(SEMANTIC_PATH, "Document semantic")
require_file(LEXICAL_PATH, "Lexical")
require_file(POOL_CACHE_PATH, "Claim-wise pool cache")

stable_reference_path = None

for candidate in STABLE_REFERENCE_CANDIDATES:
    if candidate.exists():
        stable_reference_path = candidate
        break

if stable_reference_path is None:
    matches = list(
        RUN_ROOT.rglob("best_stable_hybrid_dev_predictions.npy")
    )

    if matches:
        stable_reference_path = matches[0]

if stable_reference_path is None:
    raise FileNotFoundError(
        "best_stable_hybrid_dev_predictions.npy를 찾지 못했습니다."
    )

print("Stable reference:", stable_reference_path)
print("Output directory:", OUTPUT_DIR)

# ------------------------------------------------------------
# 7. Load primary inputs
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("LOADING INPUTS")
print("=" * 80)

start_time = time.time()

theta = np.load(THETA_PATH).astype(np.float32)
document_semantic = np.load(SEMANTIC_PATH).astype(np.float32)
lexical = np.load(LEXICAL_PATH).astype(np.float32)
stable_predictions = np.load(stable_reference_path).astype(np.int64)

with open(RECORDS_PATH, "rb") as file:
    records = pickle.load(file)

print("Theta shape:", theta.shape)
print("Document semantic shape:", document_semantic.shape)
print("Lexical shape:", lexical.shape)
print("Stable predictions shape:", stable_predictions.shape)
print("Records:", len(records))

if theta.shape != (N_PATENTS, N_CLUSTERS):
    raise ValueError(f"Unexpected theta shape: {theta.shape}")

if document_semantic.shape[0] != N_PATENTS:
    raise ValueError(
        f"Unexpected document semantic shape: {document_semantic.shape}"
    )

if lexical.shape[0] != N_PATENTS:
    raise ValueError(f"Unexpected lexical shape: {lexical.shape}")

if stable_predictions.shape[0] != N_PATENTS:
    raise ValueError(
        f"Unexpected stable prediction shape: {stable_predictions.shape}"
    )

if len(records) != N_PATENTS:
    raise ValueError(f"Unexpected record count: {len(records)}")

# ------------------------------------------------------------
# 8. CPC labels
# ------------------------------------------------------------
sections = []
classes = []
subclasses = []
patent_ids = []

for i, record in enumerate(records):
    section, class_label, subclass = extract_cpc_labels(record)

    sections.append(section)
    classes.append(class_label)
    subclasses.append(subclass)

    patent_id = get_record_value(
        record,
        ["patent_id", "publication_number", "id"],
        default=i
    )
    patent_ids.append(safe_string(patent_id))

label_sets = {
    "section": encode_labels(sections),
    "class": encode_labels(classes),
    "subclass": encode_labels(subclasses),
}

print(
    "CPC cardinalities:",
    len(np.unique(label_sets["section"])),
    len(np.unique(label_sets["class"])),
    len(np.unique(label_sets["subclass"]))
)

# ------------------------------------------------------------
# 9. Deterministic tune/holdout split
# ------------------------------------------------------------
all_indices = np.arange(N_PATENTS, dtype=np.int64)

tune_indices, holdout_indices = train_test_split(
    all_indices,
    train_size=TUNE_RATIO,
    random_state=SPLIT_SEED,
    shuffle=True
)

tune_indices = np.sort(tune_indices)
holdout_indices = np.sort(holdout_indices)

print("Tune patents:", len(tune_indices))
print("Holdout patents:", len(holdout_indices))

# ------------------------------------------------------------
# 10. Reference metrics
# ------------------------------------------------------------
reference_tune = evaluate_predictions(
    stable_predictions,
    label_sets,
    tune_indices
)

reference_holdout = evaluate_predictions(
    stable_predictions,
    label_sets,
    holdout_indices
)

reference_full = evaluate_predictions(
    stable_predictions,
    label_sets
)

print("\nREFERENCE METRICS")
print(
    f"Tune Mean NMI={reference_tune['mean_nmi']:.6f}, "
    f"Pur_p={reference_tune['mean_pur_p']:.6f}, "
    f"Pur_a={reference_tune['mean_pur_a']:.6f}"
)
print(
    f"Holdout Mean NMI={reference_holdout['mean_nmi']:.6f}, "
    f"Pur_p={reference_holdout['mean_pur_p']:.6f}, "
    f"Pur_a={reference_holdout['mean_pur_a']:.6f}"
)
print(
    f"Full Mean NMI={reference_full['mean_nmi']:.6f}, "
    f"Pur_p={reference_full['mean_pur_p']:.6f}, "
    f"Pur_a={reference_full['mean_pur_a']:.6f}"
)

# ------------------------------------------------------------
# 11. Robust loading of claim-wise pools
# ------------------------------------------------------------
def load_claim_pool_matrices(npz_path):
    archive = np.load(npz_path, allow_pickle=True)

    print("\nClaim-wise cache keys:")
    for key in archive.files:
        value = archive[key]
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    pools = {}

    # Format A:
    # uniform -> (9855, 768)
    # root8_d010 -> (9855, 768)
    for key in archive.files:
        value = archive[key]

        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.shape[0] == N_PATENTS
            and value.shape[1] >= 64
        ):
            lower_key = key.lower()

            if not any(
                marker in lower_key
                for marker in [
                    "patent_id",
                    "global_index",
                    "claim_count",
                    "depth",
                    "seen",
                    "mask"
                ]
            ):
                pools[sanitize_name(key)] = value.astype(np.float32)

    # Format B:
    # pools -> (n_configs, 9855, hidden_dim)
    if not pools:
        possible_array_keys = [
            "pools",
            "patent_pools",
            "pool_array",
            "embeddings"
        ]

        possible_name_keys = [
            "config_names",
            "pool_names",
            "names",
            "pooling_configs"
        ]

        stacked = None
        names = None

        for key in possible_array_keys:
            if key in archive.files:
                value = archive[key]

                if value.ndim == 3:
                    stacked = value
                    break

        for key in possible_name_keys:
            if key in archive.files:
                names = archive[key].tolist()
                break

        if stacked is not None:
            if stacked.shape[1] == N_PATENTS:
                config_axis_first = stacked
            elif stacked.shape[0] == N_PATENTS:
                config_axis_first = np.transpose(stacked, (1, 0, 2))
            else:
                raise ValueError(
                    f"Cannot identify patent axis in stacked pool: "
                    f"{stacked.shape}"
                )

            if names is None:
                names = [
                    f"pool_{i:02d}"
                    for i in range(config_axis_first.shape[0])
                ]

            if len(names) != config_axis_first.shape[0]:
                raise ValueError(
                    "Number of pool names does not match stacked pools."
                )

            for i, name in enumerate(names):
                pools[sanitize_name(name)] = (
                    config_axis_first[i].astype(np.float32)
                )

    if not pools:
        raise ValueError(
            "캐시에서 (9855, hidden_dim) claim-wise pool을 찾지 못했습니다."
        )

    for name, matrix in pools.items():
        if matrix.shape[0] != N_PATENTS:
            raise ValueError(
                f"Pool {name} has invalid shape: {matrix.shape}"
            )

        if not np.all(np.isfinite(matrix)):
            print(f"Warning: replacing non-finite values in {name}")
            matrix = np.nan_to_num(
                matrix,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            pools[name] = matrix.astype(np.float32)

    return pools


raw_claim_pools = load_claim_pool_matrices(POOL_CACHE_PATH)

print("\nDetected claim-wise pooling variants:")
for name, matrix in raw_claim_pools.items():
    print(f"  {name}: {matrix.shape}")

# ------------------------------------------------------------
# 12. Shared PCA for claim-wise pools
# ------------------------------------------------------------
def load_or_build_claim_pca(raw_pools):
    pool_names = list(raw_pools.keys())

    if PCA_CACHE_PATH.exists():
        cached = np.load(PCA_CACHE_PATH, allow_pickle=True)

        cached_names = [
            safe_string(x)
            for x in cached["pool_names"].tolist()
        ]

        valid = (
            cached_names == pool_names
            and all(name in cached.files for name in pool_names)
            and all(
                cached[name].shape == (N_PATENTS, CLAIM_PCA_DIM)
                for name in pool_names
            )
        )

        if valid:
            print("\nLoading cached PCA claim-wise features:")
            print(PCA_CACHE_PATH)

            return {
                name: cached[name].astype(np.float32)
                for name in pool_names
            }

        print("Existing PCA cache does not match current pools. Rebuilding.")

    # Prefer uniform pool as the shared PCA fitting source.
    fit_name = None

    for name in pool_names:
        if "uniform" in name.lower():
            fit_name = name
            break

    if fit_name is None:
        fit_name = pool_names[0]

    print("\nFitting shared randomized PCA")
    print("PCA fitting source:", fit_name)
    print("Target dimension:", CLAIM_PCA_DIM)

    fit_matrix = raw_pools[fit_name].astype(np.float32)

    pca = PCA(
        n_components=CLAIM_PCA_DIM,
        svd_solver="randomized",
        random_state=GLOBAL_SEED,
        whiten=False
    )

    pca.fit(fit_matrix)
    joblib.dump(pca, PCA_MODEL_PATH)

    reduced = {}

    for name in pool_names:
        print("Transforming:", name)
        transformed = pca.transform(
            raw_pools[name].astype(np.float32)
        ).astype(np.float32)

        transformed = l2_normalize_array(transformed)
        reduced[name] = transformed

    save_payload = {
        "pool_names": np.asarray(pool_names, dtype=object)
    }

    for name in pool_names:
        save_payload[name] = reduced[name]

    np.savez_compressed(PCA_CACHE_PATH, **save_payload)

    print("Saved PCA model:", PCA_MODEL_PATH)
    print("Saved PCA features:", PCA_CACHE_PATH)
    print(
        "Explained variance ratio:",
        float(pca.explained_variance_ratio_.sum())
    )

    return reduced


claim_features = load_or_build_claim_pca(raw_claim_pools)

# Release large raw matrices
del raw_claim_pools
gc.collect()

# ------------------------------------------------------------
# 13. Prepare normalized base feature blocks
# ------------------------------------------------------------
document_semantic = l2_normalize_array(document_semantic)
lexical = l2_normalize_array(lexical)

theta = np.clip(theta, 1e-12, None)
theta = theta / np.maximum(
    theta.sum(axis=1, keepdims=True),
    1e-12
)

theta_cache = {}

for q in sorted(set(config["q"] for config in FUSION_CONFIGS)):
    transformed = np.power(theta, q).astype(np.float32)
    theta_cache[q] = l2_normalize_array(transformed)


def build_fused_features(config, claim_matrix):
    blocks = []

    if config["t"] > 0:
        blocks.append(
            np.sqrt(config["t"]) * theta_cache[config["q"]]
        )

    if config["s"] > 0:
        blocks.append(
            np.sqrt(config["s"]) * document_semantic
        )

    if config["l"] > 0:
        blocks.append(
            np.sqrt(config["l"]) * lexical
        )

    if config["c"] > 0:
        blocks.append(
            np.sqrt(config["c"]) * claim_matrix
        )

    if not blocks:
        raise ValueError("Fusion configuration has no active features.")

    fused = np.concatenate(blocks, axis=1).astype(np.float32)
    fused = l2_normalize_array(fused)

    return fused

# ------------------------------------------------------------
# 14. Full GPU search
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("GPU CLAIM-WISE FUSION SEARCH")
print("=" * 80)

total_runs = (
    len(claim_features)
    * len(FUSION_CONFIGS)
    * len(KMEANS_SEEDS)
)

print("Pooling variants:", len(claim_features))
print("Fusion configurations:", len(FUSION_CONFIGS))
print("Seeds:", KMEANS_SEEDS)
print("Total clustering runs:", total_runs)

ranking_rows = []
prediction_store = {}

run_number = 0
search_start = time.time()

for pool_name, claim_matrix in claim_features.items():
    print("\n" + "-" * 80)
    print("POOLING VARIANT:", pool_name)
    print("-" * 80)

    for config in FUSION_CONFIGS:
        fused_features = build_fused_features(
            config,
            claim_matrix
        )

        for seed in KMEANS_SEEDS:
            run_number += 1

            variant = (
                f"{pool_name}__{config['name']}__seed{seed}"
            )

            run_start = time.time()

            predictions, mean_similarity, iterations = (
                gpu_spherical_kmeans(
                    fused_features,
                    n_clusters=N_CLUSTERS,
                    seed=seed,
                    max_iter=KMEANS_MAX_ITER,
                    tol=KMEANS_TOL
                )
            )

            tune_metrics = evaluate_predictions(
                predictions,
                label_sets,
                tune_indices
            )

            row = {
                "variant": variant,
                "pool_name": pool_name,
                "fusion_name": config["name"],
                "seed": int(seed),
                "q": float(config["q"]),
                "theta_weight": float(config["t"]),
                "document_semantic_weight": float(config["s"]),
                "lexical_weight": float(config["l"]),
                "claimwise_weight": float(config["c"]),
                "feature_dimension": int(fused_features.shape[1]),
                "iterations": int(iterations),
                "mean_cosine_similarity": float(mean_similarity),

                "tune_mean_nmi": tune_metrics["mean_nmi"],
                "tune_mean_pur_p": tune_metrics["mean_pur_p"],
                "tune_mean_pur_a": tune_metrics["mean_pur_a"],

                "tune_section_nmi": tune_metrics["section_nmi"],
                "tune_class_nmi": tune_metrics["class_nmi"],
                "tune_subclass_nmi": tune_metrics["subclass_nmi"],

                "tune_active_topics": tune_metrics["active_topics"],
                "tune_max_topic_share": tune_metrics["max_topic_share"],

                "stable": bool(
                    tune_metrics["mean_nmi"]
                    >= reference_tune["mean_nmi"]
                    + MIN_TUNE_NMI_GAIN

                    and tune_metrics["mean_pur_a"]
                    >= reference_tune["mean_pur_a"]
                    - MAX_PURA_DROP

                    and tune_metrics["active_topics"]
                    >= MIN_ACTIVE_TOPICS

                    and tune_metrics["max_topic_share"]
                    <= MAX_TOPIC_SHARE
                ),

                "runtime_sec": float(time.time() - run_start)
            }

            ranking_rows.append(row)
            prediction_store[variant] = predictions

            if (
                run_number == 1
                or run_number % 10 == 0
                or run_number == total_runs
            ):
                elapsed = (time.time() - search_start) / 60.0

                print(
                    f"[{run_number:03d}/{total_runs:03d}] "
                    f"{variant} | "
                    f"NMI={row['tune_mean_nmi']:.6f} | "
                    f"Pur_p={row['tune_mean_pur_p']:.6f} | "
                    f"Pur_a={row['tune_mean_pur_a']:.6f} | "
                    f"stable={row['stable']} | "
                    f"elapsed={elapsed:.1f} min"
                )

        del fused_features
        gc.collect()
        torch.cuda.empty_cache()

# ------------------------------------------------------------
# 15. Rank and select
# ------------------------------------------------------------
ranking_rows.sort(
    key=lambda row: (
        row["tune_mean_nmi"],
        row["tune_mean_pur_p"],
        row["tune_mean_pur_a"]
    ),
    reverse=True
)

best_raw = ranking_rows[0]
stable_rows = [row for row in ranking_rows if row["stable"]]

if stable_rows:
    stable_rows.sort(
        key=lambda row: (
            row["tune_mean_nmi"],
            row["tune_mean_pur_p"],
            row["tune_mean_pur_a"]
        ),
        reverse=True
    )

    selected = stable_rows[0]
    selection_type = "BEST_STABLE_CLAIMWISE_FUSION"
else:
    selected = best_raw
    selection_type = "BEST_RAW_CLAIMWISE_FUSION_NO_STABLE_CANDIDATE"

selected_predictions = prediction_store[selected["variant"]]

selected_holdout = evaluate_predictions(
    selected_predictions,
    label_sets,
    holdout_indices
)

selected_full = evaluate_predictions(
    selected_predictions,
    label_sets
)

# Add holdout and full metrics to selected row
for key, value in selected_holdout.items():
    selected[f"holdout_{key}"] = value

for key, value in selected_full.items():
    selected[f"full_{key}"] = value

selected["selection_type"] = selection_type

selected["holdout_nmi_delta_vs_reference"] = (
    selected_holdout["mean_nmi"]
    - reference_holdout["mean_nmi"]
)

selected["full_nmi_delta_vs_reference"] = (
    selected_full["mean_nmi"]
    - reference_full["mean_nmi"]
)

selected["full_pur_p_delta_vs_reference"] = (
    selected_full["mean_pur_p"]
    - reference_full["mean_pur_p"]
)

selected["full_pur_a_delta_vs_reference"] = (
    selected_full["mean_pur_a"]
    - reference_full["mean_pur_a"]
)

if (
    selected_full["mean_nmi"] > reference_full["mean_nmi"]
    and selected_holdout["mean_nmi"]
        > reference_holdout["mean_nmi"]
    and selected_full["mean_pur_a"]
        >= reference_full["mean_pur_a"] - MAX_PURA_DROP
):
    decision = "CLAIMWISE_FUSION_IMPROVES_NMI_AND_PASSES_STABILITY"
elif (
    selected_full["mean_nmi"] > reference_full["mean_nmi"]
    and selected_holdout["mean_nmi"]
        > reference_holdout["mean_nmi"]
):
    decision = "CLAIMWISE_FUSION_IMPROVES_NMI_BUT_NEEDS_BALANCING"
elif selected_full["mean_nmi"] > reference_full["mean_nmi"]:
    decision = "FULL_DEV_IMPROVES_BUT_HOLDOUT_DOES_NOT_CONFIRM"
else:
    decision = "KEEP_STABLE_REFERENCE"

# ------------------------------------------------------------
# 16. Save ranking CSV
# ------------------------------------------------------------
field_names = sorted(
    set(
        key
        for row in ranking_rows
        for key in row.keys()
    )
)

with open(RANKING_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()

    for row in ranking_rows:
        writer.writerow(row)

# ------------------------------------------------------------
# 17. Save selected predictions and assignments
# ------------------------------------------------------------
np.save(PREDICTIONS_PATH, selected_predictions.astype(np.int64))

with open(ASSIGNMENTS_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "global_index",
        "patent_id",
        "section",
        "class",
        "subclass",
        "stable_reference_topic",
        "selected_claimwise_topic",
        "is_tune",
        "is_holdout"
    ])

    tune_set = set(tune_indices.tolist())
    holdout_set = set(holdout_indices.tolist())

    for i in range(N_PATENTS):
        writer.writerow([
            i,
            patent_ids[i],
            sections[i],
            classes[i],
            subclasses[i],
            int(stable_predictions[i]),
            int(selected_predictions[i]),
            int(i in tune_set),
            int(i in holdout_set)
        ])

# ------------------------------------------------------------
# 18. Save summary JSON
# ------------------------------------------------------------
runtime_minutes = (time.time() - start_time) / 60.0

summary = {
    "experiment": "epoch016_gpu_claimwise_semantic_fusion_dev_search",
    "test_used": False,
    "device": str(DEVICE),
    "gpu_name": torch.cuda.get_device_name(0),

    "paths": {
        "run_root": str(RUN_ROOT),
        "theta": str(THETA_PATH),
        "records": str(RECORDS_PATH),
        "document_semantic": str(SEMANTIC_PATH),
        "lexical": str(LEXICAL_PATH),
        "claim_pool_cache": str(POOL_CACHE_PATH),
        "stable_reference": str(stable_reference_path),
        "ranking_csv": str(RANKING_PATH),
        "summary_json": str(SUMMARY_PATH),
        "predictions_npy": str(PREDICTIONS_PATH),
        "assignments_csv": str(ASSIGNMENTS_PATH)
    },

    "data": {
        "n_patents": N_PATENTS,
        "n_clusters": N_CLUSTERS,
        "tune_size": len(tune_indices),
        "holdout_size": len(holdout_indices),
        "pooling_variants": list(claim_features.keys()),
        "fusion_config_count": len(FUSION_CONFIGS),
        "seeds": KMEANS_SEEDS,
        "total_runs": total_runs
    },

    "stable_reference": {
        "tune": reference_tune,
        "holdout": reference_holdout,
        "full_dev": reference_full
    },

    "best_raw_candidate": best_raw,
    "stable_candidate_count": len(stable_rows),
    "selected_candidate": selected,

    "selected_holdout": selected_holdout,
    "selected_full_dev": selected_full,

    "decision": decision,
    "runtime_minutes": runtime_minutes
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    json.dump(
        json_safe(summary),
        file,
        indent=2,
        ensure_ascii=False
    )

# ------------------------------------------------------------
# 19. Final report
# ------------------------------------------------------------
print("\n\n" + "=" * 80)
print("BEST RAW CLAIM-WISE CANDIDATE")
print("=" * 80)
print("Variant:", best_raw["variant"])
print("Pooling:", best_raw["pool_name"])
print("Fusion:", best_raw["fusion_name"])
print("Seed:", best_raw["seed"])
print(f"Tune Mean NMI: {best_raw['tune_mean_nmi']:.6f}")
print(f"Tune Mean Pur_p: {best_raw['tune_mean_pur_p']:.6f}")
print(f"Tune Mean Pur_a: {best_raw['tune_mean_pur_a']:.6f}")
print(f"Tune Section NMI: {best_raw['tune_section_nmi']:.6f}")
print(f"Tune Class NMI: {best_raw['tune_class_nmi']:.6f}")
print(f"Tune Subclass NMI: {best_raw['tune_subclass_nmi']:.6f}")
print("Stable:", best_raw["stable"])

print("\n" + "=" * 80)
print("SELECTED CANDIDATE")
print("=" * 80)
print("Selection type:", selection_type)
print("Stable candidate count:", len(stable_rows))
print("Variant:", selected["variant"])
print("Pooling:", selected["pool_name"])
print("Fusion:", selected["fusion_name"])
print("Seed:", selected["seed"])
print(
    "Weights:",
    f"q={selected['q']:.2f},",
    f"theta={selected['theta_weight']:.2f},",
    f"doc_sem={selected['document_semantic_weight']:.2f},",
    f"lexical={selected['lexical_weight']:.2f},",
    f"claim={selected['claimwise_weight']:.2f}"
)
print(f"Tune Mean NMI: {selected['tune_mean_nmi']:.6f}")
print(f"Tune Mean Pur_p: {selected['tune_mean_pur_p']:.6f}")
print(f"Tune Mean Pur_a: {selected['tune_mean_pur_a']:.6f}")

print("\n" + "=" * 80)
print("SELECTED HOLDOUT")
print("=" * 80)
print(
    f"Reference Holdout Mean NMI: "
    f"{reference_holdout['mean_nmi']:.6f}"
)
print(
    f"Selected Holdout Mean NMI: "
    f"{selected_holdout['mean_nmi']:.6f}"
)
print(
    f"Holdout NMI Delta: "
    f"{selected['holdout_nmi_delta_vs_reference']:+.6f}"
)
print(
    f"Selected Holdout Mean Pur_p: "
    f"{selected_holdout['mean_pur_p']:.6f}"
)
print(
    f"Selected Holdout Mean Pur_a: "
    f"{selected_holdout['mean_pur_a']:.6f}"
)

print("\n" + "=" * 80)
print("SELECTED FULL DEV")
print("=" * 80)
print(f"Reference Mean NMI: {reference_full['mean_nmi']:.6f}")
print(f"Selected Mean NMI: {selected_full['mean_nmi']:.6f}")
print(
    f"Full DEV NMI Delta: "
    f"{selected['full_nmi_delta_vs_reference']:+.6f}"
)
print(f"Reference Mean Pur_p: {reference_full['mean_pur_p']:.6f}")
print(f"Selected Mean Pur_p: {selected_full['mean_pur_p']:.6f}")
print(
    f"Pur_p Delta: "
    f"{selected['full_pur_p_delta_vs_reference']:+.6f}"
)
print(f"Reference Mean Pur_a: {reference_full['mean_pur_a']:.6f}")
print(f"Selected Mean Pur_a: {selected_full['mean_pur_a']:.6f}")
print(
    f"Pur_a Delta: "
    f"{selected['full_pur_a_delta_vs_reference']:+.6f}"
)
print(f"Section NMI: {selected_full['section_nmi']:.6f}")
print(f"Class NMI: {selected_full['class_nmi']:.6f}")
print(f"Subclass NMI: {selected_full['subclass_nmi']:.6f}")
print("Active topics:", selected_full["active_topics"])
print(
    f"Max topic share: "
    f"{selected_full['max_topic_share'] * 100:.2f}%"
)

print("\n" + "=" * 80)
print("FINAL DECISION")
print("=" * 80)
print("Decision:", decision)
print("TEST executed: NO")
print(f"Runtime: {runtime_minutes:.2f} minutes")
print("Ranking CSV:", RANKING_PATH)
print("Summary JSON:", SUMMARY_PATH)
print("Predictions NPY:", PREDICTIONS_PATH)
print("Assignments CSV:", ASSIGNMENTS_PATH)

print("\n" + "=" * 80)
print("TOP 10 TUNE CANDIDATES")
print("=" * 80)

for rank, row in enumerate(ranking_rows[:10], start=1):
    print(
        f"{rank:2d}. {row['variant']} | "
        f"NMI={row['tune_mean_nmi']:.6f} | "
        f"Pur_p={row['tune_mean_pur_p']:.6f} | "
        f"Pur_a={row['tune_mean_pur_a']:.6f} | "
        f"stable={row['stable']}"
    )

print("\nExperiment complete.")



