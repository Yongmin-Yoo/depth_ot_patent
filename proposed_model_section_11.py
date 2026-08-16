# ============================================================
# FINAL FIXED TEST EVALUATION
# Root12 + Depth Decay 0.1 + Claim-Only
# T4 GPU, resumable, single-cell Colab code
#
# IMPORTANT:
# - No TEST hyperparameter tuning
# - Uses DEV-fitted PCA model
# - Seeds: 17, 42, 73
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
import shutil
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import joblib

warnings.filterwarnings("ignore")

from sklearn.metrics import normalized_mutual_info_score

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
GLOBAL_SEED = 42
KMEANS_SEEDS = [17, 42, 73]

N_CLUSTERS = 30
HIDDEN_DIM = 768
PCA_DIM = 128

ROOT_WEIGHT = 12.0
DEPTH_DECAY = 0.1

KMEANS_MAX_ITER = 100
KMEANS_TOL = 1e-5

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

if DEVICE.type != "cuda":
    raise RuntimeError(
        "GPU가 감지되지 않았습니다. "
        "Colab 메뉴에서 런타임 유형을 T4 GPU로 변경하세요."
    )

print("=" * 88)
print("FINAL FIXED TEST EVALUATION")
print("=" * 88)
print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0))
print("Pooling: root12_d010")
print("Root weight:", ROOT_WEIGHT)
print("Depth decay:", DEPTH_DECAY)
print("PCA:", PCA_DIM, "dimensions, fitted on DEV")
print("K-means seeds:", KMEANS_SEEDS)
print("TEST tuning: NO")

# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
RUN_ROOT = Path(
    "/content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/"
    "depth_ot_v2_patent_semantic_seed42_20260814_055110"
)

TEST_RECORDS_PATH = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/test_records.pkl"
)

TEST_FEATURE_DIR = Path(
    "/content/drive/MyDrive/depth_ot_patent/data/processed/"
    "token_features/full/test"
)

DEV_PCA_MODEL_PATH = (
    RUN_ROOT
    / "epoch016_gpu_claimwise_semantic_fusion_dev_search"
    / "models"
    / "claimwise_semantic_pca128.joblib"
)

OUTPUT_DIR = (
    RUN_ROOT
    / "final_fixed_root12_d010_claimwise_test_evaluation"
)

CACHE_DIR = OUTPUT_DIR / "cache"
PARTIAL_DIR = CACHE_DIR / "test_shard_partials"
LOCAL_SHARD_DIR = Path("/content/depth_ot_test_shards")

RAW_POOL_PATH = CACHE_DIR / "test_root12_d010_raw_pool.npy"
PCA_FEATURE_PATH = CACHE_DIR / "test_root12_d010_pca128.npy"

PREDICTIONS_PATH = OUTPUT_DIR / "test_3seed_predictions.npz"
ASSIGNMENTS_PATH = OUTPUT_DIR / "test_3seed_assignments.csv"
RESULTS_PATH = OUTPUT_DIR / "test_3seed_metrics.json"
RESULTS_CSV_PATH = OUTPUT_DIR / "test_3seed_metrics.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SHARD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 3. Basic utilities
# ------------------------------------------------------------
def require_file(path, description):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일을 찾을 수 없습니다:\n{path}"
        )

    print(f"{description}: {path}")


def safe_string(value):
    if value is None:
        return "UNKNOWN"

    if torch.is_tensor(value):
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            value = value.tolist()

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    value = str(value).strip()
    return value if value else "UNKNOWN"


def normalize_id(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.item()

    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    text = str(value).strip()

    if not text:
        return None

    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]

    return text


def claim_number(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.item()

    try:
        return int(value)
    except Exception:
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else None


def get_record_value(record, keys, default=None):
    if not isinstance(record, dict):
        return default

    for key in keys:
        if key not in record or record[key] is None:
            continue

        value = record[key]

        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) > 0:
                return value[0]
        else:
            return value

    return default


def encode_labels(values):
    unique_values = sorted(set(map(str, values)))
    mapping = {
        value: index
        for index, value in enumerate(unique_values)
    }

    encoded = np.asarray(
        [mapping[str(value)] for value in values],
        dtype=np.int64
    )

    return encoded, mapping


def extract_cpc_labels(record):
    section = safe_string(
        get_record_value(
            record,
            ["section", "cpc_section", "section_label"],
            "UNKNOWN"
        )
    )

    class_label = safe_string(
        get_record_value(
            record,
            ["class", "cpc_class", "class_label"],
            "UNKNOWN"
        )
    )

    subclass = safe_string(
        get_record_value(
            record,
            ["subclass", "cpc_subclass", "subclass_label"],
            "UNKNOWN"
        )
    )

    codes = record.get("cpc_codes", []) if isinstance(record, dict) else []

    if isinstance(codes, str):
        codes = [codes]

    if codes:
        code = re.sub(
            r"[^A-Z0-9]",
            "",
            safe_string(codes[0]).upper()
        )

        if section == "UNKNOWN" and len(code) >= 1:
            section = code[:1]

        if class_label == "UNKNOWN" and len(code) >= 3:
            class_label = code[:3]

        if subclass == "UNKNOWN" and len(code) >= 4:
            subclass = code[:4]

    return section, class_label, subclass


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


# ------------------------------------------------------------
# 4. Metrics
# ------------------------------------------------------------
def cluster_purity(y_true, y_pred):
    total = 0

    for cluster_id in np.unique(y_pred):
        values = y_true[y_pred == cluster_id]

        if len(values) > 0:
            total += np.bincount(values).max()

    return float(total / len(y_true))


def inverse_purity(y_true, y_pred):
    total = 0

    for category_id in np.unique(y_true):
        values = y_pred[y_true == category_id]

        if len(values) > 0:
            total += np.bincount(values).max()

    return float(total / len(y_true))


def evaluate_predictions(predictions, label_sets):
    predictions = np.asarray(
        predictions,
        dtype=np.int64
    )

    result = {}
    nmi_values = []
    pur_p_values = []
    pur_a_values = []

    for level, labels in label_sets.items():
        nmi = normalized_mutual_info_score(
            labels,
            predictions,
            average_method="arithmetic"
        )

        pur_p = cluster_purity(
            labels,
            predictions
        )

        pur_a = inverse_purity(
            labels,
            predictions
        )

        result[f"{level}_nmi"] = float(nmi)
        result[f"{level}_pur_p"] = float(pur_p)
        result[f"{level}_pur_a"] = float(pur_a)

        nmi_values.append(nmi)
        pur_p_values.append(pur_p)
        pur_a_values.append(pur_a)

    counts = np.bincount(
        predictions,
        minlength=N_CLUSTERS
    )

    result["mean_nmi"] = float(
        np.mean(nmi_values)
    )

    result["mean_pur_p"] = float(
        np.mean(pur_p_values)
    )

    result["mean_pur_a"] = float(
        np.mean(pur_a_values)
    )

    result["active_topics"] = int(
        np.sum(counts > 0)
    )

    result["max_topic_share"] = float(
        counts.max() / len(predictions)
    )

    return result


def aggregate_metrics(seed_metrics):
    metric_keys = list(
        next(iter(seed_metrics.values())).keys()
    )

    aggregate = {}

    for key in metric_keys:
        values = np.asarray(
            [
                seed_metrics[seed][key]
                for seed in KMEANS_SEEDS
            ],
            dtype=np.float64
        )

        aggregate[f"{key}_mean"] = float(
            values.mean()
        )

        aggregate[f"{key}_std"] = float(
            values.std(ddof=0)
        )

    return aggregate


# ------------------------------------------------------------
# 5. Load records and construct index/depth maps
# ------------------------------------------------------------
print("\n" + "=" * 88)
print("PATH VALIDATION")
print("=" * 88)

require_file(TEST_RECORDS_PATH, "TEST records")
require_file(DEV_PCA_MODEL_PATH, "DEV-fitted PCA model")

if not TEST_FEATURE_DIR.exists():
    raise FileNotFoundError(
        f"TEST feature directory를 찾을 수 없습니다:\n"
        f"{TEST_FEATURE_DIR}"
    )

print("TEST feature directory:", TEST_FEATURE_DIR)
print("Output directory:", OUTPUT_DIR)

with open(TEST_RECORDS_PATH, "rb") as file:
    records = pickle.load(file)

N_TEST = len(records)

print("\nTEST patents:", N_TEST)

patent_id_to_index = {}
patent_ids = []
depth_maps = []
expected_claim_count = 0

for global_index, record in enumerate(records):
    patent_id = get_record_value(
        record,
        ["patent_id", "publication_number", "id"],
        global_index
    )

    normalized_patent_id = normalize_id(patent_id)

    patent_ids.append(
        safe_string(patent_id)
    )

    patent_id_to_index[
        normalized_patent_id
    ] = global_index

    # Also support integer-like alternatives.
    try:
        patent_id_to_index[
            str(int(float(normalized_patent_id)))
        ] = global_index
    except Exception:
        pass

    record_depth = (
        record.get("depth", {})
        if isinstance(record, dict)
        else {}
    )

    normalized_depth = {}

    if isinstance(record_depth, dict):
        for claim_id, depth in record_depth.items():
            claim_id_number = claim_number(claim_id)

            if claim_id_number is None:
                continue

            try:
                normalized_depth[
                    claim_id_number
                ] = int(depth)
            except Exception:
                continue

    depth_maps.append(normalized_depth)

    claims = (
        record.get("claims", {})
        if isinstance(record, dict)
        else {}
    )

    if isinstance(claims, dict):
        expected_claim_count += len(claims)
    elif isinstance(claims, (list, tuple)):
        expected_claim_count += len(claims)

print("Expected TEST claims:", expected_claim_count)

# ------------------------------------------------------------
# 6. Locate PT shards
# ------------------------------------------------------------
shard_paths = sorted(
    TEST_FEATURE_DIR.glob("*.pt")
)

if not shard_paths:
    shard_paths = sorted(
        TEST_FEATURE_DIR.rglob("*.pt")
    )

if not shard_paths:
    raise FileNotFoundError(
        f"PT shard를 찾지 못했습니다:\n{TEST_FEATURE_DIR}"
    )

print("TEST PT shards:", len(shard_paths))
print("First shard:", shard_paths[0])
print("Last shard:", shard_paths[-1])

# ------------------------------------------------------------
# 7. Flexible shard parser
# ------------------------------------------------------------
PATENT_ID_KEYS = [
    "patent_id",
    "patent_ids",
    "publication_number",
    "publication_numbers",
    "doc_id",
    "document_id",
]

CLAIM_ID_KEYS = [
    "claim_id",
    "claim_ids",
    "claim_number",
    "claim_numbers",
    "claim_idx",
    "claim_index",
]

DEPTH_KEYS = [
    "depth",
    "depths",
    "claim_depth",
    "claim_depths",
]

MASK_KEYS = [
    "attention_mask",
    "attention_masks",
    "mask",
    "masks",
]

EMBEDDING_KEYS = [
    "token_embeddings",
    "token_embedding",
    "embeddings",
    "embedding",
    "token_features",
    "features",
    "hidden_states",
    "last_hidden_state",
]


def find_dict_value(data, candidate_keys):
    if not isinstance(data, dict):
        return None

    lower_to_original = {
        str(key).lower(): key
        for key in data.keys()
    }

    for candidate in candidate_keys:
        if candidate.lower() in lower_to_original:
            return data[
                lower_to_original[
                    candidate.lower()
                ]
            ]

    return None


def sequence_length(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.shape[0] if value.ndim > 0 else None

    if isinstance(value, np.ndarray):
        return value.shape[0] if value.ndim > 0 else None

    if isinstance(value, (list, tuple)):
        return len(value)

    return None


def sequence_item(value, index):
    if value is None:
        return None

    if torch.is_tensor(value):
        item = value[index]

        if item.numel() == 1:
            return item.item()

        return item

    if isinstance(value, np.ndarray):
        item = value[index]

        if np.asarray(item).size == 1:
            return np.asarray(item).reshape(-1)[0]

        return item

    if isinstance(value, (list, tuple)):
        return value[index]

    return value


def is_embedding_tensor(value):
    if not (
        torch.is_tensor(value)
        or isinstance(value, np.ndarray)
    ):
        return False

    shape = tuple(value.shape)

    if len(shape) < 1:
        return False

    return shape[-1] == HIDDEN_DIM


def find_embedding(data):
    if not isinstance(data, dict):
        return None, None

    lower_to_original = {
        str(key).lower(): key
        for key in data.keys()
    }

    for candidate in EMBEDDING_KEYS:
        candidate_lower = candidate.lower()

        if candidate_lower in lower_to_original:
            original_key = lower_to_original[
                candidate_lower
            ]

            value = data[original_key]

            if is_embedding_tensor(value):
                return value, original_key

    # Fallback based on key text.
    for key, value in data.items():
        key_lower = str(key).lower()

        if (
            any(
                marker in key_lower
                for marker in [
                    "embed",
                    "feature",
                    "hidden"
                ]
            )
            and is_embedding_tensor(value)
        ):
            return value, key

    return None, None


def recursive_claim_items(
    obj,
    inherited_patent_id=None,
    inherited_claim_id=None,
    inherited_depth=None
):
    """
    Yields:
      patent_id, claim_id, depth, embedding, attention_mask
    """

    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from recursive_claim_items(
                item,
                inherited_patent_id,
                inherited_claim_id,
                inherited_depth
            )
        return

    if not isinstance(obj, dict):
        return

    local_patent_id = find_dict_value(
        obj,
        PATENT_ID_KEYS
    )

    local_claim_id = find_dict_value(
        obj,
        CLAIM_ID_KEYS
    )

    local_depth = find_dict_value(
        obj,
        DEPTH_KEYS
    )

    attention_mask = find_dict_value(
        obj,
        MASK_KEYS
    )

    embedding, embedding_key = find_embedding(
        obj
    )

    context_patent_id = (
        local_patent_id
        if local_patent_id is not None
        else inherited_patent_id
    )

    context_claim_id = (
        local_claim_id
        if local_claim_id is not None
        else inherited_claim_id
    )

    context_depth = (
        local_depth
        if local_depth is not None
        else inherited_depth
    )

    if embedding is not None:
        shape = tuple(embedding.shape)

        # One already-pooled claim vector.
        if len(shape) == 1:
            yield (
                context_patent_id,
                context_claim_id,
                context_depth,
                embedding,
                attention_mask
            )
            return

        # 3D batch: [claims, tokens, hidden]
        if len(shape) == 3:
            batch_size = shape[0]

            for index in range(batch_size):
                patent_id = (
                    sequence_item(
                        context_patent_id,
                        index
                    )
                    if sequence_length(
                        context_patent_id
                    ) == batch_size
                    else context_patent_id
                )

                claim_id = (
                    sequence_item(
                        context_claim_id,
                        index
                    )
                    if sequence_length(
                        context_claim_id
                    ) == batch_size
                    else context_claim_id
                )

                depth = (
                    sequence_item(
                        context_depth,
                        index
                    )
                    if sequence_length(
                        context_depth
                    ) == batch_size
                    else context_depth
                )

                mask = (
                    sequence_item(
                        attention_mask,
                        index
                    )
                    if sequence_length(
                        attention_mask
                    ) == batch_size
                    else attention_mask
                )

                yield (
                    patent_id,
                    claim_id,
                    depth,
                    embedding[index],
                    mask
                )

            return

        # 2D can be either:
        # [tokens, hidden] for one claim, or
        # [claims, hidden] for a pre-pooled batch.
        if len(shape) == 2:
            first_dimension = shape[0]

            patent_sequence_length = sequence_length(
                context_patent_id
            )

            claim_sequence_length = sequence_length(
                context_claim_id
            )

            is_claim_batch = (
                patent_sequence_length == first_dimension
                or claim_sequence_length == first_dimension
            )

            if is_claim_batch:
                for index in range(first_dimension):
                    patent_id = (
                        sequence_item(
                            context_patent_id,
                            index
                        )
                        if patent_sequence_length
                        == first_dimension
                        else context_patent_id
                    )

                    claim_id = (
                        sequence_item(
                            context_claim_id,
                            index
                        )
                        if claim_sequence_length
                        == first_dimension
                        else context_claim_id
                    )

                    depth = (
                        sequence_item(
                            context_depth,
                            index
                        )
                        if sequence_length(
                            context_depth
                        ) == first_dimension
                        else context_depth
                    )

                    yield (
                        patent_id,
                        claim_id,
                        depth,
                        embedding[index],
                        None
                    )
            else:
                yield (
                    context_patent_id,
                    context_claim_id,
                    context_depth,
                    embedding,
                    attention_mask
                )

            return

    # Recurse through nested objects.
    for key, value in obj.items():
        if key == embedding_key:
            continue

        if isinstance(value, (dict, list, tuple)):
            yield from recursive_claim_items(
                value,
                context_patent_id,
                context_claim_id,
                context_depth
            )


# ------------------------------------------------------------
# 8. GPU mean pooling
# ------------------------------------------------------------
@torch.no_grad()
def gpu_mean_pool(embedding, attention_mask=None):
    if isinstance(embedding, np.ndarray):
        tensor = torch.from_numpy(embedding)
    else:
        tensor = embedding

    tensor = tensor.detach().to(
        device=DEVICE,
        dtype=torch.float32,
        non_blocking=True
    )

    if tensor.ndim == 1:
        pooled = tensor

    elif tensor.ndim == 2:
        if attention_mask is not None:
            if isinstance(attention_mask, np.ndarray):
                mask = torch.from_numpy(
                    attention_mask
                )
            elif torch.is_tensor(attention_mask):
                mask = attention_mask
            else:
                mask = torch.as_tensor(
                    attention_mask
                )

            mask = mask.detach().to(
                device=DEVICE,
                dtype=torch.float32,
                non_blocking=True
            ).reshape(-1)

            usable_length = min(
                tensor.shape[0],
                mask.shape[0]
            )

            tensor = tensor[:usable_length]
            mask = mask[:usable_length]

            denominator = mask.sum().clamp_min(
                1.0
            )

            pooled = (
                tensor
                * mask.unsqueeze(1)
            ).sum(dim=0) / denominator
        else:
            pooled = tensor.mean(dim=0)

    else:
        raise ValueError(
            f"Unexpected claim embedding shape: "
            f"{tuple(tensor.shape)}"
        )

    return (
        pooled.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ------------------------------------------------------------
# 9. Process one shard
# ------------------------------------------------------------
def process_one_shard(shard_path, partial_path):
    shard_start = time.time()

    local_path = (
        LOCAL_SHARD_DIR
        / shard_path.name
    )

    if local_path.exists():
        local_path.unlink()

    shutil.copy2(
        shard_path,
        local_path
    )

    source_size = shard_path.stat().st_size
    local_size = local_path.stat().st_size

    if source_size != local_size:
        local_path.unlink(missing_ok=True)

        raise IOError(
            f"Local shard copy size mismatch: "
            f"{shard_path.name}"
        )

    try:
        try:
            shard_data = torch.load(
                local_path,
                map_location="cpu",
                weights_only=False
            )
        except TypeError:
            shard_data = torch.load(
                local_path,
                map_location="cpu"
            )

        patent_sums = {}
        patent_weights = defaultdict(float)
        patent_claim_counts = defaultdict(int)

        parsed_claims = 0
        missing_patent_ids = 0
        missing_claim_ids = 0
        missing_depths = 0
        unknown_patents = 0

        for (
            patent_id,
            claim_id,
            shard_depth,
            embedding,
            attention_mask
        ) in recursive_claim_items(shard_data):

            parsed_claims += 1

            normalized_patent_id = normalize_id(
                patent_id
            )

            if normalized_patent_id is None:
                missing_patent_ids += 1
                continue

            global_index = patent_id_to_index.get(
                normalized_patent_id
            )

            if global_index is None:
                try:
                    alternative = str(
                        int(float(normalized_patent_id))
                    )

                    global_index = (
                        patent_id_to_index.get(
                            alternative
                        )
                    )
                except Exception:
                    pass

            if global_index is None:
                unknown_patents += 1
                continue

            claim_id_number = claim_number(
                claim_id
            )

            if claim_id_number is None:
                missing_claim_ids += 1

            depth = None

            if claim_id_number is not None:
                depth = depth_maps[
                    global_index
                ].get(claim_id_number)

            if depth is None and shard_depth is not None:
                try:
                    if torch.is_tensor(shard_depth):
                        if shard_depth.numel() == 1:
                            depth = int(
                                shard_depth.item()
                            )
                    else:
                        depth = int(shard_depth)
                except Exception:
                    depth = None

            if depth is None:
                missing_depths += 1

                # Conservative fallback:
                # claim 1 is treated as root,
                # otherwise dependent.
                depth = (
                    0
                    if claim_id_number == 1
                    else 1
                )

            pooled_claim = gpu_mean_pool(
                embedding,
                attention_mask
            )

            if pooled_claim.shape != (HIDDEN_DIM,):
                raise ValueError(
                    f"Unexpected pooled shape: "
                    f"{pooled_claim.shape}"
                )

            if not np.all(
                np.isfinite(pooled_claim)
            ):
                pooled_claim = np.nan_to_num(
                    pooled_claim,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                ).astype(np.float32)

            weight = (
                ROOT_WEIGHT
                if depth == 0
                else 1.0
            )

            weight *= np.exp(
                -DEPTH_DECAY * float(depth)
            )

            if global_index not in patent_sums:
                patent_sums[
                    global_index
                ] = np.zeros(
                    HIDDEN_DIM,
                    dtype=np.float64
                )

            patent_sums[global_index] += (
                float(weight)
                * pooled_claim.astype(
                    np.float64
                )
            )

            patent_weights[
                global_index
            ] += float(weight)

            patent_claim_counts[
                global_index
            ] += 1

        if parsed_claims == 0:
            raise RuntimeError(
                "No claims were parsed from shard: "
                f"{shard_path.name}"
            )

        indices = np.asarray(
            sorted(patent_sums.keys()),
            dtype=np.int64
        )

        sums = np.stack(
            [
                patent_sums[index]
                for index in indices
            ],
            axis=0
        ).astype(np.float32)

        weights = np.asarray(
            [
                patent_weights[index]
                for index in indices
            ],
            dtype=np.float32
        )

        counts = np.asarray(
            [
                patent_claim_counts[index]
                for index in indices
            ],
            dtype=np.int32
        )

        np.savez_compressed(
            partial_path,
            indices=indices,
            sums=sums,
            weights=weights,
            counts=counts,
            parsed_claims=np.asarray(
                [parsed_claims],
                dtype=np.int64
            ),
            missing_patent_ids=np.asarray(
                [missing_patent_ids],
                dtype=np.int64
            ),
            missing_claim_ids=np.asarray(
                [missing_claim_ids],
                dtype=np.int64
            ),
            missing_depths=np.asarray(
                [missing_depths],
                dtype=np.int64
            ),
            unknown_patents=np.asarray(
                [unknown_patents],
                dtype=np.int64
            ),
        )

        elapsed = time.time() - shard_start

        return {
            "parsed_claims": parsed_claims,
            "patents": len(indices),
            "missing_patent_ids": missing_patent_ids,
            "missing_claim_ids": missing_claim_ids,
            "missing_depths": missing_depths,
            "unknown_patents": unknown_patents,
            "elapsed": elapsed,
        }

    finally:
        if "shard_data" in locals():
            del shard_data

        gc.collect()
        torch.cuda.empty_cache()
        local_path.unlink(missing_ok=True)


# ------------------------------------------------------------
# 10. Resumable TEST pooling
# ------------------------------------------------------------
pooling_start = time.time()

if RAW_POOL_PATH.exists():
    print("\n" + "=" * 88)
    print("LOADING CACHED TEST RAW POOL")
    print("=" * 88)

    raw_test_pool = np.load(
        RAW_POOL_PATH
    ).astype(np.float32)

    if raw_test_pool.shape != (
        N_TEST,
        HIDDEN_DIM
    ):
        raise ValueError(
            f"Invalid cached raw pool shape: "
            f"{raw_test_pool.shape}"
        )

    print("Loaded:", RAW_POOL_PATH)
    print("Shape:", raw_test_pool.shape)

else:
    print("\n" + "=" * 88)
    print("RESUMABLE TEST CLAIM POOLING")
    print("=" * 88)

    for shard_index, shard_path in enumerate(
        shard_paths
    ):
        partial_path = (
            PARTIAL_DIR
            / f"{shard_path.stem}_root12_d010.npz"
        )

        if partial_path.exists():
            print(
                f"[{shard_index + 1:03d}/"
                f"{len(shard_paths):03d}] "
                f"Cached: {shard_path.name}"
            )
            continue

        try:
            stats = process_one_shard(
                shard_path,
                partial_path
            )

            print(
                f"[{shard_index + 1:03d}/"
                f"{len(shard_paths):03d}] "
                f"{shard_path.name} | "
                f"claims={stats['parsed_claims']:,} | "
                f"patents={stats['patents']:,} | "
                f"missing_depth={stats['missing_depths']:,} | "
                f"time={stats['elapsed']:.2f}s"
            )

        except Exception as error:
            print("\nERROR while processing:")
            print("Shard:", shard_path)
            print("Error:", repr(error))
            print(
                "완료된 shard partial cache는 보존됩니다. "
                "오류를 해결한 뒤 같은 셀을 다시 실행하면 이어집니다."
            )
            raise

    print("\nAssembling final TEST pool...")

    global_sums = np.zeros(
        (N_TEST, HIDDEN_DIM),
        dtype=np.float64
    )

    global_weights = np.zeros(
        N_TEST,
        dtype=np.float64
    )

    global_counts = np.zeros(
        N_TEST,
        dtype=np.int64
    )

    total_parsed_claims = 0
    total_missing_depths = 0
    total_unknown_patents = 0

    for shard_path in shard_paths:
        partial_path = (
            PARTIAL_DIR
            / f"{shard_path.stem}_root12_d010.npz"
        )

        if not partial_path.exists():
            raise FileNotFoundError(
                f"Missing partial cache: {partial_path}"
            )

        partial = np.load(
            partial_path
        )

        indices = partial[
            "indices"
        ].astype(np.int64)

        global_sums[indices] += partial[
            "sums"
        ].astype(np.float64)

        global_weights[indices] += partial[
            "weights"
        ].astype(np.float64)

        global_counts[indices] += partial[
            "counts"
        ].astype(np.int64)

        total_parsed_claims += int(
            partial["parsed_claims"][0]
        )

        total_missing_depths += int(
            partial["missing_depths"][0]
        )

        total_unknown_patents += int(
            partial["unknown_patents"][0]
        )

    missing_patent_indices = np.where(
        global_weights <= 0
    )[0]

    if len(missing_patent_indices) > 0:
        print(
            "Missing pooled patents:",
            len(missing_patent_indices)
        )
        print(
            "First missing indices:",
            missing_patent_indices[:20].tolist()
        )

        raise RuntimeError(
            "Some TEST patents have no pooled claims. "
            "TEST evaluation stopped."
        )

    raw_test_pool = (
        global_sums
        / global_weights[:, None]
    ).astype(np.float32)

    raw_test_pool = np.nan_to_num(
        raw_test_pool,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    np.save(
        RAW_POOL_PATH,
        raw_test_pool
    )

    print("\nPOOLING COMPLETE")
    print("Observed patents:", int(
        np.sum(global_counts > 0)
    ))
    print("Parsed claims:", total_parsed_claims)
    print("Expected claims:", expected_claim_count)
    print("Missing depths:", total_missing_depths)
    print("Unknown patents:", total_unknown_patents)
    print("Raw pool shape:", raw_test_pool.shape)
    print("Saved:", RAW_POOL_PATH)
    print(
        "Pooling runtime:",
        f"{(time.time() - pooling_start) / 60:.2f} min"
    )

# ------------------------------------------------------------
# 11. Apply DEV-fitted PCA
# ------------------------------------------------------------
print("\n" + "=" * 88)
print("APPLYING DEV-FITTED PCA")
print("=" * 88)

if PCA_FEATURE_PATH.exists():
    test_features = np.load(
        PCA_FEATURE_PATH
    ).astype(np.float32)

    if test_features.shape != (
        N_TEST,
        PCA_DIM
    ):
        raise ValueError(
            f"Invalid cached PCA shape: "
            f"{test_features.shape}"
        )

    print("Loaded cached TEST PCA features")

else:
    pca = joblib.load(
        DEV_PCA_MODEL_PATH
    )

    if not hasattr(pca, "transform"):
        raise TypeError(
            "Loaded PCA object has no transform method."
        )

    print("PCA input dimension:", raw_test_pool.shape[1])
    print("PCA model input dimension:", pca.n_features_in_)

    if raw_test_pool.shape[1] != pca.n_features_in_:
        raise ValueError(
            "TEST raw embedding dimension does not match "
            "the DEV-fitted PCA model."
        )

    test_features = pca.transform(
        raw_test_pool
    ).astype(np.float32)

    norms = np.linalg.norm(
        test_features,
        axis=1,
        keepdims=True
    )

    test_features = (
        test_features
        / np.maximum(norms, 1e-12)
    ).astype(np.float32)

    np.save(
        PCA_FEATURE_PATH,
        test_features
    )

    print("Saved TEST PCA features:", PCA_FEATURE_PATH)

print("TEST PCA feature shape:", test_features.shape)

del raw_test_pool
gc.collect()

# ------------------------------------------------------------
# 12. GPU spherical K-means
# ------------------------------------------------------------
@torch.no_grad()
def gpu_spherical_kmeans(
    features,
    n_clusters,
    seed,
    max_iter=100,
    tolerance=1e-5
):
    x = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    )

    x = torch.nn.functional.normalize(
        x,
        p=2,
        dim=1
    )

    n_samples, n_features = x.shape

    generator = torch.Generator(
        device=DEVICE
    )

    generator.manual_seed(
        int(seed)
    )

    first_index = int(
        torch.randint(
            0,
            n_samples,
            (1,),
            generator=generator,
            device=DEVICE
        ).item()
    )

    centroid_indices = [
        first_index
    ]

    closest_distance = (
        1.0
        - torch.matmul(
            x,
            x[first_index].unsqueeze(1)
        ).squeeze(1)
    )

    # GPU K-means++ initialization
    for _ in range(1, n_clusters):
        probabilities = torch.clamp(
            closest_distance,
            min=1e-8
        )

        probabilities = (
            probabilities
            / probabilities.sum()
        )

        next_index = int(
            torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator
            ).item()
        )

        centroid_indices.append(
            next_index
        )

        new_distance = (
            1.0
            - torch.matmul(
                x,
                x[next_index].unsqueeze(1)
            ).squeeze(1)
        )

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
    iteration = 0

    for iteration in range(max_iter):
        similarities = torch.matmul(
            x,
            centroids.T
        )

        best_similarity, labels = (
            similarities.max(dim=1)
        )

        new_centroids = torch.zeros(
            (n_clusters, n_features),
            dtype=torch.float32,
            device=DEVICE
        )

        new_centroids.index_add_(
            0,
            labels,
            x
        )

        counts = torch.bincount(
            labels,
            minlength=n_clusters
        ).float()

        empty_clusters = torch.where(
            counts == 0
        )[0]

        if len(empty_clusters) > 0:
            difficult_points = torch.argsort(
                best_similarity
            )[:len(empty_clusters)]

            new_centroids[
                empty_clusters
            ] = x[difficult_points]

            counts[
                empty_clusters
            ] = 1.0

        new_centroids = (
            new_centroids
            / counts.unsqueeze(1)
        )

        new_centroids = (
            torch.nn.functional.normalize(
                new_centroids,
                p=2,
                dim=1
            )
        )

        objective = float(
            best_similarity.mean().item()
        )

        centroids = new_centroids

        if previous_objective is not None:
            if (
                abs(
                    objective
                    - previous_objective
                )
                < tolerance
            ):
                break

        previous_objective = objective

    final_similarities = torch.matmul(
        x,
        centroids.T
    )

    predictions = (
        final_similarities.argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    final_objective = float(
        final_similarities.max(dim=1)
        .values.mean()
        .item()
    )

    del x
    del centroids
    del final_similarities

    torch.cuda.empty_cache()

    return (
        predictions,
        final_objective,
        iteration + 1
    )

# ------------------------------------------------------------
# 13. TEST CPC labels
# ------------------------------------------------------------
sections = []
classes = []
subclasses = []

for record in records:
    section, class_label, subclass = (
        extract_cpc_labels(record)
    )

    sections.append(section)
    classes.append(class_label)
    subclasses.append(subclass)

section_labels, section_map = encode_labels(
    sections
)

class_labels, class_map = encode_labels(
    classes
)

subclass_labels, subclass_map = encode_labels(
    subclasses
)

label_sets = {
    "section": section_labels,
    "class": class_labels,
    "subclass": subclass_labels,
}

cardinalities = {
    "section": len(section_map),
    "class": len(class_map),
    "subclass": len(subclass_map),
}

print("\n" + "=" * 88)
print("TEST LABEL CARDINALITIES")
print("=" * 88)
print(cardinalities)

# ------------------------------------------------------------
# 14. Fixed 3-seed TEST evaluation
# ------------------------------------------------------------
print("\n" + "=" * 88)
print("FIXED 3-SEED TEST CLUSTERING")
print("=" * 88)

evaluation_start = time.time()

seed_predictions = {}
seed_metrics = {}
metric_rows = []

for seed in KMEANS_SEEDS:
    run_start = time.time()

    (
        predictions,
        objective,
        iterations
    ) = gpu_spherical_kmeans(
        test_features,
        n_clusters=N_CLUSTERS,
        seed=seed,
        max_iter=KMEANS_MAX_ITER,
        tolerance=KMEANS_TOL
    )

    metrics = evaluate_predictions(
        predictions,
        label_sets
    )

    seed_predictions[seed] = predictions
    seed_metrics[seed] = metrics

    row = {
        "seed": seed,
        "objective": objective,
        "iterations": iterations,
        **metrics,
    }

    metric_rows.append(row)

    print(
        f"Seed {seed} | "
        f"NMI={metrics['mean_nmi']:.6f} | "
        f"Pur_p={metrics['mean_pur_p']:.6f} | "
        f"Pur_a={metrics['mean_pur_a']:.6f} | "
        f"Section={metrics['section_nmi']:.6f} | "
        f"Class={metrics['class_nmi']:.6f} | "
        f"Subclass={metrics['subclass_nmi']:.6f} | "
        f"time={time.time() - run_start:.2f}s"
    )

aggregate = aggregate_metrics(
    seed_metrics
)

# ------------------------------------------------------------
# 15. Save outputs
# ------------------------------------------------------------
np.savez_compressed(
    PREDICTIONS_PATH,
    seed_17=seed_predictions[17],
    seed_42=seed_predictions[42],
    seed_73=seed_predictions[73],
)

with open(
    ASSIGNMENTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "global_index",
        "patent_id",
        "section",
        "class",
        "subclass",
        "seed17_topic",
        "seed42_topic",
        "seed73_topic",
    ])

    for index in range(N_TEST):
        writer.writerow([
            index,
            patent_ids[index],
            sections[index],
            classes[index],
            subclasses[index],
            int(seed_predictions[17][index]),
            int(seed_predictions[42][index]),
            int(seed_predictions[73][index]),
        ])

csv_fieldnames = sorted(
    {
        key
        for row in metric_rows
        for key in row.keys()
    }
)

with open(
    RESULTS_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=csv_fieldnames
    )

    writer.writeheader()

    for row in metric_rows:
        writer.writerow(row)

total_runtime_minutes = (
    time.time() - evaluation_start
) / 60.0

result_summary = {
    "experiment": (
        "final_fixed_root12_d010_claimwise_test_evaluation"
    ),

    "split": "TEST",

    "configuration_frozen_before_test": True,

    "configuration": {
        "feature": "claim_only",
        "pooling": "root12_d010",
        "root_weight": ROOT_WEIGHT,
        "depth_decay": DEPTH_DECAY,
        "pca_dimension": PCA_DIM,
        "pca_fitted_on": "DEV",
        "n_clusters": N_CLUSTERS,
        "clustering": "spherical_kmeans",
        "seeds": KMEANS_SEEDS,
        "balancing": False,
        "theta_used": False,
        "document_semantic_used": False,
        "lexical_used": False,
    },

    "data": {
        "n_test_patents": N_TEST,
        "expected_claims": expected_claim_count,
        "cardinalities": cardinalities,
    },

    "per_seed": seed_metrics,
    "aggregate": aggregate,

    "paths": {
        "test_records": str(TEST_RECORDS_PATH),
        "test_features": str(TEST_FEATURE_DIR),
        "dev_pca_model": str(DEV_PCA_MODEL_PATH),
        "raw_test_pool": str(RAW_POOL_PATH),
        "test_pca_features": str(PCA_FEATURE_PATH),
        "predictions": str(PREDICTIONS_PATH),
        "assignments": str(ASSIGNMENTS_PATH),
        "metrics_csv": str(RESULTS_CSV_PATH),
        "summary_json": str(RESULTS_PATH),
    },

    "runtime_minutes": total_runtime_minutes,
}

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        json_safe(result_summary),
        file,
        indent=2,
        ensure_ascii=False
    )

# ------------------------------------------------------------
# 16. Final output
# ------------------------------------------------------------
def print_mean_std(name):
    mean_value = aggregate[
        f"{name}_mean"
    ]

    std_value = aggregate[
        f"{name}_std"
    ]

    print(
        f"{name}: "
        f"{mean_value:.6f} ± {std_value:.6f}"
    )


print("\n\n" + "=" * 88)
print("FINAL FIXED TEST RESULTS")
print("=" * 88)
print("Method: root12_d010 claim-only")
print("Configuration frozen before TEST: YES")
print("DEV-fitted PCA used: YES")
print("TEST hyperparameter tuning: NO")
print("TEST patents:", N_TEST)
print("CPC cardinalities:", cardinalities)

print("\nCPC NMI")
print_mean_std("section_nmi")
print_mean_std("class_nmi")
print_mean_std("subclass_nmi")
print_mean_std("mean_nmi")

print("\nCPC PUR_P")
print_mean_std("section_pur_p")
print_mean_std("class_pur_p")
print_mean_std("subclass_pur_p")
print_mean_std("mean_pur_p")

print("\nCPC PUR_A")
print_mean_std("section_pur_a")
print_mean_std("class_pur_a")
print_mean_std("subclass_pur_a")
print_mean_std("mean_pur_a")

print("\nCLUSTER BALANCE")
print_mean_std("active_topics")
print_mean_std("max_topic_share")

print("\n" + "=" * 88)
print("OUTPUT FILES")
print("=" * 88)
print("Raw TEST pool:", RAW_POOL_PATH)
print("TEST PCA features:", PCA_FEATURE_PATH)
print("Predictions:", PREDICTIONS_PATH)
print("Assignments:", ASSIGNMENTS_PATH)
print("Per-seed metrics:", RESULTS_CSV_PATH)
print("Summary:", RESULTS_PATH)
print(
    f"Evaluation runtime: "
    f"{total_runtime_minutes:.2f} minutes"
)

print("\nFINAL TEST EVALUATION COMPLETE.")
