# ==================================================================================================
# DEPTH-OT V2 — T4-OPTIMIZED DETERMINISTIC DEV CPC EVALUATION
#
# 평가 대상:
#   1. best_pre_hierarchy.pt (stored epoch 4)
#   2. epoch_015.pt
#   3. epoch_016.pt
#
# 입력 split:
#   DEV — 9,855 patents / 99 shards
#
# 이 셀은 학습을 수행하지 않습니다.
# ==================================================================================================


# ==================================================================================================
# 0. USER SETTINGS
# ==================================================================================================

CHECKPOINTS = {
    "best_pre_hierarchy_epoch004": (
        "/content/drive/MyDrive/depth_ot_patent/checkpoints/depth_ot_v2/"
        "depth_ot_v2_patent_semantic_seed42_20260814_055110/"
        "best_pre_hierarchy.pt"
    ),
    "epoch_015": (
        "/content/drive/MyDrive/depth_ot_patent/checkpoints/depth_ot_v2/"
        "depth_ot_v2_patent_semantic_seed42_20260814_055110/"
        "epoch_015.pt"
    ),
    "epoch_016": (
        "/content/drive/MyDrive/depth_ot_patent/checkpoints/depth_ot_v2/"
        "depth_ot_v2_patent_semantic_seed42_20260814_055110/"
        "epoch_016.pt"
    ),
}

EVALUATION_SPLIT = "dev"

RECORDS_PATH = (
    "/content/drive/MyDrive/depth_ot_patent/"
    "data/processed/dev_records.pkl"
)

FEATURE_DIR = (
    "/content/drive/MyDrive/depth_ot_patent/"
    "data/processed/token_features/full/dev"
)

RESULT_ROOT = (
    "/content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/"
    "depth_ot_v2_patent_semantic_seed42_20260814_055110/"
    "fast_t4_checkpoint_evaluation_dev"
)

EXPECTED_NUMBER_OF_PATENTS = 9_855
EXPECTED_NUMBER_OF_SHARDS = 99
PATENTS_PER_SHARD = 100

# T4 평가 설정
USE_FP16 = False
USE_LOCAL_SHARD_PREFETCH = True
RESUME_EVALUATION = True
REMOVE_LOCAL_CACHE_WHEN_DONE = True

LOCAL_SHARD_CACHE_DIR = (
    "/content/depth_ot_v2_dev_eval_shard_cache"
)

MAX_PATENTS_PER_EVAL_BATCH = 24
MAX_TOKENS_PER_EVAL_BATCH = 49_152
MAX_TOKEN_EDGES_PER_EVAL_BATCH = 850_000

# 몇 개 shard마다 진행 상황을 저장할지 설정
SAVE_PROGRESS_EVERY_SHARDS = 5

# 논문 표의 primary Pur_a
PRIMARY_PUR_A_MODE = "weighted"


# ==================================================================================================
# 1. ENVIRONMENT
# ==================================================================================================

from google.colab import drive

drive.mount(
    "/content/drive",
    force_remount=False,
)

import os
import gc
import re
import csv
import json
import time
import pickle
import shutil
import random
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix

except ImportError:
    import sys
    import subprocess

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "scikit-learn",
    ])

    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 필요합니다. "
        "Colab 런타임을 T4 GPU로 설정하세요."
    )

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GIB = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

EVALUATION_SPLIT = (
    str(EVALUATION_SPLIT)
    .strip()
    .lower()
)

if EVALUATION_SPLIT not in {
    "train",
    "dev",
    "validation",
    "valid",
    "val",
    "test",
}:
    raise ValueError(
        f"Unsupported EVALUATION_SPLIT: "
        f"{EVALUATION_SPLIT}"
    )

RECORDS_PATH = Path(RECORDS_PATH)
FEATURE_DIR = Path(FEATURE_DIR)
RESULT_ROOT = Path(RESULT_ROOT)
LOCAL_SHARD_CACHE_DIR = Path(
    LOCAL_SHARD_CACHE_DIR
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

if not RECORDS_PATH.is_file():
    raise FileNotFoundError(
        f"Records file not found: {RECORDS_PATH}"
    )

if not FEATURE_DIR.is_dir():
    raise FileNotFoundError(
        f"Feature directory not found: {FEATURE_DIR}"
    )

for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"{checkpoint_name}: {checkpoint_path}"
        )


print("=" * 100)
print(
    "DEPTH-OT V2 — T4-OPTIMIZED "
    f"{EVALUATION_SPLIT.upper()} CPC EVALUATION"
)
print("=" * 100)
print(f"Split               : {EVALUATION_SPLIT}")
print(f"GPU                 : {GPU_NAME}")
print(f"GPU memory          : {GPU_MEMORY_GIB:.2f} GiB")
print(f"PyTorch             : {torch.__version__}")
print(f"FP16                : {USE_FP16}")
print(f"Local prefetch      : {USE_LOCAL_SHARD_PREFETCH}")
print(f"Max patents/batch   : {MAX_PATENTS_PER_EVAL_BATCH}")
print(f"Max tokens/batch    : {MAX_TOKENS_PER_EVAL_BATCH:,}")
print(f"Max token edges     : {MAX_TOKEN_EDGES_PER_EVAL_BATCH:,}")
print(f"Resume              : {RESUME_EVALUATION}")
print(f"Records             : {RECORDS_PATH}")
print(f"Features            : {FEATURE_DIR}")
print(f"Results             : {RESULT_ROOT}")
print("=" * 100)

if "T4" not in GPU_NAME.upper():
    print(
        f"[WARNING] 현재 GPU는 T4가 아닙니다: "
        f"{GPU_NAME}"
    )

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

try:
    torch.set_float32_matmul_precision(
        "highest"
    )
except Exception:
    pass

torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ==================================================================================================
# 2. UTILITIES
# ==================================================================================================

MODEL_EPSILON = 1.0e-8


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_name(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    ).strip("_")


def atomic_json_save(data, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
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
        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temporary,
        path,
    )


def atomic_numpy_save(array, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary,
        "wb",
    ) as file:
        np.save(
            file,
            array,
        )
        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temporary,
        path,
    )


def normalize_integer_id(value):
    if isinstance(
        value,
        (int, np.integer),
    ):
        return int(value)

    text = str(value).strip()

    try:
        return int(text)
    except Exception:
        return text


def normalize_claims(record):
    claims = record.get(
        "claims",
        {},
    )

    if not isinstance(claims, dict):
        raise TypeError(
            "record['claims'] must be a dictionary."
        )

    return {
        normalize_integer_id(claim_id): str(text)
        for claim_id, text in claims.items()
    }


def normalize_edges(record):
    edges = record.get(
        "edges",
        [],
    )

    if edges is None:
        return []

    normalized = []

    for edge in edges:
        if (
            not isinstance(edge, (list, tuple))
            or len(edge) != 2
        ):
            raise ValueError(
                f"Invalid claim edge: {edge}"
            )

        normalized.append((
            normalize_integer_id(edge[0]),
            normalize_integer_id(edge[1]),
        ))

    return normalized


def sortable_id(value):
    if isinstance(value, int):
        return (0, value)

    return (1, str(value))


def clear_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================================================================================================
# 3. INFERENCE-ONLY MODEL
# ==================================================================================================

def segment_mean(
    values,
    segment_index,
    number_of_segments,
):
    segment_index = (
        segment_index.long().reshape(-1)
    )

    sums = values.new_zeros(
        (
            number_of_segments,
            values.shape[1],
        )
    )
    counts = values.new_zeros(
        (
            number_of_segments,
            1,
        )
    )

    sums.index_add_(
        0,
        segment_index,
        values,
    )

    counts.index_add_(
        0,
        segment_index,
        values.new_ones(
            (
                values.shape[0],
                1,
            )
        ),
    )

    return sums / counts.clamp_min(1.0)


class WeightedGraphConvolution(nn.Module):

    def __init__(
        self,
        input_dimension,
        output_dimension,
        dropout,
    ):
        super().__init__()

        self.self_projection = nn.Linear(
            input_dimension,
            output_dimension,
            bias=True,
        )

        self.neighbor_projection = nn.Linear(
            input_dimension,
            output_dimension,
            bias=False,
        )

        self.layer_norm = nn.LayerNorm(
            output_dimension
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        node_features,
        edge_index,
        edge_weight,
    ):
        number_of_nodes = int(
            node_features.shape[0]
        )

        if edge_index.shape[1] == 0:
            output = self.self_projection(
                node_features
            )
            output = self.layer_norm(
                output
            )
            output = F.gelu(
                output
            )
            return self.dropout(
                output
            )

        source = edge_index[0].long()
        target = edge_index[1].long()

        if edge_weight is None:
            edge_weight = node_features.new_ones(
                source.numel()
            )
        else:
            edge_weight = edge_weight.to(
                device=node_features.device,
                dtype=node_features.dtype,
            ).reshape(-1)

        edge_weight = edge_weight.clamp_min(
            0.0
        )

        degree = node_features.new_zeros(
            number_of_nodes
        )

        degree.index_add_(
            0,
            target,
            edge_weight,
        )

        degree = degree.clamp_min(
            MODEL_EPSILON
        )

        normalization = (
            edge_weight
            * torch.rsqrt(
                degree[source]
                * degree[target]
            )
        )

        projected_neighbors = (
            self.neighbor_projection(
                node_features
            )
        )

        messages = (
            projected_neighbors[source]
            * normalization.unsqueeze(-1)
        )

        aggregated = node_features.new_zeros(
            (
                number_of_nodes,
                self.self_projection.out_features,
            )
        )

        aggregated.index_add_(
            0,
            target,
            messages,
        )

        output = (
            self.self_projection(
                node_features
            )
            + aggregated
        )

        output = self.layer_norm(
            output
        )
        output = F.gelu(
            output
        )
        output = self.dropout(
            output
        )

        return output


class TokenGraphEncoder(nn.Module):

    def __init__(
        self,
        input_dimension,
        hidden_dimension,
        dropout,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.LayerNorm(
                hidden_dimension
            ),
            nn.GELU(),
        )

        self.graph_layer_1 = (
            WeightedGraphConvolution(
                hidden_dimension,
                hidden_dimension,
                dropout,
            )
        )

        self.graph_layer_2 = (
            WeightedGraphConvolution(
                hidden_dimension,
                hidden_dimension,
                dropout,
            )
        )

        self.output_norm = nn.LayerNorm(
            hidden_dimension
        )

    def forward(
        self,
        token_embeddings,
        token_edge_index,
        token_edge_weight,
        token_to_claim,
        number_of_claims,
    ):
        token_embeddings = (
            token_embeddings.float()
        )

        hidden = self.input_projection(
            token_embeddings
        )

        residual = hidden

        hidden = self.graph_layer_1(
            hidden,
            token_edge_index,
            token_edge_weight,
        )

        hidden = hidden + residual
        residual = hidden

        hidden = self.graph_layer_2(
            hidden,
            token_edge_index,
            token_edge_weight,
        )

        hidden = self.output_norm(
            hidden + residual
        )

        claim_representations = segment_mean(
            hidden,
            token_to_claim,
            number_of_claims,
        )

        return claim_representations


class BidirectionalClaimDependencyEncoder(
    nn.Module
):

    def __init__(
        self,
        hidden_dimension,
        output_dimension,
        dropout,
    ):
        super().__init__()

        self.context_network = nn.Sequential(
            nn.Linear(
                hidden_dimension * 3,
                output_dimension,
            ),
            nn.LayerNorm(
                output_dimension
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                output_dimension,
                output_dimension,
            ),
            nn.LayerNorm(
                output_dimension
            ),
            nn.GELU(),
        )

        self.residual_projection = (
            nn.Identity()
            if hidden_dimension == output_dimension
            else nn.Linear(
                hidden_dimension,
                output_dimension,
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        claim_representations,
        claim_edge_index,
    ):
        number_of_claims = int(
            claim_representations.shape[0]
        )
        hidden_dimension = int(
            claim_representations.shape[1]
        )

        parent_context = (
            claim_representations.new_zeros(
                (
                    number_of_claims,
                    hidden_dimension,
                )
            )
        )

        child_context = (
            claim_representations.new_zeros(
                (
                    number_of_claims,
                    hidden_dimension,
                )
            )
        )

        parent_counts = (
            claim_representations.new_zeros(
                (
                    number_of_claims,
                    1,
                )
            )
        )

        child_counts = (
            claim_representations.new_zeros(
                (
                    number_of_claims,
                    1,
                )
            )
        )

        if (
            claim_edge_index is not None
            and claim_edge_index.numel() > 0
        ):
            parent = (
                claim_edge_index[0].long()
            )
            child = (
                claim_edge_index[1].long()
            )

            parent_context.index_add_(
                0,
                child,
                claim_representations[parent],
            )

            parent_counts.index_add_(
                0,
                child,
                claim_representations.new_ones(
                    (
                        parent.numel(),
                        1,
                    )
                ),
            )

            child_context.index_add_(
                0,
                parent,
                claim_representations[child],
            )

            child_counts.index_add_(
                0,
                parent,
                claim_representations.new_ones(
                    (
                        child.numel(),
                        1,
                    )
                ),
            )

        parent_context = (
            parent_context
            / parent_counts.clamp_min(1.0)
        )

        child_context = (
            child_context
            / child_counts.clamp_min(1.0)
        )

        combined = torch.cat(
            [
                claim_representations,
                parent_context,
                child_context,
            ],
            dim=-1,
        )

        contextualized = (
            self.context_network(
                combined
            )
        )

        contextualized = (
            contextualized
            + self.residual_projection(
                claim_representations
            )
        )

        return self.dropout(
            contextualized
        )


class VariationalTopicEncoder(nn.Module):

    def __init__(
        self,
        input_dimension,
        number_of_topics,
        dropout,
    ):
        super().__init__()

        self.hidden_network = nn.Sequential(
            nn.Linear(
                input_dimension,
                input_dimension,
            ),
            nn.LayerNorm(
                input_dimension
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.mean_projection = nn.Linear(
            input_dimension,
            number_of_topics,
        )

        self.log_variance_projection = nn.Linear(
            input_dimension,
            number_of_topics,
        )

        self.posterior_norm = nn.LayerNorm(
            number_of_topics
        )

    def deterministic_theta(
        self,
        claim_representations,
    ):
        hidden = self.hidden_network(
            claim_representations
        )

        posterior_mean = (
            self.mean_projection(
                hidden
            )
        )

        normalized_latent = (
            self.posterior_norm(
                posterior_mean
            )
        )

        return F.softmax(
            normalized_latent,
            dim=-1,
        )


class FactorizedTopicWordDecoder(nn.Module):

    def __init__(
        self,
        number_of_topics,
        vocabulary_size,
        embedding_dimension,
    ):
        super().__init__()

        self.topic_embeddings = nn.Parameter(
            torch.empty(
                number_of_topics,
                embedding_dimension,
            )
        )

        self.word_embeddings = nn.Parameter(
            torch.empty(
                vocabulary_size,
                embedding_dimension,
            )
        )

        self.word_bias = nn.Parameter(
            torch.empty(
                vocabulary_size
            )
        )

        self.log_temperature = nn.Parameter(
            torch.empty(())
        )


class IndependentDepthAnchors(nn.Module):

    def __init__(
        self,
        number_of_topics,
    ):
        super().__init__()

        self.raw_coordinates = nn.Parameter(
            torch.empty(
                number_of_topics
            )
        )


class DepthOTV2InferenceModel(nn.Module):

    def __init__(
        self,
        plm_hidden_dimension,
        graph_hidden_dimension,
        claim_hidden_dimension,
        number_of_topics,
        vocabulary_size,
        topic_embedding_dimension,
    ):
        super().__init__()

        self.number_of_topics = int(
            number_of_topics
        )

        self.token_graph_encoder = (
            TokenGraphEncoder(
                plm_hidden_dimension,
                graph_hidden_dimension,
                dropout=0.10,
            )
        )

        self.claim_dependency_encoder = (
            BidirectionalClaimDependencyEncoder(
                graph_hidden_dimension,
                claim_hidden_dimension,
                dropout=0.10,
            )
        )

        self.variational_topic_encoder = (
            VariationalTopicEncoder(
                claim_hidden_dimension,
                number_of_topics,
                dropout=0.05,
            )
        )

        # strict state_dict load를 위해 유지
        self.topic_word_decoder = (
            FactorizedTopicWordDecoder(
                number_of_topics,
                vocabulary_size,
                topic_embedding_dimension,
            )
        )

        self.depth_anchors = (
            IndependentDepthAnchors(
                number_of_topics
            )
        )

    def forward_theta(
        self,
        batch,
    ):
        claim_to_patent = batch[
            "claim_to_patent"
        ]

        number_of_claims = int(
            claim_to_patent.numel()
        )

        number_of_patents = int(
            batch["num_patents"]
        )

        claim_representations = (
            self.token_graph_encoder(
                token_embeddings=(
                    batch["token_embeddings"]
                ),
                token_edge_index=(
                    batch["token_edge_index"]
                ),
                token_edge_weight=(
                    batch["token_edge_weight"]
                ),
                token_to_claim=(
                    batch["token_to_claim"]
                ),
                number_of_claims=(
                    number_of_claims
                ),
            )
        )

        claim_representations = (
            self.claim_dependency_encoder(
                claim_representations,
                batch["claim_edge_index"],
            )
        )

        claim_theta = (
            self.variational_topic_encoder
            .deterministic_theta(
                claim_representations
            )
        )

        patent_sums = claim_theta.new_zeros(
            (
                number_of_patents,
                self.number_of_topics,
            )
        )

        patent_counts = claim_theta.new_zeros(
            (
                number_of_patents,
                1,
            )
        )

        patent_sums.index_add_(
            0,
            claim_to_patent,
            claim_theta,
        )

        patent_counts.index_add_(
            0,
            claim_to_patent,
            claim_theta.new_ones(
                (
                    claim_theta.shape[0],
                    1,
                )
            ),
        )

        patent_theta = (
            patent_sums
            / patent_counts.clamp_min(1.0)
        )

        patent_theta = (
            patent_theta.clamp_min(
                MODEL_EPSILON
            )
        )

        patent_theta = (
            patent_theta
            / patent_theta.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                MODEL_EPSILON
            )
        )

        return patent_theta


def infer_architecture_from_state_dict(
    state_dict,
):
    input_projection_weight = state_dict[
        "token_graph_encoder."
        "input_projection.0.weight"
    ]

    mean_projection_weight = state_dict[
        "variational_topic_encoder."
        "mean_projection.weight"
    ]

    word_embeddings = state_dict[
        "topic_word_decoder.word_embeddings"
    ]

    topic_embeddings = state_dict[
        "topic_word_decoder.topic_embeddings"
    ]

    return {
        "plm_hidden_dimension": int(
            input_projection_weight.shape[1]
        ),
        "graph_hidden_dimension": int(
            input_projection_weight.shape[0]
        ),
        "claim_hidden_dimension": int(
            mean_projection_weight.shape[1]
        ),
        "number_of_topics": int(
            mean_projection_weight.shape[0]
        ),
        "vocabulary_size": int(
            word_embeddings.shape[0]
        ),
        "topic_embedding_dimension": int(
            topic_embeddings.shape[1]
        ),
    }


# ==================================================================================================
# 4. LOAD CHECKPOINTS
# ==================================================================================================

models = {}
checkpoint_metadata = {}

for (
    checkpoint_name,
    checkpoint_path_string,
) in CHECKPOINTS.items():

    checkpoint_name = safe_name(
        checkpoint_name
    )
    checkpoint_path = Path(
        checkpoint_path_string
    )

    print("\n" + "-" * 100)
    print(f"[LOAD] {checkpoint_name}")
    print(f"Path: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"model_state_dict not found: "
            f"{checkpoint_path}"
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    architecture = (
        infer_architecture_from_state_dict(
            state_dict
        )
    )

    model = DepthOTV2InferenceModel(
        **architecture
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            f"Checkpoint mismatch: "
            f"{incompatible}"
        )

    model = model.to(
        DEVICE
    )
    model.eval()

    models[checkpoint_name] = model

    checkpoint_metadata[
        checkpoint_name
    ] = {
        "path": str(
            checkpoint_path
        ),
        "size_mib": (
            checkpoint_path.stat().st_size
            / 1024**2
        ),
        "epoch": checkpoint.get(
            "epoch",
            checkpoint.get(
                "epoch_one_based"
            ),
        ),
        "run_name": checkpoint.get(
            "run_name"
        ),
        "model_version": checkpoint.get(
            "model_version"
        ),
        "saved_at_utc": checkpoint.get(
            "saved_at_utc"
        ),
        "architecture": architecture,
    }

    print(
        f"Epoch             : "
        f"{checkpoint_metadata[checkpoint_name]['epoch']}"
    )
    print(
        f"Architecture      : "
        f"{architecture}"
    )
    print("Strict load       : PASS")

    del checkpoint
    del state_dict
    gc.collect()

if not models:
    raise RuntimeError(
        "평가할 체크포인트가 없습니다."
    )

architecture_values = [
    metadata["architecture"]
    for metadata
    in checkpoint_metadata.values()
]

first_architecture = (
    architecture_values[0]
)

for architecture in architecture_values[1:]:
    if architecture != first_architecture:
        raise RuntimeError(
            "체크포인트 architecture가 "
            "서로 다릅니다."
        )

NUM_TOPICS = int(
    first_architecture[
        "number_of_topics"
    ]
)

clear_cuda()


# ==================================================================================================
# 5. LOAD DEV RECORDS
# ==================================================================================================

print(
    f"\n[LOAD {EVALUATION_SPLIT.upper()} RECORDS]"
)

with open(
    RECORDS_PATH,
    "rb",
) as file:
    evaluation_records = pickle.load(
        file
    )

if not isinstance(
    evaluation_records,
    list,
):
    if hasattr(
        evaluation_records,
        "to_dict",
    ):
        evaluation_records = (
            evaluation_records.to_dict(
                "records"
            )
        )
    else:
        raise TypeError(
            "Unsupported records container: "
            f"{type(evaluation_records)}"
        )

NUMBER_OF_PATENTS = len(
    evaluation_records
)

if (
    NUMBER_OF_PATENTS
    != EXPECTED_NUMBER_OF_PATENTS
):
    raise RuntimeError(
        "Unexpected number of records.\n"
        f"Split    : {EVALUATION_SPLIT}\n"
        f"Expected : "
        f"{EXPECTED_NUMBER_OF_PATENTS:,}\n"
        f"Found    : "
        f"{NUMBER_OF_PATENTS:,}"
    )

record_by_patent_id = {}
record_index_by_patent_id = {}

for (
    record_index,
    record,
) in enumerate(
    evaluation_records
):
    patent_id = normalize_integer_id(
        record["patent_id"]
    )

    if patent_id in record_by_patent_id:
        raise ValueError(
            f"Duplicate patent ID: "
            f"{patent_id}"
        )

    record_by_patent_id[
        patent_id
    ] = record

    record_index_by_patent_id[
        patent_id
    ] = record_index

print(
    f"Evaluation split    : "
    f"{EVALUATION_SPLIT}"
)
print(
    f"Evaluation patents  : "
    f"{NUMBER_OF_PATENTS:,}"
)


# ==================================================================================================
# 6. DISCOVER DEV SHARDS
# ==================================================================================================

evaluation_shards = sorted(
    FEATURE_DIR.glob(
        "shard_*.pt"
    )
)

if (
    len(evaluation_shards)
    != EXPECTED_NUMBER_OF_SHARDS
):
    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_NUMBER_OF_SHARDS} "
        f"{EVALUATION_SPLIT} shards, "
        f"found {len(evaluation_shards)} "
        f"in {FEATURE_DIR}"
    )

expected_names = [
    f"shard_{index:05d}.pt"
    for index in range(
        len(evaluation_shards)
    )
]

observed_names = [
    path.name
    for path in evaluation_shards
]

if observed_names != expected_names:
    raise RuntimeError(
        "Shard names are not contiguous."
    )

total_shard_bytes = sum(
    path.stat().st_size
    for path in evaluation_shards
)

print(
    f"Feature directory   : "
    f"{FEATURE_DIR}"
)
print(
    f"Feature shards      : "
    f"{len(evaluation_shards):,}"
)
print(
    f"Feature shard size  : "
    f"{total_shard_bytes / 1024**3:.3f} GiB"
)


# ==================================================================================================
# 7. RESUME ARRAYS
# ==================================================================================================

theta_arrays = {}
seen_arrays = {}
result_directories = {}

for checkpoint_name in models:
    result_directory = (
        RESULT_ROOT
        / checkpoint_name
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_directories[
        checkpoint_name
    ] = result_directory

    theta_progress_path = (
        result_directory
        / f"{EVALUATION_SPLIT}_theta_progress.npy"
    )

    seen_progress_path = (
        result_directory
        / f"{EVALUATION_SPLIT}_seen_progress.npy"
    )

    if (
        RESUME_EVALUATION
        and theta_progress_path.is_file()
        and seen_progress_path.is_file()
    ):
        theta = np.load(
            theta_progress_path
        )
        seen = np.load(
            seen_progress_path
        )

        expected_theta_shape = (
            NUMBER_OF_PATENTS,
            NUM_TOPICS,
        )

        if (
            theta.shape
            != expected_theta_shape
        ):
            raise RuntimeError(
                f"{checkpoint_name} theta "
                f"shape mismatch: "
                f"{theta.shape} != "
                f"{expected_theta_shape}"
            )

        if seen.shape != (
            NUMBER_OF_PATENTS,
        ):
            raise RuntimeError(
                f"{checkpoint_name} seen "
                f"shape mismatch: "
                f"{seen.shape}"
            )

        theta_arrays[
            checkpoint_name
        ] = theta.astype(
            np.float32,
            copy=False,
        )

        seen_arrays[
            checkpoint_name
        ] = seen.astype(
            bool,
            copy=False,
        )

        print(
            f"[RESUME] {checkpoint_name}: "
            f"{int(seen.sum()):,}/"
            f"{NUMBER_OF_PATENTS:,} patents"
        )

    else:
        theta_arrays[
            checkpoint_name
        ] = np.full(
            (
                NUMBER_OF_PATENTS,
                NUM_TOPICS,
            ),
            np.nan,
            dtype=np.float32,
        )

        seen_arrays[
            checkpoint_name
        ] = np.zeros(
            NUMBER_OF_PATENTS,
            dtype=bool,
        )


# ==================================================================================================
# 8. SHARD AND BATCH PREPARATION
# ==================================================================================================

def load_feature_shard(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def group_entries_by_patent(
    payload,
):
    entries = payload.get(
        "entries"
    )

    if not isinstance(entries, list):
        raise TypeError(
            "Feature shard entries "
            "must be a list."
        )

    grouped = {}

    for entry in entries:
        patent_id = normalize_integer_id(
            entry["patent_id"]
        )
        claim_id = normalize_integer_id(
            entry["claim_id"]
        )

        grouped.setdefault(
            patent_id,
            {},
        )

        if claim_id in grouped[patent_id]:
            raise ValueError(
                "Duplicate feature entry: "
                f"{patent_id}/{claim_id}"
            )

        grouped[
            patent_id
        ][claim_id] = entry

    return grouped


def make_patent_item(
    patent_id,
    feature_entries,
):
    record = record_by_patent_id[
        patent_id
    ]

    claims = normalize_claims(
        record
    )

    record_claim_ids = set(
        claims.keys()
    )

    feature_claim_ids = set(
        feature_entries.keys()
    )

    if (
        record_claim_ids
        != feature_claim_ids
    ):
        raise RuntimeError(
            f"Claim mismatch for patent "
            f"{patent_id}: "
            f"record_only="
            f"{sorted(record_claim_ids - feature_claim_ids)[:10]}, "
            f"feature_only="
            f"{sorted(feature_claim_ids - record_claim_ids)[:10]}"
        )

    number_of_tokens = 0
    number_of_token_edges = 0

    for entry in feature_entries.values():
        number_of_tokens += int(
            entry[
                "token_embeddings"
            ].shape[0]
        )

        number_of_token_edges += int(
            entry[
                "edge_index"
            ].shape[1]
        )

    return {
        "patent_id": patent_id,
        "record_index": (
            record_index_by_patent_id[
                patent_id
            ]
        ),
        "record": record,
        "feature_entries": (
            feature_entries
        ),
        "number_of_claims": len(
            feature_entries
        ),
        "number_of_tokens": (
            number_of_tokens
        ),
        "number_of_token_edges": (
            number_of_token_edges
        ),
    }


def construct_dynamic_batches(
    patent_items,
):
    batches = []

    current = []
    current_tokens = 0
    current_edges = 0

    for item in patent_items:
        item_tokens = int(
            item["number_of_tokens"]
        )

        item_edges = int(
            item["number_of_token_edges"]
        )

        would_exceed = (
            len(current)
            >= MAX_PATENTS_PER_EVAL_BATCH
            or (
                current
                and (
                    current_tokens
                    + item_tokens
                    > MAX_TOKENS_PER_EVAL_BATCH
                )
            )
            or (
                current
                and (
                    current_edges
                    + item_edges
                    > MAX_TOKEN_EDGES_PER_EVAL_BATCH
                )
            )
        )

        if would_exceed:
            batches.append(
                current
            )
            current = []
            current_tokens = 0
            current_edges = 0

        current.append(
            item
        )
        current_tokens += item_tokens
        current_edges += item_edges

    if current:
        batches.append(
            current
        )

    return batches


def collate_inference_batch(
    patent_items,
):
    token_embeddings_parts = []
    token_edge_index_parts = []
    token_edge_weight_parts = []
    token_to_claim_parts = []

    claim_to_patent = []
    claim_edges = []

    patent_ids = []
    record_indices = []

    total_tokens = 0
    total_claims = 0

    for (
        patent_local_index,
        item,
    ) in enumerate(
        patent_items
    ):
        patent_id = item[
            "patent_id"
        ]
        record = item[
            "record"
        ]
        feature_entries = item[
            "feature_entries"
        ]

        claims = normalize_claims(
            record
        )

        sorted_claim_ids = sorted(
            claims.keys(),
            key=sortable_id,
        )

        local_claim_positions = {}

        patent_ids.append(
            patent_id
        )
        record_indices.append(
            item["record_index"]
        )

        for claim_id in sorted_claim_ids:
            global_claim_index = (
                total_claims
            )

            local_claim_positions[
                claim_id
            ] = global_claim_index

            entry = feature_entries[
                claim_id
            ]

            token_embeddings = (
                entry["token_embeddings"]
                .detach()
                .contiguous()
            )

            edge_index = (
                entry["edge_index"]
                .detach()
                .long()
                .contiguous()
            )

            edge_weight = (
                entry["edge_weight"]
                .detach()
                .float()
                .contiguous()
            )

            number_of_tokens = int(
                token_embeddings.shape[0]
            )

            token_embeddings_parts.append(
                token_embeddings
            )

            token_edge_index_parts.append(
                edge_index + total_tokens
            )

            token_edge_weight_parts.append(
                edge_weight
            )

            token_to_claim_parts.append(
                torch.full(
                    (
                        number_of_tokens,
                    ),
                    global_claim_index,
                    dtype=torch.long,
                )
            )

            claim_to_patent.append(
                patent_local_index
            )

            total_tokens += (
                number_of_tokens
            )
            total_claims += 1

        for (
            parent_id,
            child_id,
        ) in normalize_edges(
            record
        ):
            if (
                parent_id
                not in local_claim_positions
                or child_id
                not in local_claim_positions
            ):
                raise KeyError(
                    "Dependency edge claim "
                    f"missing: {patent_id}/"
                    f"{parent_id}->{child_id}"
                )

            claim_edges.append((
                local_claim_positions[
                    parent_id
                ],
                local_claim_positions[
                    child_id
                ],
            ))

    if not token_embeddings_parts:
        raise RuntimeError(
            "Empty inference batch."
        )

    token_embeddings = torch.cat(
        token_embeddings_parts,
        dim=0,
    )

    token_edge_index = torch.cat(
        token_edge_index_parts,
        dim=1,
    )

    token_edge_weight = torch.cat(
        token_edge_weight_parts,
        dim=0,
    )

    token_to_claim = torch.cat(
        token_to_claim_parts,
        dim=0,
    )

    claim_to_patent = torch.tensor(
        claim_to_patent,
        dtype=torch.long,
    )

    if claim_edges:
        claim_edge_index = torch.tensor(
            claim_edges,
            dtype=torch.long,
        ).t().contiguous()

    else:
        claim_edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

    return {
        "patent_ids": patent_ids,
        "record_indices": record_indices,
        "num_patents": len(
            patent_ids
        ),
        "num_claims": total_claims,
        "num_tokens": total_tokens,
        "num_token_edges": int(
            token_edge_index.shape[1]
        ),
        "token_embeddings": (
            token_embeddings
        ),
        "token_edge_index": (
            token_edge_index
        ),
        "token_edge_weight": (
            token_edge_weight
        ),
        "token_to_claim": (
            token_to_claim
        ),
        "claim_to_patent": (
            claim_to_patent
        ),
        "claim_edge_index": (
            claim_edge_index
        ),
    }


def move_inference_batch_to_gpu(
    batch,
):
    moved = {
        "num_patents": (
            batch["num_patents"]
        ),
        "num_claims": (
            batch["num_claims"]
        ),
        "num_tokens": (
            batch["num_tokens"]
        ),
        "num_token_edges": (
            batch["num_token_edges"]
        ),
    }

    tensor_keys = [
        "token_embeddings",
        "token_edge_index",
        "token_edge_weight",
        "token_to_claim",
        "claim_to_patent",
        "claim_edge_index",
    ]

    for key in tensor_keys:
        moved[key] = batch[key].to(
            DEVICE,
            non_blocking=False,
        )

    return moved


# ==================================================================================================
# 9. INFERENCE WITH OOM SPLITTING
# ==================================================================================================

runtime_statistics = {
    "oom_splits": 0,
    "batches": 0,
    "patents": 0,
    "claims": 0,
    "tokens": 0,
    "token_edges": 0,
    "shards_processed": 0,
    "shards_skipped": 0,
    "drive_copy_seconds": 0.0,
    "shard_load_seconds": 0.0,
    "inference_seconds": 0.0,
}


def infer_patent_items(
    patent_items,
):
    cpu_batch = None
    gpu_batch = None
    outputs_by_model = None

    try:
        cpu_batch = (
            collate_inference_batch(
                patent_items
            )
        )

        gpu_batch = (
            move_inference_batch_to_gpu(
                cpu_batch
            )
        )

        torch.cuda.synchronize()

        inference_start = (
            time.perf_counter()
        )

        outputs_by_model = {}

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_FP16,
            ):
                for (
                    checkpoint_name,
                    model,
                ) in models.items():

                    patent_theta = (
                        model.forward_theta(
                            gpu_batch
                        )
                    )

                    outputs_by_model[
                        checkpoint_name
                    ] = (
                        patent_theta
                        .float()
                        .cpu()
                        .numpy()
                    )

        torch.cuda.synchronize()

        runtime_statistics[
            "inference_seconds"
        ] += (
            time.perf_counter()
            - inference_start
        )

        runtime_statistics[
            "batches"
        ] += 1

        runtime_statistics[
            "patents"
        ] += int(
            cpu_batch["num_patents"]
        )

        runtime_statistics[
            "claims"
        ] += int(
            cpu_batch["num_claims"]
        )

        runtime_statistics[
            "tokens"
        ] += int(
            cpu_batch["num_tokens"]
        )

        runtime_statistics[
            "token_edges"
        ] += int(
            cpu_batch["num_token_edges"]
        )

        record_indices = (
            cpu_batch["record_indices"]
        )

        for (
            checkpoint_name,
            theta,
        ) in outputs_by_model.items():

            theta_arrays[
                checkpoint_name
            ][record_indices] = theta

            seen_arrays[
                checkpoint_name
            ][record_indices] = True

    except torch.cuda.OutOfMemoryError:
        clear_cuda()

        if len(patent_items) <= 1:
            item = patent_items[0]

            raise RuntimeError(
                "한 개 patent도 GPU 메모리에 "
                "들어가지 않습니다. "
                f"patent_id={item['patent_id']}, "
                f"claims={item['number_of_claims']}, "
                f"tokens={item['number_of_tokens']:,}, "
                f"edges={item['number_of_token_edges']:,}"
            )

        runtime_statistics[
            "oom_splits"
        ] += 1

        midpoint = (
            len(patent_items) // 2
        )

        print(
            f"\n[OOM SPLIT] "
            f"patents={len(patent_items)} "
            f"→ {midpoint} + "
            f"{len(patent_items) - midpoint}"
        )

        infer_patent_items(
            patent_items[:midpoint]
        )

        infer_patent_items(
            patent_items[midpoint:]
        )

    finally:
        if outputs_by_model is not None:
            del outputs_by_model

        if gpu_batch is not None:
            del gpu_batch

        if cpu_batch is not None:
            del cpu_batch

        clear_cuda()


# ==================================================================================================
# 10. LOCAL SHARD PREFETCH
# ==================================================================================================

if USE_LOCAL_SHARD_PREFETCH:
    LOCAL_SHARD_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for stale_file in (
        LOCAL_SHARD_CACHE_DIR.glob(
            "*.copying"
        )
    ):
        stale_file.unlink(
            missing_ok=True
        )

    free_bytes = shutil.disk_usage(
        "/content"
    ).free

    maximum_shard_bytes = max(
        path.stat().st_size
        for path in evaluation_shards
    )

    required_free_bytes = (
        maximum_shard_bytes * 2
        + 2 * 1024**3
    )

    if free_bytes < required_free_bytes:
        print(
            "[WARNING] Local disk 공간이 "
            "부족하여 Drive 직접 로드로 "
            "전환합니다."
        )

        USE_LOCAL_SHARD_PREFETCH = False


def copy_shard_to_local(
    source_path,
):
    source_path = Path(
        source_path
    )

    target_path = (
        LOCAL_SHARD_CACHE_DIR
        / source_path.name
    )

    if (
        target_path.is_file()
        and target_path.stat().st_size
        == source_path.stat().st_size
    ):
        return target_path, 0.0

    temporary_path = (
        target_path.with_suffix(
            target_path.suffix
            + ".copying"
        )
    )

    temporary_path.unlink(
        missing_ok=True
    )

    start = time.perf_counter()

    shutil.copyfile(
        source_path,
        temporary_path,
    )

    os.replace(
        temporary_path,
        target_path,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if (
        target_path.stat().st_size
        != source_path.stat().st_size
    ):
        raise IOError(
            "Local shard copy size mismatch: "
            f"{source_path}"
        )

    return target_path, elapsed


# ==================================================================================================
# 11. DETERMINE REQUIRED SHARDS
# ==================================================================================================

required_shard_indices = []

for shard_index in range(
    len(evaluation_shards)
):
    start_index = (
        shard_index
        * PATENTS_PER_SHARD
    )

    end_index = min(
        start_index
        + PATENTS_PER_SHARD,
        NUMBER_OF_PATENTS,
    )

    all_models_complete = all(
        seen_arrays[
            checkpoint_name
        ][
            start_index:end_index
        ].all()
        for checkpoint_name in models
    )

    if all_models_complete:
        runtime_statistics[
            "shards_skipped"
        ] += 1
    else:
        required_shard_indices.append(
            shard_index
        )

print("\n" + "=" * 100)
print("INFERENCE PLAN")
print("=" * 100)
print(
    f"Split               : "
    f"{EVALUATION_SPLIT}"
)
print(
    f"Total shards        : "
    f"{len(evaluation_shards)}"
)
print(
    f"Already completed   : "
    f"{runtime_statistics['shards_skipped']}"
)
print(
    f"To process          : "
    f"{len(required_shard_indices)}"
)
print(
    f"Checkpoints         : "
    f"{list(models.keys())}"
)
print("=" * 100)


# ==================================================================================================
# 12. RUN SHARD-STREAMING INFERENCE
# ==================================================================================================

evaluation_start = (
    time.perf_counter()
)

torch.cuda.reset_peak_memory_stats()

executor = None
current_future = None

try:
    if (
        USE_LOCAL_SHARD_PREFETCH
        and required_shard_indices
    ):
        executor = ThreadPoolExecutor(
            max_workers=1
        )

        first_index = (
            required_shard_indices[0]
        )

        current_future = executor.submit(
            copy_shard_to_local,
            evaluation_shards[
                first_index
            ],
        )

    progress = tqdm(
        required_shard_indices,
        desc=(
            f"Depth-OT "
            f"{EVALUATION_SPLIT.upper()} "
            f"shards"
        ),
        unit="shard",
    )

    for (
        position,
        shard_index,
    ) in enumerate(
        progress
    ):
        source_shard_path = (
            evaluation_shards[
                shard_index
            ]
        )

        # ------------------------------------------------------------------------------------------
        # 현재 shard 확보 및 다음 shard prefetch
        # ------------------------------------------------------------------------------------------

        if USE_LOCAL_SHARD_PREFETCH:
            (
                shard_path,
                copy_seconds,
            ) = current_future.result()

            runtime_statistics[
                "drive_copy_seconds"
            ] += copy_seconds

            next_position = (
                position + 1
            )

            if (
                next_position
                < len(required_shard_indices)
            ):
                next_shard_index = (
                    required_shard_indices[
                        next_position
                    ]
                )

                current_future = (
                    executor.submit(
                        copy_shard_to_local,
                        evaluation_shards[
                            next_shard_index
                        ],
                    )
                )
            else:
                current_future = None

        else:
            shard_path = (
                source_shard_path
            )

        # ------------------------------------------------------------------------------------------
        # Shard 로드
        # ------------------------------------------------------------------------------------------

        load_start = (
            time.perf_counter()
        )

        payload = load_feature_shard(
            shard_path
        )

        runtime_statistics[
            "shard_load_seconds"
        ] += (
            time.perf_counter()
            - load_start
        )

        # 핵심 수정: test 하드코딩이 아니라 EVALUATION_SPLIT 기준으로 확인
        payload_split = str(
            payload.get(
                "split",
                "",
            )
        ).strip().lower()

        expected_split = (
            EVALUATION_SPLIT
        )

        split_aliases = {
            "dev": {
                "dev",
                "validation",
                "valid",
                "val",
            },
            "validation": {
                "dev",
                "validation",
                "valid",
                "val",
            },
            "valid": {
                "dev",
                "validation",
                "valid",
                "val",
            },
            "val": {
                "dev",
                "validation",
                "valid",
                "val",
            },
            "test": {
                "test",
            },
            "train": {
                "train",
            },
        }

        allowed_payload_splits = (
            split_aliases.get(
                expected_split,
                {expected_split},
            )
        )

        if (
            payload_split
            not in allowed_payload_splits
        ):
            raise RuntimeError(
                "Feature shard split mismatch.\n"
                f"Expected split : "
                f"{expected_split}\n"
                f"Allowed values : "
                f"{sorted(allowed_payload_splits)}\n"
                f"Found split    : "
                f"{payload_split}\n"
                f"Shard path     : "
                f"{source_shard_path}"
            )

        observed_shard_id = (
            payload.get(
                "shard_id"
            )
        )

        if (
            observed_shard_id is not None
            and int(observed_shard_id)
            != int(shard_index)
        ):
            raise RuntimeError(
                "Shard ID mismatch: "
                f"path={source_shard_path}, "
                f"payload={observed_shard_id}"
            )

        shard_patent_ids = [
            normalize_integer_id(
                patent_id
            )
            for patent_id
            in payload["patent_ids"]
        ]

        expected_start = (
            shard_index
            * PATENTS_PER_SHARD
        )

        expected_end = min(
            expected_start
            + PATENTS_PER_SHARD,
            NUMBER_OF_PATENTS,
        )

        expected_patent_ids = [
            normalize_integer_id(
                evaluation_records[
                    index
                ]["patent_id"]
            )
            for index in range(
                expected_start,
                expected_end,
            )
        ]

        if (
            shard_patent_ids
            != expected_patent_ids
        ):
            raise RuntimeError(
                "Shard/record order mismatch: "
                f"split={EVALUATION_SPLIT}, "
                f"shard={shard_index}"
            )

        entries_by_patent = (
            group_entries_by_patent(
                payload
            )
        )

        patent_items = []

        for patent_id in shard_patent_ids:
            if (
                patent_id
                not in record_by_patent_id
            ):
                raise KeyError(
                    "Unknown patent ID: "
                    f"{patent_id}"
                )

            if (
                patent_id
                not in entries_by_patent
            ):
                raise KeyError(
                    "Feature patent missing: "
                    f"{patent_id}"
                )

            patent_items.append(
                make_patent_item(
                    patent_id,
                    entries_by_patent[
                        patent_id
                    ],
                )
            )

        dynamic_batches = (
            construct_dynamic_batches(
                patent_items
            )
        )

        for batch_items in dynamic_batches:
            infer_patent_items(
                batch_items
            )

        runtime_statistics[
            "shards_processed"
        ] += 1

        # ------------------------------------------------------------------------------------------
        # Resume progress 저장
        # ------------------------------------------------------------------------------------------

        if (
            runtime_statistics[
                "shards_processed"
            ]
            % SAVE_PROGRESS_EVERY_SHARDS
            == 0
        ):
            for checkpoint_name in models:
                result_directory = (
                    result_directories[
                        checkpoint_name
                    ]
                )

                atomic_numpy_save(
                    theta_arrays[
                        checkpoint_name
                    ],
                    result_directory
                    / (
                        f"{EVALUATION_SPLIT}"
                        "_theta_progress.npy"
                    ),
                )

                atomic_numpy_save(
                    seen_arrays[
                        checkpoint_name
                    ],
                    result_directory
                    / (
                        f"{EVALUATION_SPLIT}"
                        "_seen_progress.npy"
                    ),
                )

        del dynamic_batches
        del patent_items
        del entries_by_patent
        del payload

        gc.collect()

        if USE_LOCAL_SHARD_PREFETCH:
            Path(
                shard_path
            ).unlink(
                missing_ok=True
            )

        elapsed = (
            time.perf_counter()
            - evaluation_start
        )

        first_checkpoint_name = next(
            iter(models)
        )

        completed_patents = int(
            seen_arrays[
                first_checkpoint_name
            ].sum()
        )

        progress.set_postfix({
            "patents": (
                f"{completed_patents:,}/"
                f"{NUMBER_OF_PATENTS:,}"
            ),
            "batches": (
                runtime_statistics[
                    "batches"
                ]
            ),
            "gpu": (
                f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G"
            ),
            "elapsed": (
                f"{elapsed / 60:.1f}m"
            ),
        })

finally:
    if executor is not None:
        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

evaluation_seconds = (
    time.perf_counter()
    - evaluation_start
)


# ==================================================================================================
# 13. FINAL VALIDATION
# ==================================================================================================

for checkpoint_name in models:
    seen = seen_arrays[
        checkpoint_name
    ]

    theta = theta_arrays[
        checkpoint_name
    ]

    if not seen.all():
        missing_indices = np.flatnonzero(
            ~seen
        )

        raise RuntimeError(
            f"{checkpoint_name}: "
            f"{len(missing_indices):,} "
            f"patents missing. "
            f"First missing indices: "
            f"{missing_indices[:20].tolist()}"
        )

    if not np.isfinite(
        theta
    ).all():
        raise RuntimeError(
            f"{checkpoint_name}: "
            f"theta contains NaN/Inf."
        )

    row_sums = theta.sum(
        axis=1
    )

    maximum_sum_error = float(
        np.max(
            np.abs(
                row_sums - 1.0
            )
        )
    )

    if maximum_sum_error > 1.0e-4:
        raise RuntimeError(
            f"{checkpoint_name}: "
            f"theta simplex error="
            f"{maximum_sum_error}"
        )

    # 최종 progress도 반드시 저장
    result_directory = (
        result_directories[
            checkpoint_name
        ]
    )

    atomic_numpy_save(
        theta,
        result_directory
        / (
            f"{EVALUATION_SPLIT}"
            "_theta_progress.npy"
        ),
    )

    atomic_numpy_save(
        seen,
        result_directory
        / (
            f"{EVALUATION_SPLIT}"
            "_seen_progress.npy"
        ),
    )

print(
    "\n[PASS] All theta arrays passed "
    "final validation."
)


# ==================================================================================================
# 14. CPC METRICS
# ==================================================================================================

def normalize_label(value):
    if value is None:
        return None

    text = str(
        value
    ).strip().upper()

    if not text:
        return None

    return text


def extract_cpc_label(
    record,
    level,
):
    direct = normalize_label(
        record.get(
            level
        )
    )

    if direct is not None:
        return direct

    cpc_codes = record.get(
        "cpc_codes",
        [],
    )

    if isinstance(
        cpc_codes,
        str,
    ):
        cpc_codes = [
            cpc_codes
        ]

    if not cpc_codes:
        return None

    code = normalize_label(
        cpc_codes[0]
    )

    if code is None:
        return None

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        code,
    )

    if level == "section":
        return compact[:1]

    if level == "class":
        return compact[:3]

    if level == "subclass":
        return compact[:4]

    raise ValueError(
        level
    )


def calculate_alignment_metrics(
    true_labels,
    predicted_topics,
):
    true_labels = np.asarray(
        true_labels,
        dtype=object,
    )

    predicted_topics = np.asarray(
        predicted_topics,
        dtype=np.int64,
    )

    valid = np.asarray([
        (
            label is not None
            and str(label).strip() != ""
        )
        for label in true_labels
    ])

    true_labels = (
        true_labels[valid]
    )

    predicted_topics = (
        predicted_topics[valid]
    )

    if true_labels.size == 0:
        raise RuntimeError(
            "No valid CPC labels."
        )

    (
        unique_labels,
        encoded_labels,
    ) = np.unique(
        true_labels.astype(str),
        return_inverse=True,
    )

    matrix = contingency_matrix(
        encoded_labels,
        predicted_topics,
        sparse=False,
    ).astype(
        np.int64
    )

    number_of_samples = int(
        matrix.sum()
    )

    # Pur_p: predicted topic → majority CPC label
    topic_totals = matrix.sum(
        axis=0
    )

    nonempty_topics = (
        topic_totals > 0
    )

    topic_majority_counts = (
        matrix.max(axis=0)
    )

    pur_p_weighted = float(
        topic_majority_counts.sum()
        / max(
            number_of_samples,
            1,
        )
    )

    pur_p_macro = (
        float(
            np.mean(
                topic_majority_counts[
                    nonempty_topics
                ]
                / topic_totals[
                    nonempty_topics
                ]
            )
        )
        if nonempty_topics.any()
        else 0.0
    )

    # Pur_a: CPC label → majority predicted topic
    label_totals = matrix.sum(
        axis=1
    )

    label_majority_counts = matrix.max(
        axis=1
    )

    pur_a_weighted = float(
        label_majority_counts.sum()
        / max(
            number_of_samples,
            1,
        )
    )

    pur_a_macro = float(
        np.mean(
            label_majority_counts
            / np.maximum(
                label_totals,
                1,
            )
        )
    )

    if PRIMARY_PUR_A_MODE == "weighted":
        primary_pur_a = (
            pur_a_weighted
        )

    elif PRIMARY_PUR_A_MODE == "macro":
        primary_pur_a = (
            pur_a_macro
        )

    else:
        raise ValueError(
            "PRIMARY_PUR_A_MODE must be "
            "'weighted' or 'macro'."
        )

    nmi = float(
        normalized_mutual_info_score(
            encoded_labels,
            predicted_topics,
            average_method="arithmetic",
        )
    )

    return {
        "number_of_samples": (
            number_of_samples
        ),
        "number_of_labels": int(
            len(unique_labels)
        ),
        "number_of_predicted_topics": int(
            np.unique(
                predicted_topics
            ).size
        ),
        "pur_p": (
            pur_p_weighted
        ),
        "pur_a": (
            primary_pur_a
        ),
        "nmi": nmi,
        "pur_p_weighted": (
            pur_p_weighted
        ),
        "pur_p_macro": (
            pur_p_macro
        ),
        "pur_a_weighted": (
            pur_a_weighted
        ),
        "pur_a_macro": (
            pur_a_macro
        ),
        "pur_a_primary_mode": (
            PRIMARY_PUR_A_MODE
        ),
    }


# ==================================================================================================
# 15. SAVE FINAL RESULTS
# ==================================================================================================

all_results = {}

for checkpoint_name in models:
    result_directory = (
        result_directories[
            checkpoint_name
        ]
    )

    theta = theta_arrays[
        checkpoint_name
    ].astype(
        np.float32,
        copy=False,
    )

    predicted_topics = theta.argmax(
        axis=1
    ).astype(
        np.int64
    )

    topic_counts = np.bincount(
        predicted_topics,
        minlength=NUM_TOPICS,
    )

    topic_shares = (
        topic_counts
        / NUMBER_OF_PATENTS
    )

    marginal = theta.mean(
        axis=0
    )

    marginal = (
        marginal
        / np.maximum(
            marginal.sum(),
            1.0e-12,
        )
    )

    marginal_entropy = float(
        -np.sum(
            marginal
            * np.log(
                np.maximum(
                    marginal,
                    1.0e-12,
                )
            )
        )
        / np.log(
            NUM_TOPICS
        )
    )

    assignment_entropy = float(
        np.mean(
            -np.sum(
                theta
                * np.log(
                    np.maximum(
                        theta,
                        1.0e-12,
                    )
                ),
                axis=1,
            )
        )
        / np.log(
            NUM_TOPICS
        )
    )

    model_results = {
        "checkpoint": (
            checkpoint_metadata[
                checkpoint_name
            ]
        ),
        "evaluation": {
            "created_at_utc": (
                utc_now()
            ),
            "split": (
                EVALUATION_SPLIT
            ),
            "gpu": (
                GPU_NAME
            ),
            "gpu_memory_gib": (
                GPU_MEMORY_GIB
            ),
            "fp16": (
                USE_FP16
            ),
            "deterministic_posterior_mean": True,
            "decoder_computed": False,
            "bow_computed": False,
            "sinkhorn_computed": False,
            "number_of_patents": (
                NUMBER_OF_PATENTS
            ),
            "number_of_topics": (
                NUM_TOPICS
            ),
            "evaluation_seconds": (
                evaluation_seconds
            ),
            "evaluation_minutes": (
                evaluation_seconds
                / 60.0
            ),
            "gpu_peak_allocated_gib": (
                torch.cuda.max_memory_allocated()
                / 1024**3
            ),
        },
        "topic_statistics": {
            "active_topics_nonzero": int(
                np.sum(
                    topic_counts > 0
                )
            ),
            "maximum_topic_share": float(
                topic_shares.max()
            ),
            "maximum_topic_id": int(
                topic_shares.argmax()
            ),
            "marginal_entropy_normalized": (
                marginal_entropy
            ),
            "assignment_entropy_normalized": (
                assignment_entropy
            ),
            "theta_variance": float(
                theta.var(
                    axis=0
                ).mean()
            ),
            "topic_counts": (
                topic_counts.tolist()
            ),
            "topic_shares": (
                topic_shares.tolist()
            ),
        },
        "cpc_alignment": {},
    }

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        labels = [
            extract_cpc_label(
                record,
                level,
            )
            for record
            in evaluation_records
        ]

        model_results[
            "cpc_alignment"
        ][level] = (
            calculate_alignment_metrics(
                labels,
                predicted_topics,
            )
        )

    section_metrics = (
        model_results[
            "cpc_alignment"
        ]["section"]
    )

    class_metrics = (
        model_results[
            "cpc_alignment"
        ]["class"]
    )

    subclass_metrics = (
        model_results[
            "cpc_alignment"
        ]["subclass"]
    )

    model_results[
        "checkpoint_selection"
    ] = {
        "mean_nmi": float(
            np.mean([
                section_metrics[
                    "nmi"
                ],
                class_metrics[
                    "nmi"
                ],
                subclass_metrics[
                    "nmi"
                ],
            ])
        ),
        "mean_pur_p": float(
            np.mean([
                section_metrics[
                    "pur_p"
                ],
                class_metrics[
                    "pur_p"
                ],
                subclass_metrics[
                    "pur_p"
                ],
            ])
        ),
        "mean_pur_a": float(
            np.mean([
                section_metrics[
                    "pur_a"
                ],
                class_metrics[
                    "pur_a"
                ],
                subclass_metrics[
                    "pur_a"
                ],
            ])
        ),
    }

    all_results[
        checkpoint_name
    ] = model_results

    output_prefix = (
        EVALUATION_SPLIT
    )

    atomic_numpy_save(
        theta,
        result_directory
        / f"{output_prefix}_theta.npy",
    )

    atomic_numpy_save(
        predicted_topics,
        result_directory
        / (
            f"{output_prefix}"
            "_predicted_topics.npy"
        ),
    )

    atomic_numpy_save(
        topic_counts,
        result_directory
        / (
            f"{output_prefix}"
            "_topic_counts.npy"
        ),
    )

    prediction_csv_path = (
        result_directory
        / (
            f"{output_prefix}"
            "_patent_topic_predictions.csv"
        )
    )

    temporary_csv_path = (
        prediction_csv_path.with_suffix(
            ".csv.tmp"
        )
    )

    with open(
        temporary_csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "record_index",
            "patent_id",
            "predicted_topic",
            "maximum_theta",
            "section",
            "class",
            "subclass",
        ])

        for (
            record_index,
            record,
        ) in enumerate(
            evaluation_records
        ):
            writer.writerow([
                record_index,
                record["patent_id"],
                int(
                    predicted_topics[
                        record_index
                    ]
                ),
                float(
                    theta[
                        record_index
                    ].max()
                ),
                extract_cpc_label(
                    record,
                    "section",
                ),
                extract_cpc_label(
                    record,
                    "class",
                ),
                extract_cpc_label(
                    record,
                    "subclass",
                ),
            ])

    os.replace(
        temporary_csv_path,
        prediction_csv_path,
    )

    atomic_json_save(
        model_results,
        result_directory
        / "evaluation_summary.json",
    )


# ==================================================================================================
# 16. CHECKPOINT COMPARISON AND SELECTION
# ==================================================================================================

comparison_rows = []

for (
    checkpoint_name,
    result,
) in all_results.items():

    section = result[
        "cpc_alignment"
    ]["section"]

    class_result = result[
        "cpc_alignment"
    ]["class"]

    subclass = result[
        "cpc_alignment"
    ]["subclass"]

    selection = result[
        "checkpoint_selection"
    ]

    comparison_rows.append({
        "checkpoint": (
            checkpoint_name
        ),
        "stored_epoch": (
            result["checkpoint"][
                "epoch"
            ]
        ),
        "section_pur_p": (
            section["pur_p"]
        ),
        "section_pur_a": (
            section["pur_a"]
        ),
        "section_nmi": (
            section["nmi"]
        ),
        "class_pur_p": (
            class_result["pur_p"]
        ),
        "class_pur_a": (
            class_result["pur_a"]
        ),
        "class_nmi": (
            class_result["nmi"]
        ),
        "subclass_pur_p": (
            subclass["pur_p"]
        ),
        "subclass_pur_a": (
            subclass["pur_a"]
        ),
        "subclass_nmi": (
            subclass["nmi"]
        ),
        "mean_pur_p": (
            selection["mean_pur_p"]
        ),
        "mean_pur_a": (
            selection["mean_pur_a"]
        ),
        "mean_nmi": (
            selection["mean_nmi"]
        ),
    })

comparison_rows = sorted(
    comparison_rows,
    key=lambda row: row[
        "mean_nmi"
    ],
    reverse=True,
)

for rank, row in enumerate(
    comparison_rows,
    start=1,
):
    row["mean_nmi_rank"] = rank

best_dev_checkpoint = (
    comparison_rows[0][
        "checkpoint"
    ]
)

comparison_csv_path = (
    RESULT_ROOT
    / "dev_checkpoint_comparison.csv"
)

comparison_fieldnames = [
    "mean_nmi_rank",
    "checkpoint",
    "stored_epoch",
    "section_pur_p",
    "section_pur_a",
    "section_nmi",
    "class_pur_p",
    "class_pur_a",
    "class_nmi",
    "subclass_pur_p",
    "subclass_pur_a",
    "subclass_nmi",
    "mean_pur_p",
    "mean_pur_a",
    "mean_nmi",
]

temporary_comparison_path = (
    comparison_csv_path.with_suffix(
        ".csv.tmp"
    )
)

with open(
    temporary_comparison_path,
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=(
            comparison_fieldnames
        ),
    )

    writer.writeheader()

    for row in comparison_rows:
        writer.writerow(
            row
        )

os.replace(
    temporary_comparison_path,
    comparison_csv_path,
)


# ==================================================================================================
# 17. SAVE COMPLETE SUMMARY
# ==================================================================================================

complete_summary = {
    "created_at_utc": (
        utc_now()
    ),
    "configuration": {
        "split": (
            EVALUATION_SPLIT
        ),
        "checkpoints": (
            CHECKPOINTS
        ),
        "records_path": str(
            RECORDS_PATH
        ),
        "feature_dir": str(
            FEATURE_DIR
        ),
        "result_root": str(
            RESULT_ROOT
        ),
        "expected_number_of_patents": (
            EXPECTED_NUMBER_OF_PATENTS
        ),
        "expected_number_of_shards": (
            EXPECTED_NUMBER_OF_SHARDS
        ),
        "use_fp16": (
            USE_FP16
        ),
        "use_local_shard_prefetch": (
            USE_LOCAL_SHARD_PREFETCH
        ),
        "maximum_patents_per_batch": (
            MAX_PATENTS_PER_EVAL_BATCH
        ),
        "maximum_tokens_per_batch": (
            MAX_TOKENS_PER_EVAL_BATCH
        ),
        "maximum_token_edges_per_batch": (
            MAX_TOKEN_EDGES_PER_EVAL_BATCH
        ),
        "pur_a_primary_mode": (
            PRIMARY_PUR_A_MODE
        ),
    },
    "checkpoint_selection": {
        "primary_metric": (
            "mean_nmi"
        ),
        "best_dev_checkpoint": (
            best_dev_checkpoint
        ),
        "ranking": (
            comparison_rows
        ),
    },
    "runtime_statistics": {
        **runtime_statistics,
        "total_evaluation_seconds": (
            evaluation_seconds
        ),
        "total_evaluation_minutes": (
            evaluation_seconds
            / 60.0
        ),
        "gpu_peak_allocated_gib": (
            torch.cuda.max_memory_allocated()
            / 1024**3
        ),
    },
    "results": (
        all_results
    ),
}

atomic_json_save(
    complete_summary,
    RESULT_ROOT
    / (
        "depth_ot_v2_dev_checkpoint_"
        "evaluation_complete.json"
    ),
)


# ==================================================================================================
# 18. CONSOLE REPORT AND LATEX ROWS
# ==================================================================================================

print("\n")
print("=" * 100)
print(
    "DEPTH-OT V2 — FINAL DEV CPC ALIGNMENT"
)
print("=" * 100)

for (
    checkpoint_name,
    result,
) in all_results.items():

    metadata = result[
        "checkpoint"
    ]

    print("\n" + "-" * 100)
    print(
        f"{checkpoint_name} | "
        f"epoch={metadata['epoch']}"
    )
    print("-" * 100)

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        metrics = result[
            "cpc_alignment"
        ][level]

        print(
            f"\n[{level.upper()}]"
        )
        print(
            f"Labels : "
            f"{metrics['number_of_labels']}"
        )
        print(
            f"Pur_p  : "
            f"{metrics['pur_p']:.4f}"
        )
        print(
            f"Pur_a  : "
            f"{metrics['pur_a']:.4f} "
            f"({metrics['pur_a_primary_mode']})"
        )
        print(
            f"NMI    : "
            f"{metrics['nmi']:.4f}"
        )

    topic_statistics = result[
        "topic_statistics"
    ]

    selection = result[
        "checkpoint_selection"
    ]

    print("\n[TOPIC STATISTICS]")
    print(
        f"Active topics       : "
        f"{topic_statistics['active_topics_nonzero']}/"
        f"{NUM_TOPICS}"
    )
    print(
        f"Maximum topic share : "
        f"{topic_statistics['maximum_topic_share']:.2%}"
    )
    print(
        f"Marginal entropy    : "
        f"{topic_statistics['marginal_entropy_normalized']:.6f}"
    )
    print(
        f"Assignment entropy  : "
        f"{topic_statistics['assignment_entropy_normalized']:.6f}"
    )
    print(
        f"Theta variance      : "
        f"{topic_statistics['theta_variance']:.6e}"
    )

    print("\n[DEV SELECTION SCORE]")
    print(
        f"Mean Pur_p          : "
        f"{selection['mean_pur_p']:.6f}"
    )
    print(
        f"Mean Pur_a          : "
        f"{selection['mean_pur_a']:.6f}"
    )
    print(
        f"Mean NMI            : "
        f"{selection['mean_nmi']:.6f}"
    )

    section = result[
        "cpc_alignment"
    ]["section"]

    class_result = result[
        "cpc_alignment"
    ]["class"]

    subclass = result[
        "cpc_alignment"
    ]["subclass"]

    latex_row = (
        f"Depth-OT V2 ({checkpoint_name})"
        f" & {section['pur_p']:.4f}"
        f" & {section['pur_a']:.4f}"
        f" & {section['nmi']:.4f}"
        f" & {class_result['pur_p']:.4f}"
        f" & {class_result['pur_a']:.4f}"
        f" & {class_result['nmi']:.4f}"
        f" & {subclass['pur_p']:.4f}"
        f" & {subclass['pur_a']:.4f}"
        f" & {subclass['nmi']:.4f}"
        r" \\"
    )

    latex_path = (
        result_directories[
            checkpoint_name
        ]
        / "latex_row.txt"
    )

    with open(
        latex_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            latex_row + "\n"
        )

    print("\n[LATEX ROW]")
    print(
        latex_row
    )


# ==================================================================================================
# 19. DEV CHECKPOINT RANKING
# ==================================================================================================

print("\n")
print("=" * 100)
print(
    "DEV CHECKPOINT RANKING — MEAN NMI"
)
print("=" * 100)

print(
    f"{'Rank':>4s} "
    f"{'Checkpoint':<32s} "
    f"{'Epoch':>7s} "
    f"{'Sec NMI':>10s} "
    f"{'Class NMI':>10s} "
    f"{'Sub NMI':>10s} "
    f"{'Mean NMI':>10s}"
)

print("-" * 100)

for row in comparison_rows:
    print(
        f"{row['mean_nmi_rank']:>4d} "
        f"{row['checkpoint']:<32s} "
        f"{str(row['stored_epoch']):>7s} "
        f"{row['section_nmi']:>10.4f} "
        f"{row['class_nmi']:>10.4f} "
        f"{row['subclass_nmi']:>10.4f} "
        f"{row['mean_nmi']:>10.4f}"
    )

print("-" * 100)
print(
    f"[BEST DEV CHECKPOINT] "
    f"{best_dev_checkpoint}"
)

best_row = comparison_rows[0]

print(
    f"[BEST MEAN NMI] "
    f"{best_row['mean_nmi']:.6f}"
)
print(
    f"[COMPARISON CSV] "
    f"{comparison_csv_path}"
)


# ==================================================================================================
# 20. RUNTIME REPORT
# ==================================================================================================

print("\n" + "=" * 100)
print("RUNTIME")
print("=" * 100)
print(
    f"Total time          : "
    f"{evaluation_seconds / 60:.2f} minutes"
)
print(
    f"Patents             : "
    f"{NUMBER_OF_PATENTS:,}"
)
print(
    f"Patents/second      : "
    f"{NUMBER_OF_PATENTS / max(evaluation_seconds, 1e-12):.2f}"
)
print(
    f"Processed batches   : "
    f"{runtime_statistics['batches']:,}"
)
print(
    f"OOM auto-splits     : "
    f"{runtime_statistics['oom_splits']:,}"
)
print(
    f"GPU peak allocated  : "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
)
print(
    f"Drive copy time     : "
    f"{runtime_statistics['drive_copy_seconds'] / 60:.2f} minutes"
)
print(
    f"Shard load time     : "
    f"{runtime_statistics['shard_load_seconds'] / 60:.2f} minutes"
)
print(
    f"GPU inference time  : "
    f"{runtime_statistics['inference_seconds'] / 60:.2f} minutes"
)
print(
    f"Result directory    : "
    f"{RESULT_ROOT}"
)
print("=" * 100)


# ==================================================================================================
# 21. FINAL RESULT VERIFICATION
# ==================================================================================================

required_result_files = [
    f"{EVALUATION_SPLIT}_theta.npy",
    f"{EVALUATION_SPLIT}_predicted_topics.npy",
    f"{EVALUATION_SPLIT}_topic_counts.npy",
    (
        f"{EVALUATION_SPLIT}"
        "_patent_topic_predictions.csv"
    ),
    "evaluation_summary.json",
    "latex_row.txt",
]

missing_result_files = []

complete_summary_path = (
    RESULT_ROOT
    / (
        "depth_ot_v2_dev_checkpoint_"
        "evaluation_complete.json"
    )
)

if not complete_summary_path.is_file():
    missing_result_files.append(
        complete_summary_path
    )

if not comparison_csv_path.is_file():
    missing_result_files.append(
        comparison_csv_path
    )

for checkpoint_name in models:
    checkpoint_result_directory = (
        RESULT_ROOT
        / safe_name(
            checkpoint_name
        )
    )

    for filename in required_result_files:
        result_path = (
            checkpoint_result_directory
            / filename
        )

        if (
            not result_path.is_file()
            or result_path.stat().st_size == 0
        ):
            missing_result_files.append(
                result_path
            )


# ==================================================================================================
# 22. LOCAL FEATURE CACHE CLEANUP
# ==================================================================================================

if REMOVE_LOCAL_CACHE_WHEN_DONE:
    try:
        shutil.rmtree(
            LOCAL_SHARD_CACHE_DIR,
            ignore_errors=True,
        )

        print(
            "\n[CLEANUP] "
            "Local DEV feature shard cache removed."
        )

    except Exception as error:
        print(
            "\n[WARNING] "
            "Local cache cleanup failed: "
            f"{type(error).__name__}: "
            f"{error}"
        )


# ==================================================================================================
# 23. FINAL STATUS
# ==================================================================================================

print("\n" + "=" * 100)

if not missing_result_files:
    print(
        "[PASS] DEV evaluation completed successfully."
    )
    print(
        "[PASS] Epoch 4, 15, and 16 were evaluated "
        "on the same DEV split."
    )
    print(
        f"[BEST DEV CHECKPOINT] "
        f"{best_dev_checkpoint}"
    )
    print(
        f"[DRIVE RESULTS] "
        f"{RESULT_ROOT}"
    )

else:
    print(
        "[WARNING] Evaluation finished, "
        "but some files are missing."
    )

    for missing_path in missing_result_files:
        print(
            f"[MISSING] "
            f"{missing_path}"
        )

print("=" * 100)
