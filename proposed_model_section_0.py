# ============================================================
# SECTION 0 — DEPTH-OT V2
# Environment, Paths, Reproducibility, and Configuration
# ============================================================

import os
import sys
import gc
import json
import math
import time
import random
import shutil
import subprocess
import importlib
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np


# ============================================================
# 0.1 Environment variables
# Must be configured before intensive PyTorch operations
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ============================================================
# 0.2 Google Drive
# ============================================================

try:
    from google.colab import drive

    if not Path(
        "/content/drive/MyDrive"
    ).is_dir():
        drive.mount(
            "/content/drive",
            force_remount=False,
        )

except ImportError:
    print(
        "[WARNING] Google Colab이 아닙니다. "
        "Drive가 이미 마운트되어 있어야 합니다."
    )


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/depth_ot_patent"
)

if not PROJECT_ROOT.is_dir():
    raise FileNotFoundError(
        f"Project root가 없습니다: "
        f"{PROJECT_ROOT}"
    )


# ============================================================
# 0.3 Package check
# ============================================================

REQUIRED_PACKAGES = {
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
    "einops": "einops",
    "tqdm": "tqdm",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
}

missing_packages = []

for import_name, pip_name in (
    REQUIRED_PACKAGES.items()
):
    try:
        importlib.import_module(
            import_name
        )
    except ImportError:
        missing_packages.append(
            pip_name
        )

if missing_packages:
    print(
        "[INSTALL]",
        missing_packages,
    )

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        *missing_packages,
    ])


# ============================================================
# 0.4 PyTorch environment
# ============================================================

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. "
        "Colab GPU 런타임을 활성화하세요."
    )

DEVICE = torch.device("cuda:0")

GPU_NAME = torch.cuda.get_device_name(0)

GPU_PROPERTIES = (
    torch.cuda.get_device_properties(0)
)

GPU_TOTAL_GIB = (
    GPU_PROPERTIES.total_memory
    / 2**30
)

CUDA_CAPABILITY = (
    torch.cuda.get_device_capability(0)
)

print("=" * 88)
print("PYTORCH AND CUDA")
print("=" * 88)
print(f"Python            : {sys.version.split()[0]}")
print(f"PyTorch           : {torch.__version__}")
print(f"PyTorch CUDA      : {torch.version.cuda}")
print(f"GPU               : {GPU_NAME}")
print(f"GPU memory        : {GPU_TOTAL_GIB:.2f} GiB")
print(
    f"Compute capability: "
    f"{CUDA_CAPABILITY[0]}."
    f"{CUDA_CAPABILITY[1]}"
)
print("=" * 88)


# ============================================================
# 0.5 Reproducibility
# ============================================================

STRICT_REPRODUCIBILITY = True
ENABLE_TF32 = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = (
    STRICT_REPRODUCIBILITY
)

torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = (
    ENABLE_TF32
)

torch.backends.cudnn.allow_tf32 = (
    ENABLE_TF32
)

try:
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )
except Exception as error:
    print(
        "[WARNING] Deterministic algorithms:",
        error,
    )

try:
    torch.set_float32_matmul_precision(
        "highest"
        if not ENABLE_TF32
        else "high"
    )
except Exception:
    pass

DATALOADER_GENERATOR = (
    torch.Generator()
)

DATALOADER_GENERATOR.manual_seed(
    SEED
)


def seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


# ============================================================
# 0.6 Project directories
# ============================================================

DIRS = {
    "raw": (
        PROJECT_ROOT
        / "data"
        / "raw"
    ),
    "processed": (
        PROJECT_ROOT
        / "data"
        / "processed"
    ),
    "token_features": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "token_features"
    ),
    "records": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "records"
    ),
    "vocab": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "vocabulary"
    ),
    "checkpoints": (
        PROJECT_ROOT
        / "checkpoints"
    ),
    "logs": (
        PROJECT_ROOT
        / "logs"
    ),
    "results": (
        PROJECT_ROOT
        / "results"
    ),
    "notebooks": (
        PROJECT_ROOT
        / "notebooks"
    ),
    "depth_ot_v2_ckpt": (
        PROJECT_ROOT
        / "checkpoints"
        / "depth_ot_v2"
    ),
    "depth_ot_v2_logs": (
        PROJECT_ROOT
        / "logs"
        / "depth_ot_v2"
    ),
    "depth_ot_v2_results": (
        PROJECT_ROOT
        / "results"
        / "depth_ot_v2"
    ),
    "depth_ot_v2_processed": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "depth_ot_v2"
    ),
    "hf_cache": (
        PROJECT_ROOT
        / "cache"
        / "huggingface"
    ),
    "local_cache": Path(
        "/content/depth_ot_v2_cache"
    ),
    "local_feature_cache": Path(
        "/content/depth_ot_v2_feature_cache"
    ),
}

for directory in DIRS.values():
    Path(directory).mkdir(
        parents=True,
        exist_ok=True,
    )

os.environ["HF_HOME"] = str(
    DIRS["hf_cache"]
)


# ============================================================
# 0.7 Feature source
# Existing PatentSBERTa features are reused
# ============================================================

FEATURE_RUN_NAME = "full"

FEATURE_ROOT = (
    DIRS["token_features"]
    / FEATURE_RUN_NAME
)

FEATURE_SPLIT_DIRS = {
    split: FEATURE_ROOT / split
    for split in [
        "train",
        "dev",
        "test",
    ]
}

PLM_NAME = (
    "AAUBS/PatentSBERTa_V2"
)

MAX_LENGTH = 512


# ============================================================
# 0.8 V2 run identity
# ============================================================

MODEL_VERSION = (
    "depth_ot_v2_patent_semantic"
)

RUN_TIMESTAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

RUN_NAME = (
    f"{MODEL_VERSION}"
    f"_seed{SEED}"
    f"_{RUN_TIMESTAMP}"
)

RUN_CHECKPOINT_DIR = (
    DIRS["depth_ot_v2_ckpt"]
    / RUN_NAME
)

RUN_LOG_DIR = (
    DIRS["depth_ot_v2_logs"]
    / RUN_NAME
)

RUN_RESULT_DIR = (
    DIRS["depth_ot_v2_results"]
    / RUN_NAME
)

for directory in [
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    RUN_RESULT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=False,
    )


# ============================================================
# 0.9 V2 configuration
# ============================================================

@dataclass
class DepthOTV2Config:

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    model_version: str = MODEL_VERSION
    run_name: str = RUN_NAME
    seed: int = SEED
    project_root: str = str(
        PROJECT_ROOT
    )

    # --------------------------------------------------------
    # Frozen PLM and features
    # --------------------------------------------------------

    feature_run_name: str = (
        FEATURE_RUN_NAME
    )

    plm_name: str = PLM_NAME
    max_length: int = MAX_LENGTH

    plm_hidden_dim: Optional[int] = None

    frozen_plm_dtype: str = (
        "bfloat16"
    )

    feature_storage_dtype: str = (
        "float16"
    )

    reuse_existing_features: bool = True

    # --------------------------------------------------------
    # Vocabulary
    # V1 used 2,000; V2 uses 8,000
    # --------------------------------------------------------

    vocab_size: int = 8000
    vocab_min_document_frequency: int = 10
    vocab_max_document_frequency_ratio: float = 0.70
    vocab_top_words_per_topic: int = 25

    # --------------------------------------------------------
    # Token graph encoder
    # --------------------------------------------------------

    knn_k: int = 10
    token_graph_temperature: float = 0.10

    gcn_hidden_dim: int = 256
    gcn_output_dim: int = 256
    gcn_num_layers: int = 2
    gcn_dropout: float = 0.10

    # --------------------------------------------------------
    # Dependency encoder
    # V2 uses normalized mean aggregation rather than sum
    # --------------------------------------------------------

    dependency_hidden_dim: int = 256
    dependency_num_layers: int = 2
    dependency_dropout: float = 0.10

    dependency_aggregation: str = (
        "mean"
    )

    dependency_use_residual: bool = True
    dependency_use_layer_norm: bool = True

    # --------------------------------------------------------
    # Patent and claim latent topics
    # --------------------------------------------------------

    num_topics: int = 30

    claim_representation_dim: int = 256
    patent_representation_dim: int = 256

    patent_pooling: str = (
        "attention"
    )

    patent_attention_hidden_dim: int = 128

    latent_hidden_dim: int = 256
    latent_dropout: float = 0.10

    # Claim posterior is conditioned on patent representation
    condition_claim_on_patent: bool = True

    # --------------------------------------------------------
    # Semantic topic-word decoder
    # --------------------------------------------------------

    topic_embedding_dim: int = 768

    use_fixed_vocab_embeddings: bool = True

    topic_word_temperature: float = 0.10

    topic_separation_margin: float = 0.20

    topic_word_residual_logits: bool = False

    # --------------------------------------------------------
    # Numerical constants
    # --------------------------------------------------------

    epsilon: float = 1e-8
    bow_epsilon: float = 1e-10

    # --------------------------------------------------------
    # Stable reference-regularized OT
    # --------------------------------------------------------

    ot_rho: float = 0.10
    ot_margin: float = 0.30
    ot_epsilon: float = 0.10

    sinkhorn_max_iterations: int = 300
    sinkhorn_min_iterations: int = 20
    sinkhorn_tolerance: float = 1e-6

    sinkhorn_compute_dtype: str = (
        "float64"
    )

    sinkhorn_log_domain: bool = True

    sinkhorn_check_every: int = 10

    # Final training must not silently skip invalid batches
    allow_hierarchy_batch_skip: bool = False

    # --------------------------------------------------------
    # Free topic anchors
    # V1 imposed topic-ID order; V2 learns each coordinate
    # --------------------------------------------------------

    anchor_parameterization: str = (
        "independent_sigmoid"
    )

    anchor_minimum_separation: float = 0.02

    anchor_repulsion_weight: float = 0.05

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_epochs: int = 24

    patents_per_batch: int = 8
    maximum_claims_per_batch: int = 256
    maximum_tokens_per_batch: int = 16384

    num_workers: int = 0
    pin_memory: bool = True

    learning_rate: float = 3e-4
    topic_learning_rate: float = 2e-4
    anchor_learning_rate: float = 1e-4

    weight_decay: float = 1e-5
    maximum_gradient_norm: float = 1.0

    use_amp: bool = False

    # --------------------------------------------------------
    # Three-stage training schedule
    # --------------------------------------------------------

    # Stage 1: semantic topics and reconstruction
    semantic_stage_start_epoch: int = 1

    # Stage 2: patent-level usage regularization
    usage_start_epoch: int = 5
    usage_full_epoch: int = 10

    # Stage 3: claim hierarchy and anchors
    hierarchy_start_epoch: int = 12
    hierarchy_full_epoch: int = 18
    anchor_start_epoch: int = 12

    # --------------------------------------------------------
    # KL schedule
    # --------------------------------------------------------

    kl_start_epoch: int = 3
    kl_full_epoch: int = 12
    kl_max_weight: float = 0.10

    # --------------------------------------------------------
    # Loss weights
    # --------------------------------------------------------

    # Claim BoW reconstruction
    reconstruction_weight: float = 1.00

    # Claim-to-patent semantic consistency
    claim_patent_consistency_weight: float = 0.10

    # Topic embedding and topic-word separation
    topic_separation_weight: float = 0.20

    # Mild dataset-level usage regularization
    # V1 weights 1.0–2.0 were too strong
    patent_usage_max_weight: float = 0.05

    # No mini-batch Sinkhorn balancing in V2
    use_batch_balanced_sinkhorn: bool = False
    balanced_assignment_weight: float = 0.0

    # Optional mild patent-level confidence
    patent_confidence_weight: float = 0.01

    # Hierarchy losses
    hierarchy_max_weight: float = 0.10
    reconciliation_weight: float = 0.05

    # --------------------------------------------------------
    # Dataset-level EMA topic usage
    # --------------------------------------------------------

    usage_ema_momentum: float = 0.99

    usage_target: str = (
        "uniform"
    )

    usage_minimum_probability: float = 1e-6

    # --------------------------------------------------------
    # Evaluation and model selection
    # --------------------------------------------------------

    full_dev_every_epochs: int = 2
    fast_dev_batches: int = 200

    checkpoint_interval_epochs: int = 2

    early_stopping_start_epoch: int = 12
    early_stopping_patience: int = 4

    # Usage metrics are diagnostics, not the sole criterion
    minimum_active_topics: int = 15
    maximum_topic_share: float = 0.35

    # Semantic-quality constraints
    minimum_topic_diversity: float = 0.30
    maximum_topic_cosine_similarity: float = 0.95

    # No CPC labels are used for checkpoint selection
    use_cpc_for_model_selection: bool = False

    # --------------------------------------------------------
    # Local feature cache
    # --------------------------------------------------------

    enable_local_feature_cache: bool = True

    local_feature_cache_dir: str = str(
        DIRS["local_feature_cache"]
    )

    minimum_free_local_gib: float = 8.0

    io_max_retries: int = 8
    io_retry_seconds: float = 3.0


CONFIG = DepthOTV2Config()


# ============================================================
# 0.10 Configuration validation
# ============================================================

def validate_configuration(config):
    if config.num_topics < 2:
        raise ValueError(
            "num_topics must be at least 2."
        )

    if config.vocab_size < (
        config.num_topics
        * config.vocab_top_words_per_topic
    ):
        raise ValueError(
            "Vocabulary is too small relative "
            "to topics and top-word count."
        )

    if config.patents_per_batch < 1:
        raise ValueError(
            "patents_per_batch must be positive."
        )

    if not (
        1
        <= config.kl_start_epoch
        <= config.kl_full_epoch
        <= config.num_epochs
    ):
        raise ValueError(
            "Invalid KL schedule."
        )

    if not (
        1
        <= config.usage_start_epoch
        <= config.usage_full_epoch
        <= config.num_epochs
    ):
        raise ValueError(
            "Invalid usage schedule."
        )

    if not (
        1
        <= config.hierarchy_start_epoch
        <= config.hierarchy_full_epoch
        <= config.num_epochs
    ):
        raise ValueError(
            "Invalid hierarchy schedule."
        )

    if (
        config.anchor_start_epoch
        < config.hierarchy_start_epoch
    ):
        raise ValueError(
            "Anchor training cannot start before "
            "hierarchy training."
        )

    if config.use_batch_balanced_sinkhorn:
        raise ValueError(
            "V2 must not use mini-batch balanced "
            "Sinkhorn assignments."
        )

    if config.sinkhorn_compute_dtype not in {
        "float32",
        "float64",
    }:
        raise ValueError(
            "Invalid Sinkhorn compute dtype."
        )

    if config.use_cpc_for_model_selection:
        raise ValueError(
            "CPC labels must not be used for "
            "unsupervised model selection."
        )


validate_configuration(
    CONFIG
)


# ============================================================
# 0.11 Check existing feature shards
# ============================================================

FEATURE_SHARD_COUNTS = {}

for split_name, split_directory in (
    FEATURE_SPLIT_DIRS.items()
):
    if not split_directory.is_dir():
        raise FileNotFoundError(
            f"Feature directory가 없습니다: "
            f"{split_directory}"
        )

    shard_files = sorted([
        *split_directory.glob("*.pt"),
        *split_directory.glob("*.pth"),
    ])

    FEATURE_SHARD_COUNTS[
        split_name
    ] = len(shard_files)

    if len(shard_files) == 0:
        raise FileNotFoundError(
            f"{split_name} feature shard가 없습니다: "
            f"{split_directory}"
        )


# ============================================================
# 0.12 Save run configuration
# ============================================================

CONFIG_PATH = (
    RUN_LOG_DIR
    / "section0_configuration.json"
)

ENVIRONMENT_PATH = (
    RUN_LOG_DIR
    / "section0_environment.json"
)

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        asdict(CONFIG),
        file,
        indent=2,
        ensure_ascii=False,
    )

environment_payload = {
    "created_at": (
        datetime.now().isoformat()
    ),
    "python": (
        sys.version
    ),
    "pytorch": (
        torch.__version__
    ),
    "pytorch_cuda": (
        torch.version.cuda
    ),
    "gpu": GPU_NAME,
    "gpu_total_gib": (
        GPU_TOTAL_GIB
    ),
    "cuda_capability": (
        f"{CUDA_CAPABILITY[0]}."
        f"{CUDA_CAPABILITY[1]}"
    ),
    "seed": SEED,
    "strict_reproducibility": (
        STRICT_REPRODUCIBILITY
    ),
    "tf32": ENABLE_TF32,
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "feature_shard_counts": (
        FEATURE_SHARD_COUNTS
    ),
}

with open(
    ENVIRONMENT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        environment_payload,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 0.13 Final status
# ============================================================

print("\n" + "=" * 88)
print("SECTION 0 — DEPTH-OT V2 COMPLETED")
print("=" * 88)
print(f"Run name         : {RUN_NAME}")
print(f"Model version    : {MODEL_VERSION}")
print(f"Seed             : {SEED}")
print(f"Device           : {DEVICE}")
print(f"GPU              : {GPU_NAME}")
print(f"Strict reproduce : {STRICT_REPRODUCIBILITY}")
print(f"TF32             : {ENABLE_TF32}")
print(f"Feature run      : {FEATURE_RUN_NAME}")
print(
    f"Feature shards   : "
    f"train={FEATURE_SHARD_COUNTS['train']}, "
    f"dev={FEATURE_SHARD_COUNTS['dev']}, "
    f"test={FEATURE_SHARD_COUNTS['test']}"
)
print(f"Topics           : {CONFIG.num_topics}")
print(f"Vocabulary target: {CONFIG.vocab_size:,}")
print(
    f"Batch balance    : "
    f"{CONFIG.use_batch_balanced_sinkhorn}"
)
print(
    f"Patent usage     : "
    f"EMA, max weight="
    f"{CONFIG.patent_usage_max_weight}"
)
print(
    f"Hierarchy start  : "
    f"Epoch {CONFIG.hierarchy_start_epoch}"
)
print(
    f"Sinkhorn dtype   : "
    f"{CONFIG.sinkhorn_compute_dtype}"
)
print(
    f"Sinkhorn max iter: "
    f"{CONFIG.sinkhorn_max_iterations}"
)
print(f"Checkpoint dir   : {RUN_CHECKPOINT_DIR}")
print(f"Log dir          : {RUN_LOG_DIR}")
print(f"Result dir       : {RUN_RESULT_DIR}")
print(f"Config saved     : {CONFIG_PATH}")
print("=" * 88)

print(
    "\n[PASS] 기존 feature는 변경하거나 삭제하지 않았습니다."
)
print(
    "[NEXT] 출력 결과를 확인한 뒤 Section 1을 실행합니다."
)
