# ============================================================
# SECTION 0: Environment and Configuration
# ============================================================

# ------------------------------------------------------------
# 0.0 Environment variables
# These must be set before importing PyTorch.
# ------------------------------------------------------------
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "42")

# ------------------------------------------------------------
# 0.1 Standard imports
# ------------------------------------------------------------
import sys
import json
import time
import random
import subprocess
import importlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ------------------------------------------------------------
# 0.2 Google Drive mount
# ------------------------------------------------------------
from google.colab import drive

drive.mount("/content/drive", force_remount=False)

PROJECT_ROOT = "/content/drive/MyDrive/depth_ot_patent"

DIRS = {
    "raw": f"{PROJECT_ROOT}/data/raw",
    "processed": f"{PROJECT_ROOT}/data/processed",
    "checkpoints": f"{PROJECT_ROOT}/checkpoints",
    "logs": f"{PROJECT_ROOT}/logs",
    "results": f"{PROJECT_ROOT}/results",
    "notebooks": f"{PROJECT_ROOT}/notebooks",
}

DIRS.update({
    "token_features": f"{DIRS['processed']}/token_features",
    "depth_ot_ckpt": f"{DIRS['checkpoints']}/depth_ot",
    "depth_ot_logs": f"{DIRS['logs']}/depth_ot",
    "depth_ot_results": f"{DIRS['results']}/depth_ot",
    "depth_ot_cache": f"{PROJECT_ROOT}/cache/depth_ot",
    "hf_cache": f"{PROJECT_ROOT}/cache/huggingface",
    "local_cache": "/content/depth_ot_cache",
})

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)

os.environ.setdefault("HF_HOME", DIRS["hf_cache"])

print("=== PROJECT DIRECTORIES ===")
for name, path in DIRS.items():
    print(f"{name:20s} -> {path}")

# ------------------------------------------------------------
# 0.3 Check the preinstalled PyTorch and CUDA environment
# ------------------------------------------------------------
import torch

print("\n=== PREINSTALLED PYTORCH ENVIRONMENT ===")
print(f"Python version       : {sys.version.split()[0]}")
print(f"PyTorch version      : {torch.__version__}")
print(f"PyTorch CUDA version : {torch.version.cuda}")
print(f"CUDA available       : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_idx = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(gpu_idx)
    gpu_capability = torch.cuda.get_device_capability(gpu_idx)

    print(f"GPU                   : {torch.cuda.get_device_name(gpu_idx)}")
    print(
        "Compute capability    : "
        f"{gpu_capability[0]}.{gpu_capability[1]}"
    )
    print(
        "Total GPU memory      : "
        f"{gpu_props.total_memory / (1024 ** 3):.2f} GiB"
    )
else:
    print(
        "WARNING: CUDA is unavailable. "
        "Enable a GPU runtime before feature extraction or training."
    )

# ------------------------------------------------------------
# 0.4 Install only missing packages
# PyTorch is intentionally not reinstalled.
# ------------------------------------------------------------
PACKAGE_IMPORTS = {
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
    "einops": "einops",
    "tqdm": "tqdm",
}

missing_packages = []

print("\n=== PACKAGE CHECK ===")
for pip_name, import_name in PACKAGE_IMPORTS.items():
    try:
        importlib.import_module(import_name)
        print(f"[OK] {pip_name}")
    except ImportError:
        print(f"[MISSING] {pip_name}")
        missing_packages.append(pip_name)

if missing_packages:
    print(f"\nInstalling: {missing_packages}")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        *missing_packages,
    ])

    importlib.invalidate_caches()

    for pip_name, import_name in PACKAGE_IMPORTS.items():
        importlib.import_module(import_name)

    print("Package installation and import verification completed.")
else:
    print("All required packages are already installed.")

# ------------------------------------------------------------
# 0.5 Project imports
# ------------------------------------------------------------
import math
import hashlib

import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

print(f"\nTransformers version: {transformers.__version__}")

# ------------------------------------------------------------
# 0.6 Reproducibility
# ------------------------------------------------------------
SEED = 42


def set_seed(seed: int) -> None:
    """
    Fix random seeds for Python, NumPy, and PyTorch.

    PYTHONHASHSEED is also recorded, but changing it after the Python
    process starts cannot retroactively change the current hash state.
    Code that depends on dictionary or set ordering should use sorted().
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """
    Deterministically seed each DataLoader worker.
    """
    del worker_id

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


set_seed(SEED)

DATALOADER_GENERATOR = torch.Generator()
DATALOADER_GENERATOR.manual_seed(SEED)

# Deterministic cuDNN behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable TF32 for stricter reproducibility
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# warn_only=True prevents unsupported deterministic operations from
# terminating the entire run. Any warning should still be inspected.
torch.use_deterministic_algorithms(True, warn_only=True)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

print("\n=== REPRODUCIBILITY SETTINGS ===")
print(f"Seed                         : {SEED}")
print(
    "CUBLAS_WORKSPACE_CONFIG      : "
    f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
)
print(
    "TOKENIZERS_PARALLELISM       : "
    f"{os.environ.get('TOKENIZERS_PARALLELISM')}"
)
print(
    "cuDNN deterministic          : "
    f"{torch.backends.cudnn.deterministic}"
)
print(
    "cuDNN benchmark              : "
    f"{torch.backends.cudnn.benchmark}"
)
print("Deterministic algorithms     : enabled with warn_only=True")

# ------------------------------------------------------------
# 0.7 Device and dtype configuration
# ------------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type == "cuda":
    bf16_supported = (
        torch.cuda.is_bf16_supported()
        if hasattr(torch.cuda, "is_bf16_supported")
        else False
    )

    TRAIN_AMP_DTYPE = (
        torch.bfloat16
        if bf16_supported
        else torch.float16
    )

    PLM_COMPUTE_DTYPE = TRAIN_AMP_DTYPE
else:
    bf16_supported = False
    TRAIN_AMP_DTYPE = torch.float32
    PLM_COMPUTE_DTYPE = torch.float32

# Token features are always stored as FP16 to reduce persistent storage.
FEATURE_STORAGE_DTYPE = torch.float16


def dtype_to_string(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


print("\n=== DEVICE AND DTYPE SETTINGS ===")
print(f"Device                   : {DEVICE}")
print(f"BF16 supported           : {bf16_supported}")
print(f"Training AMP dtype       : {TRAIN_AMP_DTYPE}")
print(f"Frozen PLM compute dtype : {PLM_COMPUTE_DTYPE}")
print(f"Feature storage dtype    : {FEATURE_STORAGE_DTYPE}")

# ------------------------------------------------------------
# 0.8 Frozen PLM
# ------------------------------------------------------------
PLM_NAME = "AAUBS/PatentSBERTa_V2"

print("\n=== FROZEN PLM ===")
print(f"Model name: {PLM_NAME}")
print(
    "Section 1 must use AutoModel.last_hidden_state rather than "
    "a pooled sentence-transformer embedding."
)

# ------------------------------------------------------------
# 0.9 Main configuration
# ------------------------------------------------------------
@dataclass
class DepthOTConfig:
    # General
    seed: int = SEED
    project_root: str = PROJECT_ROOT

    # Frozen PLM
    plm_name: str = PLM_NAME
    max_length: int = 512
    plm_hidden_dim: Optional[int] = None
    plm_compute_dtype: str = "float32"
    feature_storage_dtype: str = "float16"

    # Frozen feature extraction
    feature_batch_size: int = 16
    feature_shard_size: int = 1000

    # Token graph
    knn_k: int = 10
    token_graph_tau: float = 0.1

    # Token GCN and claim representation
    gcn_hidden_dim: int = 256
    gcn_num_layers: int = 2
    claim_repr_dim: int = 256
    gcn_dropout: float = 0.1

    # Dependency encoder
    dep_encoder_hidden_dim: int = 256
    dep_encoder_dropout: float = 0.1

    # Topic model
    num_topics: int = 30
    vocab_size: Optional[int] = None
    eps_x: float = 1e-10

    # KL warmup
    kl_warmup_steps: int = 5000
    kl_max_weight: float = 1.0

    # Reference-regularized OT
    ot_rho: float = 0.1
    ot_margin: float = 0.5
    ot_epsilon: float = 0.1
    sinkhorn_iters: int = 50
    sinkhorn_tolerance: float = 1e-5
    sinkhorn_log_domain: bool = True

    # Hierarchy loss
    hierarchy_warmup_steps: int = 5000
    hierarchy_max_weight: float = 1.0

    # Topic diversity
    diversity_weight: float = 0.05

    # Hierarchy extraction
    hierarchy_eps: float = 1e-10

    # Dynamic patent-level batching
    patents_per_batch: int = 16
    max_claims_per_batch: int = 256
    max_tokens_per_batch: int = 16384
    num_workers: int = 2
    pin_memory: bool = True

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 30
    early_stopping_patience: int = 5
    grad_clip_norm: float = 5.0

    # Training mixed precision
    training_amp_dtype: str = "float32"

    # Logging and checkpoints
    log_every_steps: int = 100
    validate_every_epochs: int = 1
    save_every_epochs: int = 1

    def to_dict(self):
        return asdict(self)


CONFIG = DepthOTConfig(
    plm_compute_dtype=dtype_to_string(PLM_COMPUTE_DTYPE),
    feature_storage_dtype=dtype_to_string(FEATURE_STORAGE_DTYPE),
    training_amp_dtype=dtype_to_string(TRAIN_AMP_DTYPE),
)

# ------------------------------------------------------------
# 0.10 Configuration validation
# ------------------------------------------------------------
def validate_config(config: DepthOTConfig) -> None:
    assert config.num_topics >= 2
    assert config.max_length >= 2
    assert config.knn_k >= 1
    assert config.gcn_num_layers >= 1

    assert 0.0 < config.token_graph_tau
    assert 0.0 < config.ot_rho < 1.0
    assert 0.0 < config.ot_margin <= 1.0
    assert 0.0 < config.ot_epsilon

    assert config.sinkhorn_iters >= 1
    assert config.sinkhorn_tolerance > 0.0

    assert config.patents_per_batch >= 1
    assert config.max_claims_per_batch >= 1
    assert config.max_tokens_per_batch >= 1

    assert config.feature_batch_size >= 1
    assert config.feature_shard_size >= 1

    assert config.learning_rate > 0.0
    assert config.weight_decay >= 0.0
    assert config.num_epochs >= 1
    assert config.early_stopping_patience >= 1

    assert config.kl_warmup_steps >= 0
    assert config.hierarchy_warmup_steps >= 0
    assert config.kl_max_weight >= 0.0
    assert config.hierarchy_max_weight >= 0.0
    assert config.diversity_weight >= 0.0


validate_config(CONFIG)

print("\n=== CONFIGURATION ===")
for key, value in CONFIG.to_dict().items():
    print(f"{key:30s}: {value}")

# ------------------------------------------------------------
# 0.11 Directory existence and writability checks
# ------------------------------------------------------------
def verify_writable(path: str) -> bool:
    path_obj = Path(path)
    test_path = path_obj / ".depth_ot_write_test.tmp"

    try:
        path_obj.mkdir(parents=True, exist_ok=True)

        with test_path.open("w", encoding="utf-8") as file:
            file.write("ok")

        test_path.unlink()
        return True

    except Exception as error:
        print(f"Write test failed for {path}: {error}")
        return False


print("\n=== DIRECTORY CHECK ===")

directory_status = {}
all_directories_ok = True

for name, path in DIRS.items():
    exists = os.path.isdir(path)
    writable = verify_writable(path) if exists else False

    directory_status[name] = {
        "path": path,
        "exists": exists,
        "writable": writable,
    }

    status = "OK" if exists and writable else "FAIL"
    all_directories_ok &= exists and writable

    print(
        f"[{status:4s}] "
        f"{name:20s} "
        f"exists={str(exists):5s} "
        f"writable={str(writable):5s}"
    )

if not all_directories_ok:
    raise RuntimeError(
        "One or more required directories are unavailable or not writable."
    )

# ------------------------------------------------------------
# 0.12 Environment manifest
# ------------------------------------------------------------
if torch.cuda.is_available():
    gpu_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_idx)
    gpu_capability = torch.cuda.get_device_capability(gpu_idx)
    gpu_memory_gib = (
        torch.cuda.get_device_properties(gpu_idx).total_memory
        / (1024 ** 3)
    )
else:
    gpu_name = None
    gpu_capability = None
    gpu_memory_gib = None

ENVIRONMENT_INFO = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": gpu_name,
    "gpu_capability": gpu_capability,
    "gpu_memory_gib": gpu_memory_gib,
    "device": str(DEVICE),
    "bf16_supported": bf16_supported,
    "training_amp_dtype": dtype_to_string(TRAIN_AMP_DTYPE),
    "plm_compute_dtype": dtype_to_string(PLM_COMPUTE_DTYPE),
    "feature_storage_dtype": dtype_to_string(
        FEATURE_STORAGE_DTYPE
    ),
    "cublas_workspace_config": os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ),
    "tokenizers_parallelism": os.environ.get(
        "TOKENIZERS_PARALLELISM"
    ),
}

RUN_MANIFEST = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": CONFIG.to_dict(),
    "environment": ENVIRONMENT_INFO,
    "directories": DIRS,
}

# ------------------------------------------------------------
# 0.13 Save configuration and run manifest
# ------------------------------------------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")

config_path = os.path.join(
    DIRS["depth_ot_logs"],
    f"config_{timestamp}.json",
)

manifest_path = os.path.join(
    DIRS["depth_ot_logs"],
    f"manifest_{timestamp}.json",
)

latest_config_path = os.path.join(
    DIRS["depth_ot_logs"],
    "config_latest.json",
)

latest_manifest_path = os.path.join(
    DIRS["depth_ot_logs"],
    "manifest_latest.json",
)

with open(config_path, "w", encoding="utf-8") as file:
    json.dump(CONFIG.to_dict(), file, indent=2)

with open(latest_config_path, "w", encoding="utf-8") as file:
    json.dump(CONFIG.to_dict(), file, indent=2)

with open(manifest_path, "w", encoding="utf-8") as file:
    json.dump(RUN_MANIFEST, file, indent=2)

with open(latest_manifest_path, "w", encoding="utf-8") as file:
    json.dump(RUN_MANIFEST, file, indent=2)

print("\n=== SAVED CONFIGURATION ===")
print(f"Timestamped config : {config_path}")
print(f"Latest config      : {latest_config_path}")
print(f"Timestamped manifest: {manifest_path}")
print(f"Latest manifest    : {latest_manifest_path}")

# ------------------------------------------------------------
# 0.14 Reload verification
# ------------------------------------------------------------
with open(latest_config_path, "r", encoding="utf-8") as file:
    reloaded_config = json.load(file)

with open(latest_manifest_path, "r", encoding="utf-8") as file:
    reloaded_manifest = json.load(file)

assert reloaded_config["seed"] == CONFIG.seed
assert reloaded_config["num_topics"] == CONFIG.num_topics
assert reloaded_config["plm_name"] == CONFIG.plm_name

assert (
    reloaded_manifest["environment"]["torch"]
    == torch.__version__
)
assert (
    reloaded_manifest["environment"]["transformers"]
    == transformers.__version__
)

print("Configuration and manifest reload verification passed.")

# ------------------------------------------------------------
# 0.15 Final summary
# ------------------------------------------------------------
print("\n" + "=" * 68)
print("SECTION 0 SUMMARY")
print("=" * 68)
print(f"Project root             : {PROJECT_ROOT}")
print(f"Device                   : {DEVICE}")
print(f"GPU                      : {gpu_name}")
print(f"Training AMP dtype       : {TRAIN_AMP_DTYPE}")
print(f"Frozen PLM compute dtype : {PLM_COMPUTE_DTYPE}")
print(f"Feature storage dtype    : {FEATURE_STORAGE_DTYPE}")
print(f"Frozen PLM               : {PLM_NAME}")
print(f"Random seed              : {CONFIG.seed}")
print(f"Number of topics         : {CONFIG.num_topics}")
print(f"Maximum sequence length  : {CONFIG.max_length}")
print(f"All directories writable : {all_directories_ok}")
print(f"Latest config            : {latest_config_path}")
print(f"Latest manifest          : {latest_manifest_path}")
print("=" * 68)

if DEVICE.type != "cuda":
    print(
        "WARNING: Section 0 completed on CPU, but a GPU runtime is "
        "strongly recommended for Sections 1 and 6."
    )
else:
    print(
        "Section 0 completed successfully. "
        "Proceed to Section 1: Frozen Token Feature Extraction."
    )


# ============================================================
# SECTION 1: Frozen Token Feature Extraction
# ============================================================

import os
import gc
import json
import math
import time
import pickle
from contextlib import nullcontext
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------
# 1.0 Run configuration
# ------------------------------------------------------------

# Start with a small debug run. Set to None only after verification.
DEBUG_MAX_PATENTS = None
#DEBUG_MAX_PATENTS = 20

# Smaller shards reduce corruption risk and make random access easier.
SHARD_SIZE_PATENTS = 100

# Use the Section 0 batch-size setting.
EXTRACTION_BATCH_SIZE = CONFIG.feature_batch_size

# CPU tokenization chunk size for truncation statistics.
STATS_TOKENIZE_CHUNK = 1000

# Synchronize the effective shard size with CONFIG.
CONFIG.feature_shard_size = SHARD_SIZE_PATENTS

FEATURE_RUN_NAME = (
    f"debug_{DEBUG_MAX_PATENTS}"
    if DEBUG_MAX_PATENTS is not None
    else "full"
)

FEATURE_ROOT = os.path.join(
    DIRS["token_features"],
    FEATURE_RUN_NAME,
)

for split in ["train", "dev", "test"]:
    os.makedirs(
        os.path.join(FEATURE_ROOT, split),
        exist_ok=True,
    )

print("=== SECTION 1 RUN CONFIGURATION ===")
print(f"Run name               : {FEATURE_RUN_NAME}")
print(f"Feature root           : {FEATURE_ROOT}")
print(f"Debug max patents      : {DEBUG_MAX_PATENTS}")
print(f"Patents per shard      : {SHARD_SIZE_PATENTS}")
print(f"Claims per PLM batch   : {EXTRACTION_BATCH_SIZE}")
print(f"PLM compute dtype      : {PLM_COMPUTE_DTYPE}")
print(f"Feature storage dtype  : {FEATURE_STORAGE_DTYPE}")

if DEBUG_MAX_PATENTS is None:
    print(
        "WARNING: Full extraction mode is enabled. "
        "Confirm debug runtime and storage estimates first."
    )

# ------------------------------------------------------------
# 1.1 Frozen PLM loading
# ------------------------------------------------------------

print(f"\nLoading frozen PLM: {PLM_NAME}")

plm_tokenizer = AutoTokenizer.from_pretrained(
    PLM_NAME,
    cache_dir=DIRS["hf_cache"],
)

plm_model = AutoModel.from_pretrained(
    PLM_NAME,
    cache_dir=DIRS["hf_cache"],
)

plm_model.to(DEVICE)
plm_model.eval()
plm_model.requires_grad_(False)

CONFIG.plm_hidden_dim = int(plm_model.config.hidden_size)

model_revision = getattr(
    plm_model.config,
    "_commit_hash",
    None,
)

print(f"Hidden dimension       : {CONFIG.plm_hidden_dim}")
print(f"Model revision         : {model_revision}")
print(f"Model parameter dtype  : {next(plm_model.parameters()).dtype}")
print(f"Model training mode    : {plm_model.training}")
print(
    "Any trainable parameter: "
    f"{any(p.requires_grad for p in plm_model.parameters())}"
)

assert plm_model.training is False
assert not any(
    parameter.requires_grad
    for parameter in plm_model.parameters()
)

# Update the latest configuration after resolving the PLM dimension.
with open(
    latest_config_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        CONFIG.to_dict(),
        file,
        indent=2,
    )

# ------------------------------------------------------------
# 1.2 Source record paths
# ------------------------------------------------------------

RECORD_PATHS = {
    "train": os.path.join(
        DIRS["processed"],
        "train_records.pkl",
    ),
    "dev": os.path.join(
        DIRS["processed"],
        "dev_records.pkl",
    ),
    "test": os.path.join(
        DIRS["processed"],
        "test_records.pkl",
    ),
}

for split, record_path in RECORD_PATHS.items():
    if not os.path.isfile(record_path):
        raise FileNotFoundError(
            f"Missing processed records for {split}: {record_path}"
        )

    print(
        f"[OK] {split:5s} records: "
        f"{record_path} "
        f"({os.path.getsize(record_path) / 1e6:.1f} MB)"
    )


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


# ------------------------------------------------------------
# 1.3 Longest-path depth verification
# ------------------------------------------------------------

def recompute_longest_path_depth(claim_ids, edges):
    """
    Recompute longest-path depth in a claim dependency DAG.

    Returns:
        depth: dict mapping claim ID to longest-path depth
        is_dag: whether every claim was processed
    """
    claim_ids = list(claim_ids)
    claim_set = set(claim_ids)

    parents = defaultdict(list)
    children = defaultdict(list)

    for parent_id, child_id in edges:
        if parent_id not in claim_set:
            raise KeyError(
                f"Unknown parent claim ID in edge: {parent_id}"
            )

        if child_id not in claim_set:
            raise KeyError(
                f"Unknown child claim ID in edge: {child_id}"
            )

        parents[child_id].append(parent_id)
        children[parent_id].append(child_id)

    indegree = {
        claim_id: len(parents[claim_id])
        for claim_id in claim_ids
    }

    depth = {
        claim_id: 0
        for claim_id in claim_ids
    }

    queue = deque(
        claim_id
        for claim_id in claim_ids
        if indegree[claim_id] == 0
    )

    processed = 0

    while queue:
        current = queue.popleft()
        processed += 1

        for child_id in children[current]:
            depth[child_id] = max(
                depth[child_id],
                depth[current] + 1,
            )

            indegree[child_id] -= 1

            if indegree[child_id] == 0:
                queue.append(child_id)

    is_dag = processed == len(claim_ids)

    return depth, is_dag


def verify_longest_path_depth(records, split_name):
    """
    Verify that stored depths match the longest-path definition and
    that every dependency edge follows increasing depth.
    """
    mismatch_patents = []
    cycle_patents = []
    invariant_violations = []
    invalid_records = []

    for record in tqdm(
        records,
        desc=f"{split_name} depth-check",
        leave=False,
    ):
        patent_id = record.get("patent_id")

        try:
            claims = record["claims"]
            edges = record["edges"]
            stored_depth = record["depth"]

            claim_ids = list(claims.keys())

            recomputed_depth, is_dag = (
                recompute_longest_path_depth(
                    claim_ids,
                    edges,
                )
            )

            if not is_dag:
                cycle_patents.append(patent_id)
                continue

            if recomputed_depth != stored_depth:
                mismatch_patents.append(patent_id)

            for parent_id, child_id in edges:
                if not (
                    recomputed_depth[child_id]
                    >
                    recomputed_depth[parent_id]
                ):
                    invariant_violations.append({
                        "patent_id": patent_id,
                        "parent_id": parent_id,
                        "child_id": child_id,
                        "parent_depth": recomputed_depth[parent_id],
                        "child_depth": recomputed_depth[child_id],
                    })

        except Exception as error:
            invalid_records.append({
                "patent_id": patent_id,
                "error": repr(error),
            })

    report = {
        "split": split_name,
        "checked_patents": len(records),
        "depth_mismatch_count": len(mismatch_patents),
        "cycle_count": len(cycle_patents),
        "invariant_violation_count": len(
            invariant_violations
        ),
        "invalid_record_count": len(invalid_records),
        "depth_mismatch_examples": mismatch_patents[:50],
        "cycle_examples": cycle_patents[:50],
        "invariant_violation_examples": (
            invariant_violations[:50]
        ),
        "invalid_record_examples": invalid_records[:50],
    }

    return report


def assert_valid_depth_report(report):
    invalid_count = (
        report["depth_mismatch_count"]
        + report["cycle_count"]
        + report["invariant_violation_count"]
        + report["invalid_record_count"]
    )

    if invalid_count > 0:
        raise RuntimeError(
            f"Depth verification failed for {report['split']}. "
            f"mismatch={report['depth_mismatch_count']}, "
            f"cycles={report['cycle_count']}, "
            f"violations={report['invariant_violation_count']}, "
            f"invalid={report['invalid_record_count']}. "
            "Regenerate the processed records using longest-path depth "
            "before continuing."
        )


# ------------------------------------------------------------
# 1.4 Streaming truncation statistics
# ------------------------------------------------------------

def iter_claims(records):
    """
    Yield patent ID, claim ID, and text without creating a full
    corpus-level text list.
    """
    for record in records:
        patent_id = record["patent_id"]

        for claim_id, text in record["claims"].items():
            yield (
                patent_id,
                claim_id,
                text if text is not None else "",
            )


def process_truncation_chunk(
    tokenizer,
    keys,
    texts,
    max_length,
    truncated_keys,
):
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    raw_lengths = [
        len(input_ids)
        for input_ids in encoded["input_ids"]
    ]

    for key, raw_length in zip(keys, raw_lengths):
        if raw_length > max_length:
            truncated_keys.add(key)

    return raw_lengths


def compute_truncation_stats(
    records,
    tokenizer,
    max_length,
    chunk_size,
    split_name,
):
    """
    Compute truncation statistics without retaining all claim texts or
    all token lengths in memory.

    Only the keys of truncated claims are retained.
    """
    truncated_keys = set()

    current_keys = []
    current_texts = []

    total_claims = 0
    total_raw_tokens = 0
    max_raw_length = 0

    claim_iterator = iter_claims(records)

    progress = tqdm(
        claim_iterator,
        desc=f"{split_name} truncation-stats",
        leave=False,
    )

    for patent_id, claim_id, text in progress:
        current_keys.append(
            (patent_id, claim_id)
        )
        current_texts.append(text)

        if len(current_texts) >= chunk_size:
            raw_lengths = process_truncation_chunk(
                tokenizer=tokenizer,
                keys=current_keys,
                texts=current_texts,
                max_length=max_length,
                truncated_keys=truncated_keys,
            )

            total_claims += len(raw_lengths)
            total_raw_tokens += sum(raw_lengths)

            if raw_lengths:
                max_raw_length = max(
                    max_raw_length,
                    max(raw_lengths),
                )

            current_keys.clear()
            current_texts.clear()

    if current_texts:
        raw_lengths = process_truncation_chunk(
            tokenizer=tokenizer,
            keys=current_keys,
            texts=current_texts,
            max_length=max_length,
            truncated_keys=truncated_keys,
        )

        total_claims += len(raw_lengths)
        total_raw_tokens += sum(raw_lengths)

        if raw_lengths:
            max_raw_length = max(
                max_raw_length,
                max(raw_lengths),
            )

    truncated_count = len(truncated_keys)

    report = {
        "total_claims": total_claims,
        "truncated_claims": truncated_count,
        "truncated_ratio": (
            truncated_count / total_claims
            if total_claims > 0
            else 0.0
        ),
        "mean_raw_length": (
            total_raw_tokens / total_claims
            if total_claims > 0
            else 0.0
        ),
        "max_raw_length": max_raw_length,
        "max_length": max_length,
    }

    return truncated_keys, report


# ------------------------------------------------------------
# 1.5 Batched sparse kNN token-graph construction
# ------------------------------------------------------------

def build_batched_knn_graphs(
    token_embeddings,
    valid_token_mask,
    k,
    tau,
):
    """
    Construct symmetric sparse kNN graphs for a token batch.

    Args:
        token_embeddings:
            Tensor of shape [B, L, H].
        valid_token_mask:
            Boolean tensor of shape [B, L]. True for non-padding,
            non-special tokens.
        k:
            Number of directed nearest neighbors before symmetric union.
        tau:
            Exponential edge-weight temperature.

    Returns:
        List of dictionaries with:
            token_embeddings
            edge_index
            edge_weight
            num_tokens
    """
    batch_size, sequence_length, _ = (
        token_embeddings.shape
    )

    normalized = F.normalize(
        token_embeddings.float(),
        p=2,
        dim=-1,
    )

    similarity = torch.bmm(
        normalized,
        normalized.transpose(1, 2),
    )

    pair_mask = (
        valid_token_mask.unsqueeze(2)
        & valid_token_mask.unsqueeze(1)
    )

    similarity_for_topk = similarity.masked_fill(
        ~pair_mask,
        -float("inf"),
    )

    diagonal_mask = torch.eye(
        sequence_length,
        dtype=torch.bool,
        device=token_embeddings.device,
    ).unsqueeze(0)

    similarity_for_topk = similarity_for_topk.masked_fill(
        diagonal_mask,
        -float("inf"),
    )

    if sequence_length > 1:
        batch_k = min(
            k,
            sequence_length - 1,
        )

        topk_values, topk_indices = torch.topk(
            similarity_for_topk,
            k=batch_k,
            dim=-1,
        )
    else:
        batch_k = 0
        topk_values = None
        topk_indices = None

    graph_outputs = []

    for batch_index in range(batch_size):
        valid_positions = torch.nonzero(
            valid_token_mask[batch_index],
            as_tuple=False,
        ).squeeze(1)

        num_tokens = int(valid_positions.numel())

        if num_tokens == 0:
            raise ValueError(
                "A claim produced no valid non-special tokens."
            )

        real_embeddings = token_embeddings[
            batch_index,
            valid_positions,
        ]

        if num_tokens == 1:
            edge_index = torch.tensor(
                [[0], [0]],
                dtype=torch.long,
            )

            edge_weight = torch.tensor(
                [1.0],
                dtype=FEATURE_STORAGE_DTYPE,
            )

            graph_outputs.append({
                "token_embeddings": (
                    real_embeddings.detach()
                    .to(
                        dtype=FEATURE_STORAGE_DTYPE,
                        device="cpu",
                    )
                    .contiguous()
                ),
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "num_tokens": num_tokens,
            })

            continue

        local_map = torch.full(
            (sequence_length,),
            -1,
            dtype=torch.long,
            device=token_embeddings.device,
        )

        local_map[valid_positions] = torch.arange(
            num_tokens,
            device=token_embeddings.device,
        )

        row_positions = valid_positions

        row_topk_values = topk_values[
            batch_index,
            row_positions,
        ]

        row_topk_indices = topk_indices[
            batch_index,
            row_positions,
        ]

        finite_neighbor_mask = torch.isfinite(
            row_topk_values
        )

        source_positions = (
            row_positions.unsqueeze(1)
            .expand(-1, batch_k)
        )[finite_neighbor_mask]

        target_positions = row_topk_indices[
            finite_neighbor_mask
        ]

        source_local = local_map[source_positions]
        target_local = local_map[target_positions]

        valid_neighbor_pairs = (
            (source_local >= 0)
            & (target_local >= 0)
            & (source_local != target_local)
        )

        source_local = source_local[
            valid_neighbor_pairs
        ]

        target_local = target_local[
            valid_neighbor_pairs
        ]

        # Symmetric union of directed kNN relations.
        all_source = torch.cat([
            source_local,
            target_local,
        ])

        all_target = torch.cat([
            target_local,
            source_local,
        ])

        pair_codes = (
            all_source * num_tokens
            + all_target
        )

        unique_codes = torch.unique(
            pair_codes,
            sorted=True,
        )

        graph_source = torch.div(
            unique_codes,
            num_tokens,
            rounding_mode="floor",
        )

        graph_target = unique_codes % num_tokens

        real_normalized = normalized[
            batch_index,
            valid_positions,
        ]

        graph_similarity = (
            real_normalized[graph_source]
            * real_normalized[graph_target]
        ).sum(dim=-1)

        graph_weight = torch.exp(
            graph_similarity / tau
        )

        self_index = torch.arange(
            num_tokens,
            device=token_embeddings.device,
        )

        graph_source = torch.cat([
            graph_source,
            self_index,
        ])

        graph_target = torch.cat([
            graph_target,
            self_index,
        ])

        graph_weight = torch.cat([
            graph_weight,
            torch.ones(
                num_tokens,
                dtype=graph_weight.dtype,
                device=graph_weight.device,
            ),
        ])

        edge_index = torch.stack([
            graph_source,
            graph_target,
        ]).to(
            dtype=torch.long,
            device="cpu",
        ).contiguous()

        edge_weight = graph_weight.detach().to(
            dtype=FEATURE_STORAGE_DTYPE,
            device="cpu",
        ).contiguous()

        if not torch.isfinite(
            edge_weight.float()
        ).all():
            raise FloatingPointError(
                "Non-finite kNN edge weight detected. "
                "Check token_graph_tau."
            )

        stored_embeddings = (
            real_embeddings.detach()
            .to(
                dtype=FEATURE_STORAGE_DTYPE,
                device="cpu",
            )
            .contiguous()
        )

        if not torch.isfinite(
            stored_embeddings.float()
        ).all():
            raise FloatingPointError(
                "Non-finite token embedding detected."
            )

        graph_outputs.append({
            "token_embeddings": stored_embeddings,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "num_tokens": num_tokens,
        })

    return graph_outputs


# ------------------------------------------------------------
# 1.6 Feature signature and atomic saving
# ------------------------------------------------------------

FEATURE_SIGNATURE = {
    "plm_name": CONFIG.plm_name,
    "plm_revision": model_revision,
    "max_length": CONFIG.max_length,
    "plm_hidden_dim": CONFIG.plm_hidden_dim,
    "knn_k": CONFIG.knn_k,
    "token_graph_tau": CONFIG.token_graph_tau,
    "feature_storage_dtype": (
        CONFIG.feature_storage_dtype
    ),
    "shard_size_patents": SHARD_SIZE_PATENTS,
}


def atomic_torch_save(data, final_path):
    temporary_path = final_path + ".tmp"

    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    torch.save(
        data,
        temporary_path,
    )

    os.replace(
        temporary_path,
        final_path,
    )


def atomic_json_save(data, final_path):
    temporary_path = final_path + ".tmp"

    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )

    os.replace(
        temporary_path,
        final_path,
    )


def shard_is_compatible(
    shard_path,
    metadata_path,
    expected_patent_ids,
):
    if not (
        os.path.isfile(shard_path)
        and os.path.isfile(metadata_path)
    ):
        return False

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        expected_ids = [
            str(patent_id)
            for patent_id in expected_patent_ids
        ]

        return (
            metadata.get("feature_signature")
            == FEATURE_SIGNATURE
            and metadata.get("patent_ids")
            == expected_ids
            and metadata.get("completed") is True
            and os.path.getsize(shard_path) > 0
        )

    except Exception:
        return False


# ------------------------------------------------------------
# 1.7 Frozen feature extraction for one split
# ------------------------------------------------------------

def make_autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=PLM_COMPUTE_DTYPE,
        )

    return nullcontext()


def extract_split_features(
    split_name,
    records,
    truncated_keys,
):
    split_output_dir = os.path.join(
        FEATURE_ROOT,
        split_name,
    )

    os.makedirs(
        split_output_dir,
        exist_ok=True,
    )

    number_of_patents = len(records)
    number_of_shards = math.ceil(
        number_of_patents
        / SHARD_SIZE_PATENTS
    )

    total_claims = sum(
        len(record["claims"])
        for record in records
    )

    split_start_time = time.time()

    processed_claims = 0
    processed_patents = 0
    skipped_shards = 0
    written_shards = 0

    shard_metadata_records = []

    shard_iterator = tqdm(
        range(number_of_shards),
        desc=f"{split_name} feature shards",
    )

    for shard_id in shard_iterator:
        start_index = (
            shard_id
            * SHARD_SIZE_PATENTS
        )

        end_index = min(
            start_index + SHARD_SIZE_PATENTS,
            number_of_patents,
        )

        shard_records = records[
            start_index:end_index
        ]

        expected_patent_ids = [
            record["patent_id"]
            for record in shard_records
        ]

        shard_path = os.path.join(
            split_output_dir,
            f"shard_{shard_id:05d}.pt",
        )

        metadata_path = os.path.join(
            split_output_dir,
            f"shard_{shard_id:05d}.json",
        )

        if shard_is_compatible(
            shard_path=shard_path,
            metadata_path=metadata_path,
            expected_patent_ids=expected_patent_ids,
        ):
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                existing_metadata = json.load(file)

            processed_claims += int(
                existing_metadata["number_of_claims"]
            )

            processed_patents += int(
                existing_metadata["number_of_patents"]
            )

            skipped_shards += 1
            shard_metadata_records.append(
                existing_metadata
            )
            continue

        if os.path.exists(shard_path):
            raise RuntimeError(
                f"Incompatible or incomplete shard exists: "
                f"{shard_path}. Remove it or use a new run directory."
            )

        claim_items = []

        for record in shard_records:
            patent_id = record["patent_id"]

            for claim_id, text in record["claims"].items():
                claim_items.append({
                    "patent_id": patent_id,
                    "claim_id": claim_id,
                    "text": text if text is not None else "",
                })

        shard_entries = []
        shard_start_time = time.time()

        for batch_start in range(
            0,
            len(claim_items),
            EXTRACTION_BATCH_SIZE,
        ):
            batch_items = claim_items[
                batch_start:
                batch_start + EXTRACTION_BATCH_SIZE
            ]

            batch_texts = [
                item["text"]
                for item in batch_items
            ]

            encoded = plm_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=CONFIG.max_length,
                add_special_tokens=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = encoded[
                "attention_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            special_tokens_mask = encoded[
                "special_tokens_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            with torch.inference_mode():
                with make_autocast_context():
                    outputs = plm_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    last_hidden_state = (
                        outputs.last_hidden_state
                    )

            valid_token_mask = (
                attention_mask.bool()
                & ~special_tokens_mask.bool()
            )

            graph_batch = build_batched_knn_graphs(
                token_embeddings=last_hidden_state,
                valid_token_mask=valid_token_mask,
                k=CONFIG.knn_k,
                tau=CONFIG.token_graph_tau,
            )

            if len(graph_batch) != len(batch_items):
                raise RuntimeError(
                    "Graph output count does not match "
                    "the claim batch size."
                )

            for item, graph_data in zip(
                batch_items,
                graph_batch,
            ):
                claim_key = (
                    item["patent_id"],
                    item["claim_id"],
                )

                shard_entries.append({
                    "patent_id": item["patent_id"],
                    "claim_id": item["claim_id"],
                    "token_embeddings": (
                        graph_data["token_embeddings"]
                    ),
                    "edge_index": (
                        graph_data["edge_index"]
                    ),
                    "edge_weight": (
                        graph_data["edge_weight"]
                    ),
                    "num_tokens": (
                        graph_data["num_tokens"]
                    ),
                    "truncated": (
                        claim_key in truncated_keys
                    ),
                })

            del (
                encoded,
                input_ids,
                attention_mask,
                special_tokens_mask,
                outputs,
                last_hidden_state,
                valid_token_mask,
                graph_batch,
            )

        if len(shard_entries) != len(claim_items):
            raise RuntimeError(
                f"Claim count mismatch in shard {shard_id}: "
                f"expected {len(claim_items)}, "
                f"got {len(shard_entries)}"
            )

        shard_data = {
            "split": split_name,
            "run_name": FEATURE_RUN_NAME,
            "shard_id": shard_id,
            "feature_signature": FEATURE_SIGNATURE,
            "patent_ids": expected_patent_ids,
            "entries": shard_entries,
        }

        atomic_torch_save(
            data=shard_data,
            final_path=shard_path,
        )

        shard_elapsed = (
            time.time()
            - shard_start_time
        )

        shard_size_bytes = os.path.getsize(
            shard_path
        )

        shard_metadata = {
            "completed": True,
            "split": split_name,
            "run_name": FEATURE_RUN_NAME,
            "shard_id": shard_id,
            "feature_signature": FEATURE_SIGNATURE,
            "patent_ids": [
                str(patent_id)
                for patent_id in expected_patent_ids
            ],
            "number_of_patents": len(
                shard_records
            ),
            "number_of_claims": len(
                shard_entries
            ),
            "truncated_claims": sum(
                int(entry["truncated"])
                for entry in shard_entries
            ),
            "total_real_tokens": sum(
                entry["num_tokens"]
                for entry in shard_entries
            ),
            "size_bytes": shard_size_bytes,
            "elapsed_seconds": shard_elapsed,
        }

        atomic_json_save(
            data=shard_metadata,
            final_path=metadata_path,
        )

        processed_claims += len(
            shard_entries
        )
        processed_patents += len(
            shard_records
        )
        written_shards += 1

        shard_metadata_records.append(
            shard_metadata
        )

        shard_iterator.set_postfix({
            "claims": processed_claims,
            "GB": (
                sum(
                    record["size_bytes"]
                    for record in shard_metadata_records
                )
                / (1024 ** 3)
            ),
        })

        del (
            claim_items,
            shard_entries,
            shard_data,
        )

        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_seconds = (
        time.time()
        - split_start_time
    )

    total_size_bytes = sum(
        metadata["size_bytes"]
        for metadata in shard_metadata_records
    )

    total_real_tokens = sum(
        metadata["total_real_tokens"]
        for metadata in shard_metadata_records
    )

    total_truncated_claims = sum(
        metadata["truncated_claims"]
        for metadata in shard_metadata_records
    )

    if processed_patents != number_of_patents:
        raise RuntimeError(
            f"Patent coverage mismatch for {split_name}: "
            f"expected {number_of_patents}, "
            f"got {processed_patents}"
        )

    if processed_claims != total_claims:
        raise RuntimeError(
            f"Claim coverage mismatch for {split_name}: "
            f"expected {total_claims}, "
            f"got {processed_claims}"
        )

    summary = {
        "split": split_name,
        "run_name": FEATURE_RUN_NAME,
        "number_of_patents": number_of_patents,
        "number_of_claims": total_claims,
        "number_of_shards": number_of_shards,
        "written_shards": written_shards,
        "skipped_shards": skipped_shards,
        "total_real_tokens": total_real_tokens,
        "truncated_claims": total_truncated_claims,
        "truncated_ratio": (
            total_truncated_claims / total_claims
            if total_claims > 0
            else 0.0
        ),
        "total_size_bytes": total_size_bytes,
        "total_size_gib": (
            total_size_bytes / (1024 ** 3)
        ),
        "elapsed_seconds": elapsed_seconds,
        "claims_per_second": (
            total_claims / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
        "mean_real_tokens_per_claim": (
            total_real_tokens / total_claims
            if total_claims > 0
            else 0.0
        ),
    }

    split_manifest_path = os.path.join(
        split_output_dir,
        "split_manifest.json",
    )

    atomic_json_save(
        data={
            "summary": summary,
            "feature_signature": FEATURE_SIGNATURE,
            "shards": shard_metadata_records,
        },
        final_path=split_manifest_path,
    )

    return summary


# ------------------------------------------------------------
# 1.8 Sequential split processing
# ------------------------------------------------------------

section1_summary = {
    "run_name": FEATURE_RUN_NAME,
    "feature_root": FEATURE_ROOT,
    "feature_signature": FEATURE_SIGNATURE,
    "splits": {},
}

truncation_summary = {}
depth_verification_summary = {}

for split_name in ["train", "dev", "test"]:
    print(
        "\n"
        + "=" * 68
    )
    print(
        f"PROCESSING SPLIT: {split_name.upper()}"
    )
    print(
        "=" * 68
    )

    all_records = load_pickle(
        RECORD_PATHS[split_name]
    )

    total_patents_in_source = len(
        all_records
    )

    records = (
        all_records[:DEBUG_MAX_PATENTS]
        if DEBUG_MAX_PATENTS is not None
        else all_records
    )

    print(
        f"Loaded patents: {len(records)} "
        f"of {total_patents_in_source}"
    )

    # Depth verification
    depth_report = verify_longest_path_depth(
        records=records,
        split_name=split_name,
    )

    depth_verification_summary[
        split_name
    ] = depth_report

    print(
        "Depth verification: "
        f"checked={depth_report['checked_patents']}, "
        f"mismatch={depth_report['depth_mismatch_count']}, "
        f"cycles={depth_report['cycle_count']}, "
        f"violations="
        f"{depth_report['invariant_violation_count']}, "
        f"invalid={depth_report['invalid_record_count']}"
    )

    assert_valid_depth_report(
        depth_report
    )

    # Truncation statistics
    truncated_keys, truncation_report = (
        compute_truncation_stats(
            records=records,
            tokenizer=plm_tokenizer,
            max_length=CONFIG.max_length,
            chunk_size=STATS_TOKENIZE_CHUNK,
            split_name=split_name,
        )
    )

    truncation_report[
        "source_total_patents"
    ] = total_patents_in_source

    truncation_report[
        "processed_patents"
    ] = len(records)

    truncation_summary[
        split_name
    ] = truncation_report

    print(
        "Truncation statistics: "
        f"claims={truncation_report['total_claims']}, "
        f"truncated="
        f"{truncation_report['truncated_claims']}, "
        f"ratio="
        f"{truncation_report['truncated_ratio']:.4%}, "
        f"mean_raw_length="
        f"{truncation_report['mean_raw_length']:.1f}, "
        f"max_raw_length="
        f"{truncation_report['max_raw_length']}"
    )

    # Feature extraction
    split_summary = extract_split_features(
        split_name=split_name,
        records=records,
        truncated_keys=truncated_keys,
    )

    split_summary[
        "source_total_patents"
    ] = total_patents_in_source

    section1_summary["splits"][
        split_name
    ] = split_summary

    print(
        f"Feature extraction completed for {split_name}: "
        f"patents={split_summary['number_of_patents']}, "
        f"claims={split_summary['number_of_claims']}, "
        f"shards={split_summary['number_of_shards']}, "
        f"size={split_summary['total_size_gib']:.3f} GiB, "
        f"speed={split_summary['claims_per_second']:.2f} claims/s"
    )

    del (
        truncated_keys,
        records,
        all_records,
    )

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# ------------------------------------------------------------
# 1.9 Save Section 1 reports
# ------------------------------------------------------------

depth_report_path = os.path.join(
    DIRS["depth_ot_logs"],
    f"depth_verification_{FEATURE_RUN_NAME}.json",
)

truncation_report_path = os.path.join(
    DIRS["depth_ot_logs"],
    f"truncation_stats_{FEATURE_RUN_NAME}.json",
)

section1_report_path = os.path.join(
    DIRS["depth_ot_logs"],
    f"section1_summary_{FEATURE_RUN_NAME}.json",
)

atomic_json_save(
    data=depth_verification_summary,
    final_path=depth_report_path,
)

atomic_json_save(
    data=truncation_summary,
    final_path=truncation_report_path,
)

atomic_json_save(
    data=section1_summary,
    final_path=section1_report_path,
)

# Update the latest run manifest.
if "RUN_MANIFEST" in globals():
    RUN_MANIFEST["config"] = CONFIG.to_dict()

    RUN_MANIFEST["section1"] = {
        "run_name": FEATURE_RUN_NAME,
        "feature_root": FEATURE_ROOT,
        "feature_signature": FEATURE_SIGNATURE,
        "summary_path": section1_report_path,
        "depth_report_path": depth_report_path,
        "truncation_report_path": truncation_report_path,
    }

    atomic_json_save(
        data=RUN_MANIFEST,
        final_path=latest_manifest_path,
    )

# ------------------------------------------------------------
# 1.10 Verification
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 68
)
print(
    "SECTION 1 VERIFICATION"
)
print(
    "=" * 68
)

verification_report = {}

for split_name in ["train", "dev", "test"]:
    split_output_dir = os.path.join(
        FEATURE_ROOT,
        split_name,
    )

    shard_files = sorted(
        filename
        for filename in os.listdir(
            split_output_dir
        )
        if filename.endswith(".pt")
    )

    metadata_files = sorted(
        filename
        for filename in os.listdir(
            split_output_dir
        )
        if filename.startswith("shard_")
        and filename.endswith(".json")
    )

    if not shard_files:
        raise RuntimeError(
            f"No shard files found for {split_name}."
        )

    if len(shard_files) != len(metadata_files):
        raise RuntimeError(
            f"Shard and metadata count mismatch for {split_name}: "
            f"{len(shard_files)} vs {len(metadata_files)}"
        )

    sample_path = os.path.join(
        split_output_dir,
        shard_files[0],
    )

    sample = torch.load(
        sample_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        sample.get("feature_signature")
        != FEATURE_SIGNATURE
    ):
        raise RuntimeError(
            f"Feature signature mismatch in {sample_path}"
        )

    if not sample["entries"]:
        raise RuntimeError(
            f"Empty sample shard: {sample_path}"
        )

    sample_entry = sample["entries"][0]

    token_embeddings = sample_entry[
        "token_embeddings"
    ]

    edge_index = sample_entry[
        "edge_index"
    ]

    edge_weight = sample_entry[
        "edge_weight"
    ]

    num_tokens = sample_entry[
        "num_tokens"
    ]

    assert token_embeddings.ndim == 2
    assert token_embeddings.shape[0] == num_tokens
    assert (
        token_embeddings.shape[1]
        == CONFIG.plm_hidden_dim
    )
    assert (
        token_embeddings.dtype
        == FEATURE_STORAGE_DTYPE
    )

    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long

    assert edge_weight.ndim == 1
    assert edge_weight.shape[0] == edge_index.shape[1]
    assert edge_weight.dtype == FEATURE_STORAGE_DTYPE

    assert torch.isfinite(
        token_embeddings.float()
    ).all()

    assert torch.isfinite(
        edge_weight.float()
    ).all()

    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < num_tokens

    self_loop_mask = (
        edge_index[0]
        == edge_index[1]
    )

    number_of_self_loops = int(
        self_loop_mask.sum().item()
    )

    assert number_of_self_loops == num_tokens

    if edge_index.shape[1] > num_tokens:
        nonself_mask = ~self_loop_mask

        source = edge_index[0][nonself_mask]
        target = edge_index[1][nonself_mask]

        pair_codes = (
            source * num_tokens
            + target
        )

        reverse_codes = (
            target * num_tokens
            + source
        )

        pair_set = set(
            pair_codes.tolist()
        )

        reverse_set = set(
            reverse_codes.tolist()
        )

        assert pair_set == reverse_set

    split_summary = section1_summary[
        "splits"
    ][split_name]

    verification_report[
        split_name
    ] = {
        "number_of_shards": len(
            shard_files
        ),
        "number_of_metadata_files": len(
            metadata_files
        ),
        "total_size_gib": split_summary[
            "total_size_gib"
        ],
        "number_of_patents": split_summary[
            "number_of_patents"
        ],
        "number_of_claims": split_summary[
            "number_of_claims"
        ],
        "mean_real_tokens_per_claim": (
            split_summary[
                "mean_real_tokens_per_claim"
            ]
        ),
        "claims_per_second": split_summary[
            "claims_per_second"
        ],
        "sample_token_shape": list(
            token_embeddings.shape
        ),
        "sample_edge_shape": list(
            edge_index.shape
        ),
        "sample_self_loops": (
            number_of_self_loops
        ),
        "dtype_ok": True,
        "finite_ok": True,
        "symmetric_ok": True,
    }

    print(
        f"{split_name:5s}: "
        f"shards={len(shard_files)}, "
        f"patents={split_summary['number_of_patents']}, "
        f"claims={split_summary['number_of_claims']}, "
        f"size={split_summary['total_size_gib']:.3f} GiB, "
        f"avg_tokens="
        f"{split_summary['mean_real_tokens_per_claim']:.1f}, "
        f"speed={split_summary['claims_per_second']:.2f} claims/s"
    )

verification_path = os.path.join(
    DIRS["depth_ot_logs"],
    f"section1_verification_{FEATURE_RUN_NAME}.json",
)

atomic_json_save(
    data=verification_report,
    final_path=verification_path,
)

# ------------------------------------------------------------
# 1.11 Full-run storage and runtime projection
# ------------------------------------------------------------

if DEBUG_MAX_PATENTS is not None:
    print(
        "\n=== FULL-RUN PROJECTION FROM DEBUG SAMPLE ==="
    )

    projected_total_gib = 0.0
    projected_total_hours = 0.0

    for split_name in ["train", "dev", "test"]:
        summary = section1_summary[
            "splits"
        ][split_name]

        processed_patents = summary[
            "number_of_patents"
        ]

        source_patents = summary[
            "source_total_patents"
        ]

        scale = (
            source_patents / processed_patents
            if processed_patents > 0
            else 0.0
        )

        projected_gib = (
            summary["total_size_gib"]
            * scale
        )

        projected_hours = (
            summary["elapsed_seconds"]
            * scale
            / 3600.0
        )

        projected_total_gib += (
            projected_gib
        )

        projected_total_hours += (
            projected_hours
        )

        print(
            f"{split_name:5s}: "
            f"projected_size={projected_gib:.2f} GiB, "
            f"projected_time={projected_hours:.2f} h"
        )

    print(
        f"Total projected size: "
        f"{projected_total_gib:.2f} GiB"
    )

    print(
        f"Total projected time: "
        f"{projected_total_hours:.2f} h"
    )

    print(
        "\nBefore the full run, inspect the projection above. "
        "For a more reliable estimate, rerun with "
        "DEBUG_MAX_PATENTS between 500 and 1000."
    )

# ------------------------------------------------------------
# 1.12 Cleanup and final summary
# ------------------------------------------------------------

plm_model.cpu()

del plm_model

gc.collect()

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print(
    "\n"
    + "=" * 68
)
print(
    "SECTION 1 SUMMARY"
)
print(
    "=" * 68
)
print(f"Run name              : {FEATURE_RUN_NAME}")
print(f"Feature root          : {FEATURE_ROOT}")
print(f"PLM                   : {CONFIG.plm_name}")
print(f"PLM hidden dimension  : {CONFIG.plm_hidden_dim}")
print(f"Maximum length        : {CONFIG.max_length}")
print(f"kNN k                 : {CONFIG.knn_k}")
print(f"Token graph tau       : {CONFIG.token_graph_tau}")
print(f"Storage dtype         : {FEATURE_STORAGE_DTYPE}")
print(f"Depth report          : {depth_report_path}")
print(f"Truncation report     : {truncation_report_path}")
print(f"Section 1 report      : {section1_report_path}")
print(f"Verification report   : {verification_path}")
print("=" * 68)

if DEBUG_MAX_PATENTS is not None:
    print(
        "Debug extraction completed. Inspect runtime, storage, "
        "truncation, and graph verification results before setting "
        "DEBUG_MAX_PATENTS=None."
    )
else:
    print(
        "Full feature extraction completed successfully. "
        "Proceed to Section 2: Patent-Level Dataset and Batching."
    )


# ============================================================
# SECTION 2: Patent-Level Dataset and Batching
# PART 0-8
#
# After this cell, run your existing:
#   9. Create and validate a real sample_batch
#   10. Save final Section 2 report
# ============================================================

import os
import re
import glob
import json
import math
import pickle
import hashlib
from collections import OrderedDict, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm


# ============================================================
# 0. Configuration and prerequisites
# ============================================================

if "DIRS" not in globals():
    raise RuntimeError(
        "DIRS is not defined. Run Section 0 first."
    )

if "CONFIG" not in globals():
    raise RuntimeError(
        "CONFIG is not defined. Run Section 0 first."
    )

# Must match the feature run generated in Section 1.
FEATURE_RUN_NAME = "full"

FEATURE_ROOT = os.path.join(
    DIRS["token_features"],
    FEATURE_RUN_NAME,
)

if not os.path.isdir(FEATURE_ROOT):
    raise FileNotFoundError(
        f"Feature root does not exist: {FEATURE_ROOT}. "
        "Run Section 1 with the same FEATURE_RUN_NAME."
    )

SPLITS = [
    "train",
    "dev",
    "test",
]

RECORD_PATHS = {
    split: os.path.join(
        DIRS["processed"],
        f"{split}_records.pkl",
    )
    for split in SPLITS
}

for split, record_path in RECORD_PATHS.items():
    if not os.path.isfile(record_path):
        raise FileNotFoundError(
            f"Missing {split} processed records: "
            f"{record_path}"
        )

print("=== SECTION 2 CONFIGURATION ===")
print(f"Feature run : {FEATURE_RUN_NAME}")
print(f"Feature root: {FEATURE_ROOT}")

for split, record_path in RECORD_PATHS.items():
    file_size_mb = (
        os.path.getsize(record_path)
        / 1024**2
    )

    print(
        f"{split:5s} records: {record_path} "
        f"({file_size_mb:.2f} MB)"
    )


# ============================================================
# 1. Common utilities
# ============================================================

def atomic_json_save(
    data,
    final_path,
):
    parent_directory = os.path.dirname(
        final_path
    )

    if parent_directory:
        os.makedirs(
            parent_directory,
            exist_ok=True,
        )

    temporary_path = (
        final_path + ".tmp"
    )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    os.replace(
        temporary_path,
        final_path,
    )


def normalize_patent_id(
    patent_id,
):
    if patent_id is None:
        raise ValueError(
            "patent_id cannot be None."
        )

    normalized = str(
        patent_id
    ).strip()

    if not normalized:
        raise ValueError(
            "patent_id cannot be empty."
        )

    return normalized


def normalize_claim_id(
    claim_id,
):
    try:
        normalized = int(
            claim_id
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Invalid claim ID: {claim_id}"
        ) from error

    if normalized < 1:
        raise ValueError(
            "Claim ID must be positive, "
            f"got {normalized}."
        )

    return normalized


def validate_edge_index(
    edge_index,
    num_nodes,
    name="edge_index",
):
    if not isinstance(
        edge_index,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if edge_index.dtype != torch.long:
        raise TypeError(
            f"{name} must have dtype torch.long, "
            f"got {edge_index.dtype}."
        )

    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError(
            f"{name} must have shape [2, E], "
            f"got {tuple(edge_index.shape)}."
        )

    if num_nodes < 0:
        raise ValueError(
            f"num_nodes must be nonnegative, "
            f"got {num_nodes}."
        )

    if edge_index.numel() == 0:
        return

    if num_nodes == 0:
        raise ValueError(
            f"{name} contains edges but "
            "num_nodes is zero."
        )

    minimum_index = int(
        edge_index.min().item()
    )

    maximum_index = int(
        edge_index.max().item()
    )

    if (
        minimum_index < 0
        or maximum_index >= num_nodes
    ):
        raise IndexError(
            f"{name} contains invalid indices. "
            f"Observed=[{minimum_index}, "
            f"{maximum_index}], "
            f"valid=[0, {num_nodes - 1}]."
        )


# ============================================================
# 2. Vocabulary and BoW preprocessing
# ============================================================

VOCAB_PATH = os.path.join(
    DIRS["processed"],
    "vocab.pkl",
)

if not os.path.isfile(
    VOCAB_PATH
):
    raise FileNotFoundError(
        f"Missing vocabulary: {VOCAB_PATH}"
    )

with open(
    VOCAB_PATH,
    "rb",
) as file:
    VOCAB = pickle.load(
        file
    )

if not isinstance(
    VOCAB,
    list,
):
    raise TypeError(
        "vocab.pkl must contain a list, "
        f"got {type(VOCAB)}."
    )

if not VOCAB:
    raise ValueError(
        "The vocabulary is empty."
    )

if not all(
    isinstance(word, str)
    for word in VOCAB
):
    raise TypeError(
        "Every vocabulary term must be a string."
    )

if len(VOCAB) != len(
    set(VOCAB)
):
    raise ValueError(
        "Duplicate vocabulary terms were detected."
    )

WORD_TO_ID = {
    word: word_index
    for word_index, word
    in enumerate(VOCAB)
}

BOW_TOKEN_PATTERN_STRING = (
    r"(?u)\b\w\w+\b"
)

BOW_TOKEN_PATTERN = re.compile(
    BOW_TOKEN_PATTERN_STRING
)

VOCAB_HASH = hashlib.sha256(
    "\n".join(VOCAB).encode(
        "utf-8"
    )
).hexdigest()

CONFIG.vocab_size = len(
    VOCAB
)

CONFIG.bow_lowercase = True

CONFIG.bow_token_pattern = (
    BOW_TOKEN_PATTERN_STRING
)

CONFIG.bow_normalization = "l1"

CONFIG.bow_empty_policy = (
    "zero_vector"
)

CONFIG.bow_reconstruction_policy = (
    "exclude_empty_rows"
)


def claim_text_to_bow(
    claim_text,
):
    """
    Fixed-vocabulary BoW.

    Non-empty claim BoW:
        L1-normalized, row sum = 1.

    Claim with no vocabulary terms:
        zero vector, row sum = 0.

    Empty-BoW claims remain in token and dependency graphs,
    but are excluded from the reconstruction loss in Section 5.
    """

    if claim_text is None:
        claim_text = ""

    if not isinstance(
        claim_text,
        str,
    ):
        claim_text = str(
            claim_text
        )

    tokens = (
        BOW_TOKEN_PATTERN.findall(
            claim_text.lower()
        )
    )

    bow = torch.zeros(
        len(VOCAB),
        dtype=torch.float32,
    )

    for token in tokens:
        token_id = WORD_TO_ID.get(
            token
        )

        if token_id is not None:
            bow[token_id] += 1.0

    total = bow.sum()

    if total.item() > 0.0:
        bow = (
            bow / total
        )

    return bow


print("\n=== VOCABULARY ===")
print(
    f"Vocabulary size: "
    f"{len(VOCAB):,}"
)
print(
    f"Vocabulary hash: "
    f"{VOCAB_HASH[:16]}..."
)
print(
    f"Token pattern  : "
    f"{BOW_TOKEN_PATTERN_STRING}"
)
print(
    "Normalization  : "
    "L1 for non-empty rows; zero for empty rows"
)


# ============================================================
# 3. Longest-path depth
# ============================================================

def compute_longest_path_depth(
    claim_ids,
    edges,
):
    normalized_claim_ids = [
        normalize_claim_id(
            claim_id
        )
        for claim_id in claim_ids
    ]

    if not normalized_claim_ids:
        raise ValueError(
            "A patent contains no claims."
        )

    if (
        len(normalized_claim_ids)
        != len(
            set(
                normalized_claim_ids
            )
        )
    ):
        raise ValueError(
            "Duplicate claim IDs detected."
        )

    claim_id_set = set(
        normalized_claim_ids
    )

    children = {
        claim_id: []
        for claim_id
        in normalized_claim_ids
    }

    indegree = {
        claim_id: 0
        for claim_id
        in normalized_claim_ids
    }

    seen_edges = set()

    for edge in edges:
        if (
            not isinstance(
                edge,
                (list, tuple),
            )
            or len(edge) != 2
        ):
            raise ValueError(
                f"Invalid dependency edge: {edge}"
            )

        parent_id = normalize_claim_id(
            edge[0]
        )

        child_id = normalize_claim_id(
            edge[1]
        )

        if parent_id not in claim_id_set:
            raise ValueError(
                "Unknown parent claim ID: "
                f"{parent_id}"
            )

        if child_id not in claim_id_set:
            raise ValueError(
                "Unknown child claim ID: "
                f"{child_id}"
            )

        if parent_id == child_id:
            raise ValueError(
                "Self-dependency detected: "
                f"{parent_id}"
            )

        normalized_edge = (
            parent_id,
            child_id,
        )

        if normalized_edge in seen_edges:
            raise ValueError(
                "Duplicate dependency edge: "
                f"{normalized_edge}"
            )

        seen_edges.add(
            normalized_edge
        )

        children[
            parent_id
        ].append(
            child_id
        )

        indegree[
            child_id
        ] += 1

    roots = [
        claim_id
        for claim_id
        in normalized_claim_ids
        if indegree[
            claim_id
        ] == 0
    ]

    if not roots:
        raise ValueError(
            "No root claim was found."
        )

    queue = list(
        roots
    )

    depth = {
        claim_id: 0
        for claim_id in roots
    }

    queue_position = 0
    processed_count = 0

    while queue_position < len(
        queue
    ):
        current_claim = queue[
            queue_position
        ]

        queue_position += 1
        processed_count += 1

        current_depth = depth[
            current_claim
        ]

        for child_id in children[
            current_claim
        ]:
            candidate_depth = (
                current_depth + 1
            )

            depth[
                child_id
            ] = max(
                depth.get(
                    child_id,
                    0,
                ),
                candidate_depth,
            )

            indegree[
                child_id
            ] -= 1

            if indegree[
                child_id
            ] == 0:
                queue.append(
                    child_id
                )

    if processed_count != len(
        normalized_claim_ids
    ):
        raise ValueError(
            "A cycle was detected in the claim graph."
        )

    if set(
        depth.keys()
    ) != claim_id_set:
        missing_claims = (
            claim_id_set
            - set(
                depth.keys()
            )
        )

        raise RuntimeError(
            "Depth was not assigned to all claims: "
            f"{sorted(missing_claims)}"
        )

    return depth


# ============================================================
# 4. Build patent-to-shard indices
# ============================================================

def load_existing_patent_index(
    index_path,
    split,
):
    if not os.path.isfile(
        index_path
    ):
        return None

    with open(
        index_path,
        "r",
        encoding="utf-8",
    ) as file:
        index_data = json.load(
            file
        )

    if index_data.get(
        "split"
    ) != split:
        return None

    if index_data.get(
        "run_name"
    ) != FEATURE_RUN_NAME:
        return None

    patent_index = index_data.get(
        "patent_index"
    )

    if not isinstance(
        patent_index,
        dict,
    ):
        return None

    if not patent_index:
        return None

    shard_directory = os.path.join(
        FEATURE_ROOT,
        split,
    )

    for patent_id, information in (
        patent_index.items()
    ):
        normalize_patent_id(
            patent_id
        )

        if not isinstance(
            information,
            dict,
        ):
            return None

        shard_filename = information.get(
            "shard_file"
        )

        if not shard_filename:
            return None

        shard_path = os.path.join(
            shard_directory,
            shard_filename,
        )

        if not os.path.isfile(
            shard_path
        ):
            return None

    return patent_index


def find_valid_metadata_files(
    shard_directory,
):
    candidate_paths = glob.glob(
        os.path.join(
            shard_directory,
            "shard_*.json",
        )
    )

    valid_paths = []

    metadata_pattern = re.compile(
        r"^shard_(\d+)\.json$"
    )

    for candidate_path in (
        candidate_paths
    ):
        filename = os.path.basename(
            candidate_path
        )

        if metadata_pattern.match(
            filename
        ):
            valid_paths.append(
                candidate_path
            )

    return sorted(
        valid_paths
    )


def build_patent_shard_index(
    split,
    force_rebuild=False,
):
    shard_directory = os.path.join(
        FEATURE_ROOT,
        split,
    )

    if not os.path.isdir(
        shard_directory
    ):
        raise FileNotFoundError(
            "Missing feature split directory: "
            f"{shard_directory}"
        )

    index_path = os.path.join(
        shard_directory,
        "patent_shard_index.json",
    )

    if not force_rebuild:
        existing_index = (
            load_existing_patent_index(
                index_path=index_path,
                split=split,
            )
        )

        if existing_index is not None:
            print(
                f"[{split}] Existing patent index loaded: "
                f"{len(existing_index):,} patents"
            )

            return existing_index

    patent_index = {}
    number_of_claims = 0

    metadata_files = (
        find_valid_metadata_files(
            shard_directory
        )
    )

    if metadata_files:
        print(
            f"[{split}] Building index from "
            f"{len(metadata_files):,} metadata files."
        )

        for metadata_path in tqdm(
            metadata_files,
            desc=f"{split} metadata indexing",
        ):
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(
                    file
                )

            if (
                "completed" in metadata
                and metadata[
                    "completed"
                ] is not True
            ):
                raise RuntimeError(
                    "Incomplete shard metadata: "
                    f"{metadata_path}"
                )

            if (
                metadata.get(
                    "split",
                    split,
                )
                != split
            ):
                raise ValueError(
                    "Metadata split mismatch: "
                    f"{metadata_path}"
                )

            metadata_run_name = (
                metadata.get(
                    "run_name"
                )
            )

            if (
                metadata_run_name
                is not None
                and metadata_run_name
                != FEATURE_RUN_NAME
            ):
                raise ValueError(
                    "Metadata run-name mismatch: "
                    f"{metadata_path}"
                )

            if "shard_id" in metadata:
                shard_id = int(
                    metadata[
                        "shard_id"
                    ]
                )
            else:
                filename_match = re.match(
                    r"^shard_(\d+)\.json$",
                    os.path.basename(
                        metadata_path
                    ),
                )

                if filename_match is None:
                    raise ValueError(
                        "Cannot determine shard ID: "
                        f"{metadata_path}"
                    )

                shard_id = int(
                    filename_match.group(
                        1
                    )
                )

            shard_filename = (
                f"shard_{shard_id:05d}.pt"
            )

            shard_path = os.path.join(
                shard_directory,
                shard_filename,
            )

            if not os.path.isfile(
                shard_path
            ):
                alternative_filename = (
                    f"shard_{shard_id}.pt"
                )

                alternative_path = os.path.join(
                    shard_directory,
                    alternative_filename,
                )

                if os.path.isfile(
                    alternative_path
                ):
                    shard_filename = (
                        alternative_filename
                    )
                    shard_path = (
                        alternative_path
                    )
                else:
                    raise FileNotFoundError(
                        "Missing shard tensor file for "
                        f"{metadata_path}"
                    )

            patent_ids = metadata.get(
                "patent_ids"
            )

            if not isinstance(
                patent_ids,
                list,
            ):
                raise TypeError(
                    "Metadata patent_ids must be a list: "
                    f"{metadata_path}"
                )

            for patent_id in patent_ids:
                normalized_patent_id = (
                    normalize_patent_id(
                        patent_id
                    )
                )

                if (
                    normalized_patent_id
                    in patent_index
                ):
                    raise ValueError(
                        "Duplicate patent ID across shards: "
                        f"{normalized_patent_id}"
                    )

                patent_index[
                    normalized_patent_id
                ] = {
                    "shard_file": (
                        shard_filename
                    )
                }

            number_of_claims += int(
                metadata.get(
                    "number_of_claims",
                    0,
                )
            )

    else:
        shard_files = sorted(
            glob.glob(
                os.path.join(
                    shard_directory,
                    "shard_*.pt",
                )
            )
        )

        if not shard_files:
            raise FileNotFoundError(
                "No feature shards found in "
                f"{shard_directory}"
            )

        print(
            f"[{split}] Metadata files unavailable; "
            f"scanning {len(shard_files):,} shard files."
        )

        for shard_path in tqdm(
            shard_files,
            desc=f"{split} shard indexing",
        ):
            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )

            shard_split = shard.get(
                "split"
            )

            if (
                shard_split is not None
                and shard_split != split
            ):
                raise ValueError(
                    "Shard split mismatch: "
                    f"{shard_path}"
                )

            shard_run_name = shard.get(
                "run_name"
            )

            if (
                shard_run_name is not None
                and shard_run_name
                != FEATURE_RUN_NAME
            ):
                raise ValueError(
                    "Shard run-name mismatch: "
                    f"{shard_path}"
                )

            shard_filename = os.path.basename(
                shard_path
            )

            patent_ids = shard.get(
                "patent_ids"
            )

            if patent_ids is None:
                patent_ids = sorted(
                    {
                        normalize_patent_id(
                            entry[
                                "patent_id"
                            ]
                        )
                        for entry in shard[
                            "entries"
                        ]
                    }
                )

            for patent_id in patent_ids:
                normalized_patent_id = (
                    normalize_patent_id(
                        patent_id
                    )
                )

                if (
                    normalized_patent_id
                    in patent_index
                ):
                    raise ValueError(
                        "Duplicate patent ID across shards: "
                        f"{normalized_patent_id}"
                    )

                patent_index[
                    normalized_patent_id
                ] = {
                    "shard_file": (
                        shard_filename
                    )
                }

            number_of_claims += len(
                shard.get(
                    "entries",
                    [],
                )
            )

            del shard

    if not patent_index:
        raise RuntimeError(
            f"No patents indexed for split={split}."
        )

    index_data = {
        "split": split,
        "run_name": FEATURE_RUN_NAME,
        "feature_root": FEATURE_ROOT,
        "number_of_patents": len(
            patent_index
        ),
        "number_of_claims": (
            number_of_claims
        ),
        "patent_index": (
            patent_index
        ),
    }

    atomic_json_save(
        index_data,
        index_path,
    )

    print(
        f"[{split}] Indexed patents: "
        f"{len(patent_index):,}"
    )

    return patent_index


print("\n=== PATENT SHARD INDICES ===")

PATENT_SHARD_INDEX = {
    split: build_patent_shard_index(
        split=split,
        force_rebuild=False,
    )
    for split in SPLITS
}


# ============================================================
# 5. Feature shard LRU cache
# ============================================================

class FeatureShardCache:
    def __init__(
        self,
        feature_root,
        split,
        run_name,
        max_cached_shards=2,
    ):
        if max_cached_shards < 1:
            raise ValueError(
                "max_cached_shards must be positive."
            )

        self.split = split
        self.run_name = run_name
        self.max_cached_shards = int(
            max_cached_shards
        )

        self.shard_directory = os.path.join(
            feature_root,
            split,
        )

        if not os.path.isdir(
            self.shard_directory
        ):
            raise FileNotFoundError(
                "Shard directory does not exist: "
                f"{self.shard_directory}"
            )

        self.cache = OrderedDict()
        self.load_count = 0
        self.hit_count = 0

    def _load_shard(
        self,
        shard_filename,
    ):
        shard_path = os.path.join(
            self.shard_directory,
            shard_filename,
        )

        if not os.path.isfile(
            shard_path
        ):
            raise FileNotFoundError(
                f"Missing shard: {shard_path}"
            )

        shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        shard_split = shard.get(
            "split"
        )

        if (
            shard_split is not None
            and shard_split != self.split
        ):
            raise ValueError(
                "Shard split mismatch: "
                f"{shard_path}"
            )

        shard_run_name = shard.get(
            "run_name"
        )

        if (
            shard_run_name is not None
            and shard_run_name
            != self.run_name
        ):
            raise ValueError(
                "Shard run-name mismatch: "
                f"{shard_path}"
            )

        entries = shard.get(
            "entries"
        )

        if not isinstance(
            entries,
            list,
        ):
            raise TypeError(
                "shard['entries'] must be a list: "
                f"{shard_path}"
            )

        entries_by_patent = {}

        for entry in entries:
            if not isinstance(
                entry,
                dict,
            ):
                raise TypeError(
                    "Every shard entry must be a dictionary."
                )

            patent_id = normalize_patent_id(
                entry[
                    "patent_id"
                ]
            )

            entries_by_patent.setdefault(
                patent_id,
                [],
            ).append(
                entry
            )

        for patent_entries in (
            entries_by_patent.values()
        ):
            patent_entries.sort(
                key=lambda entry: normalize_claim_id(
                    entry[
                        "claim_id"
                    ]
                )
            )

        self.load_count += 1

        return {
            "shard": shard,
            "entries_by_patent": (
                entries_by_patent
            ),
        }

    def get_shard(
        self,
        shard_filename,
    ):
        if shard_filename in self.cache:
            payload = self.cache.pop(
                shard_filename
            )

            self.cache[
                shard_filename
            ] = payload

            self.hit_count += 1

            return payload

        payload = self._load_shard(
            shard_filename
        )

        self.cache[
            shard_filename
        ] = payload

        while (
            len(self.cache)
            > self.max_cached_shards
        ):
            self.cache.popitem(
                last=False
            )

        return payload

    def get_patent_entries(
        self,
        patent_id,
        shard_filename,
    ):
        normalized_patent_id = (
            normalize_patent_id(
                patent_id
            )
        )

        payload = self.get_shard(
            shard_filename
        )

        patent_entries = payload[
            "entries_by_patent"
        ].get(
            normalized_patent_id
        )

        if not patent_entries:
            raise KeyError(
                "No features found for patent "
                f"{normalized_patent_id} in "
                f"{shard_filename}."
            )

        return patent_entries


# ============================================================
# 6. Patent-level Dataset
# ============================================================

class PatentDataset(Dataset):
    def __init__(
        self,
        split,
        record_path,
        patent_shard_index,
        feature_root,
        run_name,
        max_cached_shards=2,
        verify_depth=True,
    ):
        super().__init__()

        self.split = split
        self.record_path = record_path
        self.patent_shard_index = (
            patent_shard_index
        )

        print(
            f"[{split}] Loading processed records: "
            f"{record_path}"
        )

        with open(
            record_path,
            "rb",
        ) as file:
            all_records = pickle.load(
                file
            )

        if not isinstance(
            all_records,
            list,
        ):
            raise TypeError(
                f"{split} records must be a list."
            )

        indexed_patent_ids = set(
            patent_shard_index.keys()
        )

        self.records = []

        for record in all_records:
            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    "Every processed record must be "
                    "a dictionary."
                )

            patent_id = normalize_patent_id(
                record[
                    "patent_id"
                ]
            )

            if patent_id in (
                indexed_patent_ids
            ):
                self.records.append(
                    record
                )

        del all_records

        if not self.records:
            raise RuntimeError(
                "No matching processed records for "
                f"split={split}."
            )

        found_patent_ids = {
            normalize_patent_id(
                record[
                    "patent_id"
                ]
            )
            for record in self.records
        }

        missing_processed_records = (
            indexed_patent_ids
            - found_patent_ids
        )

        if missing_processed_records:
            raise RuntimeError(
                f"{split}: feature patents missing "
                "from processed records: "
                f"{sorted(missing_processed_records)[:20]}"
            )

        self.shard_cache = (
            FeatureShardCache(
                feature_root=feature_root,
                split=split,
                run_name=run_name,
                max_cached_shards=(
                    max_cached_shards
                ),
            )
        )

        self.num_claims = 0
        self.num_edges = 0
        self.non_adjacent_edges = 0
        self.max_depth = 0

        if verify_depth:
            self._verify_records()
        else:
            self._collect_record_statistics()

        print(
            f"[{split}] patents="
            f"{len(self.records):,}, "
            f"claims={self.num_claims:,}, "
            f"edges={self.num_edges:,}, "
            f"max_depth={self.max_depth}, "
            f"non_adjacent_edges="
            f"{self.non_adjacent_edges:,}"
        )

    def _collect_record_statistics(
        self,
    ):
        for record in self.records:
            claims = record[
                "claims"
            ]

            edges = record[
                "edges"
            ]

            depths = {
                normalize_claim_id(
                    claim_id
                ): int(depth)
                for claim_id, depth
                in record[
                    "depth"
                ].items()
            }

            self.num_claims += len(
                claims
            )

            self.num_edges += len(
                edges
            )

            if depths:
                self.max_depth = max(
                    self.max_depth,
                    max(
                        depths.values()
                    ),
                )

            for parent_id, child_id in edges:
                parent_id = normalize_claim_id(
                    parent_id
                )

                child_id = normalize_claim_id(
                    child_id
                )

                depth_difference = (
                    depths[
                        child_id
                    ]
                    - depths[
                        parent_id
                    ]
                )

                if depth_difference != 1:
                    self.non_adjacent_edges += 1

    def _verify_records(
        self,
    ):
        for record in tqdm(
            self.records,
            desc=(
                f"{self.split} "
                "depth validation"
            ),
        ):
            claims = record.get(
                "claims"
            )

            edges = record.get(
                "edges"
            )

            stored_depth_raw = (
                record.get(
                    "depth"
                )
            )

            if not isinstance(
                claims,
                dict,
            ):
                raise TypeError(
                    "record['claims'] must be "
                    "a dictionary."
                )

            if not isinstance(
                edges,
                list,
            ):
                raise TypeError(
                    "record['edges'] must be a list."
                )

            if not isinstance(
                stored_depth_raw,
                dict,
            ):
                raise TypeError(
                    "record['depth'] must be "
                    "a dictionary."
                )

            stored_depth = {
                normalize_claim_id(
                    claim_id
                ): int(depth)
                for claim_id, depth
                in stored_depth_raw.items()
            }

            recomputed_depth = (
                compute_longest_path_depth(
                    claim_ids=claims.keys(),
                    edges=edges,
                )
            )

            if (
                stored_depth
                != recomputed_depth
            ):
                raise ValueError(
                    "Depth mismatch for patent "
                    f"{record['patent_id']}."
                )

            for parent_id, child_id in edges:
                parent_id = normalize_claim_id(
                    parent_id
                )

                child_id = normalize_claim_id(
                    child_id
                )

                depth_difference = (
                    recomputed_depth[
                        child_id
                    ]
                    - recomputed_depth[
                        parent_id
                    ]
                )

                if depth_difference <= 0:
                    raise ValueError(
                        "A dependency edge does not "
                        "increase depth."
                    )

                if depth_difference != 1:
                    self.non_adjacent_edges += 1

            self.num_claims += len(
                claims
            )

            self.num_edges += len(
                edges
            )

            if recomputed_depth:
                self.max_depth = max(
                    self.max_depth,
                    max(
                        recomputed_depth.values()
                    ),
                )

    def __len__(
        self,
    ):
        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):
        record = self.records[
            index
        ]

        patent_id = normalize_patent_id(
            record[
                "patent_id"
            ]
        )

        shard_information = (
            self.patent_shard_index[
                patent_id
            ]
        )

        shard_filename = (
            shard_information[
                "shard_file"
            ]
        )

        feature_entries = (
            self.shard_cache
            .get_patent_entries(
                patent_id=patent_id,
                shard_filename=(
                    shard_filename
                ),
            )
        )

        features_by_claim = {}

        for entry in feature_entries:
            claim_id = normalize_claim_id(
                entry[
                    "claim_id"
                ]
            )

            if claim_id in features_by_claim:
                raise ValueError(
                    "Duplicate feature entry for "
                    f"patent={patent_id}, "
                    f"claim={claim_id}."
                )

            features_by_claim[
                claim_id
            ] = entry

        claims = {
            normalize_claim_id(
                claim_id
            ): text
            for claim_id, text
            in record[
                "claims"
            ].items()
        }

        depths = {
            normalize_claim_id(
                claim_id
            ): int(depth)
            for claim_id, depth
            in record[
                "depth"
            ].items()
        }

        if set(
            claims.keys()
        ) != set(
            features_by_claim.keys()
        ):
            missing_features = (
                set(
                    claims.keys()
                )
                - set(
                    features_by_claim.keys()
                )
            )

            unexpected_features = (
                set(
                    features_by_claim.keys()
                )
                - set(
                    claims.keys()
                )
            )

            raise RuntimeError(
                "Claim-feature mismatch for patent "
                f"{patent_id}. "
                f"Missing={sorted(missing_features)}, "
                f"unexpected="
                f"{sorted(unexpected_features)}."
            )

        if set(
            claims.keys()
        ) != set(
            depths.keys()
        ):
            raise RuntimeError(
                "Claim-depth mismatch for patent "
                f"{patent_id}."
            )

        claim_items = []

        for claim_id in sorted(
            claims.keys()
        ):
            feature = features_by_claim[
                claim_id
            ]

            required_feature_fields = {
                "token_embeddings",
                "edge_index",
                "edge_weight",
            }

            missing_feature_fields = (
                required_feature_fields
                - set(
                    feature.keys()
                )
            )

            if missing_feature_fields:
                raise KeyError(
                    "Missing feature fields for "
                    f"patent={patent_id}, "
                    f"claim={claim_id}: "
                    f"{sorted(missing_feature_fields)}"
                )

            token_embeddings = feature[
                "token_embeddings"
            ]

            edge_index = feature[
                "edge_index"
            ].long()

            edge_weight = feature[
                "edge_weight"
            ].float()

            if not isinstance(
                token_embeddings,
                torch.Tensor,
            ):
                raise TypeError(
                    "token_embeddings must be "
                    "a torch.Tensor."
                )

            if token_embeddings.ndim != 2:
                raise ValueError(
                    "token_embeddings must have "
                    "shape [num_tokens, hidden_dim]."
                )

            num_tokens = int(
                token_embeddings.shape[0]
            )

            if num_tokens <= 0:
                raise ValueError(
                    "Empty token features for "
                    f"patent={patent_id}, "
                    f"claim={claim_id}."
                )

            if not torch.isfinite(
                token_embeddings
            ).all():
                raise FloatingPointError(
                    "Non-finite token embeddings for "
                    f"patent={patent_id}, "
                    f"claim={claim_id}."
                )

            validate_edge_index(
                edge_index=edge_index,
                num_nodes=num_tokens,
                name=(
                    "local_token_edge_index"
                ),
            )

            if edge_weight.ndim != 1:
                raise ValueError(
                    "edge_weight must have "
                    "shape [E]."
                )

            if (
                edge_weight.shape[0]
                != edge_index.shape[1]
            ):
                raise ValueError(
                    "Local token edge-count mismatch."
                )

            if not torch.isfinite(
                edge_weight
            ).all():
                raise FloatingPointError(
                    "Non-finite token edge weights."
                )

            if torch.any(
                edge_weight < 0
            ):
                raise ValueError(
                    "Negative token edge weights "
                    "detected. Expected "
                    "exp(cosine/tau) weights."
                )

            claim_items.append(
                {
                    "claim_id": (
                        claim_id
                    ),
                    "depth": int(
                        depths[
                            claim_id
                        ]
                    ),
                    "bow": (
                        claim_text_to_bow(
                            claims[
                                claim_id
                            ]
                        )
                    ),
                    "token_embeddings": (
                        token_embeddings
                    ),
                    "edge_index": (
                        edge_index
                    ),
                    "edge_weight": (
                        edge_weight
                    ),
                    "num_tokens": (
                        num_tokens
                    ),
                    "truncated": bool(
                        feature.get(
                            "truncated",
                            False,
                        )
                    ),
                }
            )

        normalized_edges = [
            (
                normalize_claim_id(
                    parent_id
                ),
                normalize_claim_id(
                    child_id
                ),
            )
            for parent_id, child_id
            in record[
                "edges"
            ]
        ]

        return {
            "patent_id": patent_id,
            "claims": claim_items,
            "edges": normalized_edges,
        }


# ============================================================
# 7. Patent-level collate function
# ============================================================

def patent_collate_fn(
    batch,
):
    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    token_embedding_parts = []
    token_edge_parts = []
    token_edge_weight_parts = []
    token_to_claim_parts = []

    bow_parts = []
    claim_depth_values = []
    claim_to_patent_values = []

    claim_parent_indices = []
    claim_child_indices = []

    patent_ids = []
    claim_ids = []
    claim_keys = []
    truncated_flags = []

    token_offset = 0
    claim_offset = 0

    for patent_position, patent in enumerate(
        batch
    ):
        patent_id = normalize_patent_id(
            patent[
                "patent_id"
            ]
        )

        patent_ids.append(
            patent_id
        )

        claims = patent.get(
            "claims"
        )

        if not isinstance(
            claims,
            list,
        ):
            raise TypeError(
                "patent['claims'] must be a list."
            )

        if not claims:
            raise ValueError(
                f"Patent {patent_id} has no claims."
            )

        local_claim_map = {}

        for local_claim_position, claim in enumerate(
            claims
        ):
            global_claim_index = (
                claim_offset
                + local_claim_position
            )

            claim_id = normalize_claim_id(
                claim[
                    "claim_id"
                ]
            )

            if claim_id in local_claim_map:
                raise ValueError(
                    "Duplicate claim ID in patent "
                    f"{patent_id}: {claim_id}"
                )

            local_claim_map[
                claim_id
            ] = global_claim_index

            token_embeddings = claim[
                "token_embeddings"
            ]

            edge_index = claim[
                "edge_index"
            ].long()

            edge_weight = claim[
                "edge_weight"
            ].float()

            num_tokens = int(
                token_embeddings.shape[0]
            )

            token_embedding_parts.append(
                token_embeddings
            )

            if edge_index.shape[1] > 0:
                token_edge_parts.append(
                    edge_index
                    + token_offset
                )

                token_edge_weight_parts.append(
                    edge_weight
                )

            token_to_claim_parts.append(
                torch.full(
                    (
                        num_tokens,
                    ),
                    global_claim_index,
                    dtype=torch.long,
                )
            )

            bow_parts.append(
                claim[
                    "bow"
                ].float()
            )

            claim_depth_values.append(
                int(
                    claim[
                        "depth"
                    ]
                )
            )

            claim_to_patent_values.append(
                patent_position
            )

            claim_ids.append(
                claim_id
            )

            claim_keys.append(
                (
                    patent_id,
                    claim_id,
                )
            )

            truncated_flags.append(
                bool(
                    claim[
                        "truncated"
                    ]
                )
            )

            token_offset += (
                num_tokens
            )

        for parent_id, child_id in patent[
            "edges"
        ]:
            parent_id = normalize_claim_id(
                parent_id
            )

            child_id = normalize_claim_id(
                child_id
            )

            if parent_id not in local_claim_map:
                raise KeyError(
                    "Missing parent claim "
                    f"{parent_id} in patent "
                    f"{patent_id}."
                )

            if child_id not in local_claim_map:
                raise KeyError(
                    "Missing child claim "
                    f"{child_id} in patent "
                    f"{patent_id}."
                )

            claim_parent_indices.append(
                local_claim_map[
                    parent_id
                ]
            )

            claim_child_indices.append(
                local_claim_map[
                    child_id
                ]
            )

        claim_offset += len(
            claims
        )

    if not token_embedding_parts:
        raise RuntimeError(
            "The batch contains no token embeddings."
        )

    token_embeddings = torch.cat(
        token_embedding_parts,
        dim=0,
    )

    if token_edge_parts:
        token_edge_index = torch.cat(
            token_edge_parts,
            dim=1,
        )

        token_edge_weight = torch.cat(
            token_edge_weight_parts,
            dim=0,
        ).float()
    else:
        token_edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

        token_edge_weight = torch.empty(
            (
                0,
            ),
            dtype=torch.float32,
        )

    token_to_claim = torch.cat(
        token_to_claim_parts,
        dim=0,
    )

    bow = torch.stack(
        bow_parts,
        dim=0,
    )

    claim_depth = torch.tensor(
        claim_depth_values,
        dtype=torch.long,
    )

    claim_to_patent = torch.tensor(
        claim_to_patent_values,
        dtype=torch.long,
    )

    if claim_parent_indices:
        claim_edge_index = torch.tensor(
            [
                claim_parent_indices,
                claim_child_indices,
            ],
            dtype=torch.long,
        )
    else:
        claim_edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

    num_patents = len(
        patent_ids
    )

    num_claims = len(
        claim_ids
    )

    num_tokens = int(
        token_embeddings.shape[0]
    )

    return {
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
        "claim_edge_index": (
            claim_edge_index
        ),
        "claim_parent_index": (
            claim_edge_index[0]
        ),
        "claim_child_index": (
            claim_edge_index[1]
        ),
        "claim_to_patent": (
            claim_to_patent
        ),
        "claim_depth": (
            claim_depth
        ),
        "bow": bow,
        "patent_ids": (
            patent_ids
        ),
        "claim_ids": (
            claim_ids
        ),
        "claim_keys": (
            claim_keys
        ),
        "truncated": torch.tensor(
            truncated_flags,
            dtype=torch.bool,
        ),
        "num_patents": (
            num_patents
        ),
        "num_claims": (
            num_claims
        ),
        "num_tokens": (
            num_tokens
        ),
    }


# ============================================================
# 8. Build datasets and shard-aware DataLoaders
# ============================================================

# Smaller batch size reduces graph size and peak GPU memory.
# Because batches remain inside a shard, Google Drive random
# shard loading is also greatly reduced.
if FEATURE_RUN_NAME.startswith(
    "debug"
):
    PATENTS_PER_BATCH = 2
else:
    PATENTS_PER_BATCH = 8

CONFIG.patents_per_batch = (
    PATENTS_PER_BATCH
)

MAX_CACHED_SHARDS = 2

# Keep zero for Google Drive. Multiple workers create independent
# caches and may repeatedly load the same shard.
DATALOADER_NUM_WORKERS = 0

print("\n=== DATASET CREATION ===")
print(
    f"Patents per batch: "
    f"{PATENTS_PER_BATCH}"
)
print(
    f"Cached shards    : "
    f"{MAX_CACHED_SHARDS}"
)
print(
    f"DataLoader workers: "
    f"{DATALOADER_NUM_WORKERS}"
)

train_dataset = PatentDataset(
    split="train",
    record_path=(
        RECORD_PATHS[
            "train"
        ]
    ),
    patent_shard_index=(
        PATENT_SHARD_INDEX[
            "train"
        ]
    ),
    feature_root=(
        FEATURE_ROOT
    ),
    run_name=(
        FEATURE_RUN_NAME
    ),
    max_cached_shards=(
        MAX_CACHED_SHARDS
    ),
    verify_depth=True,
)

dev_dataset = PatentDataset(
    split="dev",
    record_path=(
        RECORD_PATHS[
            "dev"
        ]
    ),
    patent_shard_index=(
        PATENT_SHARD_INDEX[
            "dev"
        ]
    ),
    feature_root=(
        FEATURE_ROOT
    ),
    run_name=(
        FEATURE_RUN_NAME
    ),
    max_cached_shards=(
        MAX_CACHED_SHARDS
    ),
    verify_depth=True,
)

test_dataset = PatentDataset(
    split="test",
    record_path=(
        RECORD_PATHS[
            "test"
        ]
    ),
    patent_shard_index=(
        PATENT_SHARD_INDEX[
            "test"
        ]
    ),
    feature_root=(
        FEATURE_ROOT
    ),
    run_name=(
        FEATURE_RUN_NAME
    ),
    max_cached_shards=(
        MAX_CACHED_SHARDS
    ),
    verify_depth=True,
)


class ShardAwareBatchSampler(Sampler):
    """
    Produce batches containing patents from one shard.

    Training:
        shuffle shard order and patent order within each shard.

    Dev/Test:
        deterministic order.

    set_epoch(epoch) must be called at the beginning of each
    training epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        seed=42,
        drop_last=False,
    ):
        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        self.dataset = dataset
        self.batch_size = int(
            batch_size
        )
        self.shuffle = bool(
            shuffle
        )
        self.seed = int(
            seed
        )
        self.drop_last = bool(
            drop_last
        )
        self.epoch = 0

        shard_to_indices = defaultdict(
            list
        )

        for dataset_index, record in enumerate(
            dataset.records
        ):
            patent_id = normalize_patent_id(
                record[
                    "patent_id"
                ]
            )

            if patent_id not in (
                dataset.patent_shard_index
            ):
                raise KeyError(
                    f"Patent {patent_id} "
                    "is missing from the shard index."
                )

            shard_filename = (
                dataset
                .patent_shard_index[
                    patent_id
                ][
                    "shard_file"
                ]
            )

            shard_to_indices[
                shard_filename
            ].append(
                dataset_index
            )

        if not shard_to_indices:
            raise RuntimeError(
                "No shard groups were created."
            )

        self.shard_groups = [
            {
                "shard_file": (
                    shard_filename
                ),
                "indices": list(
                    dataset_indices
                ),
            }
            for shard_filename, dataset_indices
            in sorted(
                shard_to_indices.items()
            )
        ]

        grouped_patent_count = sum(
            len(
                group[
                    "indices"
                ]
            )
            for group
            in self.shard_groups
        )

        if grouped_patent_count != len(
            dataset
        ):
            raise RuntimeError(
                "Shard-group patent-count mismatch."
            )

        self.number_of_batches = 0

        for group in self.shard_groups:
            group_size = len(
                group[
                    "indices"
                ]
            )

            if self.drop_last:
                self.number_of_batches += (
                    group_size
                    // self.batch_size
                )
            else:
                self.number_of_batches += (
                    math.ceil(
                        group_size
                        / self.batch_size
                    )
                )

        print(
            "[ShardAwareBatchSampler] "
            f"split={dataset.split}, "
            f"patents={len(dataset):,}, "
            f"shards={len(self.shard_groups):,}, "
            f"batches={self.number_of_batches:,}, "
            f"shuffle={self.shuffle}"
        )

    def set_epoch(
        self,
        epoch,
    ):
        self.epoch = int(
            epoch
        )

    def __len__(
        self,
    ):
        return (
            self.number_of_batches
        )

    def __iter__(
        self,
    ):
        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            self.seed
            + self.epoch
        )

        number_of_shards = len(
            self.shard_groups
        )

        if self.shuffle:
            shard_order = torch.randperm(
                number_of_shards,
                generator=generator,
            ).tolist()
        else:
            shard_order = list(
                range(
                    number_of_shards
                )
            )

        for shard_position in (
            shard_order
        ):
            group_indices = list(
                self.shard_groups[
                    shard_position
                ][
                    "indices"
                ]
            )

            if self.shuffle:
                within_shard_order = (
                    torch.randperm(
                        len(
                            group_indices
                        ),
                        generator=generator,
                    ).tolist()
                )

                group_indices = [
                    group_indices[
                        position
                    ]
                    for position
                    in within_shard_order
                ]

            for start_position in range(
                0,
                len(
                    group_indices
                ),
                self.batch_size,
            ):
                batch_indices = group_indices[
                    start_position:
                    start_position
                    + self.batch_size
                ]

                if (
                    self.drop_last
                    and len(
                        batch_indices
                    )
                    < self.batch_size
                ):
                    continue

                yield batch_indices


SECTION2_SEED = int(
    getattr(
        CONFIG,
        "seed",
        42,
    )
)

train_batch_sampler = (
    ShardAwareBatchSampler(
        dataset=train_dataset,
        batch_size=(
            PATENTS_PER_BATCH
        ),
        shuffle=True,
        seed=SECTION2_SEED,
        drop_last=False,
    )
)

dev_batch_sampler = (
    ShardAwareBatchSampler(
        dataset=dev_dataset,
        batch_size=(
            PATENTS_PER_BATCH
        ),
        shuffle=False,
        seed=SECTION2_SEED,
        drop_last=False,
    )
)

test_batch_sampler = (
    ShardAwareBatchSampler(
        dataset=test_dataset,
        batch_size=(
            PATENTS_PER_BATCH
        ),
        shuffle=False,
        seed=SECTION2_SEED,
        drop_last=False,
    )
)

common_loader_options = {
    "collate_fn": (
        patent_collate_fn
    ),
    "num_workers": (
        DATALOADER_NUM_WORKERS
    ),
    "pin_memory": (
        torch.cuda.is_available()
    ),
}

train_loader = DataLoader(
    train_dataset,
    batch_sampler=(
        train_batch_sampler
    ),
    **common_loader_options,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_sampler=(
        dev_batch_sampler
    ),
    **common_loader_options,
)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=(
        test_batch_sampler
    ),
    **common_loader_options,
)

# Kept for checkpoint RNG compatibility in Section 6.
loader_generator = (
    torch.Generator()
)

loader_generator.manual_seed(
    SECTION2_SEED
)

print("\n=== SHARD-AWARE DATA LOADER SUMMARY ===")
print(
    f"train: patents="
    f"{len(train_dataset):,}, "
    f"batches={len(train_loader):,}"
)
print(
    f"dev  : patents="
    f"{len(dev_dataset):,}, "
    f"batches={len(dev_loader):,}"
)
print(
    f"test : patents="
    f"{len(test_dataset):,}, "
    f"batches={len(test_loader):,}"
)

print(
    "\n[PASS] Section 2 parts 0-8 completed. "
    "Now run Section 2 part 9 validation."
)

# ============================================================
# 9. Create and validate a real sample_batch
# ============================================================

sample_batch = next(iter(train_loader))
HAS_REAL_BATCH = True


def validate_patent_batch(batch):
    required_keys = {
        "token_embeddings",
        "token_edge_index",
        "token_edge_weight",
        "token_to_claim",
        "claim_edge_index",
        "claim_parent_index",
        "claim_child_index",
        "claim_to_patent",
        "claim_depth",
        "bow",
        "patent_ids",
        "claim_ids",
        "claim_keys",
        "truncated",
        "num_patents",
        "num_claims",
        "num_tokens",
    }

    missing_keys = required_keys - set(batch.keys())

    if missing_keys:
        raise KeyError(
            f"Missing batch keys: {sorted(missing_keys)}"
        )

    num_patents = int(batch["num_patents"])
    num_claims = int(batch["num_claims"])
    num_tokens = int(batch["num_tokens"])

    if num_patents <= 0:
        raise ValueError(
            f"num_patents must be positive, got {num_patents}."
        )

    if num_claims <= 0:
        raise ValueError(
            f"num_claims must be positive, got {num_claims}."
        )

    if num_tokens <= 0:
        raise ValueError(
            f"num_tokens must be positive, got {num_tokens}."
        )

    token_embeddings = batch["token_embeddings"]
    token_edge_index = batch["token_edge_index"]
    token_edge_weight = batch["token_edge_weight"]
    token_to_claim = batch["token_to_claim"]

    claim_edge_index = batch["claim_edge_index"]
    claim_parent_index = batch["claim_parent_index"]
    claim_child_index = batch["claim_child_index"]
    claim_to_patent = batch["claim_to_patent"]
    claim_depth = batch["claim_depth"]

    bow = batch["bow"]
    truncated = batch["truncated"]

    # --------------------------------------------------------
    # 9.1 Tensor type and dimension validation
    # --------------------------------------------------------

    tensor_fields = {
        "token_embeddings": token_embeddings,
        "token_edge_index": token_edge_index,
        "token_edge_weight": token_edge_weight,
        "token_to_claim": token_to_claim,
        "claim_edge_index": claim_edge_index,
        "claim_parent_index": claim_parent_index,
        "claim_child_index": claim_child_index,
        "claim_to_patent": claim_to_patent,
        "claim_depth": claim_depth,
        "bow": bow,
        "truncated": truncated,
    }

    for field_name, tensor in tensor_fields.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{field_name} must be a torch.Tensor, "
                f"got {type(tensor)}."
            )

    if token_embeddings.ndim != 2:
        raise ValueError(
            "token_embeddings must have shape [T, D], "
            f"got {tuple(token_embeddings.shape)}."
        )

    if token_embeddings.shape[0] != num_tokens:
        raise ValueError(
            "Token-count mismatch: "
            f"tensor={token_embeddings.shape[0]}, "
            f"metadata={num_tokens}."
        )

    detected_hidden_dim = int(
        token_embeddings.shape[1]
    )

    if detected_hidden_dim <= 0:
        raise ValueError(
            "PLM hidden dimension must be positive."
        )

    configured_hidden_dim = getattr(
        CONFIG,
        "plm_hidden_dim",
        None,
    )

    if configured_hidden_dim is None:
        CONFIG.plm_hidden_dim = detected_hidden_dim
    elif int(configured_hidden_dim) != detected_hidden_dim:
        raise ValueError(
            "PLM hidden-dimension mismatch: "
            f"CONFIG={configured_hidden_dim}, "
            f"observed={detected_hidden_dim}."
        )

    if token_edge_index.dtype != torch.long:
        raise TypeError(
            "token_edge_index must have dtype torch.long."
        )

    if token_to_claim.dtype != torch.long:
        raise TypeError(
            "token_to_claim must have dtype torch.long."
        )

    if claim_edge_index.dtype != torch.long:
        raise TypeError(
            "claim_edge_index must have dtype torch.long."
        )

    if claim_parent_index.dtype != torch.long:
        raise TypeError(
            "claim_parent_index must have dtype torch.long."
        )

    if claim_child_index.dtype != torch.long:
        raise TypeError(
            "claim_child_index must have dtype torch.long."
        )

    if claim_to_patent.dtype != torch.long:
        raise TypeError(
            "claim_to_patent must have dtype torch.long."
        )

    if claim_depth.dtype != torch.long:
        raise TypeError(
            "claim_depth must have dtype torch.long."
        )

    if truncated.dtype != torch.bool:
        raise TypeError(
            "truncated must have dtype torch.bool."
        )

    # --------------------------------------------------------
    # 9.2 Basic shape validation
    # --------------------------------------------------------

    if token_to_claim.shape != (num_tokens,):
        raise ValueError(
            "token_to_claim shape mismatch: "
            f"observed={tuple(token_to_claim.shape)}, "
            f"expected={(num_tokens,)}."
        )

    if claim_depth.shape != (num_claims,):
        raise ValueError(
            "claim_depth shape mismatch: "
            f"observed={tuple(claim_depth.shape)}, "
            f"expected={(num_claims,)}."
        )

    if claim_to_patent.shape != (num_claims,):
        raise ValueError(
            "claim_to_patent shape mismatch: "
            f"observed={tuple(claim_to_patent.shape)}, "
            f"expected={(num_claims,)}."
        )

    if truncated.shape != (num_claims,):
        raise ValueError(
            "truncated shape mismatch: "
            f"observed={tuple(truncated.shape)}, "
            f"expected={(num_claims,)}."
        )

    expected_bow_shape = (
        num_claims,
        len(VOCAB),
    )

    if bow.shape != expected_bow_shape:
        raise ValueError(
            "BoW shape mismatch: "
            f"observed={tuple(bow.shape)}, "
            f"expected={expected_bow_shape}."
        )

    # --------------------------------------------------------
    # 9.3 Finite-value validation
    # --------------------------------------------------------

    if not torch.isfinite(token_embeddings).all():
        raise FloatingPointError(
            "Non-finite token embeddings detected."
        )

    if not torch.isfinite(token_edge_weight).all():
        raise FloatingPointError(
            "Non-finite token edge weights detected."
        )

    if not torch.isfinite(bow).all():
        raise FloatingPointError(
            "Non-finite BoW values detected."
        )

    if torch.any(token_edge_weight < 0):
        raise ValueError(
            "Negative token edge weights detected. "
            "Expected non-negative exp(cosine/tau) weights."
        )

    if torch.any(bow < 0):
        raise ValueError(
            "Negative BoW values detected."
        )

    # --------------------------------------------------------
    # 9.4 Token-to-claim validation
    # --------------------------------------------------------

    if token_to_claim.numel() > 0:
        minimum_claim_index = int(
            token_to_claim.min().item()
        )

        maximum_claim_index = int(
            token_to_claim.max().item()
        )

        if (
            minimum_claim_index < 0
            or maximum_claim_index >= num_claims
        ):
            raise IndexError(
                "token_to_claim contains invalid claim indices. "
                f"Observed=[{minimum_claim_index}, "
                f"{maximum_claim_index}], "
                f"valid=[0, {num_claims - 1}]."
            )

    token_counts = torch.bincount(
        token_to_claim,
        minlength=num_claims,
    )

    if token_counts.shape[0] != num_claims:
        raise ValueError(
            "token_counts shape mismatch."
        )

    if torch.any(token_counts == 0):
        empty_token_claims = torch.nonzero(
            token_counts == 0,
            as_tuple=False,
        ).flatten().tolist()

        raise ValueError(
            "Claims without token features detected: "
            f"{empty_token_claims[:20]}"
        )

    # --------------------------------------------------------
    # 9.5 Token graph validation
    # --------------------------------------------------------

    validate_edge_index(
        token_edge_index,
        num_tokens,
        name="token_edge_index",
    )

    if token_edge_weight.ndim != 1:
        raise ValueError(
            "token_edge_weight must have shape [E], "
            f"got {tuple(token_edge_weight.shape)}."
        )

    if (
        token_edge_weight.shape[0]
        != token_edge_index.shape[1]
    ):
        raise ValueError(
            "Token edge-count mismatch: "
            f"edge_index={token_edge_index.shape[1]}, "
            f"edge_weight={token_edge_weight.shape[0]}."
        )

    if token_edge_index.shape[1] > 0:
        source_claim = token_to_claim[
            token_edge_index[0]
        ]

        target_claim = token_to_claim[
            token_edge_index[1]
        ]

        if not torch.equal(
            source_claim,
            target_claim,
        ):
            crossing_edge_mask = (
                source_claim != target_claim
            )

            number_of_crossing_edges = int(
                crossing_edge_mask.sum().item()
            )

            raise ValueError(
                "Token edges cross claim boundaries. "
                f"Crossing edges={number_of_crossing_edges}."
            )

    # --------------------------------------------------------
    # 9.6 Claim-to-patent validation
    # --------------------------------------------------------

    if claim_to_patent.numel() > 0:
        minimum_patent_index = int(
            claim_to_patent.min().item()
        )

        maximum_patent_index = int(
            claim_to_patent.max().item()
        )

        if (
            minimum_patent_index < 0
            or maximum_patent_index >= num_patents
        ):
            raise IndexError(
                "claim_to_patent contains invalid patent indices. "
                f"Observed=[{minimum_patent_index}, "
                f"{maximum_patent_index}], "
                f"valid=[0, {num_patents - 1}]."
            )

    patent_claim_counts = torch.bincount(
        claim_to_patent,
        minlength=num_patents,
    )

    if torch.any(patent_claim_counts == 0):
        raise ValueError(
            "A patent in the batch contains no claims."
        )

    # --------------------------------------------------------
    # 9.7 Claim depth validation
    # --------------------------------------------------------

    if torch.any(claim_depth < 0):
        raise ValueError(
            "Negative claim depths detected."
        )

    # --------------------------------------------------------
    # 9.8 BoW validation with empty-row support
    # --------------------------------------------------------

    bow_sums = bow.sum(dim=-1)

    valid_bow_mask = (
        bow_sums > 1e-10
    )

    empty_bow_mask = (
        ~valid_bow_mask
    )

    if valid_bow_mask.any():
        valid_bow_sums = (
            bow_sums[valid_bow_mask]
        )

        if not torch.allclose(
            valid_bow_sums,
            torch.ones_like(
                valid_bow_sums
            ),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                "Non-empty BoW rows are not L1-normalized. "
                f"Minimum row sum="
                f"{valid_bow_sums.min().item():.6f}, "
                f"maximum row sum="
                f"{valid_bow_sums.max().item():.6f}."
            )

    if empty_bow_mask.any():
        empty_bow_rows = (
            bow[empty_bow_mask]
        )

        if not torch.allclose(
            empty_bow_rows,
            torch.zeros_like(
                empty_bow_rows
            ),
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "Empty BoW rows must be exact zero vectors."
            )

    num_valid_bow_claims = int(
        valid_bow_mask.sum().item()
    )

    num_empty_bow_claims = int(
        empty_bow_mask.sum().item()
    )

    empty_bow_ratio = (
        num_empty_bow_claims
        / max(num_claims, 1)
    )

    # --------------------------------------------------------
    # 9.9 Claim dependency graph validation
    # --------------------------------------------------------

    validate_edge_index(
        claim_edge_index,
        num_claims,
        name="claim_edge_index",
    )

    if not torch.equal(
        claim_parent_index,
        claim_edge_index[0],
    ):
        raise ValueError(
            "claim_parent_index does not match "
            "claim_edge_index[0]."
        )

    if not torch.equal(
        claim_child_index,
        claim_edge_index[1],
    ):
        raise ValueError(
            "claim_child_index does not match "
            "claim_edge_index[1]."
        )

    parent_index = claim_edge_index[0]
    child_index = claim_edge_index[1]

    non_adjacent_edges = 0

    if parent_index.numel() > 0:
        same_patent = (
            claim_to_patent[parent_index]
            == claim_to_patent[child_index]
        )

        if not same_patent.all():
            number_of_cross_patent_edges = int(
                (~same_patent).sum().item()
            )

            raise ValueError(
                "Claim edges cross patent boundaries. "
                f"Cross-patent edges="
                f"{number_of_cross_patent_edges}."
            )

        depth_difference = (
            claim_depth[child_index]
            - claim_depth[parent_index]
        )

        if torch.any(
            depth_difference <= 0
        ):
            invalid_depth_edges = int(
                (
                    depth_difference <= 0
                ).sum().item()
            )

            raise ValueError(
                "A dependency edge does not increase depth. "
                f"Invalid edges={invalid_depth_edges}."
            )

        non_adjacent_edges = int(
            (
                depth_difference != 1
            ).sum().item()
        )

    # --------------------------------------------------------
    # 9.10 Metadata validation
    # --------------------------------------------------------

    if len(batch["patent_ids"]) != num_patents:
        raise ValueError(
            "Patent-ID count mismatch: "
            f"observed={len(batch['patent_ids'])}, "
            f"expected={num_patents}."
        )

    if len(batch["claim_ids"]) != num_claims:
        raise ValueError(
            "Claim-ID count mismatch: "
            f"observed={len(batch['claim_ids'])}, "
            f"expected={num_claims}."
        )

    if len(batch["claim_keys"]) != num_claims:
        raise ValueError(
            "Claim-key count mismatch: "
            f"observed={len(batch['claim_keys'])}, "
            f"expected={num_claims}."
        )

    if len(set(batch["claim_keys"])) != num_claims:
        raise ValueError(
            "Duplicate claim keys detected in the batch."
        )

    # --------------------------------------------------------
    # 9.11 Validation summary
    # --------------------------------------------------------

    return {
        "num_patents": num_patents,
        "num_claims": num_claims,
        "num_tokens": num_tokens,
        "num_token_edges": int(
            token_edge_index.shape[1]
        ),
        "num_claim_edges": int(
            claim_edge_index.shape[1]
        ),
        "valid_bow_claims": (
            num_valid_bow_claims
        ),
        "empty_bow_claims": (
            num_empty_bow_claims
        ),
        "empty_bow_ratio": (
            empty_bow_ratio
        ),
        "vocab_size": int(
            bow.shape[1]
        ),
        "plm_hidden_dim": (
            detected_hidden_dim
        ),
        "max_depth": int(
            claim_depth.max().item()
        ),
        "non_adjacent_edges": (
            non_adjacent_edges
        ),
        "min_tokens_per_claim": int(
            token_counts.min().item()
        ),
        "max_tokens_per_claim": int(
            token_counts.max().item()
        ),
        "min_claims_per_patent": int(
            patent_claim_counts.min().item()
        ),
        "max_claims_per_patent": int(
            patent_claim_counts.max().item()
        ),
        "truncated_claims": int(
            truncated.sum().item()
        ),
    }


batch_validation_summary = validate_patent_batch(
    sample_batch
)

print("\n=== REAL SAMPLE BATCH ===")

for key, value in sample_batch.items():
    if isinstance(value, torch.Tensor):
        print(
            f"{key:24s}: "
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}"
        )
    else:
        print(
            f"{key:24s}: "
            f"type={type(value).__name__}"
        )

print("\n=== BATCH VALIDATION ===")

for key, value in batch_validation_summary.items():
    print(
        f"{key:24s}: {value}"
    )

print(
    "\nBoW policy: non-empty rows are L1-normalized; "
    "empty rows remain zero vectors."
)


# ============================================================
# SECTION 3: Neural Model Components
# TokenGCNEncoder / DependencyEncoder /
# VariationalEncoder / TopicWordDecoder
# ============================================================

from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# A. Preconditions and dimensions
# ------------------------------------------------------------

if "DEVICE" not in globals():
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

if "CONFIG" not in globals():
    raise RuntimeError(
        "CONFIG is not defined. Run Section 0 first."
    )

if "sample_batch" not in globals():
    raise RuntimeError(
        "sample_batch is not defined. Section 2 must create and "
        "validate a real patent-level batch before Section 3."
    )

if not isinstance(sample_batch, dict):
    raise TypeError(
        f"sample_batch must be a dictionary, got {type(sample_batch)}."
    )

if CONFIG.num_topics < 2:
    raise ValueError(
        f"num_topics must be at least 2, got {CONFIG.num_topics}."
    )

if hasattr(CONFIG, "latent_dim"):
    if CONFIG.latent_dim != CONFIG.num_topics:
        raise ValueError(
            "The logistic-normal latent dimension must equal "
            f"num_topics. Got latent_dim={CONFIG.latent_dim} and "
            f"num_topics={CONFIG.num_topics}."
        )

if "token_embeddings" in sample_batch:
    FEATURE_KEY = "token_embeddings"
elif "token_x" in sample_batch:
    FEATURE_KEY = "token_x"
else:
    raise KeyError(
        "sample_batch must contain 'token_embeddings' or 'token_x'."
    )

feature_tensor = sample_batch[FEATURE_KEY]

if feature_tensor.ndim != 2:
    raise ValueError(
        "Token features must have shape [num_tokens, hidden_dim], "
        f"got {tuple(feature_tensor.shape)}."
    )

PLM_HIDDEN_DIM = int(feature_tensor.shape[-1])

configured_hidden_dim = getattr(
    CONFIG,
    "plm_hidden_dim",
    None,
)

if configured_hidden_dim is not None:
    if int(configured_hidden_dim) != PLM_HIDDEN_DIM:
        raise ValueError(
            "PLM hidden dimension mismatch: "
            f"CONFIG={configured_hidden_dim}, "
            f"sample_batch={PLM_HIDDEN_DIM}."
        )
else:
    CONFIG.plm_hidden_dim = PLM_HIDDEN_DIM

if "bow" not in sample_batch:
    raise KeyError(
        "sample_batch must contain 'bow' for real reconstruction-loss "
        "validation."
    )

if sample_batch["bow"].ndim != 2:
    raise ValueError(
        "sample_batch['bow'] must have shape [num_claims, vocab_size]."
    )

VOCAB_SIZE = int(sample_batch["bow"].shape[-1])

configured_vocab_size = getattr(
    CONFIG,
    "vocab_size",
    None,
)

if configured_vocab_size is not None:
    if int(configured_vocab_size) != VOCAB_SIZE:
        raise ValueError(
            "Vocabulary-size mismatch: "
            f"CONFIG={configured_vocab_size}, "
            f"sample_batch={VOCAB_SIZE}."
        )
else:
    CONFIG.vocab_size = VOCAB_SIZE

print("=== SECTION 3 CONFIG ===")
print(f"device                 : {DEVICE}")
print(f"plm_hidden_dim         : {PLM_HIDDEN_DIM}")
print(f"gcn_hidden_dim         : {CONFIG.gcn_hidden_dim}")
print(f"gcn_num_layers         : {CONFIG.gcn_num_layers}")
print(f"claim_repr_dim         : {CONFIG.claim_repr_dim}")
print(f"dep_encoder_hidden_dim : {CONFIG.dep_encoder_hidden_dim}")
print(f"num_topics             : {CONFIG.num_topics}")
print(f"vocab_size             : {CONFIG.vocab_size}")
print("batch type             : real patent-level batch")


# ------------------------------------------------------------
# B. Graph utilities
# ------------------------------------------------------------

def validate_edge_index(
    edge_index,
    num_nodes,
    name="edge_index",
):
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if edge_index.dtype != torch.long:
        raise TypeError(
            f"{name} must have dtype torch.long, "
            f"got {edge_index.dtype}."
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"{name} must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}."
        )

    if edge_index.numel() > 0:
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())

        if min_index < 0 or max_index >= num_nodes:
            raise IndexError(
                f"{name} contains invalid node indices. "
                f"Observed=[{min_index}, {max_index}], "
                f"valid=[0, {num_nodes - 1}]."
            )


def replace_with_unit_self_loops(
    edge_index,
    edge_weight,
    num_nodes,
):
    """
    Remove stored self-loops and add exactly one unit-weight
    self-loop per node.
    """

    if edge_weight.ndim != 1:
        raise ValueError(
            "edge_weight must be one-dimensional."
        )

    if edge_index.shape[1] != edge_weight.shape[0]:
        raise ValueError(
            "edge_index and edge_weight have inconsistent edge counts."
        )

    if edge_index.shape[1] > 0:
        non_self_mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, non_self_mask]
        edge_weight = edge_weight[non_self_mask]

    nodes = torch.arange(
        num_nodes,
        device=edge_index.device,
        dtype=torch.long,
    )

    self_edges = torch.stack(
        [nodes, nodes],
        dim=0,
    )

    self_weights = torch.ones(
        num_nodes,
        device=edge_weight.device,
        dtype=edge_weight.dtype,
    )

    edge_index = torch.cat(
        [edge_index, self_edges],
        dim=1,
    )
    edge_weight = torch.cat(
        [edge_weight, self_weights],
        dim=0,
    )

    return edge_index, edge_weight


def validate_token_graph_boundaries(
    edge_index,
    token_to_claim,
):
    if edge_index.shape[1] == 0:
        return

    source_claim = token_to_claim[edge_index[0]]
    target_claim = token_to_claim[edge_index[1]]

    cross_claim_mask = source_claim != target_claim

    if cross_claim_mask.any():
        count = int(cross_claim_mask.sum().item())

        raise ValueError(
            f"{count} token-graph edges cross claim boundaries. "
            "Check the collate index offsets."
        )


# ------------------------------------------------------------
# C. GCN layer
# ------------------------------------------------------------

class SimpleGCNLayer(nn.Module):
    """
    Symmetrically normalized GCN layer.

        X' = ReLU(D^{-1/2} A D^{-1/2} X W)

    This implementation assumes nonnegative graph weights:
        A_ij = exp(cos(h_i, h_j) / tau).
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_dim,
            out_dim,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        num_nodes,
    ):
        validate_edge_index(
            edge_index,
            num_nodes,
            name="token_edge_index",
        )

        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "token_edge_index and token_edge_weight have "
                "inconsistent edge counts."
            )

        if not torch.isfinite(x).all():
            raise FloatingPointError(
                "Non-finite token representations detected."
            )

        if not torch.isfinite(edge_weight).all():
            raise FloatingPointError(
                "Non-finite token edge weights detected."
            )

        if torch.any(edge_weight < 0):
            min_weight = float(edge_weight.min().item())

            raise ValueError(
                "Negative token edge weights were detected "
                f"(minimum={min_weight:.6f}). This implementation "
                "requires Section 1 to store "
                "exp(cosine_similarity / tau), not raw cosine."
            )

        source = edge_index[0]
        target = edge_index[1]

        edge_weight = edge_weight.to(
            device=x.device,
            dtype=x.dtype,
        )

        degree = torch.zeros(
            num_nodes,
            device=x.device,
            dtype=x.dtype,
        )
        degree.index_add_(
            0,
            target,
            edge_weight,
        )

        inv_sqrt_degree = degree.clamp_min(
            1e-12
        ).pow(-0.5)

        normalized_weight = (
            edge_weight
            * inv_sqrt_degree[source]
            * inv_sqrt_degree[target]
        )

        transformed = self.linear(x)
        messages = (
            transformed[source]
            * normalized_weight.unsqueeze(-1)
        )

        output = torch.zeros(
            num_nodes,
            transformed.shape[-1],
            device=transformed.device,
            dtype=transformed.dtype,
        )
        output.index_add_(
            0,
            target,
            messages,
        )

        return F.relu(output)


# ------------------------------------------------------------
# D. Token-level GCN encoder
# ------------------------------------------------------------

class TokenGCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        claim_repr_dim,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "At least one GCN layer is required."
            )

        self.in_dim = in_dim
        self.claim_repr_dim = claim_repr_dim

        self.input_projection = nn.Linear(
            in_dim,
            hidden_dim,
        )

        self.gcn_layers = nn.ModuleList(
            [
                SimpleGCNLayer(
                    hidden_dim,
                    hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            claim_repr_dim,
        )

    def forward(
        self,
        token_embeddings,
        token_edge_index,
        token_edge_weight,
        token_to_claim,
        num_claims,
    ):
        if token_embeddings.ndim != 2:
            raise ValueError(
                "token_embeddings must have shape [T, D]."
            )

        num_tokens = int(token_embeddings.shape[0])

        if num_tokens == 0:
            raise ValueError(
                "A batch with zero tokens is not allowed."
            )

        if token_embeddings.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected token dimension {self.in_dim}, "
                f"got {token_embeddings.shape[-1]}."
            )

        if token_to_claim.ndim != 1:
            raise ValueError(
                "token_to_claim must be one-dimensional."
            )

        if token_to_claim.dtype != torch.long:
            raise TypeError(
                "token_to_claim must have dtype torch.long."
            )

        if token_to_claim.shape[0] != num_tokens:
            raise ValueError(
                "token_to_claim length does not match num_tokens."
            )

        if num_claims <= 0:
            raise ValueError(
                f"num_claims must be positive, got {num_claims}."
            )

        if int(token_to_claim.min().item()) < 0:
            raise IndexError(
                "Negative claim indices detected."
            )

        if int(token_to_claim.max().item()) >= num_claims:
            raise IndexError(
                "token_to_claim contains an out-of-range claim index."
            )

        validate_edge_index(
            token_edge_index,
            num_tokens,
            name="token_edge_index",
        )

        validate_token_graph_boundaries(
            token_edge_index,
            token_to_claim,
        )

        parameter_dtype = (
            self.input_projection.weight.dtype
        )

        x = token_embeddings.to(
            dtype=parameter_dtype
        )
        x = F.relu(
            self.input_projection(x)
        )

        edge_index = token_edge_index.to(
            device=x.device,
            dtype=torch.long,
        )
        edge_weight = token_edge_weight.to(
            device=x.device,
            dtype=x.dtype,
        )

        edge_index, edge_weight = (
            replace_with_unit_self_loops(
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=num_tokens,
            )
        )

        for layer in self.gcn_layers:
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=num_tokens,
            )

        claim_sum = torch.zeros(
            num_claims,
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        claim_sum.index_add_(
            0,
            token_to_claim,
            x,
        )

        claim_count = torch.zeros(
            num_claims,
            device=x.device,
            dtype=x.dtype,
        )
        claim_count.index_add_(
            0,
            token_to_claim,
            torch.ones(
                num_tokens,
                device=x.device,
                dtype=x.dtype,
            ),
        )

        missing_claims = torch.nonzero(
            claim_count == 0,
            as_tuple=False,
        ).flatten()

        if missing_claims.numel() > 0:
            raise ValueError(
                "Claims without valid token features were detected: "
                f"{missing_claims[:20].tolist()}"
            )

        claim_mean = (
            claim_sum
            / claim_count.unsqueeze(-1)
        )

        claim_repr = self.output_projection(
            claim_mean
        )

        if not torch.isfinite(claim_repr).all():
            raise FloatingPointError(
                "Non-finite claim representations detected."
            )

        return claim_repr


# ------------------------------------------------------------
# E. Bidirectional dependency-guided claim encoder
# ------------------------------------------------------------

class DependencyEncoder(nn.Module):
    """
    Implements Eq. enc using sum aggregation:

        g_tilde_c = activation(
            W_down sum(parent representations)
            + W_up sum(child representations)
            + W_0 g_c
        ).

    claim_edge_index[0] is the parent.
    claim_edge_index[1] is the child.
    """

    def __init__(
        self,
        claim_repr_dim,
        hidden_dim,
    ):
        super().__init__()

        self.claim_repr_dim = claim_repr_dim
        self.hidden_dim = hidden_dim

        self.self_projection = nn.Linear(
            claim_repr_dim,
            hidden_dim,
            bias=True,
        )
        self.parent_projection = nn.Linear(
            claim_repr_dim,
            hidden_dim,
            bias=False,
        )
        self.child_projection = nn.Linear(
            claim_repr_dim,
            hidden_dim,
            bias=False,
        )

    @staticmethod
    def sum_aggregate(
        values,
        target_index,
        num_claims,
    ):
        output = torch.zeros(
            num_claims,
            values.shape[-1],
            device=values.device,
            dtype=values.dtype,
        )

        output.index_add_(
            0,
            target_index,
            values,
        )

        return output

    def forward(
        self,
        claim_repr,
        claim_edge_index,
        num_claims,
    ):
        if claim_repr.ndim != 2:
            raise ValueError(
                "claim_repr must have shape [C, D]."
            )

        if claim_repr.shape[0] != num_claims:
            raise ValueError(
                "claim_repr claim count does not match num_claims."
            )

        if claim_repr.shape[-1] != self.claim_repr_dim:
            raise ValueError(
                f"Expected claim dimension {self.claim_repr_dim}, "
                f"got {claim_repr.shape[-1]}."
            )

        validate_edge_index(
            claim_edge_index,
            num_claims,
            name="claim_edge_index",
        )

        if claim_edge_index.shape[1] == 0:
            parent_sum = torch.zeros_like(
                claim_repr
            )
            child_sum = torch.zeros_like(
                claim_repr
            )
        else:
            parent_index = claim_edge_index[0]
            child_index = claim_edge_index[1]

            parent_sum = self.sum_aggregate(
                values=claim_repr[parent_index],
                target_index=child_index,
                num_claims=num_claims,
            )

            child_sum = self.sum_aggregate(
                values=claim_repr[child_index],
                target_index=parent_index,
                num_claims=num_claims,
            )

        contextualized = (
            self.self_projection(claim_repr)
            + self.parent_projection(parent_sum)
            + self.child_projection(child_sum)
        )

        contextualized = F.relu(
            contextualized
        )

        if not torch.isfinite(contextualized).all():
            raise FloatingPointError(
                "Non-finite dependency-aware claim "
                "representations detected."
            )

        return contextualized


# ------------------------------------------------------------
# F. Logistic-normal variational encoder
# ------------------------------------------------------------

class VariationalEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        num_topics,
        logvar_min=-10.0,
        logvar_max=10.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_topics = num_topics
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.mean_layer = nn.Linear(
            in_dim,
            num_topics,
        )
        self.logvar_layer = nn.Linear(
            in_dim,
            num_topics,
        )

    def forward(
        self,
        claim_context,
        deterministic=None,
    ):
        if claim_context.ndim != 2:
            raise ValueError(
                "claim_context must have shape [C, D]."
            )

        if claim_context.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected context dimension {self.in_dim}, "
                f"got {claim_context.shape[-1]}."
            )

        mean = self.mean_layer(
            claim_context
        )

        logvar = self.logvar_layer(
            claim_context
        ).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )

        if deterministic is None:
            deterministic = not self.training

        if deterministic:
            latent = mean
        else:
            std = torch.exp(
                0.5 * logvar
            )
            noise = torch.randn_like(
                std
            )
            latent = mean + std * noise

        theta = F.softmax(
            latent,
            dim=-1,
        )

        if not torch.isfinite(theta).all():
            raise FloatingPointError(
                "Non-finite topic proportions detected."
            )

        return theta, mean, logvar


# ------------------------------------------------------------
# G. ProdLDA-style topic-word decoder
# ------------------------------------------------------------

class TopicWordDecoder(nn.Module):
    """
    B is the unnormalized K x V topic-word logit matrix.

        log p(w | theta) = log_softmax(theta B).

    get_beta() returns row-normalized topic-word distributions
    for diversity regularization, interpretation, and export.
    """

    def __init__(
        self,
        num_topics,
        vocab_size,
    ):
        super().__init__()

        self.num_topics = num_topics
        self.vocab_size = vocab_size

        self.topic_word_logits = nn.Parameter(
            torch.empty(
                num_topics,
                vocab_size,
            )
        )

        nn.init.normal_(
            self.topic_word_logits,
            mean=0.0,
            std=0.01,
        )

    def get_beta(self):
        return F.softmax(
            self.topic_word_logits,
            dim=-1,
        )

    def forward(self, theta):
        if theta.ndim != 2:
            raise ValueError(
                "theta must have shape [C, K]."
            )

        if theta.shape[-1] != self.num_topics:
            raise ValueError(
                f"Expected {self.num_topics} topics, "
                f"got {theta.shape[-1]}."
            )

        word_logits = (
            theta @ self.topic_word_logits
        )

        log_word_prob = F.log_softmax(
            word_logits,
            dim=-1,
        )

        if not torch.isfinite(log_word_prob).all():
            raise FloatingPointError(
                "Non-finite reconstruction log probabilities detected."
            )

        return log_word_prob, self.get_beta()


# ------------------------------------------------------------
# H. Instantiate modules
# ------------------------------------------------------------

token_gcn = TokenGCNEncoder(
    in_dim=PLM_HIDDEN_DIM,
    hidden_dim=CONFIG.gcn_hidden_dim,
    num_layers=CONFIG.gcn_num_layers,
    claim_repr_dim=CONFIG.claim_repr_dim,
).to(DEVICE)

dep_encoder = DependencyEncoder(
    claim_repr_dim=CONFIG.claim_repr_dim,
    hidden_dim=CONFIG.dep_encoder_hidden_dim,
).to(DEVICE)

var_encoder = VariationalEncoder(
    in_dim=CONFIG.dep_encoder_hidden_dim,
    num_topics=CONFIG.num_topics,
).to(DEVICE)

topic_word = TopicWordDecoder(
    num_topics=CONFIG.num_topics,
    vocab_size=CONFIG.vocab_size,
).to(DEVICE)


# ------------------------------------------------------------
# I. Prepare real batch
# ------------------------------------------------------------

def prepare_section3_batch(
    batch,
    device,
):
    token_embeddings = batch[FEATURE_KEY]

    if "claim_edge_index" in batch:
        claim_edge_index = batch[
            "claim_edge_index"
        ]
    elif (
        "claim_parent_index" in batch
        and "claim_child_index" in batch
    ):
        claim_edge_index = torch.stack(
            [
                batch["claim_parent_index"],
                batch["claim_child_index"],
            ],
            dim=0,
        )
    else:
        raise KeyError(
            "Missing claim dependency-edge indices."
        )

    if "num_claims" in batch:
        num_claims = batch["num_claims"]

        if isinstance(num_claims, torch.Tensor):
            num_claims = int(
                num_claims.item()
            )
        else:
            num_claims = int(
                num_claims
            )
    else:
        num_claims = (
            int(
                batch["token_to_claim"].max().item()
            )
            + 1
        )

    return {
        "token_embeddings": token_embeddings.to(
            device
        ),
        "token_edge_index": batch[
            "token_edge_index"
        ].to(
            device=device,
            dtype=torch.long,
        ),
        "token_edge_weight": batch[
            "token_edge_weight"
        ].to(
            device=device,
            dtype=torch.float32,
        ),
        "token_to_claim": batch[
            "token_to_claim"
        ].to(
            device=device,
            dtype=torch.long,
        ),
        "claim_edge_index": claim_edge_index.to(
            device=device,
            dtype=torch.long,
        ),
        "bow": batch["bow"].to(
            device=device,
            dtype=torch.float32,
        ),
        "num_claims": num_claims,
    }


section3_batch = prepare_section3_batch(
    sample_batch,
    DEVICE,
)


# ------------------------------------------------------------
# J. Real-batch forward and backward validation
# ------------------------------------------------------------

modules = {
    "token_gcn": token_gcn,
    "dependency_encoder": dep_encoder,
    "variational_encoder": var_encoder,
    "topic_word_decoder": topic_word,
}

for module in modules.values():
    module.train()
    module.zero_grad(
        set_to_none=True
    )

claim_repr = token_gcn(
    token_embeddings=section3_batch[
        "token_embeddings"
    ],
    token_edge_index=section3_batch[
        "token_edge_index"
    ],
    token_edge_weight=section3_batch[
        "token_edge_weight"
    ],
    token_to_claim=section3_batch[
        "token_to_claim"
    ],
    num_claims=section3_batch[
        "num_claims"
    ],
)

claim_context = dep_encoder(
    claim_repr=claim_repr,
    claim_edge_index=section3_batch[
        "claim_edge_index"
    ],
    num_claims=section3_batch[
        "num_claims"
    ],
)

theta, mean, logvar = var_encoder(
    claim_context,
    deterministic=False,
)

log_word_prob, beta = topic_word(
    theta
)

bow = section3_batch["bow"]

if bow.shape != log_word_prob.shape:
    raise ValueError(
        f"BoW and reconstruction shapes differ: "
        f"{tuple(bow.shape)} vs "
        f"{tuple(log_word_prob.shape)}."
    )

reconstruction_loss = -(
    bow * log_word_prob
).sum(dim=-1).mean()

kl_loss = -0.5 * (
    1.0
    + logvar
    - mean.pow(2)
    - logvar.exp()
).sum(dim=-1).mean()

test_loss = (
    reconstruction_loss
    + 0.01 * kl_loss
)

if not torch.isfinite(test_loss):
    raise FloatingPointError(
        f"Non-finite test loss: {test_loss.item()}."
    )

test_loss.backward()

for module_name, module in modules.items():
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    if not gradients:
        raise RuntimeError(
            f"No gradient reached {module_name}."
        )

    if not all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    ):
        raise FloatingPointError(
            f"Non-finite gradients detected in {module_name}."
        )

print("\n=== REAL-BATCH SECTION 3 CHECK ===")
print(f"claim representation : {tuple(claim_repr.shape)}")
print(f"claim context        : {tuple(claim_context.shape)}")
print(f"theta                : {tuple(theta.shape)}")
print(f"log word probability : {tuple(log_word_prob.shape)}")
print(f"beta                 : {tuple(beta.shape)}")
print(f"reconstruction loss  : {reconstruction_loss.item():.6f}")
print(f"KL loss              : {kl_loss.item():.6f}")
print(f"test loss            : {test_loss.item():.6f}")

if not torch.allclose(
    theta.sum(dim=-1),
    torch.ones_like(
        theta.sum(dim=-1)
    ),
    atol=1e-5,
):
    raise AssertionError(
        "Theta rows do not sum to one."
    )

if not torch.allclose(
    beta.sum(dim=-1),
    torch.ones_like(
        beta.sum(dim=-1)
    ),
    atol=1e-5,
):
    raise AssertionError(
        "Beta rows do not sum to one."
    )


# ------------------------------------------------------------
# K. Deterministic inference validation
# ------------------------------------------------------------

for module in modules.values():
    module.eval()

with torch.no_grad():
    claim_repr_eval = token_gcn(
        token_embeddings=section3_batch[
            "token_embeddings"
        ],
        token_edge_index=section3_batch[
            "token_edge_index"
        ],
        token_edge_weight=section3_batch[
            "token_edge_weight"
        ],
        token_to_claim=section3_batch[
            "token_to_claim"
        ],
        num_claims=section3_batch[
            "num_claims"
        ],
    )

    claim_context_eval = dep_encoder(
        claim_repr=claim_repr_eval,
        claim_edge_index=section3_batch[
            "claim_edge_index"
        ],
        num_claims=section3_batch[
            "num_claims"
        ],
    )

    theta_eval_1, _, _ = var_encoder(
        claim_context_eval,
        deterministic=True,
    )

    theta_eval_2, _, _ = var_encoder(
        claim_context_eval,
        deterministic=True,
    )

if not torch.allclose(
    theta_eval_1,
    theta_eval_2,
    atol=0.0,
    rtol=0.0,
):
    raise AssertionError(
        "Deterministic inference produced different outputs."
    )

print("[PASS] Deterministic inference check passed.")


# ------------------------------------------------------------
# L. Parameter summary
# ------------------------------------------------------------

parameter_summary = {
    module_name: sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    for module_name, module in modules.items()
}

print("\n=== PARAMETER SUMMARY ===")

for module_name, count in parameter_summary.items():
    print(f"{module_name:24s}: {count:,}")

print(
    f"{'total':24s}: "
    f"{sum(parameter_summary.values()):,}"
)

print(
    "\n[PASS] Section 3 completed with a real patent-level batch."
)


# ============================================================
# SECTION 4 - Step 1
# Adjacent-depth grouping, empirical couplings,
# marginal reconciliation, and reconciliation loss
# ============================================================

import torch


# ------------------------------------------------------------
# A. Preconditions
# ------------------------------------------------------------

if "sample_batch" not in globals():
    raise RuntimeError(
        "sample_batch is not defined. Complete Section 2 first."
    )

if "section3_batch" not in globals():
    raise RuntimeError(
        "section3_batch is not defined. Run Section 3 first."
    )

required_modules = [
    "token_gcn",
    "dep_encoder",
    "var_encoder",
]

missing_modules = [
    name
    for name in required_modules
    if name not in globals()
]

if missing_modules:
    raise RuntimeError(
        f"Missing Section 3 modules: {missing_modules}"
    )

EPS_OT = float(
    getattr(
        CONFIG,
        "eps_ot",
        1e-10,
    )
)

if EPS_OT <= 0:
    raise ValueError(
        f"eps_ot must be positive, got {EPS_OT}."
    )


# ------------------------------------------------------------
# B. Basic tensor validation
# ------------------------------------------------------------

def validate_depth_inputs(
    claim_depth,
    claim_edge_index,
    num_claims,
):
    if claim_depth.dtype != torch.long:
        raise TypeError(
            "claim_depth must have dtype torch.long."
        )

    if claim_depth.ndim != 1:
        raise ValueError(
            "claim_depth must have shape [num_claims]."
        )

    if claim_depth.shape[0] != num_claims:
        raise ValueError(
            f"claim_depth length mismatch: "
            f"{claim_depth.shape[0]} vs {num_claims}."
        )

    if torch.any(claim_depth < 0):
        raise ValueError(
            "Negative claim depths were detected."
        )

    if claim_edge_index.dtype != torch.long:
        raise TypeError(
            "claim_edge_index must have dtype torch.long."
        )

    if (
        claim_edge_index.ndim != 2
        or claim_edge_index.shape[0] != 2
    ):
        raise ValueError(
            "claim_edge_index must have shape [2, num_edges]."
        )

    if claim_edge_index.numel() > 0:
        minimum = int(
            claim_edge_index.min().item()
        )
        maximum = int(
            claim_edge_index.max().item()
        )

        if minimum < 0 or maximum >= num_claims:
            raise IndexError(
                "claim_edge_index contains invalid indices. "
                f"Observed=[{minimum}, {maximum}], "
                f"valid=[0, {num_claims - 1}]."
            )


# ------------------------------------------------------------
# C. Adjacent-depth edge grouping
# ------------------------------------------------------------

def build_depth_edge_groups(
    claim_depth,
    claim_edge_index,
):
    """
    Retain edges satisfying:

        depth(child) = depth(parent) + 1.

    Edges with a larger positive depth gap are valid dependency
    edges but are excluded from the adjacent-depth MMOT chain.
    Non-increasing edges are treated as preprocessing errors.
    """

    num_claims = int(
        claim_depth.shape[0]
    )

    validate_depth_inputs(
        claim_depth=claim_depth,
        claim_edge_index=claim_edge_index,
        num_claims=num_claims,
    )

    total_edges = int(
        claim_edge_index.shape[1]
    )

    if total_edges == 0:
        raise ValueError(
            "The batch contains no dependency edges."
        )

    parent_index = claim_edge_index[0]
    child_index = claim_edge_index[1]

    parent_depth = claim_depth[
        parent_index
    ]
    child_depth = claim_depth[
        child_index
    ]

    depth_difference = (
        child_depth - parent_depth
    )

    non_increasing_mask = (
        depth_difference <= 0
    )

    if non_increasing_mask.any():
        invalid_positions = torch.nonzero(
            non_increasing_mask,
            as_tuple=False,
        ).flatten()

        examples = []

        for position in invalid_positions[
            :10
        ].tolist():
            parent = int(
                parent_index[position].item()
            )
            child = int(
                child_index[position].item()
            )

            examples.append(
                {
                    "parent_index": parent,
                    "child_index": child,
                    "parent_depth": int(
                        claim_depth[parent].item()
                    ),
                    "child_depth": int(
                        claim_depth[child].item()
                    ),
                }
            )

        raise ValueError(
            "Non-increasing dependency edges were detected. "
            f"Count={invalid_positions.numel()}, "
            f"examples={examples}"
        )

    adjacent_mask = (
        depth_difference == 1
    )

    non_adjacent_mask = (
        depth_difference > 1
    )

    adjacent_parent = parent_index[
        adjacent_mask
    ]
    adjacent_child = child_index[
        adjacent_mask
    ]
    adjacent_parent_depth = parent_depth[
        adjacent_mask
    ]

    adjacent_edges = int(
        adjacent_mask.sum().item()
    )
    excluded_edges = int(
        non_adjacent_mask.sum().item()
    )

    if adjacent_edges == 0:
        raise ValueError(
            "No adjacent-depth dependency edges remain "
            "after filtering."
        )

    # Longest-path depth of the batch.
    max_depth = int(
        claim_depth.max().item()
    )

    if max_depth < 1:
        raise ValueError(
            "The batch maximum depth must be at least one."
        )

    groups = {}

    for depth in range(max_depth):
        depth_mask = (
            adjacent_parent_depth == depth
        )

        parent_at_depth = adjacent_parent[
            depth_mask
        ]
        child_at_next_depth = adjacent_child[
            depth_mask
        ]

        edge_count = int(
            parent_at_depth.numel()
        )

        if edge_count == 0:
            raise ValueError(
                "The adjacent-depth chain is incomplete. "
                f"No edges were found for depth "
                f"{depth}->{depth + 1}. "
                "Do not replace the missing marginal with a "
                "uniform distribution."
            )

        groups[depth] = {
            "parent_idx": parent_at_depth,
            "child_idx": child_at_next_depth,
            "n": edge_count,
        }

    diagnostics = {
        "max_depth": max_depth,
        "total_dependency_edges": total_edges,
        "adjacent_edges_used": adjacent_edges,
        "non_adjacent_edges_excluded": excluded_edges,
        "adjacent_edge_coverage": (
            adjacent_edges / total_edges
        ),
    }

    return groups, max_depth, diagnostics


# ------------------------------------------------------------
# D. Per-depth empirical coupling and marginals
# ------------------------------------------------------------

def compute_depth_transition_stats(
    theta,
    claim_depth,
    claim_edge_index,
):
    if theta.ndim != 2:
        raise ValueError(
            "theta must have shape [num_claims, num_topics]."
        )

    if not torch.isfinite(theta).all():
        raise FloatingPointError(
            "Non-finite theta values were detected."
        )

    if torch.any(theta < 0):
        raise ValueError(
            "Negative topic proportions were detected."
        )

    theta_row_sums = theta.sum(
        dim=-1
    )

    if not torch.allclose(
        theta_row_sums,
        torch.ones_like(theta_row_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Theta rows do not sum to one."
        )

    groups, max_depth, diagnostics = (
        build_depth_edge_groups(
            claim_depth=claim_depth,
            claim_edge_index=claim_edge_index,
        )
    )

    stats = {}

    for depth in range(max_depth):
        group = groups[depth]
        edge_count = group["n"]

        parent_theta = theta[
            group["parent_idx"]
        ]
        child_theta = theta[
            group["child_idx"]
        ]

        empirical_coupling = torch.einsum(
            "ei,ej->ij",
            parent_theta,
            child_theta,
        ) / float(edge_count)

        parent_marginal = (
            empirical_coupling.sum(dim=1)
        )
        child_marginal = (
            empirical_coupling.sum(dim=0)
        )

        direct_parent_marginal = (
            parent_theta.mean(dim=0)
        )
        direct_child_marginal = (
            child_theta.mean(dim=0)
        )

        if not torch.allclose(
            parent_marginal,
            direct_parent_marginal,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError(
                f"Parent marginal mismatch at depth {depth}."
            )

        if not torch.allclose(
            child_marginal,
            direct_child_marginal,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError(
                f"Child marginal mismatch at depth {depth}."
            )

        if not torch.allclose(
            empirical_coupling.sum(),
            torch.ones(
                (),
                device=theta.device,
                dtype=theta.dtype,
            ),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError(
                f"Q at depth {depth} does not sum to one."
            )

        stats[depth] = {
            "Q": empirical_coupling,
            "p": parent_marginal,
            "q": child_marginal,
            "n": edge_count,
            "parent_idx": group[
                "parent_idx"
            ],
            "child_idx": group[
                "child_idx"
            ],
        }

    return stats, max_depth, diagnostics


# ------------------------------------------------------------
# E. Numerically stable KL divergence
# ------------------------------------------------------------

def normalize_probability_vector(
    probability,
    eps=EPS_OT,
):
    probability = probability.clamp_min(
        eps
    )

    return probability / probability.sum(
        dim=-1,
        keepdim=True,
    )


def safe_kl(
    first,
    second,
    eps=EPS_OT,
):
    """
    KL(first || second), after epsilon smoothing and
    renormalization.
    """

    first_safe = normalize_probability_vector(
        first,
        eps=eps,
    )
    second_safe = normalize_probability_vector(
        second,
        eps=eps,
    )

    return torch.sum(
        first_safe
        * (
            torch.log(first_safe)
            - torch.log(second_safe)
        )
    )


# ------------------------------------------------------------
# F. Marginal reconciliation
# ------------------------------------------------------------

def reconcile_depth_marginals(
    stats,
    max_depth,
    theta_reference,
):
    """
    Endpoints:
        bar_p[0] = p^(0)
        bar_p[D] = q^(D)

    Interior depth d:
        bar_p[d] =
          (n^(d-1) q^(d) + n^(d) p^(d))
          / (n^(d-1) + n^(d))
    """

    if max_depth < 1:
        raise ValueError(
            "max_depth must be at least one."
        )

    expected_transitions = set(
        range(max_depth)
    )

    observed_transitions = set(
        stats.keys()
    )

    if (
        expected_transitions
        != observed_transitions
    ):
        missing = (
            expected_transitions
            - observed_transitions
        )

        raise ValueError(
            f"Missing depth transitions: {sorted(missing)}"
        )

    reconciled = {
        0: normalize_probability_vector(
            stats[0]["p"]
        ),
        max_depth: normalize_probability_vector(
            stats[max_depth - 1]["q"]
        ),
    }

    reconciliation_terms = []

    for depth in range(
        1,
        max_depth,
    ):
        previous_child = stats[
            depth - 1
        ]["q"]

        current_parent = stats[
            depth
        ]["p"]

        previous_count = int(
            stats[depth - 1]["n"]
        )
        current_count = int(
            stats[depth]["n"]
        )

        denominator = (
            previous_count
            + current_count
        )

        if denominator <= 0:
            raise ValueError(
                f"Invalid edge-count denominator at depth {depth}."
            )

        reconciled_depth = (
            previous_count * previous_child
            + current_count * current_parent
        ) / float(denominator)

        reconciled_depth = (
            normalize_probability_vector(
                reconciled_depth
            )
        )

        reconciled[depth] = (
            reconciled_depth
        )

        reconciliation_terms.append(
            safe_kl(
                previous_child,
                reconciled_depth,
            )
            + safe_kl(
                current_parent,
                reconciled_depth,
            )
        )

    if reconciliation_terms:
        reconciliation_loss = (
            torch.stack(
                reconciliation_terms
            ).mean()
        )
    else:
        # Keep a differentiable zero when D=1.
        reconciliation_loss = (
            theta_reference.sum() * 0.0
        )

    for depth in range(
        max_depth + 1
    ):
        marginal = reconciled[depth]

        if not torch.isfinite(
            marginal
        ).all():
            raise FloatingPointError(
                f"Non-finite reconciled marginal "
                f"at depth {depth}."
            )

        if not torch.allclose(
            marginal.sum(),
            torch.ones(
                (),
                device=marginal.device,
                dtype=marginal.dtype,
            ),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError(
                f"Reconciled marginal at depth "
                f"{depth} does not sum to one."
            )

    return reconciled, reconciliation_loss


# ------------------------------------------------------------
# G. Real-batch sanity check
# ------------------------------------------------------------

print("=== SECTION 4 STEP 1 SANITY CHECK ===")

claim_depth_s4 = sample_batch[
    "claim_depth"
].to(
    device=DEVICE,
    dtype=torch.long,
)

claim_edge_index_s4 = sample_batch[
    "claim_edge_index"
].to(
    device=DEVICE,
    dtype=torch.long,
)

for module in [
    token_gcn,
    dep_encoder,
    var_encoder,
]:
    module.train()

claim_repr_s4 = token_gcn(
    token_embeddings=section3_batch[
        "token_embeddings"
    ],
    token_edge_index=section3_batch[
        "token_edge_index"
    ],
    token_edge_weight=section3_batch[
        "token_edge_weight"
    ],
    token_to_claim=section3_batch[
        "token_to_claim"
    ],
    num_claims=section3_batch[
        "num_claims"
    ],
)

claim_context_s4 = dep_encoder(
    claim_repr=claim_repr_s4,
    claim_edge_index=section3_batch[
        "claim_edge_index"
    ],
    num_claims=section3_batch[
        "num_claims"
    ],
)

theta_s4, mean_s4, logvar_s4 = (
    var_encoder(
        claim_context_s4,
        deterministic=False,
    )
)

stats_s4, max_depth_s4, edge_diagnostics_s4 = (
    compute_depth_transition_stats(
        theta=theta_s4,
        claim_depth=claim_depth_s4,
        claim_edge_index=claim_edge_index_s4,
    )
)

bar_p_s4, reconciliation_loss_s4 = (
    reconcile_depth_marginals(
        stats=stats_s4,
        max_depth=max_depth_s4,
        theta_reference=theta_s4,
    )
)

print("\n=== EDGE DIAGNOSTICS ===")

for key, value in (
    edge_diagnostics_s4.items()
):
    print(f"{key:30s}: {value}")

print("\n=== DEPTH TRANSITIONS ===")

for depth in range(
    max_depth_s4
):
    transition = stats_s4[depth]

    print(
        f"depth {depth}->{depth + 1}: "
        f"edges={transition['n']}, "
        f"Q_sum={transition['Q'].sum().item():.6f}, "
        f"p_sum={transition['p'].sum().item():.6f}, "
        f"q_sum={transition['q'].sum().item():.6f}"
    )

print("\n=== RECONCILED MARGINALS ===")

for depth in range(
    max_depth_s4 + 1
):
    print(
        f"bar_p[{depth}]: "
        f"shape={tuple(bar_p_s4[depth].shape)}, "
        f"sum={bar_p_s4[depth].sum().item():.6f}"
    )

print(
    f"\nL_recon = "
    f"{reconciliation_loss_s4.item():.6f}"
)
print(
    f"L_recon requires_grad = "
    f"{reconciliation_loss_s4.requires_grad}"
)

if not torch.isfinite(
    reconciliation_loss_s4
):
    raise FloatingPointError(
        "Non-finite reconciliation loss detected."
    )

print(
    "\n[PASS] Section 4 Step 1 completed."
)

# ============================================================
# SECTION 4 - Steps 2, 3, and 4
# Isotonic anchor, reference-regularized Sinkhorn,
# hierarchy loss, and separate phi update
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# A. Configuration
# ============================================================

OT_MARGIN = float(
    getattr(
        CONFIG,
        "ot_margin",
        0.25,
    )
)

OT_EPSILON = float(
    getattr(
        CONFIG,
        "ot_epsilon",
        0.05,
    )
)

OT_REFERENCE_RHO = float(
    getattr(
        CONFIG,
        "ot_rho",
        0.10,
    )
)

SINKHORN_ITERS = int(
    getattr(
        CONFIG,
        "sinkhorn_iters",
        100,
    )
)

SINKHORN_TOL = float(
    getattr(
        CONFIG,
        "sinkhorn_tol",
        1e-5,
    )
)

GAP_WEIGHT = float(
    getattr(
        CONFIG,
        "gamma_gap",
        0.10,
    )
)

PHI_LEARNING_RATE = float(
    getattr(
        CONFIG,
        "phi_learning_rate",
        float(
            getattr(
                CONFIG,
                "learning_rate",
                1e-3,
            )
        ) * 0.1,
    )
)

PHI_WARMUP_EPOCHS = int(
    getattr(
        CONFIG,
        "hierarchy_warmup_epochs",
        5,
    )
)

if not 0.0 < OT_MARGIN <= 1.0:
    raise ValueError(
        f"OT_MARGIN must be in (0, 1], got {OT_MARGIN}."
    )

if OT_EPSILON <= 0:
    raise ValueError(
        f"OT_EPSILON must be positive, got {OT_EPSILON}."
    )

if not 0.0 <= OT_REFERENCE_RHO <= 1.0:
    raise ValueError(
        "OT_REFERENCE_RHO must be in [0, 1]."
    )

if SINKHORN_ITERS < 1:
    raise ValueError(
        "SINKHORN_ITERS must be positive."
    )

print("=== SECTION 4 CONFIGURATION ===")
print(f"Topics                 : {CONFIG.num_topics}")
print(f"Directional margin     : {OT_MARGIN}")
print(f"OT epsilon             : {OT_EPSILON}")
print(f"Reference smoothing rho: {OT_REFERENCE_RHO}")
print(f"Sinkhorn iterations    : {SINKHORN_ITERS}")
print(f"Sinkhorn tolerance     : {SINKHORN_TOL}")
print(f"Gap weight             : {GAP_WEIGHT}")
print(f"Phi learning rate      : {PHI_LEARNING_RATE}")
print(f"Phi warm-up epochs     : {PHI_WARMUP_EPOCHS}")


# ============================================================
# B. Step 2: Isotonic topic anchor
# ============================================================

class IsotonicTopicAnchor(nn.Module):
    """
    Learn strictly increasing topic coordinates:

        a_1 = 0,
        a_k = sum_{l<k} softplus(phi_l) / sum_l softplus(phi_l),
        a_K = 1.

    Directional cost:

        C_ij = [m + a_i - a_j]_+^2.

    A stationary transition i=j has cost m^2.
    Forward transitions remain feasible and sufficiently large
    forward movement can have zero cost.
    """

    def __init__(
        self,
        num_topics,
        margin,
    ):
        super().__init__()

        if num_topics < 2:
            raise ValueError(
                "At least two topics are required."
            )

        self.num_topics = num_topics
        self.margin = float(margin)

        # phi=0 gives equal softplus increments and therefore
        # a uniform topic grid.
        self.phi = nn.Parameter(
            torch.zeros(
                num_topics - 1,
                dtype=torch.float32,
            )
        )

    def positive_increments(self):
        return F.softplus(
            self.phi
        )

    def coordinates(self):
        increments = (
            self.positive_increments()
        )

        total = increments.sum().clamp_min(
            1e-12
        )

        cumulative = torch.cumsum(
            increments,
            dim=0,
        )

        zero = torch.zeros(
            1,
            device=increments.device,
            dtype=increments.dtype,
        )

        coordinates = torch.cat(
            [
                zero,
                cumulative / total,
            ],
            dim=0,
        )

        # Numerical protection for exact endpoints.
        coordinates = coordinates.clone()
        coordinates[0] = 0.0
        coordinates[-1] = 1.0

        return coordinates

    def gap_loss(self):
        increments = (
            self.positive_increments()
        )

        mean_increment = (
            increments.mean()
        )

        return torch.mean(
            (
                increments
                - mean_increment
            ).pow(2)
        )

    def cost_matrix(self):
        coordinates = self.coordinates()

        parent_coordinate = (
            coordinates[:, None]
        )
        child_coordinate = (
            coordinates[None, :]
        )

        cost = F.relu(
            self.margin
            + parent_coordinate
            - child_coordinate
        ).pow(2)

        return cost


topic_anchor = IsotonicTopicAnchor(
    num_topics=CONFIG.num_topics,
    margin=OT_MARGIN,
).to(DEVICE)

phi_optimizer = torch.optim.Adam(
    topic_anchor.parameters(),
    lr=PHI_LEARNING_RATE,
)

print("\n=== INITIAL ISOTONIC ANCHOR ===")

with torch.no_grad():
    initial_coordinates = (
        topic_anchor.coordinates()
    )
    initial_cost = (
        topic_anchor.cost_matrix()
    )

print(
    "Coordinates:",
    initial_coordinates.detach().cpu().tolist(),
)
print(
    f"Coordinate range: "
    f"[{initial_coordinates.min().item():.6f}, "
    f"{initial_coordinates.max().item():.6f}]"
)
print(
    f"Cost range: "
    f"[{initial_cost.min().item():.6f}, "
    f"{initial_cost.max().item():.6f}]"
)
print(
    f"Stationary cost mean: "
    f"{initial_cost.diag().mean().item():.6f}"
)

if not torch.all(
    initial_coordinates[1:]
    > initial_coordinates[:-1]
):
    raise AssertionError(
        "Initial anchor coordinates are not strictly increasing."
    )

if not torch.allclose(
    initial_cost.diag(),
    torch.full_like(
        initial_cost.diag(),
        OT_MARGIN ** 2,
    ),
    atol=1e-6,
):
    raise AssertionError(
        "Stationary transition cost does not equal margin squared."
    )


# ============================================================
# C. Reference coupling
# ============================================================

def build_reference_coupling(
    empirical_coupling,
    parent_marginal,
    child_marginal,
    rho=OT_REFERENCE_RHO,
    eps=EPS_OT,
):
    """
    Smooth the empirical parent-child coupling:

        R = (1-rho) Q + rho p q^T.

    Q retains observed aggregate parent-child topic associations,
    while p q^T provides strictly positive support.
    """

    if empirical_coupling.ndim != 2:
        raise ValueError(
            "empirical_coupling must be two-dimensional."
        )

    parent_marginal = (
        normalize_probability_vector(
            parent_marginal,
            eps=eps,
        )
    )

    child_marginal = (
        normalize_probability_vector(
            child_marginal,
            eps=eps,
        )
    )

    empirical_coupling = (
        empirical_coupling.clamp_min(0.0)
    )

    empirical_total = (
        empirical_coupling.sum()
    )

    if empirical_total <= 0:
        raise ValueError(
            "Empirical coupling has zero total mass."
        )

    empirical_coupling = (
        empirical_coupling
        / empirical_total
    )

    independent_reference = torch.outer(
        parent_marginal,
        child_marginal,
    )

    reference = (
        (1.0 - rho)
        * empirical_coupling
        + rho
        * independent_reference
    )

    reference = reference.clamp_min(
        eps
    )
    reference = (
        reference
        / reference.sum()
    )

    if not torch.isfinite(
        reference
    ).all():
        raise FloatingPointError(
            "Non-finite reference coupling detected."
        )

    return reference


# ============================================================
# D. Step 3: Log-domain reference-regularized Sinkhorn
# ============================================================

def reference_sinkhorn(
    parent_marginal,
    child_marginal,
    reference,
    cost,
    epsilon=OT_EPSILON,
    num_iterations=SINKHORN_ITERS,
    tolerance=SINKHORN_TOL,
    check_convergence=True,
):
    """
    Solve:

        min_Gamma <Gamma, C>
                  + epsilon KL(Gamma || R)

        subject to:
            Gamma 1 = parent_marginal,
            Gamma^T 1 = child_marginal.

    The corresponding kernel is:

        K = R * exp(-C / epsilon).

    Sinkhorn scaling is performed in log space.
    """

    parent_marginal = (
        normalize_probability_vector(
            parent_marginal,
            eps=EPS_OT,
        )
    )

    child_marginal = (
        normalize_probability_vector(
            child_marginal,
            eps=EPS_OT,
        )
    )

    reference = reference.clamp_min(
        EPS_OT
    )
    reference = (
        reference
        / reference.sum()
    )

    if cost.shape != reference.shape:
        raise ValueError(
            f"Cost/reference shape mismatch: "
            f"{tuple(cost.shape)} vs "
            f"{tuple(reference.shape)}."
        )

    log_parent = torch.log(
        parent_marginal
    )
    log_child = torch.log(
        child_marginal
    )

    log_kernel = (
        torch.log(reference)
        - cost / epsilon
    )

    log_u = torch.zeros_like(
        parent_marginal
    )
    log_v = torch.zeros_like(
        child_marginal
    )

    converged = False
    used_iterations = num_iterations

    for iteration in range(
        num_iterations
    ):
        previous_log_u = log_u

        log_u = (
            log_parent
            - torch.logsumexp(
                log_kernel
                + log_v.unsqueeze(0),
                dim=1,
            )
        )

        log_v = (
            log_child
            - torch.logsumexp(
                log_kernel
                + log_u.unsqueeze(1),
                dim=0,
            )
        )

        if check_convergence:
            dual_change = (
                log_u
                - previous_log_u
            ).abs().max()

            if (
                iteration >= 2
                and dual_change.item()
                < tolerance
            ):
                converged = True
                used_iterations = (
                    iteration + 1
                )
                break

    log_plan = (
        log_u.unsqueeze(1)
        + log_kernel
        + log_v.unsqueeze(0)
    )

    transport_plan = torch.exp(
        log_plan
    )

    if not torch.isfinite(
        transport_plan
    ).all():
        raise FloatingPointError(
            "Non-finite transport plan detected."
        )

    if torch.any(
        transport_plan < 0
    ):
        raise ValueError(
            "Negative transport mass detected."
        )

    row_marginal = (
        transport_plan.sum(dim=1)
    )
    column_marginal = (
        transport_plan.sum(dim=0)
    )

    row_error = (
        row_marginal
        - parent_marginal
    ).abs().max()

    column_error = (
        column_marginal
        - child_marginal
    ).abs().max()

    maximum_error = torch.maximum(
        row_error,
        column_error,
    )

    # Convergence based on final marginal feasibility.
    if maximum_error.item() <= max(
        tolerance * 10.0,
        1e-5,
    ):
        converged = True

    diagnostics = {
        "converged": converged,
        "iterations": used_iterations,
        "row_error": float(
            row_error.detach().item()
        ),
        "column_error": float(
            column_error.detach().item()
        ),
        "maximum_marginal_error": float(
            maximum_error.detach().item()
        ),
    }

    return transport_plan, diagnostics


# ============================================================
# E. OT objective components
# ============================================================

def coupling_kl_divergence(
    transport_plan,
    reference,
    eps=EPS_OT,
):
    plan_safe = transport_plan.clamp_min(
        eps
    )
    reference_safe = reference.clamp_min(
        eps
    )

    return torch.sum(
        plan_safe
        * (
            torch.log(plan_safe)
            - torch.log(reference_safe)
        )
    )


def compute_reference_ot_terms(
    stats,
    reconciled_marginals,
    max_depth,
    cost_matrix,
    detach_cost_for_main=False,
):
    """
    Compute one reference-regularized OT plan for each adjacent
    depth pair.

    For the main model update, set detach_cost_for_main=True
    so phi receives no gradient.
    """

    if detach_cost_for_main:
        effective_cost = (
            cost_matrix.detach()
        )
    else:
        effective_cost = (
            cost_matrix
        )

    transition_terms = []
    transition_weights = []
    transport_plans = {}
    references = {}
    diagnostics = {}

    total_edges = sum(
        int(stats[depth]["n"])
        for depth in range(max_depth)
    )

    if total_edges <= 0:
        raise ValueError(
            "No adjacent-depth edges are available."
        )

    for depth in range(max_depth):
        parent_marginal = (
            reconciled_marginals[depth]
        )
        child_marginal = (
            reconciled_marginals[
                depth + 1
            ]
        )

        empirical_coupling = (
            stats[depth]["Q"]
        )

        reference = (
            build_reference_coupling(
                empirical_coupling=(
                    empirical_coupling
                ),
                parent_marginal=(
                    parent_marginal
                ),
                child_marginal=(
                    child_marginal
                ),
            )
        )

        transport_plan, sinkhorn_info = (
            reference_sinkhorn(
                parent_marginal=(
                    parent_marginal
                ),
                child_marginal=(
                    child_marginal
                ),
                reference=reference,
                cost=effective_cost,
            )
        )

        transport_cost = torch.sum(
            transport_plan
            * effective_cost
        )

        reference_kl = (
            coupling_kl_divergence(
                transport_plan,
                reference,
            )
        )

        transition_objective = (
            transport_cost
            + OT_EPSILON
            * reference_kl
        )

        edge_weight = (
            float(stats[depth]["n"])
            / float(total_edges)
        )

        transition_terms.append(
            transition_objective
        )
        transition_weights.append(
            edge_weight
        )

        transport_plans[depth] = (
            transport_plan
        )
        references[depth] = reference

        diagnostics[depth] = {
            **sinkhorn_info,
            "edge_count": int(
                stats[depth]["n"]
            ),
            "edge_weight": edge_weight,
            "transport_cost": float(
                transport_cost.detach().item()
            ),
            "reference_kl": float(
                reference_kl.detach().item()
            ),
            "objective": float(
                transition_objective.detach().item()
            ),
        }

    hierarchy_loss = sum(
        weight * term
        for weight, term
        in zip(
            transition_weights,
            transition_terms,
        )
    )

    return {
        "hierarchy_loss": hierarchy_loss,
        "transport_plans": transport_plans,
        "references": references,
        "diagnostics": diagnostics,
    }


# ============================================================
# F. Step 4: Main-model hierarchy losses
# ============================================================

def compute_main_hierarchy_losses(
    theta,
    claim_depth,
    claim_edge_index,
):
    """
    Main-model pass.

    Gradients:
        theta and neural modules receive gradient.
        phi does not receive gradient because the cost is detached.
    """

    stats, max_depth, edge_diagnostics = (
        compute_depth_transition_stats(
            theta=theta,
            claim_depth=claim_depth,
            claim_edge_index=claim_edge_index,
        )
    )

    reconciled_marginals, recon_loss = (
        reconcile_depth_marginals(
            stats=stats,
            max_depth=max_depth,
            theta_reference=theta,
        )
    )

    cost_matrix = (
        topic_anchor.cost_matrix().detach()
    )

    ot_output = (
        compute_reference_ot_terms(
            stats=stats,
            reconciled_marginals=(
                reconciled_marginals
            ),
            max_depth=max_depth,
            cost_matrix=cost_matrix,
            detach_cost_for_main=True,
        )
    )

    return {
        "hierarchy_loss": (
            ot_output[
                "hierarchy_loss"
            ]
        ),
        "reconciliation_loss": (
            recon_loss
        ),
        "stats": stats,
        "reconciled_marginals": (
            reconciled_marginals
        ),
        "transport_plans": (
            ot_output[
                "transport_plans"
            ]
        ),
        "references": (
            ot_output[
                "references"
            ]
        ),
        "sinkhorn_diagnostics": (
            ot_output[
                "diagnostics"
            ]
        ),
        "edge_diagnostics": (
            edge_diagnostics
        ),
        "max_depth": max_depth,
    }


# ============================================================
# G. Separate phi loss and update
# ============================================================

def compute_phi_loss(
    theta,
    claim_depth,
    claim_edge_index,
):
    """
    Separate anchor pass.

    theta, Q, marginals, references, and optimized transport
    plans are treated as constants.

    The transport plan is recomputed using the current anchor
    cost and then detached. Phi receives gradient only through
    <stopgrad(Gamma), C(phi)> and the gap regularizer.
    """

    detached_theta = theta.detach()

    with torch.no_grad():
        stats, max_depth, edge_diagnostics = (
            compute_depth_transition_stats(
                theta=detached_theta,
                claim_depth=claim_depth,
                claim_edge_index=claim_edge_index,
            )
        )

        reconciled_marginals, _ = (
            reconcile_depth_marginals(
                stats=stats,
                max_depth=max_depth,
                theta_reference=(
                    detached_theta
                ),
            )
        )

    differentiable_cost = (
        topic_anchor.cost_matrix()
    )

    detached_plans = {}
    transition_costs = []
    transition_weights = []
    sinkhorn_diagnostics = {}

    total_edges = sum(
        int(stats[depth]["n"])
        for depth in range(max_depth)
    )

    for depth in range(max_depth):
        parent_marginal = (
            reconciled_marginals[
                depth
            ].detach()
        )
        child_marginal = (
            reconciled_marginals[
                depth + 1
            ].detach()
        )

        reference = (
            build_reference_coupling(
                empirical_coupling=(
                    stats[depth]["Q"].detach()
                ),
                parent_marginal=(
                    parent_marginal
                ),
                child_marginal=(
                    child_marginal
                ),
            )
        ).detach()

        # The optimizer plan is needed only as a fixed target for
        # the envelope-style phi gradient.
        with torch.no_grad():
            transport_plan, info = (
                reference_sinkhorn(
                    parent_marginal=(
                        parent_marginal
                    ),
                    child_marginal=(
                        child_marginal
                    ),
                    reference=reference,
                    cost=(
                        differentiable_cost.detach()
                    ),
                )
            )

        transport_plan = (
            transport_plan.detach()
        )

        transition_cost = torch.sum(
            transport_plan
            * differentiable_cost
        )

        edge_weight = (
            float(stats[depth]["n"])
            / float(total_edges)
        )

        transition_costs.append(
            transition_cost
        )
        transition_weights.append(
            edge_weight
        )

        detached_plans[depth] = (
            transport_plan
        )
        sinkhorn_diagnostics[depth] = (
            info
        )

    anchor_transport_loss = sum(
        weight * term
        for weight, term
        in zip(
            transition_weights,
            transition_costs,
        )
    )

    gap_loss = (
        topic_anchor.gap_loss()
    )

    phi_loss = (
        anchor_transport_loss
        + GAP_WEIGHT * gap_loss
    )

    return {
        "phi_loss": phi_loss,
        "anchor_transport_loss": (
            anchor_transport_loss
        ),
        "gap_loss": gap_loss,
        "transport_plans": (
            detached_plans
        ),
        "sinkhorn_diagnostics": (
            sinkhorn_diagnostics
        ),
        "edge_diagnostics": (
            edge_diagnostics
        ),
    }


def set_phi_trainable(
    trainable,
):
    for parameter in (
        topic_anchor.parameters()
    ):
        parameter.requires_grad_(
            trainable
        )


def update_phi(
    theta,
    claim_depth,
    claim_edge_index,
    epoch,
):
    """
    Perform one separate phi optimizer update.

    Phi is frozen during the hierarchy warm-up period.
    """

    if epoch < PHI_WARMUP_EPOCHS:
        set_phi_trainable(False)

        return {
            "updated": False,
            "reason": "warmup",
            "phi_loss": None,
            "gap_loss": None,
        }

    set_phi_trainable(True)

    phi_optimizer.zero_grad(
        set_to_none=True
    )

    output = compute_phi_loss(
        theta=theta,
        claim_depth=claim_depth,
        claim_edge_index=claim_edge_index,
    )

    phi_loss = output[
        "phi_loss"
    ]

    if not torch.isfinite(
        phi_loss
    ):
        raise FloatingPointError(
            "Non-finite phi loss detected."
        )

    phi_loss.backward()

    if topic_anchor.phi.grad is None:
        raise RuntimeError(
            "No gradient reached phi."
        )

    if not torch.isfinite(
        topic_anchor.phi.grad
    ).all():
        raise FloatingPointError(
            "Non-finite phi gradients detected."
        )

    torch.nn.utils.clip_grad_norm_(
        topic_anchor.parameters(),
        max_norm=1.0,
    )

    phi_optimizer.step()

    return {
        "updated": True,
        "reason": None,
        "phi_loss": float(
            output[
                "phi_loss"
            ].detach().item()
        ),
        "anchor_transport_loss": float(
            output[
                "anchor_transport_loss"
            ].detach().item()
        ),
        "gap_loss": float(
            output[
                "gap_loss"
            ].detach().item()
        ),
    }


# ============================================================
# H. Real-batch sanity checks
# ============================================================

print("\n=== SECTION 4 STEPS 2-4 SANITY CHECK ===")

main_ot_output = (
    compute_main_hierarchy_losses(
        theta=theta_s4,
        claim_depth=claim_depth_s4,
        claim_edge_index=(
            claim_edge_index_s4
        ),
    )
)

hierarchy_loss_s4 = (
    main_ot_output[
        "hierarchy_loss"
    ]
)

reconciliation_loss_s4 = (
    main_ot_output[
        "reconciliation_loss"
    ]
)

gap_loss_s4 = (
    topic_anchor.gap_loss()
)

print(
    f"Hierarchy loss      : "
    f"{hierarchy_loss_s4.item():.6f}"
)
print(
    f"Reconciliation loss : "
    f"{reconciliation_loss_s4.item():.6f}"
)
print(
    f"Gap loss            : "
    f"{gap_loss_s4.item():.6f}"
)

if not torch.isfinite(
    hierarchy_loss_s4
):
    raise FloatingPointError(
        "Non-finite hierarchy loss."
    )

if not torch.isfinite(
    reconciliation_loss_s4
):
    raise FloatingPointError(
        "Non-finite reconciliation loss."
    )

print("\n=== SINKHORN DIAGNOSTICS ===")

for depth, info in (
    main_ot_output[
        "sinkhorn_diagnostics"
    ].items()
):
    print(
        f"depth {depth}->{depth + 1}: "
        f"converged={info['converged']}, "
        f"iterations={info['iterations']}, "
        f"row_error={info['row_error']:.3e}, "
        f"column_error={info['column_error']:.3e}, "
        f"cost={info['transport_cost']:.6f}, "
        f"KL={info['reference_kl']:.6f}"
    )

    if (
        info[
            "maximum_marginal_error"
        ]
        > max(
            SINKHORN_TOL * 10.0,
            1e-4,
        )
    ):
        raise RuntimeError(
            f"Sinkhorn did not satisfy marginals "
            f"at depth {depth}."
        )


# ------------------------------------------------------------
# Main-pass gradient isolation check
# ------------------------------------------------------------

topic_anchor.zero_grad(
    set_to_none=True
)

theta_gradient = torch.autograd.grad(
    outputs=(
        hierarchy_loss_s4
        + reconciliation_loss_s4
    ),
    inputs=theta_s4,
    retain_graph=True,
    allow_unused=False,
)[0]

if theta_gradient is None:
    raise RuntimeError(
        "No hierarchy gradient reached theta."
    )

if not torch.isfinite(
    theta_gradient
).all():
    raise FloatingPointError(
        "Non-finite theta gradient detected."
    )

if topic_anchor.phi.grad is not None:
    raise RuntimeError(
        "Phi unexpectedly received gradient "
        "during the main-model hierarchy pass."
    )

print(
    "\n[PASS] Main hierarchy gradient reaches theta "
    "without reaching phi."
)


# ------------------------------------------------------------
# Phi-pass gradient isolation check
# ------------------------------------------------------------

set_phi_trainable(True)

topic_anchor.zero_grad(
    set_to_none=True
)

phi_test_output = compute_phi_loss(
    theta=theta_s4,
    claim_depth=claim_depth_s4,
    claim_edge_index=(
        claim_edge_index_s4
    ),
)

phi_test_output[
    "phi_loss"
].backward()

if topic_anchor.phi.grad is None:
    raise RuntimeError(
        "No gradient reached phi in the phi pass."
    )

if not torch.isfinite(
    topic_anchor.phi.grad
).all():
    raise FloatingPointError(
        "Non-finite phi gradient detected."
    )

print(
    "[PASS] Separate phi gradient check passed."
)
print(
    f"Phi gradient norm: "
    f"{topic_anchor.phi.grad.norm().item():.6f}"
)

topic_anchor.zero_grad(
    set_to_none=True
)


# ------------------------------------------------------------
# Final anchor checks
# ------------------------------------------------------------

with torch.no_grad():
    coordinates = (
        topic_anchor.coordinates()
    )
    cost_matrix = (
        topic_anchor.cost_matrix()
    )

if not torch.all(
    coordinates[1:]
    > coordinates[:-1]
):
    raise AssertionError(
        "Topic-anchor coordinates are not strictly increasing."
    )

if torch.any(
    cost_matrix < 0
):
    raise AssertionError(
        "Negative OT costs detected."
    )

print("\n=== FINAL STEP 2-4 SUMMARY ===")
print(
    f"Anchor coordinates shape : "
    f"{tuple(coordinates.shape)}"
)
print(
    f"Cost matrix shape        : "
    f"{tuple(cost_matrix.shape)}"
)
print(
    f"Cost range               : "
    f"[{cost_matrix.min().item():.6f}, "
    f"{cost_matrix.max().item():.6f}]"
)
print(
    f"Forward extreme cost C[0,-1]: "
    f"{cost_matrix[0, -1].item():.6f}"
)
print(
    f"Stationary mean cost     : "
    f"{cost_matrix.diag().mean().item():.6f}"
)
print(
    f"Backward extreme C[-1,0]: "
    f"{cost_matrix[-1, 0].item():.6f}"
)

print(
    "\n[PASS] Section 4 Steps 2, 3, and 4 completed."
)

# ============================================================
# SECTION 5: Full Model Assembly, Losses, and Sanity Checks
# COMPLETE SINGLE-CELL VERSION
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. Preconditions
# ============================================================

required_globals = [
    "CONFIG",
    "DEVICE",
    "sample_batch",
    "token_gcn",
    "dep_encoder",
    "var_encoder",
    "topic_word",
    "topic_anchor",
    "compute_main_hierarchy_losses",
    "compute_phi_loss",
    "update_phi",
]

missing_globals = [
    name
    for name in required_globals
    if name not in globals()
]

if missing_globals:
    raise RuntimeError(
        "Section 5 prerequisites are missing: "
        f"{missing_globals}. Run Sections 2, 3, and 4 first."
    )


# ============================================================
# 1. Objective configuration
# ============================================================

LEARNING_RATE = float(
    getattr(
        CONFIG,
        "learning_rate",
        1e-3,
    )
)

WEIGHT_DECAY = float(
    getattr(
        CONFIG,
        "weight_decay",
        1e-5,
    )
)

MAX_GRAD_NORM = float(
    getattr(
        CONFIG,
        "max_grad_norm",
        1.0,
    )
)

GAMMA_KL_MAX = float(
    getattr(
        CONFIG,
        "gamma_kl",
        1.0,
    )
)

GAMMA_H_MAX = float(
    getattr(
        CONFIG,
        "gamma_h",
        1.0,
    )
)

GAMMA_RECON = float(
    getattr(
        CONFIG,
        "gamma_r",
        0.1,
    )
)

GAMMA_DIVERSITY = float(
    getattr(
        CONFIG,
        "gamma_d",
        0.01,
    )
)

KL_WARMUP_EPOCHS = int(
    getattr(
        CONFIG,
        "kl_warmup_epochs",
        5,
    )
)

HIERARCHY_WARMUP_EPOCHS = int(
    getattr(
        CONFIG,
        "hierarchy_warmup_epochs",
        5,
    )
)

if LEARNING_RATE <= 0:
    raise ValueError(
        f"learning_rate must be positive, got {LEARNING_RATE}."
    )

if WEIGHT_DECAY < 0:
    raise ValueError(
        f"weight_decay must be nonnegative, got {WEIGHT_DECAY}."
    )

if MAX_GRAD_NORM <= 0:
    raise ValueError(
        f"max_grad_norm must be positive, got {MAX_GRAD_NORM}."
    )

for name, value in {
    "GAMMA_KL_MAX": GAMMA_KL_MAX,
    "GAMMA_H_MAX": GAMMA_H_MAX,
    "GAMMA_RECON": GAMMA_RECON,
    "GAMMA_DIVERSITY": GAMMA_DIVERSITY,
}.items():
    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative, got {value}."
        )

print("=== SECTION 5 OBJECTIVE CONFIGURATION ===")
print(f"Learning rate          : {LEARNING_RATE}")
print(f"Weight decay           : {WEIGHT_DECAY}")
print(f"Maximum gradient norm  : {MAX_GRAD_NORM}")
print(f"Maximum KL weight      : {GAMMA_KL_MAX}")
print(f"Maximum hierarchy weight: {GAMMA_H_MAX}")
print(f"Reconciliation weight  : {GAMMA_RECON}")
print(f"Diversity weight       : {GAMMA_DIVERSITY}")
print(f"KL warm-up epochs      : {KL_WARMUP_EPOCHS}")
print(
    f"Hierarchy warm-up epochs: "
    f"{HIERARCHY_WARMUP_EPOCHS}"
)


# ============================================================
# 2. Warm-up schedules
# ============================================================

def linear_warmup_weight(
    epoch,
    warmup_epochs,
    maximum_weight,
):
    """
    Linear warm-up from zero to maximum_weight.

    At epoch=0:
        weight=0 if warmup_epochs > 0.

    At epoch>=warmup_epochs:
        weight=maximum_weight.
    """

    if maximum_weight == 0:
        return 0.0

    if warmup_epochs <= 0:
        return float(maximum_weight)

    progress = min(
        max(
            float(epoch)
            / float(warmup_epochs),
            0.0,
        ),
        1.0,
    )

    return float(maximum_weight) * progress


def get_objective_weights(epoch):
    return {
        "gamma_kl": linear_warmup_weight(
            epoch=epoch,
            warmup_epochs=KL_WARMUP_EPOCHS,
            maximum_weight=GAMMA_KL_MAX,
        ),
        "gamma_h": linear_warmup_weight(
            epoch=epoch,
            warmup_epochs=(
                HIERARCHY_WARMUP_EPOCHS
            ),
            maximum_weight=GAMMA_H_MAX,
        ),
        "gamma_r": GAMMA_RECON,
        "gamma_d": GAMMA_DIVERSITY,
    }


print("\n=== WARM-UP EXAMPLES ===")

warmup_examples = sorted(
    {
        0,
        1,
        KL_WARMUP_EPOCHS,
        HIERARCHY_WARMUP_EPOCHS,
        max(
            KL_WARMUP_EPOCHS,
            HIERARCHY_WARMUP_EPOCHS,
        )
        + 1,
    }
)

for example_epoch in warmup_examples:
    weights = get_objective_weights(
        example_epoch
    )

    print(
        f"epoch={example_epoch:3d}: "
        f"gamma_kl={weights['gamma_kl']:.4f}, "
        f"gamma_h={weights['gamma_h']:.4f}, "
        f"gamma_r={weights['gamma_r']:.4f}, "
        f"gamma_d={weights['gamma_d']:.4f}"
    )


# ============================================================
# 3. Loss functions
# ============================================================

def reconstruction_loss_fn(
    bow,
    log_word_prob,
    eps=1e-10,
):
    if bow.ndim != 2:
        raise ValueError(
            f"bow must be 2-D, "
            f"got {tuple(bow.shape)}."
        )

    if log_word_prob.ndim != 2:
        raise ValueError(
            f"log_word_prob must be 2-D, "
            f"got {tuple(log_word_prob.shape)}."
        )

    if bow.shape != log_word_prob.shape:
        raise ValueError(
            "BoW/log_word_prob shape mismatch: "
            f"bow={tuple(bow.shape)}, "
            f"log_word_prob="
            f"{tuple(log_word_prob.shape)}."
        )

    if not torch.isfinite(bow).all():
        raise FloatingPointError(
            "BoW contains non-finite values."
        )

    if not torch.isfinite(
        log_word_prob
    ).all():
        raise FloatingPointError(
            "log_word_prob contains "
            "non-finite values."
        )

    if torch.any(bow < 0):
        raise ValueError(
            "BoW contains negative values."
        )

    bow_sums = bow.sum(
        dim=-1
    )

    valid_bow_mask = (
        bow_sums > eps
    )

    if valid_bow_mask.any():
        valid_sums = (
            bow_sums[valid_bow_mask]
        )

        if not torch.allclose(
            valid_sums,
            torch.ones_like(valid_sums),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError(
                "Non-empty BoW rows must "
                "be L1-normalized."
            )

        per_claim_nll = -(
            bow[valid_bow_mask]
            * log_word_prob[
                valid_bow_mask
            ]
        ).sum(dim=-1)

        reconstruction_loss = (
            per_claim_nll.mean()
        )

    else:
        # 배치 전체가 empty-BoW인 경우.
        # 그래프/계층 손실은 계속 학습할 수 있도록
        # gradient-connected zero를 반환합니다.
        reconstruction_loss = (
            log_word_prob.sum() * 0.0
        )

    if not torch.isfinite(
        reconstruction_loss
    ):
        raise FloatingPointError(
            "Reconstruction loss is not finite."
        )

    return reconstruction_loss



def logistic_normal_kl_loss(
    posterior_mean,
    posterior_logvar,
):
    """
    KL divergence from a diagonal Gaussian posterior to N(0, I):

        0.5 * sum(mu^2 + exp(logvar) - logvar - 1).

    The result is averaged over claims.
    """

    if (
        posterior_mean.shape
        != posterior_logvar.shape
    ):
        raise ValueError(
            "Mean and log-variance shapes differ."
        )

    per_claim_kl = 0.5 * torch.sum(
        posterior_mean.pow(2)
        + posterior_logvar.exp()
        - posterior_logvar
        - 1.0,
        dim=-1,
    )

    kl_loss = per_claim_kl.mean()

    if not torch.isfinite(kl_loss):
        raise FloatingPointError(
            "Non-finite KL loss detected."
        )

    return kl_loss


def topic_diversity_loss(
    beta,
    eps=1e-12,
):
    """
    Paper objective:

        L_div = (1/K) || beta_bar beta_bar^T - I ||_F^2,

    where beta_bar contains L2-normalized topic-word
    distributions.
    """

    if beta.ndim != 2:
        raise ValueError(
            "beta must have shape [num_topics, vocab_size]."
        )

    if not torch.isfinite(beta).all():
        raise FloatingPointError(
            "Non-finite topic-word values detected."
        )

    if torch.any(beta < 0):
        raise ValueError(
            "beta must be a nonnegative topic-word distribution."
        )

    beta_row_sums = beta.sum(
        dim=-1
    )

    if not torch.allclose(
        beta_row_sums,
        torch.ones_like(beta_row_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "The rows of beta must sum to one."
        )

    normalized_beta = F.normalize(
        beta,
        p=2,
        dim=1,
        eps=eps,
    )

    gram_matrix = (
        normalized_beta
        @ normalized_beta.transpose(0, 1)
    )

    identity = torch.eye(
        beta.shape[0],
        device=beta.device,
        dtype=beta.dtype,
    )

    diversity_loss = torch.sum(
        (
            gram_matrix
            - identity
        ).pow(2)
    ) / float(beta.shape[0])

    if not torch.isfinite(
        diversity_loss
    ):
        raise FloatingPointError(
            "Non-finite diversity loss detected."
        )

    return diversity_loss


# ============================================================
# 4. Batch movement utility
# ============================================================

def move_depth_ot_batch(
    batch,
    device,
    non_blocking=True,
):
    required_fields = [
        "token_embeddings",
        "token_edge_index",
        "token_edge_weight",
        "token_to_claim",
        "claim_edge_index",
        "claim_to_patent",
        "claim_depth",
        "bow",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in batch
    ]

    if missing_fields:
        raise KeyError(
            f"Missing batch fields: {missing_fields}"
        )

    moved = {
        "token_embeddings": batch[
            "token_embeddings"
        ].to(
            device=device,
            non_blocking=non_blocking,
        ),
        "token_edge_index": batch[
            "token_edge_index"
        ].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "token_edge_weight": batch[
            "token_edge_weight"
        ].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ),
        "token_to_claim": batch[
            "token_to_claim"
        ].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "claim_edge_index": batch[
            "claim_edge_index"
        ].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "claim_to_patent": batch[
            "claim_to_patent"
        ].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "claim_depth": batch[
            "claim_depth"
        ].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "bow": batch["bow"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ),
        "num_patents": int(
            batch["num_patents"]
        ),
        "num_claims": int(
            batch["num_claims"]
        ),
        "num_tokens": int(
            batch["num_tokens"]
        ),
    }

    # Keep identifiers on CPU as Python objects.
    for optional_field in [
        "patent_ids",
        "claim_ids",
        "claim_keys",
    ]:
        if optional_field in batch:
            moved[optional_field] = batch[
                optional_field
            ]

    if "truncated" in batch:
        moved["truncated"] = batch[
            "truncated"
        ]

    return moved


# ============================================================
# 5. Full neural model
# ============================================================

class DepthOTModel(nn.Module):
    """
    Neural portion of the complete model:

        frozen PLM features
        -> token GCN
        -> bidirectional dependency encoder
        -> logistic-normal variational encoder
        -> ProdLDA-style topic-word decoder
        -> reference-regularized depth OT objective

    topic_anchor is intentionally not registered here because it
    is optimized by a separate optimizer and backward pass.
    """

    def __init__(
        self,
        token_encoder,
        dependency_encoder,
        variational_encoder,
        topic_word_decoder,
    ):
        super().__init__()

        self.token_encoder = (
            token_encoder
        )
        self.dependency_encoder = (
            dependency_encoder
        )
        self.variational_encoder = (
            variational_encoder
        )
        self.topic_word_decoder = (
            topic_word_decoder
        )

    def forward(
        self,
        batch,
        deterministic=False,
    ):
        claim_representation = (
            self.token_encoder(
                token_embeddings=batch[
                    "token_embeddings"
                ],
                token_edge_index=batch[
                    "token_edge_index"
                ],
                token_edge_weight=batch[
                    "token_edge_weight"
                ],
                token_to_claim=batch[
                    "token_to_claim"
                ],
                num_claims=batch[
                    "num_claims"
                ],
            )
        )

        claim_context = (
            self.dependency_encoder(
                claim_repr=(
                    claim_representation
                ),
                claim_edge_index=batch[
                    "claim_edge_index"
                ],
                num_claims=batch[
                    "num_claims"
                ],
            )
        )

        theta, posterior_mean, posterior_logvar = (
            self.variational_encoder(
                claim_context,
                deterministic=deterministic,
            )
        )

        (
            log_word_probability,
            beta,
        ) = self.topic_word_decoder(
            theta
        )

        return {
            "claim_representation": (
                claim_representation
            ),
            "claim_context": (
                claim_context
            ),
            "theta": theta,
            "posterior_mean": (
                posterior_mean
            ),
            "posterior_logvar": (
                posterior_logvar
            ),
            "log_word_probability": (
                log_word_probability
            ),
            "beta": beta,
        }

    def compute_loss(
        self,
        batch,
        epoch,
        deterministic=False,
        compute_hierarchy=True,
    ):
        outputs = self.forward(
            batch=batch,
            deterministic=deterministic,
        )

        reconstruction_loss = (
            reconstruction_loss_fn(
                bow=batch["bow"],
                log_word_prob=(
                    outputs[
                        "log_word_probability"
                    ]
                ),
            )
        )

        kl_loss = logistic_normal_kl_loss(
            posterior_mean=outputs[
                "posterior_mean"
            ],
            posterior_logvar=outputs[
                "posterior_logvar"
            ],
        )

        diversity_loss = (
            topic_diversity_loss(
                outputs["beta"]
            )
        )

        if compute_hierarchy:
            # OT is evaluated in float32 even if the neural forward
            # pass later uses mixed precision.
            ot_output = (
                compute_main_hierarchy_losses(
                    theta=outputs[
                        "theta"
                    ].float(),
                    claim_depth=batch[
                        "claim_depth"
                    ],
                    claim_edge_index=batch[
                        "claim_edge_index"
                    ],
                )
            )

            hierarchy_loss = ot_output[
                "hierarchy_loss"
            ]

            reconciliation_loss = (
                ot_output[
                    "reconciliation_loss"
                ]
            )
        else:
            zero = (
                outputs["theta"].sum()
                * 0.0
            )

            hierarchy_loss = zero
            reconciliation_loss = zero
            ot_output = None

        weights = get_objective_weights(
            epoch
        )

        total_loss = (
            reconstruction_loss
            + weights["gamma_kl"]
            * kl_loss
            + weights["gamma_h"]
            * hierarchy_loss
            + weights["gamma_r"]
            * reconciliation_loss
            + weights["gamma_d"]
            * diversity_loss
        )

        if not torch.isfinite(
            total_loss
        ):
            raise FloatingPointError(
                "Non-finite total loss detected."
            )

        losses = {
            "total": total_loss,
            "reconstruction": (
                reconstruction_loss
            ),
            "kl": kl_loss,
            "hierarchy": hierarchy_loss,
            "reconciliation": (
                reconciliation_loss
            ),
            "diversity": diversity_loss,
        }

        return {
            "losses": losses,
            "weights": weights,
            "outputs": outputs,
            "ot_output": ot_output,
        }

    @torch.no_grad()
    def deterministic_inference(
        self,
        batch,
    ):
        was_training = self.training
        self.eval()

        outputs = self.forward(
            batch=batch,
            deterministic=True,
        )

        if was_training:
            self.train()

        return outputs


# ============================================================
# 6. Instantiate the full model
# ============================================================

depth_ot_model = DepthOTModel(
    token_encoder=token_gcn,
    dependency_encoder=dep_encoder,
    variational_encoder=var_encoder,
    topic_word_decoder=topic_word,
).to(DEVICE)

# Phi is deliberately excluded because topic_anchor is not a
# child module of depth_ot_model.
main_model_parameters = [
    parameter
    for parameter in depth_ot_model.parameters()
    if parameter.requires_grad
]

if not main_model_parameters:
    raise RuntimeError(
        "The main model has no trainable parameters."
    )

main_optimizer = torch.optim.AdamW(
    main_model_parameters,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

main_parameter_ids = {
    id(parameter)
    for parameter in main_model_parameters
}

phi_parameter_ids = {
    id(parameter)
    for parameter in topic_anchor.parameters()
}

if main_parameter_ids & phi_parameter_ids:
    raise RuntimeError(
        "Phi parameters were incorrectly included in "
        "the main-model optimizer."
    )

print("\n=== OPTIMIZER SEPARATION ===")
print(
    f"Main-model parameter tensors: "
    f"{len(main_model_parameters)}"
)
print(
    f"Phi parameter tensors       : "
    f"{len(list(topic_anchor.parameters()))}"
)
print(
    "Optimizer overlap           : False"
)


# ============================================================
# 7. Real-batch forward and backward sanity check
# ============================================================

print("\n=== SECTION 5 REAL-BATCH SANITY CHECK ===")

section5_batch = move_depth_ot_batch(
    sample_batch,
    device=DEVICE,
)

# Use an epoch after both warm-ups so every loss component
# contributes to this validation.
SANITY_EPOCH = max(
    KL_WARMUP_EPOCHS,
    HIERARCHY_WARMUP_EPOCHS,
    PHI_WARMUP_EPOCHS
    if "PHI_WARMUP_EPOCHS" in globals()
    else 0,
) + 1

depth_ot_model.train()

main_optimizer.zero_grad(
    set_to_none=True
)

topic_anchor.zero_grad(
    set_to_none=True
)

sanity_output = (
    depth_ot_model.compute_loss(
        batch=section5_batch,
        epoch=SANITY_EPOCH,
        deterministic=False,
        compute_hierarchy=True,
    )
)

sanity_losses = sanity_output[
    "losses"
]

print(f"Sanity epoch          : {SANITY_EPOCH}")
print(
    f"Total loss            : "
    f"{sanity_losses['total'].item():.6f}"
)
print(
    f"Reconstruction loss   : "
    f"{sanity_losses['reconstruction'].item():.6f}"
)
print(
    f"KL loss               : "
    f"{sanity_losses['kl'].item():.6f}"
)
print(
    f"Hierarchy loss        : "
    f"{sanity_losses['hierarchy'].item():.6f}"
)
print(
    f"Reconciliation loss   : "
    f"{sanity_losses['reconciliation'].item():.6f}"
)
print(
    f"Diversity loss        : "
    f"{sanity_losses['diversity'].item():.6f}"
)

print("\nObjective weights:")

for name, value in sanity_output[
    "weights"
].items():
    print(
        f"  {name:12s}: {value:.6f}"
    )

for name, loss in sanity_losses.items():
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite Section 5 loss: {name}."
        )

sanity_losses["total"].backward()


# ============================================================
# 8. Main-model gradient checks
# ============================================================

module_groups = {
    "token_encoder": (
        depth_ot_model.token_encoder
    ),
    "dependency_encoder": (
        depth_ot_model.dependency_encoder
    ),
    "variational_encoder": (
        depth_ot_model.variational_encoder
    ),
    "topic_word_decoder": (
        depth_ot_model.topic_word_decoder
    ),
}

gradient_report = {}

for module_name, module in (
    module_groups.items()
):
    trainable_parameters = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    parameters_with_gradient = [
        parameter
        for parameter
        in trainable_parameters
        if parameter.grad is not None
    ]

    if not parameters_with_gradient:
        raise RuntimeError(
            f"No gradient reached {module_name}."
        )

    if not all(
        torch.isfinite(
            parameter.grad
        ).all()
        for parameter
        in parameters_with_gradient
    ):
        raise FloatingPointError(
            f"Non-finite gradients in {module_name}."
        )

    squared_norm = sum(
        parameter.grad.detach().pow(2).sum()
        for parameter
        in parameters_with_gradient
    )

    gradient_norm = torch.sqrt(
        squared_norm
    )

    gradient_report[module_name] = float(
        gradient_norm.item()
    )

print("\n=== MAIN-MODEL GRADIENT NORMS ===")

for module_name, gradient_norm in (
    gradient_report.items()
):
    print(
        f"{module_name:24s}: "
        f"{gradient_norm:.6f}"
    )


# Phi must not receive gradient from the main objective.
if topic_anchor.phi.grad is not None:
    if torch.any(
        topic_anchor.phi.grad != 0
    ):
        raise RuntimeError(
            "Phi received gradient from the main-model loss."
        )

print(
    "\n[PASS] Main-model loss does not update phi."
)


# ============================================================
# 9. Gradient clipping check
# ============================================================

unclipped_gradient_norm = (
    torch.nn.utils.clip_grad_norm_(
        main_model_parameters,
        max_norm=MAX_GRAD_NORM,
    )
)

if not torch.isfinite(
    torch.as_tensor(
        unclipped_gradient_norm
    )
):
    raise FloatingPointError(
        "Non-finite global gradient norm."
    )

print(
    f"Global gradient norm before clipping: "
    f"{float(unclipped_gradient_norm):.6f}"
)

# Do not call main_optimizer.step() during the sanity check.
# Section 6 will perform actual parameter updates.
main_optimizer.zero_grad(
    set_to_none=True
)


# ============================================================
# 10. Separate phi-pass validation
# ============================================================

topic_anchor.zero_grad(
    set_to_none=True
)

theta_for_phi = sanity_output[
    "outputs"
]["theta"].detach()

phi_sanity_output = compute_phi_loss(
    theta=theta_for_phi,
    claim_depth=section5_batch[
        "claim_depth"
    ],
    claim_edge_index=section5_batch[
        "claim_edge_index"
    ],
)

phi_sanity_loss = phi_sanity_output[
    "phi_loss"
]

if not torch.isfinite(
    phi_sanity_loss
):
    raise FloatingPointError(
        "Non-finite phi sanity loss."
    )

phi_sanity_loss.backward()

if topic_anchor.phi.grad is None:
    raise RuntimeError(
        "No gradient reached phi in the separate phi pass."
    )

if not torch.isfinite(
    topic_anchor.phi.grad
).all():
    raise FloatingPointError(
        "Non-finite phi gradient."
    )

phi_gradient_norm = (
    topic_anchor.phi.grad.norm()
)

print("\n=== SEPARATE PHI PASS ===")
print(
    f"Phi loss              : "
    f"{phi_sanity_loss.item():.6f}"
)
print(
    f"Anchor transport loss : "
    f"{phi_sanity_output['anchor_transport_loss'].item():.6f}"
)
print(
    f"Gap loss              : "
    f"{phi_sanity_output['gap_loss'].item():.6f}"
)
print(
    f"Phi gradient norm     : "
    f"{phi_gradient_norm.item():.6f}"
)

# Do not call phi_optimizer.step() during the sanity check.
topic_anchor.zero_grad(
    set_to_none=True
)


# ============================================================
# 11. Deterministic inference check
# ============================================================

with torch.no_grad():
    deterministic_output_1 = (
        depth_ot_model.deterministic_inference(
            section5_batch
        )
    )

    deterministic_output_2 = (
        depth_ot_model.deterministic_inference(
            section5_batch
        )
    )

theta_1 = deterministic_output_1[
    "theta"
]
theta_2 = deterministic_output_2[
    "theta"
]

if not torch.allclose(
    theta_1,
    theta_2,
    atol=0.0,
    rtol=0.0,
):
    raise AssertionError(
        "Deterministic inference is not reproducible."
    )

if not torch.allclose(
    theta_1.sum(dim=-1),
    torch.ones_like(
        theta_1.sum(dim=-1)
    ),
    atol=1e-5,
    rtol=1e-5,
):
    raise AssertionError(
        "Deterministic theta rows do not sum to one."
    )

beta = deterministic_output_1[
    "beta"
]

if not torch.allclose(
    beta.sum(dim=-1),
    torch.ones_like(
        beta.sum(dim=-1)
    ),
    atol=1e-5,
    rtol=1e-5,
):
    raise AssertionError(
        "Beta rows do not sum to one."
    )

print(
    "\n[PASS] Deterministic inference check passed."
)


# ============================================================
# 12. Parameter summary
# ============================================================

parameter_summary = {
    module_name: sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    for module_name, module
    in module_groups.items()
}

main_parameter_count = sum(
    parameter_summary.values()
)

phi_parameter_count = sum(
    parameter.numel()
    for parameter
    in topic_anchor.parameters()
)

print("\n=== PARAMETER SUMMARY ===")

for module_name, count in (
    parameter_summary.items()
):
    print(
        f"{module_name:24s}: "
        f"{count:,}"
    )

print(
    f"{'main model total':24s}: "
    f"{main_parameter_count:,}"
)
print(
    f"{'phi anchor':24s}: "
    f"{phi_parameter_count:,}"
)


# ============================================================
# 13. Final status
# ============================================================

print("\n" + "=" * 72)
print("SECTION 5 COMPLETED SUCCESSFULLY")
print("=" * 72)
print(
    "Neural model assembly       : PASS"
)
print(
    "Reconstruction and KL losses: PASS"
)
print(
    "Hierarchy and recon losses  : PASS"
)
print(
    "Topic diversity loss        : PASS"
)
print(
    "Main-model gradient flow    : PASS"
)
print(
    "Main/phi gradient separation: PASS"
)
print(
    "Deterministic inference     : PASS"
)
print("=" * 72)

print(
    "\nThe model is ready for Section 6: "
    "Training, checkpointing, and logging."
)


# ============================================================
# SECTION 6 — COMPLETE SINGLE-CELL VERSION
# L4 OPTIMIZED + RESUME-SAFE + RNG FIX
#
# Resume run:
#   depth_ot_full_20260811_215645
#
# Requirements:
#   Run Sections 0–5 first.
# ============================================================

import os
import gc
import json
import time
import math
import random
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ============================================================
# 0. Preconditions
# ============================================================

EXPECTED_PATENTS_PER_BATCH = 8
CONFIG.patents_per_batch = 8

required_globals = [
    "CONFIG",
    "DIRS",
    "DEVICE",
    "FEATURE_RUN_NAME",
    "train_dataset",
    "dev_dataset",
    "test_dataset",
    "train_loader",
    "dev_loader",
    "test_loader",
    "depth_ot_model",
    "topic_anchor",
    "main_optimizer",
    "phi_optimizer",
    "move_depth_ot_batch",
    "get_objective_weights",
    "update_phi",
]

missing_globals = [
    name
    for name in required_globals
    if name not in globals()
]

if missing_globals:
    raise RuntimeError(
        "Section 6 prerequisites are missing: "
        f"{missing_globals}. "
        "L4 런타임에서 Sections 0~5를 먼저 실행하세요."
    )


# ============================================================
# 1. Training configuration
# ============================================================

NUM_EPOCHS = 12

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_START_EPOCH = 6
MINIMUM_IMPROVEMENT = 1e-4

MAX_GRAD_NORM = 1.0

CHECKPOINT_INTERVAL_EPOCHS = 3
SAVE_LATEST_EVERY_EPOCH = True

# ------------------------------------------------------------
# Resume configuration
# ------------------------------------------------------------

RESUME_TRAINING = True

# 기존 실행 폴더 이름
RESUME_RUN_NAME = "depth_ot_full_20260811_215645"

# checkpoint가 없을 때 새 학습을 시작하지 않음
ALLOW_NEW_RUN_IF_CHECKPOINT_MISSING = False

# ------------------------------------------------------------
# Numerical configuration
# ------------------------------------------------------------

# Sinkhorn/hierarchy 안정성을 위해 FP32 유지
USE_MIXED_PRECISION = False

KL_WARMUP_EPOCHS = 3
HIERARCHY_WARMUP_EPOCHS = 3
RECONCILIATION_WARMUP_EPOCHS = 3
DIVERSITY_WARMUP_EPOCHS = 3
PHI_WARMUP_EPOCHS = 3

# ------------------------------------------------------------
# L4 optimization
# ------------------------------------------------------------

REBUILD_LOADERS_FOR_L4 = True

L4_NUM_WORKERS = min(
    4,
    max(
        2,
        (os.cpu_count() or 2) // 2,
    ),
)

L4_PREFETCH_FACTOR = 4

# tqdm GPU→CPU 동기화 감소
TRAIN_POSTFIX_INTERVAL = 100
EVAL_POSTFIX_INTERVAL = 100

# 기존 run이 patents_per_batch=8로 시작했으므로 유지
EXPECTED_PATENTS_PER_BATCH = 8


# ============================================================
# 2. Propagate settings to CONFIG
# ============================================================

training_config_overrides = {
    "num_epochs": NUM_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_start_epoch": EARLY_STOPPING_START_EPOCH,
    "minimum_improvement": MINIMUM_IMPROVEMENT,
    "max_grad_norm": MAX_GRAD_NORM,
    "checkpoint_every_epoch": CHECKPOINT_INTERVAL_EPOCHS,
    "checkpoint_interval_epochs": CHECKPOINT_INTERVAL_EPOCHS,
    "kl_warmup_epochs": KL_WARMUP_EPOCHS,
    "hierarchy_warmup_epochs": HIERARCHY_WARMUP_EPOCHS,
    "reconciliation_warmup_epochs": RECONCILIATION_WARMUP_EPOCHS,
    "diversity_warmup_epochs": DIVERSITY_WARMUP_EPOCHS,
    "phi_warmup_epochs": PHI_WARMUP_EPOCHS,
}

for key, value in training_config_overrides.items():
    try:
        setattr(
            CONFIG,
            key,
            value,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not set CONFIG.{key}={value}."
        ) from error


# ============================================================
# 3. Validate configuration
# ============================================================

if NUM_EPOCHS < 1:
    raise ValueError(
        "NUM_EPOCHS must be positive."
    )

if EARLY_STOPPING_PATIENCE < 1:
    raise ValueError(
        "EARLY_STOPPING_PATIENCE must be positive."
    )

if not (
    1
    <= EARLY_STOPPING_START_EPOCH
    <= NUM_EPOCHS
):
    raise ValueError(
        "Invalid EARLY_STOPPING_START_EPOCH."
    )

if CHECKPOINT_INTERVAL_EPOCHS < 1:
    raise ValueError(
        "CHECKPOINT_INTERVAL_EPOCHS must be positive."
    )

if MAX_GRAD_NORM <= 0:
    raise ValueError(
        "MAX_GRAD_NORM must be positive."
    )

if MINIMUM_IMPROVEMENT < 0:
    raise ValueError(
        "MINIMUM_IMPROVEMENT must be non-negative."
    )

configured_patents_per_batch = getattr(
    CONFIG,
    "patents_per_batch",
    None,
)

if (
    configured_patents_per_batch is not None
    and int(configured_patents_per_batch)
    != EXPECTED_PATENTS_PER_BATCH
):
    raise ValueError(
        "Resume 중 patents_per_batch를 변경하면 안 됩니다. "
        f"Expected={EXPECTED_PATENTS_PER_BATCH}, "
        f"current={configured_patents_per_batch}. "
        "Sections 0~5에서 patents_per_batch=8로 설정하세요."
    )


# ============================================================
# 4. CUDA/L4 setup
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. "
        "Colab 런타임에서 L4 GPU를 선택하세요."
    )

DEVICE = torch.device("cuda:0")

depth_ot_model.to(
    DEVICE
)

topic_anchor.to(
    DEVICE
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision(
        "high"
    )
except Exception:
    pass

GPU_NAME = torch.cuda.get_device_name(0)

GPU_TOTAL_GIB = (
    torch.cuda.get_device_properties(0).total_memory
    / 2**30
)

print("=" * 90)
print("SECTION 6 — L4 OPTIMIZED RESUME-SAFE TRAINING")
print("=" * 90)
print(f"GPU                      : {GPU_NAME}")
print(f"GPU memory               : {GPU_TOTAL_GIB:.2f} GiB")
print(f"Device                   : {DEVICE}")
print(f"TF32                     : enabled")
print(f"Mixed precision          : {USE_MIXED_PRECISION}")
print(f"Resume training          : {RESUME_TRAINING}")
print(f"Resume run               : {RESUME_RUN_NAME}")
print(f"Maximum epochs           : {NUM_EPOCHS}")
print(f"Patents/batch            : {configured_patents_per_batch}")
print("=" * 90)


# ============================================================
# 5. Optimizer helper
# ============================================================

def optimizer_to_device(
    optimizer,
    target_device,
):
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(
                    target_device
                )


# ============================================================
# 6. L4 DataLoader optimization
# ============================================================

def loader_is_already_optimized(
    loader,
):
    return (
        getattr(
            loader,
            "num_workers",
            0,
        ) == L4_NUM_WORKERS
        and getattr(
            loader,
            "pin_memory",
            False,
        )
        and getattr(
            loader,
            "persistent_workers",
            False,
        )
    )


def rebuild_loader_for_l4(
    original_loader,
    loader_name,
):
    if loader_is_already_optimized(
        original_loader
    ):
        print(
            f"[L4 Loader] {loader_name} "
            "is already optimized."
        )

        return original_loader

    print(
        f"[L4 Loader] Rebuilding {loader_name}: "
        f"workers={L4_NUM_WORKERS}, "
        f"pin_memory=True, "
        f"prefetch={L4_PREFETCH_FACTOR}"
    )

    return DataLoader(
        dataset=original_loader.dataset,
        batch_sampler=original_loader.batch_sampler,
        collate_fn=original_loader.collate_fn,
        num_workers=L4_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=L4_PREFETCH_FACTOR,
        worker_init_fn=getattr(
            original_loader,
            "worker_init_fn",
            None,
        ),
    )


if REBUILD_LOADERS_FOR_L4:
    original_train_loader = train_loader
    original_dev_loader = dev_loader
    original_test_loader = test_loader

    try:
        train_loader = rebuild_loader_for_l4(
            original_train_loader,
            "train",
        )

        dev_loader = rebuild_loader_for_l4(
            original_dev_loader,
            "dev",
        )

        test_loader = rebuild_loader_for_l4(
            original_test_loader,
            "test",
        )

        print(
            "[PASS] L4 DataLoaders are ready."
        )

    except Exception as error:
        print(
            f"[WARNING] L4 DataLoader rebuild failed: "
            f"{error}"
        )
        print(
            "[Fallback] Original DataLoaders will be used."
        )

        train_loader = original_train_loader
        dev_loader = original_dev_loader
        test_loader = original_test_loader


# ============================================================
# 7. Validate loader
# ============================================================

train_batch_sampler = getattr(
    train_loader,
    "batch_sampler",
    None,
)

print("\n=== TRAIN LOADER CHECK ===")
print(
    "Batch sampler              : "
    f"{type(train_batch_sampler).__name__}"
)
print(
    "Supports set_epoch         : "
    f"{hasattr(train_batch_sampler, 'set_epoch')}"
)
print(
    f"Train batches              : "
    f"{len(train_loader):,}"
)
print(
    "Configured patents/batch   : "
    f"{configured_patents_per_batch}"
)
print(
    f"Workers                    : "
    f"{getattr(train_loader, 'num_workers', 0)}"
)
print(
    f"Pinned memory              : "
    f"{getattr(train_loader, 'pin_memory', False)}"
)

if hasattr(
    train_batch_sampler,
    "set_epoch",
):
    print(
        "[PASS] Epoch-aware batch sampler is active."
    )
else:
    print(
        "[WARNING] Batch sampler has no set_epoch()."
    )


# ============================================================
# 8. Print configuration
# ============================================================

print("\n=== TRAINING CONFIGURATION ===")
print(f"Feature run                  : {FEATURE_RUN_NAME}")
print(f"Maximum epochs               : {NUM_EPOCHS}")
print(f"KL warm-up                   : {KL_WARMUP_EPOCHS}")
print(f"Hierarchy warm-up            : {HIERARCHY_WARMUP_EPOCHS}")
print(f"Reconciliation warm-up       : {RECONCILIATION_WARMUP_EPOCHS}")
print(f"Diversity warm-up            : {DIVERSITY_WARMUP_EPOCHS}")
print(f"Phi warm-up                  : {PHI_WARMUP_EPOCHS}")
print(f"Early stopping begins        : epoch {EARLY_STOPPING_START_EPOCH}")
print(f"Early-stopping patience      : {EARLY_STOPPING_PATIENCE}")
print(f"Minimum improvement          : {MINIMUM_IMPROVEMENT}")
print(f"Maximum gradient norm        : {MAX_GRAD_NORM}")
print(f"Checkpoint interval          : {CHECKPOINT_INTERVAL_EPOCHS}")
print(f"Save latest every epoch      : {SAVE_LATEST_EVERY_EPOCH}")
print(f"Train postfix interval       : {TRAIN_POSTFIX_INTERVAL}")
print(f"Eval postfix interval        : {EVAL_POSTFIX_INTERVAL}")

print("\n=== OBJECTIVE WEIGHT SCHEDULE CHECK ===")

schedule_check_epochs = sorted(
    {
        0,
        min(1, NUM_EPOCHS - 1),
        min(2, NUM_EPOCHS - 1),
        min(3, NUM_EPOCHS - 1),
        min(
            EARLY_STOPPING_START_EPOCH - 1,
            NUM_EPOCHS - 1,
        ),
    }
)

for schedule_epoch in schedule_check_epochs:
    print(
        f"Epoch {schedule_epoch + 1:02d}: "
        f"{get_objective_weights(schedule_epoch)}"
    )


# ============================================================
# 9. Fixed run directories
# ============================================================

RUN_TIMESTAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

if RESUME_TRAINING:
    if not RESUME_RUN_NAME:
        raise ValueError(
            "RESUME_RUN_NAME must be specified."
        )

    RUN_NAME = RESUME_RUN_NAME

else:
    RUN_NAME = (
        f"depth_ot_{FEATURE_RUN_NAME}_{RUN_TIMESTAMP}"
    )

CHECKPOINT_ROOT = DIRS.get(
    "depth_ot_checkpoints",
    os.path.join(
        DIRS["checkpoints"],
        "depth_ot",
    ),
)

LOG_ROOT = DIRS.get(
    "depth_ot_logs",
    DIRS["logs"],
)

RESULT_ROOT = DIRS.get(
    "depth_ot_results",
    DIRS.get(
        "results",
        LOG_ROOT,
    ),
)

RUN_CHECKPOINT_DIR = os.path.join(
    CHECKPOINT_ROOT,
    RUN_NAME,
)

RUN_LOG_DIR = os.path.join(
    LOG_ROOT,
    RUN_NAME,
)

RUN_RESULT_DIR = os.path.join(
    RESULT_ROOT,
    RUN_NAME,
)

for directory in [
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    RUN_RESULT_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )

LATEST_CHECKPOINT_PATH = os.path.join(
    RUN_CHECKPOINT_DIR,
    "latest.pt",
)

BEST_CHECKPOINT_PATH = os.path.join(
    RUN_CHECKPOINT_DIR,
    "best.pt",
)

HISTORY_JSONL_PATH = os.path.join(
    RUN_LOG_DIR,
    "history.jsonl",
)

TRAINING_SUMMARY_PATH = os.path.join(
    RUN_LOG_DIR,
    "training_summary.json",
)

RUN_MANIFEST_PATH = os.path.join(
    RUN_LOG_DIR,
    "run_manifest.json",
)

RESUME_MANIFEST_PATH = os.path.join(
    RUN_LOG_DIR,
    "resume_execution_manifest.json",
)

ERROR_LOG_PATH = os.path.join(
    RUN_LOG_DIR,
    "training_error.txt",
)

LEARNED_ANCHOR_PATH = os.path.join(
    RUN_RESULT_DIR,
    "learned_anchor.pt",
)

print("\n=== RUN DIRECTORIES ===")
print(f"Run name       : {RUN_NAME}")
print(f"Checkpoint dir : {RUN_CHECKPOINT_DIR}")
print(f"Log dir        : {RUN_LOG_DIR}")
print(f"Result dir     : {RUN_RESULT_DIR}")


# ============================================================
# 10. Verify resume checkpoint
# ============================================================

if RESUME_TRAINING:
    print("\n=== RESUME CHECK ===")
    print(
        f"Expected checkpoint:\n"
        f"  {LATEST_CHECKPOINT_PATH}"
    )
    print(
        f"Exists: "
        f"{os.path.isfile(LATEST_CHECKPOINT_PATH)}"
    )

    if not os.path.isfile(
        LATEST_CHECKPOINT_PATH
    ):
        available_checkpoints = []

        if os.path.isdir(
            CHECKPOINT_ROOT
        ):
            for candidate_run in sorted(
                os.listdir(
                    CHECKPOINT_ROOT
                )
            ):
                candidate_latest = os.path.join(
                    CHECKPOINT_ROOT,
                    candidate_run,
                    "latest.pt",
                )

                if os.path.isfile(
                    candidate_latest
                ):
                    available_checkpoints.append(
                        candidate_latest
                    )

        if not ALLOW_NEW_RUN_IF_CHECKPOINT_MISSING:
            raise FileNotFoundError(
                "Resume checkpoint를 찾지 못했습니다.\n"
                f"Expected:\n"
                f"{LATEST_CHECKPOINT_PATH}\n\n"
                "Available checkpoints:\n"
                + "\n".join(
                    available_checkpoints[-20:]
                )
            )

        print(
            "[WARNING] Checkpoint missing. "
            "Starting a new run."
        )

        RESUME_TRAINING = False


# ============================================================
# 11. Serialization utilities
# ============================================================

def config_to_dictionary(config):
    if is_dataclass(config):
        return asdict(config)

    if isinstance(
        config,
        dict,
    ):
        return dict(config)

    result = {}

    for key in dir(config):
        if key.startswith("_"):
            continue

        try:
            value = getattr(
                config,
                key,
            )
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
                list,
                tuple,
                dict,
            ),
        ):
            result[key] = value

    return result


def atomic_json_save_local(
    data,
    final_path,
):
    directory = os.path.dirname(
        final_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temporary_path = (
        final_path + ".tmp"
    )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        file.flush()

    os.replace(
        temporary_path,
        final_path,
    )


def append_jsonl(
    data,
    path,
):
    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                data,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

        file.flush()


def atomic_torch_save(
    data,
    final_path,
):
    directory = os.path.dirname(
        final_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temporary_path = (
        final_path + ".tmp"
    )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    torch.save(
        data,
        temporary_path,
    )

    os.replace(
        temporary_path,
        final_path,
    )


def safe_torch_load(
    path,
    map_location="cpu",
):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


# ============================================================
# 12. RNG utilities — CPU ByteTensor fix
# ============================================================

def _to_cpu_byte_tensor(
    state_tensor,
):
    """
    torch RNG state는 반드시 CPU ByteTensor여야 합니다.
    """

    if torch.is_tensor(
        state_tensor
    ):
        return (
            state_tensor
            .detach()
            .to(
                device="cpu",
                dtype=torch.uint8,
            )
            .contiguous()
        )

    return torch.as_tensor(
        state_tensor,
        dtype=torch.uint8,
        device="cpu",
    ).contiguous()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )
    else:
        state["torch_cuda"] = None

    if "loader_generator" in globals():
        state["loader_generator"] = (
            loader_generator.get_state()
        )

    return state


def restore_rng_state(state):
    if not state:
        return

    # Python RNG
    if state.get(
        "python"
    ) is not None:
        random.setstate(
            state["python"]
        )

    # NumPy RNG
    if state.get(
        "numpy"
    ) is not None:
        np.random.set_state(
            state["numpy"]
        )

    # PyTorch CPU RNG
    if state.get(
        "torch_cpu"
    ) is not None:
        torch_cpu_state = (
            _to_cpu_byte_tensor(
                state["torch_cpu"]
            )
        )

        torch.set_rng_state(
            torch_cpu_state
        )

    # PyTorch CUDA RNG
    if (
        torch.cuda.is_available()
        and state.get(
            "torch_cuda"
        ) is not None
    ):
        cuda_states = state[
            "torch_cuda"
        ]

        if torch.is_tensor(
            cuda_states
        ):
            cuda_states = [
                cuda_states
            ]

        converted_cuda_states = [
            _to_cpu_byte_tensor(
                cuda_state
            )
            for cuda_state in cuda_states
        ]

        if (
            len(converted_cuda_states)
            == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(
                converted_cuda_states
            )

        elif converted_cuda_states:
            torch.cuda.set_rng_state(
                converted_cuda_states[0],
                device=DEVICE,
            )

    # DataLoader generator
    if (
        "loader_generator" in globals()
        and state.get(
            "loader_generator"
        ) is not None
    ):
        loader_state = (
            _to_cpu_byte_tensor(
                state["loader_generator"]
            )
        )

        loader_generator.set_state(
            loader_state
        )


# ============================================================
# 13. Checkpoint utilities
# ============================================================

def build_checkpoint(
    epoch,
    best_validation_loss,
    epochs_without_improvement,
    history,
):
    return {
        "run_name": RUN_NAME,
        "feature_run_name": FEATURE_RUN_NAME,
        "epoch": int(epoch),
        "epoch_one_based": int(epoch) + 1,

        "model_state_dict": (
            depth_ot_model.state_dict()
        ),

        "topic_anchor_state_dict": (
            topic_anchor.state_dict()
        ),

        "main_optimizer_state_dict": (
            main_optimizer.state_dict()
        ),

        "phi_optimizer_state_dict": (
            phi_optimizer.state_dict()
        ),

        "best_validation_loss": float(
            best_validation_loss
        ),

        "epochs_without_improvement": int(
            epochs_without_improvement
        ),

        "history": history,

        "config": config_to_dictionary(
            CONFIG
        ),

        "training_settings": {
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "early_stopping_start_epoch": (
                EARLY_STOPPING_START_EPOCH
            ),
            "minimum_improvement": (
                MINIMUM_IMPROVEMENT
            ),
            "max_grad_norm": (
                MAX_GRAD_NORM
            ),
            "checkpoint_interval_epochs": (
                CHECKPOINT_INTERVAL_EPOCHS
            ),
            "kl_warmup_epochs": (
                KL_WARMUP_EPOCHS
            ),
            "hierarchy_warmup_epochs": (
                HIERARCHY_WARMUP_EPOCHS
            ),
            "reconciliation_warmup_epochs": (
                RECONCILIATION_WARMUP_EPOCHS
            ),
            "diversity_warmup_epochs": (
                DIVERSITY_WARMUP_EPOCHS
            ),
            "phi_warmup_epochs": (
                PHI_WARMUP_EPOCHS
            ),
            "patents_per_batch": (
                configured_patents_per_batch
            ),
        },

        "rng_state": capture_rng_state(),
        "saved_at": datetime.now().isoformat(),
    }


def save_checkpoint(
    path,
    epoch,
    best_validation_loss,
    epochs_without_improvement,
    history,
):
    checkpoint = build_checkpoint(
        epoch=epoch,
        best_validation_loss=(
            best_validation_loss
        ),
        epochs_without_improvement=(
            epochs_without_improvement
        ),
        history=history,
    )

    atomic_torch_save(
        checkpoint,
        path,
    )


def periodic_checkpoint_path(
    epoch_one_based,
):
    return os.path.join(
        RUN_CHECKPOINT_DIR,
        f"epoch_{epoch_one_based:03d}.pt",
    )


def load_checkpoint(
    path,
    restore_optimizers=True,
    restore_rng=True,
):
    if not os.path.isfile(
        path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    # 중요:
    # RNG state가 CUDA tensor로 변환되지 않도록
    # checkpoint 전체를 먼저 CPU에 로드합니다.
    checkpoint = safe_torch_load(
        path,
        map_location="cpu",
    )

    checkpoint_feature_run = (
        checkpoint.get(
            "feature_run_name"
        )
    )

    if (
        checkpoint_feature_run is not None
        and checkpoint_feature_run
        != FEATURE_RUN_NAME
    ):
        raise ValueError(
            "Checkpoint feature-run mismatch: "
            f"checkpoint={checkpoint_feature_run}, "
            f"current={FEATURE_RUN_NAME}"
        )

    checkpoint_settings = checkpoint.get(
        "training_settings",
        {},
    )

    checkpoint_batch_size = (
        checkpoint_settings.get(
            "patents_per_batch"
        )
    )

    if (
        checkpoint_batch_size is not None
        and configured_patents_per_batch is not None
        and int(checkpoint_batch_size)
        != int(configured_patents_per_batch)
    ):
        raise ValueError(
            "Checkpoint patents_per_batch mismatch: "
            f"checkpoint={checkpoint_batch_size}, "
            f"current={configured_patents_per_batch}"
        )

    depth_ot_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    topic_anchor.load_state_dict(
        checkpoint[
            "topic_anchor_state_dict"
        ]
    )

    depth_ot_model.to(
        DEVICE
    )

    topic_anchor.to(
        DEVICE
    )

    if restore_optimizers:
        main_optimizer.load_state_dict(
            checkpoint[
                "main_optimizer_state_dict"
            ]
        )

        phi_optimizer.load_state_dict(
            checkpoint[
                "phi_optimizer_state_dict"
            ]
        )

        optimizer_to_device(
            main_optimizer,
            DEVICE,
        )

        optimizer_to_device(
            phi_optimizer,
            DEVICE,
        )

    if restore_rng:
        try:
            restore_rng_state(
                checkpoint.get(
                    "rng_state"
                )
            )

            print(
                "[PASS] RNG state restored."
            )

        except Exception as error:
            # RNG 복원 실패 때문에 학습 전체가 중단되지 않게 함
            print(
                "[WARNING] RNG state could not be "
                f"fully restored: {error}"
            )
            print(
                "Model and optimizer were restored; "
                "training will continue."
            )

    return checkpoint


# ============================================================
# 14. Metric accumulator
# ============================================================

LOSS_NAMES = [
    "total",
    "reconstruction",
    "kl",
    "hierarchy",
    "reconciliation",
    "diversity",
]


class EpochMetricAccumulator:
    def __init__(self):
        self.loss_sums = None

        self.num_batches = 0
        self.num_patents = 0
        self.num_claims = 0
        self.num_edges = 0

        self.gradient_norm_sum = 0.0

        self.phi_update_count = 0
        self.phi_loss_sum = 0.0
        self.phi_gap_loss_sum = 0.0

        self.maximum_sinkhorn_error = 0.0
        self.failed_sinkhorn_transitions = 0

    def update(
        self,
        losses,
        batch,
        gradient_norm=None,
        phi_info=None,
        ot_output=None,
    ):
        for name in LOSS_NAMES:
            if name not in losses:
                raise KeyError(
                    f"Missing loss '{name}'."
                )

        loss_vector = torch.stack(
            [
                losses[name]
                .detach()
                .float()
                .reshape(())
                for name in LOSS_NAMES
            ]
        )

        if not torch.isfinite(
            loss_vector
        ).all():
            diagnostics = {
                name: value
                for name, value in zip(
                    LOSS_NAMES,
                    loss_vector.cpu().tolist(),
                )
            }

            raise FloatingPointError(
                f"Non-finite loss: {diagnostics}"
            )

        if self.loss_sums is None:
            self.loss_sums = (
                loss_vector.clone()
            )
        else:
            self.loss_sums.add_(
                loss_vector
            )

        self.num_batches += 1

        self.num_patents += int(
            batch["num_patents"]
        )

        self.num_claims += int(
            batch["num_claims"]
        )

        self.num_edges += int(
            batch["claim_edge_index"].shape[1]
        )

        if gradient_norm is not None:
            self.gradient_norm_sum += float(
                gradient_norm
            )

        if (
            phi_info is not None
            and phi_info.get(
                "updated",
                False,
            )
        ):
            self.phi_update_count += 1

            self.phi_loss_sum += float(
                phi_info["phi_loss"]
            )

            self.phi_gap_loss_sum += float(
                phi_info["gap_loss"]
            )

        if (
            ot_output is not None
            and ot_output.get(
                "sinkhorn_diagnostics"
            ) is not None
        ):
            for info in ot_output[
                "sinkhorn_diagnostics"
            ].values():
                maximum_error = info[
                    "maximum_marginal_error"
                ]

                if torch.is_tensor(
                    maximum_error
                ):
                    maximum_error = float(
                        maximum_error
                        .detach()
                        .cpu()
                    )
                else:
                    maximum_error = float(
                        maximum_error
                    )

                self.maximum_sinkhorn_error = max(
                    self.maximum_sinkhorn_error,
                    maximum_error,
                )

                if not bool(
                    info["converged"]
                ):
                    self.failed_sinkhorn_transitions += 1

    def compute(self):
        if self.num_batches == 0:
            raise RuntimeError(
                "No batches were accumulated."
            )

        mean_loss_vector = (
            self.loss_sums
            / self.num_batches
        )

        mean_loss_values = (
            mean_loss_vector
            .detach()
            .cpu()
            .tolist()
        )

        metrics = {
            name: float(value)
            for name, value in zip(
                LOSS_NAMES,
                mean_loss_values,
            )
        }

        metrics.update(
            {
                "num_batches": int(
                    self.num_batches
                ),

                "num_patents": int(
                    self.num_patents
                ),

                "num_claims": int(
                    self.num_claims
                ),

                "num_edges": int(
                    self.num_edges
                ),

                "mean_gradient_norm": float(
                    self.gradient_norm_sum
                    / self.num_batches
                ),

                "phi_updates": int(
                    self.phi_update_count
                ),

                "mean_phi_loss": (
                    self.phi_loss_sum
                    / self.phi_update_count
                    if self.phi_update_count > 0
                    else None
                ),

                "mean_phi_gap_loss": (
                    self.phi_gap_loss_sum
                    / self.phi_update_count
                    if self.phi_update_count > 0
                    else None
                ),

                "maximum_sinkhorn_error": float(
                    self.maximum_sinkhorn_error
                ),

                "failed_sinkhorn_transitions": int(
                    self.failed_sinkhorn_transitions
                ),
            }
        )

        return metrics


# ============================================================
# 15. Train one epoch
# ============================================================

def train_one_epoch(epoch):
    depth_ot_model.train()
    topic_anchor.train()

    if (
        hasattr(
            train_loader,
            "batch_sampler",
        )
        and hasattr(
            train_loader.batch_sampler,
            "set_epoch",
        )
    ):
        train_loader.batch_sampler.set_epoch(
            epoch
        )

    accumulator = (
        EpochMetricAccumulator()
    )

    torch.cuda.reset_peak_memory_stats()

    epoch_start = time.time()

    progress = tqdm(
        train_loader,
        desc=f"Train {epoch + 1}/{NUM_EPOCHS}",
        leave=False,
        mininterval=2.0,
    )

    for batch_index, cpu_batch in enumerate(
        progress
    ):
        batch = move_depth_ot_batch(
            cpu_batch,
            device=DEVICE,
        )

        main_optimizer.zero_grad(
            set_to_none=True
        )

        topic_anchor.zero_grad(
            set_to_none=True
        )

        output = depth_ot_model.compute_loss(
            batch=batch,
            epoch=epoch,
            deterministic=False,
            compute_hierarchy=True,
        )

        losses = output["losses"]
        total_loss = losses["total"]

        if not torch.isfinite(
            total_loss
        ):
            raise FloatingPointError(
                "Non-finite training loss at "
                f"epoch={epoch + 1}, "
                f"batch={batch_index}."
            )

        total_loss.backward()

        if (
            topic_anchor.phi.grad is not None
            and torch.any(
                topic_anchor.phi.grad != 0
            )
        ):
            raise RuntimeError(
                "Phi received gradient during "
                "main-model backward pass."
            )

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                depth_ot_model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )
        )

        gradient_norm_value = float(
            torch.as_tensor(
                gradient_norm
            ).detach().item()
        )

        if not math.isfinite(
            gradient_norm_value
        ):
            raise FloatingPointError(
                "Non-finite main gradient norm."
            )

        main_optimizer.step()

        phi_info = update_phi(
            theta=output[
                "outputs"
            ]["theta"].detach(),

            claim_depth=batch[
                "claim_depth"
            ],

            claim_edge_index=batch[
                "claim_edge_index"
            ],

            epoch=epoch,
        )

        accumulator.update(
            losses=losses,
            batch=batch,
            gradient_norm=gradient_norm_value,
            phi_info=phi_info,
            ot_output=output[
                "ot_output"
            ],
        )

        if (
            batch_index
            % TRAIN_POSTFIX_INTERVAL
            == 0
            or batch_index + 1
            == len(train_loader)
        ):
            progress.set_postfix(
                {
                    "loss": (
                        f"{total_loss.detach().item():.4f}"
                    ),
                    "rec": (
                        f"{losses['reconstruction'].detach().item():.4f}"
                    ),
                    "hier": (
                        f"{losses['hierarchy'].detach().item():.4f}"
                    ),
                    "phi": (
                        "on"
                        if phi_info.get(
                            "updated",
                            False,
                        )
                        else "off"
                    ),
                },
                refresh=False,
            )

        del cpu_batch
        del batch
        del output
        del losses
        del total_loss
        del phi_info

    epoch_seconds = (
        time.time()
        - epoch_start
    )

    metrics = accumulator.compute()

    metrics["epoch_seconds"] = float(
        epoch_seconds
    )

    metrics["batches_per_second"] = float(
        metrics["num_batches"]
        / max(
            epoch_seconds,
            1e-9,
        )
    )

    metrics["patents_per_second"] = float(
        metrics["num_patents"]
        / max(
            epoch_seconds,
            1e-9,
        )
    )

    metrics["peak_gpu_gib"] = float(
        torch.cuda.max_memory_allocated()
        / 2**30
    )

    metrics["peak_reserved_gib"] = float(
        torch.cuda.max_memory_reserved()
        / 2**30
    )

    return metrics


# ============================================================
# 16. Evaluate one epoch
# ============================================================

@torch.inference_mode()
def evaluate_one_epoch(
    data_loader,
    epoch,
    description,
):
    depth_ot_model.eval()
    topic_anchor.eval()

    accumulator = (
        EpochMetricAccumulator()
    )

    progress = tqdm(
        data_loader,
        desc=description,
        leave=False,
        mininterval=2.0,
    )

    for batch_index, cpu_batch in enumerate(
        progress
    ):
        batch = move_depth_ot_batch(
            cpu_batch,
            device=DEVICE,
        )

        output = depth_ot_model.compute_loss(
            batch=batch,
            epoch=epoch,
            deterministic=True,
            compute_hierarchy=True,
        )

        losses = output["losses"]

        if not torch.isfinite(
            losses["total"]
        ):
            raise FloatingPointError(
                f"Non-finite evaluation loss "
                f"in {description}, "
                f"batch={batch_index}."
            )

        accumulator.update(
            losses=losses,
            batch=batch,
            gradient_norm=None,
            phi_info=None,
            ot_output=output[
                "ot_output"
            ],
        )

        if (
            batch_index
            % EVAL_POSTFIX_INTERVAL
            == 0
            or batch_index + 1
            == len(data_loader)
        ):
            progress.set_postfix(
                {
                    "loss": (
                        f"{losses['total'].detach().item():.4f}"
                    )
                },
                refresh=False,
            )

        del cpu_batch
        del batch
        del output
        del losses

    return accumulator.compute()


# ============================================================
# 17. Run manifest
# ============================================================

run_manifest = {
    "run_name": RUN_NAME,
    "execution_started_at": (
        datetime.now().isoformat()
    ),
    "resumed": RESUME_TRAINING,
    "feature_run_name": FEATURE_RUN_NAME,
    "device": str(DEVICE),
    "gpu": GPU_NAME,
    "gpu_memory_gib": GPU_TOTAL_GIB,
    "tf32_enabled": True,
    "mixed_precision": USE_MIXED_PRECISION,
    "num_epochs": NUM_EPOCHS,
    "patents_per_batch": (
        configured_patents_per_batch
    ),
    "train_batches": len(
        train_loader
    ),
    "num_workers": getattr(
        train_loader,
        "num_workers",
        0,
    ),
    "warmup": {
        "kl": KL_WARMUP_EPOCHS,
        "hierarchy": (
            HIERARCHY_WARMUP_EPOCHS
        ),
        "reconciliation": (
            RECONCILIATION_WARMUP_EPOCHS
        ),
        "diversity": (
            DIVERSITY_WARMUP_EPOCHS
        ),
        "phi": PHI_WARMUP_EPOCHS,
    },
    "early_stopping_patience": (
        EARLY_STOPPING_PATIENCE
    ),
    "early_stopping_start_epoch": (
        EARLY_STOPPING_START_EPOCH
    ),
    "minimum_improvement": (
        MINIMUM_IMPROVEMENT
    ),
    "checkpoint_directory": (
        RUN_CHECKPOINT_DIR
    ),
    "log_directory": RUN_LOG_DIR,
    "result_directory": RUN_RESULT_DIR,
    "dataset_sizes": {
        "train_patents": len(
            train_dataset
        ),
        "dev_patents": len(
            dev_dataset
        ),
        "test_patents": len(
            test_dataset
        ),
    },
    "config": config_to_dictionary(
        CONFIG
    ),
}

atomic_json_save_local(
    run_manifest,
    RESUME_MANIFEST_PATH,
)

if not os.path.isfile(
    RUN_MANIFEST_PATH
):
    atomic_json_save_local(
        run_manifest,
        RUN_MANIFEST_PATH,
    )


# ============================================================
# 18. Resume state
# ============================================================

start_epoch = 0
best_validation_loss = float("inf")
epochs_without_improvement = 0
training_history = []

if RESUME_TRAINING:
    resumed_checkpoint = load_checkpoint(
        LATEST_CHECKPOINT_PATH,
        restore_optimizers=True,
        restore_rng=True,
    )

    checkpoint_epoch = int(
        resumed_checkpoint["epoch"]
    )

    start_epoch = (
        checkpoint_epoch + 1
    )

    best_validation_loss = float(
        resumed_checkpoint.get(
            "best_validation_loss",
            float("inf"),
        )
    )

    epochs_without_improvement = int(
        resumed_checkpoint.get(
            "epochs_without_improvement",
            0,
        )
    )

    training_history = list(
        resumed_checkpoint.get(
            "history",
            [],
        )
    )

    print("\n" + "=" * 90)
    print("CHECKPOINT RESTORED")
    print("=" * 90)
    print(
        f"Completed through epoch : "
        f"{checkpoint_epoch + 1}"
    )
    print(
        f"Next epoch              : "
        f"{start_epoch + 1}"
    )
    print(
        f"Previous best dev       : "
        f"{best_validation_loss}"
    )
    print(
        f"Patience counter        : "
        f"{epochs_without_improvement}/"
        f"{EARLY_STOPPING_PATIENCE}"
    )
    print(
        f"History records         : "
        f"{len(training_history)}"
    )
    print("=" * 90)

else:
    print(
        "\nStarting a new training run."
    )


# ============================================================
# 19. Training loop
# ============================================================

already_completed = (
    start_epoch >= NUM_EPOCHS
)

if already_completed:
    print(
        f"\n[Already complete] "
        f"Checkpoint contains "
        f"{start_epoch} completed epochs."
    )

print("\n" + "=" * 90)
print("TRAINING STARTED/RESUMED")
print("=" * 90)

execution_start_time = time.time()

initial_history_length = len(
    training_history
)

stopped_early = False

try:
    if not already_completed:
        for epoch in range(
            start_epoch,
            NUM_EPOCHS,
        ):
            epoch_one_based = (
                epoch + 1
            )

            epoch_start_time = (
                time.time()
            )

            current_weights = (
                get_objective_weights(
                    epoch
                )
            )

            print("\n" + "-" * 90)
            print(
                f"Starting epoch "
                f"{epoch_one_based}/"
                f"{NUM_EPOCHS}"
            )
            print(
                f"Objective weights: "
                f"{current_weights}"
            )
            print("-" * 90)

            train_metrics = (
                train_one_epoch(
                    epoch
                )
            )

            dev_metrics = (
                evaluate_one_epoch(
                    data_loader=dev_loader,
                    epoch=epoch,
                    description=(
                        f"Dev "
                        f"{epoch_one_based}/"
                        f"{NUM_EPOCHS}"
                    ),
                )
            )

            epoch_duration = (
                time.time()
                - epoch_start_time
            )

            can_monitor_improvement = (
                epoch_one_based
                >= EARLY_STOPPING_START_EPOCH
            )

            improved = False

            if can_monitor_improvement:
                validation_loss = float(
                    dev_metrics["total"]
                )

                improved = (
                    validation_loss
                    < best_validation_loss
                    - MINIMUM_IMPROVEMENT
                )

                if improved:
                    best_validation_loss = (
                        validation_loss
                    )

                    epochs_without_improvement = 0

                else:
                    epochs_without_improvement += 1

            with torch.no_grad():
                anchor_coordinates = (
                    topic_anchor.coordinates()
                    .detach()
                    .cpu()
                    .tolist()
                )

                anchor_gap_loss = float(
                    topic_anchor.gap_loss()
                    .detach()
                    .item()
                )

            epoch_record = {
                "epoch": int(
                    epoch
                ),

                "epoch_one_based": int(
                    epoch_one_based
                ),

                "duration_seconds": float(
                    epoch_duration
                ),

                "weights": (
                    current_weights
                ),

                "train": (
                    train_metrics
                ),

                "dev": (
                    dev_metrics
                ),

                "anchor_coordinates": (
                    anchor_coordinates
                ),

                "anchor_gap_loss": (
                    anchor_gap_loss
                ),

                "early_stopping_active": (
                    can_monitor_improvement
                ),

                "best_validation_loss": (
                    None
                    if not math.isfinite(
                        best_validation_loss
                    )
                    else float(
                        best_validation_loss
                    )
                ),

                "improved": bool(
                    improved
                ),

                "epochs_without_improvement": int(
                    epochs_without_improvement
                ),

                "main_learning_rate": float(
                    main_optimizer.param_groups[
                        0
                    ]["lr"]
                ),

                "phi_learning_rate": float(
                    phi_optimizer.param_groups[
                        0
                    ]["lr"]
                ),

                "gpu_name": GPU_NAME,
            }

            # 동일 epoch 중복 방지
            training_history = [
                record
                for record in training_history
                if int(
                    record.get(
                        "epoch",
                        -1,
                    )
                ) != int(epoch)
            ]

            training_history.append(
                epoch_record
            )

            training_history.sort(
                key=lambda record: int(
                    record.get(
                        "epoch",
                        -1,
                    )
                )
            )

            append_jsonl(
                epoch_record,
                HISTORY_JSONL_PATH,
            )

            # latest.pt 매 epoch 저장
            if SAVE_LATEST_EVERY_EPOCH:
                save_checkpoint(
                    path=LATEST_CHECKPOINT_PATH,
                    epoch=epoch,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    history=training_history,
                )

            periodic_saved_path = None

            if (
                epoch_one_based
                % CHECKPOINT_INTERVAL_EPOCHS
                == 0
            ):
                periodic_saved_path = (
                    periodic_checkpoint_path(
                        epoch_one_based
                    )
                )

                save_checkpoint(
                    path=periodic_saved_path,
                    epoch=epoch,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    history=training_history,
                )

            if improved:
                save_checkpoint(
                    path=BEST_CHECKPOINT_PATH,
                    epoch=epoch,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    history=training_history,
                )

            print(
                f"\nEpoch "
                f"{epoch_one_based:03d}/"
                f"{NUM_EPOCHS:03d} | "
                f"{epoch_duration:.1f}s "
                f"({epoch_duration/3600:.2f}h)"
            )

            print(
                f"  train total="
                f"{train_metrics['total']:.6f}, "
                f"rec="
                f"{train_metrics['reconstruction']:.6f}, "
                f"kl="
                f"{train_metrics['kl']:.6f}, "
                f"hier="
                f"{train_metrics['hierarchy']:.6f}"
            )

            print(
                f"  dev   total="
                f"{dev_metrics['total']:.6f}, "
                f"rec="
                f"{dev_metrics['reconstruction']:.6f}, "
                f"kl="
                f"{dev_metrics['kl']:.6f}, "
                f"hier="
                f"{dev_metrics['hierarchy']:.6f}"
            )

            print(
                f"  phi updates="
                f"{train_metrics['phi_updates']}, "
                f"gap={anchor_gap_loss:.6f}, "
                f"max Sinkhorn error="
                f"{train_metrics['maximum_sinkhorn_error']:.3e}"
            )

            print(
                f"  speed="
                f"{train_metrics['patents_per_second']:.2f} patents/s, "
                f"peak GPU="
                f"{train_metrics['peak_gpu_gib']:.2f} GiB"
            )

            print(
                f"  latest checkpoint: "
                f"{LATEST_CHECKPOINT_PATH}"
            )

            if periodic_saved_path:
                print(
                    f"  periodic checkpoint: "
                    f"{periodic_saved_path}"
                )

            if improved:
                print(
                    f"  best checkpoint updated: "
                    f"{BEST_CHECKPOINT_PATH}"
                )

            if can_monitor_improvement:
                print(
                    f"  best dev="
                    f"{best_validation_loss:.6f}, "
                    f"patience="
                    f"{epochs_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE}"
                )
            else:
                print(
                    "  early stopping inactive; "
                    f"starts at epoch "
                    f"{EARLY_STOPPING_START_EPOCH}"
                )

            if (
                can_monitor_improvement
                and epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                stopped_early = True

                print(
                    f"\nEarly stopping triggered "
                    f"at epoch {epoch_one_based}."
                )

                break

            gc.collect()

except Exception as training_error:
    error_trace = traceback.format_exc()

    print("\n" + "!" * 90)
    print("TRAINING INTERRUPTED BY ERROR")
    print("!" * 90)
    print(str(training_error))
    print(error_trace)

    with open(
        ERROR_LOG_PATH,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n"
            + "=" * 90
            + "\n"
        )

        file.write(
            datetime.now().isoformat()
            + "\n"
        )

        file.write(
            str(training_error)
            + "\n"
        )

        file.write(
            error_trace
            + "\n"
        )

    gc.collect()
    torch.cuda.empty_cache()

    raise


execution_duration = (
    time.time()
    - execution_start_time
)


# ============================================================
# 20. Validate latest checkpoint
# ============================================================

if not training_history:
    raise RuntimeError(
        "Training history is empty."
    )

last_completed_epoch = int(
    training_history[-1]["epoch"]
)

if not os.path.isfile(
    LATEST_CHECKPOINT_PATH
):
    save_checkpoint(
        path=LATEST_CHECKPOINT_PATH,
        epoch=last_completed_epoch,
        best_validation_loss=(
            best_validation_loss
        ),
        epochs_without_improvement=(
            epochs_without_improvement
        ),
        history=training_history,
    )


# ============================================================
# 21. Ensure best checkpoint exists
# ============================================================

if not os.path.isfile(
    BEST_CHECKPOINT_PATH
):
    print(
        "[WARNING] best.pt does not exist. "
        "Using latest.pt as fallback."
    )

    latest_checkpoint = safe_torch_load(
        LATEST_CHECKPOINT_PATH,
        map_location="cpu",
    )

    atomic_torch_save(
        latest_checkpoint,
        BEST_CHECKPOINT_PATH,
    )


# ============================================================
# 22. Restore best checkpoint
# ============================================================

best_checkpoint = load_checkpoint(
    BEST_CHECKPOINT_PATH,
    restore_optimizers=False,
    restore_rng=False,
)

best_epoch = int(
    best_checkpoint["epoch"]
)

best_validation_loss = float(
    best_checkpoint.get(
        "best_validation_loss",
        best_validation_loss,
    )
)

depth_ot_model.to(
    DEVICE
)

topic_anchor.to(
    DEVICE
)

print(
    f"\nBest checkpoint restored "
    f"from epoch {best_epoch + 1}."
)


# ============================================================
# 23. Final dev/test evaluation
# ============================================================

print("\n" + "=" * 90)
print("FINAL EVALUATION")
print("=" * 90)

final_dev_metrics = evaluate_one_epoch(
    data_loader=dev_loader,
    epoch=best_epoch,
    description="Final dev evaluation",
)

final_test_metrics = evaluate_one_epoch(
    data_loader=test_loader,
    epoch=best_epoch,
    description="Final test evaluation",
)

with torch.no_grad():
    final_anchor_coordinates = (
        topic_anchor.coordinates()
        .detach()
        .cpu()
        .tolist()
    )

    final_cost_matrix = (
        topic_anchor.cost_matrix()
        .detach()
        .cpu()
    )


# ============================================================
# 24. Periodic checkpoint list
# ============================================================

periodic_checkpoint_paths = []

for completed_epoch in range(
    CHECKPOINT_INTERVAL_EPOCHS,
    NUM_EPOCHS + 1,
    CHECKPOINT_INTERVAL_EPOCHS,
):
    candidate_path = (
        periodic_checkpoint_path(
            completed_epoch
        )
    )

    if os.path.isfile(
        candidate_path
    ):
        periodic_checkpoint_paths.append(
            candidate_path
        )


# ============================================================
# 25. Save final results
# ============================================================

epochs_completed_this_execution = max(
    0,
    len(training_history)
    - initial_history_length,
)

training_summary = {
    "run_name": RUN_NAME,
    "feature_run_name": FEATURE_RUN_NAME,

    "resumed": (
        RESUME_TRAINING
    ),

    "resume_start_epoch_one_based": (
        start_epoch + 1
        if start_epoch < NUM_EPOCHS
        else None
    ),

    "execution_started_at": (
        datetime.fromtimestamp(
            execution_start_time
        ).isoformat()
    ),

    "execution_completed_at": (
        datetime.now().isoformat()
    ),

    "execution_duration_seconds": (
        execution_duration
    ),

    "stopped_early": (
        stopped_early
    ),

    "epochs_completed_this_execution": (
        epochs_completed_this_execution
    ),

    "total_epochs_recorded": len(
        training_history
    ),

    "last_completed_epoch": (
        last_completed_epoch
    ),

    "last_completed_epoch_one_based": (
        last_completed_epoch + 1
    ),

    "best_epoch": (
        best_epoch
    ),

    "best_epoch_one_based": (
        best_epoch + 1
    ),

    "best_validation_loss": float(
        best_validation_loss
    ),

    "final_dev": (
        final_dev_metrics
    ),

    "final_test": (
        final_test_metrics
    ),

    "final_anchor_coordinates": (
        final_anchor_coordinates
    ),

    "gpu": GPU_NAME,
    "gpu_memory_gib": GPU_TOTAL_GIB,

    "training_settings": {
        "num_epochs": NUM_EPOCHS,

        "patents_per_batch": (
            configured_patents_per_batch
        ),

        "warmup_epochs": {
            "kl": KL_WARMUP_EPOCHS,
            "hierarchy": (
                HIERARCHY_WARMUP_EPOCHS
            ),
            "reconciliation": (
                RECONCILIATION_WARMUP_EPOCHS
            ),
            "diversity": (
                DIVERSITY_WARMUP_EPOCHS
            ),
            "phi": PHI_WARMUP_EPOCHS,
        },

        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),

        "early_stopping_start_epoch": (
            EARLY_STOPPING_START_EPOCH
        ),

        "minimum_improvement": (
            MINIMUM_IMPROVEMENT
        ),

        "checkpoint_interval_epochs": (
            CHECKPOINT_INTERVAL_EPOCHS
        ),

        "latest_every_epoch": (
            SAVE_LATEST_EVERY_EPOCH
        ),

        "mixed_precision": (
            USE_MIXED_PRECISION
        ),

        "tf32": True,

        "num_workers": getattr(
            train_loader,
            "num_workers",
            0,
        ),
    },

    "latest_checkpoint": (
        LATEST_CHECKPOINT_PATH
    ),

    "best_checkpoint": (
        BEST_CHECKPOINT_PATH
    ),

    "periodic_checkpoints": (
        periodic_checkpoint_paths
    ),

    "history_path": (
        HISTORY_JSONL_PATH
    ),

    "manifest_path": (
        RUN_MANIFEST_PATH
    ),
}

atomic_json_save_local(
    training_summary,
    TRAINING_SUMMARY_PATH,
)

atomic_torch_save(
    {
        "coordinates": torch.tensor(
            final_anchor_coordinates,
            dtype=torch.float32,
        ),

        "cost_matrix": (
            final_cost_matrix
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_epoch_one_based": (
            best_epoch + 1
        ),

        "run_name": (
            RUN_NAME
        ),
    },
    LEARNED_ANCHOR_PATH,
)


# ============================================================
# 26. Final validation
# ============================================================

required_final_files = [
    LATEST_CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
    HISTORY_JSONL_PATH,
    TRAINING_SUMMARY_PATH,
    RUN_MANIFEST_PATH,
    LEARNED_ANCHOR_PATH,
]

missing_final_files = [
    path
    for path in required_final_files
    if not os.path.isfile(path)
]

if missing_final_files:
    raise RuntimeError(
        "Required output files are missing: "
        f"{missing_final_files}"
    )


# ============================================================
# 27. Final report
# ============================================================

print("\n" + "=" * 90)
print("SECTION 6 COMPLETED SUCCESSFULLY")
print("=" * 90)

print(f"Run name         : {RUN_NAME}")
print(f"GPU              : {GPU_NAME}")
print(f"Resumed          : {RESUME_TRAINING}")

print(
    f"Epochs this run  : "
    f"{epochs_completed_this_execution}"
)

print(
    f"Total recorded   : "
    f"{len(training_history)}"
)

print(
    f"Last epoch       : "
    f"{last_completed_epoch + 1}"
)

print(
    f"Stopped early    : "
    f"{stopped_early}"
)

print(
    f"Best epoch       : "
    f"{best_epoch + 1}"
)

print(
    f"Best dev loss    : "
    f"{best_validation_loss:.6f}"
)

print(
    f"Final dev loss   : "
    f"{final_dev_metrics['total']:.6f}"
)

print(
    f"Final test loss  : "
    f"{final_test_metrics['total']:.6f}"
)

print(
    f"Latest checkpoint: "
    f"{LATEST_CHECKPOINT_PATH}"
)

print(
    f"Best checkpoint  : "
    f"{BEST_CHECKPOINT_PATH}"
)

print(
    f"Periodic saved   : "
    f"{len(periodic_checkpoint_paths)} files"
)

for checkpoint_path in periodic_checkpoint_paths:
    print(
        f"  - {checkpoint_path}"
    )

print(
    f"Training summary : "
    f"{TRAINING_SUMMARY_PATH}"
)

print(
    f"Learned anchor   : "
    f"{LEARNED_ANCHOR_PATH}"
)

print("=" * 90)

print(
    "\nBest model is loaded and ready for "
    "Section 7 deterministic inference."
)


# ============================================================
# SECTION 7:
# Deterministic Inference and Global Topic Hierarchy Extraction
# COMPLETE SINGLE-CELL VERSION
# ============================================================

import os
import json
import math
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ============================================================
# 0. Preconditions
# ============================================================

required_section7_globals = [
    "CONFIG",
    "DIRS",
    "DEVICE",
    "FEATURE_RUN_NAME",
    "depth_ot_model",
    "topic_anchor",
    "train_dataset",
    "dev_dataset",
    "test_dataset",
    "patent_collate_fn",
    "move_depth_ot_batch",
    "VOCAB",
    "RUN_RESULT_DIR",
    "BEST_CHECKPOINT_PATH",
]

missing_section7_globals = [
    name
    for name in required_section7_globals
    if name not in globals()
]

if missing_section7_globals:
    raise RuntimeError(
        "Section 7 prerequisites are missing: "
        f"{missing_section7_globals}. "
        "Run Sections 0 through 6 first."
    )

if not os.path.isfile(
    BEST_CHECKPOINT_PATH
):
    raise FileNotFoundError(
        "Best checkpoint does not exist: "
        f"{BEST_CHECKPOINT_PATH}"
    )

if not isinstance(VOCAB, list):
    raise TypeError(
        "VOCAB must be a list."
    )

if len(VOCAB) != int(
    getattr(
        CONFIG,
        "vocab_size",
        len(VOCAB),
    )
):
    raise ValueError(
        "Vocabulary size does not match CONFIG.vocab_size."
    )


# ============================================================
# 1. Section 7 configuration
# ============================================================

SECTION7_RESULT_DIR = os.path.join(
    RUN_RESULT_DIR,
    "section7_inference",
)

INFERENCE_SHARD_DIR = os.path.join(
    SECTION7_RESULT_DIR,
    "claim_theta_shards",
)

os.makedirs(
    SECTION7_RESULT_DIR,
    exist_ok=True,
)

os.makedirs(
    INFERENCE_SHARD_DIR,
    exist_ok=True,
)

# Number of DataLoader batches stored in one inference shard.
INFERENCE_BATCHES_PER_SHARD = int(
    getattr(
        CONFIG,
        "inference_batches_per_shard",
        100,
    )
)

# Number of child topics retained for each parent topic.
GLOBAL_HIERARCHY_TOP_K = int(
    getattr(
        CONFIG,
        "global_hierarchy_top_k",
        3,
    )
)

# Minimum conditional transition score required for an edge.
GLOBAL_EDGE_MIN_CONDITIONAL = float(
    getattr(
        CONFIG,
        "global_edge_min_conditional",
        0.01,
    )
)

# Number of representative words exported per topic.
TOP_WORDS_PER_TOPIC = int(
    getattr(
        CONFIG,
        "top_words_per_topic",
        20,
    )
)

INFERENCE_BATCH_SIZE = int(
    getattr(
        CONFIG,
        "inference_patents_per_batch",
        globals().get(
            "PATENTS_PER_BATCH",
            16,
        ),
    )
)

if INFERENCE_BATCHES_PER_SHARD < 1:
    raise ValueError(
        "INFERENCE_BATCHES_PER_SHARD must be positive."
    )

if GLOBAL_HIERARCHY_TOP_K < 1:
    raise ValueError(
        "GLOBAL_HIERARCHY_TOP_K must be positive."
    )

if not (
    0.0
    <= GLOBAL_EDGE_MIN_CONDITIONAL
    <= 1.0
):
    raise ValueError(
        "GLOBAL_EDGE_MIN_CONDITIONAL must be in [0, 1]."
    )

if TOP_WORDS_PER_TOPIC < 1:
    raise ValueError(
        "TOP_WORDS_PER_TOPIC must be positive."
    )

print("=== SECTION 7 CONFIGURATION ===")
print(
    f"Feature run                 : "
    f"{FEATURE_RUN_NAME}"
)
print(
    f"Best checkpoint             : "
    f"{BEST_CHECKPOINT_PATH}"
)
print(
    f"Result directory            : "
    f"{SECTION7_RESULT_DIR}"
)
print(
    f"Inference patents per batch : "
    f"{INFERENCE_BATCH_SIZE}"
)
print(
    f"Batches per output shard    : "
    f"{INFERENCE_BATCHES_PER_SHARD}"
)
print(
    f"Global hierarchy top-k      : "
    f"{GLOBAL_HIERARCHY_TOP_K}"
)
print(
    f"Minimum conditional score   : "
    f"{GLOBAL_EDGE_MIN_CONDITIONAL}"
)
print(
    f"Top words per topic         : "
    f"{TOP_WORDS_PER_TOPIC}"
)


# ============================================================
# 2. Serialization utilities
# ============================================================

def atomic_json_save_section7(
    data,
    final_path,
):
    os.makedirs(
        os.path.dirname(final_path),
        exist_ok=True,
    )

    temporary_path = (
        final_path + ".tmp"
    )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    os.replace(
        temporary_path,
        final_path,
    )


def atomic_torch_save_section7(
    data,
    final_path,
):
    os.makedirs(
        os.path.dirname(final_path),
        exist_ok=True,
    )

    temporary_path = (
        final_path + ".tmp"
    )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    torch.save(
        data,
        temporary_path,
    )

    os.replace(
        temporary_path,
        final_path,
    )


# ============================================================
# 3. Restore best checkpoint
# ============================================================

best_checkpoint_section7 = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

checkpoint_feature_run = (
    best_checkpoint_section7.get(
        "feature_run_name"
    )
)

if (
    checkpoint_feature_run
    != FEATURE_RUN_NAME
):
    raise ValueError(
        "Best checkpoint feature-run mismatch: "
        f"checkpoint={checkpoint_feature_run}, "
        f"current={FEATURE_RUN_NAME}."
    )

depth_ot_model.load_state_dict(
    best_checkpoint_section7[
        "model_state_dict"
    ]
)

topic_anchor.load_state_dict(
    best_checkpoint_section7[
        "topic_anchor_state_dict"
    ]
)

depth_ot_model.to(
    DEVICE
)

topic_anchor.to(
    DEVICE
)

depth_ot_model.eval()
topic_anchor.eval()

BEST_EPOCH_SECTION7 = int(
    best_checkpoint_section7[
        "epoch"
    ]
)

print(
    "\nBest checkpoint restored for inference."
)
print(
    f"Best epoch: "
    f"{BEST_EPOCH_SECTION7 + 1}"
)


# ============================================================
# 4. Deterministic inference loaders
# ============================================================

inference_loader_kwargs = {
    "batch_size": (
        INFERENCE_BATCH_SIZE
    ),
    "shuffle": False,
    "collate_fn": patent_collate_fn,
    "num_workers": 0,
    "pin_memory": (
        torch.cuda.is_available()
    ),
    "persistent_workers": False,
    "drop_last": False,
}

section7_train_loader = DataLoader(
    train_dataset,
    **inference_loader_kwargs,
)

section7_dev_loader = DataLoader(
    dev_dataset,
    **inference_loader_kwargs,
)

section7_test_loader = DataLoader(
    test_dataset,
    **inference_loader_kwargs,
)

SECTION7_LOADERS = {
    "train": section7_train_loader,
    "dev": section7_dev_loader,
    "test": section7_test_loader,
}

print("\n=== DETERMINISTIC INFERENCE LOADERS ===")

for split_name, loader in (
    SECTION7_LOADERS.items()
):
    print(
        f"{split_name:5s}: "
        f"patents={len(loader.dataset):,}, "
        f"batches={len(loader):,}"
    )


# ============================================================
# 5. Model-output extraction
# ============================================================

@torch.no_grad()
def run_deterministic_forward(
    batch,
):
    """
    Use Section 5's compute_loss interface because this interface
    has already been validated during Sections 5 and 6.

    compute_hierarchy=False prevents unnecessary Sinkhorn
    computation during claim-level inference.
    """

    result = (
        depth_ot_model.compute_loss(
            batch=batch,
            epoch=BEST_EPOCH_SECTION7,
            deterministic=True,
            compute_hierarchy=False,
        )
    )

    if not isinstance(result, dict):
        raise TypeError(
            "depth_ot_model.compute_loss() must return a dictionary."
        )

    if "outputs" not in result:
        raise KeyError(
            "Model result does not contain 'outputs'."
        )

    model_outputs = result[
        "outputs"
    ]

    if "theta" not in model_outputs:
        raise KeyError(
            "Model outputs do not contain 'theta'."
        )

    theta = model_outputs[
        "theta"
    ]

    if theta.ndim != 2:
        raise ValueError(
            "theta must have shape [num_claims, num_topics], "
            f"got {tuple(theta.shape)}."
        )

    if not torch.isfinite(
        theta
    ).all():
        raise FloatingPointError(
            "Non-finite theta values detected."
        )

    if torch.any(theta < 0):
        raise ValueError(
            "Negative theta values detected."
        )

    theta_sums = theta.sum(
        dim=-1
    )

    if not torch.allclose(
        theta_sums,
        torch.ones_like(
            theta_sums
        ),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Theta rows do not sum to one."
        )

    return theta, model_outputs


# ============================================================
# 6. Deterministic reproducibility test
# ============================================================

deterministic_cpu_batch = next(
    iter(section7_dev_loader)
)

deterministic_batch = (
    move_depth_ot_batch(
        deterministic_cpu_batch,
        device=DEVICE,
    )
)

with torch.no_grad():
    deterministic_theta_1, _ = (
        run_deterministic_forward(
            deterministic_batch
        )
    )

    deterministic_theta_2, _ = (
        run_deterministic_forward(
            deterministic_batch
        )
    )

maximum_deterministic_difference = float(
    (
        deterministic_theta_1
        - deterministic_theta_2
    )
    .abs()
    .max()
    .item()
)

if not torch.equal(
    deterministic_theta_1,
    deterministic_theta_2,
):
    if not torch.allclose(
        deterministic_theta_1,
        deterministic_theta_2,
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Deterministic inference is not reproducible. "
            f"Maximum difference="
            f"{maximum_deterministic_difference:.3e}."
        )

NUM_TOPICS = int(
    deterministic_theta_1.shape[1]
)

configured_num_topics = int(
    getattr(
        CONFIG,
        "num_topics",
        NUM_TOPICS,
    )
)

if NUM_TOPICS != configured_num_topics:
    raise ValueError(
        "Topic-count mismatch: "
        f"theta={NUM_TOPICS}, "
        f"CONFIG={configured_num_topics}."
    )

print("\n=== DETERMINISTIC INFERENCE TEST ===")
print(
    f"Theta shape       : "
    f"{tuple(deterministic_theta_1.shape)}"
)
print(
    f"Number of topics  : "
    f"{NUM_TOPICS}"
)
print(
    f"Maximum difference: "
    f"{maximum_deterministic_difference:.3e}"
)
print(
    "[PASS] Deterministic inference is reproducible."
)

del deterministic_batch
del deterministic_theta_1
del deterministic_theta_2


# ============================================================
# 7. Extract topic-word distribution
# ============================================================

def get_topic_word_decoder(
    model,
):
    candidate_names = [
        "topic_word_decoder",
        "topic_word",
        "decoder",
    ]

    for candidate_name in candidate_names:
        if hasattr(
            model,
            candidate_name,
        ):
            candidate = getattr(
                model,
                candidate_name,
            )

            if candidate is not None:
                return candidate

    raise AttributeError(
        "Could not locate the topic-word decoder "
        "inside depth_ot_model."
    )


topic_word_decoder_section7 = (
    get_topic_word_decoder(
        depth_ot_model
    )
)

with torch.no_grad():
    if hasattr(
        topic_word_decoder_section7,
        "get_beta",
    ):
        beta = (
            topic_word_decoder_section7
            .get_beta()
        )
    elif hasattr(
        topic_word_decoder_section7,
        "beta",
    ):
        beta_attribute = getattr(
            topic_word_decoder_section7,
            "beta",
        )

        beta = (
            beta_attribute()
            if callable(beta_attribute)
            else beta_attribute
        )
    elif hasattr(
        topic_word_decoder_section7,
        "beta_logits",
    ):
        beta = torch.softmax(
            topic_word_decoder_section7.beta_logits,
            dim=-1,
        )
    else:
        raise AttributeError(
            "Topic-word decoder has no get_beta(), "
            "beta, or beta_logits attribute."
        )

    beta = (
        beta.detach()
        .float()
        .cpu()
    )

if beta.shape != (
    NUM_TOPICS,
    len(VOCAB),
):
    raise ValueError(
        "Beta shape mismatch: "
        f"observed={tuple(beta.shape)}, "
        f"expected={(NUM_TOPICS, len(VOCAB))}."
    )

if not torch.isfinite(
    beta
).all():
    raise FloatingPointError(
        "Beta contains non-finite values."
    )

if torch.any(beta < 0):
    raise ValueError(
        "Beta contains negative values."
    )

if not torch.allclose(
    beta.sum(dim=-1),
    torch.ones(
        NUM_TOPICS,
        dtype=beta.dtype,
    ),
    atol=1e-5,
    rtol=1e-5,
):
    raise ValueError(
        "Beta rows do not sum to one."
    )

topic_word_path = os.path.join(
    SECTION7_RESULT_DIR,
    "topic_word_distribution.pt",
)

atomic_torch_save_section7(
    {
        "beta": beta,
        "vocabulary": VOCAB,
        "number_of_topics": NUM_TOPICS,
        "vocabulary_size": len(VOCAB),
        "best_epoch": BEST_EPOCH_SECTION7,
        "feature_run_name": (
            FEATURE_RUN_NAME
        ),
    },
    topic_word_path,
)

print("\n=== TOPIC-WORD DISTRIBUTION ===")
print(f"Beta shape: {tuple(beta.shape)}")
print(f"Saved to  : {topic_word_path}")


# ============================================================
# 8. Representative topic words
# ============================================================

actual_top_words = min(
    TOP_WORDS_PER_TOPIC,
    len(VOCAB),
)

top_word_probabilities, top_word_indices = (
    torch.topk(
        beta,
        k=actual_top_words,
        dim=-1,
    )
)

topic_word_summaries = []

for topic_index in range(
    NUM_TOPICS
):
    words = []

    for rank in range(
        actual_top_words
    ):
        vocabulary_index = int(
            top_word_indices[
                topic_index,
                rank,
            ].item()
        )

        words.append(
            {
                "rank": rank + 1,
                "word": VOCAB[
                    vocabulary_index
                ],
                "vocabulary_index": (
                    vocabulary_index
                ),
                "probability": float(
                    top_word_probabilities[
                        topic_index,
                        rank,
                    ].item()
                ),
            }
        )

    topic_word_summaries.append(
        {
            "topic_id": topic_index,
            "topic_number": (
                topic_index + 1
            ),
            "top_words": words,
        }
    )


# ============================================================
# 9. Inference accumulator
# ============================================================

class GlobalTransitionAccumulator:
    """
    Aggregate claim-topic distributions and adjacent-depth
    parent-child topic-transition evidence.

    The hierarchy is constructed from train data only.
    """

    def __init__(
        self,
        number_of_topics,
    ):
        self.number_of_topics = int(
            number_of_topics
        )

        self.topic_mass = torch.zeros(
            self.number_of_topics,
            dtype=torch.float64,
        )

        self.depth_topic_mass = (
            defaultdict(
                lambda: torch.zeros(
                    self.number_of_topics,
                    dtype=torch.float64,
                )
            )
        )

        self.depth_claim_counts = (
            defaultdict(int)
        )

        self.global_joint_sum = torch.zeros(
            (
                self.number_of_topics,
                self.number_of_topics,
            ),
            dtype=torch.float64,
        )

        self.depth_joint_sum = (
            defaultdict(
                lambda: torch.zeros(
                    (
                        self.number_of_topics,
                        self.number_of_topics,
                    ),
                    dtype=torch.float64,
                )
            )
        )

        self.depth_edge_counts = (
            defaultdict(int)
        )

        self.total_claims = 0
        self.valid_bow_claims = 0
        self.empty_bow_claims = 0

        self.total_dependency_edges = 0
        self.adjacent_edges_used = 0
        self.non_adjacent_edges_excluded = 0

    def update(
        self,
        theta,
        claim_depth,
        claim_edge_index,
        bow,
    ):
        theta_cpu = (
            theta.detach()
            .double()
            .cpu()
        )

        depth_cpu = (
            claim_depth.detach()
            .long()
            .cpu()
        )

        edge_cpu = (
            claim_edge_index.detach()
            .long()
            .cpu()
        )

        bow_sums = (
            bow.detach()
            .sum(dim=-1)
            .cpu()
        )

        num_claims = int(
            theta_cpu.shape[0]
        )

        if depth_cpu.shape != (
            num_claims,
        ):
            raise ValueError(
                "claim_depth/theta mismatch."
            )

        if bow_sums.shape != (
            num_claims,
        ):
            raise ValueError(
                "BoW/theta mismatch."
            )

        self.total_claims += (
            num_claims
        )

        valid_bow_mask = (
            bow_sums > 1e-10
        )

        self.valid_bow_claims += int(
            valid_bow_mask.sum().item()
        )

        self.empty_bow_claims += int(
            (
                ~valid_bow_mask
            ).sum().item()
        )

        self.topic_mass += (
            theta_cpu.sum(dim=0)
        )

        for depth_value in torch.unique(
            depth_cpu
        ).tolist():
            depth_value = int(
                depth_value
            )

            depth_mask = (
                depth_cpu
                == depth_value
            )

            self.depth_topic_mass[
                depth_value
            ] += theta_cpu[
                depth_mask
            ].sum(dim=0)

            self.depth_claim_counts[
                depth_value
            ] += int(
                depth_mask.sum().item()
            )

        number_of_edges = int(
            edge_cpu.shape[1]
        )

        self.total_dependency_edges += (
            number_of_edges
        )

        if number_of_edges == 0:
            return

        parent_index = edge_cpu[0]
        child_index = edge_cpu[1]

        depth_difference = (
            depth_cpu[child_index]
            - depth_cpu[parent_index]
        )

        if torch.any(
            depth_difference <= 0
        ):
            raise ValueError(
                "A dependency edge does not increase depth."
            )

        adjacent_mask = (
            depth_difference == 1
        )

        adjacent_count = int(
            adjacent_mask.sum().item()
        )

        excluded_count = (
            number_of_edges
            - adjacent_count
        )

        self.adjacent_edges_used += (
            adjacent_count
        )

        self.non_adjacent_edges_excluded += (
            excluded_count
        )

        if adjacent_count == 0:
            return

        adjacent_parent = (
            parent_index[
                adjacent_mask
            ]
        )

        adjacent_child = (
            child_index[
                adjacent_mask
            ]
        )

        parent_theta = theta_cpu[
            adjacent_parent
        ]

        child_theta = theta_cpu[
            adjacent_child
        ]

        # Sum_e theta_parent(e) outer theta_child(e)
        joint_sum = torch.einsum(
            "ei,ej->ij",
            parent_theta,
            child_theta,
        )

        self.global_joint_sum += (
            joint_sum
        )

        parent_depth = depth_cpu[
            adjacent_parent
        ]

        for depth_value in torch.unique(
            parent_depth
        ).tolist():
            depth_value = int(
                depth_value
            )

            depth_mask = (
                parent_depth
                == depth_value
            )

            depth_parent_theta = (
                parent_theta[
                    depth_mask
                ]
            )

            depth_child_theta = (
                child_theta[
                    depth_mask
                ]
            )

            self.depth_joint_sum[
                depth_value
            ] += torch.einsum(
                "ei,ej->ij",
                depth_parent_theta,
                depth_child_theta,
            )

            self.depth_edge_counts[
                depth_value
            ] += int(
                depth_mask.sum().item()
            )

    def summary(self):
        edge_coverage = (
            self.adjacent_edges_used
            / max(
                self.total_dependency_edges,
                1,
            )
        )

        empty_bow_ratio = (
            self.empty_bow_claims
            / max(
                self.total_claims,
                1,
            )
        )

        return {
            "total_claims": (
                self.total_claims
            ),
            "valid_bow_claims": (
                self.valid_bow_claims
            ),
            "empty_bow_claims": (
                self.empty_bow_claims
            ),
            "empty_bow_ratio": (
                empty_bow_ratio
            ),
            "total_dependency_edges": (
                self.total_dependency_edges
            ),
            "adjacent_edges_used": (
                self.adjacent_edges_used
            ),
            "non_adjacent_edges_excluded": (
                self.non_adjacent_edges_excluded
            ),
            "adjacent_edge_coverage": (
                edge_coverage
            ),
        }


# ============================================================
# 10. Inference-shard writer
# ============================================================

def flush_inference_buffer(
    split,
    shard_index,
    buffer,
):
    if not buffer["theta"]:
        return None

    theta = torch.cat(
        buffer["theta"],
        dim=0,
    )

    claim_depth = torch.cat(
        buffer["claim_depth"],
        dim=0,
    )

    bow_valid = torch.cat(
        buffer["bow_valid"],
        dim=0,
    )

    dominant_topic = torch.cat(
        buffer["dominant_topic"],
        dim=0,
    )

    dominant_probability = torch.cat(
        buffer["dominant_probability"],
        dim=0,
    )

    claim_keys = list(
        buffer["claim_keys"]
    )

    if theta.shape[0] != len(
        claim_keys
    ):
        raise RuntimeError(
            "Inference shard claim-key count mismatch."
        )

    shard_filename = (
        f"{split}_theta_"
        f"{shard_index:05d}.pt"
    )

    shard_path = os.path.join(
        INFERENCE_SHARD_DIR,
        shard_filename,
    )

    payload = {
        "split": split,
        "feature_run_name": (
            FEATURE_RUN_NAME
        ),
        "best_epoch": (
            BEST_EPOCH_SECTION7
        ),
        "theta": theta,
        "claim_depth": (
            claim_depth
        ),
        "bow_valid": (
            bow_valid
        ),
        "dominant_topic": (
            dominant_topic
        ),
        "dominant_probability": (
            dominant_probability
        ),
        "claim_keys": (
            claim_keys
        ),
        "number_of_claims": int(
            theta.shape[0]
        ),
        "number_of_topics": int(
            theta.shape[1]
        ),
    }

    atomic_torch_save_section7(
        payload,
        shard_path,
    )

    return {
        "file": shard_filename,
        "path": shard_path,
        "number_of_claims": int(
            theta.shape[0]
        ),
    }


# ============================================================
# 11. Deterministic split inference
# ============================================================

@torch.no_grad()
def infer_split(
    split_name,
    data_loader,
    transition_accumulator=None,
):
    depth_ot_model.eval()
    topic_anchor.eval()

    shard_records = []

    buffer = {
        "theta": [],
        "claim_depth": [],
        "bow_valid": [],
        "dominant_topic": [],
        "dominant_probability": [],
        "claim_keys": [],
    }

    shard_index = 0
    total_patents = 0
    total_claims = 0
    total_empty_bow = 0

    progress = tqdm(
        data_loader,
        desc=(
            f"Section 7 inference: "
            f"{split_name}"
        ),
    )

    for batch_index, cpu_batch in enumerate(
        progress
    ):
        batch = move_depth_ot_batch(
            cpu_batch,
            device=DEVICE,
        )

        theta, _ = (
            run_deterministic_forward(
                batch
            )
        )

        dominant_probability, dominant_topic = (
            theta.max(dim=-1)
        )

        bow_sums = batch[
            "bow"
        ].sum(dim=-1)

        bow_valid = (
            bow_sums > 1e-10
        )

        number_of_empty = int(
            (
                ~bow_valid
            ).sum().item()
        )

        total_patents += int(
            cpu_batch["num_patents"]
        )

        total_claims += int(
            cpu_batch["num_claims"]
        )

        total_empty_bow += (
            number_of_empty
        )

        buffer["theta"].append(
            theta.detach()
            .float()
            .cpu()
        )

        buffer[
            "claim_depth"
        ].append(
            batch["claim_depth"]
            .detach()
            .long()
            .cpu()
        )

        buffer[
            "bow_valid"
        ].append(
            bow_valid.detach()
            .bool()
            .cpu()
        )

        buffer[
            "dominant_topic"
        ].append(
            dominant_topic.detach()
            .long()
            .cpu()
        )

        buffer[
            "dominant_probability"
        ].append(
            dominant_probability.detach()
            .float()
            .cpu()
        )

        buffer[
            "claim_keys"
        ].extend(
            [
                (
                    str(patent_id),
                    int(claim_id),
                )
                for patent_id, claim_id
                in cpu_batch[
                    "claim_keys"
                ]
            ]
        )

        # Global hierarchy is accumulated from train only.
        if (
            transition_accumulator
            is not None
        ):
            transition_accumulator.update(
                theta=theta,
                claim_depth=batch[
                    "claim_depth"
                ],
                claim_edge_index=batch[
                    "claim_edge_index"
                ],
                bow=batch["bow"],
            )

        should_flush = (
            (batch_index + 1)
            % INFERENCE_BATCHES_PER_SHARD
            == 0
        )

        if should_flush:
            shard_record = (
                flush_inference_buffer(
                    split=split_name,
                    shard_index=shard_index,
                    buffer=buffer,
                )
            )

            if shard_record is not None:
                shard_records.append(
                    shard_record
                )

                shard_index += 1

            buffer = {
                "theta": [],
                "claim_depth": [],
                "bow_valid": [],
                "dominant_topic": [],
                "dominant_probability": [],
                "claim_keys": [],
            }

        progress.set_postfix(
            {
                "claims": (
                    f"{total_claims:,}"
                ),
                "empty_bow": (
                    f"{total_empty_bow:,}"
                ),
            }
        )

        del batch
        del theta

    final_shard_record = (
        flush_inference_buffer(
            split=split_name,
            shard_index=shard_index,
            buffer=buffer,
        )
    )

    if final_shard_record is not None:
        shard_records.append(
            final_shard_record
        )

    if total_claims == 0:
        raise RuntimeError(
            f"No claims inferred for {split_name}."
        )

    return {
        "split": split_name,
        "number_of_patents": (
            total_patents
        ),
        "number_of_claims": (
            total_claims
        ),
        "empty_bow_claims": (
            total_empty_bow
        ),
        "empty_bow_ratio": (
            total_empty_bow
            / total_claims
        ),
        "number_of_shards": len(
            shard_records
        ),
        "shards": shard_records,
    }


# ============================================================
# 12. Run inference on train/dev/test
# ============================================================

train_transition_accumulator = (
    GlobalTransitionAccumulator(
        number_of_topics=NUM_TOPICS
    )
)

inference_split_summaries = {}

inference_split_summaries["train"] = (
    infer_split(
        split_name="train",
        data_loader=(
            section7_train_loader
        ),
        transition_accumulator=(
            train_transition_accumulator
        ),
    )
)

inference_split_summaries["dev"] = (
    infer_split(
        split_name="dev",
        data_loader=(
            section7_dev_loader
        ),
        transition_accumulator=None,
    )
)

inference_split_summaries["test"] = (
    infer_split(
        split_name="test",
        data_loader=(
            section7_test_loader
        ),
        transition_accumulator=None,
    )
)

print("\n=== INFERENCE SUMMARY ===")

for split_name, summary in (
    inference_split_summaries.items()
):
    print(
        f"{split_name:5s}: "
        f"patents={summary['number_of_patents']:,}, "
        f"claims={summary['number_of_claims']:,}, "
        f"empty_bow={summary['empty_bow_claims']:,} "
        f"({summary['empty_bow_ratio']:.4%}), "
        f"output_shards={summary['number_of_shards']:,}"
    )


# ============================================================
# 13. Anchor coordinates
# ============================================================

with torch.no_grad():
    anchor_coordinates_tensor = (
        topic_anchor.coordinates()
        .detach()
        .float()
        .cpu()
    )

    anchor_cost_matrix = (
        topic_anchor.cost_matrix()
        .detach()
        .float()
        .cpu()
    )

if anchor_coordinates_tensor.shape != (
    NUM_TOPICS,
):
    raise ValueError(
        "Anchor-coordinate shape mismatch."
    )

anchor_differences = (
    anchor_coordinates_tensor[1:]
    - anchor_coordinates_tensor[:-1]
)

if torch.any(
    anchor_differences <= 0
):
    raise ValueError(
        "Anchor coordinates are not strictly increasing."
    )

if anchor_cost_matrix.shape != (
    NUM_TOPICS,
    NUM_TOPICS,
):
    raise ValueError(
        "Anchor cost-matrix shape mismatch."
    )

if not torch.isfinite(
    anchor_cost_matrix
).all():
    raise FloatingPointError(
        "Anchor cost matrix contains non-finite values."
    )

if torch.any(
    anchor_cost_matrix < 0
):
    raise ValueError(
        "Anchor cost matrix contains negative values."
    )


# ============================================================
# 14. Normalize empirical transition statistics
# ============================================================

transition_summary = (
    train_transition_accumulator
    .summary()
)

number_of_adjacent_edges = (
    train_transition_accumulator
    .adjacent_edges_used
)

if number_of_adjacent_edges <= 0:
    raise RuntimeError(
        "No adjacent-depth train edges are available "
        "for global hierarchy extraction."
    )

global_joint = (
    train_transition_accumulator
    .global_joint_sum
    / float(
        number_of_adjacent_edges
    )
)

if not torch.isfinite(
    global_joint
).all():
    raise FloatingPointError(
        "Global empirical joint contains "
        "non-finite values."
    )

if torch.any(
    global_joint < 0
):
    raise ValueError(
        "Global empirical joint contains "
        "negative values."
    )

if not torch.allclose(
    global_joint.sum(),
    torch.tensor(
        1.0,
        dtype=global_joint.dtype,
    ),
    atol=1e-6,
    rtol=1e-6,
):
    raise ValueError(
        "Global empirical joint does not sum to one: "
        f"{global_joint.sum().item():.8f}."
    )

global_parent_marginal = (
    global_joint.sum(dim=1)
)

global_child_marginal = (
    global_joint.sum(dim=0)
)

topic_usage = (
    train_transition_accumulator
    .topic_mass
    / max(
        train_transition_accumulator
        .total_claims,
        1,
    )
)

topic_usage = (
    topic_usage
    / topic_usage.sum().clamp_min(
        1e-12
    )
)


# ============================================================
# 15. Construct anchor-consistent global hierarchy DAG
# ============================================================

# Only transitions from a lower anchor coordinate to a higher
# anchor coordinate are eligible.
parent_coordinate = (
    anchor_coordinates_tensor
    .double()
    .view(-1, 1)
)

child_coordinate = (
    anchor_coordinates_tensor
    .double()
    .view(1, -1)
)

directional_mask = (
    child_coordinate
    > parent_coordinate
)

directional_joint = (
    global_joint
    * directional_mask.to(
        dtype=global_joint.dtype
    )
)

directional_row_mass = (
    directional_joint.sum(
        dim=1,
        keepdim=True,
    )
)

directional_conditional = torch.where(
    directional_row_mass > 0,
    directional_joint
    / directional_row_mass.clamp_min(
        1e-12
    ),
    torch.zeros_like(
        directional_joint
    ),
)

global_hierarchy_edges = []

for parent_topic in range(
    NUM_TOPICS
):
    candidate_scores = (
        directional_conditional[
            parent_topic
        ]
    )

    valid_candidates = torch.nonzero(
        candidate_scores
        >= GLOBAL_EDGE_MIN_CONDITIONAL,
        as_tuple=False,
    ).flatten()

    valid_candidates = [
        int(child_topic)
        for child_topic
        in valid_candidates.tolist()
        if int(child_topic)
        != parent_topic
        and directional_mask[
            parent_topic,
            int(child_topic),
        ].item()
    ]

    valid_candidates.sort(
        key=lambda child_topic: (
            float(
                candidate_scores[
                    child_topic
                ].item()
            )
        ),
        reverse=True,
    )

    selected_children = (
        valid_candidates[
            :GLOBAL_HIERARCHY_TOP_K
        ]
    )

    for rank, child_topic in enumerate(
        selected_children,
        start=1,
    ):
        parent_words = [
            item["word"]
            for item in
            topic_word_summaries[
                parent_topic
            ]["top_words"][:5]
        ]

        child_words = [
            item["word"]
            for item in
            topic_word_summaries[
                child_topic
            ]["top_words"][:5]
        ]

        global_hierarchy_edges.append(
            {
                "parent_topic_id": (
                    parent_topic
                ),
                "parent_topic_number": (
                    parent_topic + 1
                ),
                "child_topic_id": (
                    child_topic
                ),
                "child_topic_number": (
                    child_topic + 1
                ),
                "rank_for_parent": rank,
                "parent_anchor_coordinate": float(
                    anchor_coordinates_tensor[
                        parent_topic
                    ].item()
                ),
                "child_anchor_coordinate": float(
                    anchor_coordinates_tensor[
                        child_topic
                    ].item()
                ),
                "anchor_gap": float(
                    (
                        anchor_coordinates_tensor[
                            child_topic
                        ]
                        - anchor_coordinates_tensor[
                            parent_topic
                        ]
                    ).item()
                ),
                "joint_probability": float(
                    global_joint[
                        parent_topic,
                        child_topic,
                    ].item()
                ),
                "directional_joint_probability": float(
                    directional_joint[
                        parent_topic,
                        child_topic,
                    ].item()
                ),
                "conditional_probability": float(
                    directional_conditional[
                        parent_topic,
                        child_topic,
                    ].item()
                ),
                "anchor_cost": float(
                    anchor_cost_matrix[
                        parent_topic,
                        child_topic,
                    ].item()
                ),
                "parent_top_words": (
                    parent_words
                ),
                "child_top_words": (
                    child_words
                ),
            }
        )


# ============================================================
# 16. Verify that the extracted graph is acyclic
# ============================================================

for edge in global_hierarchy_edges:
    parent_topic = int(
        edge[
            "parent_topic_id"
        ]
    )

    child_topic = int(
        edge[
            "child_topic_id"
        ]
    )

    if not (
        anchor_coordinates_tensor[
            child_topic
        ]
        > anchor_coordinates_tensor[
            parent_topic
        ]
    ):
        raise RuntimeError(
            "A global hierarchy edge violates "
            "the anchor ordering."
        )

# Strictly increasing anchor coordinates guarantee no cycle,
# but verify by topological sorting as an additional check.
global_children = {
    topic_id: []
    for topic_id in range(
        NUM_TOPICS
    )
}

global_indegree = {
    topic_id: 0
    for topic_id in range(
        NUM_TOPICS
    )
}

for edge in global_hierarchy_edges:
    parent_topic = int(
        edge["parent_topic_id"]
    )

    child_topic = int(
        edge["child_topic_id"]
    )

    global_children[
        parent_topic
    ].append(
        child_topic
    )

    global_indegree[
        child_topic
    ] += 1

topological_queue = [
    topic_id
    for topic_id in range(
        NUM_TOPICS
    )
    if global_indegree[
        topic_id
    ] == 0
]

topological_order = []
queue_position = 0

while queue_position < len(
    topological_queue
):
    topic_id = (
        topological_queue[
            queue_position
        ]
    )

    queue_position += 1

    topological_order.append(
        topic_id
    )

    for child_topic in (
        global_children[
            topic_id
        ]
    ):
        global_indegree[
            child_topic
        ] -= 1

        if (
            global_indegree[
                child_topic
            ]
            == 0
        ):
            topological_queue.append(
                child_topic
            )

if len(topological_order) != NUM_TOPICS:
    raise RuntimeError(
        "A cycle was detected in the extracted "
        "global topic hierarchy."
    )


# ============================================================
# 17. Per-depth statistics
# ============================================================

per_depth_statistics = []

all_depth_values = sorted(
    train_transition_accumulator
    .depth_claim_counts.keys()
)

for depth_value in all_depth_values:
    claim_count = int(
        train_transition_accumulator
        .depth_claim_counts[
            depth_value
        ]
    )

    mean_topic_distribution = (
        train_transition_accumulator
        .depth_topic_mass[
            depth_value
        ]
        / max(
            claim_count,
            1,
        )
    )

    mean_topic_distribution = (
        mean_topic_distribution
        / mean_topic_distribution
        .sum()
        .clamp_min(1e-12)
    )

    dominant_depth_topics = (
        torch.argsort(
            mean_topic_distribution,
            descending=True,
        )[
            :min(
                5,
                NUM_TOPICS,
            )
        ]
    )

    per_depth_statistics.append(
        {
            "depth": int(
                depth_value
            ),
            "number_of_claims": (
                claim_count
            ),
            "mean_topic_distribution": [
                float(value)
                for value in
                mean_topic_distribution.tolist()
            ],
            "dominant_topics": [
                {
                    "topic_id": int(
                        topic_id
                    ),
                    "topic_number": int(
                        topic_id
                    ) + 1,
                    "probability": float(
                        mean_topic_distribution[
                            topic_id
                        ].item()
                    ),
                    "top_words": [
                        item["word"]
                        for item in
                        topic_word_summaries[
                            int(topic_id)
                        ]["top_words"][:5]
                    ],
                }
                for topic_id in
                dominant_depth_topics.tolist()
            ],
        }
    )

per_depth_transition_statistics = []

for parent_depth in sorted(
    train_transition_accumulator
    .depth_edge_counts.keys()
):
    edge_count = int(
        train_transition_accumulator
        .depth_edge_counts[
            parent_depth
        ]
    )

    depth_joint = (
        train_transition_accumulator
        .depth_joint_sum[
            parent_depth
        ]
        / max(
            edge_count,
            1,
        )
    )

    per_depth_transition_statistics.append(
        {
            "parent_depth": int(
                parent_depth
            ),
            "child_depth": int(
                parent_depth + 1
            ),
            "number_of_edges": (
                edge_count
            ),
            "joint_sum": float(
                depth_joint.sum().item()
            ),
        }
    )


# ============================================================
# 18. Topic node records
# ============================================================

incoming_edge_count = defaultdict(
    int
)

outgoing_edge_count = defaultdict(
    int
)

for edge in global_hierarchy_edges:
    outgoing_edge_count[
        int(edge["parent_topic_id"])
    ] += 1

    incoming_edge_count[
        int(edge["child_topic_id"])
    ] += 1

global_topic_nodes = []

for topic_id in range(
    NUM_TOPICS
):
    incoming = int(
        incoming_edge_count[
            topic_id
        ]
    )

    outgoing = int(
        outgoing_edge_count[
            topic_id
        ]
    )

    if incoming == 0 and outgoing > 0:
        node_role = "root"
    elif incoming > 0 and outgoing == 0:
        node_role = "leaf"
    elif incoming == 0 and outgoing == 0:
        node_role = "isolated"
    else:
        node_role = "interior"

    global_topic_nodes.append(
        {
            "topic_id": topic_id,
            "topic_number": (
                topic_id + 1
            ),
            "anchor_coordinate": float(
                anchor_coordinates_tensor[
                    topic_id
                ].item()
            ),
            "train_topic_usage": float(
                topic_usage[
                    topic_id
                ].item()
            ),
            "parent_edge_marginal": float(
                global_parent_marginal[
                    topic_id
                ].item()
            ),
            "child_edge_marginal": float(
                global_child_marginal[
                    topic_id
                ].item()
            ),
            "incoming_edges": incoming,
            "outgoing_edges": outgoing,
            "role": node_role,
            "top_words": (
                topic_word_summaries[
                    topic_id
                ]["top_words"]
            ),
        }
    )


# ============================================================
# 19. Save global matrices
# ============================================================

global_matrix_path = os.path.join(
    SECTION7_RESULT_DIR,
    "global_transition_matrices.pt",
)

atomic_torch_save_section7(
    {
        "empirical_joint": (
            global_joint.float()
        ),
        "parent_marginal": (
            global_parent_marginal.float()
        ),
        "child_marginal": (
            global_child_marginal.float()
        ),
        "directional_joint": (
            directional_joint.float()
        ),
        "directional_conditional": (
            directional_conditional.float()
        ),
        "topic_usage": (
            topic_usage.float()
        ),
        "anchor_coordinates": (
            anchor_coordinates_tensor
        ),
        "anchor_cost_matrix": (
            anchor_cost_matrix
        ),
        "number_of_adjacent_edges": (
            number_of_adjacent_edges
        ),
        "feature_run_name": (
            FEATURE_RUN_NAME
        ),
        "best_epoch": (
            BEST_EPOCH_SECTION7
        ),
    },
    global_matrix_path,
)


# ============================================================
# 20. Save global hierarchy
# ============================================================

global_hierarchy = {
    "method": (
        "anchor-ordered empirical topic-transition DAG"
    ),
    "description": (
        "The global hierarchy is extracted only from train-split "
        "adjacent-depth dependency edges. Empirical topic-transition "
        "evidence is aggregated using outer products of deterministic "
        "parent and child topic mixtures. Edges inconsistent with the "
        "strictly increasing learned anchor are removed."
    ),
    "run_name": (
        globals().get(
            "RUN_NAME",
            None,
        )
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "best_epoch": (
        BEST_EPOCH_SECTION7
    ),
    "best_epoch_one_based": (
        BEST_EPOCH_SECTION7 + 1
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "number_of_topics": (
        NUM_TOPICS
    ),
    "number_of_nodes": len(
        global_topic_nodes
    ),
    "number_of_edges": len(
        global_hierarchy_edges
    ),
    "is_directed": True,
    "is_acyclic": True,
    "uses_train_split_only": True,
    "top_k_per_parent": (
        GLOBAL_HIERARCHY_TOP_K
    ),
    "minimum_conditional_probability": (
        GLOBAL_EDGE_MIN_CONDITIONAL
    ),
    "transition_diagnostics": (
        transition_summary
    ),
    "topological_order_topic_ids": (
        topological_order
    ),
    "nodes": global_topic_nodes,
    "edges": global_hierarchy_edges,
    "per_depth_topic_statistics": (
        per_depth_statistics
    ),
    "per_depth_transition_statistics": (
        per_depth_transition_statistics
    ),
}

global_hierarchy_path = os.path.join(
    SECTION7_RESULT_DIR,
    "global_topic_hierarchy.json",
)

atomic_json_save_section7(
    global_hierarchy,
    global_hierarchy_path,
)


# ============================================================
# 21. Save inference manifest
# ============================================================

inference_manifest = {
    "run_name": (
        globals().get(
            "RUN_NAME",
            None,
        )
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "best_checkpoint": (
        BEST_CHECKPOINT_PATH
    ),
    "best_epoch": (
        BEST_EPOCH_SECTION7
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "device": str(
        DEVICE
    ),
    "deterministic_maximum_difference": (
        maximum_deterministic_difference
    ),
    "number_of_topics": (
        NUM_TOPICS
    ),
    "vocabulary_size": (
        len(VOCAB)
    ),
    "inference_batches_per_shard": (
        INFERENCE_BATCHES_PER_SHARD
    ),
    "splits": (
        inference_split_summaries
    ),
    "train_transition_diagnostics": (
        transition_summary
    ),
    "files": {
        "topic_word_distribution": (
            topic_word_path
        ),
        "transition_matrices": (
            global_matrix_path
        ),
        "global_hierarchy": (
            global_hierarchy_path
        ),
        "claim_theta_shard_directory": (
            INFERENCE_SHARD_DIR
        ),
    },
}

inference_manifest_path = os.path.join(
    SECTION7_RESULT_DIR,
    "inference_manifest.json",
)

atomic_json_save_section7(
    inference_manifest,
    inference_manifest_path,
)


# ============================================================
# 22. Final validation
# ============================================================

if len(
    global_topic_nodes
) != NUM_TOPICS:
    raise RuntimeError(
        "Global topic-node count mismatch."
    )

if not os.path.isfile(
    global_hierarchy_path
):
    raise FileNotFoundError(
        "Global hierarchy file was not saved."
    )

if not os.path.isfile(
    global_matrix_path
):
    raise FileNotFoundError(
        "Global transition matrix file was not saved."
    )

if not os.path.isfile(
    topic_word_path
):
    raise FileNotFoundError(
        "Topic-word distribution file was not saved."
    )

if not os.path.isfile(
    inference_manifest_path
):
    raise FileNotFoundError(
        "Inference manifest was not saved."
    )


# ============================================================
# 23. Final status
# ============================================================

print("\n" + "=" * 72)
print("SECTION 7 COMPLETED SUCCESSFULLY")
print("=" * 72)
print(
    f"Best epoch                 : "
    f"{BEST_EPOCH_SECTION7 + 1}"
)
print(
    f"Topics                     : "
    f"{NUM_TOPICS}"
)
print(
    f"Global hierarchy nodes     : "
    f"{len(global_topic_nodes)}"
)
print(
    f"Global hierarchy edges     : "
    f"{len(global_hierarchy_edges)}"
)
print(
    f"Train dependency edges     : "
    f"{transition_summary['total_dependency_edges']:,}"
)
print(
    f"Adjacent edges used        : "
    f"{transition_summary['adjacent_edges_used']:,}"
)
print(
    f"Non-adjacent edges excluded: "
    f"{transition_summary['non_adjacent_edges_excluded']:,}"
)
print(
    f"Adjacent-edge coverage     : "
    f"{transition_summary['adjacent_edge_coverage']:.4%}"
)
print(
    f"Train empty-BoW ratio      : "
    f"{transition_summary['empty_bow_ratio']:.4%}"
)
print(
    f"Deterministic max diff     : "
    f"{maximum_deterministic_difference:.3e}"
)
print(
    f"Hierarchy JSON             : "
    f"{global_hierarchy_path}"
)
print(
    f"Transition matrices        : "
    f"{global_matrix_path}"
)
print(
    f"Topic-word distribution    : "
    f"{topic_word_path}"
)
print(
    f"Inference manifest         : "
    f"{inference_manifest_path}"
)
print(
    f"Claim-theta shards         : "
    f"{INFERENCE_SHARD_DIR}"
)
print("=" * 72)

if FEATURE_RUN_NAME.startswith(
    "debug"
):
    print(
        "\n[IMPORTANT] This hierarchy was generated from a debug "
        "feature run and must not be reported as a final result."
    )
else:
    print(
        "\n[PASS] Full deterministic inference and train-only "
        "global hierarchy extraction completed."
    )

print(
    "\nThe outputs are saved to Google Drive and are ready for "
    "Section 8: evaluation, visualization, and topic-quality analysis."
)


# ============================================================
# SECTION 8:
# Topic Quality, Hierarchy Evaluation, and Visualization
# COMPLETE SINGLE-CELL VERSION
# ============================================================

import os
import re
import csv
import json
import math
from itertools import combinations
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ============================================================
# 0. Preconditions
# ============================================================

required_section8_globals = [
    "CONFIG",
    "FEATURE_RUN_NAME",
    "VOCAB",
    "WORD_TO_ID",
    "BOW_TOKEN_PATTERN",
    "train_dataset",
    "SECTION7_RESULT_DIR",
]

missing_section8_globals = [
    name
    for name in required_section8_globals
    if name not in globals()
]

if missing_section8_globals:
    raise RuntimeError(
        "Section 8 prerequisites are missing: "
        f"{missing_section8_globals}. "
        "Run Sections 0 through 7 first."
    )

GLOBAL_HIERARCHY_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "global_topic_hierarchy.json",
)

GLOBAL_MATRIX_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "global_transition_matrices.pt",
)

TOPIC_WORD_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "topic_word_distribution.pt",
)

INFERENCE_MANIFEST_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "inference_manifest.json",
)

required_section7_files = [
    GLOBAL_HIERARCHY_PATH,
    GLOBAL_MATRIX_PATH,
    TOPIC_WORD_PATH,
    INFERENCE_MANIFEST_PATH,
]

missing_section7_files = [
    path
    for path in required_section7_files
    if not os.path.isfile(path)
]

if missing_section7_files:
    raise FileNotFoundError(
        "Section 7 output files are missing: "
        f"{missing_section7_files}"
    )


# ============================================================
# 1. Evaluation configuration
# ============================================================

SECTION8_RESULT_DIR = os.path.join(
    SECTION7_RESULT_DIR,
    "section8_evaluation",
)

SECTION8_FIGURE_DIR = os.path.join(
    SECTION8_RESULT_DIR,
    "figures",
)

os.makedirs(
    SECTION8_RESULT_DIR,
    exist_ok=True,
)

os.makedirs(
    SECTION8_FIGURE_DIR,
    exist_ok=True,
)

TOP_WORDS_FOR_DIVERSITY = int(
    getattr(
        CONFIG,
        "evaluation_top_words",
        25,
    )
)

TOP_WORDS_FOR_NPMI = int(
    getattr(
        CONFIG,
        "npmi_top_words",
        10,
    )
)

TOP_WORDS_FOR_EXCLUSIVITY = int(
    getattr(
        CONFIG,
        "exclusivity_top_words",
        20,
    )
)

if TOP_WORDS_FOR_DIVERSITY < 2:
    raise ValueError(
        "TOP_WORDS_FOR_DIVERSITY must be at least 2."
    )

if TOP_WORDS_FOR_NPMI < 2:
    raise ValueError(
        "TOP_WORDS_FOR_NPMI must be at least 2."
    )

if TOP_WORDS_FOR_EXCLUSIVITY < 1:
    raise ValueError(
        "TOP_WORDS_FOR_EXCLUSIVITY must be positive."
    )

TOP_WORDS_FOR_DIVERSITY = min(
    TOP_WORDS_FOR_DIVERSITY,
    len(VOCAB),
)

TOP_WORDS_FOR_NPMI = min(
    TOP_WORDS_FOR_NPMI,
    len(VOCAB),
)

TOP_WORDS_FOR_EXCLUSIVITY = min(
    TOP_WORDS_FOR_EXCLUSIVITY,
    len(VOCAB),
)

print("=== SECTION 8 CONFIGURATION ===")
print(f"Feature run             : {FEATURE_RUN_NAME}")
print(f"Result directory        : {SECTION8_RESULT_DIR}")
print(f"Top words for diversity : {TOP_WORDS_FOR_DIVERSITY}")
print(f"Top words for NPMI      : {TOP_WORDS_FOR_NPMI}")
print(f"Top words exclusivity   : {TOP_WORDS_FOR_EXCLUSIVITY}")


# ============================================================
# 2. Serialization helpers
# ============================================================

def atomic_json_save_section8(
    data,
    final_path,
):
    os.makedirs(
        os.path.dirname(final_path),
        exist_ok=True,
    )

    temporary_path = final_path + ".tmp"

    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    with open(
        temporary_path,
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
        temporary_path,
        final_path,
    )


def finite_or_none(value):
    value = float(value)

    if math.isfinite(value):
        return value

    return None


# ============================================================
# 3. Load Section 7 outputs
# ============================================================

with open(
    GLOBAL_HIERARCHY_PATH,
    "r",
    encoding="utf-8",
) as file:
    global_hierarchy = json.load(file)

with open(
    INFERENCE_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    inference_manifest = json.load(file)

global_matrices = torch.load(
    GLOBAL_MATRIX_PATH,
    map_location="cpu",
    weights_only=False,
)

topic_word_payload = torch.load(
    TOPIC_WORD_PATH,
    map_location="cpu",
    weights_only=False,
)

beta = (
    topic_word_payload["beta"]
    .detach()
    .float()
    .cpu()
)

empirical_joint = (
    global_matrices[
        "empirical_joint"
    ]
    .detach()
    .double()
    .cpu()
)

directional_joint = (
    global_matrices[
        "directional_joint"
    ]
    .detach()
    .double()
    .cpu()
)

directional_conditional = (
    global_matrices[
        "directional_conditional"
    ]
    .detach()
    .double()
    .cpu()
)

anchor_coordinates = (
    global_matrices[
        "anchor_coordinates"
    ]
    .detach()
    .float()
    .cpu()
)

topic_usage = (
    global_matrices[
        "topic_usage"
    ]
    .detach()
    .float()
    .cpu()
)

NUM_TOPICS = int(
    beta.shape[0]
)

VOCAB_SIZE = int(
    beta.shape[1]
)

if VOCAB_SIZE != len(VOCAB):
    raise ValueError(
        "Beta/vocabulary size mismatch."
    )

if anchor_coordinates.shape != (
    NUM_TOPICS,
):
    raise ValueError(
        "Anchor-coordinate shape mismatch."
    )

if empirical_joint.shape != (
    NUM_TOPICS,
    NUM_TOPICS,
):
    raise ValueError(
        "Empirical transition matrix shape mismatch."
    )

if not torch.isfinite(beta).all():
    raise FloatingPointError(
        "Beta contains non-finite values."
    )

if torch.any(beta < 0):
    raise ValueError(
        "Beta contains negative values."
    )

if not torch.allclose(
    beta.sum(dim=-1),
    torch.ones(NUM_TOPICS),
    atol=1e-5,
    rtol=1e-5,
):
    raise ValueError(
        "Beta rows do not sum to one."
    )

print("\n=== LOADED MODEL OUTPUTS ===")
print(f"Topics             : {NUM_TOPICS}")
print(f"Vocabulary size    : {VOCAB_SIZE}")
print(f"Hierarchy nodes    : {len(global_hierarchy['nodes'])}")
print(f"Hierarchy edges    : {len(global_hierarchy['edges'])}")


# ============================================================
# 4. Topic Diversity
# ============================================================

diversity_top_indices = torch.topk(
    beta,
    k=TOP_WORDS_FOR_DIVERSITY,
    dim=-1,
).indices

all_diversity_indices = (
    diversity_top_indices
    .reshape(-1)
    .tolist()
)

number_of_unique_top_words = len(
    set(
        int(index)
        for index in all_diversity_indices
    )
)

topic_diversity = (
    number_of_unique_top_words
    / (
        NUM_TOPICS
        * TOP_WORDS_FOR_DIVERSITY
    )
)

if not (
    0.0 <= topic_diversity <= 1.0
):
    raise RuntimeError(
        "Topic diversity is outside [0, 1]."
    )


# ============================================================
# 5. Topic entropy
# ============================================================

beta_safe = beta.clamp_min(
    1e-12
)

topic_entropy = -(
    beta_safe
    * beta_safe.log()
).sum(dim=-1)

normalized_topic_entropy = (
    topic_entropy
    / math.log(
        max(VOCAB_SIZE, 2)
    )
)

mean_normalized_topic_entropy = float(
    normalized_topic_entropy
    .mean()
    .item()
)


# ============================================================
# 6. Topic exclusivity
# ============================================================

word_probability_across_topics = (
    beta.sum(
        dim=0,
        keepdim=True,
    )
    .clamp_min(1e-12)
)

topic_word_exclusivity = (
    beta
    / word_probability_across_topics
)

exclusivity_top_indices = torch.topk(
    beta,
    k=TOP_WORDS_FOR_EXCLUSIVITY,
    dim=-1,
).indices

per_topic_exclusivity = []

for topic_id in range(
    NUM_TOPICS
):
    indices = exclusivity_top_indices[
        topic_id
    ]

    score = (
        topic_word_exclusivity[
            topic_id,
            indices,
        ]
        .mean()
        .item()
    )

    per_topic_exclusivity.append(
        float(score)
    )

mean_topic_exclusivity = float(
    np.mean(
        per_topic_exclusivity
    )
)


# ============================================================
# 7. Topic redundancy using cosine similarity
# ============================================================

normalized_beta = F.normalize(
    beta,
    p=2,
    dim=-1,
)

topic_cosine_similarity = (
    normalized_beta
    @ normalized_beta.T
)

off_diagonal_mask = (
    ~torch.eye(
        NUM_TOPICS,
        dtype=torch.bool,
    )
)

if NUM_TOPICS > 1:
    off_diagonal_similarity = (
        topic_cosine_similarity[
            off_diagonal_mask
        ]
    )

    mean_topic_cosine_similarity = float(
        off_diagonal_similarity
        .mean()
        .item()
    )

    maximum_topic_cosine_similarity = float(
        off_diagonal_similarity
        .max()
        .item()
    )
else:
    mean_topic_cosine_similarity = 0.0
    maximum_topic_cosine_similarity = 0.0

per_topic_maximum_similarity = []

for topic_id in range(
    NUM_TOPICS
):
    if NUM_TOPICS == 1:
        maximum_similarity = 0.0
    else:
        row = (
            topic_cosine_similarity[
                topic_id
            ].clone()
        )

        row[topic_id] = -1.0

        maximum_similarity = float(
            row.max().item()
        )

    per_topic_maximum_similarity.append(
        maximum_similarity
    )


# ============================================================
# 8. Prepare NPMI word pairs
# ============================================================

npmi_top_indices = torch.topk(
    beta,
    k=TOP_WORDS_FOR_NPMI,
    dim=-1,
).indices

target_word_ids = set()
target_word_pairs = set()
topic_npmi_pairs = []

for topic_id in range(
    NUM_TOPICS
):
    word_ids = [
        int(index)
        for index in
        npmi_top_indices[
            topic_id
        ].tolist()
    ]

    target_word_ids.update(
        word_ids
    )

    current_topic_pairs = []

    for first_word, second_word in combinations(
        word_ids,
        2,
    ):
        pair = tuple(
            sorted(
                (
                    int(first_word),
                    int(second_word),
                )
            )
        )

        target_word_pairs.add(
            pair
        )

        current_topic_pairs.append(
            pair
        )

    topic_npmi_pairs.append(
        current_topic_pairs
    )

document_frequency = {
    word_id: 0
    for word_id in target_word_ids
}

pair_document_frequency = {
    pair: 0
    for pair in target_word_pairs
}


# ============================================================
# 9. Train-claim document-frequency statistics
# ============================================================

number_of_documents = 0
number_of_empty_vocabulary_documents = 0

print("\n=== TRAIN CORPUS NPMI STATISTICS ===")

for record in tqdm(
    train_dataset.records,
    desc="Counting claim co-occurrences",
):
    claims = record.get(
        "claims",
        {}
    )

    if not isinstance(claims, dict):
        raise TypeError(
            "record['claims'] must be a dictionary."
        )

    for claim_text in claims.values():
        number_of_documents += 1

        if claim_text is None:
            claim_text = ""

        if not isinstance(
            claim_text,
            str,
        ):
            claim_text = str(
                claim_text
            )

        tokens = BOW_TOKEN_PATTERN.findall(
            claim_text.lower()
        )

        present_word_ids = {
            int(WORD_TO_ID[token])
            for token in tokens
            if token in WORD_TO_ID
            and int(WORD_TO_ID[token])
            in target_word_ids
        }

        if not present_word_ids:
            number_of_empty_vocabulary_documents += 1
            continue

        for word_id in present_word_ids:
            document_frequency[
                word_id
            ] += 1

        for first_word, second_word in combinations(
            sorted(present_word_ids),
            2,
        ):
            pair = (
                first_word,
                second_word,
            )

            if pair in pair_document_frequency:
                pair_document_frequency[
                    pair
                ] += 1

if number_of_documents <= 0:
    raise RuntimeError(
        "No train claim documents were found."
    )

print(
    f"Train claim documents : "
    f"{number_of_documents:,}"
)
print(
    f"Empty target-vocab docs: "
    f"{number_of_empty_vocabulary_documents:,} "
    f"({number_of_empty_vocabulary_documents / number_of_documents:.4%})"
)


# ============================================================
# 10. NPMI coherence
# ============================================================

def calculate_npmi(
    first_word,
    second_word,
):
    first_frequency = (
        document_frequency.get(
            first_word,
            0,
        )
    )

    second_frequency = (
        document_frequency.get(
            second_word,
            0,
        )
    )

    pair = tuple(
        sorted(
            (
                first_word,
                second_word,
            )
        )
    )

    pair_frequency = (
        pair_document_frequency.get(
            pair,
            0,
        )
    )

    if (
        first_frequency <= 0
        or second_frequency <= 0
        or pair_frequency <= 0
    ):
        # No observed co-occurrence indicates the lowest
        # document-level NPMI coherence.
        return -1.0

    probability_first = (
        first_frequency
        / number_of_documents
    )

    probability_second = (
        second_frequency
        / number_of_documents
    )

    probability_pair = (
        pair_frequency
        / number_of_documents
    )

    pmi = math.log(
        probability_pair
        / (
            probability_first
            * probability_second
        )
    )

    denominator = -math.log(
        probability_pair
    )

    if denominator <= 0:
        return 0.0

    npmi = (
        pmi
        / denominator
    )

    return float(
        max(
            -1.0,
            min(
                1.0,
                npmi,
            ),
        )
    )


per_topic_npmi = []

for topic_id in range(
    NUM_TOPICS
):
    pair_scores = [
        calculate_npmi(
            first_word=pair[0],
            second_word=pair[1],
        )
        for pair in
        topic_npmi_pairs[
            topic_id
        ]
    ]

    if pair_scores:
        topic_score = float(
            np.mean(pair_scores)
        )
    else:
        topic_score = 0.0

    per_topic_npmi.append(
        topic_score
    )

mean_topic_npmi = float(
    np.mean(
        per_topic_npmi
    )
)

median_topic_npmi = float(
    np.median(
        per_topic_npmi
    )
)


# ============================================================
# 11. Hierarchy directional evaluation
# ============================================================

total_empirical_mass = float(
    empirical_joint.sum().item()
)

directional_empirical_mass = float(
    directional_joint.sum().item()
)

if total_empirical_mass <= 0:
    raise RuntimeError(
        "Empirical transition joint has zero mass."
    )

directional_mass_ratio = (
    directional_empirical_mass
    / total_empirical_mass
)

hierarchy_edges = (
    global_hierarchy["edges"]
)

hierarchy_nodes = (
    global_hierarchy["nodes"]
)

edge_conditional_scores = [
    float(
        edge[
            "conditional_probability"
        ]
    )
    for edge in hierarchy_edges
]

edge_joint_scores = [
    float(
        edge[
            "joint_probability"
        ]
    )
    for edge in hierarchy_edges
]

if edge_conditional_scores:
    mean_hierarchy_edge_conditional = float(
        np.mean(
            edge_conditional_scores
        )
    )

    minimum_hierarchy_edge_conditional = float(
        np.min(
            edge_conditional_scores
        )
    )
else:
    mean_hierarchy_edge_conditional = 0.0
    minimum_hierarchy_edge_conditional = 0.0

selected_edge_joint_mass = float(
    sum(
        edge_joint_scores
    )
)

selected_edge_mass_ratio = (
    selected_edge_joint_mass
    / total_empirical_mass
)

directional_violations = 0

for edge in hierarchy_edges:
    parent_topic = int(
        edge[
            "parent_topic_id"
        ]
    )

    child_topic = int(
        edge[
            "child_topic_id"
        ]
    )

    if not (
        anchor_coordinates[
            child_topic
        ]
        > anchor_coordinates[
            parent_topic
        ]
    ):
        directional_violations += 1

if directional_violations > 0:
    raise RuntimeError(
        "The extracted hierarchy contains "
        "anchor-direction violations."
    )


# ============================================================
# 12. Depth-anchor alignment
# ============================================================

per_depth_statistics = (
    global_hierarchy.get(
        "per_depth_topic_statistics",
        [],
    )
)

depth_values = []
expected_anchor_values = []

for depth_record in (
    per_depth_statistics
):
    depth_value = float(
        depth_record["depth"]
    )

    mean_theta = torch.tensor(
        depth_record[
            "mean_topic_distribution"
        ],
        dtype=torch.float32,
    )

    if mean_theta.shape != (
        NUM_TOPICS,
    ):
        raise ValueError(
            "Per-depth topic-distribution shape mismatch."
        )

    expected_anchor = float(
        torch.dot(
            mean_theta,
            anchor_coordinates,
        ).item()
    )

    depth_values.append(
        depth_value
    )

    expected_anchor_values.append(
        expected_anchor
    )


def pearson_correlation(
    first_values,
    second_values,
):
    first = np.asarray(
        first_values,
        dtype=np.float64,
    )

    second = np.asarray(
        second_values,
        dtype=np.float64,
    )

    if (
        first.size < 2
        or second.size < 2
    ):
        return float("nan")

    first_centered = (
        first - first.mean()
    )

    second_centered = (
        second - second.mean()
    )

    denominator = math.sqrt(
        float(
            np.sum(
                first_centered ** 2
            )
            * np.sum(
                second_centered ** 2
            )
        )
    )

    if denominator <= 0:
        return float("nan")

    return float(
        np.sum(
            first_centered
            * second_centered
        )
        / denominator
    )


def average_ranks(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    position = 0

    while position < len(values):
        end_position = (
            position + 1
        )

        while (
            end_position < len(values)
            and values[
                order[end_position]
            ]
            == values[
                order[position]
            ]
        ):
            end_position += 1

        average_rank = (
            position
            + end_position
            - 1
        ) / 2.0

        ranks[
            order[
                position:end_position
            ]
        ] = average_rank

        position = end_position

    return ranks


depth_anchor_pearson = (
    pearson_correlation(
        depth_values,
        expected_anchor_values,
    )
)

if len(depth_values) >= 2:
    depth_anchor_spearman = (
        pearson_correlation(
            average_ranks(
                depth_values
            ),
            average_ranks(
                expected_anchor_values
            ),
        )
    )
else:
    depth_anchor_spearman = float(
        "nan"
    )

adjacent_anchor_increases = []

for index in range(
    len(expected_anchor_values) - 1
):
    adjacent_anchor_increases.append(
        expected_anchor_values[
            index + 1
        ]
        > expected_anchor_values[
            index
        ]
    )

if adjacent_anchor_increases:
    monotonic_depth_transition_ratio = (
        sum(
            adjacent_anchor_increases
        )
        / len(
            adjacent_anchor_increases
        )
    )
else:
    monotonic_depth_transition_ratio = float(
        "nan"
    )


# ============================================================
# 13. Per-topic summaries
# ============================================================

hierarchy_node_by_topic = {
    int(node["topic_id"]): node
    for node in hierarchy_nodes
}

per_topic_records = []

for topic_id in range(
    NUM_TOPICS
):
    top_indices = torch.topk(
        beta[topic_id],
        k=min(
            20,
            VOCAB_SIZE,
        ),
    ).indices.tolist()

    top_words = [
        VOCAB[int(index)]
        for index in top_indices
    ]

    hierarchy_node = (
        hierarchy_node_by_topic.get(
            topic_id,
            {},
        )
    )

    per_topic_records.append(
        {
            "topic_id": topic_id,
            "topic_number": (
                topic_id + 1
            ),
            "anchor_coordinate": float(
                anchor_coordinates[
                    topic_id
                ].item()
            ),
            "topic_usage": float(
                topic_usage[
                    topic_id
                ].item()
            ),
            "npmi": float(
                per_topic_npmi[
                    topic_id
                ]
            ),
            "exclusivity": float(
                per_topic_exclusivity[
                    topic_id
                ]
            ),
            "normalized_entropy": float(
                normalized_topic_entropy[
                    topic_id
                ].item()
            ),
            "maximum_cosine_similarity": float(
                per_topic_maximum_similarity[
                    topic_id
                ]
            ),
            "role": hierarchy_node.get(
                "role",
                "unknown",
            ),
            "incoming_edges": int(
                hierarchy_node.get(
                    "incoming_edges",
                    0,
                )
            ),
            "outgoing_edges": int(
                hierarchy_node.get(
                    "outgoing_edges",
                    0,
                )
            ),
            "top_words": top_words,
        }
    )


# ============================================================
# 14. Save per-topic CSV
# ============================================================

topic_summary_csv_path = os.path.join(
    SECTION8_RESULT_DIR,
    "per_topic_evaluation.csv",
)

with open(
    topic_summary_csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = [
        "topic_id",
        "topic_number",
        "anchor_coordinate",
        "topic_usage",
        "npmi",
        "exclusivity",
        "normalized_entropy",
        "maximum_cosine_similarity",
        "role",
        "incoming_edges",
        "outgoing_edges",
        "top_words",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for record in per_topic_records:
        csv_record = dict(
            record
        )

        csv_record["top_words"] = (
            ", ".join(
                record[
                    "top_words"
                ]
            )
        )

        writer.writerow(
            csv_record
        )


# ============================================================
# 15. Save depth-alignment CSV
# ============================================================

depth_alignment_csv_path = os.path.join(
    SECTION8_RESULT_DIR,
    "depth_anchor_alignment.csv",
)

with open(
    depth_alignment_csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "depth",
            "expected_anchor_coordinate",
        ],
    )

    writer.writeheader()

    for depth_value, expected_anchor in zip(
        depth_values,
        expected_anchor_values,
    ):
        writer.writerow(
            {
                "depth": int(
                    depth_value
                ),
                "expected_anchor_coordinate": (
                    expected_anchor
                ),
            }
        )


# ============================================================
# 16. Visualizations
# ============================================================

figure_paths = {}

try:
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # 16.1 Per-topic NPMI
    # --------------------------------------------------------
    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        np.arange(NUM_TOPICS),
        per_topic_npmi,
        color="steelblue",
    )

    plt.axhline(
        mean_topic_npmi,
        color="red",
        linestyle="--",
        label=(
            f"Mean={mean_topic_npmi:.3f}"
        ),
    )

    plt.xlabel("Topic ID")
    plt.ylabel("NPMI")
    plt.title(
        "Per-topic NPMI coherence"
    )
    plt.legend()
    plt.tight_layout()

    npmi_figure_path = os.path.join(
        SECTION8_FIGURE_DIR,
        "topic_npmi.png",
    )

    plt.savefig(
        npmi_figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    figure_paths[
        "topic_npmi"
    ] = npmi_figure_path

    # --------------------------------------------------------
    # 16.2 Topic usage
    # --------------------------------------------------------
    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        np.arange(NUM_TOPICS),
        topic_usage.numpy(),
        color="darkseagreen",
    )

    plt.xlabel("Topic ID")
    plt.ylabel("Train topic usage")
    plt.title(
        "Mean train topic usage"
    )
    plt.tight_layout()

    usage_figure_path = os.path.join(
        SECTION8_FIGURE_DIR,
        "topic_usage.png",
    )

    plt.savefig(
        usage_figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    figure_paths[
        "topic_usage"
    ] = usage_figure_path

    # --------------------------------------------------------
    # 16.3 Anchor coordinates
    # --------------------------------------------------------
    plt.figure(
        figsize=(10, 5)
    )

    plt.scatter(
        np.arange(NUM_TOPICS),
        anchor_coordinates.numpy(),
        c=np.arange(NUM_TOPICS),
        cmap="viridis",
        s=60,
    )

    plt.plot(
        np.arange(NUM_TOPICS),
        anchor_coordinates.numpy(),
        color="gray",
        alpha=0.5,
    )

    plt.xlabel("Topic ID")
    plt.ylabel("Anchor coordinate")
    plt.title(
        "Learned isotonic topic anchors"
    )
    plt.tight_layout()

    anchor_figure_path = os.path.join(
        SECTION8_FIGURE_DIR,
        "topic_anchor_coordinates.png",
    )

    plt.savefig(
        anchor_figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    figure_paths[
        "anchor_coordinates"
    ] = anchor_figure_path

    # --------------------------------------------------------
    # 16.4 Depth-anchor alignment
    # --------------------------------------------------------
    if depth_values:
        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            depth_values,
            expected_anchor_values,
            marker="o",
            linewidth=2,
        )

        plt.xlabel("Claim depth")
        plt.ylabel(
            "Expected anchor coordinate"
        )
        plt.title(
            "Depth-anchor alignment"
        )
        plt.grid(
            alpha=0.3
        )
        plt.tight_layout()

        depth_figure_path = os.path.join(
            SECTION8_FIGURE_DIR,
            "depth_anchor_alignment.png",
        )

        plt.savefig(
            depth_figure_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        figure_paths[
            "depth_anchor_alignment"
        ] = depth_figure_path

    # --------------------------------------------------------
    # 16.5 Empirical transition heatmap
    # --------------------------------------------------------
    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        empirical_joint.numpy(),
        aspect="auto",
        cmap="magma",
    )

    plt.colorbar(
        label="Empirical joint probability"
    )

    plt.xlabel("Child topic")
    plt.ylabel("Parent topic")
    plt.title(
        "Train empirical topic-transition joint"
    )
    plt.tight_layout()

    transition_figure_path = os.path.join(
        SECTION8_FIGURE_DIR,
        "empirical_transition_heatmap.png",
    )

    plt.savefig(
        transition_figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    figure_paths[
        "transition_heatmap"
    ] = transition_figure_path

    print(
        "\n[PASS] Evaluation figures saved."
    )

except ImportError:
    print(
        "\n[WARNING] matplotlib is unavailable. "
        "Numerical evaluation completed without figures."
    )

except Exception as visualization_error:
    print(
        "\n[WARNING] Figure generation failed: "
        f"{visualization_error}"
    )


# ============================================================
# 17. Final evaluation report
# ============================================================

evaluation_report = {
    "run_name": global_hierarchy.get(
        "run_name"
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "evaluation_protocol": {
        "topic_quality_corpus": (
            "train claim documents"
        ),
        "hierarchy_source": (
            "train adjacent-depth dependency edges"
        ),
        "test_split_used_for_model_selection": False,
        "npmi_unobserved_pair_score": -1.0,
        "topic_diversity_top_words": (
            TOP_WORDS_FOR_DIVERSITY
        ),
        "npmi_top_words": (
            TOP_WORDS_FOR_NPMI
        ),
        "exclusivity_top_words": (
            TOP_WORDS_FOR_EXCLUSIVITY
        ),
    },
    "model": {
        "number_of_topics": (
            NUM_TOPICS
        ),
        "vocabulary_size": (
            VOCAB_SIZE
        ),
        "best_epoch": (
            global_hierarchy.get(
                "best_epoch"
            )
        ),
    },
    "corpus": {
        "train_claim_documents": (
            number_of_documents
        ),
        "empty_target_vocabulary_documents": (
            number_of_empty_vocabulary_documents
        ),
        "empty_target_vocabulary_ratio": (
            number_of_empty_vocabulary_documents
            / number_of_documents
        ),
    },
    "topic_quality": {
        "topic_diversity": (
            topic_diversity
        ),
        "unique_top_words": (
            number_of_unique_top_words
        ),
        "mean_npmi": (
            mean_topic_npmi
        ),
        "median_npmi": (
            median_topic_npmi
        ),
        "mean_exclusivity": (
            mean_topic_exclusivity
        ),
        "mean_normalized_entropy": (
            mean_normalized_topic_entropy
        ),
        "mean_topic_cosine_similarity": (
            mean_topic_cosine_similarity
        ),
        "maximum_topic_cosine_similarity": (
            maximum_topic_cosine_similarity
        ),
    },
    "hierarchy_quality": {
        "number_of_nodes": len(
            hierarchy_nodes
        ),
        "number_of_edges": len(
            hierarchy_edges
        ),
        "is_acyclic": bool(
            global_hierarchy.get(
                "is_acyclic",
                False,
            )
        ),
        "directional_violations": (
            directional_violations
        ),
        "directional_empirical_mass_ratio": (
            directional_mass_ratio
        ),
        "selected_edge_joint_mass_ratio": (
            selected_edge_mass_ratio
        ),
        "mean_selected_edge_conditional": (
            mean_hierarchy_edge_conditional
        ),
        "minimum_selected_edge_conditional": (
            minimum_hierarchy_edge_conditional
        ),
    },
    "depth_anchor_alignment": {
        "number_of_depth_levels": len(
            depth_values
        ),
        "pearson_correlation": finite_or_none(
            depth_anchor_pearson
        ),
        "spearman_correlation": finite_or_none(
            depth_anchor_spearman
        ),
        "monotonic_transition_ratio": finite_or_none(
            monotonic_depth_transition_ratio
        ),
        "depths": [
            int(value)
            for value in depth_values
        ],
        "expected_anchor_coordinates": [
            float(value)
            for value in expected_anchor_values
        ],
    },
    "per_topic": (
        per_topic_records
    ),
    "files": {
        "topic_summary_csv": (
            topic_summary_csv_path
        ),
        "depth_alignment_csv": (
            depth_alignment_csv_path
        ),
        "figures": figure_paths,
        "global_hierarchy": (
            GLOBAL_HIERARCHY_PATH
        ),
        "transition_matrices": (
            GLOBAL_MATRIX_PATH
        ),
    },
}

evaluation_report_path = os.path.join(
    SECTION8_RESULT_DIR,
    "section8_evaluation_report.json",
)

atomic_json_save_section8(
    evaluation_report,
    evaluation_report_path,
)


# ============================================================
# 18. Final checks
# ============================================================

if not os.path.isfile(
    evaluation_report_path
):
    raise FileNotFoundError(
        "Evaluation report was not saved."
    )

if not os.path.isfile(
    topic_summary_csv_path
):
    raise FileNotFoundError(
        "Per-topic CSV was not saved."
    )

if directional_violations != 0:
    raise RuntimeError(
        "Global hierarchy contains directional violations."
    )

if not bool(
    global_hierarchy.get(
        "is_acyclic",
        False,
    )
):
    raise RuntimeError(
        "Global hierarchy is not marked as acyclic."
    )


# ============================================================
# 19. Final status
# ============================================================

print("\n" + "=" * 72)
print("SECTION 8 COMPLETED SUCCESSFULLY")
print("=" * 72)
print(f"Topics                         : {NUM_TOPICS}")
print(f"Train claim documents          : {number_of_documents:,}")
print(f"Topic diversity                : {topic_diversity:.6f}")
print(f"Mean NPMI                      : {mean_topic_npmi:.6f}")
print(f"Median NPMI                    : {median_topic_npmi:.6f}")
print(f"Mean topic exclusivity         : {mean_topic_exclusivity:.6f}")
print(
    f"Mean normalized entropy        : "
    f"{mean_normalized_topic_entropy:.6f}"
)
print(
    f"Mean topic cosine similarity   : "
    f"{mean_topic_cosine_similarity:.6f}"
)
print(
    f"Maximum topic cosine similarity: "
    f"{maximum_topic_cosine_similarity:.6f}"
)
print(
    f"Directional empirical mass     : "
    f"{directional_mass_ratio:.6f}"
)
print(
    f"Selected hierarchy mass        : "
    f"{selected_edge_mass_ratio:.6f}"
)
print(
    f"Depth-anchor Pearson           : "
    f"{depth_anchor_pearson:.6f}"
    if math.isfinite(depth_anchor_pearson)
    else "Depth-anchor Pearson           : unavailable"
)
print(
    f"Depth-anchor Spearman          : "
    f"{depth_anchor_spearman:.6f}"
    if math.isfinite(depth_anchor_spearman)
    else "Depth-anchor Spearman          : unavailable"
)
print(f"Hierarchy edges                : {len(hierarchy_edges)}")
print(f"Directional violations         : {directional_violations}")
print(f"Evaluation report              : {evaluation_report_path}")
print(f"Per-topic CSV                  : {topic_summary_csv_path}")
print(f"Figure directory               : {SECTION8_FIGURE_DIR}")
print("=" * 72)

if FEATURE_RUN_NAME.startswith(
    "debug"
):
    print(
        "\n[IMPORTANT] These are debug-run diagnostics and must not "
        "be reported as final experimental results."
    )
else:
    print(
        "\n[PASS] Full-run topic quality and hierarchy evaluation "
        "completed."
    )

print(
    "\nSection 8 outputs are saved to Google Drive. "
    "The next stage is Section 9: patent-level hierarchy export, "
    "qualitative examples, and publication-ready result tables."
)

# ============================================================
# SECTION 9:
# Patent-Level Hierarchy Export, Qualitative Examples,
# and Publication-Ready Tables
# COMPLETE SINGLE-CELL VERSION
# ============================================================

import os
import csv
import json
import math
import heapq
from collections import defaultdict
from datetime import datetime

import torch
from tqdm.auto import tqdm


# ============================================================
# 0. Preconditions
# ============================================================

required_section9_globals = [
    "CONFIG",
    "FEATURE_RUN_NAME",
    "VOCAB",
    "train_dataset",
    "dev_dataset",
    "test_dataset",
    "SECTION7_RESULT_DIR",
    "SECTION8_RESULT_DIR",
]

missing_section9_globals = [
    name
    for name in required_section9_globals
    if name not in globals()
]

if missing_section9_globals:
    raise RuntimeError(
        "Section 9 prerequisites are missing: "
        f"{missing_section9_globals}. "
        "Run Sections 0 through 8 first."
    )

GLOBAL_HIERARCHY_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "global_topic_hierarchy.json",
)

GLOBAL_MATRIX_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "global_transition_matrices.pt",
)

TOPIC_WORD_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "topic_word_distribution.pt",
)

INFERENCE_MANIFEST_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "inference_manifest.json",
)

SECTION8_REPORT_PATH = os.path.join(
    SECTION8_RESULT_DIR,
    "section8_evaluation_report.json",
)

required_input_files = [
    GLOBAL_HIERARCHY_PATH,
    GLOBAL_MATRIX_PATH,
    TOPIC_WORD_PATH,
    INFERENCE_MANIFEST_PATH,
    SECTION8_REPORT_PATH,
]

missing_input_files = [
    path
    for path in required_input_files
    if not os.path.isfile(path)
]

if missing_input_files:
    raise FileNotFoundError(
        "Required Section 7/8 output files are missing: "
        f"{missing_input_files}"
    )


# ============================================================
# 1. Configuration
# ============================================================

SECTION9_RESULT_DIR = os.path.join(
    SECTION7_RESULT_DIR,
    "section9_patent_exports",
)

PATENT_EXPORT_DIR = os.path.join(
    SECTION9_RESULT_DIR,
    "patent_hierarchies",
)

TABLE_DIR = os.path.join(
    SECTION9_RESULT_DIR,
    "tables",
)

QUALITATIVE_DIR = os.path.join(
    SECTION9_RESULT_DIR,
    "qualitative_examples",
)

for directory in [
    SECTION9_RESULT_DIR,
    PATENT_EXPORT_DIR,
    TABLE_DIR,
    QUALITATIVE_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )

CLAIM_TOP_K_TOPICS = int(
    getattr(
        CONFIG,
        "claim_top_k_topics",
        3,
    )
)

TOP_WORDS_PER_TOPIC_EXPORT = int(
    getattr(
        CONFIG,
        "export_top_words_per_topic",
        10,
    )
)

QUALITATIVE_PATENTS_PER_SPLIT = int(
    getattr(
        CONFIG,
        "qualitative_patents_per_split",
        5,
    )
)

CLAIM_TEXT_SNIPPET_LENGTH = int(
    getattr(
        CONFIG,
        "claim_text_snippet_length",
        500,
    )
)

EXPORT_FULL_THETA = bool(
    getattr(
        CONFIG,
        "export_full_theta",
        False,
    )
)

if CLAIM_TOP_K_TOPICS < 1:
    raise ValueError(
        "CLAIM_TOP_K_TOPICS must be positive."
    )

if TOP_WORDS_PER_TOPIC_EXPORT < 1:
    raise ValueError(
        "TOP_WORDS_PER_TOPIC_EXPORT must be positive."
    )

if QUALITATIVE_PATENTS_PER_SPLIT < 1:
    raise ValueError(
        "QUALITATIVE_PATENTS_PER_SPLIT must be positive."
    )

if CLAIM_TEXT_SNIPPET_LENGTH < 1:
    raise ValueError(
        "CLAIM_TEXT_SNIPPET_LENGTH must be positive."
    )

print("=== SECTION 9 CONFIGURATION ===")
print(f"Feature run                    : {FEATURE_RUN_NAME}")
print(f"Result directory               : {SECTION9_RESULT_DIR}")
print(f"Claim top-k topics             : {CLAIM_TOP_K_TOPICS}")
print(f"Topic words per export         : {TOP_WORDS_PER_TOPIC_EXPORT}")
print(f"Qualitative patents per split  : {QUALITATIVE_PATENTS_PER_SPLIT}")
print(f"Claim text snippet length      : {CLAIM_TEXT_SNIPPET_LENGTH}")
print(f"Export full theta              : {EXPORT_FULL_THETA}")


# ============================================================
# 2. Serialization utilities
# ============================================================

def atomic_json_save_section9(
    data,
    final_path,
):
    os.makedirs(
        os.path.dirname(final_path),
        exist_ok=True,
    )

    temporary_path = final_path + ".tmp"

    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    with open(
        temporary_path,
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
        temporary_path,
        final_path,
    )


def sanitize_json_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            value = value.item()
        else:
            return [
                sanitize_json_value(item)
                for item in value.tolist()
            ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return float(value)

    if isinstance(value, dict):
        return {
            str(key): sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            sanitize_json_value(item)
            for item in value
        ]

    return value


def normalize_patent_id_section9(
    patent_id,
):
    normalized = str(
        patent_id
    ).strip()

    if not normalized:
        raise ValueError(
            "Empty patent ID detected."
        )

    return normalized


def normalize_claim_id_section9(
    claim_id,
):
    try:
        normalized = int(
            claim_id
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid claim ID: {claim_id}"
        ) from error

    if normalized < 1:
        raise ValueError(
            f"Claim ID must be positive, got {normalized}."
        )

    return normalized


# ============================================================
# 3. Load Section 7 and Section 8 outputs
# ============================================================

with open(
    GLOBAL_HIERARCHY_PATH,
    "r",
    encoding="utf-8",
) as file:
    global_hierarchy = json.load(file)

with open(
    INFERENCE_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    inference_manifest = json.load(file)

with open(
    SECTION8_REPORT_PATH,
    "r",
    encoding="utf-8",
) as file:
    section8_report = json.load(file)

global_matrices = torch.load(
    GLOBAL_MATRIX_PATH,
    map_location="cpu",
    weights_only=False,
)

topic_word_payload = torch.load(
    TOPIC_WORD_PATH,
    map_location="cpu",
    weights_only=False,
)

beta = (
    topic_word_payload["beta"]
    .detach()
    .float()
    .cpu()
)

anchor_coordinates = (
    global_matrices[
        "anchor_coordinates"
    ]
    .detach()
    .float()
    .cpu()
)

directional_conditional = (
    global_matrices[
        "directional_conditional"
    ]
    .detach()
    .float()
    .cpu()
)

empirical_joint = (
    global_matrices[
        "empirical_joint"
    ]
    .detach()
    .float()
    .cpu()
)

topic_usage = (
    global_matrices[
        "topic_usage"
    ]
    .detach()
    .float()
    .cpu()
)

NUM_TOPICS = int(
    beta.shape[0]
)

VOCAB_SIZE = int(
    beta.shape[1]
)

if VOCAB_SIZE != len(VOCAB):
    raise ValueError(
        "Beta/vocabulary size mismatch."
    )

if anchor_coordinates.shape != (
    NUM_TOPICS,
):
    raise ValueError(
        "Anchor-coordinate shape mismatch."
    )

if directional_conditional.shape != (
    NUM_TOPICS,
    NUM_TOPICS,
):
    raise ValueError(
        "Directional conditional matrix shape mismatch."
    )

if empirical_joint.shape != (
    NUM_TOPICS,
    NUM_TOPICS,
):
    raise ValueError(
        "Empirical joint matrix shape mismatch."
    )

if topic_usage.shape != (
    NUM_TOPICS,
):
    raise ValueError(
        "Topic-usage shape mismatch."
    )

if not torch.isfinite(beta).all():
    raise FloatingPointError(
        "Non-finite beta values detected."
    )

if not torch.isfinite(
    anchor_coordinates
).all():
    raise FloatingPointError(
        "Non-finite anchor coordinates detected."
    )

if not torch.isfinite(
    directional_conditional
).all():
    raise FloatingPointError(
        "Non-finite directional conditional values detected."
    )

print("\n=== LOADED OUTPUTS ===")
print(f"Number of topics : {NUM_TOPICS}")
print(f"Vocabulary size  : {VOCAB_SIZE}")
print(f"Hierarchy nodes  : {len(global_hierarchy['nodes'])}")
print(f"Hierarchy edges  : {len(global_hierarchy['edges'])}")


# ============================================================
# 4. Topic summaries
# ============================================================

actual_topic_word_count = min(
    TOP_WORDS_PER_TOPIC_EXPORT,
    VOCAB_SIZE,
)

top_word_probabilities, top_word_indices = torch.topk(
    beta,
    k=actual_topic_word_count,
    dim=-1,
)

topic_summaries = {}

for topic_id in range(
    NUM_TOPICS
):
    top_words = []

    for rank in range(
        actual_topic_word_count
    ):
        vocabulary_index = int(
            top_word_indices[
                topic_id,
                rank,
            ].item()
        )

        top_words.append(
            {
                "rank": rank + 1,
                "word": VOCAB[
                    vocabulary_index
                ],
                "probability": float(
                    top_word_probabilities[
                        topic_id,
                        rank,
                    ].item()
                ),
            }
        )

    topic_summaries[
        topic_id
    ] = {
        "topic_id": topic_id,
        "topic_number": topic_id + 1,
        "anchor_coordinate": float(
            anchor_coordinates[
                topic_id
            ].item()
        ),
        "train_topic_usage": float(
            topic_usage[
                topic_id
            ].item()
        ),
        "top_words": top_words,
    }


# ============================================================
# 5. Global hierarchy masks
# ============================================================

selected_hierarchy_mask = torch.zeros(
    (
        NUM_TOPICS,
        NUM_TOPICS,
    ),
    dtype=torch.float32,
)

global_edge_lookup = {}

for edge in global_hierarchy[
    "edges"
]:
    parent_topic = int(
        edge[
            "parent_topic_id"
        ]
    )

    child_topic = int(
        edge[
            "child_topic_id"
        ]
    )

    if not (
        0 <= parent_topic < NUM_TOPICS
        and 0 <= child_topic < NUM_TOPICS
    ):
        raise IndexError(
            "Global hierarchy contains an invalid topic ID."
        )

    if not (
        anchor_coordinates[
            child_topic
        ]
        > anchor_coordinates[
            parent_topic
        ]
    ):
        raise ValueError(
            "Global hierarchy contains an anchor-direction violation."
        )

    selected_hierarchy_mask[
        parent_topic,
        child_topic,
    ] = 1.0

    global_edge_lookup[
        (
            parent_topic,
            child_topic,
        )
    ] = edge


# ============================================================
# 6. Processed-record lookup tables
# ============================================================

def build_record_lookup(
    dataset,
    split_name,
):
    lookup = {}

    for record in tqdm(
        dataset.records,
        desc=f"{split_name} record lookup",
    ):
        patent_id = (
            normalize_patent_id_section9(
                record["patent_id"]
            )
        )

        if patent_id in lookup:
            raise ValueError(
                f"Duplicate patent ID in {split_name}: "
                f"{patent_id}"
            )

        lookup[
            patent_id
        ] = record

    if not lookup:
        raise RuntimeError(
            f"No records found for split={split_name}."
        )

    return lookup


DATASETS_BY_SPLIT = {
    "train": train_dataset,
    "dev": dev_dataset,
    "test": test_dataset,
}

RECORD_LOOKUPS = {}

for split_name, dataset in (
    DATASETS_BY_SPLIT.items()
):
    RECORD_LOOKUPS[
        split_name
    ] = build_record_lookup(
        dataset=dataset,
        split_name=split_name,
    )


# ============================================================
# 7. Inference-shard resolution
# ============================================================

def resolve_inference_shard_path(
    shard_record,
):
    candidate_path = (
        shard_record.get(
            "path"
        )
    )

    if (
        candidate_path is not None
        and os.path.isfile(
            candidate_path
        )
    ):
        return candidate_path

    shard_filename = (
        shard_record.get(
            "file"
        )
    )

    if not shard_filename:
        raise KeyError(
            "Inference shard record has no file name."
        )

    fallback_path = os.path.join(
        SECTION7_RESULT_DIR,
        "claim_theta_shards",
        shard_filename,
    )

    if not os.path.isfile(
        fallback_path
    ):
        raise FileNotFoundError(
            "Inference shard not found: "
            f"{fallback_path}"
        )

    return fallback_path


# ============================================================
# 8. Claim-topic summary helper
# ============================================================

def create_claim_topic_summary(
    theta_row,
):
    if theta_row.ndim != 1:
        raise ValueError(
            "theta_row must be one-dimensional."
        )

    if theta_row.shape[0] != NUM_TOPICS:
        raise ValueError(
            "theta_row topic-count mismatch."
        )

    if not torch.isfinite(
        theta_row
    ).all():
        raise FloatingPointError(
            "Non-finite theta detected."
        )

    if torch.any(theta_row < 0):
        raise ValueError(
            "Negative theta detected."
        )

    theta_sum = float(
        theta_row.sum().item()
    )

    if not math.isclose(
        theta_sum,
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError(
            f"Theta does not sum to one: {theta_sum:.8f}"
        )

    actual_top_k = min(
        CLAIM_TOP_K_TOPICS,
        NUM_TOPICS,
    )

    top_probabilities, top_indices = torch.topk(
        theta_row,
        k=actual_top_k,
    )

    dominant_topic = int(
        top_indices[0].item()
    )

    expected_anchor = float(
        torch.dot(
            theta_row,
            anchor_coordinates,
        ).item()
    )

    top_topics = []

    for rank in range(
        actual_top_k
    ):
        topic_id = int(
            top_indices[
                rank
            ].item()
        )

        top_topics.append(
            {
                "rank": rank + 1,
                "topic_id": topic_id,
                "topic_number": (
                    topic_id + 1
                ),
                "probability": float(
                    top_probabilities[
                        rank
                    ].item()
                ),
                "anchor_coordinate": float(
                    anchor_coordinates[
                        topic_id
                    ].item()
                ),
                "top_words": [
                    item["word"]
                    for item in
                    topic_summaries[
                        topic_id
                    ]["top_words"][:5]
                ],
            }
        )

    result = {
        "dominant_topic_id": (
            dominant_topic
        ),
        "dominant_topic_number": (
            dominant_topic + 1
        ),
        "dominant_topic_probability": float(
            theta_row[
                dominant_topic
            ].item()
        ),
        "expected_anchor_coordinate": (
            expected_anchor
        ),
        "top_topics": top_topics,
    }

    if EXPORT_FULL_THETA:
        result["theta"] = [
            float(value)
            for value in
            theta_row.tolist()
        ]

    return result


# ============================================================
# 9. Patent payload construction
# ============================================================

def build_patent_payload(
    split_name,
    patent_id,
    record,
    claim_ids,
    theta_rows,
    claim_depths,
    bow_valid_flags,
):
    patent_id = (
        normalize_patent_id_section9(
            patent_id
        )
    )

    claim_ids = [
        normalize_claim_id_section9(
            claim_id
        )
        for claim_id in claim_ids
    ]

    if theta_rows.ndim != 2:
        raise ValueError(
            "theta_rows must have shape [C, K]."
        )

    number_of_claims = len(
        claim_ids
    )

    if theta_rows.shape != (
        number_of_claims,
        NUM_TOPICS,
    ):
        raise ValueError(
            "Patent theta shape mismatch: "
            f"observed={tuple(theta_rows.shape)}, "
            f"expected={(number_of_claims, NUM_TOPICS)}."
        )

    if claim_depths.shape != (
        number_of_claims,
    ):
        raise ValueError(
            "Patent claim-depth shape mismatch."
        )

    if bow_valid_flags.shape != (
        number_of_claims,
    ):
        raise ValueError(
            "Patent BoW-valid shape mismatch."
        )

    if len(set(claim_ids)) != number_of_claims:
        raise ValueError(
            f"Duplicate claim IDs in patent {patent_id}."
        )

    local_claim_index = {
        claim_id: position
        for position, claim_id
        in enumerate(claim_ids)
    }

    record_claims = {
        normalize_claim_id_section9(
            claim_id
        ): text
        for claim_id, text
        in record["claims"].items()
    }

    record_depth = {
        normalize_claim_id_section9(
            claim_id
        ): int(depth)
        for claim_id, depth
        in record["depth"].items()
    }

    if set(claim_ids) != set(
        record_claims.keys()
    ):
        raise ValueError(
            f"Inference/record claim mismatch for patent "
            f"{patent_id}."
        )

    claim_payloads = []
    claim_summary_lookup = {}

    for position, claim_id in enumerate(
        claim_ids
    ):
        inferred_depth = int(
            claim_depths[
                position
            ].item()
        )

        stored_depth = int(
            record_depth[
                claim_id
            ]
        )

        if inferred_depth != stored_depth:
            raise ValueError(
                f"Depth mismatch for patent {patent_id}, "
                f"claim {claim_id}: "
                f"inference={inferred_depth}, "
                f"record={stored_depth}."
            )

        topic_summary = (
            create_claim_topic_summary(
                theta_rows[
                    position
                ]
            )
        )

        claim_payload = {
            "claim_id": claim_id,
            "depth": inferred_depth,
            "bow_valid": bool(
                bow_valid_flags[
                    position
                ].item()
            ),
            **topic_summary,
        }

        claim_payloads.append(
            claim_payload
        )

        claim_summary_lookup[
            claim_id
        ] = {
            "theta": theta_rows[
                position
            ],
            "summary": claim_payload,
        }

    dependency_edges = []

    number_of_adjacent_edges = 0
    number_of_non_adjacent_edges = 0
    number_of_directional_edges = 0
    number_of_dominant_global_edges = 0

    hierarchy_support_values = []
    directional_support_values = []
    expected_anchor_gaps = []

    for parent_id, child_id in record[
        "edges"
    ]:
        parent_id = (
            normalize_claim_id_section9(
                parent_id
            )
        )

        child_id = (
            normalize_claim_id_section9(
                child_id
            )
        )

        if parent_id not in local_claim_index:
            raise KeyError(
                f"Unknown parent claim {parent_id} "
                f"in patent {patent_id}."
            )

        if child_id not in local_claim_index:
            raise KeyError(
                f"Unknown child claim {child_id} "
                f"in patent {patent_id}."
            )

        parent_payload = (
            claim_summary_lookup[
                parent_id
            ]
        )

        child_payload = (
            claim_summary_lookup[
                child_id
            ]
        )

        parent_theta = (
            parent_payload[
                "theta"
            ]
        )

        child_theta = (
            child_payload[
                "theta"
            ]
        )

        parent_depth = int(
            parent_payload[
                "summary"
            ]["depth"]
        )

        child_depth = int(
            child_payload[
                "summary"
            ]["depth"]
        )

        depth_difference = (
            child_depth
            - parent_depth
        )

        if depth_difference <= 0:
            raise ValueError(
                f"Dependency edge does not increase depth: "
                f"patent={patent_id}, "
                f"edge={parent_id}->{child_id}."
            )

        adjacent_depth = (
            depth_difference == 1
        )

        if adjacent_depth:
            number_of_adjacent_edges += 1
        else:
            number_of_non_adjacent_edges += 1

        parent_expected_anchor = float(
            parent_payload[
                "summary"
            ][
                "expected_anchor_coordinate"
            ]
        )

        child_expected_anchor = float(
            child_payload[
                "summary"
            ][
                "expected_anchor_coordinate"
            ]
        )

        expected_anchor_gap = (
            child_expected_anchor
            - parent_expected_anchor
        )

        direction_consistent = (
            expected_anchor_gap > 0.0
        )

        if direction_consistent:
            number_of_directional_edges += 1

        hierarchy_support = float(
            torch.dot(
                parent_theta,
                torch.mv(
                    selected_hierarchy_mask,
                    child_theta,
                ),
            ).item()
        )

        directional_support = float(
            torch.dot(
                parent_theta,
                torch.mv(
                    directional_conditional,
                    child_theta,
                ),
            ).item()
        )

        parent_dominant_topic = int(
            parent_payload[
                "summary"
            ][
                "dominant_topic_id"
            ]
        )

        child_dominant_topic = int(
            child_payload[
                "summary"
            ][
                "dominant_topic_id"
            ]
        )

        dominant_global_edge = (
            parent_dominant_topic,
            child_dominant_topic,
        ) in global_edge_lookup

        if dominant_global_edge:
            number_of_dominant_global_edges += 1

        hierarchy_support_values.append(
            hierarchy_support
        )

        directional_support_values.append(
            directional_support
        )

        expected_anchor_gaps.append(
            expected_anchor_gap
        )

        dependency_edges.append(
            {
                "parent_claim_id": parent_id,
                "child_claim_id": child_id,
                "parent_depth": parent_depth,
                "child_depth": child_depth,
                "depth_difference": depth_difference,
                "adjacent_depth": adjacent_depth,
                "parent_dominant_topic_id": (
                    parent_dominant_topic
                ),
                "parent_dominant_topic_number": (
                    parent_dominant_topic + 1
                ),
                "child_dominant_topic_id": (
                    child_dominant_topic
                ),
                "child_dominant_topic_number": (
                    child_dominant_topic + 1
                ),
                "dominant_topics_form_global_edge": (
                    dominant_global_edge
                ),
                "parent_expected_anchor": (
                    parent_expected_anchor
                ),
                "child_expected_anchor": (
                    child_expected_anchor
                ),
                "expected_anchor_gap": (
                    expected_anchor_gap
                ),
                "direction_consistent": (
                    direction_consistent
                ),
                "global_hierarchy_support": (
                    hierarchy_support
                ),
                "directional_transition_support": (
                    directional_support
                ),
            }
        )

    number_of_edges = len(
        dependency_edges
    )

    if number_of_edges > 0:
        directional_edge_ratio = (
            number_of_directional_edges
            / number_of_edges
        )

        dominant_global_edge_ratio = (
            number_of_dominant_global_edges
            / number_of_edges
        )

        mean_hierarchy_support = float(
            sum(
                hierarchy_support_values
            )
            / number_of_edges
        )

        mean_directional_support = float(
            sum(
                directional_support_values
            )
            / number_of_edges
        )

        mean_expected_anchor_gap = float(
            sum(
                expected_anchor_gaps
            )
            / number_of_edges
        )
    else:
        directional_edge_ratio = None
        dominant_global_edge_ratio = None
        mean_hierarchy_support = None
        mean_directional_support = None
        mean_expected_anchor_gap = None

    number_of_empty_bow_claims = sum(
        not bool(flag.item())
        for flag in bow_valid_flags
    )

    return {
        "split": split_name,
        "patent_id": patent_id,
        "cpc_codes": record.get(
            "cpc_codes"
        ),
        "section": record.get(
            "section"
        ),
        "class": record.get(
            "class"
        ),
        "subclass": record.get(
            "subclass"
        ),
        "number_of_claims": (
            number_of_claims
        ),
        "number_of_dependency_edges": (
            number_of_edges
        ),
        "number_of_adjacent_edges": (
            number_of_adjacent_edges
        ),
        "number_of_non_adjacent_edges": (
            number_of_non_adjacent_edges
        ),
        "maximum_depth": max(
            int(value.item())
            for value in claim_depths
        ),
        "valid_bow_claims": (
            number_of_claims
            - number_of_empty_bow_claims
        ),
        "empty_bow_claims": (
            number_of_empty_bow_claims
        ),
        "empty_bow_ratio": (
            number_of_empty_bow_claims
            / max(
                number_of_claims,
                1,
            )
        ),
        "directional_edge_ratio": (
            directional_edge_ratio
        ),
        "dominant_global_edge_ratio": (
            dominant_global_edge_ratio
        ),
        "mean_global_hierarchy_support": (
            mean_hierarchy_support
        ),
        "mean_directional_transition_support": (
            mean_directional_support
        ),
        "mean_expected_anchor_gap": (
            mean_expected_anchor_gap
        ),
        "claims": claim_payloads,
        "dependency_edges": (
            dependency_edges
        ),
    }


# ============================================================
# DEPTH-OT CPC ALIGNMENT EVALUATION
#
# Run immediately after Section 9 in the same notebook.
#
# Outputs:
#   - Section/Class/Subclass Pur_p, Pur_a, NMI
#   - all-claim and valid-BoW-only sensitivity results
#   - patent predictions
#   - publication-ready LaTeX row
# ============================================================

import os
import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sklearn.metrics import normalized_mutual_info_score


# ============================================================
# 0. Configuration
# ============================================================

# 기존 baseline에서 빈 BoW claim을 제외했다면 이 설정을 유지합니다.
PRIMARY_POLICY = "valid_bow_only"

# 비교를 위해 두 방식 모두 평가합니다.
EVALUATION_POLICIES = [
    "valid_bow_only",
    "all_claims",
]

EXPECTED_LABEL_COUNTS = {
    "section": 9,
    "class": 121,
    "subclass": 466,
}

ROUND_DIGITS = 4


# ============================================================
# 1. Preconditions
# ============================================================

required_globals = [
    "SECTION7_RESULT_DIR",
    "test_dataset",
]

missing_globals = [
    name
    for name in required_globals
    if name not in globals()
]

if missing_globals:
    raise RuntimeError(
        "필수 Section 7/9 객체가 없습니다: "
        f"{missing_globals}. "
        "Section 7과 Section 9를 실행한 같은 노트북에서 "
        "이 셀을 실행하세요."
    )

SECTION7_RESULT_DIR = Path(
    SECTION7_RESULT_DIR
)

INFERENCE_MANIFEST_PATH = (
    SECTION7_RESULT_DIR
    / "inference_manifest.json"
)

THETA_SHARD_DIR = (
    SECTION7_RESULT_DIR
    / "claim_theta_shards"
)

CPC_RESULT_DIR = (
    SECTION7_RESULT_DIR
    / "cpc_alignment"
)

CPC_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_NAME_CPC = str(
    globals().get(
        "RUN_NAME",
        SECTION7_RESULT_DIR.parent.name,
    )
)

FEATURE_RUN_NAME_CPC = str(
    globals().get(
        "FEATURE_RUN_NAME",
        "unknown",
    )
)

if not INFERENCE_MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        f"Inference manifest가 없습니다: "
        f"{INFERENCE_MANIFEST_PATH}"
    )

if not THETA_SHARD_DIR.is_dir():
    raise FileNotFoundError(
        f"Theta shard 폴더가 없습니다: "
        f"{THETA_SHARD_DIR}"
    )

if not hasattr(
    test_dataset,
    "records",
):
    raise AttributeError(
        "test_dataset에 records 속성이 없습니다."
    )

print("=" * 80)
print("DEPTH-OT PATENT-LEVEL CPC ALIGNMENT")
print("=" * 80)
print(f"Run name         : {RUN_NAME_CPC}")
print(f"Feature run      : {FEATURE_RUN_NAME_CPC}")
print(f"Section 7 result : {SECTION7_RESULT_DIR}")
print(f"Theta shard dir  : {THETA_SHARD_DIR}")
print(f"Primary policy   : {PRIMARY_POLICY}")
print("=" * 80)


# ============================================================
# 2. Utility functions
# ============================================================

def normalize_patent_id(
    patent_id,
):
    normalized = str(
        patent_id
    ).strip()

    if not normalized:
        raise ValueError(
            "빈 patent_id가 발견되었습니다."
        )

    return normalized


def normalize_single_label(
    value,
    field_name,
    patent_id,
):
    """
    CPC 평가에서는 patent마다 하나의 정답 label이 필요합니다.

    scalar이면 그대로 사용하고, 길이 1짜리 list이면 첫 값을
    사용합니다. 여러 값이 있으면 preprocessing에서 지정된
    첫 번째 primary label을 사용하고 경고 개수를 기록합니다.
    """

    if value is None:
        return None, False

    if isinstance(
        value,
        torch.Tensor,
    ):
        if value.ndim == 0:
            value = value.item()
        else:
            value = value.detach().cpu().tolist()

    if isinstance(
        value,
        np.ndarray,
    ):
        value = value.tolist()

    if isinstance(
        value,
        (list, tuple, set),
    ):
        values = [
            item
            for item in value
            if item is not None
            and str(item).strip()
        ]

        if not values:
            return None, False

        was_multivalued = (
            len(values) > 1
        )

        value = values[0]

    else:
        was_multivalued = False

    normalized = str(
        value
    ).strip()

    if not normalized:
        return None, was_multivalued

    if normalized.lower() in {
        "none",
        "nan",
        "null",
        "na",
    }:
        return None, was_multivalued

    return normalized, was_multivalued


def resolve_shard_path(
    shard_record,
):
    stored_path = shard_record.get(
        "path"
    )

    if stored_path:
        stored_path = Path(
            stored_path
        )

        if stored_path.is_file():
            return stored_path

    filename = shard_record.get(
        "file"
    )

    if not filename:
        raise KeyError(
            "Shard manifest에 file/path가 없습니다."
        )

    fallback_path = (
        THETA_SHARD_DIR
        / filename
    )

    if not fallback_path.is_file():
        raise FileNotFoundError(
            f"Theta shard를 찾지 못했습니다: "
            f"{fallback_path}"
        )

    return fallback_path


def calculate_clustering_metrics(
    true_labels,
    predicted_topics,
):
    true_labels = np.asarray(
        true_labels,
        dtype=str,
    )

    predicted_topics = np.asarray(
        predicted_topics,
        dtype=np.int64,
    )

    if len(true_labels) != len(
        predicted_topics
    ):
        raise ValueError(
            "정답과 예측 길이가 다릅니다."
        )

    if len(true_labels) == 0:
        raise ValueError(
            "평가할 patent가 없습니다."
        )

    contingency = pd.crosstab(
        pd.Series(
            predicted_topics,
            name="predicted_topic",
        ),
        pd.Series(
            true_labels,
            name="true_label",
        ),
        dropna=False,
    )

    matrix = contingency.to_numpy(
        dtype=np.int64
    )

    number_of_samples = int(
        matrix.sum()
    )

    if number_of_samples == 0:
        raise ValueError(
            "Contingency matrix가 비어 있습니다."
        )

    # Predicted-cluster purity:
    # 각 predicted cluster에서 가장 많은 true label을 선택
    pur_p = float(
        matrix.max(axis=1).sum()
        / number_of_samples
    )

    # Inverse label-wise purity:
    # 각 true label에서 가장 많은 predicted cluster를 선택
    pur_a = float(
        matrix.max(axis=0).sum()
        / number_of_samples
    )

    nmi = float(
        normalized_mutual_info_score(
            true_labels,
            predicted_topics,
            average_method="arithmetic",
        )
    )

    return {
        "pur_p": pur_p,
        "pur_a": pur_a,
        "nmi": nmi,
        "number_of_samples": (
            number_of_samples
        ),
        "number_of_true_labels": int(
            np.unique(
                true_labels
            ).size
        ),
        "number_of_predicted_topics": int(
            np.unique(
                predicted_topics
            ).size
        ),
    }


def atomic_json_save(
    payload,
    output_path,
):
    output_path = Path(
        output_path
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix + ".tmp"
        )
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )

    os.replace(
        temporary_path,
        output_path,
    )


# ============================================================
# 3. Load inference manifest
# ============================================================

with open(
    INFERENCE_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    inference_manifest_cpc = json.load(
        file
    )

if "splits" not in inference_manifest_cpc:
    raise KeyError(
        "Inference manifest에 splits가 없습니다."
    )

if "test" not in inference_manifest_cpc[
    "splits"
]:
    raise KeyError(
        "Inference manifest에 test split이 없습니다."
    )

test_manifest = (
    inference_manifest_cpc[
        "splits"
    ]["test"]
)

test_shard_records = (
    test_manifest.get(
        "shards",
        [],
    )
)

if not test_shard_records:
    raise RuntimeError(
        "Test theta shard 기록이 없습니다."
    )

expected_test_patents = int(
    test_manifest.get(
        "number_of_patents",
        len(test_dataset.records),
    )
)

expected_test_claims = int(
    test_manifest.get(
        "number_of_claims",
        0,
    )
)

print("\n=== TEST INFERENCE MANIFEST ===")
print(
    f"Expected patents: "
    f"{expected_test_patents:,}"
)
print(
    f"Expected claims : "
    f"{expected_test_claims:,}"
)
print(
    f"Theta shards    : "
    f"{len(test_shard_records):,}"
)


# ============================================================
# 4. Build patent-level CPC label lookup
# ============================================================

label_lookup = {}
multivalued_label_counts = {
    "section": 0,
    "class": 0,
    "subclass": 0,
}

missing_label_counts = {
    "section": 0,
    "class": 0,
    "subclass": 0,
}

for record in tqdm(
    test_dataset.records,
    desc="Building CPC label lookup",
):
    if "patent_id" not in record:
        raise KeyError(
            "Test record에 patent_id가 없습니다."
        )

    patent_id = normalize_patent_id(
        record["patent_id"]
    )

    if patent_id in label_lookup:
        raise ValueError(
            f"중복 test patent_id: {patent_id}"
        )

    normalized_labels = {}

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        label, was_multivalued = (
            normalize_single_label(
                record.get(level),
                field_name=level,
                patent_id=patent_id,
            )
        )

        normalized_labels[
            level
        ] = label

        if was_multivalued:
            multivalued_label_counts[
                level
            ] += 1

        if label is None:
            missing_label_counts[
                level
            ] += 1

    label_lookup[
        patent_id
    ] = normalized_labels

print("\n=== CPC LABEL LOOKUP ===")
print(
    f"Test patent records: "
    f"{len(label_lookup):,}"
)

for level in [
    "section",
    "class",
    "subclass",
]:
    observed_labels = {
        labels[level]
        for labels in label_lookup.values()
        if labels[level] is not None
    }

    print(
        f"{level:8s}: "
        f"labels={len(observed_labels):,}, "
        f"missing={missing_label_counts[level]:,}, "
        f"multi-valued={multivalued_label_counts[level]:,}"
    )

    expected_count = (
        EXPECTED_LABEL_COUNTS[level]
    )

    if len(observed_labels) != expected_count:
        print(
            f"[WARNING] {level} label 수가 예상과 다릅니다: "
            f"expected={expected_count}, "
            f"observed={len(observed_labels)}"
        )


# ============================================================
# 5. Aggregate claim theta to patent theta
# ============================================================

all_theta_sum = {}
all_claim_count = defaultdict(int)

valid_theta_sum = {}
valid_claim_count = defaultdict(int)

seen_claim_keys = set()

number_of_topics = None
loaded_claims = 0
loaded_valid_claims = 0
loaded_empty_claims = 0

for shard_record in tqdm(
    test_shard_records,
    desc="Aggregating test theta shards",
):
    shard_path = resolve_shard_path(
        shard_record
    )

    shard = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    if shard.get("split") != "test":
        raise ValueError(
            f"Test가 아닌 shard가 포함되었습니다: "
            f"{shard_path}"
        )

    theta = (
        shard["theta"]
        .detach()
        .double()
        .cpu()
    )

    bow_valid = (
        shard["bow_valid"]
        .detach()
        .bool()
        .cpu()
    )

    claim_keys = shard[
        "claim_keys"
    ]

    if theta.ndim != 2:
        raise ValueError(
            f"Theta는 2차원이어야 합니다: "
            f"{tuple(theta.shape)}"
        )

    shard_claim_count = int(
        theta.shape[0]
    )

    shard_topic_count = int(
        theta.shape[1]
    )

    if number_of_topics is None:
        number_of_topics = (
            shard_topic_count
        )

    elif number_of_topics != shard_topic_count:
        raise ValueError(
            "Shard 간 topic 수가 다릅니다."
        )

    if len(claim_keys) != shard_claim_count:
        raise ValueError(
            f"Claim key/theta 길이 불일치: "
            f"{shard_path}"
        )

    if bow_valid.shape != (
        shard_claim_count,
    ):
        raise ValueError(
            f"bow_valid shape 불일치: "
            f"{shard_path}"
        )

    if not torch.isfinite(
        theta
    ).all():
        raise FloatingPointError(
            f"Theta에 NaN/Inf가 있습니다: "
            f"{shard_path}"
        )

    if torch.any(
        theta < -1e-8
    ):
        raise ValueError(
            f"Theta에 음수가 있습니다: "
            f"{shard_path}"
        )

    theta = torch.clamp(
        theta,
        min=0.0,
    )

    theta_row_sums = theta.sum(
        dim=1,
        keepdim=True,
    )

    if torch.any(
        theta_row_sums <= 0
    ):
        raise ValueError(
            f"Theta 합이 0인 claim이 있습니다: "
            f"{shard_path}"
        )

    theta = theta / theta_row_sums

    for position, claim_key in enumerate(
        claim_keys
    ):
        if (
            not isinstance(
                claim_key,
                (list, tuple),
            )
            or len(claim_key) != 2
        ):
            raise ValueError(
                f"잘못된 claim key: {claim_key}"
            )

        patent_id = normalize_patent_id(
            claim_key[0]
        )

        claim_id = int(
            claim_key[1]
        )

        normalized_claim_key = (
            patent_id,
            claim_id,
        )

        if normalized_claim_key in seen_claim_keys:
            raise ValueError(
                f"중복 claim key: "
                f"{normalized_claim_key}"
            )

        seen_claim_keys.add(
            normalized_claim_key
        )

        theta_row = theta[
            position
        ].numpy()

        if patent_id not in all_theta_sum:
            all_theta_sum[
                patent_id
            ] = np.zeros(
                number_of_topics,
                dtype=np.float64,
            )

        all_theta_sum[
            patent_id
        ] += theta_row

        all_claim_count[
            patent_id
        ] += 1

        is_valid_bow = bool(
            bow_valid[
                position
            ].item()
        )

        if is_valid_bow:
            if patent_id not in valid_theta_sum:
                valid_theta_sum[
                    patent_id
                ] = np.zeros(
                    number_of_topics,
                    dtype=np.float64,
                )

            valid_theta_sum[
                patent_id
            ] += theta_row

            valid_claim_count[
                patent_id
            ] += 1

            loaded_valid_claims += 1

        else:
            loaded_empty_claims += 1

        loaded_claims += 1

    del shard
    del theta
    del bow_valid

if expected_test_claims > 0:
    if loaded_claims != expected_test_claims:
        raise RuntimeError(
            "Test claim 수가 manifest와 다릅니다: "
            f"loaded={loaded_claims:,}, "
            f"expected={expected_test_claims:,}"
        )

if len(all_theta_sum) != expected_test_patents:
    raise RuntimeError(
        "Test patent 수가 manifest와 다릅니다: "
        f"aggregated={len(all_theta_sum):,}, "
        f"expected={expected_test_patents:,}"
    )

print("\n=== THETA AGGREGATION ===")
print(
    f"Topics           : "
    f"{number_of_topics}"
)
print(
    f"Claims loaded    : "
    f"{loaded_claims:,}"
)
print(
    f"Valid-BoW claims : "
    f"{loaded_valid_claims:,}"
)
print(
    f"Empty-BoW claims : "
    f"{loaded_empty_claims:,}"
)
print(
    f"All patents      : "
    f"{len(all_theta_sum):,}"
)
print(
    f"Patents with >=1 valid claim: "
    f"{len(valid_theta_sum):,}"
)


# ============================================================
# 6. Create patent predictions for each policy
# ============================================================

prediction_frames = {}

for policy in EVALUATION_POLICIES:
    prediction_rows = []

    for patent_id in sorted(
        all_theta_sum.keys()
    ):
        if patent_id not in label_lookup:
            continue

        if policy == "all_claims":
            theta_sum = all_theta_sum[
                patent_id
            ]

            claim_count = all_claim_count[
                patent_id
            ]

        elif policy == "valid_bow_only":
            if valid_claim_count[
                patent_id
            ] <= 0:
                continue

            theta_sum = valid_theta_sum[
                patent_id
            ]

            claim_count = valid_claim_count[
                patent_id
            ]

        else:
            raise ValueError(
                f"알 수 없는 policy: {policy}"
            )

        patent_theta = (
            theta_sum
            / max(
                claim_count,
                1,
            )
        )

        patent_theta_sum = float(
            patent_theta.sum()
        )

        if (
            not math.isfinite(
                patent_theta_sum
            )
            or patent_theta_sum <= 0
        ):
            raise FloatingPointError(
                f"유효하지 않은 patent theta: "
                f"{patent_id}"
            )

        patent_theta = (
            patent_theta
            / patent_theta_sum
        )

        predicted_topic = int(
            np.argmax(
                patent_theta
            )
        )

        dominant_probability = float(
            patent_theta[
                predicted_topic
            ]
        )

        labels = label_lookup[
            patent_id
        ]

        prediction_rows.append(
            {
                "run_name": RUN_NAME_CPC,
                "feature_run": (
                    FEATURE_RUN_NAME_CPC
                ),
                "aggregation_policy": policy,
                "patent_id": patent_id,
                "predicted_topic": (
                    predicted_topic
                ),
                "predicted_topic_number": (
                    predicted_topic + 1
                ),
                "dominant_probability": (
                    dominant_probability
                ),
                "claims_used": int(
                    claim_count
                ),
                "all_claims": int(
                    all_claim_count[
                        patent_id
                    ]
                ),
                "valid_bow_claims": int(
                    valid_claim_count[
                        patent_id
                    ]
                ),
                "section": labels[
                    "section"
                ],
                "class": labels[
                    "class"
                ],
                "subclass": labels[
                    "subclass"
                ],
            }
        )

    prediction_frame = pd.DataFrame(
        prediction_rows
    )

    if prediction_frame.empty:
        raise RuntimeError(
            f"{policy} prediction이 비어 있습니다."
        )

    prediction_frames[
        policy
    ] = prediction_frame

    coverage = (
        len(prediction_frame)
        / max(
            len(label_lookup),
            1,
        )
    )

    print(
        f"\nPolicy={policy}: "
        f"patents={len(prediction_frame):,}, "
        f"coverage={coverage:.4%}"
    )


# ============================================================
# 7. Evaluate Section/Class/Subclass
# ============================================================

metric_rows = []

for policy in EVALUATION_POLICIES:
    predictions = prediction_frames[
        policy
    ]

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        evaluation_frame = (
            predictions[
                predictions[level].notna()
            ]
            .copy()
        )

        metrics = (
            calculate_clustering_metrics(
                true_labels=(
                    evaluation_frame[
                        level
                    ].to_numpy()
                ),
                predicted_topics=(
                    evaluation_frame[
                        "predicted_topic"
                    ].to_numpy()
                ),
            )
        )

        metric_row = {
            "run_name": RUN_NAME_CPC,
            "feature_run": (
                FEATURE_RUN_NAME_CPC
            ),
            "aggregation_policy": policy,
            "level": level,
            **metrics,
        }

        metric_rows.append(
            metric_row
        )

        print(
            f"{policy:16s} | "
            f"{level:8s} | "
            f"Pur_p={metrics['pur_p']:.4f} | "
            f"Pur_a={metrics['pur_a']:.4f} | "
            f"NMI={metrics['nmi']:.4f} | "
            f"N={metrics['number_of_samples']:,} | "
            f"labels={metrics['number_of_true_labels']:,} | "
            f"topics={metrics['number_of_predicted_topics']:,}"
        )

metrics_df = pd.DataFrame(
    metric_rows
)


# ============================================================
# 8. Save outputs
# ============================================================

metrics_csv_path = (
    CPC_RESULT_DIR
    / "depth_ot_cpc_alignment_metrics.csv"
)

predictions_csv_path = (
    CPC_RESULT_DIR
    / "depth_ot_patent_predictions.csv"
)

summary_json_path = (
    CPC_RESULT_DIR
    / "depth_ot_cpc_alignment_summary.json"
)

metrics_df.to_csv(
    metrics_csv_path,
    index=False,
)

all_prediction_frames = pd.concat(
    [
        prediction_frames[
            policy
        ]
        for policy in EVALUATION_POLICIES
    ],
    ignore_index=True,
)

all_prediction_frames.to_csv(
    predictions_csv_path,
    index=False,
)

summary_payload = {
    "run_name": RUN_NAME_CPC,
    "feature_run": FEATURE_RUN_NAME_CPC,
    "created_at": (
        datetime.now().isoformat()
    ),
    "number_of_topics": (
        number_of_topics
    ),
    "primary_policy": (
        PRIMARY_POLICY
    ),
    "policy_description": {
        "valid_bow_only": (
            "Patent theta is the normalized mean of claim theta "
            "over claims containing at least one fixed-vocabulary "
            "term. Patents with no valid-BoW claims are excluded."
        ),
        "all_claims": (
            "Patent theta is the normalized mean of all claim "
            "theta values, including empty-BoW claims."
        ),
    },
    "claim_counts": {
        "total": loaded_claims,
        "valid_bow": loaded_valid_claims,
        "empty_bow": loaded_empty_claims,
    },
    "patent_counts": {
        "total_test_records": len(
            label_lookup
        ),
        "all_claim_predictions": len(
            prediction_frames[
                "all_claims"
            ]
        ),
        "valid_bow_predictions": len(
            prediction_frames[
                "valid_bow_only"
            ]
        ),
    },
    "metrics": (
        metrics_df.to_dict(
            orient="records"
        )
    ),
    "files": {
        "metrics_csv": str(
            metrics_csv_path
        ),
        "predictions_csv": str(
            predictions_csv_path
        ),
    },
}

atomic_json_save(
    summary_payload,
    summary_json_path,
)


# ============================================================
# 9. Print publication-ready result
# ============================================================

primary_metrics = (
    metrics_df[
        metrics_df[
            "aggregation_policy"
        ] == PRIMARY_POLICY
    ]
    .set_index("level")
)

required_levels = {
    "section",
    "class",
    "subclass",
}

if not required_levels.issubset(
    set(primary_metrics.index)
):
    raise RuntimeError(
        "Primary policy의 CPC 결과가 불완전합니다."
    )

latex_values = []

for level in [
    "section",
    "class",
    "subclass",
]:
    for metric in [
        "pur_p",
        "pur_a",
        "nmi",
    ]:
        value = float(
            primary_metrics.loc[
                level,
                metric,
            ]
        )

        latex_values.append(
            f"{value:.{ROUND_DIGITS}f}"
        )

latex_row = (
    r"\textbf{Depth-OT (ours)}"
    + " & "
    + " & ".join(
        rf"\textbf{{{value}}}"
        for value in latex_values
    )
    + r" \\"
)

print("\n" + "=" * 80)
print("DEPTH-OT CPC ALIGNMENT COMPLETED")
print("=" * 80)
print(
    f"Primary policy: "
    f"{PRIMARY_POLICY}"
)

for level in [
    "section",
    "class",
    "subclass",
]:
    row = primary_metrics.loc[
        level
    ]

    print(
        f"{level:8s}: "
        f"Pur_p={row['pur_p']:.4f}, "
        f"Pur_a={row['pur_a']:.4f}, "
        f"NMI={row['nmi']:.4f}, "
        f"N={int(row['number_of_samples']):,}"
    )

print("\n=== LATEX TABLE ROW ===")
print(latex_row)

print("\n=== SAVED FILES ===")
print(f"Metrics     : {metrics_csv_path}")
print(f"Predictions : {predictions_csv_path}")
print(f"Summary JSON: {summary_json_path}")
print("=" * 80)


# ============================================================
# 10. Qualitative-example helper
# ============================================================

def add_claim_text_snippets(
    patent_payload,
    record,
):
    claims_by_id = {
        normalize_claim_id_section9(
            claim_id
        ): text
        for claim_id, text
        in record["claims"].items()
    }

    result = json.loads(
        json.dumps(
            sanitize_json_value(
                patent_payload
            ),
            ensure_ascii=False,
        )
    )

    for claim in result[
        "claims"
    ]:
        claim_id = int(
            claim["claim_id"]
        )

        claim_text = (
            claims_by_id.get(
                claim_id,
                "",
            )
        )

        if claim_text is None:
            claim_text = ""

        if not isinstance(
            claim_text,
            str,
        ):
            claim_text = str(
                claim_text
            )

        normalized_text = " ".join(
            claim_text.split()
        )

        claim[
            "text_snippet"
        ] = normalized_text[
            :CLAIM_TEXT_SNIPPET_LENGTH
        ]

        claim[
            "text_was_truncated_for_export"
        ] = (
            len(normalized_text)
            > CLAIM_TEXT_SNIPPET_LENGTH
        )

    return result


# ============================================================
# 11. Export one split
# ============================================================

def export_split_patent_hierarchies(
    split_name,
):
    split_manifest = (
        inference_manifest[
            "splits"
        ][split_name]
    )

    shard_records = (
        split_manifest[
            "shards"
        ]
    )

    record_lookup = (
        RECORD_LOOKUPS[
            split_name
        ]
    )

    final_jsonl_path = os.path.join(
        PATENT_EXPORT_DIR,
        f"{split_name}_patent_hierarchies.jsonl",
    )

    temporary_jsonl_path = (
        final_jsonl_path + ".tmp"
    )

    if os.path.exists(
        temporary_jsonl_path
    ):
        os.remove(
            temporary_jsonl_path
        )

    number_of_patents = 0
    number_of_claims = 0
    number_of_edges = 0
    number_of_adjacent_edges = 0
    number_of_non_adjacent_edges = 0
    number_of_directional_edges = 0
    number_of_empty_bow_claims = 0

    hierarchy_support_sum = 0.0
    hierarchy_support_count = 0

    directional_support_sum = 0.0
    directional_support_count = 0

    seen_patent_ids = set()

    qualitative_heap = []
    qualitative_counter = 0

    with open(
        temporary_jsonl_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        for shard_record in tqdm(
            shard_records,
            desc=f"Exporting {split_name} patents",
        ):
            shard_path = (
                resolve_inference_shard_path(
                    shard_record
                )
            )

            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )

            if shard.get(
                "split"
            ) != split_name:
                raise ValueError(
                    f"Inference shard split mismatch: "
                    f"{shard_path}"
                )

            theta = (
                shard["theta"]
                .detach()
                .float()
                .cpu()
            )

            claim_depth = (
                shard["claim_depth"]
                .detach()
                .long()
                .cpu()
            )

            bow_valid = (
                shard["bow_valid"]
                .detach()
                .bool()
                .cpu()
            )

            claim_keys = (
                shard["claim_keys"]
            )

            number_of_shard_claims = int(
                theta.shape[0]
            )

            if len(
                claim_keys
            ) != number_of_shard_claims:
                raise ValueError(
                    f"Claim-key count mismatch in {shard_path}."
                )

            if claim_depth.shape != (
                number_of_shard_claims,
            ):
                raise ValueError(
                    f"Claim-depth shape mismatch in {shard_path}."
                )

            if bow_valid.shape != (
                number_of_shard_claims,
            ):
                raise ValueError(
                    f"BoW-valid shape mismatch in {shard_path}."
                )

            patent_claim_positions = (
                defaultdict(list)
            )

            for position, claim_key in enumerate(
                claim_keys
            ):
                if (
                    not isinstance(
                        claim_key,
                        (list, tuple),
                    )
                    or len(claim_key) != 2
                ):
                    raise ValueError(
                        f"Invalid claim key in {shard_path}: "
                        f"{claim_key}"
                    )

                patent_id = (
                    normalize_patent_id_section9(
                        claim_key[0]
                    )
                )

                claim_id = (
                    normalize_claim_id_section9(
                        claim_key[1]
                    )
                )

                patent_claim_positions[
                    patent_id
                ].append(
                    (
                        position,
                        claim_id,
                    )
                )

            for patent_id, positions in (
                patent_claim_positions.items()
            ):
                if patent_id in seen_patent_ids:
                    raise RuntimeError(
                        "A patent appears in multiple inference "
                        f"shards: {patent_id}"
                    )

                seen_patent_ids.add(
                    patent_id
                )

                if patent_id not in record_lookup:
                    raise KeyError(
                        f"No processed record for patent "
                        f"{patent_id} in split={split_name}."
                    )

                positions.sort(
                    key=lambda item: item[1]
                )

                tensor_positions = [
                    int(item[0])
                    for item in positions
                ]

                patent_claim_ids = [
                    int(item[1])
                    for item in positions
                ]

                patent_theta = theta[
                    tensor_positions
                ]

                patent_depth = claim_depth[
                    tensor_positions
                ]

                patent_bow_valid = bow_valid[
                    tensor_positions
                ]

                record = record_lookup[
                    patent_id
                ]

                patent_payload = (
                    build_patent_payload(
                        split_name=split_name,
                        patent_id=patent_id,
                        record=record,
                        claim_ids=(
                            patent_claim_ids
                        ),
                        theta_rows=(
                            patent_theta
                        ),
                        claim_depths=(
                            patent_depth
                        ),
                        bow_valid_flags=(
                            patent_bow_valid
                        ),
                    )
                )

                sanitized_payload = (
                    sanitize_json_value(
                        patent_payload
                    )
                )

                output_file.write(
                    json.dumps(
                        sanitized_payload,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

                number_of_patents += 1
                number_of_claims += int(
                    patent_payload[
                        "number_of_claims"
                    ]
                )
                number_of_edges += int(
                    patent_payload[
                        "number_of_dependency_edges"
                    ]
                )
                number_of_adjacent_edges += int(
                    patent_payload[
                        "number_of_adjacent_edges"
                    ]
                )
                number_of_non_adjacent_edges += int(
                    patent_payload[
                        "number_of_non_adjacent_edges"
                    ]
                )
                number_of_empty_bow_claims += int(
                    patent_payload[
                        "empty_bow_claims"
                    ]
                )

                for edge in patent_payload[
                    "dependency_edges"
                ]:
                    if edge[
                        "direction_consistent"
                    ]:
                        number_of_directional_edges += 1

                    hierarchy_support_sum += float(
                        edge[
                            "global_hierarchy_support"
                        ]
                    )
                    hierarchy_support_count += 1

                    directional_support_sum += float(
                        edge[
                            "directional_transition_support"
                        ]
                    )
                    directional_support_count += 1

                if (
                    patent_payload[
                        "number_of_dependency_edges"
                    ]
                    > 0
                ):
                    directional_ratio = float(
                        patent_payload[
                            "directional_edge_ratio"
                        ]
                    )

                    hierarchy_support = float(
                        patent_payload[
                            "mean_global_hierarchy_support"
                        ]
                    )

                    directional_support = float(
                        patent_payload[
                            "mean_directional_transition_support"
                        ]
                    )

                    maximum_depth = float(
                        patent_payload[
                            "maximum_depth"
                        ]
                    )

                    # Representative examples should contain
                    # multiple claims and meaningful hierarchy evidence.
                    qualitative_score = (
                        0.40
                        * directional_ratio
                        + 0.35
                        * hierarchy_support
                        + 0.20
                        * directional_support
                        + 0.05
                        * min(
                            maximum_depth / 5.0,
                            1.0,
                        )
                    )

                    qualitative_counter += 1

                    heap_item = (
                        qualitative_score,
                        qualitative_counter,
                        patent_payload,
                    )

                    if (
                        len(qualitative_heap)
                        < QUALITATIVE_PATENTS_PER_SPLIT
                    ):
                        heapq.heappush(
                            qualitative_heap,
                            heap_item,
                        )
                    elif (
                        qualitative_score
                        > qualitative_heap[0][0]
                    ):
                        heapq.heapreplace(
                            qualitative_heap,
                            heap_item,
                        )

            del shard
            del theta
            del claim_depth
            del bow_valid

    os.replace(
        temporary_jsonl_path,
        final_jsonl_path,
    )

    expected_patents = int(
        split_manifest[
            "number_of_patents"
        ]
    )

    expected_claims = int(
        split_manifest[
            "number_of_claims"
        ]
    )

    if number_of_patents != expected_patents:
        raise RuntimeError(
            f"{split_name} patent export count mismatch: "
            f"exported={number_of_patents}, "
            f"expected={expected_patents}."
        )

    if number_of_claims != expected_claims:
        raise RuntimeError(
            f"{split_name} claim export count mismatch: "
            f"exported={number_of_claims}, "
            f"expected={expected_claims}."
        )

    qualitative_heap.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    qualitative_examples = []

    for rank, (
        score,
        _,
        patent_payload,
    ) in enumerate(
        qualitative_heap,
        start=1,
    ):
        patent_id = (
            patent_payload[
                "patent_id"
            ]
        )

        record = record_lookup[
            patent_id
        ]

        qualitative_payload = (
            add_claim_text_snippets(
                patent_payload=(
                    patent_payload
                ),
                record=record,
            )
        )

        qualitative_payload[
            "qualitative_rank"
        ] = rank

        qualitative_payload[
            "qualitative_selection_score"
        ] = float(score)

        qualitative_examples.append(
            qualitative_payload
        )

    qualitative_path = os.path.join(
        QUALITATIVE_DIR,
        f"{split_name}_qualitative_examples.json",
    )

    atomic_json_save_section9(
        {
            "split": split_name,
            "selection_method": (
                "weighted directional consistency, global "
                "hierarchy support, directional transition "
                "support, and maximum claim depth"
            ),
            "number_of_examples": len(
                qualitative_examples
            ),
            "examples": (
                qualitative_examples
            ),
        },
        qualitative_path,
    )

    directional_edge_ratio = (
        number_of_directional_edges
        / max(
            number_of_edges,
            1,
        )
    )

    adjacent_edge_coverage = (
        number_of_adjacent_edges
        / max(
            number_of_edges,
            1,
        )
    )

    mean_hierarchy_support = (
        hierarchy_support_sum
        / hierarchy_support_count
        if hierarchy_support_count > 0
        else None
    )

    mean_directional_support = (
        directional_support_sum
        / directional_support_count
        if directional_support_count > 0
        else None
    )

    return {
        "split": split_name,
        "number_of_patents": (
            number_of_patents
        ),
        "number_of_claims": (
            number_of_claims
        ),
        "number_of_dependency_edges": (
            number_of_edges
        ),
        "number_of_adjacent_edges": (
            number_of_adjacent_edges
        ),
        "number_of_non_adjacent_edges": (
            number_of_non_adjacent_edges
        ),
        "adjacent_edge_coverage": (
            adjacent_edge_coverage
        ),
        "number_of_directional_edges": (
            number_of_directional_edges
        ),
        "directional_edge_ratio": (
            directional_edge_ratio
        ),
        "empty_bow_claims": (
            number_of_empty_bow_claims
        ),
        "empty_bow_ratio": (
            number_of_empty_bow_claims
            / max(
                number_of_claims,
                1,
            )
        ),
        "mean_global_hierarchy_support": (
            mean_hierarchy_support
        ),
        "mean_directional_transition_support": (
            mean_directional_support
        ),
        "patent_hierarchy_jsonl": (
            final_jsonl_path
        ),
        "qualitative_examples": (
            qualitative_path
        ),
    }


# ============================================================
# 12. Export train/dev/test patent hierarchies
# ============================================================

split_export_summaries = {}

for split_name in [
    "train",
    "dev",
    "test",
]:
    split_export_summaries[
        split_name
    ] = export_split_patent_hierarchies(
        split_name
    )

print("\n=== PATENT EXPORT SUMMARY ===")

for split_name, summary in (
    split_export_summaries.items()
):
    print(
        f"{split_name:5s}: "
        f"patents={summary['number_of_patents']:,}, "
        f"claims={summary['number_of_claims']:,}, "
        f"edges={summary['number_of_dependency_edges']:,}, "
        f"directional={summary['directional_edge_ratio']:.4%}, "
        f"adjacent={summary['adjacent_edge_coverage']:.4%}"
    )


# ============================================================
# 13. Publication table: topic summary
# ============================================================

per_topic_evaluation = {
    int(record["topic_id"]): record
    for record in section8_report[
        "per_topic"
    ]
}

topic_table_path = os.path.join(
    TABLE_DIR,
    "table_topic_summary.csv",
)

with open(
    topic_table_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = [
        "topic_id",
        "topic_number",
        "anchor_coordinate",
        "train_topic_usage",
        "npmi",
        "exclusivity",
        "normalized_entropy",
        "maximum_cosine_similarity",
        "role",
        "incoming_edges",
        "outgoing_edges",
        "top_words",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    node_lookup = {
        int(node["topic_id"]): node
        for node in global_hierarchy[
            "nodes"
        ]
    }

    for topic_id in range(
        NUM_TOPICS
    ):
        evaluation = (
            per_topic_evaluation.get(
                topic_id,
                {},
            )
        )

        node = node_lookup.get(
            topic_id,
            {},
        )

        writer.writerow(
            {
                "topic_id": topic_id,
                "topic_number": topic_id + 1,
                "anchor_coordinate": float(
                    anchor_coordinates[
                        topic_id
                    ].item()
                ),
                "train_topic_usage": float(
                    topic_usage[
                        topic_id
                    ].item()
                ),
                "npmi": evaluation.get(
                    "npmi"
                ),
                "exclusivity": evaluation.get(
                    "exclusivity"
                ),
                "normalized_entropy": (
                    evaluation.get(
                        "normalized_entropy"
                    )
                ),
                "maximum_cosine_similarity": (
                    evaluation.get(
                        "maximum_cosine_similarity"
                    )
                ),
                "role": node.get(
                    "role",
                    "unknown",
                ),
                "incoming_edges": int(
                    node.get(
                        "incoming_edges",
                        0,
                    )
                ),
                "outgoing_edges": int(
                    node.get(
                        "outgoing_edges",
                        0,
                    )
                ),
                "top_words": ", ".join(
                    item["word"]
                    for item in
                    topic_summaries[
                        topic_id
                    ]["top_words"]
                ),
            }
        )


# ============================================================
# 14. Publication table: global hierarchy edges
# ============================================================

hierarchy_edge_table_path = os.path.join(
    TABLE_DIR,
    "table_global_hierarchy_edges.csv",
)

with open(
    hierarchy_edge_table_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = [
        "parent_topic_id",
        "parent_topic_number",
        "child_topic_id",
        "child_topic_number",
        "rank_for_parent",
        "parent_anchor",
        "child_anchor",
        "anchor_gap",
        "joint_probability",
        "conditional_probability",
        "anchor_cost",
        "parent_top_words",
        "child_top_words",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    sorted_hierarchy_edges = sorted(
        global_hierarchy[
            "edges"
        ],
        key=lambda edge: (
            int(
                edge[
                    "parent_topic_id"
                ]
            ),
            int(
                edge[
                    "rank_for_parent"
                ]
            ),
        ),
    )

    for edge in sorted_hierarchy_edges:
        writer.writerow(
            {
                "parent_topic_id": edge[
                    "parent_topic_id"
                ],
                "parent_topic_number": edge[
                    "parent_topic_number"
                ],
                "child_topic_id": edge[
                    "child_topic_id"
                ],
                "child_topic_number": edge[
                    "child_topic_number"
                ],
                "rank_for_parent": edge[
                    "rank_for_parent"
                ],
                "parent_anchor": edge[
                    "parent_anchor_coordinate"
                ],
                "child_anchor": edge[
                    "child_anchor_coordinate"
                ],
                "anchor_gap": edge[
                    "anchor_gap"
                ],
                "joint_probability": edge[
                    "joint_probability"
                ],
                "conditional_probability": edge[
                    "conditional_probability"
                ],
                "anchor_cost": edge[
                    "anchor_cost"
                ],
                "parent_top_words": ", ".join(
                    edge[
                        "parent_top_words"
                    ]
                ),
                "child_top_words": ", ".join(
                    edge[
                        "child_top_words"
                    ]
                ),
            }
        )


# ============================================================
# 15. Publication table: split evaluation
# ============================================================

split_table_path = os.path.join(
    TABLE_DIR,
    "table_split_hierarchy_metrics.csv",
)

with open(
    split_table_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = [
        "split",
        "number_of_patents",
        "number_of_claims",
        "number_of_dependency_edges",
        "adjacent_edge_coverage",
        "directional_edge_ratio",
        "empty_bow_ratio",
        "mean_global_hierarchy_support",
        "mean_directional_transition_support",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for split_name in [
        "train",
        "dev",
        "test",
    ]:
        summary = (
            split_export_summaries[
                split_name
            ]
        )

        writer.writerow(
            {
                field_name: summary.get(
                    field_name
                )
                for field_name in fieldnames
            }
        )


# ============================================================
# 16. Publication table: overall model metrics
# ============================================================

overall_metrics_path = os.path.join(
    TABLE_DIR,
    "table_overall_model_metrics.csv",
)

topic_quality = section8_report[
    "topic_quality"
]

hierarchy_quality = section8_report[
    "hierarchy_quality"
]

depth_anchor_alignment = (
    section8_report[
        "depth_anchor_alignment"
    ]
)

overall_metric_rows = [
    (
        "Number of topics",
        NUM_TOPICS,
    ),
    (
        "Vocabulary size",
        VOCAB_SIZE,
    ),
    (
        "Topic diversity",
        topic_quality.get(
            "topic_diversity"
        ),
    ),
    (
        "Mean NPMI",
        topic_quality.get(
            "mean_npmi"
        ),
    ),
    (
        "Median NPMI",
        topic_quality.get(
            "median_npmi"
        ),
    ),
    (
        "Mean exclusivity",
        topic_quality.get(
            "mean_exclusivity"
        ),
    ),
    (
        "Mean normalized entropy",
        topic_quality.get(
            "mean_normalized_entropy"
        ),
    ),
    (
        "Mean topic cosine similarity",
        topic_quality.get(
            "mean_topic_cosine_similarity"
        ),
    ),
    (
        "Maximum topic cosine similarity",
        topic_quality.get(
            "maximum_topic_cosine_similarity"
        ),
    ),
    (
        "Global hierarchy nodes",
        hierarchy_quality.get(
            "number_of_nodes"
        ),
    ),
    (
        "Global hierarchy edges",
        hierarchy_quality.get(
            "number_of_edges"
        ),
    ),
    (
        "Directional empirical mass ratio",
        hierarchy_quality.get(
            "directional_empirical_mass_ratio"
        ),
    ),
    (
        "Selected hierarchy mass ratio",
        hierarchy_quality.get(
            "selected_edge_joint_mass_ratio"
        ),
    ),
    (
        "Depth-anchor Pearson correlation",
        depth_anchor_alignment.get(
            "pearson_correlation"
        ),
    ),
    (
        "Depth-anchor Spearman correlation",
        depth_anchor_alignment.get(
            "spearman_correlation"
        ),
    ),
    (
        "Depth monotonic transition ratio",
        depth_anchor_alignment.get(
            "monotonic_transition_ratio"
        ),
    ),
]

with open(
    overall_metrics_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "metric",
            "value",
        ],
    )

    writer.writeheader()

    for metric_name, metric_value in (
        overall_metric_rows
    ):
        writer.writerow(
            {
                "metric": metric_name,
                "value": metric_value,
            }
        )


# ============================================================
# 17. Save compact publication summary
# ============================================================

publication_summary = {
    "run_name": global_hierarchy.get(
        "run_name"
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "model": {
        "number_of_topics": (
            NUM_TOPICS
        ),
        "vocabulary_size": (
            VOCAB_SIZE
        ),
        "best_epoch": (
            global_hierarchy.get(
                "best_epoch"
            )
        ),
    },
    "topic_quality": (
        topic_quality
    ),
    "hierarchy_quality": (
        hierarchy_quality
    ),
    "depth_anchor_alignment": (
        depth_anchor_alignment
    ),
    "split_hierarchy_metrics": (
        split_export_summaries
    ),
    "important_interpretation": {
        "global_hierarchy_training_split": (
            "train"
        ),
        "test_used_for_hierarchy_construction": (
            False
        ),
        "test_used_for_model_selection": (
            False
        ),
        "empty_bow_claim_policy": (
            "Claims with zero fixed-vocabulary terms remain "
            "in graph and hierarchy objectives but are excluded "
            "from reconstruction loss."
        ),
        "non_adjacent_edge_policy": (
            "Retained in patent export; excluded from adjacent-depth "
            "OT transition estimation."
        ),
        "global_structure_type": (
            "directed acyclic graph"
        ),
    },
}

publication_summary_path = os.path.join(
    SECTION9_RESULT_DIR,
    "publication_summary.json",
)

atomic_json_save_section9(
    publication_summary,
    publication_summary_path,
)


# ============================================================
# 18. Final manifest
# ============================================================

section9_manifest = {
    "run_name": global_hierarchy.get(
        "run_name"
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "number_of_topics": (
        NUM_TOPICS
    ),
    "claim_top_k_topics": (
        CLAIM_TOP_K_TOPICS
    ),
    "export_full_theta": (
        EXPORT_FULL_THETA
    ),
    "splits": (
        split_export_summaries
    ),
    "files": {
        "publication_summary": (
            publication_summary_path
        ),
        "topic_summary_table": (
            topic_table_path
        ),
        "hierarchy_edge_table": (
            hierarchy_edge_table_path
        ),
        "split_metric_table": (
            split_table_path
        ),
        "overall_metric_table": (
            overall_metrics_path
        ),
        "patent_hierarchy_directory": (
            PATENT_EXPORT_DIR
        ),
        "qualitative_example_directory": (
            QUALITATIVE_DIR
        ),
    },
}

section9_manifest_path = os.path.join(
    SECTION9_RESULT_DIR,
    "section9_manifest.json",
)

atomic_json_save_section9(
    section9_manifest,
    section9_manifest_path,
)


# ============================================================
# 19. Final validation
# ============================================================

required_output_files = [
    publication_summary_path,
    topic_table_path,
    hierarchy_edge_table_path,
    split_table_path,
    overall_metrics_path,
    section9_manifest_path,
]

for split_name in [
    "train",
    "dev",
    "test",
]:
    required_output_files.append(
        split_export_summaries[
            split_name
        ][
            "patent_hierarchy_jsonl"
        ]
    )

    required_output_files.append(
        split_export_summaries[
            split_name
        ][
            "qualitative_examples"
        ]
    )

missing_output_files = [
    path
    for path in required_output_files
    if not os.path.isfile(path)
]

if missing_output_files:
    raise FileNotFoundError(
        "Section 9 output files are missing: "
        f"{missing_output_files}"
    )

for split_name, summary in (
    split_export_summaries.items()
):
    if summary[
        "number_of_patents"
    ] <= 0:
        raise RuntimeError(
            f"No patents exported for split={split_name}."
        )

    if summary[
        "number_of_claims"
    ] <= 0:
        raise RuntimeError(
            f"No claims exported for split={split_name}."
        )


# ============================================================
# 20. Final status
# ============================================================

print("\n" + "=" * 72)
print("SECTION 9 COMPLETED SUCCESSFULLY")
print("=" * 72)
print(f"Feature run            : {FEATURE_RUN_NAME}")
print(f"Topics                 : {NUM_TOPICS}")
print(f"Vocabulary             : {VOCAB_SIZE:,}")

for split_name in [
    "train",
    "dev",
    "test",
]:
    summary = (
        split_export_summaries[
            split_name
        ]
    )

    print(
        f"{split_name:5s} patents        : "
        f"{summary['number_of_patents']:,}"
    )
    print(
        f"{split_name:5s} claims         : "
        f"{summary['number_of_claims']:,}"
    )
    print(
        f"{split_name:5s} directional    : "
        f"{summary['directional_edge_ratio']:.4%}"
    )
    print(
        f"{split_name:5s} hierarchy sup. : "
        f"{summary['mean_global_hierarchy_support']}"
    )

print(f"Patent exports         : {PATENT_EXPORT_DIR}")
print(f"Qualitative examples   : {QUALITATIVE_DIR}")
print(f"Publication tables     : {TABLE_DIR}")
print(f"Publication summary    : {publication_summary_path}")
print(f"Section 9 manifest     : {section9_manifest_path}")
print("=" * 72)

if FEATURE_RUN_NAME.startswith(
    "debug"
):
    print(
        "\n[IMPORTANT] These patent examples and tables come from "
        "a debug run and must not be used as final results."
    )
else:
    print(
        "\n[PASS] Full patent-level hierarchy export and "
        "publication-table generation completed."
    )

print(
    "\nAll Section 9 outputs are saved to Google Drive. "
    "The experimental pipeline is now complete."
)
# ============================================================
# SECTION 9:
# Patent-Level Hierarchy Export, Qualitative Examples,
# and Publication-Ready Tables
# COMPLETE SINGLE-CELL VERSION
# ============================================================

import os
import csv
import json
import math
import heapq
from collections import defaultdict
from datetime import datetime

import torch
from tqdm.auto import tqdm


# ============================================================
# 0. Preconditions
# ============================================================

required_section9_globals = [
    "CONFIG",
    "FEATURE_RUN_NAME",
    "VOCAB",
    "train_dataset",
    "dev_dataset",
    "test_dataset",
    "SECTION7_RESULT_DIR",
    "SECTION8_RESULT_DIR",
]

missing_section9_globals = [
    name
    for name in required_section9_globals
    if name not in globals()
]

if missing_section9_globals:
    raise RuntimeError(
        "Section 9 prerequisites are missing: "
        f"{missing_section9_globals}. "
        "Run Sections 0 through 8 first."
    )

GLOBAL_HIERARCHY_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "global_topic_hierarchy.json",
)

GLOBAL_MATRIX_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "global_transition_matrices.pt",
)

TOPIC_WORD_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "topic_word_distribution.pt",
)

INFERENCE_MANIFEST_PATH = os.path.join(
    SECTION7_RESULT_DIR,
    "inference_manifest.json",
)

SECTION8_REPORT_PATH = os.path.join(
    SECTION8_RESULT_DIR,
    "section8_evaluation_report.json",
)

required_input_files = [
    GLOBAL_HIERARCHY_PATH,
    GLOBAL_MATRIX_PATH,
    TOPIC_WORD_PATH,
    INFERENCE_MANIFEST_PATH,
    SECTION8_REPORT_PATH,
]

missing_input_files = [
    path
    for path in required_input_files
    if not os.path.isfile(path)
]

if missing_input_files:
    raise FileNotFoundError(
        "Required Section 7/8 output files are missing: "
        f"{missing_input_files}"
    )


# ============================================================
# 1. Configuration
# ============================================================

SECTION9_RESULT_DIR = os.path.join(
    SECTION7_RESULT_DIR,
    "section9_patent_exports",
)

PATENT_EXPORT_DIR = os.path.join(
    SECTION9_RESULT_DIR,
    "patent_hierarchies",
)

TABLE_DIR = os.path.join(
    SECTION9_RESULT_DIR,
    "tables",
)

QUALITATIVE_DIR = os.path.join(
    SECTION9_RESULT_DIR,
    "qualitative_examples",
)

for directory in [
    SECTION9_RESULT_DIR,
    PATENT_EXPORT_DIR,
    TABLE_DIR,
    QUALITATIVE_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )

CLAIM_TOP_K_TOPICS = int(
    getattr(
        CONFIG,
        "claim_top_k_topics",
        3,
    )
)

TOP_WORDS_PER_TOPIC_EXPORT = int(
    getattr(
        CONFIG,
        "export_top_words_per_topic",
        10,
    )
)

QUALITATIVE_PATENTS_PER_SPLIT = int(
    getattr(
        CONFIG,
        "qualitative_patents_per_split",
        5,
    )
)

CLAIM_TEXT_SNIPPET_LENGTH = int(
    getattr(
        CONFIG,
        "claim_text_snippet_length",
        500,
    )
)

EXPORT_FULL_THETA = bool(
    getattr(
        CONFIG,
        "export_full_theta",
        False,
    )
)

if CLAIM_TOP_K_TOPICS < 1:
    raise ValueError(
        "CLAIM_TOP_K_TOPICS must be positive."
    )

if TOP_WORDS_PER_TOPIC_EXPORT < 1:
    raise ValueError(
        "TOP_WORDS_PER_TOPIC_EXPORT must be positive."
    )

if QUALITATIVE_PATENTS_PER_SPLIT < 1:
    raise ValueError(
        "QUALITATIVE_PATENTS_PER_SPLIT must be positive."
    )

if CLAIM_TEXT_SNIPPET_LENGTH < 1:
    raise ValueError(
        "CLAIM_TEXT_SNIPPET_LENGTH must be positive."
    )

print("=== SECTION 9 CONFIGURATION ===")
print(f"Feature run                    : {FEATURE_RUN_NAME}")
print(f"Result directory               : {SECTION9_RESULT_DIR}")
print(f"Claim top-k topics             : {CLAIM_TOP_K_TOPICS}")
print(f"Topic words per export         : {TOP_WORDS_PER_TOPIC_EXPORT}")
print(f"Qualitative patents per split  : {QUALITATIVE_PATENTS_PER_SPLIT}")
print(f"Claim text snippet length      : {CLAIM_TEXT_SNIPPET_LENGTH}")
print(f"Export full theta              : {EXPORT_FULL_THETA}")


# ============================================================
# 2. Serialization utilities
# ============================================================

def atomic_json_save_section9(
    data,
    final_path,
):
    os.makedirs(
        os.path.dirname(final_path),
        exist_ok=True,
    )

    temporary_path = final_path + ".tmp"

    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    with open(
        temporary_path,
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
        temporary_path,
        final_path,
    )


def sanitize_json_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            value = value.item()
        else:
            return [
                sanitize_json_value(item)
                for item in value.tolist()
            ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return float(value)

    if isinstance(value, dict):
        return {
            str(key): sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            sanitize_json_value(item)
            for item in value
        ]

    return value


def normalize_patent_id_section9(
    patent_id,
):
    normalized = str(
        patent_id
    ).strip()

    if not normalized:
        raise ValueError(
            "Empty patent ID detected."
        )

    return normalized


def normalize_claim_id_section9(
    claim_id,
):
    try:
        normalized = int(
            claim_id
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid claim ID: {claim_id}"
        ) from error

    if normalized < 1:
        raise ValueError(
            f"Claim ID must be positive, got {normalized}."
        )

    return normalized


# ============================================================
# 3. Load Section 7 and Section 8 outputs
# ============================================================

with open(
    GLOBAL_HIERARCHY_PATH,
    "r",
    encoding="utf-8",
) as file:
    global_hierarchy = json.load(file)

with open(
    INFERENCE_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    inference_manifest = json.load(file)

with open(
    SECTION8_REPORT_PATH,
    "r",
    encoding="utf-8",
) as file:
    section8_report = json.load(file)

global_matrices = torch.load(
    GLOBAL_MATRIX_PATH,
    map_location="cpu",
    weights_only=False,
)

topic_word_payload = torch.load(
    TOPIC_WORD_PATH,
    map_location="cpu",
    weights_only=False,
)

beta = (
    topic_word_payload["beta"]
    .detach()
    .float()
    .cpu()
)

anchor_coordinates = (
    global_matrices[
        "anchor_coordinates"
    ]
    .detach()
    .float()
    .cpu()
)

directional_conditional = (
    global_matrices[
        "directional_conditional"
    ]
    .detach()
    .float()
    .cpu()
)

empirical_joint = (
    global_matrices[
        "empirical_joint"
    ]
    .detach()
    .float()
    .cpu()
)

topic_usage = (
    global_matrices[
        "topic_usage"
    ]
    .detach()
    .float()
    .cpu()
)

NUM_TOPICS = int(
    beta.shape[0]
)

VOCAB_SIZE = int(
    beta.shape[1]
)

if VOCAB_SIZE != len(VOCAB):
    raise ValueError(
        "Beta/vocabulary size mismatch."
    )

if anchor_coordinates.shape != (
    NUM_TOPICS,
):
    raise ValueError(
        "Anchor-coordinate shape mismatch."
    )

if directional_conditional.shape != (
    NUM_TOPICS,
    NUM_TOPICS,
):
    raise ValueError(
        "Directional conditional matrix shape mismatch."
    )

if empirical_joint.shape != (
    NUM_TOPICS,
    NUM_TOPICS,
):
    raise ValueError(
        "Empirical joint matrix shape mismatch."
    )

if topic_usage.shape != (
    NUM_TOPICS,
):
    raise ValueError(
        "Topic-usage shape mismatch."
    )

if not torch.isfinite(beta).all():
    raise FloatingPointError(
        "Non-finite beta values detected."
    )

if not torch.isfinite(
    anchor_coordinates
).all():
    raise FloatingPointError(
        "Non-finite anchor coordinates detected."
    )

if not torch.isfinite(
    directional_conditional
).all():
    raise FloatingPointError(
        "Non-finite directional conditional values detected."
    )

print("\n=== LOADED OUTPUTS ===")
print(f"Number of topics : {NUM_TOPICS}")
print(f"Vocabulary size  : {VOCAB_SIZE}")
print(f"Hierarchy nodes  : {len(global_hierarchy['nodes'])}")
print(f"Hierarchy edges  : {len(global_hierarchy['edges'])}")


# ============================================================
# 4. Topic summaries
# ============================================================

actual_topic_word_count = min(
    TOP_WORDS_PER_TOPIC_EXPORT,
    VOCAB_SIZE,
)

top_word_probabilities, top_word_indices = torch.topk(
    beta,
    k=actual_topic_word_count,
    dim=-1,
)

topic_summaries = {}

for topic_id in range(
    NUM_TOPICS
):
    top_words = []

    for rank in range(
        actual_topic_word_count
    ):
        vocabulary_index = int(
            top_word_indices[
                topic_id,
                rank,
            ].item()
        )

        top_words.append(
            {
                "rank": rank + 1,
                "word": VOCAB[
                    vocabulary_index
                ],
                "probability": float(
                    top_word_probabilities[
                        topic_id,
                        rank,
                    ].item()
                ),
            }
        )

    topic_summaries[
        topic_id
    ] = {
        "topic_id": topic_id,
        "topic_number": topic_id + 1,
        "anchor_coordinate": float(
            anchor_coordinates[
                topic_id
            ].item()
        ),
        "train_topic_usage": float(
            topic_usage[
                topic_id
            ].item()
        ),
        "top_words": top_words,
    }


# ============================================================
# 5. Global hierarchy masks
# ============================================================

selected_hierarchy_mask = torch.zeros(
    (
        NUM_TOPICS,
        NUM_TOPICS,
    ),
    dtype=torch.float32,
)

global_edge_lookup = {}

for edge in global_hierarchy[
    "edges"
]:
    parent_topic = int(
        edge[
            "parent_topic_id"
        ]
    )

    child_topic = int(
        edge[
            "child_topic_id"
        ]
    )

    if not (
        0 <= parent_topic < NUM_TOPICS
        and 0 <= child_topic < NUM_TOPICS
    ):
        raise IndexError(
            "Global hierarchy contains an invalid topic ID."
        )

    if not (
        anchor_coordinates[
            child_topic
        ]
        > anchor_coordinates[
            parent_topic
        ]
    ):
        raise ValueError(
            "Global hierarchy contains an anchor-direction violation."
        )

    selected_hierarchy_mask[
        parent_topic,
        child_topic,
    ] = 1.0

    global_edge_lookup[
        (
            parent_topic,
            child_topic,
        )
    ] = edge


# ============================================================
# 6. Processed-record lookup tables
# ============================================================

def build_record_lookup(
    dataset,
    split_name,
):
    lookup = {}

    for record in tqdm(
        dataset.records,
        desc=f"{split_name} record lookup",
    ):
        patent_id = (
            normalize_patent_id_section9(
                record["patent_id"]
            )
        )

        if patent_id in lookup:
            raise ValueError(
                f"Duplicate patent ID in {split_name}: "
                f"{patent_id}"
            )

        lookup[
            patent_id
        ] = record

    if not lookup:
        raise RuntimeError(
            f"No records found for split={split_name}."
        )

    return lookup


DATASETS_BY_SPLIT = {
    "train": train_dataset,
    "dev": dev_dataset,
    "test": test_dataset,
}

RECORD_LOOKUPS = {}

for split_name, dataset in (
    DATASETS_BY_SPLIT.items()
):
    RECORD_LOOKUPS[
        split_name
    ] = build_record_lookup(
        dataset=dataset,
        split_name=split_name,
    )


# ============================================================
# 7. Inference-shard resolution
# ============================================================

def resolve_inference_shard_path(
    shard_record,
):
    candidate_path = (
        shard_record.get(
            "path"
        )
    )

    if (
        candidate_path is not None
        and os.path.isfile(
            candidate_path
        )
    ):
        return candidate_path

    shard_filename = (
        shard_record.get(
            "file"
        )
    )

    if not shard_filename:
        raise KeyError(
            "Inference shard record has no file name."
        )

    fallback_path = os.path.join(
        SECTION7_RESULT_DIR,
        "claim_theta_shards",
        shard_filename,
    )

    if not os.path.isfile(
        fallback_path
    ):
        raise FileNotFoundError(
            "Inference shard not found: "
            f"{fallback_path}"
        )

    return fallback_path


# ============================================================
# 8. Claim-topic summary helper
# ============================================================

def create_claim_topic_summary(
    theta_row,
):
    if theta_row.ndim != 1:
        raise ValueError(
            "theta_row must be one-dimensional."
        )

    if theta_row.shape[0] != NUM_TOPICS:
        raise ValueError(
            "theta_row topic-count mismatch."
        )

    if not torch.isfinite(
        theta_row
    ).all():
        raise FloatingPointError(
            "Non-finite theta detected."
        )

    if torch.any(theta_row < 0):
        raise ValueError(
            "Negative theta detected."
        )

    theta_sum = float(
        theta_row.sum().item()
    )

    if not math.isclose(
        theta_sum,
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError(
            f"Theta does not sum to one: {theta_sum:.8f}"
        )

    actual_top_k = min(
        CLAIM_TOP_K_TOPICS,
        NUM_TOPICS,
    )

    top_probabilities, top_indices = torch.topk(
        theta_row,
        k=actual_top_k,
    )

    dominant_topic = int(
        top_indices[0].item()
    )

    expected_anchor = float(
        torch.dot(
            theta_row,
            anchor_coordinates,
        ).item()
    )

    top_topics = []

    for rank in range(
        actual_top_k
    ):
        topic_id = int(
            top_indices[
                rank
            ].item()
        )

        top_topics.append(
            {
                "rank": rank + 1,
                "topic_id": topic_id,
                "topic_number": (
                    topic_id + 1
                ),
                "probability": float(
                    top_probabilities[
                        rank
                    ].item()
                ),
                "anchor_coordinate": float(
                    anchor_coordinates[
                        topic_id
                    ].item()
                ),
                "top_words": [
                    item["word"]
                    for item in
                    topic_summaries[
                        topic_id
                    ]["top_words"][:5]
                ],
            }
        )

    result = {
        "dominant_topic_id": (
            dominant_topic
        ),
        "dominant_topic_number": (
            dominant_topic + 1
        ),
        "dominant_topic_probability": float(
            theta_row[
                dominant_topic
            ].item()
        ),
        "expected_anchor_coordinate": (
            expected_anchor
        ),
        "top_topics": top_topics,
    }

    if EXPORT_FULL_THETA:
        result["theta"] = [
            float(value)
            for value in
            theta_row.tolist()
        ]

    return result


# ============================================================
# 9. Patent payload construction
# ============================================================

def build_patent_payload(
    split_name,
    patent_id,
    record,
    claim_ids,
    theta_rows,
    claim_depths,
    bow_valid_flags,
):
    patent_id = (
        normalize_patent_id_section9(
            patent_id
        )
    )

    claim_ids = [
        normalize_claim_id_section9(
            claim_id
        )
        for claim_id in claim_ids
    ]

    if theta_rows.ndim != 2:
        raise ValueError(
            "theta_rows must have shape [C, K]."
        )

    number_of_claims = len(
        claim_ids
    )

    if theta_rows.shape != (
        number_of_claims,
        NUM_TOPICS,
    ):
        raise ValueError(
            "Patent theta shape mismatch: "
            f"observed={tuple(theta_rows.shape)}, "
            f"expected={(number_of_claims, NUM_TOPICS)}."
        )

    if claim_depths.shape != (
        number_of_claims,
    ):
        raise ValueError(
            "Patent claim-depth shape mismatch."
        )

    if bow_valid_flags.shape != (
        number_of_claims,
    ):
        raise ValueError(
            "Patent BoW-valid shape mismatch."
        )

    if len(set(claim_ids)) != number_of_claims:
        raise ValueError(
            f"Duplicate claim IDs in patent {patent_id}."
        )

    local_claim_index = {
        claim_id: position
        for position, claim_id
        in enumerate(claim_ids)
    }

    record_claims = {
        normalize_claim_id_section9(
            claim_id
        ): text
        for claim_id, text
        in record["claims"].items()
    }

    record_depth = {
        normalize_claim_id_section9(
            claim_id
        ): int(depth)
        for claim_id, depth
        in record["depth"].items()
    }

    if set(claim_ids) != set(
        record_claims.keys()
    ):
        raise ValueError(
            f"Inference/record claim mismatch for patent "
            f"{patent_id}."
        )

    claim_payloads = []
    claim_summary_lookup = {}

    for position, claim_id in enumerate(
        claim_ids
    ):
        inferred_depth = int(
            claim_depths[
                position
            ].item()
        )

        stored_depth = int(
            record_depth[
                claim_id
            ]
        )

        if inferred_depth != stored_depth:
            raise ValueError(
                f"Depth mismatch for patent {patent_id}, "
                f"claim {claim_id}: "
                f"inference={inferred_depth}, "
                f"record={stored_depth}."
            )

        topic_summary = (
            create_claim_topic_summary(
                theta_rows[
                    position
                ]
            )
        )

        claim_payload = {
            "claim_id": claim_id,
            "depth": inferred_depth,
            "bow_valid": bool(
                bow_valid_flags[
                    position
                ].item()
            ),
            **topic_summary,
        }

        claim_payloads.append(
            claim_payload
        )

        claim_summary_lookup[
            claim_id
        ] = {
            "theta": theta_rows[
                position
            ],
            "summary": claim_payload,
        }

    dependency_edges = []

    number_of_adjacent_edges = 0
    number_of_non_adjacent_edges = 0
    number_of_directional_edges = 0
    number_of_dominant_global_edges = 0

    hierarchy_support_values = []
    directional_support_values = []
    expected_anchor_gaps = []

    for parent_id, child_id in record[
        "edges"
    ]:
        parent_id = (
            normalize_claim_id_section9(
                parent_id
            )
        )

        child_id = (
            normalize_claim_id_section9(
                child_id
            )
        )

        if parent_id not in local_claim_index:
            raise KeyError(
                f"Unknown parent claim {parent_id} "
                f"in patent {patent_id}."
            )

        if child_id not in local_claim_index:
            raise KeyError(
                f"Unknown child claim {child_id} "
                f"in patent {patent_id}."
            )

        parent_payload = (
            claim_summary_lookup[
                parent_id
            ]
        )

        child_payload = (
            claim_summary_lookup[
                child_id
            ]
        )

        parent_theta = (
            parent_payload[
                "theta"
            ]
        )

        child_theta = (
            child_payload[
                "theta"
            ]
        )

        parent_depth = int(
            parent_payload[
                "summary"
            ]["depth"]
        )

        child_depth = int(
            child_payload[
                "summary"
            ]["depth"]
        )

        depth_difference = (
            child_depth
            - parent_depth
        )

        if depth_difference <= 0:
            raise ValueError(
                f"Dependency edge does not increase depth: "
                f"patent={patent_id}, "
                f"edge={parent_id}->{child_id}."
            )

        adjacent_depth = (
            depth_difference == 1
        )

        if adjacent_depth:
            number_of_adjacent_edges += 1
        else:
            number_of_non_adjacent_edges += 1

        parent_expected_anchor = float(
            parent_payload[
                "summary"
            ][
                "expected_anchor_coordinate"
            ]
        )

        child_expected_anchor = float(
            child_payload[
                "summary"
            ][
                "expected_anchor_coordinate"
            ]
        )

        expected_anchor_gap = (
            child_expected_anchor
            - parent_expected_anchor
        )

        direction_consistent = (
            expected_anchor_gap > 0.0
        )

        if direction_consistent:
            number_of_directional_edges += 1

        hierarchy_support = float(
            torch.dot(
                parent_theta,
                torch.mv(
                    selected_hierarchy_mask,
                    child_theta,
                ),
            ).item()
        )

        directional_support = float(
            torch.dot(
                parent_theta,
                torch.mv(
                    directional_conditional,
                    child_theta,
                ),
            ).item()
        )

        parent_dominant_topic = int(
            parent_payload[
                "summary"
            ][
                "dominant_topic_id"
            ]
        )

        child_dominant_topic = int(
            child_payload[
                "summary"
            ][
                "dominant_topic_id"
            ]
        )

        dominant_global_edge = (
            parent_dominant_topic,
            child_dominant_topic,
        ) in global_edge_lookup

        if dominant_global_edge:
            number_of_dominant_global_edges += 1

        hierarchy_support_values.append(
            hierarchy_support
        )

        directional_support_values.append(
            directional_support
        )

        expected_anchor_gaps.append(
            expected_anchor_gap
        )

        dependency_edges.append(
            {
                "parent_claim_id": parent_id,
                "child_claim_id": child_id,
                "parent_depth": parent_depth,
                "child_depth": child_depth,
                "depth_difference": depth_difference,
                "adjacent_depth": adjacent_depth,
                "parent_dominant_topic_id": (
                    parent_dominant_topic
                ),
                "parent_dominant_topic_number": (
                    parent_dominant_topic + 1
                ),
                "child_dominant_topic_id": (
                    child_dominant_topic
                ),
                "child_dominant_topic_number": (
                    child_dominant_topic + 1
                ),
                "dominant_topics_form_global_edge": (
                    dominant_global_edge
                ),
                "parent_expected_anchor": (
                    parent_expected_anchor
                ),
                "child_expected_anchor": (
                    child_expected_anchor
                ),
                "expected_anchor_gap": (
                    expected_anchor_gap
                ),
                "direction_consistent": (
                    direction_consistent
                ),
                "global_hierarchy_support": (
                    hierarchy_support
                ),
                "directional_transition_support": (
                    directional_support
                ),
            }
        )

    number_of_edges = len(
        dependency_edges
    )

    if number_of_edges > 0:
        directional_edge_ratio = (
            number_of_directional_edges
            / number_of_edges
        )

        dominant_global_edge_ratio = (
            number_of_dominant_global_edges
            / number_of_edges
        )

        mean_hierarchy_support = float(
            sum(
                hierarchy_support_values
            )
            / number_of_edges
        )

        mean_directional_support = float(
            sum(
                directional_support_values
            )
            / number_of_edges
        )

        mean_expected_anchor_gap = float(
            sum(
                expected_anchor_gaps
            )
            / number_of_edges
        )
    else:
        directional_edge_ratio = None
        dominant_global_edge_ratio = None
        mean_hierarchy_support = None
        mean_directional_support = None
        mean_expected_anchor_gap = None

    number_of_empty_bow_claims = sum(
        not bool(flag.item())
        for flag in bow_valid_flags
    )

    return {
        "split": split_name,
        "patent_id": patent_id,
        "cpc_codes": record.get(
            "cpc_codes"
        ),
        "section": record.get(
            "section"
        ),
        "class": record.get(
            "class"
        ),
        "subclass": record.get(
            "subclass"
        ),
        "number_of_claims": (
            number_of_claims
        ),
        "number_of_dependency_edges": (
            number_of_edges
        ),
        "number_of_adjacent_edges": (
            number_of_adjacent_edges
        ),
        "number_of_non_adjacent_edges": (
            number_of_non_adjacent_edges
        ),
        "maximum_depth": max(
            int(value.item())
            for value in claim_depths
        ),
        "valid_bow_claims": (
            number_of_claims
            - number_of_empty_bow_claims
        ),
        "empty_bow_claims": (
            number_of_empty_bow_claims
        ),
        "empty_bow_ratio": (
            number_of_empty_bow_claims
            / max(
                number_of_claims,
                1,
            )
        ),
        "directional_edge_ratio": (
            directional_edge_ratio
        ),
        "dominant_global_edge_ratio": (
            dominant_global_edge_ratio
        ),
        "mean_global_hierarchy_support": (
            mean_hierarchy_support
        ),
        "mean_directional_transition_support": (
            mean_directional_support
        ),
        "mean_expected_anchor_gap": (
            mean_expected_anchor_gap
        ),
        "claims": claim_payloads,
        "dependency_edges": (
            dependency_edges
        ),
    }


# ============================================================
# 10. Qualitative-example helper
# ============================================================

def add_claim_text_snippets(
    patent_payload,
    record,
):
    claims_by_id = {
        normalize_claim_id_section9(
            claim_id
        ): text
        for claim_id, text
        in record["claims"].items()
    }

    result = json.loads(
        json.dumps(
            sanitize_json_value(
                patent_payload
            ),
            ensure_ascii=False,
        )
    )

    for claim in result[
        "claims"
    ]:
        claim_id = int(
            claim["claim_id"]
        )

        claim_text = (
            claims_by_id.get(
                claim_id,
                "",
            )
        )

        if claim_text is None:
            claim_text = ""

        if not isinstance(
            claim_text,
            str,
        ):
            claim_text = str(
                claim_text
            )

        normalized_text = " ".join(
            claim_text.split()
        )

        claim[
            "text_snippet"
        ] = normalized_text[
            :CLAIM_TEXT_SNIPPET_LENGTH
        ]

        claim[
            "text_was_truncated_for_export"
        ] = (
            len(normalized_text)
            > CLAIM_TEXT_SNIPPET_LENGTH
        )

    return result


# ============================================================
# 11. Export one split
# ============================================================

def export_split_patent_hierarchies(
    split_name,
):
    split_manifest = (
        inference_manifest[
            "splits"
        ][split_name]
    )

    shard_records = (
        split_manifest[
            "shards"
        ]
    )

    record_lookup = (
        RECORD_LOOKUPS[
            split_name
        ]
    )

    final_jsonl_path = os.path.join(
        PATENT_EXPORT_DIR,
        f"{split_name}_patent_hierarchies.jsonl",
    )

    temporary_jsonl_path = (
        final_jsonl_path + ".tmp"
    )

    if os.path.exists(
        temporary_jsonl_path
    ):
        os.remove(
            temporary_jsonl_path
        )

    number_of_patents = 0
    number_of_claims = 0
    number_of_edges = 0
    number_of_adjacent_edges = 0
    number_of_non_adjacent_edges = 0
    number_of_directional_edges = 0
    number_of_empty_bow_claims = 0

    hierarchy_support_sum = 0.0
    hierarchy_support_count = 0

    directional_support_sum = 0.0
    directional_support_count = 0

    seen_patent_ids = set()

    qualitative_heap = []
    qualitative_counter = 0

    with open(
        temporary_jsonl_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        for shard_record in tqdm(
            shard_records,
            desc=f"Exporting {split_name} patents",
        ):
            shard_path = (
                resolve_inference_shard_path(
                    shard_record
                )
            )

            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )

            if shard.get(
                "split"
            ) != split_name:
                raise ValueError(
                    f"Inference shard split mismatch: "
                    f"{shard_path}"
                )

            theta = (
                shard["theta"]
                .detach()
                .float()
                .cpu()
            )

            claim_depth = (
                shard["claim_depth"]
                .detach()
                .long()
                .cpu()
            )

            bow_valid = (
                shard["bow_valid"]
                .detach()
                .bool()
                .cpu()
            )

            claim_keys = (
                shard["claim_keys"]
            )

            number_of_shard_claims = int(
                theta.shape[0]
            )

            if len(
                claim_keys
            ) != number_of_shard_claims:
                raise ValueError(
                    f"Claim-key count mismatch in {shard_path}."
                )

            if claim_depth.shape != (
                number_of_shard_claims,
            ):
                raise ValueError(
                    f"Claim-depth shape mismatch in {shard_path}."
                )

            if bow_valid.shape != (
                number_of_shard_claims,
            ):
                raise ValueError(
                    f"BoW-valid shape mismatch in {shard_path}."
                )

            patent_claim_positions = (
                defaultdict(list)
            )

            for position, claim_key in enumerate(
                claim_keys
            ):
                if (
                    not isinstance(
                        claim_key,
                        (list, tuple),
                    )
                    or len(claim_key) != 2
                ):
                    raise ValueError(
                        f"Invalid claim key in {shard_path}: "
                        f"{claim_key}"
                    )

                patent_id = (
                    normalize_patent_id_section9(
                        claim_key[0]
                    )
                )

                claim_id = (
                    normalize_claim_id_section9(
                        claim_key[1]
                    )
                )

                patent_claim_positions[
                    patent_id
                ].append(
                    (
                        position,
                        claim_id,
                    )
                )

            for patent_id, positions in (
                patent_claim_positions.items()
            ):
                if patent_id in seen_patent_ids:
                    raise RuntimeError(
                        "A patent appears in multiple inference "
                        f"shards: {patent_id}"
                    )

                seen_patent_ids.add(
                    patent_id
                )

                if patent_id not in record_lookup:
                    raise KeyError(
                        f"No processed record for patent "
                        f"{patent_id} in split={split_name}."
                    )

                positions.sort(
                    key=lambda item: item[1]
                )

                tensor_positions = [
                    int(item[0])
                    for item in positions
                ]

                patent_claim_ids = [
                    int(item[1])
                    for item in positions
                ]

                patent_theta = theta[
                    tensor_positions
                ]

                patent_depth = claim_depth[
                    tensor_positions
                ]

                patent_bow_valid = bow_valid[
                    tensor_positions
                ]

                record = record_lookup[
                    patent_id
                ]

                patent_payload = (
                    build_patent_payload(
                        split_name=split_name,
                        patent_id=patent_id,
                        record=record,
                        claim_ids=(
                            patent_claim_ids
                        ),
                        theta_rows=(
                            patent_theta
                        ),
                        claim_depths=(
                            patent_depth
                        ),
                        bow_valid_flags=(
                            patent_bow_valid
                        ),
                    )
                )

                sanitized_payload = (
                    sanitize_json_value(
                        patent_payload
                    )
                )

                output_file.write(
                    json.dumps(
                        sanitized_payload,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

                number_of_patents += 1
                number_of_claims += int(
                    patent_payload[
                        "number_of_claims"
                    ]
                )
                number_of_edges += int(
                    patent_payload[
                        "number_of_dependency_edges"
                    ]
                )
                number_of_adjacent_edges += int(
                    patent_payload[
                        "number_of_adjacent_edges"
                    ]
                )
                number_of_non_adjacent_edges += int(
                    patent_payload[
                        "number_of_non_adjacent_edges"
                    ]
                )
                number_of_empty_bow_claims += int(
                    patent_payload[
                        "empty_bow_claims"
                    ]
                )

                for edge in patent_payload[
                    "dependency_edges"
                ]:
                    if edge[
                        "direction_consistent"
                    ]:
                        number_of_directional_edges += 1

                    hierarchy_support_sum += float(
                        edge[
                            "global_hierarchy_support"
                        ]
                    )
                    hierarchy_support_count += 1

                    directional_support_sum += float(
                        edge[
                            "directional_transition_support"
                        ]
                    )
                    directional_support_count += 1

                if (
                    patent_payload[
                        "number_of_dependency_edges"
                    ]
                    > 0
                ):
                    directional_ratio = float(
                        patent_payload[
                            "directional_edge_ratio"
                        ]
                    )

                    hierarchy_support = float(
                        patent_payload[
                            "mean_global_hierarchy_support"
                        ]
                    )

                    directional_support = float(
                        patent_payload[
                            "mean_directional_transition_support"
                        ]
                    )

                    maximum_depth = float(
                        patent_payload[
                            "maximum_depth"
                        ]
                    )

                    # Representative examples should contain
                    # multiple claims and meaningful hierarchy evidence.
                    qualitative_score = (
                        0.40
                        * directional_ratio
                        + 0.35
                        * hierarchy_support
                        + 0.20
                        * directional_support
                        + 0.05
                        * min(
                            maximum_depth / 5.0,
                            1.0,
                        )
                    )

                    qualitative_counter += 1

                    heap_item = (
                        qualitative_score,
                        qualitative_counter,
                        patent_payload,
                    )

                    if (
                        len(qualitative_heap)
                        < QUALITATIVE_PATENTS_PER_SPLIT
                    ):
                        heapq.heappush(
                            qualitative_heap,
                            heap_item,
                        )
                    elif (
                        qualitative_score
                        > qualitative_heap[0][0]
                    ):
                        heapq.heapreplace(
                            qualitative_heap,
                            heap_item,
                        )

            del shard
            del theta
            del claim_depth
            del bow_valid

    os.replace(
        temporary_jsonl_path,
        final_jsonl_path,
    )

    expected_patents = int(
        split_manifest[
            "number_of_patents"
        ]
    )

    expected_claims = int(
        split_manifest[
            "number_of_claims"
        ]
    )

    if number_of_patents != expected_patents:
        raise RuntimeError(
            f"{split_name} patent export count mismatch: "
            f"exported={number_of_patents}, "
            f"expected={expected_patents}."
        )

    if number_of_claims != expected_claims:
        raise RuntimeError(
            f"{split_name} claim export count mismatch: "
            f"exported={number_of_claims}, "
            f"expected={expected_claims}."
        )

    qualitative_heap.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    qualitative_examples = []

    for rank, (
        score,
        _,
        patent_payload,
    ) in enumerate(
        qualitative_heap,
        start=1,
    ):
        patent_id = (
            patent_payload[
                "patent_id"
            ]
        )

        record = record_lookup[
            patent_id
        ]

        qualitative_payload = (
            add_claim_text_snippets(
                patent_payload=(
                    patent_payload
                ),
                record=record,
            )
        )

        qualitative_payload[
            "qualitative_rank"
        ] = rank

        qualitative_payload[
            "qualitative_selection_score"
        ] = float(score)

        qualitative_examples.append(
            qualitative_payload
        )

    qualitative_path = os.path.join(
        QUALITATIVE_DIR,
        f"{split_name}_qualitative_examples.json",
    )

    atomic_json_save_section9(
        {
            "split": split_name,
            "selection_method": (
                "weighted directional consistency, global "
                "hierarchy support, directional transition "
                "support, and maximum claim depth"
            ),
            "number_of_examples": len(
                qualitative_examples
            ),
            "examples": (
                qualitative_examples
            ),
        },
        qualitative_path,
    )

    directional_edge_ratio = (
        number_of_directional_edges
        / max(
            number_of_edges,
            1,
        )
    )

    adjacent_edge_coverage = (
        number_of_adjacent_edges
        / max(
            number_of_edges,
            1,
        )
    )

    mean_hierarchy_support = (
        hierarchy_support_sum
        / hierarchy_support_count
        if hierarchy_support_count > 0
        else None
    )

    mean_directional_support = (
        directional_support_sum
        / directional_support_count
        if directional_support_count > 0
        else None
    )

    return {
        "split": split_name,
        "number_of_patents": (
            number_of_patents
        ),
        "number_of_claims": (
            number_of_claims
        ),
        "number_of_dependency_edges": (
            number_of_edges
        ),
        "number_of_adjacent_edges": (
            number_of_adjacent_edges
        ),
        "number_of_non_adjacent_edges": (
            number_of_non_adjacent_edges
        ),
        "adjacent_edge_coverage": (
            adjacent_edge_coverage
        ),
        "number_of_directional_edges": (
            number_of_directional_edges
        ),
        "directional_edge_ratio": (
            directional_edge_ratio
        ),
        "empty_bow_claims": (
            number_of_empty_bow_claims
        ),
        "empty_bow_ratio": (
            number_of_empty_bow_claims
            / max(
                number_of_claims,
                1,
            )
        ),
        "mean_global_hierarchy_support": (
            mean_hierarchy_support
        ),
        "mean_directional_transition_support": (
            mean_directional_support
        ),
        "patent_hierarchy_jsonl": (
            final_jsonl_path
        ),
        "qualitative_examples": (
            qualitative_path
        ),
    }


# ============================================================
# 12. Export train/dev/test patent hierarchies
# ============================================================

split_export_summaries = {}

for split_name in [
    "train",
    "dev",
    "test",
]:
    split_export_summaries[
        split_name
    ] = export_split_patent_hierarchies(
        split_name
    )

print("\n=== PATENT EXPORT SUMMARY ===")

for split_name, summary in (
    split_export_summaries.items()
):
    print(
        f"{split_name:5s}: "
        f"patents={summary['number_of_patents']:,}, "
        f"claims={summary['number_of_claims']:,}, "
        f"edges={summary['number_of_dependency_edges']:,}, "
        f"directional={summary['directional_edge_ratio']:.4%}, "
        f"adjacent={summary['adjacent_edge_coverage']:.4%}"
    )


# ============================================================
# 13. Publication table: topic summary
# ============================================================

per_topic_evaluation = {
    int(record["topic_id"]): record
    for record in section8_report[
        "per_topic"
    ]
}

topic_table_path = os.path.join(
    TABLE_DIR,
    "table_topic_summary.csv",
)

with open(
    topic_table_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = [
        "topic_id",
        "topic_number",
        "anchor_coordinate",
        "train_topic_usage",
        "npmi",
        "exclusivity",
        "normalized_entropy",
        "maximum_cosine_similarity",
        "role",
        "incoming_edges",
        "outgoing_edges",
        "top_words",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    node_lookup = {
        int(node["topic_id"]): node
        for node in global_hierarchy[
            "nodes"
        ]
    }

    for topic_id in range(
        NUM_TOPICS
    ):
        evaluation = (
            per_topic_evaluation.get(
                topic_id,
                {},
            )
        )

        node = node_lookup.get(
            topic_id,
            {},
        )

        writer.writerow(
            {
                "topic_id": topic_id,
                "topic_number": topic_id + 1,
                "anchor_coordinate": float(
                    anchor_coordinates[
                        topic_id
                    ].item()
                ),
                "train_topic_usage": float(
                    topic_usage[
                        topic_id
                    ].item()
                ),
                "npmi": evaluation.get(
                    "npmi"
                ),
                "exclusivity": evaluation.get(
                    "exclusivity"
                ),
                "normalized_entropy": (
                    evaluation.get(
                        "normalized_entropy"
                    )
                ),
                "maximum_cosine_similarity": (
                    evaluation.get(
                        "maximum_cosine_similarity"
                    )
                ),
                "role": node.get(
                    "role",
                    "unknown",
                ),
                "incoming_edges": int(
                    node.get(
                        "incoming_edges",
                        0,
                    )
                ),
                "outgoing_edges": int(
                    node.get(
                        "outgoing_edges",
                        0,
                    )
                ),
                "top_words": ", ".join(
                    item["word"]
                    for item in
                    topic_summaries[
                        topic_id
                    ]["top_words"]
                ),
            }
        )


# ============================================================
# 14. Publication table: global hierarchy edges
# ============================================================

hierarchy_edge_table_path = os.path.join(
    TABLE_DIR,
    "table_global_hierarchy_edges.csv",
)

with open(
    hierarchy_edge_table_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = [
        "parent_topic_id",
        "parent_topic_number",
        "child_topic_id",
        "child_topic_number",
        "rank_for_parent",
        "parent_anchor",
        "child_anchor",
        "anchor_gap",
        "joint_probability",
        "conditional_probability",
        "anchor_cost",
        "parent_top_words",
        "child_top_words",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    sorted_hierarchy_edges = sorted(
        global_hierarchy[
            "edges"
        ],
        key=lambda edge: (
            int(
                edge[
                    "parent_topic_id"
                ]
            ),
            int(
                edge[
                    "rank_for_parent"
                ]
            ),
        ),
    )

    for edge in sorted_hierarchy_edges:
        writer.writerow(
            {
                "parent_topic_id": edge[
                    "parent_topic_id"
                ],
                "parent_topic_number": edge[
                    "parent_topic_number"
                ],
                "child_topic_id": edge[
                    "child_topic_id"
                ],
                "child_topic_number": edge[
                    "child_topic_number"
                ],
                "rank_for_parent": edge[
                    "rank_for_parent"
                ],
                "parent_anchor": edge[
                    "parent_anchor_coordinate"
                ],
                "child_anchor": edge[
                    "child_anchor_coordinate"
                ],
                "anchor_gap": edge[
                    "anchor_gap"
                ],
                "joint_probability": edge[
                    "joint_probability"
                ],
                "conditional_probability": edge[
                    "conditional_probability"
                ],
                "anchor_cost": edge[
                    "anchor_cost"
                ],
                "parent_top_words": ", ".join(
                    edge[
                        "parent_top_words"
                    ]
                ),
                "child_top_words": ", ".join(
                    edge[
                        "child_top_words"
                    ]
                ),
            }
        )


# ============================================================
# 15. Publication table: split evaluation
# ============================================================

split_table_path = os.path.join(
    TABLE_DIR,
    "table_split_hierarchy_metrics.csv",
)

with open(
    split_table_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = [
        "split",
        "number_of_patents",
        "number_of_claims",
        "number_of_dependency_edges",
        "adjacent_edge_coverage",
        "directional_edge_ratio",
        "empty_bow_ratio",
        "mean_global_hierarchy_support",
        "mean_directional_transition_support",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for split_name in [
        "train",
        "dev",
        "test",
    ]:
        summary = (
            split_export_summaries[
                split_name
            ]
        )

        writer.writerow(
            {
                field_name: summary.get(
                    field_name
                )
                for field_name in fieldnames
            }
        )


# ============================================================
# 16. Publication table: overall model metrics
# ============================================================

overall_metrics_path = os.path.join(
    TABLE_DIR,
    "table_overall_model_metrics.csv",
)

topic_quality = section8_report[
    "topic_quality"
]

hierarchy_quality = section8_report[
    "hierarchy_quality"
]

depth_anchor_alignment = (
    section8_report[
        "depth_anchor_alignment"
    ]
)

overall_metric_rows = [
    (
        "Number of topics",
        NUM_TOPICS,
    ),
    (
        "Vocabulary size",
        VOCAB_SIZE,
    ),
    (
        "Topic diversity",
        topic_quality.get(
            "topic_diversity"
        ),
    ),
    (
        "Mean NPMI",
        topic_quality.get(
            "mean_npmi"
        ),
    ),
    (
        "Median NPMI",
        topic_quality.get(
            "median_npmi"
        ),
    ),
    (
        "Mean exclusivity",
        topic_quality.get(
            "mean_exclusivity"
        ),
    ),
    (
        "Mean normalized entropy",
        topic_quality.get(
            "mean_normalized_entropy"
        ),
    ),
    (
        "Mean topic cosine similarity",
        topic_quality.get(
            "mean_topic_cosine_similarity"
        ),
    ),
    (
        "Maximum topic cosine similarity",
        topic_quality.get(
            "maximum_topic_cosine_similarity"
        ),
    ),
    (
        "Global hierarchy nodes",
        hierarchy_quality.get(
            "number_of_nodes"
        ),
    ),
    (
        "Global hierarchy edges",
        hierarchy_quality.get(
            "number_of_edges"
        ),
    ),
    (
        "Directional empirical mass ratio",
        hierarchy_quality.get(
            "directional_empirical_mass_ratio"
        ),
    ),
    (
        "Selected hierarchy mass ratio",
        hierarchy_quality.get(
            "selected_edge_joint_mass_ratio"
        ),
    ),
    (
        "Depth-anchor Pearson correlation",
        depth_anchor_alignment.get(
            "pearson_correlation"
        ),
    ),
    (
        "Depth-anchor Spearman correlation",
        depth_anchor_alignment.get(
            "spearman_correlation"
        ),
    ),
    (
        "Depth monotonic transition ratio",
        depth_anchor_alignment.get(
            "monotonic_transition_ratio"
        ),
    ),
]

with open(
    overall_metrics_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "metric",
            "value",
        ],
    )

    writer.writeheader()

    for metric_name, metric_value in (
        overall_metric_rows
    ):
        writer.writerow(
            {
                "metric": metric_name,
                "value": metric_value,
            }
        )


# ============================================================
# 17. Save compact publication summary
# ============================================================

publication_summary = {
    "run_name": global_hierarchy.get(
        "run_name"
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "model": {
        "number_of_topics": (
            NUM_TOPICS
        ),
        "vocabulary_size": (
            VOCAB_SIZE
        ),
        "best_epoch": (
            global_hierarchy.get(
                "best_epoch"
            )
        ),
    },
    "topic_quality": (
        topic_quality
    ),
    "hierarchy_quality": (
        hierarchy_quality
    ),
    "depth_anchor_alignment": (
        depth_anchor_alignment
    ),
    "split_hierarchy_metrics": (
        split_export_summaries
    ),
    "important_interpretation": {
        "global_hierarchy_training_split": (
            "train"
        ),
        "test_used_for_hierarchy_construction": (
            False
        ),
        "test_used_for_model_selection": (
            False
        ),
        "empty_bow_claim_policy": (
            "Claims with zero fixed-vocabulary terms remain "
            "in graph and hierarchy objectives but are excluded "
            "from reconstruction loss."
        ),
        "non_adjacent_edge_policy": (
            "Retained in patent export; excluded from adjacent-depth "
            "OT transition estimation."
        ),
        "global_structure_type": (
            "directed acyclic graph"
        ),
    },
}

publication_summary_path = os.path.join(
    SECTION9_RESULT_DIR,
    "publication_summary.json",
)

atomic_json_save_section9(
    publication_summary,
    publication_summary_path,
)


# ============================================================
# 18. Final manifest
# ============================================================

section9_manifest = {
    "run_name": global_hierarchy.get(
        "run_name"
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "created_at": (
        datetime.now().isoformat()
    ),
    "number_of_topics": (
        NUM_TOPICS
    ),
    "claim_top_k_topics": (
        CLAIM_TOP_K_TOPICS
    ),
    "export_full_theta": (
        EXPORT_FULL_THETA
    ),
    "splits": (
        split_export_summaries
    ),
    "files": {
        "publication_summary": (
            publication_summary_path
        ),
        "topic_summary_table": (
            topic_table_path
        ),
        "hierarchy_edge_table": (
            hierarchy_edge_table_path
        ),
        "split_metric_table": (
            split_table_path
        ),
        "overall_metric_table": (
            overall_metrics_path
        ),
        "patent_hierarchy_directory": (
            PATENT_EXPORT_DIR
        ),
        "qualitative_example_directory": (
            QUALITATIVE_DIR
        ),
    },
}

section9_manifest_path = os.path.join(
    SECTION9_RESULT_DIR,
    "section9_manifest.json",
)

atomic_json_save_section9(
    section9_manifest,
    section9_manifest_path,
)


# ============================================================
# 19. Final validation
# ============================================================

required_output_files = [
    publication_summary_path,
    topic_table_path,
    hierarchy_edge_table_path,
    split_table_path,
    overall_metrics_path,
    section9_manifest_path,
]

for split_name in [
    "train",
    "dev",
    "test",
]:
    required_output_files.append(
        split_export_summaries[
            split_name
        ][
            "patent_hierarchy_jsonl"
        ]
    )

    required_output_files.append(
        split_export_summaries[
            split_name
        ][
            "qualitative_examples"
        ]
    )

missing_output_files = [
    path
    for path in required_output_files
    if not os.path.isfile(path)
]

if missing_output_files:
    raise FileNotFoundError(
        "Section 9 output files are missing: "
        f"{missing_output_files}"
    )

for split_name, summary in (
    split_export_summaries.items()
):
    if summary[
        "number_of_patents"
    ] <= 0:
        raise RuntimeError(
            f"No patents exported for split={split_name}."
        )

    if summary[
        "number_of_claims"
    ] <= 0:
        raise RuntimeError(
            f"No claims exported for split={split_name}."
        )


# ============================================================
# 20. Final status
# ============================================================

print("\n" + "=" * 72)
print("SECTION 9 COMPLETED SUCCESSFULLY")
print("=" * 72)
print(f"Feature run            : {FEATURE_RUN_NAME}")
print(f"Topics                 : {NUM_TOPICS}")
print(f"Vocabulary             : {VOCAB_SIZE:,}")

for split_name in [
    "train",
    "dev",
    "test",
]:
    summary = (
        split_export_summaries[
            split_name
        ]
    )

    print(
        f"{split_name:5s} patents        : "
        f"{summary['number_of_patents']:,}"
    )
    print(
        f"{split_name:5s} claims         : "
        f"{summary['number_of_claims']:,}"
    )
    print(
        f"{split_name:5s} directional    : "
        f"{summary['directional_edge_ratio']:.4%}"
    )
    print(
        f"{split_name:5s} hierarchy sup. : "
        f"{summary['mean_global_hierarchy_support']}"
    )

print(f"Patent exports         : {PATENT_EXPORT_DIR}")
print(f"Qualitative examples   : {QUALITATIVE_DIR}")
print(f"Publication tables     : {TABLE_DIR}")
print(f"Publication summary    : {publication_summary_path}")
print(f"Section 9 manifest     : {section9_manifest_path}")
print("=" * 72)

if FEATURE_RUN_NAME.startswith(
    "debug"
):
    print(
        "\n[IMPORTANT] These patent examples and tables come from "
        "a debug run and must not be used as final results."
    )
else:
    print(
        "\n[PASS] Full patent-level hierarchy export and "
        "publication-table generation completed."
    )

print(
    "\nAll Section 9 outputs are saved to Google Drive. "
    "The experimental pipeline is now complete."
)


# ============================================================
# SECTION 10: Final Audit, Reproducibility Manifest,
#             and Publication Bundle
# COMPLETE SINGLE-CELL VERSION
#
# This section:
#   1. Validates outputs from Sections 6-9.
#   2. Creates a final file inventory and SHA-256 checksums.
#   3. Saves reproducibility/environment information.
#   4. Creates a final README.
#   5. Creates a publication/share ZIP bundle.
#
# No model training or inference occurs in this section.
# ============================================================

import os
import sys
import csv
import json
import math
import shutil
import hashlib
import zipfile
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass

import numpy as np
import torch


# ============================================================
# 0. User configuration
# ============================================================

# Include best.pt in the final ZIP.
#
# WARNING:
# Section 6 checkpoints include optimizer states and can be large.
# Leave False for a lightweight publication bundle.
INCLUDE_BEST_CHECKPOINT_IN_BUNDLE = False

# Include claim-level theta shards from Section 7.
#
# These files can be very large, so the default is False.
INCLUDE_THETA_SHARDS_IN_BUNDLE = False

# Include periodic checkpoints:
# epoch_003.pt, epoch_006.pt, ...
INCLUDE_PERIODIC_CHECKPOINTS_IN_BUNDLE = False

# Include figures generated by Section 8.
INCLUDE_SECTION8_FIGURES_IN_BUNDLE = True

# Create the ZIP archive.
CREATE_PUBLICATION_ZIP = True

# Calculate SHA-256 only for files at or below this size.
# Larger files remain in the inventory but are marked as skipped.
MAX_CHECKSUM_FILE_SIZE_BYTES = 2 * 1024**3  # 2 GiB

# Read at most this many bytes at once when computing SHA-256.
HASH_CHUNK_SIZE_BYTES = 8 * 1024**2  # 8 MiB


# ============================================================
# 1. Preconditions
# ============================================================

required_globals = [
    "CONFIG",
    "DIRS",
    "FEATURE_RUN_NAME",
    "RUN_NAME",
    "RUN_RESULT_DIR",
    "RUN_CHECKPOINT_DIR",
    "BEST_CHECKPOINT_PATH",
    "LATEST_CHECKPOINT_PATH",
]

missing_globals = [
    name
    for name in required_globals
    if name not in globals()
]

if missing_globals:
    raise RuntimeError(
        "Section 10 prerequisites are missing: "
        f"{missing_globals}. Run Sections 0 through 9 first."
    )


# ============================================================
# 2. Resolve Section 7-10 directories
# ============================================================

RUN_RESULT_DIR = os.path.abspath(
    str(RUN_RESULT_DIR)
)

RUN_CHECKPOINT_DIR = os.path.abspath(
    str(RUN_CHECKPOINT_DIR)
)

# Prefer the global Section 7 path if it exists.
if (
    "SECTION7_RESULT_DIR" in globals()
    and SECTION7_RESULT_DIR is not None
):
    resolved_section7_directory = os.path.abspath(
        str(SECTION7_RESULT_DIR)
    )
else:
    resolved_section7_directory = os.path.join(
        RUN_RESULT_DIR,
        "section7_inference",
    )

SECTION7_RESULT_DIR = resolved_section7_directory

# Prefer global Section 8 path if defined.
section8_global_candidates = [
    globals().get("SECTION8_RESULT_DIR"),
    globals().get("SECTION8_EVALUATION_DIR"),
]

SECTION8_RESULT_DIR = None

for candidate in section8_global_candidates:
    if candidate is None:
        continue

    candidate = os.path.abspath(
        str(candidate)
    )

    if os.path.isdir(candidate):
        SECTION8_RESULT_DIR = candidate
        break

if SECTION8_RESULT_DIR is None:
    SECTION8_RESULT_DIR = os.path.join(
        SECTION7_RESULT_DIR,
        "section8_evaluation",
    )

# Prefer global Section 9 path if defined.
section9_global_candidates = [
    globals().get("SECTION9_RESULT_DIR"),
    globals().get("SECTION9_EXPORT_DIR"),
    globals().get("PATENT_EXPORT_DIR"),
]

SECTION9_RESULT_DIR = None

for candidate in section9_global_candidates:
    if candidate is None:
        continue

    candidate = os.path.abspath(
        str(candidate)
    )

    if os.path.isdir(candidate):
        SECTION9_RESULT_DIR = candidate
        break

if SECTION9_RESULT_DIR is None:
    SECTION9_RESULT_DIR = os.path.join(
        SECTION7_RESULT_DIR,
        "section9_patent_exports",
    )

SECTION10_RESULT_DIR = os.path.join(
    RUN_RESULT_DIR,
    "section10_final_bundle",
)

SECTION10_AUDIT_DIR = os.path.join(
    SECTION10_RESULT_DIR,
    "audit",
)

SECTION10_BUNDLE_STAGING_DIR = os.path.join(
    SECTION10_RESULT_DIR,
    "publication_bundle",
)

SECTION10_ZIP_PATH = os.path.join(
    SECTION10_RESULT_DIR,
    f"{RUN_NAME}_publication_bundle.zip",
)

for directory in [
    SECTION10_RESULT_DIR,
    SECTION10_AUDIT_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )

print("=== SECTION 10 DIRECTORIES ===")
print(f"Run result         : {RUN_RESULT_DIR}")
print(f"Section 7          : {SECTION7_RESULT_DIR}")
print(f"Section 8          : {SECTION8_RESULT_DIR}")
print(f"Section 9          : {SECTION9_RESULT_DIR}")
print(f"Section 10         : {SECTION10_RESULT_DIR}")
print(f"Publication ZIP    : {SECTION10_ZIP_PATH}")


# ============================================================
# 3. Output file paths
# ============================================================

section6_files = {
    "best_checkpoint": os.path.abspath(
        str(BEST_CHECKPOINT_PATH)
    ),
    "latest_checkpoint": os.path.abspath(
        str(LATEST_CHECKPOINT_PATH)
    ),
    "training_summary": os.path.join(
        os.path.dirname(
            os.path.abspath(
                str(globals().get(
                    "TRAINING_SUMMARY_PATH",
                    os.path.join(
                        RUN_RESULT_DIR,
                        "training_summary.json",
                    ),
                ))
            )
        ),
        os.path.basename(
            str(globals().get(
                "TRAINING_SUMMARY_PATH",
                "training_summary.json",
            ))
        ),
    ),
    "run_manifest": os.path.abspath(
        str(globals().get(
            "RUN_MANIFEST_PATH",
            os.path.join(
                RUN_RESULT_DIR,
                "run_manifest.json",
            ),
        ))
    ),
    "history": os.path.abspath(
        str(globals().get(
            "HISTORY_JSONL_PATH",
            os.path.join(
                RUN_RESULT_DIR,
                "history.jsonl",
            ),
        ))
    ),
    "learned_anchor": os.path.join(
        RUN_RESULT_DIR,
        "learned_anchor.pt",
    ),
}

section7_files = {
    "inference_manifest": os.path.join(
        SECTION7_RESULT_DIR,
        "inference_manifest.json",
    ),
    "global_topic_hierarchy": os.path.join(
        SECTION7_RESULT_DIR,
        "global_topic_hierarchy.json",
    ),
    "global_transition_matrices": os.path.join(
        SECTION7_RESULT_DIR,
        "global_transition_matrices.pt",
    ),
    "topic_word_distribution": os.path.join(
        SECTION7_RESULT_DIR,
        "topic_word_distribution.pt",
    ),
}

section7_theta_shard_directory = os.path.join(
    SECTION7_RESULT_DIR,
    "claim_theta_shards",
)

section8_files = {
    "evaluation_report": os.path.join(
        SECTION8_RESULT_DIR,
        "section8_evaluation_report.json",
    ),
    "per_topic_evaluation": os.path.join(
        SECTION8_RESULT_DIR,
        "per_topic_evaluation.csv",
    ),
    "depth_anchor_alignment": os.path.join(
        SECTION8_RESULT_DIR,
        "depth_anchor_alignment.csv",
    ),
}

section8_figure_directory = os.path.join(
    SECTION8_RESULT_DIR,
    "figures",
)

section9_files = {
    "section9_manifest": os.path.join(
        SECTION9_RESULT_DIR,
        "section9_manifest.json",
    ),
    "publication_summary": os.path.join(
        SECTION9_RESULT_DIR,
        "publication_summary.json",
    ),
}

section9_patent_hierarchy_directory = os.path.join(
    SECTION9_RESULT_DIR,
    "patent_hierarchies",
)

section9_qualitative_directory = os.path.join(
    SECTION9_RESULT_DIR,
    "qualitative_examples",
)

section9_table_directory = os.path.join(
    SECTION9_RESULT_DIR,
    "tables",
)

expected_section9_patent_files = [
    os.path.join(
        section9_patent_hierarchy_directory,
        f"{split}_patent_hierarchies.jsonl",
    )
    for split in ["train", "dev", "test"]
]

expected_section9_qualitative_files = [
    os.path.join(
        section9_qualitative_directory,
        f"{split}_qualitative_examples.json",
    )
    for split in ["train", "dev", "test"]
]

expected_section9_table_files = [
    os.path.join(
        section9_table_directory,
        "table_topic_summary.csv",
    ),
    os.path.join(
        section9_table_directory,
        "table_global_hierarchy_edges.csv",
    ),
    os.path.join(
        section9_table_directory,
        "table_split_hierarchy_metrics.csv",
    ),
    os.path.join(
        section9_table_directory,
        "table_overall_model_metrics.csv",
    ),
]


# ============================================================
# 4. Generic utility functions
# ============================================================

def config_to_dictionary_section10(config):
    if is_dataclass(config):
        return asdict(config)

    if isinstance(config, dict):
        return dict(config)

    result = {}

    for key in dir(config):
        if key.startswith("_"):
            continue

        try:
            value = getattr(
                config,
                key,
            )
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
                list,
                tuple,
                dict,
            ),
        ):
            result[key] = value

    return result


def atomic_json_save_section10(
    data,
    final_path,
):
    final_path = os.path.abspath(
        str(final_path)
    )

    os.makedirs(
        os.path.dirname(final_path),
        exist_ok=True,
    )

    temporary_path = (
        final_path + ".tmp"
    )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    os.replace(
        temporary_path,
        final_path,
    )


def atomic_text_save_section10(
    text,
    final_path,
):
    final_path = os.path.abspath(
        str(final_path)
    )

    os.makedirs(
        os.path.dirname(final_path),
        exist_ok=True,
    )

    temporary_path = (
        final_path + ".tmp"
    )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            str(text)
        )

    os.replace(
        temporary_path,
        final_path,
    )


def load_json_section10(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def count_jsonl_records_section10(path):
    count = 0

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at {path}, "
                    f"line {line_number}."
                ) from exc

            count += 1

    return count


def inspect_csv_section10(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(
                f"CSV file is empty: {path}"
            )

        if not header:
            raise ValueError(
                f"CSV header is empty: {path}"
            )

        row_count = sum(
            1
            for row in reader
            if row
        )

    return {
        "columns": header,
        "num_columns": len(header),
        "num_data_rows": row_count,
    }


def sha256_file_section10(
    path,
    chunk_size=HASH_CHUNK_SIZE_BYTES,
):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def human_readable_size_section10(
    num_bytes,
):
    num_bytes = float(
        num_bytes
    )

    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    unit_index = 0

    while (
        num_bytes >= 1024.0
        and unit_index < len(units) - 1
    ):
        num_bytes /= 1024.0
        unit_index += 1

    return (
        f"{num_bytes:.2f} "
        f"{units[unit_index]}"
    )


def list_files_recursively_section10(
    root_directory,
    excluded_directories=None,
):
    root_directory = os.path.abspath(
        str(root_directory)
    )

    excluded_directories = {
        os.path.abspath(
            str(path)
        )
        for path in (
            excluded_directories or []
        )
    }

    result = []

    if not os.path.isdir(
        root_directory
    ):
        return result

    for current_root, directory_names, file_names in os.walk(
        root_directory
    ):
        current_root_absolute = os.path.abspath(
            current_root
        )

        retained_directories = []

        for directory_name in directory_names:
            candidate = os.path.abspath(
                os.path.join(
                    current_root_absolute,
                    directory_name,
                )
            )

            should_exclude = any(
                candidate == excluded
                or candidate.startswith(
                    excluded + os.sep
                )
                for excluded in excluded_directories
            )

            if not should_exclude:
                retained_directories.append(
                    directory_name
                )

        directory_names[:] = retained_directories

        for file_name in file_names:
            file_path = os.path.join(
                current_root_absolute,
                file_name,
            )

            if os.path.isfile(
                file_path
            ):
                result.append(
                    os.path.abspath(
                        file_path
                    )
                )

    return sorted(
        set(result)
    )


def safe_relative_path_section10(
    path,
):
    path = os.path.abspath(
        str(path)
    )

    candidate_roots = [
        (
            "results",
            RUN_RESULT_DIR,
        ),
        (
            "checkpoints",
            RUN_CHECKPOINT_DIR,
        ),
    ]

    for root_label, root_path in candidate_roots:
        root_path = os.path.abspath(
            str(root_path)
        )

        try:
            common_path = os.path.commonpath(
                [path, root_path]
            )
        except ValueError:
            continue

        if common_path == root_path:
            return os.path.join(
                root_label,
                os.path.relpath(
                    path,
                    root_path,
                ),
            )

    return os.path.join(
        "external",
        os.path.basename(path),
    )


def copy_file_to_bundle_section10(
    source_path,
    bundle_root,
):
    source_path = os.path.abspath(
        str(source_path)
    )

    if not os.path.isfile(
        source_path
    ):
        raise FileNotFoundError(
            f"Cannot bundle missing file: {source_path}"
        )

    relative_path = (
        safe_relative_path_section10(
            source_path
        )
    )

    destination_path = os.path.join(
        bundle_root,
        relative_path,
    )

    os.makedirs(
        os.path.dirname(
            destination_path
        ),
        exist_ok=True,
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    return destination_path


# ============================================================
# 5. Required-output validation
# ============================================================

required_files = {
    # Section 6
    "section6.best_checkpoint": (
        section6_files[
            "best_checkpoint"
        ]
    ),
    "section6.latest_checkpoint": (
        section6_files[
            "latest_checkpoint"
        ]
    ),
    "section6.training_summary": (
        section6_files[
            "training_summary"
        ]
    ),
    "section6.run_manifest": (
        section6_files[
            "run_manifest"
        ]
    ),
    "section6.history": (
        section6_files[
            "history"
        ]
    ),
    "section6.learned_anchor": (
        section6_files[
            "learned_anchor"
        ]
    ),

    # Section 7
    "section7.inference_manifest": (
        section7_files[
            "inference_manifest"
        ]
    ),
    "section7.global_topic_hierarchy": (
        section7_files[
            "global_topic_hierarchy"
        ]
    ),
    "section7.global_transition_matrices": (
        section7_files[
            "global_transition_matrices"
        ]
    ),
    "section7.topic_word_distribution": (
        section7_files[
            "topic_word_distribution"
        ]
    ),

    # Section 8
    "section8.evaluation_report": (
        section8_files[
            "evaluation_report"
        ]
    ),
    "section8.per_topic_evaluation": (
        section8_files[
            "per_topic_evaluation"
        ]
    ),
    "section8.depth_anchor_alignment": (
        section8_files[
            "depth_anchor_alignment"
        ]
    ),

    # Section 9
    "section9.manifest": (
        section9_files[
            "section9_manifest"
        ]
    ),
    "section9.publication_summary": (
        section9_files[
            "publication_summary"
        ]
    ),
}

for path in (
    expected_section9_patent_files
    + expected_section9_qualitative_files
    + expected_section9_table_files
):
    label = (
        "section9."
        + os.path.basename(path)
    )

    required_files[
        label
    ] = path

missing_required_files = {
    label: path
    for label, path in required_files.items()
    if not os.path.isfile(path)
}

if missing_required_files:
    missing_lines = "\n".join(
        f"  - {label}: {path}"
        for label, path in (
            missing_required_files.items()
        )
    )

    raise FileNotFoundError(
        "Required Section 6-9 outputs are missing:\n"
        f"{missing_lines}"
    )

if not os.path.isdir(
    section7_theta_shard_directory
):
    raise FileNotFoundError(
        "Section 7 theta-shard directory is missing: "
        f"{section7_theta_shard_directory}"
    )

print("\n[PASS] All required Section 6-9 files exist.")


# ============================================================
# 6. JSON, JSONL, CSV, and PyTorch validation
# ============================================================

validation_details = {
    "json": {},
    "jsonl": {},
    "csv": {},
    "torch": {},
}

json_paths = [
    section6_files[
        "training_summary"
    ],
    section6_files[
        "run_manifest"
    ],
    section7_files[
        "inference_manifest"
    ],
    section7_files[
        "global_topic_hierarchy"
    ],
    section8_files[
        "evaluation_report"
    ],
    section9_files[
        "section9_manifest"
    ],
    section9_files[
        "publication_summary"
    ],
] + expected_section9_qualitative_files

for json_path in json_paths:
    loaded_json = load_json_section10(
        json_path
    )

    validation_details[
        "json"
    ][safe_relative_path_section10(
        json_path
    )] = {
        "valid": True,
        "top_level_type": type(
            loaded_json
        ).__name__,
        "top_level_size": (
            len(loaded_json)
            if hasattr(
                loaded_json,
                "__len__",
            )
            else None
        ),
    }

jsonl_paths = [
    section6_files[
        "history"
    ],
] + expected_section9_patent_files

for jsonl_path in jsonl_paths:
    num_records = (
        count_jsonl_records_section10(
            jsonl_path
        )
    )

    validation_details[
        "jsonl"
    ][safe_relative_path_section10(
        jsonl_path
    )] = {
        "valid": True,
        "num_records": (
            num_records
        ),
    }

csv_paths = [
    section8_files[
        "per_topic_evaluation"
    ],
    section8_files[
        "depth_anchor_alignment"
    ],
] + expected_section9_table_files

for csv_path in csv_paths:
    csv_information = (
        inspect_csv_section10(
            csv_path
        )
    )

    validation_details[
        "csv"
    ][safe_relative_path_section10(
        csv_path
    )] = {
        "valid": True,
        **csv_information,
    }

torch_paths = [
    section6_files[
        "learned_anchor"
    ],
    section7_files[
        "global_transition_matrices"
    ],
    section7_files[
        "topic_word_distribution"
    ],
]

for torch_path in torch_paths:
    torch_object = torch.load(
        torch_path,
        map_location="cpu",
        weights_only=False,
    )

    validation_details[
        "torch"
    ][safe_relative_path_section10(
        torch_path
    )] = {
        "valid": True,
        "top_level_type": type(
            torch_object
        ).__name__,
        "keys": (
            sorted(
                str(key)
                for key in torch_object.keys()
            )
            if isinstance(
                torch_object,
                dict,
            )
            else None
        ),
        "tensor_shape": (
            list(
                torch_object.shape
            )
            if isinstance(
                torch_object,
                torch.Tensor,
            )
            else None
        ),
    }

print("[PASS] JSON, JSONL, CSV, and PyTorch outputs are readable.")


# ============================================================
# 7. Semantic cross-checks
# ============================================================

training_summary = load_json_section10(
    section6_files[
        "training_summary"
    ]
)

inference_manifest = load_json_section10(
    section7_files[
        "inference_manifest"
    ]
)

global_hierarchy = load_json_section10(
    section7_files[
        "global_topic_hierarchy"
    ]
)

section8_report = load_json_section10(
    section8_files[
        "evaluation_report"
    ]
)

section9_manifest = load_json_section10(
    section9_files[
        "section9_manifest"
    ]
)

publication_summary = load_json_section10(
    section9_files[
        "publication_summary"
    ]
)

run_names_found = {}

for object_name, object_value in [
    (
        "training_summary",
        training_summary,
    ),
    (
        "inference_manifest",
        inference_manifest,
    ),
    (
        "section8_report",
        section8_report,
    ),
    (
        "section9_manifest",
        section9_manifest,
    ),
    (
        "publication_summary",
        publication_summary,
    ),
]:
    if isinstance(
        object_value,
        dict,
    ):
        candidate_run_name = (
            object_value.get(
                "run_name"
            )
        )

        if candidate_run_name is not None:
            run_names_found[
                object_name
            ] = str(
                candidate_run_name
            )

run_name_mismatches = {
    name: found_run_name
    for name, found_run_name in (
        run_names_found.items()
    )
    if found_run_name != RUN_NAME
}

if run_name_mismatches:
    raise ValueError(
        "Run-name mismatch across saved outputs: "
        f"expected={RUN_NAME}, "
        f"mismatches={run_name_mismatches}"
    )

theta_shard_files = sorted(
    str(path)
    for path in Path(
        section7_theta_shard_directory
    ).glob("*.pt")
)

if not theta_shard_files:
    raise RuntimeError(
        "No theta shard files were found in: "
        f"{section7_theta_shard_directory}"
    )

split_hierarchy_counts = {}

for split, jsonl_path in zip(
    ["train", "dev", "test"],
    expected_section9_patent_files,
):
    split_hierarchy_counts[
        split
    ] = count_jsonl_records_section10(
        jsonl_path
    )

best_epoch_one_based = (
    training_summary.get(
        "best_epoch_one_based"
    )
)

best_validation_loss = (
    training_summary.get(
        "best_validation_loss"
    )
)

semantic_checks = {
    "run_name": RUN_NAME,
    "run_names_found": (
        run_names_found
    ),
    "run_name_consistent": True,
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "best_epoch_one_based": (
        best_epoch_one_based
    ),
    "best_validation_loss": (
        best_validation_loss
    ),
    "num_theta_shard_files": len(
        theta_shard_files
    ),
    "patent_hierarchy_record_counts": (
        split_hierarchy_counts
    ),
    "global_hierarchy_top_level_type": (
        type(
            global_hierarchy
        ).__name__
    ),
}

print("[PASS] Cross-section run-name consistency check passed.")
print(
    f"[PASS] Theta shards found: "
    f"{len(theta_shard_files):,}"
)
print(
    "[PASS] Patent hierarchy records: "
    f"{split_hierarchy_counts}"
)


# ============================================================
# 8. Build complete file inventory
# ============================================================

# Exclude Section 10 itself to prevent recursive inventory growth.
result_files = list_files_recursively_section10(
    RUN_RESULT_DIR,
    excluded_directories=[
        SECTION10_RESULT_DIR,
    ],
)

checkpoint_files = list_files_recursively_section10(
    RUN_CHECKPOINT_DIR,
)

all_source_files = sorted(
    set(
        result_files
        + checkpoint_files
        + list(
            required_files.values()
        )
    )
)

inventory_records = []
total_inventory_bytes = 0
checksum_count = 0
checksum_skipped_count = 0

print("\n=== BUILDING FILE INVENTORY ===")
print(
    f"Files discovered: {len(all_source_files):,}"
)

for file_index, file_path in enumerate(
    all_source_files,
    start=1,
):
    file_size = os.path.getsize(
        file_path
    )

    total_inventory_bytes += file_size

    modified_timestamp = os.path.getmtime(
        file_path
    )

    if (
        file_size
        <= MAX_CHECKSUM_FILE_SIZE_BYTES
    ):
        sha256_value = (
            sha256_file_section10(
                file_path
            )
        )

        checksum_status = (
            "computed"
        )

        checksum_count += 1
    else:
        sha256_value = None
        checksum_status = (
            "skipped_due_to_size"
        )

        checksum_skipped_count += 1

    inventory_records.append(
        {
            "relative_path": (
                safe_relative_path_section10(
                    file_path
                )
            ),
            "absolute_path": (
                file_path
            ),
            "size_bytes": (
                file_size
            ),
            "size_human": (
                human_readable_size_section10(
                    file_size
                )
            ),
            "modified_at": (
                datetime.fromtimestamp(
                    modified_timestamp
                ).isoformat()
            ),
            "sha256": (
                sha256_value
            ),
            "checksum_status": (
                checksum_status
            ),
        }
    )

    if (
        file_index % 100 == 0
        or file_index == len(
            all_source_files
        )
    ):
        print(
            f"  inventoried "
            f"{file_index:,}/"
            f"{len(all_source_files):,} files"
        )

inventory_path = os.path.join(
    SECTION10_AUDIT_DIR,
    "file_inventory.json",
)

atomic_json_save_section10(
    {
        "run_name": RUN_NAME,
        "created_at": (
            datetime.now().isoformat()
        ),
        "num_files": len(
            inventory_records
        ),
        "total_size_bytes": (
            total_inventory_bytes
        ),
        "total_size_human": (
            human_readable_size_section10(
                total_inventory_bytes
            )
        ),
        "checksums_computed": (
            checksum_count
        ),
        "checksums_skipped": (
            checksum_skipped_count
        ),
        "files": inventory_records,
    },
    inventory_path,
)

checksum_text_lines = []

for record in inventory_records:
    if record["sha256"] is None:
        continue

    checksum_text_lines.append(
        f"{record['sha256']}  "
        f"{record['relative_path']}"
    )

checksum_text = (
    "\n".join(
        checksum_text_lines
    )
    + (
        "\n"
        if checksum_text_lines
        else ""
    )
)

checksum_path = os.path.join(
    SECTION10_AUDIT_DIR,
    "SHA256SUMS.txt",
)

atomic_text_save_section10(
    checksum_text,
    checksum_path,
)

print(
    "[PASS] File inventory created: "
    f"{len(inventory_records):,} files, "
    f"{human_readable_size_section10(total_inventory_bytes)}"
)


# ============================================================
# 9. Save reproducibility/environment information
# ============================================================

cuda_information = {
    "available": torch.cuda.is_available(),
    "torch_cuda_version": (
        torch.version.cuda
    ),
    "cudnn_version": (
        torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None
    ),
    "device_count": (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0
    ),
    "devices": [],
}

if torch.cuda.is_available():
    for device_index in range(
        torch.cuda.device_count()
    ):
        device_properties = (
            torch.cuda.get_device_properties(
                device_index
            )
        )

        cuda_information[
            "devices"
        ].append(
            {
                "index": (
                    device_index
                ),
                "name": (
                    device_properties.name
                ),
                "total_memory_bytes": (
                    device_properties.total_memory
                ),
                "total_memory_human": (
                    human_readable_size_section10(
                        device_properties.total_memory
                    )
                ),
                "compute_capability": (
                    f"{device_properties.major}."
                    f"{device_properties.minor}"
                ),
            }
        )

environment_report = {
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(),
    "feature_run_name": FEATURE_RUN_NAME,
    "python": {
        "version": sys.version,
        "executable": sys.executable,
        "implementation": (
            platform.python_implementation()
        ),
    },
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    },
    "libraries": {
        "torch": torch.__version__,
        "numpy": np.__version__,
    },
    "cuda": cuda_information,
    "configuration": (
        config_to_dictionary_section10(
            CONFIG
        )
    ),
    "important_paths": {
        "run_result_directory": (
            RUN_RESULT_DIR
        ),
        "run_checkpoint_directory": (
            RUN_CHECKPOINT_DIR
        ),
        "section7_directory": (
            SECTION7_RESULT_DIR
        ),
        "section8_directory": (
            SECTION8_RESULT_DIR
        ),
        "section9_directory": (
            SECTION9_RESULT_DIR
        ),
        "section10_directory": (
            SECTION10_RESULT_DIR
        ),
    },
}

environment_report_path = os.path.join(
    SECTION10_AUDIT_DIR,
    "reproducibility_environment.json",
)

atomic_json_save_section10(
    environment_report,
    environment_report_path,
)


# ============================================================
# 10. Save final audit report
# ============================================================

final_audit_report = {
    "status": "passed",
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(),
    "feature_run_name": FEATURE_RUN_NAME,
    "validation": {
        "all_required_files_exist": True,
        "required_file_count": len(
            required_files
        ),
        "structured_file_validation": (
            validation_details
        ),
        "semantic_checks": (
            semantic_checks
        ),
    },
    "inventory": {
        "num_files": len(
            inventory_records
        ),
        "total_size_bytes": (
            total_inventory_bytes
        ),
        "total_size_human": (
            human_readable_size_section10(
                total_inventory_bytes
            )
        ),
        "checksums_computed": (
            checksum_count
        ),
        "checksums_skipped": (
            checksum_skipped_count
        ),
        "inventory_path": (
            inventory_path
        ),
        "checksum_path": (
            checksum_path
        ),
    },
    "training": {
        "best_epoch_one_based": (
            best_epoch_one_based
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "stopped_early": (
            training_summary.get(
                "stopped_early"
            )
        ),
        "final_dev": (
            training_summary.get(
                "final_dev"
            )
        ),
        "final_test": (
            training_summary.get(
                "final_test"
            )
        ),
    },
    "section7": {
        "num_theta_shard_files": len(
            theta_shard_files
        ),
    },
    "section9": {
        "patent_hierarchy_record_counts": (
            split_hierarchy_counts
        ),
    },
    "bundle_configuration": {
        "include_best_checkpoint": (
            INCLUDE_BEST_CHECKPOINT_IN_BUNDLE
        ),
        "include_theta_shards": (
            INCLUDE_THETA_SHARDS_IN_BUNDLE
        ),
        "include_periodic_checkpoints": (
            INCLUDE_PERIODIC_CHECKPOINTS_IN_BUNDLE
        ),
        "include_section8_figures": (
            INCLUDE_SECTION8_FIGURES_IN_BUNDLE
        ),
    },
}

final_audit_report_path = os.path.join(
    SECTION10_AUDIT_DIR,
    "final_audit_report.json",
)

atomic_json_save_section10(
    final_audit_report,
    final_audit_report_path,
)

print(
    "[PASS] Final audit report saved: "
    f"{final_audit_report_path}"
)


# ============================================================
# 11. Create final README
# ============================================================

readme_text = f"""# Depth-OT Patent Topic Hierarchy: Final Results

## Run information

- Run name: `{RUN_NAME}`
- Feature run: `{FEATURE_RUN_NAME}`
- Section 10 completed: `{datetime.now().isoformat()}`
- Best epoch: `{best_epoch_one_based}`
- Best validation loss: `{best_validation_loss}`
- Early stopping used: `{training_summary.get("stopped_early")}`

## Pipeline

1. Sections 0-2: environment, vocabulary, patent dataset and batching
2. Sections 3-4: dependency encoder, topic hierarchy and optimal transport
3. Section 5: model, losses and optimizers
4. Section 6: model training and checkpointing
5. Section 7: deterministic inference and global hierarchy extraction
6. Section 8: topic-quality and hierarchy evaluation
7. Section 9: patent-level export and publication tables
8. Section 10: final validation, reproducibility audit and packaging

## Principal outputs

### Section 6

- Best model checkpoint: `{section6_files["best_checkpoint"]}`
- Latest checkpoint: `{section6_files["latest_checkpoint"]}`
- Training summary: `{section6_files["training_summary"]}`
- Learned anchor: `{section6_files["learned_anchor"]}`

### Section 7

- Global hierarchy: `{section7_files["global_topic_hierarchy"]}`
- Transition matrices: `{section7_files["global_transition_matrices"]}`
- Topic-word distribution: `{section7_files["topic_word_distribution"]}`
- Inference manifest: `{section7_files["inference_manifest"]}`
- Claim-theta shard count: `{len(theta_shard_files)}`

### Section 8

- Evaluation report: `{section8_files["evaluation_report"]}`
- Per-topic table: `{section8_files["per_topic_evaluation"]}`
- Depth-anchor alignment: `{section8_files["depth_anchor_alignment"]}`

### Section 9

- Section 9 manifest: `{section9_files["section9_manifest"]}`
- Publication summary: `{section9_files["publication_summary"]}`
- Patent hierarchy counts: `{split_hierarchy_counts}`

### Section 10

- Final audit report: `{final_audit_report_path}`
- File inventory: `{inventory_path}`
- SHA-256 checksums: `{checksum_path}`
- Reproducibility environment: `{environment_report_path}`

## Bundle policy

- Best checkpoint included in ZIP: `{INCLUDE_BEST_CHECKPOINT_IN_BUNDLE}`
- Claim theta shards included in ZIP: `{INCLUDE_THETA_SHARDS_IN_BUNDLE}`
- Periodic checkpoints included in ZIP: `{INCLUDE_PERIODIC_CHECKPOINTS_IN_BUNDLE}`
- Section 8 figures included in ZIP: `{INCLUDE_SECTION8_FIGURES_IN_BUNDLE}`

Large checkpoints and claim-theta shards are excluded by default to keep
the publication bundle manageable. Their original locations are recorded
in the audit reports.

## Integrity verification

The file `audit/SHA256SUMS.txt` contains SHA-256 checksums. On Linux,
the included files can be verified using:

```bash
sha256sum -c SHA256SUMS.txt




