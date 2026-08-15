# ==================================================================================================
# FORCE RESUME FROM THE ORIGINAL DEPTH-OT V2 RUN
# ==================================================================================================

from pathlib import Path
import torch
import json
import os

ORIGINAL_RUN_NAME = (
    "depth_ot_v2_patent_semantic_seed42_20260814_055110"
)

ORIGINAL_CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/depth_ot_patent/checkpoints/depth_ot_v2"
) / ORIGINAL_RUN_NAME

ORIGINAL_LATEST_PATH = (
    ORIGINAL_CHECKPOINT_DIR
    / "latest.pt"
)

print("=" * 100)
print("DEPTH-OT V2 ORIGINAL RUN RECOVERY")
print("=" * 100)
print("Original run name     :", ORIGINAL_RUN_NAME)
print("Checkpoint directory  :", ORIGINAL_CHECKPOINT_DIR)
print("Latest checkpoint     :", ORIGINAL_LATEST_PATH)
print("Latest exists         :", ORIGINAL_LATEST_PATH.is_file())

if not ORIGINAL_LATEST_PATH.is_file():
    raise FileNotFoundError(
        f"Original latest.pt not found:\n"
        f"{ORIGINAL_LATEST_PATH}"
    )

# ----------------------------------------------------------------------------------------------
# 체크포인트 자체 검증
# ----------------------------------------------------------------------------------------------

checkpoint = torch.load(
    ORIGINAL_LATEST_PATH,
    map_location="cpu",
    weights_only=False,
)

checkpoint_epoch = checkpoint.get(
    "epoch",
    checkpoint.get("epoch_one_based")
)

checkpoint_run_name = checkpoint.get(
    "run_name"
)

print("Checkpoint epoch      :", checkpoint_epoch)
print("Checkpoint run name   :", checkpoint_run_name)
print("Checkpoint keys       :", list(checkpoint.keys()))

if checkpoint_run_name != ORIGINAL_RUN_NAME:
    raise RuntimeError(
        "Checkpoint run-name mismatch: "
        f"observed={checkpoint_run_name}, "
        f"expected={ORIGINAL_RUN_NAME}"
    )

if checkpoint_epoch is None:
    raise RuntimeError(
        "Checkpoint epoch metadata is missing."
    )

# ----------------------------------------------------------------------------------------------
# CONFIG run name 복구
# ----------------------------------------------------------------------------------------------

if "CONFIG" not in globals():
    raise RuntimeError(
        "CONFIG가 없습니다. Section 0을 먼저 실행한 후 "
        "이 복구 셀을 실행하세요."
    )

print("\nBefore CONFIG.run_name:", getattr(CONFIG, "run_name", None))

CONFIG.run_name = ORIGINAL_RUN_NAME

print("After CONFIG.run_name :", CONFIG.run_name)

# 모델 버전은 그대로 유지
if hasattr(CONFIG, "model_version"):
    print("Model version         :", CONFIG.model_version)

# ----------------------------------------------------------------------------------------------
# 흔히 사용되는 전역 run/checkpoint 경로도 기존 run으로 교정
# ----------------------------------------------------------------------------------------------

possible_run_name_variables = [
    "RUN_NAME",
    "CURRENT_RUN_NAME",
    "SECTION6_RUN_NAME",
]

for variable_name in possible_run_name_variables:
    if variable_name in globals():
        old_value = globals()[variable_name]
        globals()[variable_name] = ORIGINAL_RUN_NAME

        print(
            f"[UPDATED] {variable_name}: "
            f"{old_value} -> {ORIGINAL_RUN_NAME}"
        )

possible_checkpoint_directory_variables = [
    "CHECKPOINT_DIR",
    "CHECKPOINT_DIRECTORY",
    "CHECKPOINT_ROOT",
    "SECTION6_CHECKPOINT_DIR",
    "SECTION6_CHECKPOINT_DIRECTORY",
    "RUN_CHECKPOINT_DIR",
    "RUN_CHECKPOINT_DIRECTORY",
]

for variable_name in possible_checkpoint_directory_variables:
    if variable_name in globals():
        old_value = globals()[variable_name]

        # CHECKPOINT_ROOT처럼 상위 디렉터리일 수 있는 변수는 함부로 바꾸지 않음
        old_text = str(old_value)

        if (
            "20260815_055640" in old_text
            or old_text.endswith(
                "depth_ot_v2_patent_semantic_seed42_20260815_055640"
            )
        ):
            globals()[variable_name] = (
                ORIGINAL_CHECKPOINT_DIR
            )

            print(
                f"[UPDATED] {variable_name}: "
                f"{old_value} -> "
                f"{ORIGINAL_CHECKPOINT_DIR}"
            )

# ----------------------------------------------------------------------------------------------
# Section 6에서 사용하는 latest 경로 변수도 교정
# ----------------------------------------------------------------------------------------------

possible_latest_variables = [
    "LATEST_CHECKPOINT_PATH",
    "LATEST_PATH",
    "SECTION6_LATEST_PATH",
    "LATEST_CHECKPOINT",
]

for variable_name in possible_latest_variables:
    if variable_name in globals():
        old_value = globals()[variable_name]
        old_text = str(old_value)

        if (
            "20260815_055640" in old_text
            or old_text.endswith("latest.pt")
        ):
            globals()[variable_name] = (
                ORIGINAL_LATEST_PATH
            )

            print(
                f"[UPDATED] {variable_name}: "
                f"{old_value} -> "
                f"{ORIGINAL_LATEST_PATH}"
            )

# ----------------------------------------------------------------------------------------------
# 현재 checkpoint 관련 전역 변수 출력
# ----------------------------------------------------------------------------------------------

print("\n" + "-" * 100)
print("CURRENT RUN/CHECKPOINT GLOBALS")
print("-" * 100)

for variable_name, value in sorted(
    list(globals().items()),
    key=lambda item: item[0],
):
    upper_name = variable_name.upper()

    if (
        "CHECKPOINT" in upper_name
        or upper_name in {
            "RUN_NAME",
            "CURRENT_RUN_NAME",
            "SECTION6_RUN_NAME",
        }
    ):
        if isinstance(
            value,
            (str, Path, int, float, bool, type(None)),
        ):
            print(
                f"{variable_name:40s}: "
                f"{value}"
            )

# ----------------------------------------------------------------------------------------------
# 새로 잘못 생성된 run에는 아무것도 복사하지 않음
# ----------------------------------------------------------------------------------------------

ACCIDENTAL_RUN_NAME = (
    "depth_ot_v2_patent_semantic_seed42_20260815_055640"
)

ACCIDENTAL_CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/depth_ot_patent/checkpoints/depth_ot_v2"
) / ACCIDENTAL_RUN_NAME

print("\n" + "-" * 100)
print("ACCIDENTAL RUN")
print("-" * 100)
print("Accidental directory :", ACCIDENTAL_CHECKPOINT_DIR)
print("Exists               :", ACCIDENTAL_CHECKPOINT_DIR.exists())

if ACCIDENTAL_CHECKPOINT_DIR.exists():
    accidental_files = list(
        ACCIDENTAL_CHECKPOINT_DIR.glob("*.pt")
    )

    print(
        "Accidental checkpoints:",
        [path.name for path in accidental_files],
    )

    print(
        "[INFO] 이 디렉터리는 현재 삭제하지 않습니다. "
        "기존 run resume 확인 후 정리할 수 있습니다."
    )

del checkpoint

print("\n" + "=" * 100)
print(
    f"[PASS] CONFIG.run_name restored to "
    f"{ORIGINAL_RUN_NAME}"
)
print(
    f"[EXPECTED RESUME] Last completed epoch: "
    f"{checkpoint_epoch}"
)
print(
    f"[EXPECTED START] Epoch: "
    f"{int(checkpoint_epoch) + 1}"
)
print("=" * 100)
print("[NEXT] Section 6 학습 셀을 다시 실행하세요.")


# ============================================================
# SECTION 6 — DEPTH-OT V2 L4 TRAINING
# Stage-wise training + resume + local cache + atomic checkpoint
# ============================================================

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ------------------------------------------------------------
# 6.0 Preconditions
# ------------------------------------------------------------

_REQUIRED_GLOBALS = [
    "CONFIG",
    "DEVICE",
    "depth_ot_v2_model",
    "patent_usage_tracker",
    "compute_depth_ot_v2_objective",
    "compute_anchor_objective",
    "neural_parameters",
    "anchor_parameters",
    "train_loader",
    "dev_loader",
    "test_loader",
    "RUN_LOG_DIR",
]

_missing_globals = [
    name for name in _REQUIRED_GLOBALS
    if name not in globals()
]

if _missing_globals:
    raise RuntimeError(
        "Section 6 실행 전에 필요한 전역 변수가 없습니다: "
        + ", ".join(_missing_globals)
        + "\nSection 0–5와 BoW 패치를 먼저 실행하세요."
    )

DEVICE = torch.device(DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        f"Section 6은 L4 CUDA 학습용입니다. Current device={DEVICE}"
    )

GPU_NAME = torch.cuda.get_device_name(DEVICE)
GPU_TOTAL_GIB = (
    torch.cuda.get_device_properties(DEVICE).total_memory
    / (1024 ** 3)
)

print(f"[GPU] {GPU_NAME} ({GPU_TOTAL_GIB:.2f} GiB)")

if "L4" not in GPU_NAME.upper():
    print(
        "[WARNING] NVIDIA L4가 아닙니다. 코드는 실행할 수 있지만 "
        "속도와 메모리 사용량이 예상과 다를 수 있습니다."
    )


# ------------------------------------------------------------
# 6.1 Configuration helper
# ------------------------------------------------------------

def _cfg(
    name: str,
    default: Any,
    *alternative_names: str,
) -> Any:
    for candidate in (name, *alternative_names):
        if hasattr(CONFIG, candidate):
            value = getattr(CONFIG, candidate)

            if value is not None:
                return value

    return default


RUN_NAME = str(
    _cfg(
        "run_name",
        "depth_ot_v2_patent_semantic_seed42",
    )
)

MODEL_VERSION = str(
    _cfg(
        "model_version",
        "depth_ot_v2_patent_semantic",
    )
)

SEED = int(_cfg("seed", 42))
NUM_EPOCHS = int(_cfg("num_epochs", 24))

BASE_LEARNING_RATE = float(
    _cfg(
        "base_learning_rate",
        3.0e-4,
        "learning_rate",
        "base_lr",
    )
)

TOPIC_LEARNING_RATE = float(
    _cfg(
        "topic_learning_rate",
        2.0e-4,
        "topic_lr",
    )
)

ANCHOR_LEARNING_RATE = float(
    _cfg(
        "anchor_learning_rate",
        1.0e-4,
        "anchor_lr",
    )
)

WEIGHT_DECAY = float(
    _cfg("weight_decay", 1.0e-5)
)

MAX_GRAD_NORM = float(
    _cfg("max_grad_norm", 1.0)
)

USE_AMP = bool(_cfg("use_amp", False))

if USE_AMP:
    print(
        "[WARNING] CONFIG.use_amp=True였지만 V2 검증 실험에서는 "
        "FP32를 사용하도록 비활성화합니다."
    )

USE_AMP = False

# The hierarchy solver itself remains float64 in Section 5.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


# ------------------------------------------------------------
# 6.2 Schedule configuration
# ------------------------------------------------------------

KL_START_EPOCH = int(
    _cfg("kl_start_epoch", 3)
)
KL_FULL_EPOCH = int(
    _cfg("kl_full_epoch", 12)
)
KL_MAX_WEIGHT = float(
    _cfg("kl_max_weight", 0.10)
)

USAGE_START_EPOCH = int(
    _cfg(
        "patent_usage_start_epoch",
        5,
        "usage_start_epoch",
    )
)
USAGE_FULL_EPOCH = int(
    _cfg(
        "patent_usage_full_epoch",
        10,
        "usage_full_epoch",
    )
)
USAGE_MAX_WEIGHT = float(
    _cfg(
        "patent_usage_max_weight",
        0.05,
    )
)

HIERARCHY_START_EPOCH = int(
    _cfg("hierarchy_start_epoch", 12)
)
HIERARCHY_FULL_EPOCH = int(
    _cfg("hierarchy_full_epoch", 18)
)
HIERARCHY_MAX_WEIGHT = float(
    _cfg("hierarchy_max_weight", 0.15)
)

ANCHOR_START_EPOCH = int(
    _cfg("anchor_start_epoch", 12)
)

RECONCILIATION_WEIGHT = float(
    globals().get(
        "RECONSTRUCTION_WEIGHT",
        _cfg("reconciliation_weight", 0.10),
    )
)

SEPARATION_FULL_EPOCH = int(
    _cfg("separation_full_epoch", 6)
)

TOPIC_EMBEDDING_SEPARATION_MAX_WEIGHT = float(
    globals().get(
        "TOPIC_EMBEDDING_SEPARATION_MAX_WEIGHT",
        _cfg(
            "topic_embedding_separation_max_weight",
            0.10,
        ),
    )
)

BETA_SEPARATION_MAX_WEIGHT = float(
    globals().get(
        "BETA_SEPARATION_MAX_WEIGHT",
        _cfg(
            "beta_separation_max_weight",
            0.02,
        ),
    )
)


def linear_warmup_after_start(
    epoch: int,
    start_epoch: int,
    full_epoch: int,
    maximum_weight: float,
) -> float:
    """
    The weight is zero through start_epoch and begins increasing in
    the following epoch. This matches the previous schedule semantics:
    hierarchy_start_epoch=12 means hierarchy begins at Epoch 13.
    """

    epoch = int(epoch)
    start_epoch = int(start_epoch)
    full_epoch = int(full_epoch)
    maximum_weight = float(maximum_weight)

    if epoch <= start_epoch:
        return 0.0

    if epoch >= full_epoch:
        return maximum_weight

    denominator = max(1, full_epoch - start_epoch)

    return maximum_weight * (
        (epoch - start_epoch) / denominator
    )


def linear_warmup_from_first_epoch(
    epoch: int,
    full_epoch: int,
    maximum_weight: float,
) -> float:
    if epoch <= 0:
        return 0.0

    if epoch >= full_epoch:
        return float(maximum_weight)

    return float(maximum_weight) * (
        epoch / max(1, full_epoch)
    )


def objective_weights_for_epoch(
    epoch: int,
) -> Dict[str, float]:
    hierarchy_weight = linear_warmup_after_start(
        epoch=epoch,
        start_epoch=HIERARCHY_START_EPOCH,
        full_epoch=HIERARCHY_FULL_EPOCH,
        maximum_weight=HIERARCHY_MAX_WEIGHT,
    )

    return {
        "kl": linear_warmup_after_start(
            epoch=epoch,
            start_epoch=KL_START_EPOCH,
            full_epoch=KL_FULL_EPOCH,
            maximum_weight=KL_MAX_WEIGHT,
        ),
        "hierarchy": hierarchy_weight,
        "reconciliation": (
            RECONCILIATION_WEIGHT
            if hierarchy_weight > 0.0
            else 0.0
        ),
        "topic_embedding_separation": (
            linear_warmup_from_first_epoch(
                epoch=epoch,
                full_epoch=SEPARATION_FULL_EPOCH,
                maximum_weight=(
                    TOPIC_EMBEDDING_SEPARATION_MAX_WEIGHT
                ),
            )
        ),
        "beta_separation": (
            linear_warmup_from_first_epoch(
                epoch=epoch,
                full_epoch=SEPARATION_FULL_EPOCH,
                maximum_weight=(
                    BETA_SEPARATION_MAX_WEIGHT
                ),
            )
        ),
        "patent_usage": linear_warmup_after_start(
            epoch=epoch,
            start_epoch=USAGE_START_EPOCH,
            full_epoch=USAGE_FULL_EPOCH,
            maximum_weight=USAGE_MAX_WEIGHT,
        ),
    }


def anchor_enabled_for_epoch(epoch: int) -> bool:
    return int(epoch) > ANCHOR_START_EPOCH


def hierarchy_enabled_for_epoch(epoch: int) -> bool:
    return (
        objective_weights_for_epoch(epoch)["hierarchy"]
        > 0.0
    )


def training_stage_for_epoch(epoch: int) -> str:
    if epoch <= KL_START_EPOCH:
        return "semantic_reconstruction"

    if epoch <= USAGE_START_EPOCH:
        return "semantic_plus_kl"

    if epoch <= HIERARCHY_START_EPOCH:
        return "semantic_plus_patent_usage"

    if epoch < HIERARCHY_FULL_EPOCH:
        return "hierarchy_warmup"

    return "full_depth_ot"


# ------------------------------------------------------------
# 6.3 Evaluation and checkpoint configuration
# ------------------------------------------------------------

FAST_DEV_MAX_BATCHES = int(
    _cfg("fast_dev_max_batches", 200)
)

FULL_DEV_INTERVAL = int(
    _cfg("full_dev_interval", 2)
)

CHECKPOINT_INTERVAL = int(
    _cfg("checkpoint_interval", 3)
)

CHECKPOINT_SELECTION_START_EPOCH = int(
    _cfg(
        "checkpoint_selection_start_epoch",
        HIERARCHY_START_EPOCH + 1,
    )
)

EARLY_STOP_START_EPOCH = int(
    _cfg(
        "early_stopping_start_epoch",
        HIERARCHY_FULL_EPOCH,
    )
)

EARLY_STOP_PATIENCE = int(
    _cfg("early_stopping_patience", 3)
)

EARLY_STOP_MIN_IMPROVEMENT = float(
    _cfg(
        "early_stopping_min_improvement",
        1.0e-4,
    )
)

RESUME_TRAINING = bool(
    _cfg("resume", True, "resume_training")
)

LOG_INTERVAL = int(
    _cfg("log_interval", 250)
)

DIAGNOSTIC_ACTIVE_TOPIC_MINIMUM = int(
    _cfg("diagnostic_active_topic_minimum", 20)
)

DIAGNOSTIC_MAX_SHARE_LIMIT = float(
    _cfg("diagnostic_max_share_limit", 0.30)
)

DIAGNOSTIC_ENTROPY_MINIMUM = float(
    _cfg("diagnostic_entropy_minimum", 0.70)
)

DIAGNOSTIC_BETA_MAX_COSINE_LIMIT = float(
    _cfg(
        "diagnostic_beta_max_cosine_limit",
        0.95,
    )
)


# ------------------------------------------------------------
# 6.4 Directories
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    globals().get(
        "PROJECT_ROOT",
        "/content/drive/MyDrive/depth_ot_patent",
    )
)

RUN_LOG_DIR = Path(RUN_LOG_DIR)

RUN_RESULT_DIR = Path(
    globals().get(
        "RUN_RESULT_DIR",
        PROJECT_ROOT
        / "results"
        / "depth_ot_v2"
        / RUN_NAME,
    )
)

RUN_CHECKPOINT_DIR = Path(
    globals().get(
        "RUN_CHECKPOINT_DIR",
        PROJECT_ROOT
        / "checkpoints"
        / "depth_ot_v2"
        / RUN_NAME,
    )
)

RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
RUN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT_PATH = (
    RUN_CHECKPOINT_DIR / "latest.pt"
)
BEST_CHECKPOINT_PATH = (
    RUN_CHECKPOINT_DIR / "best.pt"
)
BEST_PRE_HIERARCHY_CHECKPOINT_PATH = (
    RUN_CHECKPOINT_DIR / "best_pre_hierarchy.pt"
)

TRAINING_HISTORY_JSON_PATH = (
    RUN_LOG_DIR / "section6_training_history.json"
)
TRAINING_HISTORY_CSV_PATH = (
    RUN_LOG_DIR / "section6_training_history.csv"
)
SECTION6_MANIFEST_PATH = (
    RUN_LOG_DIR / "section6_training_manifest.json"
)
TRAINING_SUMMARY_PATH = (
    RUN_RESULT_DIR / "section6_training_summary.json"
)
OOM_LOG_PATH = (
    RUN_LOG_DIR / "section6_oom_diagnostics.jsonl"
)


# ------------------------------------------------------------
# 6.5 Reproducibility
# ------------------------------------------------------------

def set_training_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass


set_training_seed(SEED)


# ------------------------------------------------------------
# 6.6 Feature shard local cache
# ------------------------------------------------------------

ENABLE_LOCAL_SHARD_CACHE = bool(
    _cfg("enable_local_shard_cache", True)
)

LOCAL_SHARD_CACHE_DIR = Path(
    _cfg(
        "local_shard_cache_dir",
        "/content/depth_ot_v2_feature_cache",
    )
)

LOCAL_CACHE_MINIMUM_FREE_GIB = float(
    _cfg("local_cache_minimum_free_gib", 8.0)
)

LOCAL_CACHE_MAXIMUM_GIB = float(
    _cfg("local_cache_maximum_gib", 12.0)
)

IO_RETRY_COUNT = int(
    _cfg("io_retry_count", 8)
)

IO_RETRY_INITIAL_SECONDS = float(
    _cfg("io_retry_initial_seconds", 1.0)
)

FEATURE_ROOT_FOR_CACHE = Path(
    globals().get(
        "FEATURE_ROOT",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "token_features"
        / "full",
    )
)

LOCAL_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "copies": 0,
    "evictions": 0,
    "copy_failures": 0,
    "bytes_copied": 0,
}


def _path_is_inside(
    path: Path,
    root: Path,
) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
        return True
    except Exception:
        return False


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0

    for child in path.iterdir():
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue

    return total


def _local_free_bytes() -> int:
    return int(
        shutil.disk_usage("/content").free
    )


def _cache_file_for_source(source_path: Path) -> Path:
    digest = hashlib.sha1(
        str(source_path.absolute()).encode("utf-8")
    ).hexdigest()[:16]

    return LOCAL_SHARD_CACHE_DIR / (
        f"{digest}_{source_path.name}"
    )


def _evict_local_cache_if_needed(
    required_bytes: int,
    protected_path: Optional[Path] = None,
) -> None:
    LOCAL_SHARD_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    maximum_bytes = int(
        LOCAL_CACHE_MAXIMUM_GIB * (1024 ** 3)
    )
    minimum_free_bytes = int(
        LOCAL_CACHE_MINIMUM_FREE_GIB * (1024 ** 3)
    )

    while True:
        current_size = _directory_size_bytes(
            LOCAL_SHARD_CACHE_DIR
        )
        current_free = _local_free_bytes()

        size_ok = (
            current_size + required_bytes
            <= maximum_bytes
        )
        free_ok = (
            current_free - required_bytes
            >= minimum_free_bytes
        )

        if size_ok and free_ok:
            return

        candidates = []

        for child in LOCAL_SHARD_CACHE_DIR.iterdir():
            if not child.is_file():
                continue

            if (
                protected_path is not None
                and child == protected_path
            ):
                continue

            if child.name.endswith(".tmp"):
                continue

            try:
                stat = child.stat()
                candidates.append(
                    (stat.st_atime, stat.st_mtime, child)
                )
            except OSError:
                continue

        if not candidates:
            raise RuntimeError(
                "Local feature cache 공간을 확보할 수 없습니다. "
                f"Required={required_bytes / 2**30:.2f} GiB, "
                f"Free={current_free / 2**30:.2f} GiB."
            )

        candidates.sort(key=lambda item: (item[0], item[1]))
        victim = candidates[0][2]

        try:
            victim.unlink()
            LOCAL_CACHE_STATS["evictions"] += 1
        except OSError:
            time.sleep(0.2)


def _copy_feature_to_local_cache(
    source_path: Path,
) -> Path:
    source_size = int(source_path.stat().st_size)
    cache_path = _cache_file_for_source(source_path)

    if cache_path.exists():
        try:
            if cache_path.stat().st_size == source_size:
                os.utime(cache_path, None)
                LOCAL_CACHE_STATS["hits"] += 1
                return cache_path
        except OSError:
            pass

        cache_path.unlink(missing_ok=True)

    LOCAL_CACHE_STATS["misses"] += 1

    _evict_local_cache_if_needed(
        required_bytes=source_size,
        protected_path=cache_path,
    )

    for attempt in range(1, IO_RETRY_COUNT + 1):
        temporary_path = cache_path.with_name(
            cache_path.name
            + f".{os.getpid()}.{attempt}.tmp"
        )

        try:
            temporary_path.unlink(missing_ok=True)

            shutil.copy2(
                source_path,
                temporary_path,
            )

            copied_size = int(
                temporary_path.stat().st_size
            )

            if copied_size != source_size:
                raise IOError(
                    f"Copied size mismatch: "
                    f"{copied_size} != {source_size}"
                )

            os.replace(
                temporary_path,
                cache_path,
            )

            LOCAL_CACHE_STATS["copies"] += 1
            LOCAL_CACHE_STATS["bytes_copied"] += (
                source_size
            )

            return cache_path

        except Exception:
            LOCAL_CACHE_STATS["copy_failures"] += 1
            temporary_path.unlink(missing_ok=True)

            if attempt >= IO_RETRY_COUNT:
                # If local caching fails, the caller can still load
                # directly from Drive.
                return source_path

            delay = min(
                30.0,
                IO_RETRY_INITIAL_SECONDS
                * (2 ** (attempt - 1)),
            )
            time.sleep(delay)

    return source_path


if not hasattr(
    torch,
    "_depth_ot_v2_original_load",
):
    torch._depth_ot_v2_original_load = torch.load

_ORIGINAL_TORCH_LOAD = (
    torch._depth_ot_v2_original_load
)


def _depth_ot_v2_cached_torch_load(
    file: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if (
        ENABLE_LOCAL_SHARD_CACHE
        and isinstance(file, (str, os.PathLike, Path))
    ):
        source_path = Path(file)

        if (
            source_path.exists()
            and source_path.is_file()
            and _path_is_inside(
                source_path,
                FEATURE_ROOT_FOR_CACHE,
            )
        ):
            local_path = _copy_feature_to_local_cache(
                source_path
            )

            last_error = None

            for attempt in range(
                1,
                IO_RETRY_COUNT + 1,
            ):
                try:
                    return _ORIGINAL_TORCH_LOAD(
                        local_path,
                        *args,
                        **kwargs,
                    )
                except Exception as error:
                    last_error = error

                    if (
                        local_path != source_path
                        and local_path.exists()
                    ):
                        local_path.unlink(
                            missing_ok=True
                        )

                    if attempt >= IO_RETRY_COUNT:
                        break

                    time.sleep(
                        min(
                            30.0,
                            IO_RETRY_INITIAL_SECONDS
                            * (2 ** (attempt - 1)),
                        )
                    )

            raise RuntimeError(
                f"Feature shard load failed after "
                f"{IO_RETRY_COUNT} attempts: {source_path}"
            ) from last_error

    return _ORIGINAL_TORCH_LOAD(
        file,
        *args,
        **kwargs,
    )


if ENABLE_LOCAL_SHARD_CACHE:
    LOCAL_SHARD_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.load = _depth_ot_v2_cached_torch_load
    print(
        f"[LOCAL CACHE] Enabled: "
        f"{LOCAL_SHARD_CACHE_DIR}"
    )
else:
    print("[LOCAL CACHE] Disabled")


# ------------------------------------------------------------
# 6.7 Device utilities
# ------------------------------------------------------------

def move_batch_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if torch.is_tensor(value):
        return value.to(
            device,
            non_blocking=True,
        )

    if isinstance(value, dict):
        return {
            key: move_batch_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            move_batch_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            move_batch_to_device(item, device)
            for item in value
        )

    return value


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def extract_batch_identity(
    batch: Mapping[str, Any],
) -> Dict[str, Any]:
    patent_ids = batch.get("patent_ids")
    claim_ids = batch.get("claim_ids")

    def convert(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (list, tuple)):
            return list(value)

        return value

    output = {
        "patent_ids": convert(patent_ids),
        "claim_ids_preview": None,
    }

    converted_claim_ids = convert(claim_ids)

    if isinstance(converted_claim_ids, list):
        output["claim_ids_preview"] = (
            converted_claim_ids[:20]
        )

    return output


# ------------------------------------------------------------
# 6.8 Optimizers
# ------------------------------------------------------------

anchor_parameter_ids = {
    id(parameter)
    for parameter in anchor_parameters
}

topic_parameters = list(
    depth_ot_v2_model
    .topic_word_decoder
    .parameters()
)

topic_parameter_ids = {
    id(parameter)
    for parameter in topic_parameters
}

base_parameters = [
    parameter
    for parameter in depth_ot_v2_model.parameters()
    if (
        id(parameter) not in anchor_parameter_ids
        and id(parameter) not in topic_parameter_ids
    )
]

if not base_parameters:
    raise RuntimeError(
        "No base neural parameters were found."
    )

if not topic_parameters:
    raise RuntimeError(
        "No topic decoder parameters were found."
    )

neural_optimizer = torch.optim.AdamW(
    [
        {
            "params": base_parameters,
            "lr": BASE_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "name": "base_neural",
        },
        {
            "params": topic_parameters,
            "lr": TOPIC_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "name": "topic_decoder",
        },
    ],
    betas=(0.9, 0.999),
    eps=1.0e-8,
)

anchor_optimizer = torch.optim.AdamW(
    [
        {
            "params": anchor_parameters,
            "lr": ANCHOR_LEARNING_RATE,
            "weight_decay": 0.0,
            "name": "depth_anchors",
        }
    ],
    betas=(0.9, 0.999),
    eps=1.0e-8,
)


# ------------------------------------------------------------
# 6.9 Epoch statistics
# ------------------------------------------------------------

TRACKED_LOSSES = [
    "total",
    "reconstruction",
    "kl",
    "hierarchy",
    "reconciliation",
    "topic_embedding_separation",
    "beta_separation",
    "patent_usage",
]


class EpochAccumulator:

    def __init__(self, number_of_topics: int) -> None:
        self.number_of_topics = int(number_of_topics)

        self.loss_sums = defaultdict(float)
        self.loss_weight = 0.0

        self.number_of_batches = 0
        self.number_of_patents = 0
        self.number_of_claims = 0
        self.number_of_valid_bow_claims = 0
        self.number_of_empty_bow_claims = 0

        self.soft_topic_sum = torch.zeros(
            self.number_of_topics,
            dtype=torch.float64,
        )
        self.soft_topic_square_sum = torch.zeros(
            self.number_of_topics,
            dtype=torch.float64,
        )
        self.hard_topic_counts = torch.zeros(
            self.number_of_topics,
            dtype=torch.long,
        )

        self.assignment_entropy_sum = 0.0
        self.assignment_max_probability_sum = 0.0

        self.hierarchy_input_edges = 0
        self.hierarchy_adjacent_edges = 0
        self.maximum_sinkhorn_error = 0.0
        self.sinkhorn_iteration_sum = 0.0
        self.sinkhorn_iteration_count = 0

        self.anchor_total_sum = 0.0
        self.anchor_directional_sum = 0.0
        self.anchor_updates = 0

    def update(
        self,
        losses: Mapping[str, Any],
        outputs: Mapping[str, Any],
        anchor_losses: Optional[Mapping[str, Any]] = None,
    ) -> None:
        patent_theta = (
            outputs["patent_theta"]
            .detach()
            .float()
            .cpu()
        )

        claim_theta = outputs["claim_theta"]

        number_of_patents = int(
            patent_theta.shape[0]
        )
        number_of_claims = int(
            claim_theta.shape[0]
        )

        batch_weight = max(1, number_of_patents)

        for name in TRACKED_LOSSES:
            value = losses[name]

            if torch.is_tensor(value):
                value = float(
                    value.detach().cpu().item()
                )

            self.loss_sums[name] += (
                float(value) * batch_weight
            )

        self.loss_weight += batch_weight
        self.number_of_batches += 1
        self.number_of_patents += number_of_patents
        self.number_of_claims += number_of_claims

        self.number_of_valid_bow_claims += int(
            losses["valid_bow_claims"]
            .detach()
            .cpu()
            .item()
        )
        self.number_of_empty_bow_claims += int(
            losses["empty_bow_claims"]
            .detach()
            .cpu()
            .item()
        )

        self.soft_topic_sum += (
            patent_theta.double().sum(dim=0)
        )
        self.soft_topic_square_sum += (
            patent_theta.double().pow(2).sum(dim=0)
        )

        hard_topic = patent_theta.argmax(dim=-1)

        self.hard_topic_counts += torch.bincount(
            hard_topic,
            minlength=self.number_of_topics,
        )

        entropy = -torch.sum(
            patent_theta.clamp_min(1.0e-8)
            * torch.log(
                patent_theta.clamp_min(1.0e-8)
            ),
            dim=-1,
        ) / math.log(self.number_of_topics)

        self.assignment_entropy_sum += float(
            entropy.sum().item()
        )
        self.assignment_max_probability_sum += float(
            patent_theta.max(dim=-1).values.sum().item()
        )

        self.hierarchy_input_edges += int(
            losses["hierarchy_number_of_input_edges"]
        )
        self.hierarchy_adjacent_edges += int(
            losses[
                "hierarchy_number_of_adjacent_edges"
            ]
        )

        self.maximum_sinkhorn_error = max(
            self.maximum_sinkhorn_error,
            float(losses["sinkhorn_maximum_error"]),
        )

        if (
            int(
                losses[
                    "hierarchy_number_of_depth_groups"
                ]
            )
            > 0
        ):
            self.sinkhorn_iteration_sum += float(
                losses["sinkhorn_mean_iterations"]
            )
            self.sinkhorn_iteration_count += 1

        if anchor_losses is not None:
            self.anchor_total_sum += float(
                anchor_losses["anchor_total"]
                .detach()
                .cpu()
                .item()
            )
            self.anchor_directional_sum += float(
                anchor_losses["anchor_directional"]
                .detach()
                .cpu()
                .item()
            )
            self.anchor_updates += 1

    def finalize(
        self,
        model: nn.Module,
        elapsed_seconds: float,
        peak_gpu_gib: float,
    ) -> Dict[str, Any]:
        if self.number_of_patents <= 0:
            raise RuntimeError(
                "No patents were processed in this epoch."
            )

        soft_marginal = (
            self.soft_topic_sum
            / self.soft_topic_sum.sum().clamp_min(
                1.0e-12
            )
        )

        hard_marginal = (
            self.hard_topic_counts.double()
            / self.hard_topic_counts.sum().clamp_min(1)
        )

        soft_entropy = float(
            (
                -torch.sum(
                    soft_marginal.clamp_min(1.0e-12)
                    * torch.log(
                        soft_marginal.clamp_min(1.0e-12)
                    )
                )
                / math.log(self.number_of_topics)
            ).item()
        )

        hard_entropy = float(
            (
                -torch.sum(
                    hard_marginal.clamp_min(1.0e-12)
                    * torch.log(
                        hard_marginal.clamp_min(1.0e-12)
                    )
                )
                / math.log(self.number_of_topics)
            ).item()
        )

        mean_theta = (
            self.soft_topic_sum
            / self.number_of_patents
        )
        mean_theta_square = (
            self.soft_topic_square_sum
            / self.number_of_patents
        )

        mean_topic_variance = float(
            (
                mean_theta_square
                - mean_theta.pow(2)
            )
            .clamp_min(0.0)
            .mean()
            .item()
        )

        with torch.no_grad():
            beta = model.get_beta().detach()
            normalized_beta = torch.nn.functional.normalize(
                beta,
                p=2,
                dim=-1,
                eps=1.0e-8,
            )
            beta_cosine = torch.matmul(
                normalized_beta,
                normalized_beta.transpose(0, 1),
            )
            mask = ~torch.eye(
                beta_cosine.shape[0],
                dtype=torch.bool,
                device=beta_cosine.device,
            )
            beta_values = beta_cosine[mask]

            topic_embeddings = (
                model.get_topic_embeddings().detach()
            )
            normalized_topic_embeddings = (
                torch.nn.functional.normalize(
                    topic_embeddings,
                    p=2,
                    dim=-1,
                    eps=1.0e-8,
                )
            )
            topic_cosine = torch.matmul(
                normalized_topic_embeddings,
                normalized_topic_embeddings.transpose(0, 1),
            )
            topic_values = topic_cosine[mask]

        result = {
            name: (
                self.loss_sums[name]
                / max(1.0, self.loss_weight)
            )
            for name in TRACKED_LOSSES
        }

        result.update({
            "number_of_batches": self.number_of_batches,
            "number_of_patents": self.number_of_patents,
            "number_of_claims": self.number_of_claims,
            "valid_bow_claims": (
                self.number_of_valid_bow_claims
            ),
            "empty_bow_claims": (
                self.number_of_empty_bow_claims
            ),
            "active_topics": int(
                (self.hard_topic_counts > 0)
                .sum()
                .item()
            ),
            "maximum_topic_share": float(
                hard_marginal.max().item()
            ),
            "marginal_entropy": hard_entropy,
            "soft_marginal_entropy": soft_entropy,
            "mean_assignment_entropy": (
                self.assignment_entropy_sum
                / self.number_of_patents
            ),
            "mean_max_probability": (
                self.assignment_max_probability_sum
                / self.number_of_patents
            ),
            "mean_topic_variance": (
                mean_topic_variance
            ),
            "hard_topic_counts": (
                self.hard_topic_counts.tolist()
            ),
            "soft_topic_marginal": (
                soft_marginal.tolist()
            ),
            "beta_mean_cosine": float(
                beta_values.mean().item()
            ),
            "beta_max_cosine": float(
                beta_values.max().item()
            ),
            "topic_embedding_mean_cosine": float(
                topic_values.mean().item()
            ),
            "topic_embedding_max_cosine": float(
                topic_values.max().item()
            ),
            "hierarchy_input_edges": (
                self.hierarchy_input_edges
            ),
            "hierarchy_adjacent_edges": (
                self.hierarchy_adjacent_edges
            ),
            "sinkhorn_maximum_error": (
                self.maximum_sinkhorn_error
            ),
            "sinkhorn_mean_iterations": (
                self.sinkhorn_iteration_sum
                / max(
                    1,
                    self.sinkhorn_iteration_count,
                )
            ),
            "anchor_total": (
                self.anchor_total_sum
                / max(1, self.anchor_updates)
            ),
            "anchor_directional": (
                self.anchor_directional_sum
                / max(1, self.anchor_updates)
            ),
            "anchor_updates": self.anchor_updates,
            "elapsed_seconds": float(
                elapsed_seconds
            ),
            "patents_per_second": float(
                self.number_of_patents
                / max(1.0e-8, elapsed_seconds)
            ),
            "peak_gpu_gib": float(peak_gpu_gib),
        })

        return result


# ------------------------------------------------------------
# 6.10 Diagnostic validity and checkpoint score
# ------------------------------------------------------------

def compute_checkpoint_diagnostics(
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    active_topics = int(metrics["active_topics"])
    maximum_share = float(
        metrics["maximum_topic_share"]
    )
    marginal_entropy = float(
        metrics["marginal_entropy"]
    )
    beta_max_cosine = float(
        metrics["beta_max_cosine"]
    )

    valid = (
        active_topics
        >= DIAGNOSTIC_ACTIVE_TOPIC_MINIMUM
        and maximum_share
        <= DIAGNOSTIC_MAX_SHARE_LIMIT
        and marginal_entropy
        >= DIAGNOSTIC_ENTROPY_MINIMUM
        and beta_max_cosine
        <= DIAGNOSTIC_BETA_MAX_COSINE_LIMIT
    )

    collapse_penalty = (
        4.0
        * max(
            0.0,
            maximum_share - 0.25,
        )
        + 2.0
        * max(
            0.0,
            0.70 - marginal_entropy,
        )
        + max(
            0.0,
            (
                DIAGNOSTIC_ACTIVE_TOPIC_MINIMUM
                - active_topics
            )
            / max(
                1,
                DIAGNOSTIC_ACTIVE_TOPIC_MINIMUM,
            ),
        )
    )

    uniform_assignment_penalty = max(
        0.0,
        float(metrics["mean_assignment_entropy"])
        - 0.95,
    )

    low_variance_penalty = max(
        0.0,
        1.0e-4
        - float(metrics["mean_topic_variance"]),
    ) * 100.0

    score = (
        float(metrics["reconstruction"])
        + 0.25
        * float(metrics["beta_max_cosine"])
        + 0.10
        * float(metrics["beta_mean_cosine"])
        + collapse_penalty
        + uniform_assignment_penalty
        + low_variance_penalty
    )

    return {
        "valid": bool(valid),
        "selection_score": float(score),
        "collapse_penalty": float(
            collapse_penalty
        ),
        "uniform_assignment_penalty": float(
            uniform_assignment_penalty
        ),
        "low_variance_penalty": float(
            low_variance_penalty
        ),
    }


def candidate_is_better(
    candidate: Mapping[str, Any],
    incumbent: Optional[Mapping[str, Any]],
    minimum_improvement: float,
) -> bool:
    if incumbent is None:
        return True

    candidate_valid = bool(
        candidate["valid"]
    )
    incumbent_valid = bool(
        incumbent["valid"]
    )

    if candidate_valid and not incumbent_valid:
        return True

    if incumbent_valid and not candidate_valid:
        return False

    return (
        float(candidate["selection_score"])
        < float(incumbent["selection_score"])
        - float(minimum_improvement)
    )


# ------------------------------------------------------------
# 6.11 OOM diagnostics
# ------------------------------------------------------------

def append_oom_diagnostic(
    epoch: int,
    batch_index: int,
    batch: Mapping[str, Any],
    error: BaseException,
) -> None:
    identity = extract_batch_identity(batch)

    record = {
        "timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "error": repr(error),
        "gpu": GPU_NAME,
        "allocated_gib": (
            torch.cuda.memory_allocated(DEVICE)
            / (1024 ** 3)
        ),
        "reserved_gib": (
            torch.cuda.memory_reserved(DEVICE)
            / (1024 ** 3)
        ),
        **identity,
    }

    with OOM_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ------------------------------------------------------------
# 6.12 Training epoch
# ------------------------------------------------------------

def set_loader_epoch(
    loader: Any,
    epoch: int,
) -> None:
    candidates = [
        getattr(loader, "batch_sampler", None),
        getattr(loader, "sampler", None),
    ]

    for candidate in candidates:
        if (
            candidate is not None
            and hasattr(candidate, "set_epoch")
        ):
            candidate.set_epoch(int(epoch))
            return


def train_one_epoch(
    epoch: int,
) -> Dict[str, Any]:
    depth_ot_v2_model.train()
    set_loader_epoch(train_loader, epoch)

    weights = objective_weights_for_epoch(epoch)
    compute_hierarchy = (
        weights["hierarchy"] > 0.0
    )
    compute_anchor = anchor_enabled_for_epoch(
        epoch
    )

    accumulator = EpochAccumulator(
        number_of_topics=int(NUM_TOPICS)
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    start_time = time.perf_counter()

    progress = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} TRAIN",
        dynamic_ncols=True,
    )

    for batch_index, batch_cpu in progress:
        batch = None

        try:
            batch = move_batch_to_device(
                batch_cpu,
                DEVICE,
            )

            neural_optimizer.zero_grad(
                set_to_none=True
            )
            anchor_optimizer.zero_grad(
                set_to_none=True
            )

            outputs = depth_ot_v2_model(
                batch,
                sample=True,
                decode=True,
            )

            losses = compute_depth_ot_v2_objective(
                model=depth_ot_v2_model,
                outputs=outputs,
                weights=weights,
                update_usage_ema=True,
                compute_hierarchy=compute_hierarchy,
            )

            losses["total"].backward()

            neural_gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    neural_parameters,
                    max_norm=MAX_GRAD_NORM,
                    error_if_nonfinite=True,
                )
            )

            if not torch.isfinite(
                neural_gradient_norm
            ):
                raise FloatingPointError(
                    "Non-finite neural gradient norm."
                )

            neural_optimizer.step()

            anchor_losses = None

            if compute_anchor:
                anchor_optimizer.zero_grad(
                    set_to_none=True
                )

                anchor_losses = compute_anchor_objective(
                    model=depth_ot_v2_model,
                    outputs=outputs,
                    compute_anchor_loss=True,
                )

                anchor_losses[
                    "anchor_total"
                ].backward()

                anchor_gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        anchor_parameters,
                        max_norm=MAX_GRAD_NORM,
                        error_if_nonfinite=True,
                    )
                )

                if not torch.isfinite(
                    anchor_gradient_norm
                ):
                    raise FloatingPointError(
                        "Non-finite anchor gradient norm."
                    )

                anchor_optimizer.step()

            accumulator.update(
                losses=losses,
                outputs=outputs,
                anchor_losses=anchor_losses,
            )

            if (
                batch_index == 1
                or batch_index % LOG_INTERVAL == 0
                or batch_index == len(train_loader)
            ):
                progress.set_postfix({
                    "loss": (
                        f"{float(losses['total'].detach().item()):.4f}"
                    ),
                    "rec": (
                        f"{float(losses['reconstruction'].detach().item()):.4f}"
                    ),
                    "beta": (
                        f"{float(losses['beta_max_cosine'].detach().item()):.3f}"
                    ),
                    "gpu": (
                        f"{torch.cuda.max_memory_allocated(DEVICE) / 2**30:.1f}G"
                    ),
                })

            del losses
            del outputs
            del batch

        except torch.cuda.OutOfMemoryError as error:
            neural_optimizer.zero_grad(
                set_to_none=True
            )
            anchor_optimizer.zero_grad(
                set_to_none=True
            )

            diagnostic_batch = (
                batch_cpu
                if isinstance(batch_cpu, Mapping)
                else {}
            )

            append_oom_diagnostic(
                epoch=epoch,
                batch_index=batch_index,
                batch=diagnostic_batch,
                error=error,
            )

            if batch is not None:
                del batch

            gc.collect()
            torch.cuda.empty_cache()

            raise RuntimeError(
                f"L4 CUDA OOM at Epoch {epoch}, "
                f"batch {batch_index}. "
                f"Patent IDs={extract_batch_identity(diagnostic_batch).get('patent_ids')}. "
                f"Diagnostic saved to {OOM_LOG_PATH}. "
                "이 배치를 자동으로 건너뛰지 않았습니다."
            ) from error

        except Exception:
            neural_optimizer.zero_grad(
                set_to_none=True
            )
            anchor_optimizer.zero_grad(
                set_to_none=True
            )
            raise

    elapsed_seconds = (
        time.perf_counter() - start_time
    )
    peak_gpu_gib = (
        torch.cuda.max_memory_allocated(DEVICE)
        / (1024 ** 3)
    )

    result = accumulator.finalize(
        model=depth_ot_v2_model,
        elapsed_seconds=elapsed_seconds,
        peak_gpu_gib=peak_gpu_gib,
    )

    result["epoch"] = int(epoch)
    result["stage"] = training_stage_for_epoch(
        epoch
    )
    result["weights"] = weights
    result["hierarchy_enabled"] = (
        compute_hierarchy
    )
    result["anchor_enabled"] = compute_anchor

    return result


# ------------------------------------------------------------
# 6.13 Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(
    loader: Any,
    epoch: int,
    description: str,
    maximum_batches: Optional[int] = None,
) -> Dict[str, Any]:
    depth_ot_v2_model.eval()

    weights = objective_weights_for_epoch(epoch)
    compute_hierarchy = (
        weights["hierarchy"] > 0.0
    )

    accumulator = EpochAccumulator(
        number_of_topics=int(NUM_TOPICS)
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    start_time = time.perf_counter()

    if maximum_batches is None:
        total_batches = len(loader)
    else:
        total_batches = min(
            len(loader),
            int(maximum_batches),
        )

    progress = tqdm(
        enumerate(loader, start=1),
        total=total_batches,
        desc=description,
        dynamic_ncols=True,
    )

    for batch_index, batch_cpu in progress:
        if (
            maximum_batches is not None
            and batch_index > maximum_batches
        ):
            break

        batch = move_batch_to_device(
            batch_cpu,
            DEVICE,
        )

        outputs = depth_ot_v2_model(
            batch,
            sample=False,
            decode=True,
        )

        losses = compute_depth_ot_v2_objective(
            model=depth_ot_v2_model,
            outputs=outputs,
            weights=weights,
            update_usage_ema=False,
            compute_hierarchy=compute_hierarchy,
        )

        accumulator.update(
            losses=losses,
            outputs=outputs,
            anchor_losses=None,
        )

        del losses
        del outputs
        del batch

    elapsed_seconds = (
        time.perf_counter() - start_time
    )
    peak_gpu_gib = (
        torch.cuda.max_memory_allocated(DEVICE)
        / (1024 ** 3)
    )

    result = accumulator.finalize(
        model=depth_ot_v2_model,
        elapsed_seconds=elapsed_seconds,
        peak_gpu_gib=peak_gpu_gib,
    )

    diagnostics = compute_checkpoint_diagnostics(
        result
    )
    result.update(diagnostics)

    result["epoch"] = int(epoch)
    result["description"] = description
    result["weights"] = weights
    result["maximum_batches"] = (
        maximum_batches
    )
    result["full_evaluation"] = (
        maximum_batches is None
    )

    return result


# ------------------------------------------------------------
# 6.14 Atomic save/load
# ------------------------------------------------------------

def get_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: Optional[Mapping[str, Any]],
) -> None:
    if not state:
        return

    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch_cpu" in state:
        torch.set_rng_state(
            state["torch_cpu"]
        )

    if (
        "torch_cuda" in state
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            state["torch_cuda"]
        )


def atomic_torch_save(
    payload: Mapping[str, Any],
    destination: Path,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_temporary = Path(
        tempfile.mkstemp(
            prefix="depth_ot_v2_checkpoint_",
            suffix=".pt",
            dir="/content",
        )[1]
    )

    drive_temporary = destination.with_name(
        destination.name
        + f".{os.getpid()}.tmp"
    )

    try:
        torch.save(
            payload,
            local_temporary,
        )

        local_size = int(
            local_temporary.stat().st_size
        )

        if local_size <= 0:
            raise IOError(
                "Local checkpoint has zero size."
            )

        last_error = None

        for attempt in range(
            1,
            IO_RETRY_COUNT + 1,
        ):
            try:
                drive_temporary.unlink(
                    missing_ok=True
                )

                shutil.copy2(
                    local_temporary,
                    drive_temporary,
                )

                if (
                    int(drive_temporary.stat().st_size)
                    != local_size
                ):
                    raise IOError(
                        "Drive checkpoint size mismatch."
                    )

                os.replace(
                    drive_temporary,
                    destination,
                )

                return

            except Exception as error:
                last_error = error
                drive_temporary.unlink(
                    missing_ok=True
                )

                if attempt >= IO_RETRY_COUNT:
                    raise

                time.sleep(
                    min(
                        30.0,
                        IO_RETRY_INITIAL_SECONDS
                        * (2 ** (attempt - 1)),
                    )
                )

        raise RuntimeError(
            f"Failed to save checkpoint: {destination}"
        ) from last_error

    finally:
        local_temporary.unlink(
            missing_ok=True
        )
        drive_temporary.unlink(
            missing_ok=True
        )


def build_checkpoint_payload(
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    best_pre_hierarchy: Optional[Mapping[str, Any]],
    best_post_hierarchy: Optional[Mapping[str, Any]],
    full_dev_no_improvement_count: int,
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "run_name": RUN_NAME,
        "model_version": MODEL_VERSION,
        "epoch": int(epoch),
        "epoch_one_based": int(epoch),
        "model_state_dict": (
            depth_ot_v2_model.state_dict()
        ),
        "patent_usage_tracker_state_dict": (
            patent_usage_tracker.state_dict()
        ),
        "neural_optimizer_state_dict": (
            neural_optimizer.state_dict()
        ),
        "anchor_optimizer_state_dict": (
            anchor_optimizer.state_dict()
        ),
        "history": list(history),
        "best_pre_hierarchy": (
            best_pre_hierarchy
        ),
        "best_post_hierarchy": (
            best_post_hierarchy
        ),
        "full_dev_no_improvement_count": int(
            full_dev_no_improvement_count
        ),
        "rng_state": get_rng_state(),
        "configuration": {
            "seed": SEED,
            "num_epochs": NUM_EPOCHS,
            "base_learning_rate": (
                BASE_LEARNING_RATE
            ),
            "topic_learning_rate": (
                TOPIC_LEARNING_RATE
            ),
            "anchor_learning_rate": (
                ANCHOR_LEARNING_RATE
            ),
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "use_amp": False,
            "tf32": False,
        },
        "saved_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }


def load_checkpoint(
    checkpoint_path: Path,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)

    checkpoint = _ORIGINAL_TORCH_LOAD(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_run_name = checkpoint.get(
        "run_name"
    )

    if (
        checkpoint_run_name is not None
        and checkpoint_run_name != RUN_NAME
    ):
        raise RuntimeError(
            f"Checkpoint run mismatch: "
            f"{checkpoint_run_name} != {RUN_NAME}"
        )

    depth_ot_v2_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if (
        "patent_usage_tracker_state_dict"
        in checkpoint
    ):
        patent_usage_tracker.load_state_dict(
            checkpoint[
                "patent_usage_tracker_state_dict"
            ],
            strict=True,
        )

    if (
        "neural_optimizer_state_dict"
        in checkpoint
    ):
        neural_optimizer.load_state_dict(
            checkpoint[
                "neural_optimizer_state_dict"
            ]
        )
        optimizer_to_device(
            neural_optimizer,
            DEVICE,
        )

    if (
        "anchor_optimizer_state_dict"
        in checkpoint
    ):
        anchor_optimizer.load_state_dict(
            checkpoint[
                "anchor_optimizer_state_dict"
            ]
        )
        optimizer_to_device(
            anchor_optimizer,
            DEVICE,
        )

    depth_ot_v2_model.to(DEVICE)
    patent_usage_tracker.to(DEVICE)

    restore_rng_state(
        checkpoint.get("rng_state")
    )

    return checkpoint


# ------------------------------------------------------------
# 6.15 JSON/CSV logging
# ------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(
        value,
        (np.integer, np.floating),
    ):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def atomic_json_save(
    payload: Any,
    destination: Path,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name
        + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                json_safe(payload),
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary,
            destination,
        )

    finally:
        temporary.unlink(
            missing_ok=True
        )


def write_history_csv(
    history: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    rows = []

    for record in history:
        train_metrics = record["train"]
        dev_metrics = record["dev"]
        weights = record["weights"]

        rows.append({
            "epoch": record["epoch"],
            "stage": record["stage"],
            "dev_mode": record["dev_mode"],
            "train_total": train_metrics["total"],
            "train_reconstruction": train_metrics[
                "reconstruction"
            ],
            "dev_total": dev_metrics["total"],
            "dev_reconstruction": dev_metrics[
                "reconstruction"
            ],
            "dev_active_topics": dev_metrics[
                "active_topics"
            ],
            "dev_maximum_topic_share": dev_metrics[
                "maximum_topic_share"
            ],
            "dev_marginal_entropy": dev_metrics[
                "marginal_entropy"
            ],
            "dev_soft_marginal_entropy": dev_metrics[
                "soft_marginal_entropy"
            ],
            "dev_mean_assignment_entropy": dev_metrics[
                "mean_assignment_entropy"
            ],
            "dev_mean_topic_variance": dev_metrics[
                "mean_topic_variance"
            ],
            "dev_beta_mean_cosine": dev_metrics[
                "beta_mean_cosine"
            ],
            "dev_beta_max_cosine": dev_metrics[
                "beta_max_cosine"
            ],
            "dev_valid": dev_metrics["valid"],
            "selection_score": dev_metrics[
                "selection_score"
            ],
            "gamma_kl": weights["kl"],
            "gamma_hierarchy": weights[
                "hierarchy"
            ],
            "gamma_reconciliation": weights[
                "reconciliation"
            ],
            "lambda_topic_separation": weights[
                "topic_embedding_separation"
            ],
            "lambda_beta_separation": weights[
                "beta_separation"
            ],
            "lambda_patent_usage": weights[
                "patent_usage"
            ],
            "train_patents_per_second": train_metrics[
                "patents_per_second"
            ],
            "train_peak_gpu_gib": train_metrics[
                "peak_gpu_gib"
            ],
        })

    if not rows:
        return

    temporary = destination.with_name(
        destination.name
        + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary,
            destination,
        )

    finally:
        temporary.unlink(
            missing_ok=True
        )


# ------------------------------------------------------------
# 6.16 Initial validation
# ------------------------------------------------------------

def validate_training_interfaces() -> None:
    batch_cpu = next(iter(train_loader))
    batch = move_batch_to_device(
        batch_cpu,
        DEVICE,
    )

    depth_ot_v2_model.eval()

    with torch.no_grad():
        resolved = (
            depth_ot_v2_model.resolve_batch_tensors(
                batch
            )
        )

        claim_bow = resolved.get("claim_bow")

        if claim_bow is None:
            raise RuntimeError(
                "claim_bow is None. "
                "bow_normalized 패치를 먼저 실행하세요."
            )

        if (
            claim_bow.ndim != 2
            or claim_bow.shape[1] != VOCAB_SIZE
        ):
            raise RuntimeError(
                f"Invalid claim BoW shape: "
                f"{tuple(claim_bow.shape)}"
            )

        outputs = depth_ot_v2_model(
            batch,
            sample=False,
            decode=True,
        )

        validation_weights = (
            objective_weights_for_epoch(1)
        )

        losses = compute_depth_ot_v2_objective(
            model=depth_ot_v2_model,
            outputs=outputs,
            weights=validation_weights,
            update_usage_ema=False,
            compute_hierarchy=False,
        )

        if not torch.isfinite(
            losses["total"]
        ):
            raise FloatingPointError(
                "Initial training objective is non-finite."
            )

    del outputs
    del losses
    del batch

    depth_ot_v2_model.train()
    torch.cuda.empty_cache()


validate_training_interfaces()


# ------------------------------------------------------------
# 6.17 Manifest
# ------------------------------------------------------------

section6_manifest = {
    "created_at_utc": (
        datetime.now(timezone.utc).isoformat()
    ),
    "run_name": RUN_NAME,
    "model_version": MODEL_VERSION,
    "gpu": {
        "name": GPU_NAME,
        "total_gib": GPU_TOTAL_GIB,
        "device": str(DEVICE),
    },
    "training": {
        "seed": SEED,
        "epochs": NUM_EPOCHS,
        "train_batches": len(train_loader),
        "dev_batches": len(dev_loader),
        "test_batches": len(test_loader),
        "base_learning_rate": (
            BASE_LEARNING_RATE
        ),
        "topic_learning_rate": (
            TOPIC_LEARNING_RATE
        ),
        "anchor_learning_rate": (
            ANCHOR_LEARNING_RATE
        ),
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "amp": False,
        "tf32": False,
        "workers_expected": 0,
    },
    "schedules": {
        "kl": {
            "start_epoch": KL_START_EPOCH,
            "full_epoch": KL_FULL_EPOCH,
            "maximum_weight": KL_MAX_WEIGHT,
        },
        "patent_usage": {
            "start_epoch": USAGE_START_EPOCH,
            "full_epoch": USAGE_FULL_EPOCH,
            "maximum_weight": USAGE_MAX_WEIGHT,
        },
        "hierarchy": {
            "start_epoch": (
                HIERARCHY_START_EPOCH
            ),
            "first_active_epoch": (
                HIERARCHY_START_EPOCH + 1
            ),
            "full_epoch": (
                HIERARCHY_FULL_EPOCH
            ),
            "maximum_weight": (
                HIERARCHY_MAX_WEIGHT
            ),
        },
        "anchor": {
            "start_epoch": ANCHOR_START_EPOCH,
            "first_active_epoch": (
                ANCHOR_START_EPOCH + 1
            ),
        },
        "separation": {
            "full_epoch": (
                SEPARATION_FULL_EPOCH
            ),
            "topic_embedding_maximum": (
                TOPIC_EMBEDDING_SEPARATION_MAX_WEIGHT
            ),
            "beta_maximum": (
                BETA_SEPARATION_MAX_WEIGHT
            ),
        },
    },
    "evaluation": {
        "fast_dev_max_batches": (
            FAST_DEV_MAX_BATCHES
        ),
        "full_dev_interval": (
            FULL_DEV_INTERVAL
        ),
        "checkpoint_selection_start_epoch": (
            CHECKPOINT_SELECTION_START_EPOCH
        ),
        "cpc_used_for_selection": False,
    },
    "cache": {
        "enabled": ENABLE_LOCAL_SHARD_CACHE,
        "directory": str(
            LOCAL_SHARD_CACHE_DIR
        ),
        "maximum_gib": (
            LOCAL_CACHE_MAXIMUM_GIB
        ),
        "minimum_free_gib": (
            LOCAL_CACHE_MINIMUM_FREE_GIB
        ),
        "feature_root": str(
            FEATURE_ROOT_FOR_CACHE
        ),
    },
    "paths": {
        "checkpoint_directory": str(
            RUN_CHECKPOINT_DIR
        ),
        "latest_checkpoint": str(
            LATEST_CHECKPOINT_PATH
        ),
        "best_checkpoint": str(
            BEST_CHECKPOINT_PATH
        ),
        "best_pre_hierarchy_checkpoint": str(
            BEST_PRE_HIERARCHY_CHECKPOINT_PATH
        ),
        "history_json": str(
            TRAINING_HISTORY_JSON_PATH
        ),
        "history_csv": str(
            TRAINING_HISTORY_CSV_PATH
        ),
    },
}

atomic_json_save(
    section6_manifest,
    SECTION6_MANIFEST_PATH,
)


# ------------------------------------------------------------
# 6.18 Resume
# ------------------------------------------------------------

start_epoch = 1
history: List[Dict[str, Any]] = []
best_pre_hierarchy = None
best_post_hierarchy = None
full_dev_no_improvement_count = 0

if (
    RESUME_TRAINING
    and LATEST_CHECKPOINT_PATH.exists()
):
    print(
        f"[RESUME] Loading: "
        f"{LATEST_CHECKPOINT_PATH}"
    )

    checkpoint = load_checkpoint(
        LATEST_CHECKPOINT_PATH
    )

    last_completed_epoch = int(
        checkpoint.get(
            "epoch_one_based",
            checkpoint.get("epoch", 0),
        )
    )

    start_epoch = (
        last_completed_epoch + 1
    )
    history = list(
        checkpoint.get("history", [])
    )
    best_pre_hierarchy = checkpoint.get(
        "best_pre_hierarchy"
    )
    best_post_hierarchy = checkpoint.get(
        "best_post_hierarchy"
    )
    full_dev_no_improvement_count = int(
        checkpoint.get(
            "full_dev_no_improvement_count",
            0,
        )
    )

    print(
        f"[RESUME] Last completed epoch: "
        f"{last_completed_epoch}"
    )
    print(
        f"[RESUME] Starting epoch: "
        f"{start_epoch}"
    )

else:
    print("[START] New Depth-OT V2 training run")


# ------------------------------------------------------------
# 6.19 Configuration summary
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("SECTION 6 — DEPTH-OT V2 L4 TRAINING")
print("=" * 100)
print(f"Run name                 : {RUN_NAME}")
print(f"GPU                      : {GPU_NAME} ({GPU_TOTAL_GIB:.2f} GiB)")
print(f"Start epoch              : {start_epoch}")
print(f"Target final epoch       : {NUM_EPOCHS}")
print(f"Train batches            : {len(train_loader):,}")
print(f"DEV batches              : {len(dev_loader):,}")
print(f"TEST batches             : {len(test_loader):,}")
print(f"Mixed precision          : False")
print(f"TF32                     : False")
print(f"Base LR                  : {BASE_LEARNING_RATE:.2e}")
print(f"Topic LR                 : {TOPIC_LEARNING_RATE:.2e}")
print(f"Anchor LR                : {ANCHOR_LEARNING_RATE:.2e}")
print(f"Local shard cache        : {ENABLE_LOCAL_SHARD_CACHE}")
print(f"Local cache directory    : {LOCAL_SHARD_CACHE_DIR}")
print(f"Hierarchy first active   : Epoch {HIERARCHY_START_EPOCH + 1}")
print(f"Anchor first active      : Epoch {ANCHOR_START_EPOCH + 1}")
print(f"Checkpoint directory     : {RUN_CHECKPOINT_DIR}")
print(f"CPC in training/selection: False")
print("=" * 100)


# ------------------------------------------------------------
# 6.20 Training loop
# ------------------------------------------------------------

training_stopped_early = False
last_completed_epoch = start_epoch - 1

for epoch in range(
    start_epoch,
    NUM_EPOCHS + 1,
):
    epoch_weights = objective_weights_for_epoch(
        epoch
    )
    stage = training_stage_for_epoch(epoch)

    print("\n" + "=" * 100)
    print(
        f"EPOCH {epoch}/{NUM_EPOCHS} — "
        f"{stage}"
    )
    print("=" * 100)
    print(
        "Weights: "
        f"KL={epoch_weights['kl']:.6f}, "
        f"Hierarchy={epoch_weights['hierarchy']:.6f}, "
        f"Recon={epoch_weights['reconciliation']:.6f}, "
        f"TopicSep={epoch_weights['topic_embedding_separation']:.6f}, "
        f"BetaSep={epoch_weights['beta_separation']:.6f}, "
        f"Usage={epoch_weights['patent_usage']:.6f}"
    )
    print(
        f"Anchor enabled: "
        f"{anchor_enabled_for_epoch(epoch)}"
    )

    train_metrics = train_one_epoch(
        epoch=epoch
    )

    full_dev = (
        epoch % FULL_DEV_INTERVAL == 0
        or epoch == NUM_EPOCHS
    )

    if full_dev:
        dev_mode = "FULL"
        dev_metrics = evaluate_loader(
            loader=dev_loader,
            epoch=epoch,
            description=(
                f"Epoch {epoch:02d} FULL DEV"
            ),
            maximum_batches=None,
        )
    else:
        dev_mode = "FAST"
        dev_metrics = evaluate_loader(
            loader=dev_loader,
            epoch=epoch,
            description=(
                f"Epoch {epoch:02d} FAST DEV"
            ),
            maximum_batches=(
                FAST_DEV_MAX_BATCHES
            ),
        )

    epoch_record = {
        "epoch": int(epoch),
        "stage": stage,
        "dev_mode": dev_mode,
        "weights": epoch_weights,
        "train": train_metrics,
        "dev": dev_metrics,
        "cache_stats": dict(
            LOCAL_CACHE_STATS
        ),
        "timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }

    history.append(epoch_record)
    last_completed_epoch = epoch

    print("\n" + "-" * 100)
    print(
        f"[EPOCH {epoch}] "
        f"Train loss={train_metrics['total']:.6f} | "
        f"DEV loss={dev_metrics['total']:.6f} "
        f"({dev_mode})"
    )
    print(
        f"Train speed={train_metrics['patents_per_second']:.2f} patents/s | "
        f"GPU peak={train_metrics['peak_gpu_gib']:.2f} GiB"
    )
    print(
        f"DEV active topics="
        f"{dev_metrics['active_topics']}/{NUM_TOPICS} | "
        f"max share="
        f"{100.0 * dev_metrics['maximum_topic_share']:.2f}% | "
        f"marginal H="
        f"{dev_metrics['marginal_entropy']:.4f}"
    )
    print(
        f"DEV assignment H="
        f"{dev_metrics['mean_assignment_entropy']:.4f} | "
        f"theta variance="
        f"{dev_metrics['mean_topic_variance']:.6e}"
    )
    print(
        f"DEV beta cosine: "
        f"mean={dev_metrics['beta_mean_cosine']:.4f}, "
        f"max={dev_metrics['beta_max_cosine']:.4f}"
    )
    print(
        f"DEV valid={dev_metrics['valid']} | "
        f"selection score="
        f"{dev_metrics['selection_score']:.6f}"
    )

    if epoch_weights["hierarchy"] > 0.0:
        print(
            f"Sinkhorn max error="
            f"{dev_metrics['sinkhorn_maximum_error']:.3e} | "
            f"adjacent edges="
            f"{dev_metrics['hierarchy_adjacent_edges']:,}"
        )

    print(
        f"Local cache: "
        f"hits={LOCAL_CACHE_STATS['hits']}, "
        f"misses={LOCAL_CACHE_STATS['misses']}, "
        f"copies={LOCAL_CACHE_STATS['copies']}, "
        f"evictions={LOCAL_CACHE_STATS['evictions']}"
    )
    print("-" * 100)

    # Save logs before checkpointing.
    atomic_json_save(
        history,
        TRAINING_HISTORY_JSON_PATH,
    )
    write_history_csv(
        history,
        TRAINING_HISTORY_CSV_PATH,
    )

    # Pre-hierarchy diagnostic checkpoint.
    if (
        full_dev
        and epoch
        <= HIERARCHY_START_EPOCH
    ):
        if candidate_is_better(
            candidate=dev_metrics,
            incumbent=best_pre_hierarchy,
            minimum_improvement=(
                EARLY_STOP_MIN_IMPROVEMENT
            ),
        ):
            best_pre_hierarchy = {
                "epoch": int(epoch),
                "valid": bool(
                    dev_metrics["valid"]
                ),
                "selection_score": float(
                    dev_metrics[
                        "selection_score"
                    ]
                ),
                "metrics": dev_metrics,
            }

            pre_payload = build_checkpoint_payload(
                epoch=epoch,
                history=history,
                best_pre_hierarchy=(
                    best_pre_hierarchy
                ),
                best_post_hierarchy=(
                    best_post_hierarchy
                ),
                full_dev_no_improvement_count=(
                    full_dev_no_improvement_count
                ),
            )

            atomic_torch_save(
                pre_payload,
                BEST_PRE_HIERARCHY_CHECKPOINT_PATH,
            )

            print(
                f"[BEST PRE-HIERARCHY] "
                f"Epoch {epoch} saved."
            )

    # Post-hierarchy final model selection.
    post_hierarchy_improved = False

    if (
        full_dev
        and epoch
        >= CHECKPOINT_SELECTION_START_EPOCH
    ):
        if candidate_is_better(
            candidate=dev_metrics,
            incumbent=best_post_hierarchy,
            minimum_improvement=(
                EARLY_STOP_MIN_IMPROVEMENT
            ),
        ):
            post_hierarchy_improved = True

            best_post_hierarchy = {
                "epoch": int(epoch),
                "valid": bool(
                    dev_metrics["valid"]
                ),
                "selection_score": float(
                    dev_metrics[
                        "selection_score"
                    ]
                ),
                "metrics": dev_metrics,
            }

            full_dev_no_improvement_count = 0

            best_payload = build_checkpoint_payload(
                epoch=epoch,
                history=history,
                best_pre_hierarchy=(
                    best_pre_hierarchy
                ),
                best_post_hierarchy=(
                    best_post_hierarchy
                ),
                full_dev_no_improvement_count=(
                    full_dev_no_improvement_count
                ),
            )

            atomic_torch_save(
                best_payload,
                BEST_CHECKPOINT_PATH,
            )

            print(
                f"[BEST POST-HIERARCHY] "
                f"Epoch {epoch} saved: "
                f"score="
                f"{dev_metrics['selection_score']:.6f}"
            )

        elif epoch >= EARLY_STOP_START_EPOCH:
            full_dev_no_improvement_count += 1

            print(
                f"[EARLY STOP COUNTER] "
                f"{full_dev_no_improvement_count}/"
                f"{EARLY_STOP_PATIENCE}"
            )

    # latest.pt is always written after the epoch.
    latest_payload = build_checkpoint_payload(
        epoch=epoch,
        history=history,
        best_pre_hierarchy=(
            best_pre_hierarchy
        ),
        best_post_hierarchy=(
            best_post_hierarchy
        ),
        full_dev_no_improvement_count=(
            full_dev_no_improvement_count
        ),
    )

    atomic_torch_save(
        latest_payload,
        LATEST_CHECKPOINT_PATH,
    )

    if (
        epoch % CHECKPOINT_INTERVAL == 0
        or epoch == NUM_EPOCHS
    ):
        epoch_checkpoint_path = (
            RUN_CHECKPOINT_DIR
            / f"epoch_{epoch:03d}.pt"
        )

        atomic_torch_save(
            latest_payload,
            epoch_checkpoint_path,
        )

        print(
            f"[CHECKPOINT] Saved: "
            f"{epoch_checkpoint_path.name}"
        )

    if (
        epoch >= EARLY_STOP_START_EPOCH
        and full_dev_no_improvement_count
        >= EARLY_STOP_PATIENCE
    ):
        training_stopped_early = True

        print(
            f"[EARLY STOP] No FULL DEV improvement "
            f"for {EARLY_STOP_PATIENCE} evaluations."
        )
        break

    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------
# 6.21 Ensure a final checkpoint exists
# ------------------------------------------------------------

if not LATEST_CHECKPOINT_PATH.exists():
    raise RuntimeError(
        "Training ended without latest.pt."
    )

if not BEST_CHECKPOINT_PATH.exists():
    print(
        "[WARNING] No post-hierarchy best checkpoint exists. "
        "Using latest.pt as best.pt."
    )

    shutil.copy2(
        LATEST_CHECKPOINT_PATH,
        BEST_CHECKPOINT_PATH,
    )


# ------------------------------------------------------------
# 6.22 Restore selected checkpoint and final evaluation
# ------------------------------------------------------------

print(
    f"\n[FINAL RESTORE] "
    f"{BEST_CHECKPOINT_PATH}"
)

selected_checkpoint = load_checkpoint(
    BEST_CHECKPOINT_PATH
)

selected_epoch = int(
    selected_checkpoint.get(
        "epoch_one_based",
        selected_checkpoint.get(
            "epoch",
            last_completed_epoch,
        ),
    )
)

final_dev_metrics = evaluate_loader(
    loader=dev_loader,
    epoch=selected_epoch,
    description=(
        f"FINAL FULL DEV — Epoch {selected_epoch}"
    ),
    maximum_batches=None,
)

final_test_metrics = evaluate_loader(
    loader=test_loader,
    epoch=selected_epoch,
    description=(
        f"FINAL FULL TEST — Epoch {selected_epoch}"
    ),
    maximum_batches=None,
)


# ------------------------------------------------------------
# 6.23 Save training summary
# ------------------------------------------------------------

training_summary = {
    "completed_at_utc": (
        datetime.now(timezone.utc).isoformat()
    ),
    "run_name": RUN_NAME,
    "model_version": MODEL_VERSION,
    "gpu": {
        "name": GPU_NAME,
        "total_gib": GPU_TOTAL_GIB,
    },
    "training": {
        "requested_epochs": NUM_EPOCHS,
        "start_epoch": start_epoch,
        "last_completed_epoch": (
            last_completed_epoch
        ),
        "stopped_early": (
            training_stopped_early
        ),
    },
    "selected_checkpoint": {
        "path": str(
            BEST_CHECKPOINT_PATH
        ),
        "epoch": selected_epoch,
    },
    "best_pre_hierarchy": (
        best_pre_hierarchy
    ),
    "best_post_hierarchy": (
        best_post_hierarchy
    ),
    "final_dev": final_dev_metrics,
    "final_test": final_test_metrics,
    "cache_statistics": dict(
        LOCAL_CACHE_STATS
    ),
    "paths": {
        "latest_checkpoint": str(
            LATEST_CHECKPOINT_PATH
        ),
        "best_checkpoint": str(
            BEST_CHECKPOINT_PATH
        ),
        "best_pre_hierarchy_checkpoint": str(
            BEST_PRE_HIERARCHY_CHECKPOINT_PATH
        ),
        "history_json": str(
            TRAINING_HISTORY_JSON_PATH
        ),
        "history_csv": str(
            TRAINING_HISTORY_CSV_PATH
        ),
        "manifest": str(
            SECTION6_MANIFEST_PATH
        ),
    },
    "cpc_used_for_training": False,
    "cpc_used_for_selection": False,
}

atomic_json_save(
    training_summary,
    TRAINING_SUMMARY_PATH,
)


# ------------------------------------------------------------
# 6.24 Final report
# ------------------------------------------------------------

print("\n" + "=" * 104)
print("SECTION 6 — DEPTH-OT V2 TRAINING COMPLETED")
print("=" * 104)
print(f"Run name                 : {RUN_NAME}")
print(f"GPU                      : {GPU_NAME} ({GPU_TOTAL_GIB:.2f} GiB)")
print(f"Last completed epoch     : {last_completed_epoch}")
print(f"Stopped early            : {training_stopped_early}")
print(f"Selected epoch           : {selected_epoch}")
print("-" * 104)

if best_pre_hierarchy is not None:
    print(
        f"Best pre-hierarchy epoch : "
        f"{best_pre_hierarchy['epoch']} "
        f"(score={best_pre_hierarchy['selection_score']:.6f}, "
        f"valid={best_pre_hierarchy['valid']})"
    )

if best_post_hierarchy is not None:
    print(
        f"Best post-hierarchy epoch: "
        f"{best_post_hierarchy['epoch']} "
        f"(score={best_post_hierarchy['selection_score']:.6f}, "
        f"valid={best_post_hierarchy['valid']})"
    )

print("-" * 104)
print(
    f"FINAL DEV loss           : "
    f"{final_dev_metrics['total']:.6f}"
)
print(
    f"FINAL DEV active topics  : "
    f"{final_dev_metrics['active_topics']}/{NUM_TOPICS}"
)
print(
    f"FINAL DEV max share      : "
    f"{100.0 * final_dev_metrics['maximum_topic_share']:.2f}%"
)
print(
    f"FINAL DEV marginal H     : "
    f"{final_dev_metrics['marginal_entropy']:.6f}"
)
print(
    f"FINAL DEV assignment H   : "
    f"{final_dev_metrics['mean_assignment_entropy']:.6f}"
)
print(
    f"FINAL DEV theta variance : "
    f"{final_dev_metrics['mean_topic_variance']:.6e}"
)
print(
    f"FINAL DEV beta cosine    : "
    f"mean={final_dev_metrics['beta_mean_cosine']:.6f}, "
    f"max={final_dev_metrics['beta_max_cosine']:.6f}"
)
print(
    f"FINAL DEV valid          : "
    f"{final_dev_metrics['valid']}"
)
print("-" * 104)
print(
    f"FINAL TEST loss          : "
    f"{final_test_metrics['total']:.6f}"
)
print(
    f"FINAL TEST active topics : "
    f"{final_test_metrics['active_topics']}/{NUM_TOPICS}"
)
print(
    f"FINAL TEST max share     : "
    f"{100.0 * final_test_metrics['maximum_topic_share']:.2f}%"
)
print(
    f"FINAL TEST marginal H    : "
    f"{final_test_metrics['marginal_entropy']:.6f}"
)
print(
    f"FINAL TEST beta max cos  : "
    f"{final_test_metrics['beta_max_cosine']:.6f}"
)
print("-" * 104)
print(
    f"Local cache hits/misses  : "
    f"{LOCAL_CACHE_STATS['hits']}/"
    f"{LOCAL_CACHE_STATS['misses']}"
)
print(
    f"Local cache copied       : "
    f"{LOCAL_CACHE_STATS['bytes_copied'] / 2**30:.2f} GiB"
)
print(f"Latest checkpoint        : {LATEST_CHECKPOINT_PATH}")
print(f"Best checkpoint          : {BEST_CHECKPOINT_PATH}")
print(f"Pre-hierarchy checkpoint : {BEST_PRE_HIERARCHY_CHECKPOINT_PATH}")
print(f"History JSON             : {TRAINING_HISTORY_JSON_PATH}")
print(f"History CSV              : {TRAINING_HISTORY_CSV_PATH}")
print(f"Training summary         : {TRAINING_SUMMARY_PATH}")
print("-" * 104)
print("CPC used in training     : False")
print("CPC used for selection   : False")
print("=" * 104)
print("[PASS] Section 6 training and final evaluation completed.")
