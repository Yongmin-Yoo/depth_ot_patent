# ============================================================
# SECTION 6 — T4-OPTIMIZED ANTI-COLLAPSE TRAINING
# Fresh run / patents_per_batch=8 / no resume
# ============================================================

import os
import gc
import json
import time
import math
import random
import traceback
from pathlib import Path
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ============================================================
# 0. Preconditions
# ============================================================

REQUIRED_GLOBALS = [
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
    name for name in REQUIRED_GLOBALS
    if name not in globals()
]

if missing_globals:
    raise RuntimeError(
        "Sections 0–5를 먼저 실행하세요. "
        f"Missing globals: {missing_globals}"
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. Colab에서 T4 GPU를 선택하세요."
    )

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_TOTAL_GIB = (
    torch.cuda.get_device_properties(0).total_memory / 2**30
)

if "T4" not in GPU_NAME:
    print(
        f"[WARNING] 현재 GPU는 {GPU_NAME}입니다. "
        "코드는 T4 기준으로 설정되었습니다."
    )


# ============================================================
# 1. Core experiment settings
# ============================================================

SEED = 42
PATENTS_PER_BATCH = 8
NUM_EPOCHS = 24

# 완전 신규 실행
RESUME_TRAINING = False

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = (
    f"depth_ot_anticollapse_seed{SEED}_{RUN_TIMESTAMP}"
)

# ------------------------------------------------------------
# Objective schedule
# ------------------------------------------------------------

# Epoch 1–3: reconstruction/topic separation 중심
KL_START_EPOCH = 3
KL_FULL_EPOCH = 12
KL_MAX_WEIGHT = 0.10

# Epoch 1–8: hierarchy 비활성화
HIERARCHY_START_EPOCH = 8
HIERARCHY_FULL_EPOCH = 16
HIERARCHY_MAX_WEIGHT = 0.15

# Phi 역시 hierarchy와 함께 늦게 시작
PHI_START_EPOCH = 8

# ------------------------------------------------------------
# Anti-collapse regularization
# ------------------------------------------------------------

BALANCE_WEIGHT_INITIAL = 2.00
BALANCE_WEIGHT_FINAL = 0.60

USAGE_WEIGHT_INITIAL = 1.00
USAGE_WEIGHT_FINAL = 0.30

CONFIDENCE_WEIGHT_INITIAL = 0.05
CONFIDENCE_WEIGHT_FINAL = 0.02

BALANCE_TEMPERATURE_INITIAL = 0.30
BALANCE_TEMPERATURE_FINAL = 0.15

BALANCE_SINKHORN_ITERATIONS = 5

# ------------------------------------------------------------
# Best-checkpoint requirements
# ------------------------------------------------------------

MIN_ACTIVE_TOPICS = 20
MAX_TOPIC_SHARE = 0.25
MIN_MARGINAL_ENTROPY = 0.70

# ------------------------------------------------------------
# Evaluation frequency
# ------------------------------------------------------------

# 짝수 epoch 및 마지막 epoch: 전체 DEV
FULL_DEV_EVERY_EPOCHS = 2

# 나머지 epoch: 앞부분 DEV batch만 빠르게 평가
FAST_DEV_BATCHES = 200

# ------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------

EARLY_STOPPING_START_EPOCH = 12

# Full DEV 평가 기준 3회
EARLY_STOPPING_PATIENCE = 3
MINIMUM_IMPROVEMENT = 1e-4

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

MAX_GRAD_NORM = 1.0
CHECKPOINT_INTERVAL_EPOCHS = 3

TRAIN_POSTFIX_INTERVAL = 100
EVAL_POSTFIX_INTERVAL = 100

# Sinkhorn 안정성을 위해 전체 AMP는 사용하지 않음.
# TF32는 활성화하므로 T4 Tensor Core를 일부 활용함.
USE_MIXED_PRECISION = False

# DataLoader
T4_NUM_WORKERS = min(
    4,
    max(2, (os.cpu_count() or 2) // 2),
)

T4_PREFETCH_FACTOR = 4


# ============================================================
# 2. Seed and CUDA setup
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

depth_ot_model.to(DEVICE)
topic_anchor.to(DEVICE)

CONFIG.seed = SEED
CONFIG.patents_per_batch = PATENTS_PER_BATCH

if int(CONFIG.patents_per_batch) != PATENTS_PER_BATCH:
    raise RuntimeError(
        "CONFIG.patents_per_batch는 반드시 8이어야 합니다."
    )


# ============================================================
# 3. DataLoader optimization
# ============================================================

def loader_is_t4_optimized(loader):
    return (
        getattr(loader, "num_workers", 0)
        == T4_NUM_WORKERS
        and bool(
            getattr(loader, "pin_memory", False)
        )
        and bool(
            getattr(loader, "persistent_workers", False)
        )
    )


def rebuild_loader_for_t4(
    original_loader,
    loader_name,
):
    if loader_is_t4_optimized(original_loader):
        print(
            f"[Loader] {loader_name}: already optimized"
        )
        return original_loader

    print(
        f"[Loader] {loader_name}: rebuilding | "
        f"workers={T4_NUM_WORKERS}, "
        f"pin_memory=True, "
        f"prefetch={T4_PREFETCH_FACTOR}"
    )

    return DataLoader(
        dataset=original_loader.dataset,
        batch_sampler=original_loader.batch_sampler,
        collate_fn=original_loader.collate_fn,
        num_workers=T4_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=T4_PREFETCH_FACTOR,
        worker_init_fn=getattr(
            original_loader,
            "worker_init_fn",
            None,
        ),
    )


original_train_loader = train_loader
original_dev_loader = dev_loader
original_test_loader = test_loader

try:
    train_loader = rebuild_loader_for_t4(
        original_train_loader,
        "train",
    )

    dev_loader = rebuild_loader_for_t4(
        original_dev_loader,
        "dev",
    )

    test_loader = rebuild_loader_for_t4(
        original_test_loader,
        "test",
    )

except Exception as loader_error:
    print(
        f"[WARNING] DataLoader 최적화 실패: "
        f"{loader_error}"
    )
    print("[Fallback] 기존 DataLoader를 사용합니다.")

    train_loader = original_train_loader
    dev_loader = original_dev_loader
    test_loader = original_test_loader


# ============================================================
# 4. CONFIG overrides
# ============================================================

CONFIG.num_epochs = NUM_EPOCHS
CONFIG.patents_per_batch = PATENTS_PER_BATCH

CONFIG.kl_warmup_epochs = KL_FULL_EPOCH
CONFIG.hierarchy_warmup_epochs = HIERARCHY_FULL_EPOCH
CONFIG.phi_warmup_epochs = PHI_START_EPOCH

CONFIG.kl_warmup_steps = (
    len(train_loader) * KL_FULL_EPOCH
)

CONFIG.hierarchy_warmup_steps = (
    len(train_loader) * HIERARCHY_FULL_EPOCH
)

CONFIG.kl_max_weight = KL_MAX_WEIGHT
CONFIG.hierarchy_max_weight = HIERARCHY_MAX_WEIGHT

# 프로젝트 내부에서 다른 이름을 사용하는 경우에도 적용
CONFIG_ALIASES = {
    "gamma_kl": KL_MAX_WEIGHT,
    "kl_weight": KL_MAX_WEIGHT,
    "gamma_h": HIERARCHY_MAX_WEIGHT,
    "gamma_hierarchy": HIERARCHY_MAX_WEIGHT,
    "hierarchy_weight": HIERARCHY_MAX_WEIGHT,
}

for config_name, config_value in CONFIG_ALIASES.items():
    if hasattr(CONFIG, config_name):
        setattr(
            CONFIG,
            config_name,
            config_value,
        )

# ============================================================
# 5. Objective weight override — recursion-safe
# ============================================================

_BASE_OBJECTIVE_WEIGHT_FUNCTION = get_objective_weights


def linear_ramp(
    epoch,
    start_epoch,
    full_epoch,
    maximum,
):
    epoch_one_based = epoch + 1

    if epoch_one_based <= start_epoch:
        return 0.0

    if epoch_one_based >= full_epoch:
        return float(maximum)

    progress = (
        epoch_one_based - start_epoch
    ) / max(
        full_epoch - start_epoch,
        1,
    )

    return float(maximum * progress)


def make_scheduled_objective_weight_function(base_function):

    def scheduled_objective_weights(epoch):
        original_weights = base_function(epoch)
        weights = dict(original_weights)

        kl_weight = linear_ramp(
            epoch=epoch,
            start_epoch=KL_START_EPOCH,
            full_epoch=KL_FULL_EPOCH,
            maximum=KL_MAX_WEIGHT,
        )

        hierarchy_weight = linear_ramp(
            epoch=epoch,
            start_epoch=HIERARCHY_START_EPOCH,
            full_epoch=HIERARCHY_FULL_EPOCH,
            maximum=HIERARCHY_MAX_WEIGHT,
        )

        found_kl = False
        found_hierarchy = False

        for key in list(weights):
            normalized_key = str(key).lower()

            if "kl" in normalized_key:
                weights[key] = kl_weight
                found_kl = True

            elif (
                "hier" in normalized_key
                and "recon" not in normalized_key
            ):
                weights[key] = hierarchy_weight
                found_hierarchy = True

        if not found_kl:
            weights["kl"] = kl_weight

        if not found_hierarchy:
            weights["hierarchy"] = hierarchy_weight

        return weights

    scheduled_objective_weights._anti_collapse_override = True
    return scheduled_objective_weights


get_objective_weights = (
    make_scheduled_objective_weight_function(
        _BASE_OBJECTIVE_WEIGHT_FUNCTION
    )
)



# ============================================================
# 6. Anti-collapse schedule
# ============================================================

def interpolate_schedule(
    epoch,
    initial_value,
    final_value,
):
    if NUM_EPOCHS <= 1:
        return float(final_value)

    progress = epoch / (NUM_EPOCHS - 1)

    return float(
        initial_value
        + progress
        * (final_value - initial_value)
    )


def anti_collapse_schedule(epoch):
    return {
        "balance_weight": interpolate_schedule(
            epoch,
            BALANCE_WEIGHT_INITIAL,
            BALANCE_WEIGHT_FINAL,
        ),
        "usage_weight": interpolate_schedule(
            epoch,
            USAGE_WEIGHT_INITIAL,
            USAGE_WEIGHT_FINAL,
        ),
        "confidence_weight": interpolate_schedule(
            epoch,
            CONFIDENCE_WEIGHT_INITIAL,
            CONFIDENCE_WEIGHT_FINAL,
        ),
        "temperature": interpolate_schedule(
            epoch,
            BALANCE_TEMPERATURE_INITIAL,
            BALANCE_TEMPERATURE_FINAL,
        ),
    }


# ============================================================
# 7. Balanced Sinkhorn target
# ============================================================

@torch.no_grad()
def make_balanced_sinkhorn_target(
    theta,
    temperature,
    iterations,
):
    if theta.ndim != 2:
        raise ValueError(
            f"theta must be 2D, got {theta.shape}"
        )

    num_documents, num_topics = theta.shape

    if num_documents == 0:
        raise RuntimeError("Empty theta batch.")

    logits = torch.log(
        theta.float().clamp_min(1e-8)
    )

    logits = logits / max(
        float(temperature),
        1e-4,
    )

    logits = logits - logits.max(
        dim=1,
        keepdim=True,
    ).values

    # Topic × document
    assignment = torch.exp(logits).t()
    assignment = assignment.clamp_min(1e-12)
    assignment = (
        assignment
        / assignment.sum().clamp_min(1e-12)
    )

    for _ in range(iterations):
        # Topic marginal을 균등하게 설정
        assignment = (
            assignment
            / assignment.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        assignment = assignment / num_topics

        # Document marginal을 균등하게 설정
        assignment = (
            assignment
            / assignment.sum(
                dim=0,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        assignment = assignment / num_documents

    assignment = assignment * num_documents

    return assignment.t().contiguous()


# ============================================================
# 8. Anti-collapse losses
# ============================================================

def calculate_anti_collapse_losses(
    theta,
    epoch,
):
    theta = theta.float()

    if theta.ndim != 2:
        raise ValueError(
            f"Unexpected theta shape: {theta.shape}"
        )

    theta = theta.clamp_min(1e-8)

    theta = theta / theta.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)

    schedule = anti_collapse_schedule(epoch)

    balanced_target = (
        make_balanced_sinkhorn_target(
            theta=theta.detach(),
            temperature=schedule["temperature"],
            iterations=(
                BALANCE_SINKHORN_ITERATIONS
            ),
        )
    )

    # Balanced pseudo-target와 theta 사이의 KL
    balance_loss = (
        balanced_target
        * (
            torch.log(
                balanced_target.clamp_min(1e-8)
            )
            - torch.log(theta)
        )
    ).sum(dim=1).mean()

    # Corpus/batch topic marginal 균형
    topic_marginal = theta.mean(dim=0)

    topic_marginal = (
        topic_marginal
        / topic_marginal.sum().clamp_min(1e-8)
    )

    num_topics = theta.shape[1]

    usage_loss = (
        topic_marginal
        * torch.log(
            topic_marginal.clamp_min(1e-8)
            * num_topics
        )
    ).sum()

    # 각 문서의 theta가 완전 uniform이 되는 것을 방지
    document_entropy = -(
        theta * torch.log(theta)
    ).sum(dim=1).mean()

    normalized_document_entropy = (
        document_entropy
        / math.log(max(num_topics, 2))
    )

    confidence_loss = (
        normalized_document_entropy
    )

    regularizer = (
        schedule["balance_weight"]
        * balance_loss
        + schedule["usage_weight"]
        * usage_loss
        + schedule["confidence_weight"]
        * confidence_loss
    )

    return {
        "balance": balance_loss,
        "usage": usage_loss,
        "confidence": confidence_loss,
        "regularizer": regularizer,
        "schedule": schedule,
    }


# ============================================================
# 9. Run directories
# ============================================================

PROJECT_ROOT = Path(
    getattr(
        CONFIG,
        "project_root",
        "/content/drive/MyDrive/depth_ot_patent",
    )
)

CHECKPOINT_ROOT = Path(
    DIRS.get(
        "depth_ot_ckpt",
        PROJECT_ROOT / "checkpoints" / "depth_ot",
    )
)

LOG_ROOT = Path(
    DIRS.get(
        "depth_ot_logs",
        PROJECT_ROOT / "logs" / "depth_ot",
    )
)

RESULT_ROOT = Path(
    DIRS.get(
        "depth_ot_results",
        PROJECT_ROOT / "results" / "depth_ot",
    )
)

RUN_CHECKPOINT_DIR = (
    CHECKPOINT_ROOT / RUN_NAME
)
RUN_LOG_DIR = LOG_ROOT / RUN_NAME
RUN_RESULT_DIR = RESULT_ROOT / RUN_NAME

for directory in [
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    RUN_RESULT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

LATEST_CHECKPOINT_PATH = (
    RUN_CHECKPOINT_DIR / "latest.pt"
)
BEST_CHECKPOINT_PATH = (
    RUN_CHECKPOINT_DIR / "best.pt"
)
HISTORY_JSONL_PATH = (
    RUN_LOG_DIR / "history.jsonl"
)
TRAINING_SUMMARY_PATH = (
    RUN_LOG_DIR / "training_summary.json"
)
RUN_MANIFEST_PATH = (
    RUN_LOG_DIR / "run_manifest.json"
)
ERROR_LOG_PATH = (
    RUN_LOG_DIR / "training_error.txt"
)
LEARNED_ANCHOR_PATH = (
    RUN_RESULT_DIR / "learned_anchor.pt"
)


# Section 7 호환을 위해 문자열로 유지
RUN_CHECKPOINT_DIR = str(RUN_CHECKPOINT_DIR)
RUN_LOG_DIR = str(RUN_LOG_DIR)
RUN_RESULT_DIR = str(RUN_RESULT_DIR)

LATEST_CHECKPOINT_PATH = str(
    LATEST_CHECKPOINT_PATH
)
BEST_CHECKPOINT_PATH = str(
    BEST_CHECKPOINT_PATH
)
HISTORY_JSONL_PATH = str(
    HISTORY_JSONL_PATH
)
TRAINING_SUMMARY_PATH = str(
    TRAINING_SUMMARY_PATH
)
RUN_MANIFEST_PATH = str(
    RUN_MANIFEST_PATH
)
ERROR_LOG_PATH = str(
    ERROR_LOG_PATH
)
LEARNED_ANCHOR_PATH = str(
    LEARNED_ANCHOR_PATH
)


# ============================================================
# 10. Serialization utilities
# ============================================================

def config_to_dictionary(config):
    if is_dataclass(config):
        return asdict(config)

    if isinstance(config, dict):
        return dict(config)

    result = {}

    for key in dir(config):
        if key.startswith("_"):
            continue

        try:
            value = getattr(config, key)
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


def atomic_json_save(data, path):
    temporary_path = path + ".tmp"

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
        path,
    )


def append_jsonl(data, path):
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


def atomic_torch_save(data, path):
    temporary_path = path + ".tmp"

    torch.save(
        data,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
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
# 11. Optimizer utility
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
# 12. Metrics
# ============================================================

BASE_LOSS_NAMES = [
    "reconstruction",
    "kl",
    "hierarchy",
    "reconciliation",
    "diversity",
]


class MetricAccumulator:
    def __init__(self, num_topics):
        self.num_topics = int(num_topics)

        self.num_batches = 0
        self.num_patents = 0
        self.num_claims = 0

        self.loss_sums = {
            "total": 0.0,
            "base_total": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "hierarchy": 0.0,
            "reconciliation": 0.0,
            "diversity": 0.0,
            "balance": 0.0,
            "usage": 0.0,
            "confidence": 0.0,
            "regularizer": 0.0,
        }

        self.topic_counts = torch.zeros(
            self.num_topics,
            dtype=torch.long,
        )

        self.gradient_norm_sum = 0.0
        self.phi_update_count = 0
        self.maximum_sinkhorn_error = 0.0

    def update(
        self,
        losses,
        regularizers,
        theta,
        batch,
        gradient_norm=None,
        phi_info=None,
        ot_output=None,
    ):
        values = {
            "total": losses["total"],
            "base_total": losses["base_total"],
            "reconstruction": losses[
                "reconstruction"
            ],
            "kl": losses["kl"],
            "hierarchy": losses["hierarchy"],
            "reconciliation": losses[
                "reconciliation"
            ],
            "diversity": losses["diversity"],
            "balance": regularizers["balance"],
            "usage": regularizers["usage"],
            "confidence": regularizers[
                "confidence"
            ],
            "regularizer": regularizers[
                "regularizer"
            ],
        }

        for name, value in values.items():
            scalar = float(
                value.detach().float().item()
            )

            if not math.isfinite(scalar):
                raise FloatingPointError(
                    f"Non-finite metric: "
                    f"{name}={scalar}"
                )

            self.loss_sums[name] += scalar

        predictions = (
            theta.detach()
            .argmax(dim=1)
            .cpu()
        )

        self.topic_counts += torch.bincount(
            predictions,
            minlength=self.num_topics,
        )

        self.num_batches += 1
        self.num_patents += int(
            batch["num_patents"]
        )
        self.num_claims += int(
            batch["num_claims"]
        )

        if gradient_norm is not None:
            self.gradient_norm_sum += float(
                gradient_norm
            )

        if (
            phi_info is not None
            and phi_info.get("updated", False)
        ):
            self.phi_update_count += 1

        if (
            isinstance(ot_output, dict)
            and ot_output.get(
                "sinkhorn_diagnostics"
            ) is not None
        ):
            diagnostics = ot_output[
                "sinkhorn_diagnostics"
            ]

            for info in diagnostics.values():
                error = info.get(
                    "maximum_marginal_error",
                    0.0,
                )

                if torch.is_tensor(error):
                    error = float(
                        error.detach().cpu()
                    )

                self.maximum_sinkhorn_error = max(
                    self.maximum_sinkhorn_error,
                    float(error),
                )

    def compute(self):
        if self.num_batches == 0:
            raise RuntimeError(
                "No batches were accumulated."
            )

        metrics = {
            name: value / self.num_batches
            for name, value
            in self.loss_sums.items()
        }

        total_assignments = int(
            self.topic_counts.sum().item()
        )

        active_topics = int(
            (self.topic_counts > 0)
            .sum()
            .item()
        )

        maximum_topic_share = float(
            self.topic_counts.max().item()
            / max(total_assignments, 1)
        )

        marginal = (
            self.topic_counts.float()
            / max(total_assignments, 1)
        )

        positive = marginal > 0

        marginal_entropy = float(
            -(
                marginal[positive]
                * torch.log(marginal[positive])
            ).sum().item()
            / math.log(max(self.num_topics, 2))
        )

        metrics.update({
            "num_batches": int(
                self.num_batches
            ),
            "num_patents": int(
                self.num_patents
            ),
            "num_claims": int(
                self.num_claims
            ),
            "active_topics": active_topics,
            "maximum_topic_share": (
                maximum_topic_share
            ),
            "normalized_marginal_entropy": (
                marginal_entropy
            ),
            "topic_counts": (
                self.topic_counts.tolist()
            ),
            "mean_gradient_norm": (
                self.gradient_norm_sum
                / max(self.num_batches, 1)
            ),
            "phi_updates": int(
                self.phi_update_count
            ),
            "maximum_sinkhorn_error": float(
                self.maximum_sinkhorn_error
            ),
        })

        return metrics


# ============================================================
# 13. Model objective
# ============================================================

def should_compute_hierarchy(epoch):
    return (
        epoch + 1 > HIERARCHY_START_EPOCH
    )


def compute_objective(
    batch,
    epoch,
    deterministic,
):
    hierarchy_enabled = (
        should_compute_hierarchy(epoch)
    )

    output = depth_ot_model.compute_loss(
        batch=batch,
        epoch=epoch,
        deterministic=deterministic,
        compute_hierarchy=hierarchy_enabled,
    )

    losses = output["losses"]
    theta = output["outputs"]["theta"]

    regularizers = (
        calculate_anti_collapse_losses(
            theta=theta,
            epoch=epoch,
        )
    )

    base_total = losses["total"]
    final_total = (
        base_total
        + regularizers["regularizer"]
    )

    losses["base_total"] = base_total
    losses["total"] = final_total

    # compute_hierarchy=False인 경우에도
    # MetricAccumulator가 요구하는 키를 보장
    reference_tensor = final_total

    for name in BASE_LOSS_NAMES:
        if name not in losses:
            losses[name] = (
                reference_tensor
                * 0.0
            )

    return (
        output,
        losses,
        regularizers,
        theta,
    )


# ============================================================
# 14. Train one epoch
# ============================================================

def train_one_epoch(epoch):
    depth_ot_model.train()
    topic_anchor.train()

    batch_sampler = getattr(
        train_loader,
        "batch_sampler",
        None,
    )

    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)

    accumulator = MetricAccumulator(
        num_topics=CONFIG.num_topics
    )

    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

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

        (
            output,
            losses,
            regularizers,
            theta,
        ) = compute_objective(
            batch=batch,
            epoch=epoch,
            deterministic=False,
        )

        total_loss = losses["total"]

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite training loss: "
                f"epoch={epoch + 1}, "
                f"batch={batch_index}"
            )

        total_loss.backward()

        if (
            topic_anchor.phi.grad is not None
            and torch.any(
                topic_anchor.phi.grad != 0
            )
        ):
            raise RuntimeError(
                "topic_anchor.phi received gradients "
                "during main-model backward."
            )

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                depth_ot_model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )
        )

        gradient_norm = float(
            torch.as_tensor(
                gradient_norm
            ).detach().item()
        )

        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                "Non-finite gradient norm."
            )

        main_optimizer.step()

        if epoch + 1 > PHI_START_EPOCH:
            phi_info = update_phi(
                theta=theta.detach(),
                claim_depth=batch["claim_depth"],
                claim_edge_index=batch[
                    "claim_edge_index"
                ],
                epoch=epoch,
            )
        else:
            phi_info = {
                "updated": False
            }

        accumulator.update(
            losses=losses,
            regularizers=regularizers,
            theta=theta,
            batch=batch,
            gradient_norm=gradient_norm,
            phi_info=phi_info,
            ot_output=output.get("ot_output"),
        )

        if (
            batch_index
            % TRAIN_POSTFIX_INTERVAL
            == 0
            or batch_index + 1
            == len(train_loader)
        ):
            progress.set_postfix({
                "loss": (
                    f"{total_loss.detach().item():.4f}"
                ),
                "bal": (
                    f"{regularizers['balance'].detach().item():.3f}"
                ),
                "use": (
                    f"{regularizers['usage'].detach().item():.3f}"
                ),
                "hier": (
                    "on"
                    if should_compute_hierarchy(epoch)
                    else "off"
                ),
            }, refresh=False)

        del cpu_batch
        del batch
        del output
        del losses
        del regularizers
        del theta
        del total_loss
        del phi_info

    elapsed = time.time() - start_time
    metrics = accumulator.compute()

    metrics["epoch_seconds"] = float(
        elapsed
    )

    metrics["patents_per_second"] = float(
        metrics["num_patents"]
        / max(elapsed, 1e-9)
    )

    metrics["peak_gpu_gib"] = float(
        torch.cuda.max_memory_allocated()
        / 2**30
    )

    metrics["hierarchy_computed"] = bool(
        should_compute_hierarchy(epoch)
    )

    return metrics


# ============================================================
# 15. Evaluate
# ============================================================

@torch.inference_mode()
def evaluate_one_epoch(
    data_loader,
    epoch,
    description,
    maximum_batches=None,
):
    depth_ot_model.eval()
    topic_anchor.eval()

    accumulator = MetricAccumulator(
        num_topics=CONFIG.num_topics
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
        if (
            maximum_batches is not None
            and batch_index >= maximum_batches
        ):
            break

        batch = move_depth_ot_batch(
            cpu_batch,
            device=DEVICE,
        )

        (
            output,
            losses,
            regularizers,
            theta,
        ) = compute_objective(
            batch=batch,
            epoch=epoch,
            deterministic=True,
        )

        if not torch.isfinite(
            losses["total"]
        ):
            raise FloatingPointError(
                f"Non-finite evaluation loss: "
                f"{description}, "
                f"batch={batch_index}"
            )

        accumulator.update(
            losses=losses,
            regularizers=regularizers,
            theta=theta,
            batch=batch,
            gradient_norm=None,
            phi_info=None,
            ot_output=output.get("ot_output"),
        )

        if (
            batch_index
            % EVAL_POSTFIX_INTERVAL
            == 0
        ):
            progress.set_postfix({
                "loss": (
                    f"{losses['total'].item():.4f}"
                )
            }, refresh=False)

        del cpu_batch
        del batch
        del output
        del losses
        del regularizers
        del theta

    return accumulator.compute()


# ============================================================
# 16. Checkpoint selection
# ============================================================

def build_selection_record(
    dev_metrics,
):
    eligible = (
        dev_metrics["active_topics"]
        >= MIN_ACTIVE_TOPICS
        and dev_metrics["maximum_topic_share"]
        <= MAX_TOPIC_SHARE
        and dev_metrics[
            "normalized_marginal_entropy"
        ]
        >= MIN_MARGINAL_ENTROPY
    )

    return {
        "eligible": bool(eligible),
        "active_topics": int(
            dev_metrics["active_topics"]
        ),
        "maximum_topic_share": float(
            dev_metrics["maximum_topic_share"]
        ),
        "marginal_entropy": float(
            dev_metrics[
                "normalized_marginal_entropy"
            ]
        ),
        "dev_total": float(
            dev_metrics["total"]
        ),
    }


def selection_score(selection):
    return (
        2.0 * selection["active_topics"]
        + 10.0 * selection["marginal_entropy"]
        - 20.0 * selection[
            "maximum_topic_share"
        ]
    )


def candidate_is_better(
    candidate,
    best_selection,
):
    if best_selection is None:
        return True

    if (
        candidate["eligible"]
        != best_selection["eligible"]
    ):
        return candidate["eligible"]

    if candidate["eligible"]:
        return (
            candidate["dev_total"]
            < best_selection["dev_total"]
            - MINIMUM_IMPROVEMENT
        )

    return (
        selection_score(candidate)
        > selection_score(best_selection)
    )


# ============================================================
# 17. Checkpoint utilities
# ============================================================

def build_checkpoint(
    epoch,
    history,
    best_selection,
):
    return {
        "run_name": RUN_NAME,
        "feature_run_name": FEATURE_RUN_NAME,
        "epoch": int(epoch),
        "epoch_one_based": int(epoch + 1),
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
        "history": history,
        "best_selection": best_selection,
        "best_validation_loss": (
            None
            if best_selection is None
            else best_selection["dev_total"]
        ),
        "config": config_to_dictionary(
            CONFIG
        ),
        "training_settings": {
            "fresh_run": True,
            "resume": False,
            "seed": SEED,
            "num_epochs": NUM_EPOCHS,
            "patents_per_batch": (
                PATENTS_PER_BATCH
            ),
            "anti_collapse": True,
            "balance_weight_initial": (
                BALANCE_WEIGHT_INITIAL
            ),
            "balance_weight_final": (
                BALANCE_WEIGHT_FINAL
            ),
            "usage_weight_initial": (
                USAGE_WEIGHT_INITIAL
            ),
            "usage_weight_final": (
                USAGE_WEIGHT_FINAL
            ),
            "kl_max_weight": KL_MAX_WEIGHT,
            "hierarchy_max_weight": (
                HIERARCHY_MAX_WEIGHT
            ),
            "minimum_active_topics": (
                MIN_ACTIVE_TOPICS
            ),
            "maximum_topic_share": (
                MAX_TOPIC_SHARE
            ),
            "minimum_marginal_entropy": (
                MIN_MARGINAL_ENTROPY
            ),
        },
        "saved_at": (
            datetime.now().isoformat()
        ),
    }


def save_checkpoint(
    path,
    epoch,
    history,
    best_selection,
):
    checkpoint = build_checkpoint(
        epoch=epoch,
        history=history,
        best_selection=best_selection,
    )

    atomic_torch_save(
        checkpoint,
        path,
    )


# ============================================================
# 18. Manifest and configuration report
# ============================================================

run_manifest = {
    "run_name": RUN_NAME,
    "feature_run_name": FEATURE_RUN_NAME,
    "started_at": datetime.now().isoformat(),
    "fresh_training": True,
    "resume": False,
    "seed": SEED,
    "gpu": GPU_NAME,
    "gpu_memory_gib": GPU_TOTAL_GIB,
    "patents_per_batch": PATENTS_PER_BATCH,
    "num_epochs": NUM_EPOCHS,
    "train_batches": len(train_loader),
    "dev_batches": len(dev_loader),
    "test_batches": len(test_loader),
    "num_workers": getattr(
        train_loader,
        "num_workers",
        0,
    ),
    "full_dev_every_epochs": (
        FULL_DEV_EVERY_EPOCHS
    ),
    "fast_dev_batches": FAST_DEV_BATCHES,
    "config": config_to_dictionary(CONFIG),
}

atomic_json_save(
    run_manifest,
    RUN_MANIFEST_PATH,
)

print("=" * 90)
print("SECTION 6 — T4-OPTIMIZED ANTI-COLLAPSE TRAINING")
print("=" * 90)
print(f"Run name              : {RUN_NAME}")
print(f"GPU                   : {GPU_NAME}")
print(f"GPU memory            : {GPU_TOTAL_GIB:.2f} GiB")
print(f"Seed                  : {SEED}")
print(f"Epochs                : {NUM_EPOCHS}")
print(f"Patents/batch         : {PATENTS_PER_BATCH}")
print(f"Topics                : {CONFIG.num_topics}")
print(f"Train batches         : {len(train_loader):,}")
print(f"DEV batches           : {len(dev_loader):,}")
print(f"Workers               : {getattr(train_loader, 'num_workers', 0)}")
print(f"KL max                : {KL_MAX_WEIGHT}")
print(f"Hierarchy max         : {HIERARCHY_MAX_WEIGHT}")
print(f"Hierarchy starts      : epoch {HIERARCHY_START_EPOCH + 1}")
print(f"Full DEV interval     : {FULL_DEV_EVERY_EPOCHS}")
print(f"Fast DEV batches      : {FAST_DEV_BATCHES}")
print(f"Minimum active topics : {MIN_ACTIVE_TOPICS}")
print(f"Maximum topic share   : {MAX_TOPIC_SHARE:.0%}")
print(f"Minimum marginal H    : {MIN_MARGINAL_ENTROPY}")
print(f"Checkpoint directory  : {RUN_CHECKPOINT_DIR}")
print("=" * 90)

print("\nObjective schedule:")

for check_epoch in [
    0,
    2,
    5,
    7,
    8,
    11,
    15,
    23,
]:
    if check_epoch < NUM_EPOCHS:
        print(
            f"Epoch {check_epoch + 1:02d}: "
            f"weights="
            f"{get_objective_weights(check_epoch)}, "
            f"hierarchy="
            f"{should_compute_hierarchy(check_epoch)}, "
            f"anti="
            f"{anti_collapse_schedule(check_epoch)}"
        )


# ============================================================
# 19. Training loop
# ============================================================

history = []

best_selection = None
best_epoch = None

full_dev_evaluations_without_improvement = 0
stopped_early = False

execution_start_time = time.time()

try:
    for epoch in range(NUM_EPOCHS):
        epoch_one_based = epoch + 1

        run_full_dev = (
            epoch_one_based
            % FULL_DEV_EVERY_EPOCHS
            == 0
            or epoch_one_based == NUM_EPOCHS
        )

        print("\n" + "-" * 90)
        print(
            f"Epoch {epoch_one_based}/{NUM_EPOCHS}"
        )
        print(
            f"Hierarchy enabled : "
            f"{should_compute_hierarchy(epoch)}"
        )
        print(
            f"DEV mode          : "
            f"{'FULL' if run_full_dev else 'FAST'}"
        )
        print(
            f"Objective weights : "
            f"{get_objective_weights(epoch)}"
        )
        print(
            f"Anti-collapse     : "
            f"{anti_collapse_schedule(epoch)}"
        )
        print("-" * 90)

        epoch_start_time = time.time()

        train_metrics = train_one_epoch(epoch)

        if run_full_dev:
            dev_metrics = evaluate_one_epoch(
                data_loader=dev_loader,
                epoch=epoch,
                description=(
                    f"Full DEV "
                    f"{epoch_one_based}/{NUM_EPOCHS}"
                ),
                maximum_batches=None,
            )
        else:
            dev_metrics = evaluate_one_epoch(
                data_loader=dev_loader,
                epoch=epoch,
                description=(
                    f"Fast DEV "
                    f"{epoch_one_based}/{NUM_EPOCHS}"
                ),
                maximum_batches=FAST_DEV_BATCHES,
            )

        candidate_selection = (
            build_selection_record(
                dev_metrics
            )
        )

        improved = False

        # Best checkpoint는 전체 DEV 평가에서만 갱신
        if run_full_dev:
            improved = candidate_is_better(
                candidate=candidate_selection,
                best_selection=best_selection,
            )

            if improved:
                best_selection = (
                    candidate_selection
                )
                best_epoch = epoch

                full_dev_evaluations_without_improvement = 0
            elif (
                epoch_one_based
                >= EARLY_STOPPING_START_EPOCH
                and best_selection is not None
                and best_selection["eligible"]
            ):
                full_dev_evaluations_without_improvement += 1

        epoch_duration = (
            time.time() - epoch_start_time
        )

        epoch_record = {
            "epoch": int(epoch),
            "epoch_one_based": int(
                epoch_one_based
            ),
            "duration_seconds": float(
                epoch_duration
            ),
            "full_dev_evaluation": bool(
                run_full_dev
            ),
            "objective_weights": (
                get_objective_weights(epoch)
            ),
            "anti_collapse_schedule": (
                anti_collapse_schedule(epoch)
            ),
            "hierarchy_computed": bool(
                should_compute_hierarchy(epoch)
            ),
            "train": train_metrics,
            "dev": dev_metrics,
            "candidate_selection": (
                candidate_selection
            ),
            "best_selection": (
                best_selection
            ),
            "improved": bool(improved),
            "full_dev_evaluations_without_improvement": int(
                full_dev_evaluations_without_improvement
            ),
        }

        history.append(epoch_record)

        append_jsonl(
            epoch_record,
            HISTORY_JSONL_PATH,
        )

        # latest는 매 epoch 저장
        save_checkpoint(
            path=LATEST_CHECKPOINT_PATH,
            epoch=epoch,
            history=history,
            best_selection=best_selection,
        )

        if (
            epoch_one_based
            % CHECKPOINT_INTERVAL_EPOCHS
            == 0
        ):
            periodic_path = os.path.join(
                RUN_CHECKPOINT_DIR,
                f"epoch_{epoch_one_based:03d}.pt",
            )

            save_checkpoint(
                path=periodic_path,
                epoch=epoch,
                history=history,
                best_selection=best_selection,
            )

        if run_full_dev and improved:
            save_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                epoch=epoch,
                history=history,
                best_selection=best_selection,
            )

        print(
            f"\nEpoch {epoch_one_based:02d} | "
            f"{epoch_duration / 60:.1f} min | "
            f"train={train_metrics['total']:.5f} | "
            f"dev={dev_metrics['total']:.5f}"
        )

        print(
            f"Train speed       : "
            f"{train_metrics['patents_per_second']:.2f} patents/s"
        )

        print(
            f"Peak GPU          : "
            f"{train_metrics['peak_gpu_gib']:.2f} GiB"
        )

        print(
            f"DEV active topics : "
            f"{dev_metrics['active_topics']}/"
            f"{CONFIG.num_topics}"
        )

        print(
            f"DEV max share     : "
            f"{dev_metrics['maximum_topic_share']:.2%}"
        )

        print(
            f"DEV marginal H    : "
            f"{dev_metrics['normalized_marginal_entropy']:.4f}"
        )

        print(
            f"Candidate valid   : "
            f"{candidate_selection['eligible']}"
        )

        if run_full_dev:
            print(
                f"Best updated      : {improved}"
            )

        if best_selection is not None:
            print(
                f"Current best      : "
                f"active={best_selection['active_topics']}, "
                f"share="
                f"{best_selection['maximum_topic_share']:.2%}, "
                f"H="
                f"{best_selection['marginal_entropy']:.4f}"
            )

        if (
            run_full_dev
            and epoch_one_based
            >= EARLY_STOPPING_START_EPOCH
            and best_selection is not None
            and best_selection["eligible"]
            and full_dev_evaluations_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True

            print(
                "\nEarly stopping activated."
            )
            break

        gc.collect()

except Exception as training_error:
    error_trace = traceback.format_exc()

    print("\n" + "!" * 90)
    print("TRAINING ERROR")
    print("!" * 90)
    print(error_trace)

    with open(
        ERROR_LOG_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(error_trace)

    gc.collect()
    torch.cuda.empty_cache()

    raise


# ============================================================
# 20. Ensure best checkpoint exists
# ============================================================

if not os.path.isfile(BEST_CHECKPOINT_PATH):
    print(
        "[WARNING] 유효한 best checkpoint가 없어 "
        "latest.pt를 best.pt로 사용합니다."
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
# 21. Restore best checkpoint
# ============================================================

best_checkpoint = safe_torch_load(
    BEST_CHECKPOINT_PATH,
    map_location="cpu",
)

depth_ot_model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

topic_anchor.load_state_dict(
    best_checkpoint[
        "topic_anchor_state_dict"
    ]
)

depth_ot_model.to(DEVICE)
topic_anchor.to(DEVICE)

best_epoch = int(
    best_checkpoint["epoch"]
)

print(
    f"\nBest checkpoint restored: "
    f"epoch {best_epoch + 1}"
)


# ============================================================
# 22. Final full evaluation
# ============================================================

print("\n" + "=" * 90)
print("FINAL FULL EVALUATION")
print("=" * 90)

final_dev_metrics = evaluate_one_epoch(
    data_loader=dev_loader,
    epoch=best_epoch,
    description="Final full DEV",
    maximum_batches=None,
)

final_test_metrics = evaluate_one_epoch(
    data_loader=test_loader,
    epoch=best_epoch,
    description="Final full TEST",
    maximum_batches=None,
)

with torch.no_grad():
    final_anchor_coordinates = (
        topic_anchor.coordinates()
        .detach()
        .cpu()
    )

    final_cost_matrix = (
        topic_anchor.cost_matrix()
        .detach()
        .cpu()
    )


# ============================================================
# 23. Save final outputs
# ============================================================

execution_duration = (
    time.time() - execution_start_time
)

training_summary = {
    "run_name": RUN_NAME,
    "feature_run_name": FEATURE_RUN_NAME,
    "seed": SEED,
    "fresh_training": True,
    "resume": False,
    "completed_at": datetime.now().isoformat(),
    "duration_seconds": float(
        execution_duration
    ),
    "stopped_early": bool(
        stopped_early
    ),
    "epochs_recorded": len(history),
    "best_epoch": int(best_epoch),
    "best_epoch_one_based": int(
        best_epoch + 1
    ),
    "best_selection": (
        best_checkpoint.get(
            "best_selection"
        )
    ),
    "final_dev": final_dev_metrics,
    "final_test": final_test_metrics,
    "latest_checkpoint": (
        LATEST_CHECKPOINT_PATH
    ),
    "best_checkpoint": (
        BEST_CHECKPOINT_PATH
    ),
    "history_path": (
        HISTORY_JSONL_PATH
    ),
    "run_manifest": (
        RUN_MANIFEST_PATH
    ),
    "learned_anchor": (
        LEARNED_ANCHOR_PATH
    ),
    "training_settings": {
        "patents_per_batch": (
            PATENTS_PER_BATCH
        ),
        "num_epochs": NUM_EPOCHS,
        "anti_collapse": True,
        "t4_optimized": True,
        "mixed_precision": (
            USE_MIXED_PRECISION
        ),
        "tf32": True,
        "full_dev_every_epochs": (
            FULL_DEV_EVERY_EPOCHS
        ),
        "fast_dev_batches": (
            FAST_DEV_BATCHES
        ),
        "num_workers": getattr(
            train_loader,
            "num_workers",
            0,
        ),
    },
}

atomic_json_save(
    training_summary,
    TRAINING_SUMMARY_PATH,
)

atomic_torch_save(
    {
        "coordinates": (
            final_anchor_coordinates
        ),
        "cost_matrix": final_cost_matrix,
        "best_epoch": int(best_epoch),
        "best_epoch_one_based": int(
            best_epoch + 1
        ),
        "run_name": RUN_NAME,
    },
    LEARNED_ANCHOR_PATH,
)


# ============================================================
# 24. Validate outputs
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
        "필수 출력 파일이 없습니다: "
        f"{missing_final_files}"
    )


# ============================================================
# 25. Final report
# ============================================================

print("\n" + "=" * 90)
print("SECTION 6 COMPLETED SUCCESSFULLY")
print("=" * 90)

print(f"Run name          : {RUN_NAME}")
print(f"GPU               : {GPU_NAME}")
print(f"Best epoch        : {best_epoch + 1}")
print(f"Stopped early     : {stopped_early}")

print(
    f"DEV active topics : "
    f"{final_dev_metrics['active_topics']}/"
    f"{CONFIG.num_topics}"
)

print(
    f"DEV max share     : "
    f"{final_dev_metrics['maximum_topic_share']:.2%}"
)

print(
    f"DEV marginal H    : "
    f"{final_dev_metrics['normalized_marginal_entropy']:.4f}"
)

print(
    f"TEST active topics: "
    f"{final_test_metrics['active_topics']}/"
    f"{CONFIG.num_topics}"
)

print(
    f"TEST max share    : "
    f"{final_test_metrics['maximum_topic_share']:.2%}"
)

print(
    f"TEST marginal H   : "
    f"{final_test_metrics['normalized_marginal_entropy']:.4f}"
)

print(f"Best checkpoint   : {BEST_CHECKPOINT_PATH}")
print(f"Latest checkpoint : {LATEST_CHECKPOINT_PATH}")
print(f"Training summary  : {TRAINING_SUMMARY_PATH}")
print(f"Learned anchor    : {LEARNED_ANCHOR_PATH}")

print("=" * 90)

anti_collapse_passed = (
    final_dev_metrics["active_topics"]
    >= MIN_ACTIVE_TOPICS
    and final_dev_metrics[
        "maximum_topic_share"
    ]
    <= MAX_TOPIC_SHARE
    and final_dev_metrics[
        "normalized_marginal_entropy"
    ]
    >= MIN_MARGINAL_ENTROPY
)

if anti_collapse_passed:
    print(
        "\n[PASS] Anti-collapse 기준을 충족했습니다."
    )
    print(
        "Section 7 inference를 실행할 수 있습니다."
    )
else:
    print(
        "\n[WARNING] Anti-collapse 기준을 충족하지 못했습니다."
    )
    print(
        "3개 seed로 확장하지 말고 현재 로그를 먼저 확인하세요."
    )
