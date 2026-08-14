# ============================================================
# SECTION 4/5 BOW FIX — USE NORMALIZED CLAIM BOW
# ============================================================

from typing import Any, Dict, Mapping
import torch


def _resolve_batch_tensors_with_normalized_bow(
    self,
    batch: Mapping[str, Any],
) -> Dict[str, Any]:
    # Call the original Section 4 resolver, not the failed hotfix.
    original_resolver = (
        DepthOTV2Model
        ._original_resolve_batch_tensors_before_bow_hotfix
    )

    resolved = original_resolver(self, batch)

    if "bow_normalized" not in batch:
        raise KeyError(
            "Expected `bow_normalized` in the Section 3 batch. "
            f"Available keys: {sorted(batch.keys())}"
        )

    claim_bow = batch["bow_normalized"]

    if not torch.is_tensor(claim_bow):
        raise TypeError(
            "`bow_normalized` must be a torch.Tensor."
        )

    number_of_claims = int(
        resolved["claim_to_patent"].numel()
    )

    expected_shape = (
        number_of_claims,
        int(self.vocabulary_size),
    )

    if tuple(claim_bow.shape) != expected_shape:
        raise ValueError(
            f"bow_normalized shape={tuple(claim_bow.shape)}, "
            f"expected={expected_shape}"
        )

    resolved["claim_bow"] = claim_bow
    resolved["claim_bow_counts"] = batch.get("bow_counts")
    resolved["claim_bow_valid"] = batch.get("bow_valid")
    resolved["claim_bow_source_key"] = "bow_normalized"

    return resolved


DepthOTV2Model.resolve_batch_tensors = (
    _resolve_batch_tensors_with_normalized_bow
)


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

if "_section5_move_to_device" in globals():
    sample_batch_for_bow_check = (
        _section5_move_to_device(
            section4_sample_batch,
            DEVICE,
        )
    )
else:
    sample_batch_for_bow_check = _move_to_device(
        section4_sample_batch,
        DEVICE,
    )

depth_ot_v2_model.eval()

with torch.no_grad():
    resolved_check = (
        depth_ot_v2_model.resolve_batch_tensors(
            sample_batch_for_bow_check
        )
    )

claim_bow = resolved_check["claim_bow"]
row_sums = claim_bow.sum(dim=-1)
valid_mask = row_sums > 0

if not torch.isfinite(claim_bow).all():
    raise RuntimeError(
        "Non-finite values found in bow_normalized."
    )

if valid_mask.any():
    maximum_valid_sum_error = float(
        torch.max(
            torch.abs(
                row_sums[valid_mask]
                - torch.ones_like(row_sums[valid_mask])
            )
        ).item()
    )
else:
    maximum_valid_sum_error = 0.0

if maximum_valid_sum_error > 1.0e-4:
    raise RuntimeError(
        "bow_normalized rows do not sum to one. "
        f"Maximum error={maximum_valid_sum_error:.3e}"
    )

print("=" * 80)
print("NORMALIZED CLAIM BOW FIX INSTALLED")
print("=" * 80)
print("Selected key          : bow_normalized")
print(f"BoW shape             : {tuple(claim_bow.shape)}")
print(f"BoW dtype             : {claim_bow.dtype}")
print(
    f"Valid BoW rows        : "
    f"{int(valid_mask.sum().item())}/{claim_bow.shape[0]}"
)
print(
    f"Empty BoW rows        : "
    f"{int((~valid_mask).sum().item())}"
)
print(
    f"Max valid sum error   : "
    f"{maximum_valid_sum_error:.3e}"
)
print("=" * 80)
print("[PASS] Section 5 전체 셀을 다시 실행하세요.")


# ============================================================
# SECTION 5 — DEPTH-OT V2 OBJECTIVE AND HIERARCHY OT
# Patent-level EMA usage + topic separation + float64 Sinkhorn
# ============================================================

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 5.0 Preconditions
# ------------------------------------------------------------

_REQUIRED_GLOBALS = [
    "CONFIG",
    "DEVICE",
    "RUN_LOG_DIR",
    "depth_ot_v2_model",
    "section4_sample_batch",
    "NUM_TOPICS",
    "VOCAB_SIZE",
]

_missing_globals = [
    name for name in _REQUIRED_GLOBALS
    if name not in globals()
]

if _missing_globals:
    raise RuntimeError(
        "Section 5 실행 전에 필요한 전역 변수가 없습니다: "
        + ", ".join(_missing_globals)
        + "\nSection 0–4를 먼저 실행하세요."
    )

DEVICE = torch.device(DEVICE)
RUN_LOG_DIR = Path(RUN_LOG_DIR)
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

SECTION5_MANIFEST_PATH = (
    RUN_LOG_DIR / "section5_objective_manifest.json"
)

MODEL_EPSILON = float(
    globals().get("MODEL_EPSILON", 1.0e-8)
)


# ------------------------------------------------------------
# 5.1 Objective and OT configuration
# ------------------------------------------------------------

RECONSTRUCTION_WEIGHT = float(
    getattr(CONFIG, "reconciliation_weight", 0.10)
)

TOPIC_EMBEDDING_SEPARATION_MAX_WEIGHT = float(
    getattr(
        CONFIG,
        "topic_embedding_separation_max_weight",
        0.10,
    )
)

BETA_SEPARATION_MAX_WEIGHT = float(
    getattr(
        CONFIG,
        "beta_separation_max_weight",
        0.02,
    )
)

TOPIC_COSINE_MARGIN = float(
    getattr(CONFIG, "topic_cosine_margin", 0.20)
)

PATENT_USAGE_EMA_DECAY = float(
    getattr(CONFIG, "patent_usage_ema_decay", 0.99)
)

PATENT_USAGE_CURRENT_BLEND = float(
    getattr(CONFIG, "patent_usage_current_blend", 0.25)
)

PATENT_USAGE_MAX_WEIGHT = float(
    getattr(CONFIG, "patent_usage_max_weight", 0.05)
)

HIERARCHY_REFERENCE_MIX = float(
    getattr(CONFIG, "hierarchy_reference_mix", 0.10)
)

HIERARCHY_DIRECTION_MARGIN = float(
    getattr(CONFIG, "hierarchy_direction_margin", 0.03)
)

HIERARCHY_OT_EPSILON = float(
    getattr(CONFIG, "hierarchy_ot_epsilon", 0.05)
)

SINKHORN_MAX_ITERATIONS = int(
    getattr(CONFIG, "sinkhorn_max_iterations", 300)
)

SINKHORN_MIN_ITERATIONS = int(
    getattr(CONFIG, "sinkhorn_min_iterations", 20)
)

SINKHORN_TOLERANCE = float(
    getattr(CONFIG, "sinkhorn_tolerance", 1.0e-6)
)

SINKHORN_COMPUTE_DTYPE = torch.float64

ANCHOR_REPULSION_WEIGHT = float(
    getattr(CONFIG, "anchor_repulsion_weight", 0.05)
)

ANCHOR_MINIMUM_SEPARATION = float(
    getattr(CONFIG, "anchor_minimum_separation", 0.02)
)

MAX_ACCEPTABLE_OT_MARGINAL_ERROR = float(
    getattr(CONFIG, "maximum_ot_marginal_error", 1.0e-4)
)

if not 0.0 < PATENT_USAGE_EMA_DECAY < 1.0:
    raise ValueError(
        "PATENT_USAGE_EMA_DECAY must be in (0, 1), "
        f"got {PATENT_USAGE_EMA_DECAY}"
    )

if not 0.0 < PATENT_USAGE_CURRENT_BLEND <= 1.0:
    raise ValueError(
        "PATENT_USAGE_CURRENT_BLEND must be in (0, 1], "
        f"got {PATENT_USAGE_CURRENT_BLEND}"
    )

if not 0.0 <= HIERARCHY_REFERENCE_MIX <= 1.0:
    raise ValueError(
        "HIERARCHY_REFERENCE_MIX must be in [0, 1], "
        f"got {HIERARCHY_REFERENCE_MIX}"
    )

if HIERARCHY_OT_EPSILON <= 0.0:
    raise ValueError(
        "HIERARCHY_OT_EPSILON must be positive."
    )

if SINKHORN_MIN_ITERATIONS > SINKHORN_MAX_ITERATIONS:
    raise ValueError(
        "sinkhorn_min_iterations cannot exceed "
        "sinkhorn_max_iterations."
    )


# ------------------------------------------------------------
# 5.2 Utility functions
# ------------------------------------------------------------

def _section5_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _section5_atomic_json_dump(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)

        os.replace(temporary_path, path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink(missing_ok=True)


def _section5_move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)

    if isinstance(value, dict):
        return {
            key: _section5_move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _section5_move_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _section5_move_to_device(item, device)
            for item in value
        )

    return value


def _normalize_probability_vector(
    vector: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    vector = vector.clamp_min(epsilon)
    return vector / vector.sum().clamp_min(epsilon)


def _normalize_probability_matrix(
    matrix: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    matrix = matrix.clamp_min(epsilon)
    return matrix / matrix.sum().clamp_min(epsilon)


def _safe_kl_divergence(
    distribution_p: torch.Tensor,
    distribution_q: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    p = distribution_p.clamp_min(epsilon)
    q = distribution_q.clamp_min(epsilon)

    p = p / p.sum().clamp_min(epsilon)
    q = q / q.sum().clamp_min(epsilon)

    return torch.sum(p * (torch.log(p) - torch.log(q)))


def _normalized_entropy(
    distribution: torch.Tensor,
    dimension: int = -1,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    distribution = distribution.clamp_min(epsilon)
    distribution = distribution / distribution.sum(
        dim=dimension,
        keepdim=True,
    ).clamp_min(epsilon)

    entropy = -torch.sum(
        distribution * torch.log(distribution),
        dim=dimension,
    )

    number_of_categories = int(distribution.shape[dimension])

    if number_of_categories <= 1:
        return torch.zeros_like(entropy)

    return entropy / math.log(number_of_categories)


def _maximum_off_diagonal_cosine(
    matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normalized = F.normalize(
        matrix,
        p=2,
        dim=-1,
        eps=MODEL_EPSILON,
    )

    cosine_matrix = torch.matmul(
        normalized,
        normalized.transpose(0, 1),
    )

    number_of_rows = int(cosine_matrix.shape[0])

    if number_of_rows <= 1:
        zero = cosine_matrix.new_zeros(())
        return zero, zero

    mask = ~torch.eye(
        number_of_rows,
        dtype=torch.bool,
        device=cosine_matrix.device,
    )

    values = cosine_matrix[mask]

    return values.mean(), values.max()


def _global_gradient_norm(
    parameters: Sequence[torch.nn.Parameter],
) -> float:
    squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():
            return float("nan")

        squared_norm += float(
            torch.sum(gradient.float().pow(2)).item()
        )

    return math.sqrt(squared_norm)


# ------------------------------------------------------------
# 5.3 Patent-level EMA usage tracker
# ------------------------------------------------------------

class PatentTopicUsageEMA(nn.Module):
    """
    Tracks the global patent-level topic marginal.

    Unlike the previous batch-balanced Sinkhorn loss, this module does
    not force each mini-batch of eight patents to use all 30 topics.
    The loss is computed from a blend of the detached running marginal
    and the differentiable current patent marginal.
    """

    def __init__(
        self,
        number_of_topics: int,
        decay: float,
        current_blend: float,
    ) -> None:
        super().__init__()

        self.number_of_topics = int(number_of_topics)
        self.decay = float(decay)
        self.current_blend = float(current_blend)

        uniform = torch.full(
            (self.number_of_topics,),
            1.0 / self.number_of_topics,
            dtype=torch.float32,
        )

        self.register_buffer(
            "running_marginal",
            uniform.clone(),
        )
        self.register_buffer(
            "number_of_updates",
            torch.zeros((), dtype=torch.long),
        )

    @torch.no_grad()
    def reset(self) -> None:
        self.running_marginal.fill_(
            1.0 / self.number_of_topics
        )
        self.number_of_updates.zero_()

    @torch.no_grad()
    def update(
        self,
        current_marginal: torch.Tensor,
    ) -> None:
        current = current_marginal.detach().float()
        current = _normalize_probability_vector(current)

        self.running_marginal.mul_(self.decay)
        self.running_marginal.add_(
            current,
            alpha=1.0 - self.decay,
        )
        self.running_marginal.copy_(
            _normalize_probability_vector(
                self.running_marginal
            )
        )

        self.number_of_updates.add_(1)

    def compute(
        self,
        patent_theta: torch.Tensor,
        *,
        update_ema: bool,
    ) -> Dict[str, torch.Tensor]:
        if patent_theta.ndim != 2:
            raise ValueError(
                "patent_theta must have shape [P, K], "
                f"got {tuple(patent_theta.shape)}"
            )

        if patent_theta.shape[1] != self.number_of_topics:
            raise ValueError(
                f"Expected {self.number_of_topics} topics, "
                f"got {patent_theta.shape[1]}"
            )

        current_marginal = patent_theta.mean(dim=0)
        current_marginal = _normalize_probability_vector(
            current_marginal
        )

        running_marginal = self.running_marginal.detach().to(
            device=patent_theta.device,
            dtype=patent_theta.dtype,
        )
        running_marginal = _normalize_probability_vector(
            running_marginal
        )

        surrogate_marginal = (
            (1.0 - self.current_blend) * running_marginal
            + self.current_blend * current_marginal
        )
        surrogate_marginal = _normalize_probability_vector(
            surrogate_marginal
        )

        uniform = torch.full_like(
            surrogate_marginal,
            1.0 / self.number_of_topics,
        )

        # KL(surrogate || uniform), minimized at uniform global usage.
        usage_loss = _safe_kl_divergence(
            surrogate_marginal,
            uniform,
        )

        current_entropy = _normalized_entropy(
            current_marginal
        )
        running_entropy = _normalized_entropy(
            running_marginal
        )
        surrogate_entropy = _normalized_entropy(
            surrogate_marginal
        )

        if update_ema:
            self.update(current_marginal)

        return {
            "loss": usage_loss,
            "current_marginal": current_marginal,
            "running_marginal": running_marginal,
            "surrogate_marginal": surrogate_marginal,
            "current_entropy": current_entropy,
            "running_entropy": running_entropy,
            "surrogate_entropy": surrogate_entropy,
            "current_max_share": current_marginal.max(),
            "running_max_share": running_marginal.max(),
        }


patent_usage_tracker = PatentTopicUsageEMA(
    number_of_topics=int(NUM_TOPICS),
    decay=PATENT_USAGE_EMA_DECAY,
    current_blend=PATENT_USAGE_CURRENT_BLEND,
).to(DEVICE)


# ------------------------------------------------------------
# 5.4 Base neural losses
# ------------------------------------------------------------

def compute_bow_reconstruction_loss(
    log_word_probabilities: torch.Tensor,
    claim_bow: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if log_word_probabilities is None:
        raise ValueError(
            "log_word_probabilities cannot be None when "
            "computing reconstruction loss."
        )

    if claim_bow is None:
        raise ValueError(
            "claim_bow is required for reconstruction loss."
        )

    claim_bow = claim_bow.to(
        device=log_word_probabilities.device,
        dtype=log_word_probabilities.dtype,
    )

    if claim_bow.shape != log_word_probabilities.shape:
        raise ValueError(
            "BoW/output shape mismatch: "
            f"bow={tuple(claim_bow.shape)}, "
            f"output={tuple(log_word_probabilities.shape)}"
        )

    row_mass = claim_bow.sum(dim=-1)
    valid_mask = row_mass > 0

    per_claim_loss = -torch.sum(
        claim_bow * log_word_probabilities,
        dim=-1,
    )

    if valid_mask.any():
        reconstruction_loss = per_claim_loss[
            valid_mask
        ].mean()
    else:
        reconstruction_loss = (
            log_word_probabilities.sum() * 0.0
        )

    return {
        "loss": reconstruction_loss,
        "per_claim_loss": per_claim_loss,
        "valid_mask": valid_mask,
        "valid_claims": valid_mask.sum(),
        "empty_claims": (~valid_mask).sum(),
    }


def compute_logistic_normal_kl_loss(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
) -> torch.Tensor:
    if posterior_mean.shape != posterior_log_variance.shape:
        raise ValueError(
            "Posterior mean/log-variance shape mismatch: "
            f"{tuple(posterior_mean.shape)} vs "
            f"{tuple(posterior_log_variance.shape)}"
        )

    per_claim_kl = 0.5 * torch.sum(
        torch.exp(posterior_log_variance)
        + posterior_mean.pow(2)
        - 1.0
        - posterior_log_variance,
        dim=-1,
    )

    return per_claim_kl.mean()


def compute_patent_confidence_diagnostic(
    patent_theta: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    patent_entropy = _normalized_entropy(
        patent_theta,
        dimension=-1,
    )

    confidence = 1.0 - patent_entropy

    return {
        "mean_normalized_entropy": patent_entropy.mean(),
        "mean_confidence": confidence.mean(),
        "maximum_topic_probability": (
            patent_theta.max(dim=-1).values.mean()
        ),
    }


# ------------------------------------------------------------
# 5.5 Sinkhorn solver
# ------------------------------------------------------------

@dataclass
class SinkhornResult:
    coupling: torch.Tensor
    iterations: int
    converged: bool
    source_error: float
    target_error: float
    maximum_error: float


def solve_sinkhorn_scaling(
    kernel: torch.Tensor,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    *,
    maximum_iterations: int = SINKHORN_MAX_ITERATIONS,
    minimum_iterations: int = SINKHORN_MIN_ITERATIONS,
    tolerance: float = SINKHORN_TOLERANCE,
) -> SinkhornResult:
    """
    Projects a positive kernel onto the requested source and target
    marginals using float64 Sinkhorn scaling.
    """

    kernel = kernel.to(dtype=SINKHORN_COMPUTE_DTYPE)
    source_marginal = source_marginal.to(
        device=kernel.device,
        dtype=SINKHORN_COMPUTE_DTYPE,
    )
    target_marginal = target_marginal.to(
        device=kernel.device,
        dtype=SINKHORN_COMPUTE_DTYPE,
    )

    source_marginal = _normalize_probability_vector(
        source_marginal,
        epsilon=1.0e-15,
    )
    target_marginal = _normalize_probability_vector(
        target_marginal,
        epsilon=1.0e-15,
    )

    if kernel.ndim != 2:
        raise ValueError(
            f"Kernel must be rank 2, got {tuple(kernel.shape)}"
        )

    if kernel.shape != (
        source_marginal.numel(),
        target_marginal.numel(),
    ):
        raise ValueError(
            f"Kernel shape={tuple(kernel.shape)} does not match "
            f"marginals=({source_marginal.numel()}, "
            f"{target_marginal.numel()})"
        )

    if not torch.isfinite(kernel).all():
        raise FloatingPointError(
            "Non-finite value found in Sinkhorn kernel."
        )

    kernel = kernel.clamp_min(1.0e-300)
    kernel = kernel / kernel.max().clamp_min(1.0e-300)

    source_scaling = torch.ones_like(source_marginal)
    target_scaling = torch.ones_like(target_marginal)

    converged = False
    source_error = float("inf")
    target_error = float("inf")
    maximum_error = float("inf")
    iteration = 0

    for iteration in range(1, maximum_iterations + 1):
        kernel_times_target = torch.mv(
            kernel,
            target_scaling,
        ).clamp_min(1.0e-300)

        source_scaling = (
            source_marginal / kernel_times_target
        )

        kernel_transpose_times_source = torch.mv(
            kernel.transpose(0, 1),
            source_scaling,
        ).clamp_min(1.0e-300)

        target_scaling = (
            target_marginal
            / kernel_transpose_times_source
        )

        if (
            iteration >= minimum_iterations
            and (
                iteration == minimum_iterations
                or iteration % 5 == 0
                or iteration == maximum_iterations
            )
        ):
            coupling = (
                source_scaling.unsqueeze(1)
                * kernel
                * target_scaling.unsqueeze(0)
            )

            actual_source = coupling.sum(dim=1)
            actual_target = coupling.sum(dim=0)

            source_error = float(
                torch.max(
                    torch.abs(
                        actual_source - source_marginal
                    )
                ).item()
            )
            target_error = float(
                torch.max(
                    torch.abs(
                        actual_target - target_marginal
                    )
                ).item()
            )
            maximum_error = max(
                source_error,
                target_error,
            )

            if maximum_error <= tolerance:
                converged = True
                break

    coupling = (
        source_scaling.unsqueeze(1)
        * kernel
        * target_scaling.unsqueeze(0)
    )

    coupling = coupling.clamp_min(0.0)
    coupling = coupling / coupling.sum().clamp_min(
        1.0e-300
    )

    actual_source = coupling.sum(dim=1)
    actual_target = coupling.sum(dim=0)

    source_error = float(
        torch.max(
            torch.abs(
                actual_source - source_marginal
            )
        ).item()
    )
    target_error = float(
        torch.max(
            torch.abs(
                actual_target - target_marginal
            )
        ).item()
    )
    maximum_error = max(source_error, target_error)

    if not torch.isfinite(coupling).all():
        raise FloatingPointError(
            "Sinkhorn produced non-finite coupling."
        )

    if maximum_error > MAX_ACCEPTABLE_OT_MARGINAL_ERROR:
        raise RuntimeError(
            "Sinkhorn failed to satisfy its requested marginals. "
            f"source_error={source_error:.3e}, "
            f"target_error={target_error:.3e}, "
            f"iterations={iteration}. "
            "This error is computed from the same marginals supplied "
            "to the solver; it is not the previous parent-marginal "
            "comparison bug."
        )

    return SinkhornResult(
        coupling=coupling,
        iterations=int(iteration),
        converged=bool(converged),
        source_error=source_error,
        target_error=target_error,
        maximum_error=maximum_error,
    )


# ------------------------------------------------------------
# 5.6 Depth-transition statistics
# ------------------------------------------------------------

def _prepare_hierarchy_edges(
    claim_edge_index: Optional[torch.Tensor],
    claim_depth: Optional[torch.Tensor],
    claim_to_patent: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = claim_to_patent.device

    empty_long = torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )

    if (
        claim_edge_index is None
        or claim_depth is None
        or claim_edge_index.numel() == 0
    ):
        return {
            "parent": empty_long,
            "child": empty_long,
            "parent_depth": empty_long,
            "adjacent_mask": torch.empty(
                0,
                dtype=torch.bool,
                device=device,
            ),
            "same_patent_mask": torch.empty(
                0,
                dtype=torch.bool,
                device=device,
            ),
        }

    claim_edge_index = claim_edge_index.long()
    claim_depth = claim_depth.long().reshape(-1)
    claim_to_patent = claim_to_patent.long().reshape(-1)

    if (
        claim_edge_index.ndim != 2
        or claim_edge_index.shape[0] != 2
    ):
        raise ValueError(
            "claim_edge_index must have shape [2, E], "
            f"got {tuple(claim_edge_index.shape)}"
        )

    number_of_claims = int(claim_to_patent.numel())

    if claim_depth.numel() != number_of_claims:
        raise ValueError(
            f"claim_depth length={claim_depth.numel()} does not "
            f"match claims={number_of_claims}"
        )

    if claim_edge_index.numel() > 0:
        minimum_index = int(claim_edge_index.min().item())
        maximum_index = int(claim_edge_index.max().item())

        if minimum_index < 0 or maximum_index >= number_of_claims:
            raise IndexError(
                f"Claim edge index out of range: "
                f"min={minimum_index}, max={maximum_index}, "
                f"claims={number_of_claims}"
            )

    parent_all = claim_edge_index[0]
    child_all = claim_edge_index[1]

    same_patent_mask = (
        claim_to_patent[parent_all]
        == claim_to_patent[child_all]
    )

    adjacent_mask = (
        claim_depth[child_all]
        == claim_depth[parent_all] + 1
    )

    valid_mask = same_patent_mask & adjacent_mask

    parent = parent_all[valid_mask]
    child = child_all[valid_mask]
    parent_depth = claim_depth[parent]

    return {
        "parent": parent,
        "child": child,
        "parent_depth": parent_depth,
        "adjacent_mask": adjacent_mask,
        "same_patent_mask": same_patent_mask,
        "all_parent": parent_all,
        "all_child": child_all,
        "valid_mask": valid_mask,
    }


def compute_empirical_transition_coupling(
    parent_theta: torch.Tensor,
    child_theta: torch.Tensor,
) -> torch.Tensor:
    if parent_theta.shape != child_theta.shape:
        raise ValueError(
            "Parent and child theta shapes differ: "
            f"{tuple(parent_theta.shape)} vs "
            f"{tuple(child_theta.shape)}"
        )

    if parent_theta.ndim != 2:
        raise ValueError(
            "Parent and child theta must be rank 2."
        )

    number_of_edges = int(parent_theta.shape[0])

    if number_of_edges == 0:
        raise ValueError(
            "Cannot compute a coupling from zero edges."
        )

    coupling = torch.einsum(
        "ek,el->kl",
        parent_theta,
        child_theta,
    )

    coupling = coupling / float(number_of_edges)
    coupling = coupling.clamp_min(MODEL_EPSILON)
    coupling = coupling / coupling.sum().clamp_min(
        MODEL_EPSILON
    )

    return coupling


def compute_directional_cost(
    anchor_coordinates: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    parent_coordinate = anchor_coordinates.unsqueeze(1)
    child_coordinate = anchor_coordinates.unsqueeze(0)

    violation = (
        float(margin)
        + parent_coordinate
        - child_coordinate
    )

    return F.relu(violation).pow(2)


def build_reference_transport_kernel(
    empirical_coupling: torch.Tensor,
    reconciled_marginal: torch.Tensor,
    directional_cost: torch.Tensor,
    *,
    reference_mix: float,
    ot_epsilon: float,
) -> torch.Tensor:
    empirical = empirical_coupling.to(
        dtype=SINKHORN_COMPUTE_DTYPE
    )
    reconciled = reconciled_marginal.to(
        dtype=SINKHORN_COMPUTE_DTYPE
    )
    cost = directional_cost.to(
        dtype=SINKHORN_COMPUTE_DTYPE
    )

    independent = torch.outer(
        reconciled,
        reconciled,
    )

    reference = (
        (1.0 - float(reference_mix)) * empirical
        + float(reference_mix) * independent
    )
    reference = reference.clamp_min(1.0e-15)
    reference = reference / reference.sum().clamp_min(
        1.0e-15
    )

    shifted_cost = cost - cost.min()

    transport_kernel = reference * torch.exp(
        -shifted_cost / float(ot_epsilon)
    )

    transport_kernel = transport_kernel.clamp_min(
        1.0e-300
    )

    return transport_kernel


# ------------------------------------------------------------
# 5.7 Hierarchy loss
# ------------------------------------------------------------

def compute_hierarchy_objective(
    model: nn.Module,
    outputs: Mapping[str, Any],
    *,
    compute_ot: bool,
) -> Dict[str, Any]:
    claim_theta = outputs["claim_theta"]
    claim_edge_index = outputs.get("claim_edge_index")
    claim_depth = outputs.get("claim_depth")
    claim_to_patent = outputs["claim_to_patent"]

    zero = claim_theta.sum() * 0.0

    prepared_edges = _prepare_hierarchy_edges(
        claim_edge_index=claim_edge_index,
        claim_depth=claim_depth,
        claim_to_patent=claim_to_patent,
    )

    parent = prepared_edges["parent"]
    child = prepared_edges["child"]
    parent_depth = prepared_edges["parent_depth"]

    total_input_edges = 0

    if claim_edge_index is not None:
        total_input_edges = int(claim_edge_index.shape[1])

    number_of_adjacent_edges = int(parent.numel())

    if not compute_ot or number_of_adjacent_edges == 0:
        return {
            "hierarchy_loss": zero,
            "reconciliation_loss": zero,
            "number_of_input_edges": total_input_edges,
            "number_of_adjacent_edges": number_of_adjacent_edges,
            "number_of_depth_groups": 0,
            "maximum_sinkhorn_error": 0.0,
            "mean_sinkhorn_iterations": 0.0,
            "sinkhorn_all_converged": True,
            "directional_violation_mass": zero.detach(),
            "depth_statistics": [],
        }

    unique_depths = torch.unique(
        parent_depth,
        sorted=True,
    )

    hierarchy_weighted_sum = zero
    reconciliation_weighted_sum = zero
    directional_weighted_sum = zero

    total_group_weight = 0
    maximum_sinkhorn_error = 0.0
    sinkhorn_iterations: List[int] = []
    sinkhorn_convergence: List[bool] = []
    depth_statistics: List[Dict[str, Any]] = []

    # Neural hierarchy loss must not update anchor parameters.
    detached_anchors = (
        model.get_anchor_coordinates().detach()
    )

    detached_directional_cost = compute_directional_cost(
        anchor_coordinates=detached_anchors,
        margin=HIERARCHY_DIRECTION_MARGIN,
    )

    for depth_tensor in unique_depths:
        depth_value = int(depth_tensor.item())
        depth_mask = parent_depth == depth_tensor

        group_parent = parent[depth_mask]
        group_child = child[depth_mask]
        group_edge_count = int(group_parent.numel())

        if group_edge_count == 0:
            continue

        empirical_coupling = (
            compute_empirical_transition_coupling(
                parent_theta=claim_theta[group_parent],
                child_theta=claim_theta[group_child],
            )
        )

        source_marginal = empirical_coupling.sum(dim=1)
        target_marginal = empirical_coupling.sum(dim=0)

        reconciled_marginal = 0.5 * (
            source_marginal + target_marginal
        )
        reconciled_marginal = (
            _normalize_probability_vector(
                reconciled_marginal
            )
        )

        reconciliation_loss = 0.5 * (
            _safe_kl_divergence(
                source_marginal,
                reconciled_marginal,
            )
            + _safe_kl_divergence(
                target_marginal,
                reconciled_marginal,
            )
        )

        # The OT solution is a detached target for the neural encoder.
        with torch.no_grad():
            detached_empirical = (
                empirical_coupling.detach().to(
                    dtype=SINKHORN_COMPUTE_DTYPE
                )
            )
            detached_marginal = (
                reconciled_marginal.detach().to(
                    dtype=SINKHORN_COMPUTE_DTYPE
                )
            )

            transport_kernel = (
                build_reference_transport_kernel(
                    empirical_coupling=detached_empirical,
                    reconciled_marginal=detached_marginal,
                    directional_cost=(
                        detached_directional_cost
                    ),
                    reference_mix=(
                        HIERARCHY_REFERENCE_MIX
                    ),
                    ot_epsilon=HIERARCHY_OT_EPSILON,
                )
            )

            sinkhorn_result = solve_sinkhorn_scaling(
                kernel=transport_kernel,
                source_marginal=detached_marginal,
                target_marginal=detached_marginal,
            )

            ot_target = sinkhorn_result.coupling.to(
                device=claim_theta.device,
                dtype=claim_theta.dtype,
            )

        hierarchy_loss = _safe_kl_divergence(
            empirical_coupling,
            ot_target,
        )

        directional_violation_mass = torch.sum(
            empirical_coupling
            * detached_directional_cost.to(
                device=claim_theta.device,
                dtype=claim_theta.dtype,
            )
        )

        hierarchy_weighted_sum = (
            hierarchy_weighted_sum
            + group_edge_count * hierarchy_loss
        )
        reconciliation_weighted_sum = (
            reconciliation_weighted_sum
            + group_edge_count * reconciliation_loss
        )
        directional_weighted_sum = (
            directional_weighted_sum
            + group_edge_count
            * directional_violation_mass
        )

        total_group_weight += group_edge_count

        maximum_sinkhorn_error = max(
            maximum_sinkhorn_error,
            sinkhorn_result.maximum_error,
        )

        sinkhorn_iterations.append(
            sinkhorn_result.iterations
        )
        sinkhorn_convergence.append(
            sinkhorn_result.converged
        )

        depth_statistics.append({
            "parent_depth": depth_value,
            "number_of_edges": group_edge_count,
            "sinkhorn_iterations": (
                sinkhorn_result.iterations
            ),
            "sinkhorn_converged": (
                sinkhorn_result.converged
            ),
            "sinkhorn_source_error": (
                sinkhorn_result.source_error
            ),
            "sinkhorn_target_error": (
                sinkhorn_result.target_error
            ),
            "sinkhorn_maximum_error": (
                sinkhorn_result.maximum_error
            ),
            "source_target_l1": float(
                torch.sum(
                    torch.abs(
                        source_marginal.detach()
                        - target_marginal.detach()
                    )
                ).item()
            ),
        })

    if total_group_weight <= 0:
        return {
            "hierarchy_loss": zero,
            "reconciliation_loss": zero,
            "number_of_input_edges": total_input_edges,
            "number_of_adjacent_edges": number_of_adjacent_edges,
            "number_of_depth_groups": 0,
            "maximum_sinkhorn_error": 0.0,
            "mean_sinkhorn_iterations": 0.0,
            "sinkhorn_all_converged": True,
            "directional_violation_mass": zero.detach(),
            "depth_statistics": [],
        }

    hierarchy_loss = (
        hierarchy_weighted_sum / total_group_weight
    )
    reconciliation_loss = (
        reconciliation_weighted_sum
        / total_group_weight
    )
    directional_violation_mass = (
        directional_weighted_sum
        / total_group_weight
    )

    return {
        "hierarchy_loss": hierarchy_loss,
        "reconciliation_loss": reconciliation_loss,
        "number_of_input_edges": total_input_edges,
        "number_of_adjacent_edges": number_of_adjacent_edges,
        "number_of_depth_groups": len(depth_statistics),
        "maximum_sinkhorn_error": (
            maximum_sinkhorn_error
        ),
        "mean_sinkhorn_iterations": (
            float(sum(sinkhorn_iterations))
            / max(1, len(sinkhorn_iterations))
        ),
        "sinkhorn_all_converged": all(
            sinkhorn_convergence
        ),
        "directional_violation_mass": (
            directional_violation_mass.detach()
        ),
        "depth_statistics": depth_statistics,
    }


# ------------------------------------------------------------
# 5.8 Separate anchor objective
# ------------------------------------------------------------

def compute_anchor_objective(
    model: nn.Module,
    outputs: Mapping[str, Any],
    *,
    compute_anchor_loss: bool,
) -> Dict[str, Any]:
    claim_theta = outputs["claim_theta"]
    claim_edge_index = outputs.get("claim_edge_index")
    claim_depth = outputs.get("claim_depth")
    claim_to_patent = outputs["claim_to_patent"]

    anchors = model.get_anchor_coordinates()
    zero = anchors.sum() * 0.0

    anchor_repulsion = model.anchor_repulsion_loss(
        minimum_separation=ANCHOR_MINIMUM_SEPARATION
    )

    prepared_edges = _prepare_hierarchy_edges(
        claim_edge_index=claim_edge_index,
        claim_depth=claim_depth,
        claim_to_patent=claim_to_patent,
    )

    parent = prepared_edges["parent"]
    child = prepared_edges["child"]
    parent_depth = prepared_edges["parent_depth"]

    if (
        not compute_anchor_loss
        or parent.numel() == 0
    ):
        return {
            "anchor_total": zero,
            "anchor_directional": zero,
            "anchor_repulsion": anchor_repulsion,
            "number_of_adjacent_edges": int(
                parent.numel()
            ),
            "number_of_depth_groups": 0,
        }

    directional_cost = compute_directional_cost(
        anchor_coordinates=anchors,
        margin=HIERARCHY_DIRECTION_MARGIN,
    )

    unique_depths = torch.unique(
        parent_depth,
        sorted=True,
    )

    directional_sum = zero
    total_edges = 0
    valid_groups = 0

    for depth_tensor in unique_depths:
        depth_mask = parent_depth == depth_tensor

        group_parent = parent[depth_mask]
        group_child = child[depth_mask]
        group_edge_count = int(group_parent.numel())

        if group_edge_count == 0:
            continue

        # Theta is detached so the anchor optimizer cannot alter
        # neural encoder parameters.
        empirical_coupling = (
            compute_empirical_transition_coupling(
                parent_theta=(
                    claim_theta[group_parent].detach()
                ),
                child_theta=(
                    claim_theta[group_child].detach()
                ),
            )
        )

        directional_loss = torch.sum(
            empirical_coupling * directional_cost
        )

        directional_sum = (
            directional_sum
            + group_edge_count * directional_loss
        )

        total_edges += group_edge_count
        valid_groups += 1

    if total_edges > 0:
        anchor_directional = (
            directional_sum / total_edges
        )
    else:
        anchor_directional = zero

    anchor_total = (
        anchor_directional
        + ANCHOR_REPULSION_WEIGHT * anchor_repulsion
    )

    return {
        "anchor_total": anchor_total,
        "anchor_directional": anchor_directional,
        "anchor_repulsion": anchor_repulsion,
        "number_of_adjacent_edges": total_edges,
        "number_of_depth_groups": valid_groups,
    }


# ------------------------------------------------------------
# 5.9 Complete neural objective
# ------------------------------------------------------------

DEFAULT_OBJECTIVE_WEIGHTS = {
    "kl": 0.0,
    "hierarchy": 0.0,
    "reconciliation": RECONSTRUCTION_WEIGHT,
    "topic_embedding_separation": 0.0,
    "beta_separation": 0.0,
    "patent_usage": 0.0,
}


def compute_depth_ot_v2_objective(
    model: nn.Module,
    outputs: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, float]] = None,
    update_usage_ema: bool,
    compute_hierarchy: bool,
) -> Dict[str, Any]:
    effective_weights = dict(DEFAULT_OBJECTIVE_WEIGHTS)

    if weights is not None:
        for name, value in weights.items():
            if name not in effective_weights:
                raise KeyError(
                    f"Unknown objective weight: {name}"
                )

            effective_weights[name] = float(value)

    reconstruction = compute_bow_reconstruction_loss(
        log_word_probabilities=outputs[
            "log_word_probabilities"
        ],
        claim_bow=outputs["claim_bow"],
    )

    kl_loss = compute_logistic_normal_kl_loss(
        posterior_mean=outputs["posterior_mean"],
        posterior_log_variance=outputs[
            "posterior_log_variance"
        ],
    )

    usage = patent_usage_tracker.compute(
        patent_theta=outputs["patent_theta"],
        update_ema=update_usage_ema,
    )

    topic_embedding_separation = (
        model.topic_embedding_separation_loss(
            margin=TOPIC_COSINE_MARGIN
        )
    )

    beta_separation = (
        model.beta_cosine_separation_loss(
            margin=TOPIC_COSINE_MARGIN
        )
    )

    hierarchy = compute_hierarchy_objective(
        model=model,
        outputs=outputs,
        compute_ot=compute_hierarchy,
    )

    total = reconstruction["loss"]
    total = (
        total
        + effective_weights["kl"] * kl_loss
        + effective_weights["hierarchy"]
        * hierarchy["hierarchy_loss"]
        + effective_weights["reconciliation"]
        * hierarchy["reconciliation_loss"]
        + effective_weights[
            "topic_embedding_separation"
        ]
        * topic_embedding_separation
        + effective_weights["beta_separation"]
        * beta_separation
        + effective_weights["patent_usage"]
        * usage["loss"]
    )

    confidence = compute_patent_confidence_diagnostic(
        outputs["patent_theta"]
    )

    beta_mean_cosine, beta_max_cosine = (
        _maximum_off_diagonal_cosine(
            outputs["beta"]
        )
    )

    topic_mean_cosine, topic_max_cosine = (
        _maximum_off_diagonal_cosine(
            model.get_topic_embeddings()
        )
    )

    result = {
        "total": total,
        "reconstruction": reconstruction["loss"],
        "kl": kl_loss,
        "hierarchy": hierarchy["hierarchy_loss"],
        "reconciliation": hierarchy[
            "reconciliation_loss"
        ],
        "topic_embedding_separation": (
            topic_embedding_separation
        ),
        "beta_separation": beta_separation,
        "patent_usage": usage["loss"],
        "weights": effective_weights,

        "valid_bow_claims": reconstruction[
            "valid_claims"
        ],
        "empty_bow_claims": reconstruction[
            "empty_claims"
        ],

        "usage_current_marginal": usage[
            "current_marginal"
        ],
        "usage_running_marginal": usage[
            "running_marginal"
        ],
        "usage_surrogate_marginal": usage[
            "surrogate_marginal"
        ],
        "usage_current_entropy": usage[
            "current_entropy"
        ],
        "usage_running_entropy": usage[
            "running_entropy"
        ],
        "usage_current_max_share": usage[
            "current_max_share"
        ],
        "usage_running_max_share": usage[
            "running_max_share"
        ],

        "patent_mean_entropy": confidence[
            "mean_normalized_entropy"
        ],
        "patent_mean_confidence": confidence[
            "mean_confidence"
        ],
        "patent_mean_max_probability": confidence[
            "maximum_topic_probability"
        ],

        "beta_mean_cosine": beta_mean_cosine,
        "beta_max_cosine": beta_max_cosine,
        "topic_mean_cosine": topic_mean_cosine,
        "topic_max_cosine": topic_max_cosine,

        "hierarchy_number_of_input_edges": hierarchy[
            "number_of_input_edges"
        ],
        "hierarchy_number_of_adjacent_edges": hierarchy[
            "number_of_adjacent_edges"
        ],
        "hierarchy_number_of_depth_groups": hierarchy[
            "number_of_depth_groups"
        ],
        "sinkhorn_maximum_error": hierarchy[
            "maximum_sinkhorn_error"
        ],
        "sinkhorn_mean_iterations": hierarchy[
            "mean_sinkhorn_iterations"
        ],
        "sinkhorn_all_converged": hierarchy[
            "sinkhorn_all_converged"
        ],
        "directional_violation_mass": hierarchy[
            "directional_violation_mass"
        ],
        "depth_statistics": hierarchy[
            "depth_statistics"
        ],
    }

    for name in [
        "total",
        "reconstruction",
        "kl",
        "hierarchy",
        "reconciliation",
        "topic_embedding_separation",
        "beta_separation",
        "patent_usage",
    ]:
        value = result[name]

        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Non-finite objective component: {name}"
            )

    return result


# Compatibility alias used by Section 6.
compute_objective = compute_depth_ot_v2_objective


# ------------------------------------------------------------
# 5.10 Neural and anchor parameter separation
# ------------------------------------------------------------

def get_anchor_parameters(
    model: nn.Module,
) -> List[nn.Parameter]:
    return list(model.depth_anchors.parameters())


def get_neural_parameters(
    model: nn.Module,
) -> List[nn.Parameter]:
    anchor_parameter_ids = {
        id(parameter)
        for parameter in get_anchor_parameters(model)
    }

    return [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in anchor_parameter_ids
    ]


anchor_parameters = get_anchor_parameters(
    depth_ot_v2_model
)
neural_parameters = get_neural_parameters(
    depth_ot_v2_model
)

anchor_parameter_ids = {
    id(parameter)
    for parameter in anchor_parameters
}
neural_parameter_ids = {
    id(parameter)
    for parameter in neural_parameters
}

if anchor_parameter_ids & neural_parameter_ids:
    raise RuntimeError(
        "Anchor and neural parameter groups overlap."
    )

if (
    len(anchor_parameter_ids | neural_parameter_ids)
    != len(
        {
            id(parameter)
            for parameter
            in depth_ot_v2_model.parameters()
        }
    )
):
    raise RuntimeError(
        "Some model parameters are missing from neural/anchor "
        "parameter groups."
    )


# ------------------------------------------------------------
# 5.11 Section validation
# ------------------------------------------------------------

SECTION5_VALIDATION_WEIGHTS = {
    "kl": 0.01,
    "hierarchy": 0.01,
    "reconciliation": RECONSTRUCTION_WEIGHT,
    "topic_embedding_separation": 0.10,
    "beta_separation": 0.02,
    "patent_usage": 0.05,
}

section5_sample_batch_device = (
    _section5_move_to_device(
        section4_sample_batch,
        DEVICE,
    )
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

depth_ot_v2_model.train()
depth_ot_v2_model.zero_grad(set_to_none=True)

section5_outputs = depth_ot_v2_model(
    section5_sample_batch_device,
    sample=True,
    decode=True,
)

section5_losses = compute_depth_ot_v2_objective(
    model=depth_ot_v2_model,
    outputs=section5_outputs,
    weights=SECTION5_VALIDATION_WEIGHTS,
    update_usage_ema=False,
    compute_hierarchy=True,
)

section5_anchor_losses = compute_anchor_objective(
    model=depth_ot_v2_model,
    outputs=section5_outputs,
    compute_anchor_loss=True,
)

# Validate the separate anchor gradient first.
anchor_gradient_tuple = torch.autograd.grad(
    section5_anchor_losses["anchor_total"],
    anchor_parameters,
    retain_graph=True,
    allow_unused=True,
)

anchor_gradient_is_finite = True
anchor_gradient_squared_norm = 0.0

for gradient in anchor_gradient_tuple:
    if gradient is None:
        continue

    if not torch.isfinite(gradient).all():
        anchor_gradient_is_finite = False

    anchor_gradient_squared_norm += float(
        torch.sum(gradient.detach().float().pow(2)).item()
    )

anchor_gradient_norm = math.sqrt(
    anchor_gradient_squared_norm
)

if not anchor_gradient_is_finite:
    raise FloatingPointError(
        "Non-finite gradient found in anchor objective."
    )

# Validate the main neural objective.
section5_losses["total"].backward()

neural_gradient_norm = _global_gradient_norm(
    neural_parameters
)

if not math.isfinite(neural_gradient_norm):
    raise FloatingPointError(
        "Non-finite neural gradient found."
    )

# Neural objective must not update independent anchors.
anchor_gradient_from_neural_objective = (
    _global_gradient_norm(anchor_parameters)
)

if not math.isfinite(
    anchor_gradient_from_neural_objective
):
    raise FloatingPointError(
        "Non-finite anchor gradient found after neural backward."
    )

if anchor_gradient_from_neural_objective > 1.0e-10:
    raise RuntimeError(
        "Neural objective unexpectedly produced an anchor "
        f"gradient: {anchor_gradient_from_neural_objective:.3e}"
    )

depth_ot_v2_model.zero_grad(set_to_none=True)

if torch.cuda.is_available():
    section5_peak_gpu_gib = (
        torch.cuda.max_memory_allocated(DEVICE)
        / (1024 ** 3)
    )
else:
    section5_peak_gpu_gib = 0.0


# ------------------------------------------------------------
# 5.12 Validation values
# ------------------------------------------------------------

def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())

    return float(value)


validation_summary = {
    "total": _scalar(section5_losses["total"]),
    "reconstruction": _scalar(
        section5_losses["reconstruction"]
    ),
    "kl": _scalar(section5_losses["kl"]),
    "hierarchy": _scalar(
        section5_losses["hierarchy"]
    ),
    "reconciliation": _scalar(
        section5_losses["reconciliation"]
    ),
    "topic_embedding_separation": _scalar(
        section5_losses[
            "topic_embedding_separation"
        ]
    ),
    "beta_separation": _scalar(
        section5_losses["beta_separation"]
    ),
    "patent_usage": _scalar(
        section5_losses["patent_usage"]
    ),
    "usage_current_entropy": _scalar(
        section5_losses["usage_current_entropy"]
    ),
    "usage_current_max_share": _scalar(
        section5_losses["usage_current_max_share"]
    ),
    "patent_mean_entropy": _scalar(
        section5_losses["patent_mean_entropy"]
    ),
    "patent_mean_confidence": _scalar(
        section5_losses["patent_mean_confidence"]
    ),
    "beta_mean_cosine": _scalar(
        section5_losses["beta_mean_cosine"]
    ),
    "beta_max_cosine": _scalar(
        section5_losses["beta_max_cosine"]
    ),
    "topic_mean_cosine": _scalar(
        section5_losses["topic_mean_cosine"]
    ),
    "topic_max_cosine": _scalar(
        section5_losses["topic_max_cosine"]
    ),
    "anchor_total": _scalar(
        section5_anchor_losses["anchor_total"]
    ),
    "anchor_directional": _scalar(
        section5_anchor_losses[
            "anchor_directional"
        ]
    ),
    "anchor_repulsion": _scalar(
        section5_anchor_losses[
            "anchor_repulsion"
        ]
    ),
    "neural_gradient_norm": neural_gradient_norm,
    "anchor_gradient_norm": anchor_gradient_norm,
    "anchor_gradient_from_neural_objective": (
        anchor_gradient_from_neural_objective
    ),
    "peak_gpu_gib": section5_peak_gpu_gib,
}


# ------------------------------------------------------------
# 5.13 Save manifest
# ------------------------------------------------------------

section5_manifest = {
    "created_at_utc": _section5_utc_now(),
    "run_name": str(
        getattr(CONFIG, "run_name", "depth_ot_v2")
    ),
    "model_version": str(
        getattr(CONFIG, "model_version", "depth_ot_v2")
    ),
    "device": str(DEVICE),
    "objective": {
        "reconstruction": (
            "claim-level normalized BoW negative log-likelihood"
        ),
        "kl": (
            "claim-level logistic-normal KL divergence"
        ),
        "patent_usage": (
            "patent-level EMA surrogate marginal KL to uniform"
        ),
        "claim_batch_balanced_sinkhorn": False,
        "topic_embedding_separation": (
            "off-diagonal cosine margin penalty"
        ),
        "beta_separation": (
            "topic-word distribution cosine margin penalty"
        ),
        "hierarchy": (
            "empirical adjacent-depth coupling KL to "
            "directional reference OT coupling"
        ),
        "reconciliation": (
            "source/target marginal reconciliation"
        ),
        "anchor_optimization": "separate optimizer",
        "cpc_used": False,
    },
    "usage_configuration": {
        "ema_decay": PATENT_USAGE_EMA_DECAY,
        "current_blend": PATENT_USAGE_CURRENT_BLEND,
        "maximum_weight": PATENT_USAGE_MAX_WEIGHT,
    },
    "separation_configuration": {
        "topic_cosine_margin": TOPIC_COSINE_MARGIN,
        "topic_embedding_max_weight": (
            TOPIC_EMBEDDING_SEPARATION_MAX_WEIGHT
        ),
        "beta_max_weight": (
            BETA_SEPARATION_MAX_WEIGHT
        ),
    },
    "hierarchy_configuration": {
        "reference_mix": HIERARCHY_REFERENCE_MIX,
        "direction_margin": HIERARCHY_DIRECTION_MARGIN,
        "ot_epsilon": HIERARCHY_OT_EPSILON,
        "sinkhorn_dtype": "float64",
        "sinkhorn_max_iterations": (
            SINKHORN_MAX_ITERATIONS
        ),
        "sinkhorn_min_iterations": (
            SINKHORN_MIN_ITERATIONS
        ),
        "sinkhorn_tolerance": (
            SINKHORN_TOLERANCE
        ),
        "maximum_acceptable_marginal_error": (
            MAX_ACCEPTABLE_OT_MARGINAL_ERROR
        ),
        "anchor_minimum_separation": (
            ANCHOR_MINIMUM_SEPARATION
        ),
        "anchor_repulsion_weight": (
            ANCHOR_REPULSION_WEIGHT
        ),
    },
    "parameter_groups": {
        "neural_parameter_tensors": len(
            neural_parameters
        ),
        "anchor_parameter_tensors": len(
            anchor_parameters
        ),
        "overlap": False,
    },
    "validation_weights": (
        SECTION5_VALIDATION_WEIGHTS
    ),
    "validation": validation_summary,
    "hierarchy_validation": {
        "input_edges": int(
            section5_losses[
                "hierarchy_number_of_input_edges"
            ]
        ),
        "adjacent_edges": int(
            section5_losses[
                "hierarchy_number_of_adjacent_edges"
            ]
        ),
        "depth_groups": int(
            section5_losses[
                "hierarchy_number_of_depth_groups"
            ]
        ),
        "sinkhorn_maximum_error": float(
            section5_losses[
                "sinkhorn_maximum_error"
            ]
        ),
        "sinkhorn_mean_iterations": float(
            section5_losses[
                "sinkhorn_mean_iterations"
            ]
        ),
        "sinkhorn_all_converged": bool(
            section5_losses[
                "sinkhorn_all_converged"
            ]
        ),
        "depth_statistics": (
            section5_losses["depth_statistics"]
        ),
    },
}

_section5_atomic_json_dump(
    section5_manifest,
    SECTION5_MANIFEST_PATH,
)


# ------------------------------------------------------------
# 5.14 Cleanup
# ------------------------------------------------------------

depth_ot_v2_model.zero_grad(set_to_none=True)
depth_ot_v2_model.train()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ------------------------------------------------------------
# 5.15 Final report
# ------------------------------------------------------------

print("\n" + "=" * 94)
print("SECTION 5 — DEPTH-OT V2 OBJECTIVE AND HIERARCHY OT COMPLETED")
print("=" * 94)
print(f"Device                         : {DEVICE}")
print(f"Topics                         : {NUM_TOPICS}")
print(f"Vocabulary                     : {VOCAB_SIZE:,}")
print("-" * 94)
print(
    f"Total validation loss          : "
    f"{validation_summary['total']:.6f}"
)
print(
    f"BoW reconstruction             : "
    f"{validation_summary['reconstruction']:.6f}"
)
print(
    f"KL                              : "
    f"{validation_summary['kl']:.6f}"
)
print(
    f"Hierarchy                       : "
    f"{validation_summary['hierarchy']:.6f}"
)
print(
    f"Reconciliation                  : "
    f"{validation_summary['reconciliation']:.6f}"
)
print(
    f"Topic embedding separation      : "
    f"{validation_summary['topic_embedding_separation']:.6f}"
)
print(
    f"Beta separation                 : "
    f"{validation_summary['beta_separation']:.6f}"
)
print(
    f"Patent usage EMA loss           : "
    f"{validation_summary['patent_usage']:.6f}"
)
print("-" * 94)
print(
    f"Patent current usage entropy    : "
    f"{validation_summary['usage_current_entropy']:.6f}"
)
print(
    f"Patent current max share        : "
    f"{100.0 * validation_summary['usage_current_max_share']:.2f}%"
)
print(
    f"Patent mean assignment entropy  : "
    f"{validation_summary['patent_mean_entropy']:.6f}"
)
print(
    f"Patent mean confidence          : "
    f"{validation_summary['patent_mean_confidence']:.6f}"
)
print("-" * 94)
print(
    f"Beta mean cosine                : "
    f"{validation_summary['beta_mean_cosine']:.6f}"
)
print(
    f"Beta max cosine                 : "
    f"{validation_summary['beta_max_cosine']:.6f}"
)
print(
    f"Topic embedding mean cosine     : "
    f"{validation_summary['topic_mean_cosine']:.6f}"
)
print(
    f"Topic embedding max cosine      : "
    f"{validation_summary['topic_max_cosine']:.6f}"
)
print("-" * 94)
print(
    f"Hierarchy input edges           : "
    f"{section5_losses['hierarchy_number_of_input_edges']}"
)
print(
    f"Adjacent-depth edges            : "
    f"{section5_losses['hierarchy_number_of_adjacent_edges']}"
)
print(
    f"Depth groups                    : "
    f"{section5_losses['hierarchy_number_of_depth_groups']}"
)
print(
    f"Sinkhorn maximum error          : "
    f"{section5_losses['sinkhorn_maximum_error']:.3e}"
)
print(
    f"Sinkhorn mean iterations        : "
    f"{section5_losses['sinkhorn_mean_iterations']:.1f}"
)
print(
    f"Sinkhorn all converged          : "
    f"{section5_losses['sinkhorn_all_converged']}"
)
print("-" * 94)
print(
    f"Anchor total                    : "
    f"{validation_summary['anchor_total']:.6f}"
)
print(
    f"Anchor directional              : "
    f"{validation_summary['anchor_directional']:.6f}"
)
print(
    f"Anchor repulsion                : "
    f"{validation_summary['anchor_repulsion']:.6f}"
)
print(
    f"Neural gradient norm            : "
    f"{validation_summary['neural_gradient_norm']:.6f}"
)
print(
    f"Anchor gradient norm            : "
    f"{validation_summary['anchor_gradient_norm']:.6f}"
)
print(
    f"Anchor grad from neural loss     : "
    f"{validation_summary['anchor_gradient_from_neural_objective']:.3e}"
)
print(
    f"Validation peak GPU memory       : "
    f"{validation_summary['peak_gpu_gib']:.2f} GiB"
)
print("-" * 94)
print("Claim batch-balanced Sinkhorn   : DISABLED")
print("Patent usage regularization     : EMA-based")
print("Hierarchy Sinkhorn dtype        : float64")
print("Neural/anchor optimizer groups  : SEPARATE")
print("CPC used in objective           : False")
print(f"Manifest                        : {SECTION5_MANIFEST_PATH}")
print("=" * 94)
print("[PASS] Section 5 objective and gradient validation completed.")
print("[NEXT] 위의 전체 마지막 요약을 보내주세요.")
