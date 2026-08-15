# ======================================================================================
# TraCo T4/L4 TURBO TRAINING + RESUME + EVALUATION
#
# 사용 방법:
#   1. 현재 느리게 실행 중인 기존 셀 중지
#   2. 이 셀 전체를 바로 실행
#
# 전제:
#   기존 코드 Section 0~12가 실행되어 다음 객체들이 메모리에 있어야 함:
#   train_bow, test_bow, vocab, NUM_TOPICS_LIST, RESULT_DIR,
#   CHECKPOINT_DIR, ARRAY_DIR, TOPIC_DIR, HIERARCHY_DIR,
#   TraCo 및 평가/저장 utility functions
# ======================================================================================

import os
import gc
import time
import json
import random
import types
import pickle
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


# ======================================================================================
# 0. Preflight
# ======================================================================================

REQUIRED_OBJECTS = [
    "train_bow",
    "test_bow",
    "vocab",
    "TraCo",
    "NUM_TOPICS_LIST",
    "RESULT_DIR",
    "CHECKPOINT_DIR",
    "ARRAY_DIR",
    "TOPIC_DIR",
    "HIERARCHY_DIR",
    "valid_cpc_mask",
    "CPC_LABELS",
    "MATCHED_LEVEL_MAP",
    "evaluate_matched_cpc",
    "evaluate_all_cross_levels",
    "get_top_word_rows",
    "build_hierarchy_edge_rows",
    "atomic_torch_save",
    "atomic_json_save",
    "get_rng_state",
    "restore_rng_state",
    "optimizer_to_device",
]

missing_objects = [
    name
    for name in REQUIRED_OBJECTS
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        "기존 코드 Section 0~12 실행이 필요합니다.\n"
        f"누락 객체: {missing_objects}"
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. Colab 런타임을 T4/L4로 설정하세요."
    )

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GIB = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print("=" * 100)
print("TraCo T4/L4 TURBO TRAINING")
print("=" * 100)
print(f"GPU         : {GPU_NAME}")
print(f"GPU memory  : {GPU_MEMORY_GIB:.2f} GiB")
print(f"PyTorch     : {torch.__version__}")
print(f"Train BoW   : {train_bow.shape}")
print(f"Test BoW    : {test_bow.shape}")
print("=" * 100)


# ======================================================================================
# 1. Turbo configuration
# ======================================================================================

SEEDS = [42, 43, 44]

EPOCHS = 200

# T4 15GB 권장값.
# OOM이면 768 또는 512로 낮추면 됨.
BATCH_SIZE = 1024

LEARNING_RATE = 0.002

ENCODER_UNITS = 300
DROPOUT = 0.0
EMBED_SIZE = 200

BIAS_TOPK = 20
BIAS_P = 5.0
BETA_TEMP = 0.1

WEIGHT_LOSS_TPD = 20.0
SINKHORN_ALPHA = 20.0
SINKHORN_MAX_ITER = 1000

GRADIENT_CLIP_NORM = 5.0

CHECKPOINT_INTERVAL = 10
NUM_TOP_WORDS = 25

# 최종 inference
INFERENCE_BATCH_SIZE = 2048

# T4에서는 encoder/decoder만 FP16.
# Sinkhorn, beta, KL은 FP32 유지.
USE_MIXED_PRECISION = True

# tqdm 및 loss.item() 호출 간격
PROGRESS_UPDATE_INTERVAL = 20
FINITE_CHECK_INTERVAL = 20

# Train BoW 전체를 GPU에 올릴 최대 비율.
# 15GB T4에서 모델/activation 여유 공간을 남긴다.
GPU_BOW_MAX_MEMORY_FRACTION = 0.55


# ======================================================================================
# 2. Clean interrupted model objects
# ======================================================================================

# 중단된 기존 training loop의 모델/배치가 GPU에 남아 있을 수 있음.
for object_name in [
    "model",
    "optimizer",
    "scaler",
    "batch",
    "output",
    "loss",
    "theta_list",
    "beta_list",
    "phi_list",
    "train_theta_list",
    "test_theta_list",
]:
    globals().pop(object_name, None)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()


# ======================================================================================
# 3. Fast CUDA settings
# ======================================================================================

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = True

if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def set_seed_fast(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


set_seed_fast(42)


# ======================================================================================
# 4. GPU train BoW cache
# ======================================================================================

TRAIN_BOW_GPU = None

required_train_bow_bytes = (
    int(train_bow.shape[0])
    * int(train_bow.shape[1])
    * 4
)

free_gpu_bytes, total_gpu_bytes = (
    torch.cuda.mem_get_info()
)

print("\n" + "=" * 100)
print("GPU TRAIN BOW CACHE")
print("=" * 100)
print(
    f"Required dense FP32 memory : "
    f"{required_train_bow_bytes / 2**30:.2f} GiB"
)
print(
    f"Currently free GPU memory  : "
    f"{free_gpu_bytes / 2**30:.2f} GiB"
)
print(
    f"Total GPU memory           : "
    f"{total_gpu_bytes / 2**30:.2f} GiB"
)

cache_allowed_bytes = min(
    int(total_gpu_bytes * GPU_BOW_MAX_MEMORY_FRACTION),
    int(free_gpu_bytes * 0.80),
)

if required_train_bow_bytes <= cache_allowed_bytes:
    try:
        print("[GPU CACHE] Allocating train BoW on GPU")

        TRAIN_BOW_GPU = torch.empty(
            train_bow.shape,
            dtype=torch.float32,
            device=DEVICE,
        )

        GPU_CACHE_CHUNK_SIZE = 2048

        for cache_start in tqdm(
            range(
                0,
                train_bow.shape[0],
                GPU_CACHE_CHUNK_SIZE,
            ),
            desc="Train BoW -> GPU",
            dynamic_ncols=True,
        ):
            cache_end = min(
                cache_start + GPU_CACHE_CHUNK_SIZE,
                train_bow.shape[0],
            )

            dense_chunk = (
                train_bow[
                    cache_start:cache_end
                ]
                .toarray()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            chunk_tensor = torch.from_numpy(
                dense_chunk
            )

            TRAIN_BOW_GPU[
                cache_start:cache_end
            ].copy_(
                chunk_tensor,
                non_blocking=False,
            )

            del dense_chunk
            del chunk_tensor

        torch.cuda.synchronize()

        print(
            f"[GPU CACHE READY] "
            f"shape={tuple(TRAIN_BOW_GPU.shape)}"
        )
        print(
            f"[GPU CACHE READY] "
            f"allocated="
            f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB"
        )

    except torch.cuda.OutOfMemoryError:
        print(
            "[GPU CACHE WARNING] OOM 발생. "
            "CPU CSR fallback을 사용합니다."
        )

        TRAIN_BOW_GPU = None

        gc.collect()
        torch.cuda.empty_cache()

else:
    print(
        "[GPU CACHE SKIP] Train BoW가 너무 커서 "
        "GPU 전체 캐시는 사용하지 않습니다."
    )

print("=" * 100)


# ======================================================================================
# 5. Fast batch transfer
# ======================================================================================

def fast_batch_to_gpu(
    csr_matrix,
    indices,
):
    global TRAIN_BOW_GPU

    # Train matrix가 GPU에 캐시된 경우
    if (
        TRAIN_BOW_GPU is not None
        and csr_matrix is train_bow
    ):
        gpu_indices = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=DEVICE,
        )

        return torch.index_select(
            TRAIN_BOW_GPU,
            dim=0,
            index=gpu_indices,
        )

    # Test 또는 cache 미사용
    dense = (
        csr_matrix[indices]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    tensor = torch.from_numpy(dense)

    return tensor.to(
        DEVICE,
        non_blocking=False,
    )


# 기존 이름도 교체
sparse_batch_to_gpu = fast_batch_to_gpu


# ======================================================================================
# 6. AMP-safe TraCo forward
# ======================================================================================

def install_amp_safe_traco_forward(model):
    """
    선택적 mixed precision:

    FP32:
      - TPD / Sinkhorn
      - beta distance/softmax
      - KL

    AMP FP16:
      - MLP encoder
      - theta/beta reconstruction
    """

    def amp_safe_forward(self, input_bow):
        # ------------------------------------------------------
        # TPD/Sinkhorn: FP32 유지
        # ------------------------------------------------------
        with torch.autocast(
            device_type="cuda",
            enabled=False,
        ):
            loss_tpd, transport_plans = self.TPD(
                self.topic_embeddings_list,
                self.weight_loss_TPD,
            )

        self.transp_list = transport_plans

        # ------------------------------------------------------
        # Encoder/theta: 바깥쪽 autocast 사용
        # ------------------------------------------------------
        theta_list, mu, logvar = self.get_theta(
            input_bow
        )

        # ------------------------------------------------------
        # KL 및 beta: FP32
        # ------------------------------------------------------
        with torch.autocast(
            device_type="cuda",
            enabled=False,
        ):
            loss_kl = self.compute_loss_KL(
                mu.float(),
                logvar.float(),
            ).mean()

            beta_list = self.get_beta()

        # ------------------------------------------------------
        # Decoder: 바깥쪽 autocast 사용
        # ------------------------------------------------------
        reconstruction_loss = self.CDDecoder(
            input_bow,
            theta_list,
            beta_list,
        )

        total_loss = (
            loss_tpd
            + loss_kl
            + reconstruction_loss
        )

        return {
            "loss": total_loss,
        }

    model.forward = types.MethodType(
        amp_safe_forward,
        model,
    )

    return model


def create_grad_scaler():
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=USE_MIXED_PRECISION,
        )

    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=USE_MIXED_PRECISION,
        )


# ======================================================================================
# 7. Fast inference
# ======================================================================================

@torch.inference_mode()
def refresh_transport_plans_fast(model):
    model.eval()

    with torch.autocast(
        device_type="cuda",
        enabled=False,
    ):
        _, transport_plans = model.TPD(
            model.topic_embeddings_list,
            model.weight_loss_TPD,
        )

    model.transp_list = transport_plans


@torch.inference_mode()
def infer_hierarchical_theta_fast(
    model,
    csr_matrix,
    batch_size=2048,
    description="Theta inference",
):
    model.eval()
    refresh_transport_plans_fast(model)

    outputs = [
        []
        for _ in NUM_TOPICS_LIST
    ]

    num_documents = csr_matrix.shape[0]

    for start in tqdm(
        range(
            0,
            num_documents,
            batch_size,
        ),
        desc=description,
        leave=False,
        dynamic_ncols=True,
    ):
        end = min(
            start + batch_size,
            num_documents,
        )

        indices = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        batch = fast_batch_to_gpu(
            csr_matrix,
            indices,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_MIXED_PRECISION,
        ):
            theta_list = model.get_theta(batch)

        if isinstance(theta_list, tuple):
            theta_list = theta_list[0]

        for level, theta in enumerate(
            theta_list
        ):
            outputs[level].append(
                theta.float()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        del batch
        del theta_list

    return [
        np.concatenate(
            level_outputs,
            axis=0,
        )
        for level_outputs in outputs
    ]


@torch.inference_mode()
def get_beta_list_fast(model):
    model.eval()

    with torch.autocast(
        device_type="cuda",
        enabled=False,
    ):
        beta_tensors = model.get_beta()

    return [
        beta.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
        for beta in beta_tensors
    ]


@torch.inference_mode()
def get_phi_list_fast(model):
    model.eval()
    refresh_transport_plans_fast(model)

    return [
        phi.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
        for phi in model.get_phi_list()
    ]


# 기존 함수 이름 교체
infer_hierarchical_theta = (
    infer_hierarchical_theta_fast
)
get_beta_list = get_beta_list_fast
get_phi_list = get_phi_list_fast
refresh_transport_plans = (
    refresh_transport_plans_fast
)


# ======================================================================================
# 8. Configuration update
# ======================================================================================

if "configuration" not in globals():
    configuration = {}

configuration.update({
    "model": "TraCo",
    "official_implementation": "TopMost",
    "acceleration_mode": "T4_L4_turbo",
    "seeds": SEEDS,
    "num_topics_list": NUM_TOPICS_LIST,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "encoder_units": ENCODER_UNITS,
    "dropout": DROPOUT,
    "embedding_size": EMBED_SIZE,
    "bias_topk": BIAS_TOPK,
    "bias_p": BIAS_P,
    "beta_temp": BETA_TEMP,
    "weight_loss_tpd": WEIGHT_LOSS_TPD,
    "sinkhorn_alpha": SINKHORN_ALPHA,
    "sinkhorn_max_iter": SINKHORN_MAX_ITER,
    "gradient_clip_norm": GRADIENT_CLIP_NORM,
    "checkpoint_interval": CHECKPOINT_INTERVAL,
    "inference_batch_size": INFERENCE_BATCH_SIZE,
    "mixed_precision": USE_MIXED_PRECISION,
    "mixed_precision_policy": (
        "FP16 encoder/decoder; "
        "FP32 Sinkhorn/beta/KL"
    ),
    "tf32_enabled": True,
    "cudnn_benchmark": True,
    "deterministic_algorithms": False,
    "train_bow_gpu_cached": (
        TRAIN_BOW_GPU is not None
    ),
    "gpu": GPU_NAME,
    "gpu_memory_gib": GPU_MEMORY_GIB,
    "updated_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
})

atomic_json_save(
    configuration,
    RESULT_DIR / "configuration_turbo.json",
)


# ======================================================================================
# 9. Three-seed training
# ======================================================================================

all_seed_results = []
all_matched_rows = []
all_cross_rows = []

total_start = time.time()

for seed_number, seed in enumerate(
    SEEDS,
    start=1,
):
    print("\n" + "=" * 100)
    print(
        f"TraCo TURBO SEED {seed} — "
        f"{seed_number}/{len(SEEDS)}"
    )
    print("=" * 100)

    latest_checkpoint_path = (
        CHECKPOINT_DIR
        / f"seed_{seed}_latest.pt"
    )

    final_model_path = (
        CHECKPOINT_DIR
        / f"seed_{seed}_final.pt"
    )

    metrics_path = (
        RESULT_DIR
        / f"seed_{seed}_metrics.json"
    )

    cross_metrics_path = (
        RESULT_DIR
        / f"seed_{seed}_all_cross_level_metrics.csv"
    )

    # ------------------------------------------------------------------
    # Completed seed
    # ------------------------------------------------------------------
    if (
        final_model_path.exists()
        and metrics_path.exists()
    ):
        print(
            f"[SKIP] Seed {seed}는 이미 완료됐습니다."
        )

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            seed_result = json.load(handle)

        all_seed_results.append(seed_result)

        for cpc_level in [
            "section",
            "class",
            "subclass",
        ]:
            metrics = seed_result[
                "matched_metrics"
            ][cpc_level]

            diagnostics = seed_result[
                "level_diagnostics"
            ][cpc_level]

            all_matched_rows.append({
                "seed": seed,
                "cpc_level": cpc_level,
                "hierarchy_level": metrics[
                    "hierarchy_level"
                ],
                "num_topics": metrics[
                    "num_topics"
                ],
                "pur_p": metrics["pur_p"],
                "pur_a": metrics["pur_a"],
                "nmi": metrics["nmi"],
                "active_topics": diagnostics[
                    "active_topics"
                ],
                "max_topic_share": diagnostics[
                    "max_topic_share"
                ],
                "elapsed_seconds": seed_result[
                    "elapsed_seconds"
                ],
                "peak_gpu_gib": seed_result[
                    "peak_gpu_gib"
                ],
            })

        if cross_metrics_path.exists():
            cross_df = pd.read_csv(
                cross_metrics_path
            )

            all_cross_rows.extend(
                cross_df.to_dict(
                    orient="records"
                )
            )

        continue

    # ------------------------------------------------------------------
    # Seed setup
    # ------------------------------------------------------------------
    set_seed_fast(seed)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = TraCo(
        vocab_size=len(vocab),
        num_topics_list=NUM_TOPICS_LIST,
        en_units=ENCODER_UNITS,
        dropout=DROPOUT,
        embed_size=EMBED_SIZE,
        bias_topk=BIAS_TOPK,
        bias_p=BIAS_P,
        beta_temp=BETA_TEMP,
        weight_loss_TPD=WEIGHT_LOSS_TPD,
        sinkhorn_alpha=SINKHORN_ALPHA,
        sinkhorn_max_iter=SINKHORN_MAX_ITER,
    ).to(DEVICE)

    if USE_MIXED_PRECISION:
        model = install_amp_safe_traco_forward(
            model
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scaler = create_grad_scaler()

    start_epoch = 1
    previous_elapsed = 0.0
    training_history = []

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    if latest_checkpoint_path.exists():
        print(
            f"[RESUME] {latest_checkpoint_path}"
        )

        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if int(checkpoint["seed"]) != seed:
            raise RuntimeError(
                "Checkpoint seed mismatch: "
                f"{checkpoint['seed']} != {seed}"
            )

        if list(
            checkpoint["num_topics_list"]
        ) != list(NUM_TOPICS_LIST):
            raise RuntimeError(
                "Checkpoint hierarchy mismatch."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        model.to(DEVICE)

        optimizer_to_device(
            optimizer,
            DEVICE,
        )

        if (
            "scaler_state_dict" in checkpoint
            and USE_MIXED_PRECISION
        ):
            try:
                scaler.load_state_dict(
                    checkpoint[
                        "scaler_state_dict"
                    ]
                )
            except Exception as error:
                print(
                    "[WARNING] GradScaler state를 "
                    f"불러오지 못했습니다: {error}"
                )

        start_epoch = (
            int(checkpoint["epoch"]) + 1
        )

        previous_elapsed = float(
            checkpoint.get(
                "elapsed_seconds",
                0.0,
            )
        )

        training_history = list(
            checkpoint.get(
                "training_history",
                [],
            )
        )

        restore_rng_state(
            checkpoint.get("rng_state")
        )

        print(
            f"[RESUME] Last epoch="
            f"{start_epoch - 1}, "
            f"next epoch={start_epoch}"
        )

        old_batch_size = (
            checkpoint.get(
                "configuration",
                {},
            ).get("batch_size")
        )

        if (
            old_batch_size is not None
            and int(old_batch_size)
            != BATCH_SIZE
        ):
            print(
                "[RESUME NOTICE] Checkpoint batch size가 "
                f"{old_batch_size}이고 현재는 "
                f"{BATCH_SIZE}입니다."
            )

    seed_start = time.time()
    num_train = train_bow.shape[0]

    num_batches = int(
        np.ceil(
            num_train / BATCH_SIZE
        )
    )

    print(
        f"[TRAIN] Documents={num_train:,}, "
        f"batch={BATCH_SIZE:,}, "
        f"batches/epoch={num_batches:,}, "
        f"AMP={USE_MIXED_PRECISION}, "
        f"GPU-BoW-cache="
        f"{TRAIN_BOW_GPU is not None}"
    )

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        model.train()

        permutation = np.random.permutation(
            num_train
        )

        epoch_loss_sum_gpu = torch.zeros(
            (),
            dtype=torch.float64,
            device=DEVICE,
        )

        epoch_patents = 0
        epoch_start = time.time()

        progress = tqdm(
            range(
                0,
                num_train,
                BATCH_SIZE,
            ),
            total=num_batches,
            desc=(
                f"Seed {seed} "
                f"Epoch {epoch:03d}/{EPOCHS}"
            ),
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
        )

        for batch_number, start in enumerate(
            progress,
            start=1,
        ):
            indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch = fast_batch_to_gpu(
                train_bow,
                indices,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_MIXED_PRECISION,
            ):
                output = model(batch)
                loss = output["loss"]

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP_NORM,
                    error_if_nonfinite=False,
                )
            )

            scaler.step(optimizer)
            scaler.update()

            current_batch_size = len(indices)

            epoch_loss_sum_gpu.add_(
                loss.detach().double()
                * current_batch_size
            )

            epoch_patents += current_batch_size

            # 매 batch .item()을 호출하면 GPU가 대기하므로
            # 일정 간격으로만 확인한다.
            if (
                batch_number
                % FINITE_CHECK_INTERVAL
                == 0
                or batch_number
                == num_batches
            ):
                loss_value = float(
                    loss.detach().item()
                )

                if not np.isfinite(loss_value):
                    raise FloatingPointError(
                        "Non-finite TraCo loss: "
                        f"seed={seed}, "
                        f"epoch={epoch}, "
                        f"batch={batch_number}, "
                        f"loss={loss_value}"
                    )

            if (
                batch_number
                % PROGRESS_UPDATE_INTERVAL
                == 0
                or batch_number
                == num_batches
            ):
                progress.set_postfix({
                    "loss": (
                        f"{float(loss.detach().item()):.3f}"
                    ),
                    "alloc": (
                        f"{torch.cuda.memory_allocated() / 2**30:.1f}G"
                    ),
                    "peak": (
                        f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G"
                    ),
                    "scale": (
                        f"{scaler.get_scale():.0f}"
                        if USE_MIXED_PRECISION
                        else "off"
                    ),
                })

            del batch
            del output
            del loss
            del gradient_norm

        # Epoch 마지막에 한 번만 동기화
        epoch_loss_sum = float(
            epoch_loss_sum_gpu.item()
        )

        del epoch_loss_sum_gpu

        epoch_elapsed = (
            time.time() - epoch_start
        )

        mean_epoch_loss = (
            epoch_loss_sum
            / max(1, epoch_patents)
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        history_record = {
            "epoch": epoch,
            "mean_loss": mean_epoch_loss,
            "elapsed_seconds": epoch_elapsed,
            "learning_rate": current_lr,
            "batch_size": BATCH_SIZE,
            "mixed_precision": (
                USE_MIXED_PRECISION
            ),
            "train_bow_gpu_cached": (
                TRAIN_BOW_GPU is not None
            ),
        }

        training_history.append(
            history_record
        )

        documents_per_second = (
            epoch_patents
            / max(epoch_elapsed, 1e-9)
        )

        print(
            f"[SEED {seed} EPOCH "
            f"{epoch:03d}/{EPOCHS}] "
            f"loss={mean_epoch_loss:.6f} | "
            f"lr={current_lr:.3e} | "
            f"time={epoch_elapsed / 60:.2f}m | "
            f"speed={documents_per_second:.1f} docs/s | "
            f"peak="
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
        )

        # --------------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------------
        if (
            epoch % CHECKPOINT_INTERVAL == 0
            or epoch == EPOCHS
        ):
            current_elapsed = (
                previous_elapsed
                + time.time()
                - seed_start
            )

            checkpoint_payload = {
                "format_version": 2,
                "model": "TraCo",
                "acceleration_mode": (
                    "T4_L4_turbo"
                ),
                "seed": seed,
                "epoch": epoch,
                "num_topics_list": (
                    NUM_TOPICS_LIST
                ),
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scaler_state_dict": (
                    scaler.state_dict()
                ),
                "training_history": (
                    training_history
                ),
                "elapsed_seconds": (
                    current_elapsed
                ),
                "rng_state": get_rng_state(),
                "configuration": (
                    configuration
                ),
                "vocab": vocab,
                "saved_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            atomic_torch_save(
                checkpoint_payload,
                latest_checkpoint_path,
            )

            pd.DataFrame(
                training_history
            ).to_csv(
                RESULT_DIR
                / f"seed_{seed}_training_history.csv",
                index=False,
            )

            print(
                f"[CHECKPOINT] Seed={seed}, "
                f"epoch={epoch}: "
                f"{latest_checkpoint_path}"
            )

    total_seed_elapsed = (
        previous_elapsed
        + time.time()
        - seed_start
    )

    # ==================================================================
    # 10. Final inference
    # ==================================================================

    print(
        f"[INFERENCE] Seed {seed} train theta"
    )

    train_theta_list = (
        infer_hierarchical_theta_fast(
            model=model,
            csr_matrix=train_bow,
            batch_size=INFERENCE_BATCH_SIZE,
            description=(
                f"Seed {seed} train theta"
            ),
        )
    )

    print(
        f"[INFERENCE] Seed {seed} test theta"
    )

    test_theta_list = (
        infer_hierarchical_theta_fast(
            model=model,
            csr_matrix=test_bow,
            batch_size=INFERENCE_BATCH_SIZE,
            description=(
                f"Seed {seed} test theta"
            ),
        )
    )

    beta_list = get_beta_list_fast(model)
    phi_list = get_phi_list_fast(model)

    predicted_by_level = [
        theta.argmax(axis=1)
        for theta in test_theta_list
    ]

    matched_metrics = evaluate_matched_cpc(
        predicted_by_level
    )

    cross_rows = evaluate_all_cross_levels(
        predicted_by_level
    )

    for row in cross_rows:
        row["seed"] = seed

    cross_df = pd.DataFrame(
        cross_rows
    )

    cross_df.to_csv(
        cross_metrics_path,
        index=False,
    )

    all_cross_rows.extend(
        cross_rows
    )

    # ==================================================================
    # 11. Diagnostics
    # ==================================================================

    raw_level_diagnostics = []
    level_diagnostics_by_cpc = {}

    for level, predicted in enumerate(
        predicted_by_level
    ):
        counts = np.bincount(
            predicted,
            minlength=NUM_TOPICS_LIST[level],
        )

        active_topics = int(
            np.count_nonzero(counts)
        )

        max_topic_share = float(
            counts.max()
            / max(1, counts.sum())
        )

        raw_level_diagnostics.append({
            "hierarchy_level": level,
            "num_topics": (
                NUM_TOPICS_LIST[level]
            ),
            "active_topics": (
                active_topics
            ),
            "max_topic_share": (
                max_topic_share
            ),
            "topic_counts": (
                counts.tolist()
            ),
        })

    for cpc_level, hierarchy_level in (
        MATCHED_LEVEL_MAP.items()
    ):
        level_diagnostics_by_cpc[
            cpc_level
        ] = raw_level_diagnostics[
            hierarchy_level
        ]

    peak_gpu_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    # ==================================================================
    # 12. Save arrays
    # ==================================================================

    for level, num_topics in enumerate(
        NUM_TOPICS_LIST
    ):
        np.save(
            ARRAY_DIR
            / (
                f"seed_{seed}_level_{level}_"
                f"k{num_topics}_train_theta.npy"
            ),
            train_theta_list[
                level
            ].astype(np.float32),
        )

        np.save(
            ARRAY_DIR
            / (
                f"seed_{seed}_level_{level}_"
                f"k{num_topics}_test_theta.npy"
            ),
            test_theta_list[
                level
            ].astype(np.float32),
        )

        np.save(
            ARRAY_DIR
            / (
                f"seed_{seed}_level_{level}_"
                f"k{num_topics}_beta.npy"
            ),
            beta_list[
                level
            ].astype(np.float32),
        )

        np.save(
            ARRAY_DIR
            / (
                f"seed_{seed}_level_{level}_"
                f"k{num_topics}_test_assignments.npy"
            ),
            predicted_by_level[
                level
            ].astype(np.int16),
        )

    for parent_level, phi in enumerate(
        phi_list
    ):
        child_level = parent_level + 1

        np.save(
            ARRAY_DIR
            / (
                f"seed_{seed}_phi_"
                f"level_{parent_level}_"
                f"k{NUM_TOPICS_LIST[parent_level]}"
                f"_to_level_{child_level}_"
                f"k{NUM_TOPICS_LIST[child_level]}.npy"
            ),
            phi.astype(np.float32),
        )

    # ==================================================================
    # 13. Topic words and hierarchy
    # ==================================================================

    top_word_rows = get_top_word_rows(
        beta_list=beta_list,
        vocab=vocab,
        num_words=NUM_TOP_WORDS,
    )

    pd.DataFrame(
        top_word_rows
    ).to_csv(
        TOPIC_DIR
        / f"seed_{seed}_top_words.csv",
        index=False,
    )

    hierarchy_edge_rows = (
        build_hierarchy_edge_rows(
            phi_list
        )
    )

    pd.DataFrame(
        hierarchy_edge_rows
    ).to_csv(
        HIERARCHY_DIR
        / f"seed_{seed}_hierarchy_edges.csv",
        index=False,
    )

    # ==================================================================
    # 14. Seed result
    # ==================================================================

    seed_result = {
        "seed": seed,
        "acceleration_mode": (
            "T4_L4_turbo"
        ),
        "batch_size": BATCH_SIZE,
        "mixed_precision": (
            USE_MIXED_PRECISION
        ),
        "train_bow_gpu_cached": (
            TRAIN_BOW_GPU is not None
        ),
        "matched_metrics": (
            matched_metrics
        ),
        "level_diagnostics": (
            level_diagnostics_by_cpc
        ),
        "raw_level_diagnostics": (
            raw_level_diagnostics
        ),
        "elapsed_seconds": (
            total_seed_elapsed
        ),
        "peak_gpu_gib": (
            peak_gpu_gib
        ),
        "train_theta_shapes": [
            list(theta.shape)
            for theta in train_theta_list
        ],
        "test_theta_shapes": [
            list(theta.shape)
            for theta in test_theta_list
        ],
        "beta_shapes": [
            list(beta.shape)
            for beta in beta_list
        ],
        "phi_shapes": [
            list(phi.shape)
            for phi in phi_list
        ],
    }

    atomic_json_save(
        seed_result,
        metrics_path,
    )

    final_payload = {
        "format_version": 2,
        "model": "TraCo",
        "acceleration_mode": (
            "T4_L4_turbo"
        ),
        "seed": seed,
        "epoch": EPOCHS,
        "num_topics_list": (
            NUM_TOPICS_LIST
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "configuration": (
            configuration
        ),
        "vocab": vocab,
        "matched_metrics": (
            matched_metrics
        ),
        "raw_level_diagnostics": (
            raw_level_diagnostics
        ),
        "saved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    atomic_torch_save(
        final_payload,
        final_model_path,
    )

    all_seed_results.append(
        seed_result
    )

    for cpc_level in [
        "section",
        "class",
        "subclass",
    ]:
        metrics = matched_metrics[
            cpc_level
        ]

        diagnostics = (
            level_diagnostics_by_cpc[
                cpc_level
            ]
        )

        all_matched_rows.append({
            "seed": seed,
            "cpc_level": cpc_level,
            "hierarchy_level": metrics[
                "hierarchy_level"
            ],
            "num_topics": metrics[
                "num_topics"
            ],
            "pur_p": metrics["pur_p"],
            "pur_a": metrics["pur_a"],
            "nmi": metrics["nmi"],
            "active_topics": diagnostics[
                "active_topics"
            ],
            "max_topic_share": diagnostics[
                "max_topic_share"
            ],
            "elapsed_seconds": (
                total_seed_elapsed
            ),
            "peak_gpu_gib": (
                peak_gpu_gib
            ),
        })

    print("\n" + "-" * 100)
    print(
        f"[SEED {seed} MATCHED CPC ALIGNMENT]"
    )

    for title, cpc_level in [
        ("Section ", "section"),
        ("Class   ", "class"),
        ("Subclass", "subclass"),
    ]:
        metrics = matched_metrics[
            cpc_level
        ]

        diagnostics = (
            level_diagnostics_by_cpc[
                cpc_level
            ]
        )

        print(
            f"{title} "
            f"[L{metrics['hierarchy_level']}, "
            f"K={metrics['num_topics']}] : "
            f"Pur_p={metrics['pur_p']:.4f} | "
            f"Pur_a={metrics['pur_a']:.4f} | "
            f"NMI={metrics['nmi']:.4f} | "
            f"active="
            f"{diagnostics['active_topics']}/"
            f"{metrics['num_topics']} | "
            f"max share="
            f"{diagnostics['max_topic_share']:.2%}"
        )

    print(
        f"Time={total_seed_elapsed / 3600:.2f}h | "
        f"GPU peak={peak_gpu_gib:.2f} GiB"
    )
    print("-" * 100)

    del model
    del optimizer
    del scaler
    del train_theta_list
    del test_theta_list
    del beta_list
    del phi_list
    del predicted_by_level

    gc.collect()
    torch.cuda.empty_cache()


# ======================================================================================
# 15. Aggregate results
# ======================================================================================

matched_seed_df = pd.DataFrame(
    all_matched_rows
)

if matched_seed_df.empty:
    raise RuntimeError(
        "집계할 seed 결과가 없습니다."
    )

matched_seed_df = (
    matched_seed_df
    .sort_values(
        ["seed", "hierarchy_level"]
    )
    .reset_index(drop=True)
)

matched_seed_df.to_csv(
    RESULT_DIR
    / "traco_matched_seed_level_results.csv",
    index=False,
)

cross_level_df = pd.DataFrame(
    all_cross_rows
)

if not cross_level_df.empty:
    cross_level_df = (
        cross_level_df
        .sort_values([
            "seed",
            "hierarchy_level",
            "cpc_level",
        ])
        .reset_index(drop=True)
    )

    cross_level_df.to_csv(
        RESULT_DIR
        / "traco_all_cross_level_results.csv",
        index=False,
    )

summary_rows = []

for cpc_level in [
    "section",
    "class",
    "subclass",
]:
    level_df = matched_seed_df[
        matched_seed_df["cpc_level"]
        == cpc_level
    ]

    row = {
        "cpc_level": cpc_level,
        "hierarchy_level": int(
            MATCHED_LEVEL_MAP[
                cpc_level
            ]
        ),
        "num_topics": int(
            NUM_TOPICS_LIST[
                MATCHED_LEVEL_MAP[
                    cpc_level
                ]
            ]
        ),
    }

    for metric in [
        "pur_p",
        "pur_a",
        "nmi",
    ]:
        values = level_df[
            metric
        ].to_numpy(dtype=float)

        row[f"{metric}_mean"] = float(
            values.mean()
        )

        row[f"{metric}_std"] = float(
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        )

        row[f"{metric}_min"] = float(
            values.min()
        )

        row[f"{metric}_max"] = float(
            values.max()
        )

    summary_rows.append(row)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    RESULT_DIR
    / "traco_3seed_matched_summary.csv",
    index=False,
)

complete_result = {
    "configuration": configuration,
    "seed_results": all_seed_results,
    "matched_summary": summary_rows,
    "total_elapsed_seconds": (
        time.time() - total_start
    ),
}

atomic_json_save(
    complete_result,
    RESULT_DIR
    / "traco_3seed_complete_results.json",
)


# ======================================================================================
# 16. LaTeX and final report
# ======================================================================================

def summary_value(
    cpc_level,
    metric,
    statistic="mean",
):
    row = summary_df[
        summary_df["cpc_level"]
        == cpc_level
    ].iloc[0]

    return float(
        row[
            f"{metric}_{statistic}"
        ]
    )


latex_mean = (
    "TraCo "
    + " & ".join(
        f"{summary_value(cpc_level, metric):.4f}"
        for cpc_level in [
            "section",
            "class",
            "subclass",
        ]
        for metric in [
            "pur_p",
            "pur_a",
            "nmi",
        ]
    )
    + r" \\"
)

latex_mean_std = (
    "TraCo "
    + " & ".join(
        f"${summary_value(cpc_level, metric):.4f}"
        f"\\pm"
        f"{summary_value(cpc_level, metric, 'std'):.4f}$"
        for cpc_level in [
            "section",
            "class",
            "subclass",
        ]
        for metric in [
            "pur_p",
            "pur_a",
            "nmi",
        ]
    )
    + r" \\"
)

with (
    RESULT_DIR
    / "latex_row_mean.txt"
).open(
    "w",
    encoding="utf-8",
) as handle:
    handle.write(
        latex_mean + "\n"
    )

with (
    RESULT_DIR
    / "latex_row_mean_std.txt"
).open(
    "w",
    encoding="utf-8",
) as handle:
    handle.write(
        latex_mean_std + "\n"
    )


print("\n" + "=" * 100)
print("TraCo 3-SEED TURBO FINAL RESULTS")
print("=" * 100)

for title, cpc_level in [
    ("SECTION", "section"),
    ("CLASS", "class"),
    ("SUBCLASS", "subclass"),
]:
    hierarchy_level = (
        MATCHED_LEVEL_MAP[
            cpc_level
        ]
    )

    print(
        f"\n[{title}] "
        f"Hierarchy L{hierarchy_level}, "
        f"K={NUM_TOPICS_LIST[hierarchy_level]}"
    )

    for metric_title, metric in [
        ("Pur_p", "pur_p"),
        ("Pur_a", "pur_a"),
        ("NMI", "nmi"),
    ]:
        mean_value = summary_value(
            cpc_level,
            metric,
        )

        std_value = summary_value(
            cpc_level,
            metric,
            "std",
        )

        print(
            f"{metric_title:<5} = "
            f"{mean_value:.4f} "
            f"± {std_value:.4f}"
        )

print("\n[LATEX ROW — MEAN]")
print(latex_mean)

print("\n[LATEX ROW — MEAN ± STD]")
print(latex_mean_std)

print("\n[PER-SEED RESULTS]")
display(
    matched_seed_df[
        [
            "seed",
            "cpc_level",
            "hierarchy_level",
            "num_topics",
            "pur_p",
            "pur_a",
            "nmi",
            "active_topics",
            "max_topic_share",
            "peak_gpu_gib",
        ]
    ]
)

print("\n[3-SEED SUMMARY]")
display(summary_df)

print("\n" + "=" * 100)
print("[COMPLETE]")
print(
    f"Total elapsed : "
    f"{(time.time() - total_start) / 3600:.2f} hours"
)
print(f"Batch size    : {BATCH_SIZE}")
print(
    f"Mixed precision: "
    f"{USE_MIXED_PRECISION}"
)
print(
    f"GPU BoW cache : "
    f"{TRAIN_BOW_GPU is not None}"
)
print(f"Result dir    : {RESULT_DIR}")
print(
    "Main JSON     : "
    f"{RESULT_DIR / 'traco_3seed_complete_results.json'}"
)
print(
    "Summary CSV   : "
    f"{RESULT_DIR / 'traco_3seed_matched_summary.csv'}"
)
print(
    "Seed CSV      : "
    f"{RESULT_DIR / 'traco_matched_seed_level_results.csv'}"
)
print(f"Arrays        : {ARRAY_DIR}")
print(f"Topics        : {TOPIC_DIR}")
print(f"Hierarchies   : {HIERARCHY_DIR}")
print(f"Checkpoints   : {CHECKPOINT_DIR}")
print("=" * 100)
