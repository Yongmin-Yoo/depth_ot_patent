# ============================================================
# ECRTM — Colab standalone runner
# Mount -> install -> load -> train seeds 43/44
# -> reuse seed 42 -> patent-level CPC evaluation -> save
# ============================================================

import os
import sys
import gc
import json
import time
import pickle
import random
import inspect
import traceback
import subprocess
import importlib
from pathlib import Path

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path("/content/drive/MyDrive/depth_ot_patent")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULT_DIR = PROJECT_ROOT / "results" / "cpc_alignment"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_BOW_PATH = PROCESSED_DIR / "bow_train.npz"
TEST_BOW_PATH = PROCESSED_DIR / "bow_test.npz"
VOCAB_PATH = PROCESSED_DIR / "vocab.pkl"

TEST_RECORDS_PATH = PROCESSED_DIR / "test_records.pkl"
REF_TEST_PATH = PROCESSED_DIR / "ref_test.pkl"

# CPC 정답 참조 파일
CPC_REFERENCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "cpc_alignment"
    / "etm_patent_predictions_by_seed.csv"
)

K = 30
EPOCHS = 200

# 반드시 기존 seed 42 실행과 동일해야 합니다.
BATCH_SIZE = 1024
LEARNING_RATE = 2e-3

# 권장:
# 첫 번째 Colab 세션: [43]
# 두 번째 Colab 세션: [44]
#
# 한 번에 모두 실행하려면 [43, 44]
TRAIN_SEEDS = [43, 44]

# 최종 평가는 저장된 seed 42 + 새 seed 43/44를 모두 사용
EVAL_SEEDS = [42, 43, 44]

# 저장된 theta가 있으면 해당 seed 학습을 건너뜁니다.
SKIP_COMPLETED_SEEDS = True

# 엄격한 재현성 설정
DETERMINISTIC = True

for directory in [CHECKPOINT_DIR, RESULT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

ERROR_LOG_PATH = LOG_DIR / "ecrtm_standalone_errors.txt"

# ------------------------------------------------------------
# 1. Google Drive mount
# ------------------------------------------------------------

from google.colab import drive

drive.mount("/content/drive", force_remount=False)

print("=" * 80)
print("ECRTM STANDALONE RUNNER")
print("=" * 80)
print(f"Project root    : {PROJECT_ROOT}")
print(f"Train seeds     : {TRAIN_SEEDS}")
print(f"Evaluation seeds: {EVAL_SEEDS}")
print(f"Topics          : {K}")
print(f"Epochs          : {EPOCHS}")
print(f"Batch size      : {BATCH_SIZE}")
print("=" * 80)

# ------------------------------------------------------------
# 2. Install/import required packages
# ------------------------------------------------------------

required_packages = {
    "topmost": "topmost",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
}

missing_packages = []

for import_name, pip_name in required_packages.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        missing_packages.append(pip_name)

if missing_packages:
    print("Installing:", missing_packages)

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing_packages,
        ]
    )

import numpy as np
import pandas as pd
import scipy
import torch
import topmost

from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score

print("\n=== ENVIRONMENT ===")
print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("NumPy  :", np.__version__)
print("SciPy  :", scipy.__version__)
print("TopMost:", getattr(topmost, "__version__", "unknown"))

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU를 찾지 못했습니다. "
        "Colab 런타임 유형에서 GPU를 활성화하세요."
    )

device = torch.device("cuda")

print("Device :", device)
print("GPU    :", torch.cuda.get_device_name(0))
print(
    "GPU RAM:",
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB",
)

# ------------------------------------------------------------
# 3. Reproducibility
# ------------------------------------------------------------

def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if DETERMINISTIC:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # seed 42와 동일한 설정이 우선입니다.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )


# ------------------------------------------------------------
# 4. Validate input files
# ------------------------------------------------------------

required_files = [
    TRAIN_BOW_PATH,
    TEST_BOW_PATH,
    VOCAB_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"필수 파일이 없습니다: {path}")

train_bow = sparse.load_npz(TRAIN_BOW_PATH).tocsr()
test_bow = sparse.load_npz(TEST_BOW_PATH).tocsr()

if train_bow.shape[1] != test_bow.shape[1]:
    raise ValueError(
        f"Train/test vocabulary dimension mismatch: "
        f"train={train_bow.shape}, test={test_bow.shape}"
    )

if not np.isfinite(train_bow.data).all():
    raise ValueError("Train BoW에 NaN 또는 Inf가 있습니다.")

if not np.isfinite(test_bow.data).all():
    raise ValueError("Test BoW에 NaN 또는 Inf가 있습니다.")

if train_bow.data.size and train_bow.data.min() < 0:
    raise ValueError("Train BoW에 음수가 있습니다.")

if test_bow.data.size and test_bow.data.min() < 0:
    raise ValueError("Test BoW에 음수가 있습니다.")

with open(VOCAB_PATH, "rb") as file:
    vocab_object = pickle.load(file)

if isinstance(vocab_object, dict):
    if all(isinstance(value, (int, np.integer))
           for value in vocab_object.values()):
        vocab = [
            token
            for token, index in sorted(
                vocab_object.items(),
                key=lambda item: int(item[1]),
            )
        ]
    else:
        vocab = list(vocab_object.keys())
else:
    vocab = list(vocab_object)

vocab = [str(token) for token in vocab]

if len(vocab) != train_bow.shape[1]:
    raise ValueError(
        f"Vocabulary length mismatch: "
        f"vocab={len(vocab)}, bow={train_bow.shape[1]}"
    )

print("\n=== DATA ===")
print(
    f"Train: {train_bow.shape}, "
    f"nnz={train_bow.nnz:,}"
)
print(
    f"Test : {test_bow.shape}, "
    f"nnz={test_bow.nnz:,}"
)
print(f"Vocab: {len(vocab):,}")

# ------------------------------------------------------------
# 5. Prepare a TopMost-compatible data directory
# ------------------------------------------------------------

TOPMOST_DATA_DIR = Path("/content/topmost_patent_data")
TOPMOST_DATA_DIR.mkdir(parents=True, exist_ok=True)

topmost_train_path = TOPMOST_DATA_DIR / "train_bow.npz"
topmost_test_path = TOPMOST_DATA_DIR / "test_bow.npz"
topmost_vocab_path = TOPMOST_DATA_DIR / "vocab.txt"

for source, destination in [
    (TRAIN_BOW_PATH, topmost_train_path),
    (TEST_BOW_PATH, topmost_test_path),
]:
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    os.symlink(source, destination)

with open(topmost_vocab_path, "w", encoding="utf-8") as file:
    for token in vocab:
        file.write(token.replace("\n", " ") + "\n")

# ------------------------------------------------------------
# 6. Construct TopMost BasicDataset
# ------------------------------------------------------------

def get_basic_dataset_class():
    if hasattr(topmost, "BasicDataset"):
        return topmost.BasicDataset

    if hasattr(topmost, "data") and hasattr(
        topmost.data,
        "BasicDataset",
    ):
        return topmost.data.BasicDataset

    try:
        from topmost.data import BasicDataset
        return BasicDataset
    except ImportError as error:
        raise ImportError(
            "설치된 TopMost에서 BasicDataset을 찾지 못했습니다."
        ) from error


def make_topmost_dataset():
    dataset_class = get_basic_dataset_class()
    signature = inspect.signature(dataset_class)

    kwargs = {}

    if "batch_size" in signature.parameters:
        kwargs["batch_size"] = BATCH_SIZE

    if "device" in signature.parameters:
        kwargs["device"] = device

    if "read_labels" in signature.parameters:
        kwargs["read_labels"] = False

    if "as_tensor" in signature.parameters:
        kwargs["as_tensor"] = True

    path_parameter = None

    for candidate in [
        "dataset_dir",
        "data_dir",
        "path",
        "root",
    ]:
        if candidate in signature.parameters:
            path_parameter = candidate
            break

    if path_parameter is not None:
        kwargs[path_parameter] = str(TOPMOST_DATA_DIR)
        dataset = dataset_class(**kwargs)
    else:
        dataset = dataset_class(
            str(TOPMOST_DATA_DIR),
            **kwargs,
        )

    return dataset


print("\nConstructing TopMost dataset...")
tm_dataset = make_topmost_dataset()

print("[PASS] TopMost dataset constructed.")
print(
    "TopMost vocab size:",
    getattr(tm_dataset, "vocab_size", "unknown"),
)

if int(tm_dataset.vocab_size) != len(vocab):
    raise ValueError(
        f"TopMost vocab mismatch: "
        f"{tm_dataset.vocab_size} != {len(vocab)}"
    )

# 원본 sparse matrix는 평가용 non-empty mask만 남기고 정리
test_nonempty_mask = np.asarray(
    test_bow.getnnz(axis=1) > 0
).reshape(-1)

number_of_test_claims = test_bow.shape[0]

del train_bow
gc.collect()

# ------------------------------------------------------------
# 7. Checkpoint helpers
# ------------------------------------------------------------

def theta_pkl_path(seed):
    return (
        CHECKPOINT_DIR
        / f"ecrtm_seed{seed}_theta_test.pkl"
    )


def theta_npy_path(seed):
    return (
        CHECKPOINT_DIR
        / f"ecrtm_seed{seed}_theta_test.npy"
    )


def topwords_path(seed):
    return (
        CHECKPOINT_DIR
        / f"ecrtm_seed{seed}_topwords.pkl"
    )


def model_path(seed):
    return (
        CHECKPOINT_DIR
        / f"ecrtm_seed{seed}_model.pt"
    )


def atomic_pickle_save(value, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "wb") as file:
        pickle.dump(
            value,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(temporary_path, path)


def atomic_numpy_save(value, path):
    path = Path(path)
    temporary_path = path.with_suffix(".tmp.npy")

    np.save(temporary_path, value)
    os.replace(temporary_path, path)


def load_saved_theta(seed):
    npy_path = theta_npy_path(seed)
    pkl_path = theta_pkl_path(seed)

    if npy_path.exists():
        theta = np.load(
            npy_path,
            mmap_mode=None,
        )

    elif pkl_path.exists():
        with open(pkl_path, "rb") as file:
            theta = pickle.load(file)

    else:
        raise FileNotFoundError(
            f"Seed {seed} theta가 없습니다."
        )

    if isinstance(theta, dict):
        for key in [
            "theta",
            "test_theta",
            "theta_test",
            "data",
        ]:
            if key in theta:
                theta = theta[key]
                break

    if torch.is_tensor(theta):
        theta = theta.detach().cpu().numpy()

    theta = np.asarray(theta, dtype=np.float32)

    if theta.ndim != 2:
        raise ValueError(
            f"Seed {seed}: theta가 2차원이 아닙니다: "
            f"{theta.shape}"
        )

    if theta.shape != (number_of_test_claims, K):
        raise ValueError(
            f"Seed {seed}: theta shape mismatch: "
            f"expected={(number_of_test_claims, K)}, "
            f"actual={theta.shape}"
        )

    if not np.isfinite(theta).all():
        raise ValueError(
            f"Seed {seed}: theta에 NaN/Inf가 있습니다."
        )

    if theta.min() < -1e-7:
        raise ValueError(
            f"Seed {seed}: theta에 음수가 있습니다."
        )

    theta = np.maximum(theta, 0.0)

    row_sums = theta.sum(axis=1, keepdims=True)
    valid_rows = row_sums[:, 0] > 0

    theta[valid_rows] /= row_sums[valid_rows]

    if (~valid_rows).any():
        theta[~valid_rows] = 1.0 / K

    return theta


# ------------------------------------------------------------
# 8. Locate TopMost model/trainer classes
# ------------------------------------------------------------

def get_ecrtm_class():
    if hasattr(topmost, "ECRTM"):
        return topmost.ECRTM

    if (
        hasattr(topmost, "models")
        and hasattr(topmost.models, "ECRTM")
    ):
        return topmost.models.ECRTM

    raise ImportError(
        "설치된 TopMost에서 ECRTM을 찾지 못했습니다."
    )


def get_basic_trainer_class():
    if hasattr(topmost, "BasicTrainer"):
        return topmost.BasicTrainer

    if (
        hasattr(topmost, "trainers")
        and hasattr(topmost.trainers, "BasicTrainer")
    ):
        return topmost.trainers.BasicTrainer

    raise ImportError(
        "설치된 TopMost에서 BasicTrainer를 찾지 못했습니다."
    )


ECRTMClass = get_ecrtm_class()
BasicTrainerClass = get_basic_trainer_class()

# ------------------------------------------------------------
# 9. Train only unfinished seeds
# ------------------------------------------------------------

training_records = []
failed_seeds = []

for seed in TRAIN_SEEDS:
    print("\n" + "=" * 80)
    print(f"ECRTM SEED {seed}")
    print("=" * 80)

    if (
        SKIP_COMPLETED_SEEDS
        and (
            theta_npy_path(seed).exists()
            or theta_pkl_path(seed).exists()
        )
    ):
        print(
            f"[SKIP] Seed {seed} theta가 이미 존재합니다."
        )

        # 저장 파일이 실제로 정상인지 검증
        theta_check = load_saved_theta(seed)

        print(
            f"[PASS] Existing theta: "
            f"shape={theta_check.shape}"
        )

        del theta_check
        continue

    set_all_seeds(seed)

    model = None
    trainer = None

    try:
        torch.cuda.empty_cache()

        model = ECRTMClass(
            tm_dataset.vocab_size,
            num_topics=K,
        ).to(device)

        trainer = BasicTrainerClass(
            model,
            tm_dataset,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            verbose=True,
        )

        started_at = time.time()

        top_words, _ = trainer.train()

        training_seconds = time.time() - started_at

        print(
            f"\nSeed {seed} training took "
            f"{training_seconds / 3600:.2f} hours."
        )

        print("Inferring test theta...")
        test_theta = trainer.test(
            tm_dataset.test_data
        )

        if torch.is_tensor(test_theta):
            test_theta = (
                test_theta
                .detach()
                .cpu()
                .numpy()
            )

        test_theta = np.asarray(
            test_theta,
            dtype=np.float32,
        )

        if test_theta.shape != (
            number_of_test_claims,
            K,
        ):
            raise ValueError(
                f"Seed {seed}: test theta shape mismatch: "
                f"{test_theta.shape}"
            )

        if not np.isfinite(test_theta).all():
            raise FloatingPointError(
                f"Seed {seed}: theta에 NaN/Inf가 있습니다."
            )

        test_theta = np.maximum(
            test_theta,
            0.0,
        )

        theta_sums = test_theta.sum(
            axis=1,
            keepdims=True,
        )

        valid_theta_rows = theta_sums[:, 0] > 0

        test_theta[valid_theta_rows] /= (
            theta_sums[valid_theta_rows]
        )

        if (~valid_theta_rows).any():
            test_theta[~valid_theta_rows] = 1.0 / K

        atomic_numpy_save(
            test_theta,
            theta_npy_path(seed),
        )

        atomic_pickle_save(
            test_theta,
            theta_pkl_path(seed),
        )

        atomic_pickle_save(
            top_words,
            topwords_path(seed),
        )

        torch.save(
            {
                "seed": seed,
                "num_topics": K,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "model_state_dict": model.state_dict(),
            },
            model_path(seed),
        )

        training_record = {
            "seed": seed,
            "status": "completed",
            "training_seconds": training_seconds,
            "theta_shape": list(test_theta.shape),
            "theta_min": float(test_theta.min()),
            "theta_max": float(test_theta.max()),
            "mean_row_sum": float(
                test_theta.sum(axis=1).mean()
            ),
        }

        training_records.append(training_record)

        print(f"[SAVED] {theta_npy_path(seed)}")
        print(f"[SAVED] {theta_pkl_path(seed)}")
        print(f"[SAVED] {topwords_path(seed)}")
        print(f"[SAVED] {model_path(seed)}")

    except KeyboardInterrupt:
        print(
            f"\n[INTERRUPTED] Seed {seed} 학습이 중단되었습니다."
        )
        print(
            "TopMost BasicTrainer는 epoch 중간 resume를 "
            "지원하지 않으므로 이 seed는 다음 실행에서 "
            "처음부터 다시 시작됩니다."
        )
        raise

    except Exception as error:
        failed_seeds.append(seed)

        error_message = (
            f"Seed {seed} failed: {error}\n"
            f"{traceback.format_exc()}\n"
        )

        print(error_message)

        with open(
            ERROR_LOG_PATH,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(error_message + "\n")

    finally:
        if trainer is not None:
            del trainer

        if model is not None:
            del model

        gc.collect()
        torch.cuda.empty_cache()

# ------------------------------------------------------------
# 10. Patent/claim mapping
# ------------------------------------------------------------

def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def extract_record_patent_ids(records):
    candidate_columns = [
        "patent_id",
        "publication_number",
        "publication_id",
        "doc_id",
        "document_id",
        "id",
    ]

    if isinstance(records, pd.DataFrame):
        for column in candidate_columns:
            if column in records.columns:
                return records[column].astype(str).to_numpy()

    if isinstance(records, dict):
        for column in candidate_columns:
            if column in records:
                values = np.asarray(records[column])

                if values.ndim == 1:
                    return values.astype(str)

        for key in ["records", "data", "items"]:
            if key in records:
                return extract_record_patent_ids(
                    records[key]
                )

    if isinstance(records, (list, tuple, np.ndarray)):
        records_list = list(records)

        if not records_list:
            raise ValueError("test_records가 비어 있습니다.")

        first = records_list[0]

        if isinstance(first, dict):
            for column in candidate_columns:
                if column in first:
                    return np.asarray(
                        [
                            str(record[column])
                            for record in records_list
                        ]
                    )

        if isinstance(first, (str, int, np.integer)):
            return np.asarray(
                records_list,
                dtype=str,
            )

    raise ValueError(
        "test_records에서 patent ID를 찾지 못했습니다."
    )


def collect_reference_candidates(obj, expected_length):
    candidates = []

    if isinstance(obj, pd.DataFrame):
        for column in obj.columns:
            values = obj[column].to_numpy()

            if len(values) == expected_length:
                candidates.append(
                    (f"column:{column}", values)
                )

    elif isinstance(obj, dict):
        priority_keys = [
            "patent_idx",
            "patent_index",
            "record_idx",
            "record_index",
            "claim_to_patent",
            "claim_patent_idx",
            "references",
            "ref",
            "indices",
            "patent_id",
            "patent_ids",
        ]

        for key in priority_keys:
            if key in obj:
                values = np.asarray(obj[key])

                if (
                    values.ndim == 1
                    and len(values) == expected_length
                ):
                    candidates.append(
                        (f"key:{key}", values)
                    )

        for key, value in obj.items():
            values = np.asarray(value)

            if (
                values.ndim == 1
                and len(values) == expected_length
            ):
                candidates.append(
                    (f"key:{key}", values)
                )

    else:
        values = np.asarray(obj, dtype=object)

        if (
            values.ndim == 1
            and len(values) == expected_length
        ):
            candidates.append(("array", values))

        elif (
            values.ndim == 2
            and values.shape[0] == expected_length
        ):
            for column_index in range(values.shape[1]):
                candidates.append(
                    (
                        f"array-column:{column_index}",
                        values[:, column_index],
                    )
                )

    return candidates


def create_claim_patent_ids(
    ref_test,
    record_patent_ids,
    expected_length,
):
    candidates = collect_reference_candidates(
        ref_test,
        expected_length,
    )

    if not candidates:
        raise ValueError(
            "ref_test에서 claim-to-patent mapping을 "
            "찾지 못했습니다."
        )

    number_of_patents = len(record_patent_ids)
    patent_id_set = set(
        record_patent_ids.astype(str)
    )

    for candidate_name, candidate in candidates:
        candidate_array = np.asarray(candidate)

        # 정수 index mapping 확인
        try:
            numeric = candidate_array.astype(np.int64)

            if (
                numeric.min() >= 0
                and numeric.max() < number_of_patents
            ):
                print(
                    f"Reference mapping: "
                    f"{candidate_name}, zero-based"
                )

                return (
                    record_patent_ids[numeric],
                    "zero-based",
                )

            if (
                numeric.min() >= 1
                and numeric.max() <= number_of_patents
            ):
                print(
                    f"Reference mapping: "
                    f"{candidate_name}, one-based"
                )

                return (
                    record_patent_ids[numeric - 1],
                    "one-based",
                )

        except (ValueError, TypeError, OverflowError):
            pass

        # 직접 patent ID mapping 확인
        candidate_strings = candidate_array.astype(str)

        matched_fraction = np.mean(
            [
                value in patent_id_set
                for value in candidate_strings
            ]
        )

        if matched_fraction > 0.95:
            print(
                f"Reference mapping: "
                f"{candidate_name}, direct patent ID"
            )

            return (
                candidate_strings,
                "direct-patent-id",
            )

    raise ValueError(
        "ref_test mapping 후보를 patent record와 "
        "연결하지 못했습니다."
    )


if not TEST_RECORDS_PATH.exists():
    raise FileNotFoundError(
        f"test_records 파일이 없습니다: "
        f"{TEST_RECORDS_PATH}"
    )

if not REF_TEST_PATH.exists():
    raise FileNotFoundError(
        f"ref_test 파일이 없습니다: {REF_TEST_PATH}"
    )

test_records = load_pickle(TEST_RECORDS_PATH)
ref_test = load_pickle(REF_TEST_PATH)

record_patent_ids = extract_record_patent_ids(
    test_records
)

claim_patent_ids, reference_type = (
    create_claim_patent_ids(
        ref_test,
        record_patent_ids,
        number_of_test_claims,
    )
)

claim_patent_ids = np.asarray(
    claim_patent_ids,
    dtype=str,
)

if len(claim_patent_ids) != number_of_test_claims:
    raise ValueError(
        f"Claim mapping length mismatch: "
        f"{len(claim_patent_ids)} != "
        f"{number_of_test_claims}"
    )

print("\n=== CLAIM/PATENT MAPPING ===")
print(f"Patent records : {len(record_patent_ids):,}")
print(f"Test claims    : {len(claim_patent_ids):,}")
print(
    f"Unique patents: "
    f"{len(np.unique(claim_patent_ids)):,}"
)
print(f"Reference type : {reference_type}")
print(
    f"Empty claims   : "
    f"{np.sum(~test_nonempty_mask):,}"
)

# ------------------------------------------------------------
# 11. Load CPC reference labels
# ------------------------------------------------------------

def normalize_cpc_reference(frame):
    frame = frame.copy()

    lowercase_mapping = {
        str(column).lower(): column
        for column in frame.columns
    }

    aliases = {
        "patent_id": [
            "patent_id",
            "publication_number",
            "publication_id",
            "doc_id",
            "document_id",
        ],
        "section": [
            "section",
            "cpc_section",
        ],
        "class": [
            "class",
            "cpc_class",
        ],
        "subclass": [
            "subclass",
            "cpc_subclass",
        ],
    }

    rename_mapping = {}

    for target, candidates in aliases.items():
        found_column = None

        for candidate in candidates:
            if candidate in lowercase_mapping:
                found_column = lowercase_mapping[candidate]
                break

        if found_column is None:
            raise ValueError(
                f"CPC reference에 {target} 열이 없습니다. "
                f"Columns={list(frame.columns)}"
            )

        rename_mapping[found_column] = target

    frame = frame.rename(
        columns=rename_mapping
    )

    frame = frame[
        [
            "patent_id",
            "section",
            "class",
            "subclass",
        ]
    ].copy()

    frame["patent_id"] = (
        frame["patent_id"].astype(str)
    )

    for column in [
        "section",
        "class",
        "subclass",
    ]:
        frame[column] = frame[column].astype(str)

    # seed별 예측 파일이면 같은 patent가 여러 번 있을 수 있음
    frame = frame.drop_duplicates(
        subset=[
            "patent_id",
            "section",
            "class",
            "subclass",
        ]
    )

    # 한 patent에 동일한 정답 한 행만 유지
    conflicting = (
        frame.groupby("patent_id")[
            ["section", "class", "subclass"]
        ]
        .nunique()
        .max(axis=1)
    )

    if (conflicting > 1).any():
        raise ValueError(
            "동일 patent_id에 서로 다른 CPC 정답이 있습니다."
        )

    frame = frame.drop_duplicates(
        subset=["patent_id"]
    )

    return frame


if not CPC_REFERENCE_PATH.exists():
    raise FileNotFoundError(
        "CPC reference 파일을 찾지 못했습니다:\n"
        f"{CPC_REFERENCE_PATH}\n"
        "경로가 다르면 코드 상단의 "
        "CPC_REFERENCE_PATH를 수정하세요."
    )

cpc_reference = normalize_cpc_reference(
    pd.read_csv(CPC_REFERENCE_PATH)
)

print("\n=== CPC REFERENCE ===")
print(f"Path   : {CPC_REFERENCE_PATH}")
print(f"Patents: {len(cpc_reference):,}")
print(
    "Sections:",
    cpc_reference["section"].nunique(),
)
print(
    "Classes:",
    cpc_reference["class"].nunique(),
)
print(
    "Subclasses:",
    cpc_reference["subclass"].nunique(),
)

# ------------------------------------------------------------
# 12. Patent-level aggregation
# ------------------------------------------------------------

def aggregate_claim_theta_to_patents(
    claim_theta,
    claim_to_patent_ids,
    nonempty_mask,
):
    claim_theta = np.asarray(
        claim_theta,
        dtype=np.float32,
    )

    valid_mask = (
        np.asarray(nonempty_mask, dtype=bool)
        & np.isfinite(claim_theta).all(axis=1)
        & (claim_theta.sum(axis=1) > 0)
    )

    valid_theta = claim_theta[valid_mask]
    valid_patent_ids = np.asarray(
        claim_to_patent_ids,
        dtype=str,
    )[valid_mask]

    patent_ids, inverse_indices = np.unique(
        valid_patent_ids,
        return_inverse=True,
    )

    patent_theta = np.zeros(
        (len(patent_ids), claim_theta.shape[1]),
        dtype=np.float64,
    )

    counts = np.zeros(
        len(patent_ids),
        dtype=np.int64,
    )

    np.add.at(
        patent_theta,
        inverse_indices,
        valid_theta,
    )

    np.add.at(
        counts,
        inverse_indices,
        1,
    )

    patent_theta /= np.maximum(
        counts[:, None],
        1,
    )

    row_sums = patent_theta.sum(
        axis=1,
        keepdims=True,
    )

    valid_rows = row_sums[:, 0] > 0
    patent_theta[valid_rows] /= row_sums[valid_rows]

    return (
        patent_ids,
        patent_theta.astype(np.float32),
        counts,
    )


# ------------------------------------------------------------
# 13. Purity/NMI metrics
# ------------------------------------------------------------

def calculate_metrics(
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

    contingency = pd.crosstab(
        pd.Series(
            predicted_topics,
            name="predicted_topic",
        ),
        pd.Series(
            true_labels,
            name="true_label",
        ),
    )

    number_of_samples = contingency.to_numpy().sum()

    if number_of_samples == 0:
        raise ValueError(
            "평가할 sample이 없습니다."
        )

    # Predicted-cluster purity
    pur_p = (
        contingency.max(axis=1).sum()
        / number_of_samples
    )

    # Inverse label-wise purity
    pur_a = (
        contingency.max(axis=0).sum()
        / number_of_samples
    )

    nmi = normalized_mutual_info_score(
        true_labels,
        predicted_topics,
        average_method="arithmetic",
    )

    return {
        "pur_p": float(pur_p),
        "pur_a": float(pur_a),
        "nmi": float(nmi),
        "n_samples": int(number_of_samples),
        "n_predicted_topics": int(
            np.unique(predicted_topics).size
        ),
        "n_true_labels": int(
            np.unique(true_labels).size
        ),
    }


# ------------------------------------------------------------
# 14. Seed-wise evaluation
# ------------------------------------------------------------

metric_rows = []
prediction_frames = []
evaluation_failures = []

for seed in EVAL_SEEDS:
    print("\n" + "=" * 80)
    print(f"EVALUATING ECRTM SEED {seed}")
    print("=" * 80)

    try:
        claim_theta = load_saved_theta(seed)

        (
            patent_ids,
            patent_theta,
            patent_claim_counts,
        ) = aggregate_claim_theta_to_patents(
            claim_theta,
            claim_patent_ids,
            test_nonempty_mask,
        )

        predicted_topics = np.argmax(
            patent_theta,
            axis=1,
        )

        topic_probabilities = np.max(
            patent_theta,
            axis=1,
        )

        predictions = pd.DataFrame(
            {
                "patent_id": patent_ids.astype(str),
                "seed": seed,
                "predicted_topic": predicted_topics,
                "topic_probability": topic_probabilities,
                "number_of_claims": patent_claim_counts,
            }
        )

        merged = predictions.merge(
            cpc_reference,
            on="patent_id",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) == 0:
            raise ValueError(
                "CPC reference와 patent predictions의 "
                "공통 patent_id가 없습니다."
            )

        coverage = len(merged) / len(cpc_reference)

        print(
            f"Matched patents: "
            f"{len(merged):,}/{len(cpc_reference):,} "
            f"({coverage:.2%})"
        )

        if coverage < 0.95:
            raise ValueError(
                f"CPC patent coverage가 너무 낮습니다: "
                f"{coverage:.2%}"
            )

        for level in [
            "section",
            "class",
            "subclass",
        ]:
            metrics = calculate_metrics(
                merged[level].to_numpy(),
                merged["predicted_topic"].to_numpy(),
            )

            metric_row = {
                "model": "ECRTM",
                "seed": seed,
                "level": level,
                **metrics,
            }

            metric_rows.append(metric_row)

            print(
                f"{level:8s} | "
                f"Pur_p={metrics['pur_p']:.4f} | "
                f"Pur_a={metrics['pur_a']:.4f} | "
                f"NMI={metrics['nmi']:.4f} | "
                f"N={metrics['n_samples']:,}"
            )

        prediction_frames.append(merged)

        del claim_theta
        del patent_theta
        gc.collect()

    except Exception as error:
        evaluation_failures.append(seed)

        error_message = (
            f"Evaluation seed {seed} failed: {error}\n"
            f"{traceback.format_exc()}\n"
        )

        print(error_message)

        with open(
            ERROR_LOG_PATH,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(error_message + "\n")

if evaluation_failures:
    raise RuntimeError(
        f"평가 실패 seed: {evaluation_failures}. "
        f"로그를 확인하세요: {ERROR_LOG_PATH}"
    )

metrics_df = pd.DataFrame(metric_rows)

if metrics_df.empty:
    raise RuntimeError(
        "평가 지표가 생성되지 않았습니다."
    )

# ------------------------------------------------------------
# 15. Aggregate scalar metrics across seeds
# ------------------------------------------------------------

summary_rows = []

for level in [
    "section",
    "class",
    "subclass",
]:
    level_frame = metrics_df[
        metrics_df["level"] == level
    ]

    for metric in [
        "pur_p",
        "pur_a",
        "nmi",
    ]:
        values = level_frame[metric].to_numpy(
            dtype=float
        )

        summary_rows.append(
            {
                "model": "ECRTM",
                "level": level,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
                "number_of_seeds": int(len(values)),
                "seeds": ",".join(
                    map(
                        str,
                        level_frame["seed"].tolist(),
                    )
                ),
            }
        )

summary_df = pd.DataFrame(summary_rows)

by_seed_output_path = (
    RESULT_DIR
    / "ecrtm_patent_cpc_alignment_by_seed.csv"
)

summary_output_path = (
    RESULT_DIR
    / "ecrtm_patent_cpc_alignment_summary.csv"
)

summary_json_path = (
    RESULT_DIR
    / "ecrtm_summary.json"
)

prediction_output_path = (
    RESULT_DIR
    / "ecrtm_patent_predictions_by_seed.csv"
)

metrics_df.to_csv(
    by_seed_output_path,
    index=False,
)

summary_df.to_csv(
    summary_output_path,
    index=False,
)

if prediction_frames:
    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_csv(
        prediction_output_path,
        index=False,
    )

summary_json = {
    "model": "ECRTM",
    "topics": K,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "evaluation_seeds": EVAL_SEEDS,
    "training_records": training_records,
    "failed_training_seeds": failed_seeds,
    "failed_evaluation_seeds": evaluation_failures,
    "metrics_by_seed": metrics_df.to_dict(
        orient="records"
    ),
    "metric_summary": summary_df.to_dict(
        orient="records"
    ),
    "paths": {
        "by_seed_csv": str(by_seed_output_path),
        "summary_csv": str(summary_output_path),
        "predictions_csv": str(
            prediction_output_path
        ),
    },
}

with open(
    summary_json_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary_json,
        file,
        ensure_ascii=False,
        indent=2,
    )

# ------------------------------------------------------------
# 16. Final report
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ECRTM FINAL RESULTS — MEAN ± STD")
print("=" * 80)

for level in [
    "section",
    "class",
    "subclass",
]:
    print(f"\n[{level.upper()}]")

    level_summary = summary_df[
        summary_df["level"] == level
    ]

    for metric in [
        "pur_p",
        "pur_a",
        "nmi",
    ]:
        row = level_summary[
            level_summary["metric"] == metric
        ].iloc[0]

        print(
            f"{metric:5s}: "
            f"{row['mean']:.4f} "
            f"± {row['std']:.4f}"
        )

print("\n=== SAVED FILES ===")
print("Per-seed metrics :", by_seed_output_path)
print("Summary metrics  :", summary_output_path)
print("Summary JSON     :", summary_json_path)
print("Predictions      :", prediction_output_path)
print("Error log        :", ERROR_LOG_PATH)

print("\n=== LATEX ROW — MEAN VALUES ONLY ===")

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
        value = summary_df.loc[
            (
                summary_df["level"] == level
            )
            & (
                summary_df["metric"] == metric
            ),
            "mean",
        ].iloc[0]

        latex_values.append(f"{value:.4f}")

print(
    "ECRTM & "
    + " & ".join(latex_values)
    + r" \\"
)

print("\n[DONE] ECRTM training and evaluation completed.")
