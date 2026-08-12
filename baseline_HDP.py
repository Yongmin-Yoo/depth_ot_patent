# ============================================================
# Standalone Gensim HDP Pipeline for Google Colab
# - Google Drive mount
# - Package installation
# - Train/test BoW and vocabulary loading
# - Fixed training sample across seeds
# - HDP training for seeds 42, 43, 44
# - Batch-wise full theta inference
# - Per-seed evaluation: Pur_p, Pur_a, NMI
# - Model/theta/metrics saving
# - No invalid raw-theta averaging across seeds
# ============================================================

# ============================================================
# 0. Google Drive 마운트
# ============================================================
from google.colab import drive
drive.mount("/content/drive", force_remount=False)


# ============================================================
# 1. 패키지 설치
# ============================================================
import sys
import subprocess
import importlib.util

required_packages = {
    "gensim": "gensim",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "pandas": "pandas",
    "tqdm": "tqdm",
}

missing_packages = [
    pip_name
    for module_name, pip_name in required_packages.items()
    if importlib.util.find_spec(module_name) is None
]

if missing_packages:
    print("Installing missing packages:", missing_packages)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )
else:
    print("[PASS] Required packages are already installed.")


# ============================================================
# 2. Imports
# ============================================================
import os
import gc
import json
import time
import pickle
import random
import logging
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

import gensim
from gensim.models import HdpModel
from gensim.matutils import Sparse2Corpus

from sklearn.metrics import normalized_mutual_info_score
from tqdm.auto import tqdm


# ============================================================
# 3. 사용자 설정
# ============================================================

# 프로젝트 루트
PROJECT_ROOT = Path("/content/drive/MyDrive/depth_ot_patent")

# ------------------------------------------------------------
# 데이터 파일 경로
# ------------------------------------------------------------
# 정확한 경로를 알고 있다면 문자열로 지정하세요.
# None이면 PROJECT_ROOT 안에서 알려진 파일명을 자동으로 찾습니다.
#
# 예:
# TRAIN_BOW_PATH = PROJECT_ROOT / "data/bow_train.npz"
# TEST_BOW_PATH  = PROJECT_ROOT / "data/bow_test.npz"
# VOCAB_PATH     = PROJECT_ROOT / "data/vocab.pkl"
# TEST_LABEL_PATH = PROJECT_ROOT / "data/test_labels.csv"

TRAIN_BOW_PATH = None
TEST_BOW_PATH = None
VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "vocab.pkl"

# CPC 평가를 하지 않으려면 None으로 두세요.
# 자동 탐색하지 않고, 명시적으로 지정하는 것을 권장합니다.
TEST_LABEL_PATH = None

# CSV에 여러 CPC 계층 열이 있다면 여기에 지정합니다.
# None이면 로드된 모든 열을 평가합니다.
#
# 예:
# LABEL_COLUMNS = ["section", "class", "subclass"]
LABEL_COLUMNS = None


# ------------------------------------------------------------
# HDP 설정
# ------------------------------------------------------------
SEEDS = [42, 43, 44]

# 모든 seed가 동일한 5만 개 문서를 사용
HDP_SAMPLE_SEED = 42
HDP_TRAIN_SAMPLE_SIZE = 50_000

# HDP truncation 설정
HDP_T = 150
HDP_K = 15

# 온라인 HDP 설정
HDP_CHUNKSIZE = 1000
HDP_KAPPA = 0.9
HDP_TAU = 64.0
HDP_ALPHA = 1.0
HDP_GAMMA = 1.0
HDP_ETA = 0.01

# 테스트 theta 추론 배치
HDP_INFERENCE_BATCH_SIZE = 2000

# 기존 학습 모델이 있으면 재사용
REUSE_EXISTING_MODEL = True

# theta가 이미 있어도 다시 추론할지 여부
FORCE_REINFERENCE = False

# 모델 저장 여부
SAVE_HDP_MODEL = True

# 로그 수준
#logging.getLogger("gensim").setLevel(logging.WARNING)
logging.getLogger("gensim.models.hdpmodel").setLevel(logging.ERROR)


# ============================================================
# 4. 결과 폴더 생성
# ============================================================
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"hdp_standalone_{RUN_TIMESTAMP}"

HDP_ROOT = PROJECT_ROOT / "topic_model" / "hdp"
MODEL_DIR = HDP_ROOT / "models"
THETA_DIR = HDP_ROOT / "theta"
RESULT_DIR = HDP_ROOT / "results"
LOG_DIR = HDP_ROOT / "logs"
SAMPLE_DIR = HDP_ROOT / "samples"

for directory in [
    HDP_ROOT,
    MODEL_DIR,
    THETA_DIR,
    RESULT_DIR,
    LOG_DIR,
    SAMPLE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

RUN_LOG_PATH = LOG_DIR / f"{RUN_NAME}.json"
METRICS_CSV_PATH = RESULT_DIR / f"{RUN_NAME}_metrics.csv"
SUMMARY_JSON_PATH = RESULT_DIR / f"{RUN_NAME}_summary.json"
SAMPLE_IDX_PATH = SAMPLE_DIR / (
    f"hdp_train_sample_idx_n{HDP_TRAIN_SAMPLE_SIZE}"
    f"_seed{HDP_SAMPLE_SEED}.npy"
)

print("\n=== Output directories ===")
print("Project root :", PROJECT_ROOT)
print("HDP root     :", HDP_ROOT)
print("Model dir    :", MODEL_DIR)
print("Theta dir    :", THETA_DIR)
print("Result dir   :", RESULT_DIR)
print("Log dir      :", LOG_DIR)


# ============================================================
# 5. 파일 탐색 유틸리티
# ============================================================
def resolve_data_path(explicit_path, candidate_names, description, required=True):
    """
    명시적 경로가 있으면 그것을 사용하고,
    없으면 PROJECT_ROOT 아래에서 후보 파일명을 탐색합니다.
    """
    if explicit_path is not None:
        path = Path(explicit_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"{description} 파일이 없습니다:\n{path}"
            )

        return path

    matches = []

    for candidate_name in candidate_names:
        matches.extend(PROJECT_ROOT.rglob(candidate_name))

    # 중복 제거
    unique_matches = sorted(
        set(path.resolve() for path in matches),
        key=lambda x: str(x)
    )

    if len(unique_matches) == 1:
        print(f"[AUTO] {description}: {unique_matches[0]}")
        return Path(unique_matches[0])

    if len(unique_matches) == 0:
        if required:
            raise FileNotFoundError(
                f"\n{description} 파일을 자동으로 찾지 못했습니다.\n"
                f"PROJECT_ROOT: {PROJECT_ROOT}\n"
                f"찾은 파일명 후보: {candidate_names}\n\n"
                f"코드 상단에서 경로를 직접 지정하세요."
            )

        return None

    print(f"\n[WARNING] {description} 후보가 여러 개 발견되었습니다.")

    for index, path in enumerate(unique_matches, start=1):
        print(f"  {index}. {path}")

    raise RuntimeError(
        f"\n{description} 후보가 여러 개입니다. "
        f"코드 상단에서 정확한 경로를 직접 지정하세요."
    )


TRAIN_BOW_PATH = resolve_data_path(
    TRAIN_BOW_PATH,
    candidate_names=[
        "bow_train.npz",
        "train_bow.npz",
        "bow_train.pkl",
        "train_bow.pkl",
        "bow_train.pickle",
        "train_bow.pickle",
    ],
    description="Train BoW",
    required=True,
)

TEST_BOW_PATH = resolve_data_path(
    TEST_BOW_PATH,
    candidate_names=[
        "bow_test.npz",
        "test_bow.npz",
        "bow_test.pkl",
        "test_bow.pkl",
        "bow_test.pickle",
        "test_bow.pickle",
    ],
    description="Test BoW",
    required=True,
)

VOCAB_PATH = resolve_data_path(
    VOCAB_PATH,
    candidate_names=[
        "vocab.pkl",
        "vocabulary.pkl",
        "vocab.pickle",
        "vocabulary.pickle",
        "vocab.npy",
        "vocabulary.npy",
        "vocab.json",
        "vocabulary.json",
        "vocab.txt",
        "vocabulary.txt",
    ],
    description="Vocabulary",
    required=True,
)

if TEST_LABEL_PATH is not None:
    TEST_LABEL_PATH = resolve_data_path(
        TEST_LABEL_PATH,
        candidate_names=[],
        description="Test CPC labels",
        required=True,
    )


# ============================================================
# 6. 데이터 로드 유틸리티
# ============================================================
def unwrap_sparse_object(obj, preferred_keys=None):
    """
    pickle 내부가 sparse matrix 자체이거나,
    dictionary 형태인 경우 sparse matrix를 추출합니다.
    """
    if sp.issparse(obj):
        return obj

    if preferred_keys is None:
        preferred_keys = [
            "bow",
            "matrix",
            "data",
            "x",
            "X",
            "train_bow",
            "test_bow",
            "bow_train",
            "bow_test",
        ]

    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj and sp.issparse(obj[key]):
                return obj[key]

        sparse_values = [
            value for value in obj.values()
            if sp.issparse(value)
        ]

        if len(sparse_values) == 1:
            return sparse_values[0]

    raise TypeError(
        "파일에서 scipy sparse matrix를 추출하지 못했습니다. "
        f"로드된 객체 타입: {type(obj)}"
    )


def load_sparse_matrix(path):
    path = Path(path)
    suffix = path.suffix.lower()

    print(f"Loading sparse matrix: {path}")

    if suffix == ".npz":
        matrix = sp.load_npz(path)

    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            obj = pickle.load(file)

        matrix = unwrap_sparse_object(obj)

    else:
        raise ValueError(
            f"지원하지 않는 sparse matrix 형식입니다: {suffix}"
        )

    matrix = matrix.tocsr()

    if matrix.ndim != 2:
        raise ValueError(
            f"BoW matrix는 2차원이어야 합니다: shape={matrix.shape}"
        )

    if matrix.data.size > 0:
        if not np.isfinite(matrix.data).all():
            raise FloatingPointError(
                f"{path.name}에 NaN 또는 Inf가 있습니다."
            )

        if np.any(matrix.data < 0):
            raise ValueError(
                f"{path.name}에 음수 BoW 값이 있습니다."
            )

    matrix.sort_indices()

    return matrix


def normalize_vocab_object(obj):
    """
    다양한 vocab 저장 형식을 id 순서의 문자열 리스트로 변환합니다.
    """
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, pd.Series):
        obj = obj.tolist()

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(
                "Vocabulary DataFrame에는 열이 하나만 있어야 합니다."
            )

        obj = obj.iloc[:, 0].tolist()

    if isinstance(obj, (list, tuple)):
        return [str(word) for word in obj]

    if isinstance(obj, dict):
        # {integer_id: word}
        if all(isinstance(key, (int, np.integer)) for key in obj.keys()):
            ordered_keys = sorted(obj.keys())
            return [str(obj[key]) for key in ordered_keys]

        # {word: integer_id}
        if all(isinstance(value, (int, np.integer)) for value in obj.values()):
            sorted_items = sorted(
                obj.items(),
                key=lambda item: int(item[1])
            )
            return [str(word) for word, _ in sorted_items]

        # {"vocab": [...]}, {"words": [...]}
        for key in ["vocab", "vocabulary", "words", "tokens", "id2word"]:
            if key in obj:
                return normalize_vocab_object(obj[key])

    # Gensim Dictionary 형태
    if hasattr(obj, "id2token") and obj.id2token:
        return [
            str(obj.id2token[index])
            for index in range(len(obj.id2token))
        ]

    if hasattr(obj, "token2id") and obj.token2id:
        sorted_items = sorted(
            obj.token2id.items(),
            key=lambda item: int(item[1])
        )
        return [str(word) for word, _ in sorted_items]

    raise TypeError(
        "Vocabulary 객체를 문자열 리스트로 변환하지 못했습니다. "
        f"타입: {type(obj)}"
    )


def load_vocabulary(path):
    path = Path(path)
    suffix = path.suffix.lower()

    print(f"Loading vocabulary: {path}")

    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            obj = pickle.load(file)

    elif suffix == ".npy":
        obj = np.load(path, allow_pickle=True)

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            obj = json.load(file)

    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as file:
            obj = [
                line.rstrip("\n")
                for line in file
                if line.strip()
            ]

    else:
        raise ValueError(
            f"지원하지 않는 vocabulary 형식입니다: {suffix}"
        )

    vocab_list = normalize_vocab_object(obj)

    if len(vocab_list) == 0:
        raise ValueError("Vocabulary가 비어 있습니다.")

    return vocab_list


def load_labels(path):
    """
    CPC test labels를 DataFrame으로 반환합니다.
    """
    if path is None:
        return None

    path = Path(path)
    suffix = path.suffix.lower()

    print(f"Loading test labels: {path}")

    if suffix == ".csv":
        labels = pd.read_csv(path)

    elif suffix == ".parquet":
        labels = pd.read_parquet(path)

    elif suffix == ".npy":
        values = np.load(path, allow_pickle=True)

        if values.ndim == 1:
            labels = pd.DataFrame({"label": values})
        else:
            labels = pd.DataFrame(values)

    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            obj = pickle.load(file)

        if isinstance(obj, pd.DataFrame):
            labels = obj.copy()

        elif isinstance(obj, pd.Series):
            labels = obj.to_frame(name=obj.name or "label")

        elif isinstance(obj, dict):
            labels = pd.DataFrame(obj)

        else:
            values = np.asarray(obj)

            if values.ndim == 1:
                labels = pd.DataFrame({"label": values})
            else:
                labels = pd.DataFrame(values)

    else:
        raise ValueError(
            f"지원하지 않는 label 형식입니다: {suffix}"
        )

    if LABEL_COLUMNS is not None:
        missing_columns = [
            column for column in LABEL_COLUMNS
            if column not in labels.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Label 파일에 다음 열이 없습니다: {missing_columns}\n"
                f"현재 열: {list(labels.columns)}"
            )

        labels = labels[LABEL_COLUMNS].copy()

    return labels.reset_index(drop=True)


# ============================================================
# 7. 실제 데이터 로드 및 검증
# ============================================================
bow_train = load_sparse_matrix(TRAIN_BOW_PATH)
bow_test = load_sparse_matrix(TEST_BOW_PATH)
vocab = load_vocabulary(VOCAB_PATH)
test_labels = load_labels(TEST_LABEL_PATH)

print("\n=== Loaded data ===")
print(f"Train BoW : {bow_train.shape}, nnz={bow_train.nnz:,}")
print(f"Test BoW  : {bow_test.shape}, nnz={bow_test.nnz:,}")
print(f"Vocabulary: {len(vocab):,}")

if bow_train.shape[1] != len(vocab):
    raise ValueError(
        "Train BoW와 vocabulary 크기가 다릅니다.\n"
        f"bow_train.shape[1]={bow_train.shape[1]:,}\n"
        f"len(vocab)={len(vocab):,}"
    )

if bow_test.shape[1] != len(vocab):
    raise ValueError(
        "Test BoW와 vocabulary 크기가 다릅니다.\n"
        f"bow_test.shape[1]={bow_test.shape[1]:,}\n"
        f"len(vocab)={len(vocab):,}"
    )

if test_labels is not None:
    print(f"Test labels: {test_labels.shape}")
    print(f"Label columns: {list(test_labels.columns)}")

    if len(test_labels) != bow_test.shape[0]:
        raise ValueError(
            "Test label 개수와 test 문서 수가 다릅니다.\n"
            f"labels={len(test_labels):,}\n"
            f"test documents={bow_test.shape[0]:,}"
        )
else:
    print("Test labels: not configured; CPC evaluation will be skipped.")


# ============================================================
# 8. 동일한 학습 샘플 준비
# ============================================================
nonempty_train_indices = np.flatnonzero(
    np.asarray(bow_train.getnnz(axis=1)).ravel() > 0
)

if len(nonempty_train_indices) == 0:
    raise ValueError("Train BoW에 비어 있지 않은 문서가 없습니다.")

actual_sample_size = min(
    HDP_TRAIN_SAMPLE_SIZE,
    len(nonempty_train_indices)
)

if SAMPLE_IDX_PATH.is_file():
    sample_idx = np.load(SAMPLE_IDX_PATH)

    valid_saved_sample = (
        sample_idx.ndim == 1
        and len(sample_idx) == actual_sample_size
        and sample_idx.min(initial=0) >= 0
        and sample_idx.max(initial=0) < bow_train.shape[0]
    )

    if valid_saved_sample:
        print(f"[LOAD] Existing sample indices: {SAMPLE_IDX_PATH}")
    else:
        print("[WARNING] Saved sample indices are invalid. Regenerating.")
        sample_idx = None
else:
    sample_idx = None

if sample_idx is None:
    sample_rng = np.random.RandomState(HDP_SAMPLE_SEED)

    sample_idx = sample_rng.choice(
        nonempty_train_indices,
        size=actual_sample_size,
        replace=False,
    )

    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    np.save(SAMPLE_IDX_PATH, sample_idx)

    print(f"[SAVE] Sample indices: {SAMPLE_IDX_PATH}")

bow_train_sample = bow_train[sample_idx].tocsr()
bow_train_sample.sort_indices()

if np.any(np.asarray(bow_train_sample.getnnz(axis=1)).ravel() == 0):
    raise ValueError("샘플에 빈 학습 문서가 포함되어 있습니다.")

id2word = {
    index: str(word)
    for index, word in enumerate(vocab)
}

print("\n=== HDP configuration ===")
print(f"Gensim version         : {gensim.__version__}")
print(f"NumPy version          : {np.__version__}")
print(f"SciPy version          : {scipy.__version__}")
print(f"Train documents        : {bow_train.shape[0]:,}")
print(f"Non-empty train docs   : {len(nonempty_train_indices):,}")
print(f"Sampled train docs     : {len(sample_idx):,}")
print(f"Test documents         : {bow_test.shape[0]:,}")
print(f"Vocabulary size        : {len(vocab):,}")
print(f"Seeds                  : {SEEDS}")
print(f"T / K                  : {HDP_T} / {HDP_K}")
print(f"Chunksize              : {HDP_CHUNKSIZE:,}")
print(f"Approx. training chunks: {int(np.ceil(len(sample_idx) / HDP_CHUNKSIZE)):,}")
print(f"Inference batch size   : {HDP_INFERENCE_BATCH_SIZE:,}")
print("GPU usage              : Gensim HDP is CPU-based")


# ============================================================
# 9. 평가 함수
# ============================================================
def calculate_clustering_metrics(true_labels, predicted_topics):
    """
    Pur_p:
        각 예측 토픽에서 가장 많은 실제 CPC의 비율을 합산한 purity.

    Pur_a:
        각 실제 CPC에서 가장 많은 예측 토픽의 비율을 합산한
        inverse purity.

    NMI:
        실제 CPC와 예측 토픽 간 normalized mutual information.
    """
    true_labels = np.asarray(true_labels)
    predicted_topics = np.asarray(predicted_topics)

    if len(true_labels) != len(predicted_topics):
        raise ValueError(
            "Label과 predicted topic 길이가 다릅니다."
        )

    valid_mask = (
        ~pd.isna(true_labels)
        & ~pd.isna(predicted_topics)
    )

    true_labels = true_labels[valid_mask]
    predicted_topics = predicted_topics[valid_mask]

    if len(true_labels) == 0:
        return {
            "n_documents": 0,
            "pur_p": np.nan,
            "pur_a": np.nan,
            "nmi": np.nan,
        }

    contingency = pd.crosstab(
        pd.Series(predicted_topics, name="predicted_topic"),
        pd.Series(true_labels, name="actual_label"),
        dropna=False,
    )

    total = contingency.to_numpy().sum()

    pur_p = (
        contingency.max(axis=1).sum() / total
        if contingency.shape[0] > 0
        else np.nan
    )

    pur_a = (
        contingency.max(axis=0).sum() / total
        if contingency.shape[1] > 0
        else np.nan
    )

    nmi = normalized_mutual_info_score(
        true_labels.astype(str),
        predicted_topics.astype(str),
    )

    return {
        "n_documents": int(total),
        "pur_p": float(pur_p),
        "pur_a": float(pur_a),
        "nmi": float(nmi),
    }


def evaluate_theta(seed, theta, labels_df):
    if labels_df is None:
        return []

    nonempty_theta = theta.sum(axis=1) > 0
    predicted_topics = np.full(
        theta.shape[0],
        fill_value=-1,
        dtype=np.int32,
    )

    predicted_topics[nonempty_theta] = theta[
        nonempty_theta
    ].argmax(axis=1)

    results = []

    for label_column in labels_df.columns:
        valid = (
            nonempty_theta
            & labels_df[label_column].notna().to_numpy()
        )

        metrics = calculate_clustering_metrics(
            labels_df.loc[valid, label_column].to_numpy(),
            predicted_topics[valid],
        )

        result = {
            "model": "HDP",
            "seed": int(seed),
            "label_level": str(label_column),
            "n_documents": metrics["n_documents"],
            "pur_p": metrics["pur_p"],
            "pur_a": metrics["pur_a"],
            "nmi": metrics["nmi"],
        }

        results.append(result)

        print(
            f"  [{label_column}] "
            f"Pur_p={metrics['pur_p']:.4f} | "
            f"Pur_a={metrics['pur_a']:.4f} | "
            f"NMI={metrics['nmi']:.4f} | "
            f"N={metrics['n_documents']:,}"
        )

    return results


# ============================================================
# 10. HDP theta 배치 추론
# ============================================================
def infer_hdp_theta(
    hdp,
    bow_matrix,
    batch_size=HDP_INFERENCE_BATCH_SIZE,
):
    """
    hdp[doc]의 기본 eps=0.01 확률 절단을 피하고,
    hdp.inference()를 배치 단위로 호출합니다.

    반환:
        theta: test_documents × T, float32
    """
    n_documents = bow_matrix.shape[0]
    n_topics = int(hdp.m_T)

    theta = np.zeros(
        (n_documents, n_topics),
        dtype=np.float32,
    )

    progress = tqdm(
        range(0, n_documents, batch_size),
        desc="HDP test inference",
    )

    for start in progress:
        end = min(start + batch_size, n_documents)

        batch_matrix = bow_matrix[start:end].tocsr()
        batch_nnz = np.asarray(
            batch_matrix.getnnz(axis=1)
        ).ravel()

        local_nonempty_indices = np.flatnonzero(batch_nnz > 0)

        if len(local_nonempty_indices) == 0:
            continue

        nonempty_matrix = batch_matrix[
            local_nonempty_indices
        ].tocsr()

        batch_corpus = Sparse2Corpus(
            nonempty_matrix,
            documents_columns=False,
        )

        batch_documents = list(batch_corpus)

        gamma = hdp.inference(batch_documents)
        gamma = np.asarray(gamma, dtype=np.float64)

        if gamma.ndim != 2:
            raise ValueError(
                f"예상하지 못한 gamma shape: {gamma.shape}"
            )

        if gamma.shape[1] != n_topics:
            raise ValueError(
                f"Gamma topic dimension mismatch: "
                f"gamma={gamma.shape}, expected T={n_topics}"
            )

        if not np.isfinite(gamma).all():
            raise FloatingPointError(
                f"추론 gamma에 NaN/Inf가 있습니다: "
                f"documents {start}:{end}"
            )

        row_sums = gamma.sum(axis=1, keepdims=True)

        normalized_gamma = np.divide(
            gamma,
            row_sums,
            out=np.zeros_like(gamma),
            where=row_sums > 0,
        )

        global_indices = start + local_nonempty_indices

        theta[global_indices] = normalized_gamma.astype(
            np.float32
        )

        progress.set_postfix(
            processed=f"{end:,}/{n_documents:,}",
            refresh=False,
        )

    if not np.isfinite(theta).all():
        raise FloatingPointError(
            "최종 theta에 NaN 또는 Inf가 있습니다."
        )

    valid_rows = theta.sum(axis=1) > 0

    if np.any(valid_rows):
        row_sums = theta[valid_rows].sum(axis=1)

        if not np.allclose(row_sums, 1.0, atol=1e-5):
            max_error = float(
                np.max(np.abs(row_sums - 1.0))
            )

            raise ValueError(
                f"Theta 행 합 정규화 오류: max error={max_error}"
            )

    return theta


# ============================================================
# 11. Seed별 HDP 학습 함수
# ============================================================
def train_or_load_hdp(seed):
    model_path = MODEL_DIR / (
        f"hdp_seed{seed}"
        f"_sample{len(sample_idx)}"
        f"_T{HDP_T}_K{HDP_K}.model"
    )

    if REUSE_EXISTING_MODEL and model_path.is_file():
        print(f"[LOAD] Existing HDP model: {model_path}")

        hdp = HdpModel.load(str(model_path))

        if int(hdp.m_T) != int(HDP_T):
            raise ValueError(
                f"저장 모델 T={hdp.m_T}, 현재 T={HDP_T}"
            )

        return hdp, model_path, 0.0, True

    print(f"[TRAIN] HDP seed={seed}")

    # seed마다 corpus iterable을 새로 생성
    train_corpus = Sparse2Corpus(
        bow_train_sample,
        documents_columns=False,
    )

    start_time = time.time()

    hdp = HdpModel(
        corpus=train_corpus,
        id2word=id2word,
        random_state=seed,
        T=HDP_T,
        K=HDP_K,
        chunksize=HDP_CHUNKSIZE,
        kappa=HDP_KAPPA,
        tau=HDP_TAU,
        alpha=HDP_ALPHA,
        gamma=HDP_GAMMA,
        eta=HDP_ETA,
    )

    training_seconds = time.time() - start_time

    print(
        f"[DONE] seed={seed} training time: "
        f"{training_seconds / 60:.2f} minutes"
    )

    if SAVE_HDP_MODEL:
        hdp.save(str(model_path))
        print(f"[SAVE] HDP model: {model_path}")

    return hdp, model_path, training_seconds, False


# ============================================================
# 12. Seed별 학습·추론·평가
# ============================================================
all_metric_rows = []
run_records = []
successful_seeds = []
failed_seeds = []

for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"HDP SEED {seed}")
    print("=" * 70)

    # 재현성 보조 설정
    random.seed(seed)
    np.random.seed(seed)

    theta_path = THETA_DIR / (
        f"hdp_theta_test_seed{seed}"
        f"_sample{len(sample_idx)}"
        f"_T{HDP_T}_K{HDP_K}.npy"
    )

    try:
        hdp, model_path, training_seconds, reused_model = (
            train_or_load_hdp(seed)
        )

        if theta_path.is_file() and not FORCE_REINFERENCE:
            print(f"[LOAD] Existing theta: {theta_path}")
            theta_test = np.load(
                theta_path,
                mmap_mode=None,
            )

            inference_seconds = 0.0
            reused_theta = True

        else:
            print(f"[INFERENCE] HDP seed={seed}")

            inference_start = time.time()

            theta_test = infer_hdp_theta(
                hdp,
                bow_test,
                batch_size=HDP_INFERENCE_BATCH_SIZE,
            )

            inference_seconds = time.time() - inference_start
            reused_theta = False

            np.save(theta_path, theta_test)

            print(f"[SAVE] Test theta: {theta_path}")
            print(
                f"[DONE] Inference time: "
                f"{inference_seconds / 60:.2f} minutes"
            )

        expected_shape = (
            bow_test.shape[0],
            int(hdp.m_T),
        )

        if theta_test.shape != expected_shape:
            raise ValueError(
                f"Theta shape mismatch: "
                f"actual={theta_test.shape}, "
                f"expected={expected_shape}"
            )

        valid_documents = theta_test.sum(axis=1) > 0
        empty_documents = int((~valid_documents).sum())

        if np.any(valid_documents):
            assigned_topics = theta_test[
                valid_documents
            ].argmax(axis=1)

            effective_topics = int(
                np.unique(assigned_topics).size
            )
        else:
            effective_topics = 0

        print("\n=== Seed result ===")
        print(f"Seed                     : {seed}")
        print(f"Theta shape              : {theta_test.shape}")
        print(f"Truncation dimension T   : {hdp.m_T}")
        print(f"Effective assigned topics: {effective_topics}")
        print(f"Empty test documents     : {empty_documents:,}")

        seed_metrics = evaluate_theta(
            seed=seed,
            theta=theta_test,
            labels_df=test_labels,
        )

        all_metric_rows.extend(seed_metrics)
        successful_seeds.append(seed)

        run_records.append({
            "seed": int(seed),
            "status": "success",
            "model_path": str(model_path),
            "theta_path": str(theta_path),
            "theta_shape": list(theta_test.shape),
            "truncation_T": int(hdp.m_T),
            "effective_assigned_topics": effective_topics,
            "empty_test_documents": empty_documents,
            "training_seconds": float(training_seconds),
            "inference_seconds": float(inference_seconds),
            "reused_model": bool(reused_model),
            "reused_theta": bool(reused_theta),
        })

        del theta_test
        del hdp
        gc.collect()

    except Exception as error:
        failed_seeds.append(seed)

        error_traceback = traceback.format_exc()

        print(f"\n[FAILED] HDP seed={seed}")
        print(f"Error: {error}")
        print(error_traceback)

        run_records.append({
            "seed": int(seed),
            "status": "failed",
            "error": str(error),
            "traceback": error_traceback,
        })

        gc.collect()


# ============================================================
# 13. 평가 결과 저장 및 평균±표준편차 계산
# ============================================================
metric_summary_rows = []

if all_metric_rows:
    metrics_df = pd.DataFrame(all_metric_rows)

    metrics_df.to_csv(
        METRICS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[SAVE] Per-seed metrics: {METRICS_CSV_PATH}")

    for label_level, group in metrics_df.groupby("label_level"):
        summary_row = {
            "model": "HDP",
            "label_level": label_level,
            "successful_seed_count": int(group["seed"].nunique()),
            "pur_p_mean": float(group["pur_p"].mean()),
            "pur_p_std": float(group["pur_p"].std(ddof=1))
                if len(group) > 1 else 0.0,
            "pur_a_mean": float(group["pur_a"].mean()),
            "pur_a_std": float(group["pur_a"].std(ddof=1))
                if len(group) > 1 else 0.0,
            "nmi_mean": float(group["nmi"].mean()),
            "nmi_std": float(group["nmi"].std(ddof=1))
                if len(group) > 1 else 0.0,
        }

        metric_summary_rows.append(summary_row)

    metric_summary_df = pd.DataFrame(metric_summary_rows)

    metric_summary_path = RESULT_DIR / (
        f"{RUN_NAME}_metric_mean_std.csv"
    )

    metric_summary_df.to_csv(
        metric_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[SAVE] Metric mean/std: {metric_summary_path}")

    print("\n=== HDP metric summary: mean ± std ===")

    for row in metric_summary_rows:
        print(
            f"[{row['label_level']}] "
            f"Pur_p={row['pur_p_mean']:.4f}"
            f"±{row['pur_p_std']:.4f} | "
            f"Pur_a={row['pur_a_mean']:.4f}"
            f"±{row['pur_a_std']:.4f} | "
            f"NMI={row['nmi_mean']:.4f}"
            f"±{row['nmi_std']:.4f}"
        )

else:
    print(
        "\n[INFO] Test label 경로가 설정되지 않았거나 "
        "평가 결과가 없어 metric CSV를 생성하지 않았습니다."
    )


# ============================================================
# 14. 전체 실행 요약 저장
# ============================================================
final_summary = {
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(),
    "project_root": str(PROJECT_ROOT),
    "data": {
        "train_bow_path": str(TRAIN_BOW_PATH),
        "test_bow_path": str(TEST_BOW_PATH),
        "vocab_path": str(VOCAB_PATH),
        "test_label_path": (
            str(TEST_LABEL_PATH)
            if TEST_LABEL_PATH is not None
            else None
        ),
        "train_shape": list(bow_train.shape),
        "test_shape": list(bow_test.shape),
        "vocabulary_size": int(len(vocab)),
        "sample_size": int(len(sample_idx)),
        "sample_seed": int(HDP_SAMPLE_SEED),
        "sample_index_path": str(SAMPLE_IDX_PATH),
    },
    "hdp_config": {
        "seeds": [int(seed) for seed in SEEDS],
        "T": int(HDP_T),
        "K": int(HDP_K),
        "chunksize": int(HDP_CHUNKSIZE),
        "kappa": float(HDP_KAPPA),
        "tau": float(HDP_TAU),
        "alpha": float(HDP_ALPHA),
        "gamma": float(HDP_GAMMA),
        "eta": float(HDP_ETA),
        "inference_batch_size": int(
            HDP_INFERENCE_BATCH_SIZE
        ),
    },
    "successful_seeds": [
        int(seed) for seed in successful_seeds
    ],
    "failed_seeds": [
        int(seed) for seed in failed_seeds
    ],
    "runs": run_records,
    "metric_summary": metric_summary_rows,
    "important_note": (
        "Seed별 topic index는 의미적으로 정렬되어 있지 않으므로 "
        "theta를 직접 평균하지 않았습니다. "
        "대신 seed별 평가 지표의 평균과 표준편차를 계산했습니다."
    ),
}

with open(
    SUMMARY_JSON_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_summary,
        file,
        ensure_ascii=False,
        indent=2,
    )

with open(
    RUN_LOG_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_summary,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 15. 최종 출력
# ============================================================
print("\n" + "=" * 70)
print("HDP PIPELINE FINISHED")
print("=" * 70)
print(f"Successful seeds : {successful_seeds}")
print(f"Failed seeds     : {failed_seeds}")
print(f"Model directory  : {MODEL_DIR}")
print(f"Theta directory  : {THETA_DIR}")
print(f"Result directory : {RESULT_DIR}")
print(f"Summary JSON     : {SUMMARY_JSON_PATH}")

if all_metric_rows:
    print(f"Metrics CSV      : {METRICS_CSV_PATH}")

if failed_seeds:
    raise RuntimeError(
        f"HDP 학습 또는 추론에 실패한 seed가 있습니다: {failed_seeds}. "
        f"위 traceback을 확인하세요."
    )

print("\n[PASS] 모든 HDP seed의 학습·추론이 완료되었습니다.")
print(
    "[NOTE] seed별 theta는 직접 평균하지 않았으며, "
    "평가 지표의 평균±표준편차만 계산했습니다."
)
