# ======================================================================================
# TSNTM STATIC 5×5 — 3-SEED CPC BENCHMARK
#
# Model:
#   Tree-Structured Neural Topic Model (TSNTM), ACL 2020
#   Official source: https://github.com/misonuma/tsntm
#
# Evaluation:
#   Tree nodes = root 1 + level-2 5 + level-3 25 = 31
#   CPC evaluation = 30 non-root topics
#   Seeds = [42, 43, 44]
#
# Runtime:
#   Google Colab + Tesla T4
#   tensorflow.compat.v1 source patch
#   Automatic checkpoint resume
#   Automatic runtime termination after successful completion
# ======================================================================================


# ======================================================================================
# 0. Environment configuration — must precede TensorFlow import
# ======================================================================================

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import sys
import subprocess
import importlib.util


def ensure_package(package, import_name=None):
    if import_name is None:
        import_name = package.split("==")[0].replace("-", "_")

    if importlib.util.find_spec(import_name) is None:
        print(f"[INSTALL] {package}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package,
        ])


# TensorFlow is normally preinstalled in Colab.
# tf-keras supplies legacy tf.layers compatibility for current Keras environments.
ensure_package("tf-keras", "tf_keras")
ensure_package("scikit-learn", "sklearn")
ensure_package("scipy", "scipy")
ensure_package("pandas", "pandas")
ensure_package("joblib", "joblib")
ensure_package("tqdm", "tqdm")

import gc
import re
import json
import time
import math
import pickle
import random
import shutil
import hashlib
import warnings
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

print("=" * 100)
print("TSNTM STATIC 5×5 — 3-SEED CPC BENCHMARK")
print("=" * 100)


# ======================================================================================
# 1. Mount Google Drive
# ======================================================================================

from google.colab import drive


def contains_project_records(path):
    path = Path(path)

    return (
        path.exists()
        and (
            (path / "data/processed/train_records.pkl").exists()
            or len(list(path.rglob("train_records.pkl"))) > 0
        )
        and (
            (path / "data/processed/test_records.pkl").exists()
            or len(list(path.rglob("test_records.pkl"))) > 0
        )
    )


candidate_project_roots = [
    Path("/content/depth_ot_evaluation_drive/MyDrive/depth_ot_patent"),
    Path("/content/drive/MyDrive/depth_ot_patent"),
    Path("/content/gdrive/MyDrive/depth_ot_patent"),
    Path("/content/fastopic_drive/MyDrive/depth_ot_patent"),
    Path("/content/tsntm_drive/MyDrive/depth_ot_patent"),
]

PROJECT_ROOT = None

for candidate in candidate_project_roots:
    if contains_project_records(candidate):
        PROJECT_ROOT = candidate
        print(f"[VERIFIED EXISTING MOUNT] {PROJECT_ROOT}")
        break

if PROJECT_ROOT is None:
    mount_point = Path("/content/tsntm_drive")

    if not (mount_point / "MyDrive").exists():
        print(f"[MOUNT] Google Drive → {mount_point}")
        drive.mount(str(mount_point), force_remount=False)

    my_drive = mount_point / "MyDrive"

    project_candidates = [
        path.parent.parent
        for path in my_drive.rglob("data/processed/train_records.pkl")
        if (
            path.parent / "test_records.pkl"
        ).exists()
    ]

    if not project_candidates:
        project_candidates = [
            path
            for path in my_drive.rglob("depth_ot_patent")
            if contains_project_records(path)
        ]

    if not project_candidates:
        raise FileNotFoundError(
            "Google Drive에서 depth_ot_patent 프로젝트와 "
            "train_records.pkl/test_records.pkl을 찾지 못했습니다."
        )

    PROJECT_ROOT = sorted(
        project_candidates,
        key=lambda path: len(str(path)),
    )[0]

print(f"[PROJECT ROOT] {PROJECT_ROOT}")


def find_required_file(filename, preferred_path):
    preferred_path = Path(preferred_path)

    if preferred_path.exists() and preferred_path.stat().st_size > 0:
        return preferred_path

    candidates = list(PROJECT_ROOT.rglob(filename))

    if not candidates:
        raise FileNotFoundError(
            f"{filename}을 찾지 못했습니다: {PROJECT_ROOT}"
        )

    candidates = sorted(
        candidates,
        key=lambda path: (
            "processed" not in str(path),
            len(str(path)),
        ),
    )

    return candidates[0]


TRAIN_RECORD_PATH = find_required_file(
    "train_records.pkl",
    PROJECT_ROOT / "data/processed/train_records.pkl",
)

TEST_RECORD_PATH = find_required_file(
    "test_records.pkl",
    PROJECT_ROOT / "data/processed/test_records.pkl",
)

print(f"[TRAIN RECORDS] {TRAIN_RECORD_PATH}")
print(f"[TEST RECORDS ] {TEST_RECORD_PATH}")
print(
    f"[TRAIN SIZE] "
    f"{TRAIN_RECORD_PATH.stat().st_size / (1024**2):.2f} MiB"
)
print(
    f"[TEST SIZE ] "
    f"{TEST_RECORD_PATH.stat().st_size / (1024**2):.2f} MiB"
)


# ======================================================================================
# 2. Experiment configuration
# ======================================================================================

SEEDS = [42, 43, 44]

VOCAB_SIZE = 8000
MIN_DOC_COUNT = 10
MAX_DOC_FREQ = 0.70

TREE_CODE = 55
TREE_DESCRIPTION = "static_5x5"
TOTAL_TREE_NODES = 31
EVAL_TOPICS = 30

MAX_EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.01
OPTIMIZER = "Adagrad"

KEEP_PROB = 0.8
REG_WEIGHT = 1.0
GRAD_CLIP = 5.0

DIM_HIDDEN_BOW = 256
DIM_LATENT_BOW = 32
DIM_EMBEDDING = 256
DEPTH_TEMPERATURE = 10.0

DEV_FRACTION = 0.10
DEV_EVAL_INTERVAL = 5
EARLY_STOPPING_PATIENCE = 8  # 8 checks × 5 epochs = 40 epochs
MC_TEST_SAMPLES = 5

RESULT_DIR = (
    PROJECT_ROOT
    / "results/baselines"
    / "tsntm_static55_nonroot_k30_seeds_42_43_44"
)

CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
ARRAY_DIR = RESULT_DIR / "arrays"
TOPIC_DIR = RESULT_DIR / "topics"
CACHE_DIR = Path("/content/tsntm_cpc_cache")
SOURCE_DIR = Path("/content/tsntm_official")

for directory in [
    RESULT_DIR,
    CHECKPOINT_DIR,
    ARRAY_DIR,
    TOPIC_DIR,
    CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIGURATION = {
    "model": "TSNTM",
    "model_description": (
        "Official TSNTM source patched with tensorflow.compat.v1"
    ),
    "tree": TREE_DESCRIPTION,
    "tree_code": TREE_CODE,
    "dynamic_tree": False,
    "total_tree_nodes": TOTAL_TREE_NODES,
    "evaluation_topics": EVAL_TOPICS,
    "evaluation_rule": "drop root and renormalize 30 non-root probabilities",
    "seeds": SEEDS,
    "vocabulary_size": VOCAB_SIZE,
    "min_doc_count": MIN_DOC_COUNT,
    "max_doc_frequency": MAX_DOC_FREQ,
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optimizer": OPTIMIZER,
    "keep_prob": KEEP_PROB,
    "regularization_weight": REG_WEIGHT,
    "gradient_clip": GRAD_CLIP,
    "hidden_dimension": DIM_HIDDEN_BOW,
    "latent_dimension": DIM_LATENT_BOW,
    "embedding_dimension": DIM_EMBEDDING,
    "depth_temperature": DEPTH_TEMPERATURE,
    "dev_fraction": DEV_FRACTION,
    "dev_evaluation_interval": DEV_EVAL_INTERVAL,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "mc_test_samples": MC_TEST_SAMPLES,
    "train_record_path": str(TRAIN_RECORD_PATH),
    "test_record_path": str(TEST_RECORD_PATH),
    "result_directory": str(RESULT_DIR),
}

with open(
    RESULT_DIR / "configuration.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        CONFIGURATION,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"[RESULT DIR] {RESULT_DIR}")


# ======================================================================================
# 3. Clone and patch official TSNTM
# ======================================================================================

if not (SOURCE_DIR / ".git").exists():
    print("[CLONE] Official TSNTM repository")

    subprocess.check_call([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/misonuma/tsntm.git",
        str(SOURCE_DIR),
    ])
else:
    print(f"[SOURCE CACHE HIT] {SOURCE_DIR}")

patch_targets = [
    SOURCE_DIR / "hntm.py",
    SOURCE_DIR / "nn.py",
    SOURCE_DIR / "components.py",
    SOURCE_DIR / "tree.py",
]

for source_path in patch_targets:
    if not source_path.exists():
        continue

    text = source_path.read_text(encoding="utf-8")

    text = text.replace(
        "import tensorflow as tf",
        "import tensorflow.compat.v1 as tf\n"
        "tf.disable_v2_behavior()",
    )

    text = text.replace(
        "tf.contrib.layers.xavier_initializer()",
        "tf.glorot_uniform_initializer()",
    )

    source_path.write_text(
        text,
        encoding="utf-8",
    )

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

print("[PATCH COMPLETE] tensorflow.compat.v1 + legacy Keras")


# ======================================================================================
# 4. Import TensorFlow and verify T4
# ======================================================================================

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

print(f"[TENSORFLOW] {tf.__version__}")
print(
    f"[TF LEGACY LAYERS] "
    f"{hasattr(tf.layers, 'Dense')}"
)

if not hasattr(tf.layers, "Dense"):
    raise RuntimeError(
        "tf.layers.Dense를 사용할 수 없습니다. "
        "새 Colab 런타임에서 이 셀을 가장 먼저 실행해야 합니다. "
        "런타임을 재시작한 뒤 다른 TensorFlow import 없이 다시 실행하세요."
    )

gpu_devices = tf.config.list_physical_devices("GPU")

print(f"[TF GPU DEVICES] {gpu_devices}")

if not gpu_devices:
    raise RuntimeError(
        "TensorFlow에서 GPU가 감지되지 않습니다. "
        "Colab 메뉴에서 런타임 → 런타임 유형 변경 → T4 GPU를 선택하세요."
    )

from hntm import HierarchicalNeuralTopicModel
from tree import get_tree_idxs

print("[OFFICIAL TSNTM IMPORT SUCCESS]")


# ======================================================================================
# 5. Reproducibility
# ======================================================================================

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


# ======================================================================================
# 6. Record loading and robust field extraction
# ======================================================================================

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")

    if isinstance(data, dict):
        for key in [
            "records",
            "data",
            "items",
            "patents",
            "documents",
        ]:
            if key in data and isinstance(data[key], (list, tuple)):
                data = data[key]
                break

    return list(data)


def normalize_text_value(value):
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        preferred_keys = [
            "text",
            "claim_text",
            "content",
            "value",
            "abstract",
            "title",
        ]

        parts = []

        for key in preferred_keys:
            if key in value:
                text = normalize_text_value(value[key])

                if text.strip():
                    parts.append(text)

        if not parts:
            parts = [
                normalize_text_value(item)
                for item in value.values()
            ]

        return " ".join(parts)

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return " ".join(
            normalize_text_value(item)
            for item in value
            if item is not None
        )

    return str(value)


def extract_document_text(record):
    if not isinstance(record, dict):
        return normalize_text_value(record).strip()

    claim_candidates = [
        "claims",
        "claim",
        "claim_text",
        "claims_text",
        "patent_claims",
        "independent_claims",
    ]

    for key in claim_candidates:
        if key in record:
            text = normalize_text_value(record[key]).strip()

            if text:
                return text

    fallback_parts = []

    for key in [
        "title",
        "abstract",
        "description",
        "summary",
        "text",
    ]:
        if key in record:
            text = normalize_text_value(record[key]).strip()

            if text:
                fallback_parts.append(text)

    return " ".join(fallback_parts).strip()


CPC_PATTERN = re.compile(r"\b([A-HY])\s*([0-9]{2})\s*([A-Z])\b")


def find_cpc_string(value):
    if value is None:
        return None

    if isinstance(value, dict):
        preferred_keys = [
            "cpc",
            "code",
            "symbol",
            "classification",
            "classification_code",
            "primary_cpc",
            "main_cpc",
        ]

        for key in preferred_keys:
            if key in value:
                result = find_cpc_string(value[key])

                if result is not None:
                    return result

        for item in value.values():
            result = find_cpc_string(item)

            if result is not None:
                return result

        return None

    if isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            result = find_cpc_string(item)

            if result is not None:
                return result

        return None

    text = normalize_text_value(value).upper()
    match = CPC_PATTERN.search(text)

    if match is None:
        return None

    return "".join(match.groups())


def extract_primary_cpc(record):
    if not isinstance(record, dict):
        return find_cpc_string(record)

    preferred_keys = [
        "primary_cpc",
        "main_cpc",
        "cpc_primary",
        "primary_cpc_code",
        "main_classification",
        "cpc",
        "cpc_codes",
        "classifications",
        "classification",
    ]

    for key in preferred_keys:
        if key in record:
            result = find_cpc_string(record[key])

            if result is not None:
                return result

    return find_cpc_string(record)


print("[LOAD] Train records")
train_records = load_pickle(TRAIN_RECORD_PATH)

print("[LOAD] Test records")
test_records = load_pickle(TEST_RECORD_PATH)

print(f"[TRAIN PATENTS] {len(train_records):,}")
print(f"[TEST PATENTS ] {len(test_records):,}")

if len(train_records) != 49599:
    print(
        f"[WARNING] Expected 49,599 train patents, "
        f"found {len(train_records):,}"
    )

if len(test_records) != 9881:
    print(
        f"[WARNING] Expected 9,881 test patents, "
        f"found {len(test_records):,}"
    )

print("[TEXT] Extracting train documents")
train_docs = [
    extract_document_text(record)
    for record in tqdm(train_records)
]

print("[TEXT] Extracting test documents")
test_docs = [
    extract_document_text(record)
    for record in tqdm(test_records)
]

empty_train = sum(not text.strip() for text in train_docs)
empty_test = sum(not text.strip() for text in test_docs)

print(f"[EMPTY TRAIN TEXT] {empty_train}")
print(f"[EMPTY TEST TEXT ] {empty_test}")

if empty_train > 0 or empty_test > 0:
    raise RuntimeError(
        "빈 특허 텍스트가 발견되었습니다. "
        "기존 FASTopic에서 사용한 필드 추출 규칙을 확인하세요."
    )

print("[CPC] Extracting test labels")

test_cpc = [
    extract_primary_cpc(record)
    for record in tqdm(test_records)
]

missing_cpc = sum(code is None for code in test_cpc)

if missing_cpc > 0:
    raise RuntimeError(
        f"Test CPC가 없는 문서가 {missing_cpc:,}개 있습니다."
    )

test_section_labels = np.asarray([
    code[:1]
    for code in test_cpc
])

test_class_labels = np.asarray([
    code[:3]
    for code in test_cpc
])

test_subclass_labels = np.asarray([
    code[:4]
    for code in test_cpc
])

label_counts = {
    "section": int(np.unique(test_section_labels).size),
    "class": int(np.unique(test_class_labels).size),
    "subclass": int(np.unique(test_subclass_labels).size),
}

print(f"[CPC LABEL COUNTS] {label_counts}")

expected_label_counts = {
    "section": 9,
    "class": 121,
    "subclass": 466,
}

if label_counts != expected_label_counts:
    raise RuntimeError(
        f"CPC label count mismatch: "
        f"expected={expected_label_counts}, "
        f"actual={label_counts}"
    )

# Large raw records are no longer needed.
del train_records
del test_records
gc.collect()


# ======================================================================================
# 7. Train-only vocabulary and sparse BoW cache
# ======================================================================================

data_signature = hashlib.sha1(
    (
        str(TRAIN_RECORD_PATH)
        + str(TRAIN_RECORD_PATH.stat().st_size)
        + str(TEST_RECORD_PATH)
        + str(TEST_RECORD_PATH.stat().st_size)
        + str(VOCAB_SIZE)
        + str(MIN_DOC_COUNT)
        + str(MAX_DOC_FREQ)
    ).encode("utf-8")
).hexdigest()[:16]

VECTOR_CACHE = CACHE_DIR / f"vectorizer_{data_signature}.joblib"
TRAIN_BOW_CACHE = CACHE_DIR / f"train_bow_{data_signature}.npz"
TEST_BOW_CACHE = CACHE_DIR / f"test_bow_{data_signature}.npz"

if (
    VECTOR_CACHE.exists()
    and TRAIN_BOW_CACHE.exists()
    and TEST_BOW_CACHE.exists()
):
    print("[BOW CACHE HIT] Loading vectorizer and sparse matrices")

    vectorizer = joblib.load(VECTOR_CACHE)
    train_bow = sp.load_npz(TRAIN_BOW_CACHE).tocsr()
    test_bow = sp.load_npz(TEST_BOW_CACHE).tocsr()

else:
    print("[BOW] Fitting train-only vocabulary")

    vectorizer = CountVectorizer(
        max_features=VOCAB_SIZE,
        min_df=MIN_DOC_COUNT,
        max_df=MAX_DOC_FREQ,
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
        dtype=np.float32,
    )

    train_bow = vectorizer.fit_transform(train_docs).tocsr()
    test_bow = vectorizer.transform(test_docs).tocsr()

    joblib.dump(vectorizer, VECTOR_CACHE)
    sp.save_npz(TRAIN_BOW_CACHE, train_bow)
    sp.save_npz(TEST_BOW_CACHE, test_bow)

actual_vocab_size = len(vectorizer.get_feature_names_out())

print(f"[VOCABULARY] {actual_vocab_size:,}")
print(f"[TRAIN BOW] {train_bow.shape}, nnz={train_bow.nnz:,}")
print(f"[TEST BOW ] {test_bow.shape}, nnz={test_bow.nnz:,}")

if actual_vocab_size != VOCAB_SIZE:
    raise RuntimeError(
        f"Expected vocabulary size {VOCAB_SIZE:,}, "
        f"found {actual_vocab_size:,}"
    )

if train_bow.shape != (len(train_docs), VOCAB_SIZE):
    raise RuntimeError(
        f"Unexpected train BoW shape: {train_bow.shape}"
    )

if test_bow.shape != (len(test_docs), VOCAB_SIZE):
    raise RuntimeError(
        f"Unexpected test BoW shape: {test_bow.shape}"
    )

feature_names = np.asarray(
    vectorizer.get_feature_names_out()
)

# Raw strings are no longer needed after BoW construction.
del train_docs
del test_docs
gc.collect()


# ======================================================================================
# 8. Fixed train/DEV split
# ======================================================================================

all_train_indices = np.arange(
    train_bow.shape[0],
    dtype=np.int64,
)

fit_indices, dev_indices = train_test_split(
    all_train_indices,
    test_size=DEV_FRACTION,
    random_state=42,
    shuffle=True,
)

fit_indices = np.asarray(
    fit_indices,
    dtype=np.int64,
)

dev_indices = np.asarray(
    dev_indices,
    dtype=np.int64,
)

print(f"[FIT DOCUMENTS] {len(fit_indices):,}")
print(f"[DEV DOCUMENTS] {len(dev_indices):,}")
print(f"[TEST DOCUMENTS] {test_bow.shape[0]:,}")


# ======================================================================================
# 9. CPC evaluation metrics
# ======================================================================================

def predicted_cluster_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_correct = 0

    for cluster in np.unique(y_pred):
        labels = y_true[y_pred == cluster]

        if labels.size == 0:
            continue

        counts = Counter(labels.tolist())
        total_correct += counts.most_common(1)[0][1]

    return float(total_correct / len(y_true))


def inverse_label_wise_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_correct = 0

    for label in np.unique(y_true):
        clusters = y_pred[y_true == label]

        if clusters.size == 0:
            continue

        counts = Counter(clusters.tolist())
        total_correct += counts.most_common(1)[0][1]

    return float(total_correct / len(y_true))


def evaluate_level(y_true, predicted_topics):
    return {
        "pur_p": predicted_cluster_purity(
            y_true,
            predicted_topics,
        ),
        "pur_a": inverse_label_wise_purity(
            y_true,
            predicted_topics,
        ),
        "nmi": float(
            normalized_mutual_info_score(
                y_true,
                predicted_topics,
                average_method="arithmetic",
            )
        ),
    }


def evaluate_all_levels(predicted_topics):
    return {
        "section": evaluate_level(
            test_section_labels,
            predicted_topics,
        ),
        "class": evaluate_level(
            test_class_labels,
            predicted_topics,
        ),
        "subclass": evaluate_level(
            test_subclass_labels,
            predicted_topics,
        ),
    }


# ======================================================================================
# 10. TSNTM utilities
# ======================================================================================

def make_model_config(seed):
    config = SimpleNamespace()

    config.seed = int(seed)
    config.tree_idxs = get_tree_idxs(TREE_CODE)

    config.dim_bow = VOCAB_SIZE
    config.dim_hidden_bow = DIM_HIDDEN_BOW
    config.dim_latent_bow = DIM_LATENT_BOW
    config.dim_emb = DIM_EMBEDDING

    config.depth_temperature = DEPTH_TEMPERATURE

    config.opt = OPTIMIZER
    config.lr = LEARNING_RATE
    config.reg = REG_WEIGHT
    config.grad_clip = GRAD_CLIP
    config.keep_prob = KEEP_PROB

    config.static = True
    config.add_threshold = 0.05
    config.remove_threshold = 0.05
    config.cell = "rnn"

    return config


def make_tf_session():
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
    )

    tf_config.gpu_options.allow_growth = True

    return tf.Session(config=tf_config)


def iterate_batches(
    matrix,
    indices,
    batch_size,
    shuffle,
    seed,
):
    indices = np.asarray(indices).copy()

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[
            start:start + batch_size
        ]

        dense_batch = matrix[
            batch_indices
        ].toarray().astype(
            np.float32,
            copy=False,
        )

        yield batch_indices, dense_batch


def evaluate_loss(
    sess,
    model,
    matrix,
    indices,
):
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_reg = 0.0
    total_docs = 0

    for _, dense_bow in iterate_batches(
        matrix=matrix,
        indices=indices,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=0,
    ):
        feed_dict = {
            model.t_variables["bow"]: dense_bow,
            model.t_variables["keep_prob"]: 1.0,
        }

        (
            loss,
            recon,
            kl,
            reg,
        ) = sess.run(
            [
                model.loss,
                model.topic_loss_recon,
                model.topic_loss_kl,
                model.topic_loss_reg,
            ],
            feed_dict=feed_dict,
        )

        batch_docs = dense_bow.shape[0]

        total_loss += float(loss) * batch_docs
        total_recon += float(recon) * batch_docs
        total_kl += float(kl) * batch_docs
        total_reg += float(reg) * batch_docs
        total_docs += batch_docs

    return {
        "loss": total_loss / total_docs,
        "recon": total_recon / total_docs,
        "kl": total_kl / total_docs,
        "reg": total_reg / total_docs,
    }


def infer_probabilities(
    sess,
    model,
    matrix,
    mc_samples=1,
):
    all_probabilities = []

    all_indices = np.arange(
        matrix.shape[0],
        dtype=np.int64,
    )

    for _, dense_bow in tqdm(
        iterate_batches(
            matrix=matrix,
            indices=all_indices,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=0,
        ),
        total=math.ceil(matrix.shape[0] / BATCH_SIZE),
        desc="TSNTM test inference",
    ):
        feed_dict = {
            model.t_variables["bow"]: dense_bow,
            model.t_variables["keep_prob"]: 1.0,
        }

        batch_probabilities = []

        for _ in range(mc_samples):
            prob_topic = sess.run(
                model.prob_topic,
                feed_dict=feed_dict,
            )

            batch_probabilities.append(
                np.asarray(
                    prob_topic,
                    dtype=np.float32,
                )
            )

        mean_probabilities = np.mean(
            batch_probabilities,
            axis=0,
            dtype=np.float32,
        )

        all_probabilities.append(
            mean_probabilities
        )

    return np.concatenate(
        all_probabilities,
        axis=0,
    )


def atomic_json_save(data, path):
    path = Path(path)
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


# ======================================================================================
# 11. Train and evaluate three seeds
# ======================================================================================

all_seed_results = []
all_metric_rows = []

total_start_time = time.time()

for seed_number, seed in enumerate(SEEDS, start=1):
    print("\n" + "=" * 100)
    print(
        f"TSNTM SEED {seed} — "
        f"{seed_number}/{len(SEEDS)}"
    )
    print("=" * 100)

    seed_start_time = time.time()
    set_seed(seed)

    seed_checkpoint_dir = (
        CHECKPOINT_DIR / f"seed_{seed}"
    )
    seed_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_prefix = (
        seed_checkpoint_dir / "best_model"
    )
    latest_checkpoint_prefix = (
        seed_checkpoint_dir / "latest_model"
    )

    state_path = (
        seed_checkpoint_dir / "training_state.json"
    )
    complete_result_path = (
        RESULT_DIR / f"seed_{seed}_metrics.json"
    )

    # Already completed seeds can be skipped.
    if complete_result_path.exists():
        print(
            f"[SEED {seed} ALREADY COMPLETE] "
            "Loading saved result"
        )

        with open(
            complete_result_path,
            "r",
            encoding="utf-8",
        ) as f:
            seed_result = json.load(f)

        all_seed_results.append(seed_result)

        for level in [
            "section",
            "class",
            "subclass",
        ]:
            all_metric_rows.append({
                "seed": seed,
                "level": level,
                **seed_result["metrics"][level],
                "active_topics": seed_result[
                    "active_topics"
                ],
                "max_topic_share": seed_result[
                    "max_topic_share"
                ],
            })

        continue

    model_config = make_model_config(seed)
    model = HierarchicalNeuralTopicModel(
        model_config
    )

    if len(model.topic_idxs) != TOTAL_TREE_NODES:
        raise RuntimeError(
            f"Expected {TOTAL_TREE_NODES} tree nodes, "
            f"found {len(model.topic_idxs)}: "
            f"{model.topic_idxs}"
        )

    print(f"[TREE TOPIC IDS] {model.topic_idxs}")
    print(
        f"[NON-ROOT TOPICS] "
        f"{len(model.topic_idxs) - 1}"
    )

    session = make_tf_session()
    saver = tf.train.Saver(max_to_keep=2)

    session.run(tf.global_variables_initializer())

    start_epoch = 1
    best_dev_loss = float("inf")
    early_stop_count = 0
    training_history = []

    latest_checkpoint = tf.train.latest_checkpoint(
        str(seed_checkpoint_dir)
    )

    if latest_checkpoint is not None and state_path.exists():
        print(
            f"[RESUME] Restoring {latest_checkpoint}"
        )

        saver.restore(
            session,
            latest_checkpoint,
        )

        with open(
            state_path,
            "r",
            encoding="utf-8",
        ) as f:
            saved_state = json.load(f)

        start_epoch = (
            int(saved_state["last_epoch"]) + 1
        )

        best_dev_loss = float(
            saved_state["best_dev_loss"]
        )

        early_stop_count = int(
            saved_state["early_stop_count"]
        )

        training_history = saved_state.get(
            "training_history",
            [],
        )

        print(
            f"[RESUME] start_epoch={start_epoch}, "
            f"best_dev={best_dev_loss:.6f}, "
            f"patience={early_stop_count}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

    stopped_epoch = MAX_EPOCHS

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        epoch_start = time.time()

        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_reg_sum = 0.0
        train_document_count = 0

        progress = tqdm(
            iterate_batches(
                matrix=train_bow,
                indices=fit_indices,
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=seed * 100000 + epoch,
            ),
            total=math.ceil(
                len(fit_indices) / BATCH_SIZE
            ),
            desc=f"Seed {seed} epoch {epoch:03d}",
            leave=False,
        )

        for _, dense_bow in progress:
            feed_dict = {
                model.t_variables["bow"]: dense_bow,
                model.t_variables["keep_prob"]: KEEP_PROB,
            }

            (
                _,
                loss,
                recon,
                kl,
                reg,
                global_step,
            ) = session.run(
                [
                    model.opt,
                    model.loss,
                    model.topic_loss_recon,
                    model.topic_loss_kl,
                    model.topic_loss_reg,
                    model.global_step,
                ],
                feed_dict=feed_dict,
            )

            batch_documents = dense_bow.shape[0]

            train_loss_sum += (
                float(loss) * batch_documents
            )
            train_recon_sum += (
                float(recon) * batch_documents
            )
            train_kl_sum += (
                float(kl) * batch_documents
            )
            train_reg_sum += (
                float(reg) * batch_documents
            )
            train_document_count += batch_documents

            progress.set_postfix({
                "loss": f"{float(loss):.2f}",
                "step": int(global_step),
            })

        train_metrics = {
            "loss": (
                train_loss_sum / train_document_count
            ),
            "recon": (
                train_recon_sum / train_document_count
            ),
            "kl": (
                train_kl_sum / train_document_count
            ),
            "reg": (
                train_reg_sum / train_document_count
            ),
        }

        evaluate_dev = (
            epoch == 1
            or epoch % DEV_EVAL_INTERVAL == 0
            or epoch == MAX_EPOCHS
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "elapsed_seconds": (
                time.time() - epoch_start
            ),
        }

        if evaluate_dev:
            dev_metrics = evaluate_loss(
                session,
                model,
                train_bow,
                dev_indices,
            )

            epoch_record["dev"] = dev_metrics

            improved = (
                dev_metrics["loss"]
                < best_dev_loss - 1e-6
            )

            if improved:
                best_dev_loss = dev_metrics["loss"]
                early_stop_count = 0

                saver.save(
                    session,
                    str(best_checkpoint_prefix),
                )

                print(
                    f"[BEST] seed={seed} "
                    f"epoch={epoch:03d} | "
                    f"train={train_metrics['loss']:.4f} | "
                    f"DEV={dev_metrics['loss']:.4f}"
                )

            else:
                early_stop_count += 1

                print(
                    f"[EPOCH] seed={seed} "
                    f"epoch={epoch:03d} | "
                    f"train={train_metrics['loss']:.4f} | "
                    f"DEV={dev_metrics['loss']:.4f} | "
                    f"patience={early_stop_count}/"
                    f"{EARLY_STOPPING_PATIENCE}"
                )

        else:
            print(
                f"[EPOCH] seed={seed} "
                f"epoch={epoch:03d} | "
                f"train={train_metrics['loss']:.4f}"
            )

        training_history.append(epoch_record)

        # Save resumable latest checkpoint at every DEV interval.
        if evaluate_dev:
            saver.save(
                session,
                str(latest_checkpoint_prefix),
                global_step=epoch,
            )

            atomic_json_save(
                {
                    "last_epoch": epoch,
                    "best_dev_loss": best_dev_loss,
                    "early_stop_count": early_stop_count,
                    "training_history": training_history,
                },
                state_path,
            )

            try:
                os.sync()
            except Exception:
                pass

        if (
            evaluate_dev
            and early_stop_count
            >= EARLY_STOPPING_PATIENCE
        ):
            stopped_epoch = epoch

            print(
                f"[EARLY STOP] seed={seed}, "
                f"epoch={epoch}, "
                f"best_dev={best_dev_loss:.6f}"
            )
            break

    # Restore unsupervised DEV-selected checkpoint.
    best_index_path = Path(
        str(best_checkpoint_prefix) + ".index"
    )

    if not best_index_path.exists():
        raise RuntimeError(
            f"Best checkpoint not found: "
            f"{best_index_path}"
        )

    saver.restore(
        session,
        str(best_checkpoint_prefix),
    )

    print(
        f"[BEST CHECKPOINT RESTORED] "
        f"{best_checkpoint_prefix}"
    )

    # ------------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------------
    all_node_probabilities = infer_probabilities(
        session,
        model,
        test_bow,
        mc_samples=MC_TEST_SAMPLES,
    )

    if all_node_probabilities.shape != (
        test_bow.shape[0],
        TOTAL_TREE_NODES,
    ):
        raise RuntimeError(
            "Unexpected TSNTM probability shape: "
            f"{all_node_probabilities.shape}"
        )

    # topic_idxs order begins with root topic 0.
    # Drop the root to obtain exactly 30 comparison topics.
    nonroot_probabilities = (
        all_node_probabilities[:, 1:]
    )

    nonroot_denominator = np.maximum(
        nonroot_probabilities.sum(
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    nonroot_theta = (
        nonroot_probabilities
        / nonroot_denominator
    )

    predicted_topics = nonroot_theta.argmax(
        axis=1
    ).astype(np.int16)

    metrics = evaluate_all_levels(
        predicted_topics
    )

    topic_counts = np.bincount(
        predicted_topics,
        minlength=EVAL_TOPICS,
    )

    active_topics = int(
        np.count_nonzero(topic_counts)
    )

    max_topic_share = float(
        topic_counts.max()
        / topic_counts.sum()
    )

    # ------------------------------------------------------------------
    # Topic-word distributions
    # ------------------------------------------------------------------
    topic_word_distribution = session.run(
        model.topic_bow
    ).astype(np.float32)

    nonroot_beta = topic_word_distribution[1:]

    if nonroot_beta.shape != (
        EVAL_TOPICS,
        VOCAB_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected non-root beta shape: "
            f"{nonroot_beta.shape}"
        )

    top_word_rows = []
    top_word_count = 25

    nonroot_topic_ids = model.topic_idxs[1:]

    for evaluation_topic, tree_topic_id in enumerate(
        nonroot_topic_ids
    ):
        top_indices = np.argsort(
            nonroot_beta[evaluation_topic]
        )[::-1][:top_word_count]

        top_words = feature_names[
            top_indices
        ].tolist()

        top_word_rows.append({
            "seed": seed,
            "evaluation_topic": evaluation_topic,
            "tree_topic_id": int(tree_topic_id),
            "tree_depth": int(
                model.tree_depth[tree_topic_id]
            ),
            "top_words": " ".join(top_words),
        })

    pd.DataFrame(top_word_rows).to_csv(
        TOPIC_DIR
        / f"seed_{seed}_top_words.csv",
        index=False,
    )

    np.save(
        ARRAY_DIR
        / f"seed_{seed}_all_node_theta.npy",
        all_node_probabilities,
    )

    np.save(
        ARRAY_DIR
        / f"seed_{seed}_nonroot_theta.npy",
        nonroot_theta.astype(np.float32),
    )

    np.save(
        ARRAY_DIR
        / f"seed_{seed}_nonroot_beta.npy",
        nonroot_beta.astype(np.float32),
    )

    np.save(
        ARRAY_DIR
        / f"seed_{seed}_test_topic_assignments.npy",
        predicted_topics,
    )

    np.save(
        ARRAY_DIR
        / f"seed_{seed}_test_topic_counts.npy",
        topic_counts.astype(np.int32),
    )

    elapsed_seconds = (
        time.time() - seed_start_time
    )

    best_epoch = None

    for record in training_history:
        if "dev" in record:
            if (
                abs(
                    record["dev"]["loss"]
                    - best_dev_loss
                )
                < 1e-6
            ):
                best_epoch = record["epoch"]

    seed_result = {
        "seed": seed,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_dev_loss": best_dev_loss,
        "active_topics": active_topics,
        "max_topic_share": max_topic_share,
        "elapsed_seconds": elapsed_seconds,
        "tree_topic_ids": [
            int(topic_id)
            for topic_id in model.topic_idxs
        ],
        "nonroot_topic_ids": [
            int(topic_id)
            for topic_id in nonroot_topic_ids
        ],
        "all_node_theta_shape": list(
            all_node_probabilities.shape
        ),
        "nonroot_theta_shape": list(
            nonroot_theta.shape
        ),
        "nonroot_beta_shape": list(
            nonroot_beta.shape
        ),
        "checkpoint": str(
            best_checkpoint_prefix
        ),
    }

    atomic_json_save(
        seed_result,
        complete_result_path,
    )

    pd.DataFrame(
        training_history
    ).to_json(
        RESULT_DIR
        / f"seed_{seed}_training_history.json",
        orient="records",
        indent=2,
    )

    all_seed_results.append(seed_result)

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        all_metric_rows.append({
            "seed": seed,
            "level": level,
            **metrics[level],
            "active_topics": active_topics,
            "max_topic_share": max_topic_share,
        })

    print("\n" + "-" * 100)
    print(f"[SEED {seed} RESULT]")

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        print(
            f"{level.capitalize():8s}: "
            f"Pur_p={metrics[level]['pur_p']:.4f} | "
            f"Pur_a={metrics[level]['pur_a']:.4f} | "
            f"NMI={metrics[level]['nmi']:.4f}"
        )

    print(
        f"Active topics={active_topics}/{EVAL_TOPICS} | "
        f"max share={max_topic_share:.2%} | "
        f"best epoch={best_epoch} | "
        f"time={elapsed_seconds / 3600:.2f} h"
    )
    print("-" * 100)

    session.close()

    del model
    del session
    del saver
    del all_node_probabilities
    del nonroot_probabilities
    del nonroot_theta
    del predicted_topics
    del topic_counts
    del topic_word_distribution
    del nonroot_beta

    tf.reset_default_graph()
    gc.collect()

    try:
        os.sync()
    except Exception:
        pass


# ======================================================================================
# 12. Aggregate three seeds
# ======================================================================================

if len(all_seed_results) != len(SEEDS):
    raise RuntimeError(
        f"Expected {len(SEEDS)} seed results, "
        f"found {len(all_seed_results)}"
    )

seed_result_df = pd.DataFrame(
    all_metric_rows
)

seed_result_df.to_csv(
    RESULT_DIR
    / "tsntm_seed_level_results.csv",
    index=False,
)

summary_rows = []

for level in [
    "section",
    "class",
    "subclass",
]:
    level_df = seed_result_df[
        seed_result_df["level"] == level
    ]

    summary = {
        "level": level,
    }

    for metric in [
        "pur_p",
        "pur_a",
        "nmi",
    ]:
        values = level_df[
            metric
        ].to_numpy(dtype=float)

        summary[f"{metric}_mean"] = float(
            values.mean()
        )

        summary[f"{metric}_std"] = float(
            values.std(ddof=1)
        )

        summary[f"{metric}_min"] = float(
            values.min()
        )

        summary[f"{metric}_max"] = float(
            values.max()
        )

    summary_rows.append(summary)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    RESULT_DIR
    / "tsntm_3seed_summary.csv",
    index=False,
)

final_results = {
    "configuration": CONFIGURATION,
    "seed_results": all_seed_results,
    "summary": summary_rows,
    "total_elapsed_seconds": (
        time.time() - total_start_time
    ),
}

atomic_json_save(
    final_results,
    RESULT_DIR
    / "tsntm_3seed_complete_results.json",
)


# ======================================================================================
# 13. Final report
# ======================================================================================

def get_summary_value(
    level,
    metric,
    statistic="mean",
):
    row = summary_df[
        summary_df["level"] == level
    ].iloc[0]

    return float(
        row[f"{metric}_{statistic}"]
    )


print("\n" + "=" * 100)
print("TSNTM STATIC 5×5 — 3-SEED FINAL CPC ALIGNMENT")
print("CPC evaluation: 30 non-root tree topics")
print("=" * 100)

for display_name, level in [
    ("SECTION", "section"),
    ("CLASS", "class"),
    ("SUBCLASS", "subclass"),
]:
    print(f"\n[{display_name}]")

    for metric_display, metric in [
        ("Pur_p", "pur_p"),
        ("Pur_a", "pur_a"),
        ("NMI", "nmi"),
    ]:
        print(
            f"{metric_display:5s} = "
            f"{get_summary_value(level, metric):.4f} "
            f"± "
            f"{get_summary_value(level, metric, 'std'):.4f}"
        )

latex_mean_values = []

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
        latex_mean_values.append(
            get_summary_value(
                level,
                metric,
            )
        )

latex_mean_row = (
    r"TSNTM (static $5{\times}5$) & "
    + " & ".join(
        f"{value:.4f}"
        for value in latex_mean_values
    )
    + r" \\"
)

latex_std_row = (
    r"TSNTM (static $5{\times}5$) & "
    + " & ".join(
        f"${get_summary_value(level, metric):.4f}"
        f"\\pm"
        f"{get_summary_value(level, metric, 'std'):.4f}$"
        for level in [
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

with open(
    RESULT_DIR / "latex_row_mean_only.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_mean_row + "\n")

with open(
    RESULT_DIR / "latex_row_mean_std.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_std_row + "\n")

print("\n[LATEX ROW — MEAN ONLY]")
print(latex_mean_row)

print("\n[LATEX ROW — MEAN ± STD]")
print(latex_std_row)

print("\n[PER-SEED RESULTS]")
display(seed_result_df)

print("\n[3-SEED SUMMARY]")
display(summary_df)

print("\n" + "=" * 100)
print("[COMPLETE]")
print(
    f"Total elapsed: "
    f"{(time.time() - total_start_time) / 3600:.2f} hours"
)
print(f"Result directory: {RESULT_DIR}")
print(
    "Final JSON: "
    f"{RESULT_DIR / 'tsntm_3seed_complete_results.json'}"
)
print(
    "Summary CSV: "
    f"{RESULT_DIR / 'tsntm_3seed_summary.csv'}"
)
print("=" * 100)


# ======================================================================================
# 14. Final Drive flush and automatic Colab runtime release
# 정상적으로 COMPLETE까지 도달한 경우에만 실행됨
# ======================================================================================

print("\n" + "=" * 100)
print("[AUTO SHUTDOWN] TSNTM 3-seed 실험이 완료되었습니다.")
print("[AUTO SHUTDOWN] Google Drive 저장을 최종 동기화합니다.")
print("=" * 100)

gc.collect()

try:
    os.sync()
except Exception as error:
    print(f"[WARNING] os.sync(): {error}")

print(f"[AUTO SHUTDOWN] Result directory: {RESULT_DIR}")
print("[AUTO SHUTDOWN] 저장 안정화를 위해 60초 대기합니다.")

for remaining_seconds in range(60, 0, -10):
    print(
        f"  Runtime termination in "
        f"{remaining_seconds} seconds..."
    )
    time.sleep(10)

try:
    os.sync()
except Exception:
    pass

print("[AUTO SHUTDOWN] Colab T4 runtime을 해제합니다.")

from google.colab import runtime
runtime.unassign()
