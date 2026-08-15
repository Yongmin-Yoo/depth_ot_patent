# ============================================================
# FASTopic 1.0.1 NumPy -> Torch transform compatibility patch
# ============================================================
import numpy as np
import torch
from fastopic import FASTopic

if not getattr(FASTopic.transform, "_numpy_tensor_patch", False):
    _original_fastopic_transform = FASTopic.transform

    def _patched_fastopic_transform(
        self,
        docs=None,
        doc_embeddings=None,
        *args,
        **kwargs,
    ):
        if isinstance(doc_embeddings, np.ndarray):
            try:
                target_device = next(self.model.parameters()).device
            except Exception:
                target_device = torch.device(
                    getattr(
                        self,
                        "device",
                        "cuda" if torch.cuda.is_available() else "cpu",
                    )
                )

            doc_embeddings = torch.tensor(
                np.asarray(doc_embeddings),
                dtype=torch.float32,
                device=target_device,
            )

        return _original_fastopic_transform(
            self,
            docs=docs,
            doc_embeddings=doc_embeddings,
            *args,
            **kwargs,
        )

    _patched_fastopic_transform._numpy_tensor_patch = True
    FASTopic.transform = _patched_fastopic_transform

    print("[PATCH APPLIED] FASTopic.transform: NumPy -> Torch Tensor")
else:
    print("[PATCH ALREADY ACTIVE]")

print("CUDA available:", torch.cuda.is_available())
print("Patch status:", getattr(FASTopic.transform, "_numpy_tensor_patch", False))


# ======================================================================================
# FASTopic FAST 3-SEED PATENT CPC BENCHMARK — ONE-CELL COLAB / NVIDIA T4
#
# Speed-oriented but still reasonable configuration:
#   Seeds      : 42, 43, 44
#   Topics     : 30
#   Epochs     : 50
#   LR         : 0.01
#   Train only : FASTopic fitting
#   Test only  : CPC evaluation
#
# Important:
#   이 설정은 기존 200 epochs / LR 0.002 실험과 다른 새로운 FASTopic 설정입니다.
# ======================================================================================


# ======================================================================================
# 0. USER SETTINGS
# ======================================================================================

PROJECT_ROOT = (
    "/content/drive/MyDrive/depth_ot_patent"
)

TRAIN_RECORD_PATH = (
    f"{PROJECT_ROOT}/data/processed/train_records.pkl"
)

TEST_RECORD_PATH = (
    f"{PROJECT_ROOT}/data/processed/test_records.pkl"
)

RESULT_BASE_DIR = (
    f"{PROJECT_ROOT}/results/baselines"
)

# 3-seed benchmark
SEEDS = [42, 43, 44]

NUM_TOPICS = 30
VOCAB_SIZE = 8000
MIN_DOC_COUNT = 10
MAX_DOC_FREQ = 0.70

# 빠른 공식 권장 계열 설정
FAST_TOPIC_EPOCHS = 50
FAST_TOPIC_LR = 0.01

EMBED_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# T4에서 먼저 512로 시도하고 OOM이면 자동으로 절반으로 줄임
INITIAL_EMBED_BATCH_SIZE = 512
MINIMUM_EMBED_BATCH_SIZE = 64

NORMALIZE_EMBEDDINGS = False

# FASTopic mode
#   "auto"       : 예상 dense BOW가 충분히 작으면 full-memory
#   "full"       : 무조건 full-memory
#   "low_memory" : 무조건 mini-batch
TRAINING_MODE = "auto"

LOW_MEMORY_BATCH_SIZE = 16_384

# full-memory 모드 허용 예상 dense BOW 크기
# T4 15GB와 Colab CPU RAM을 모두 고려하여 보수적으로 5 GiB 사용
MAX_ESTIMATED_DENSE_BOW_GIB_FOR_FULL_MODE = 5.0

# 모델 zip 저장은 속도에 큰 영향은 없지만 필요 없으면 False
SAVE_MODELS = True

# train theta는 최종 CPC 표에는 필요하지 않음
# Drive 저장 시간을 줄이려면 False 권장
SAVE_TRAIN_THETA = False

EXPECTED_TEST_PATENTS = 9_881
EXPECTED_SECTION_LABELS = 9
EXPECTED_CLASS_LABELS = 121
EXPECTED_SUBCLASS_LABELS = 466

DEVICE = "cuda"

# 동일한 record 파일과 embedding model이면 Drive cache 재사용
EMBED_CACHE_DIR = (
    f"{PROJECT_ROOT}/data/cache/fastopic_embeddings"
)


# ======================================================================================
# 1. ENVIRONMENT
# ======================================================================================

import os
import sys
import gc
import re
import json
import time
import pickle
import random
import hashlib
import warnings
import subprocess
import importlib
import importlib.metadata
from pathlib import Path
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

warnings.filterwarnings("ignore")


# ======================================================================================
# 2. INSTALL REQUIRED PACKAGES
# ======================================================================================

def package_version(package_name):
    try:
        return importlib.metadata.version(
            package_name
        )
    except Exception:
        return None


packages_to_install = []

if package_version("fastopic") != "1.0.1":
    packages_to_install.append(
        "fastopic==1.0.1"
    )

required_imports = [
    ("topmost", "topmost"),
    (
        "sentence_transformers",
        "sentence-transformers",
    ),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
]

for module_name, pip_name in required_imports:
    try:
        importlib.import_module(
            module_name
        )
    except Exception:
        packages_to_install.append(
            pip_name
        )

if packages_to_install:
    print(
        "[INSTALL] "
        + " ".join(packages_to_install)
    )

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        *packages_to_install,
    ])


# ======================================================================================
# 3. IMPORTS
# ======================================================================================

import numpy as np
import pandas as pd
import torch

from google.colab import drive
from IPython.display import display

from sklearn.metrics import (
    normalized_mutual_info_score,
)

from sentence_transformers import (
    SentenceTransformer,
)

from fastopic import FASTopic
from topmost import Preprocess


# ======================================================================================
# 4. GOOGLE DRIVE AND PATH VALIDATION
# ======================================================================================

drive.mount(
    "/content/drive",
    force_remount=False,
)

PROJECT_ROOT = Path(
    PROJECT_ROOT
)

TRAIN_RECORD_PATH = Path(
    TRAIN_RECORD_PATH
)

TEST_RECORD_PATH = Path(
    TEST_RECORD_PATH
)

RESULT_BASE_DIR = Path(
    RESULT_BASE_DIR
)

EMBED_CACHE_DIR = Path(
    EMBED_CACHE_DIR
)

if not PROJECT_ROOT.is_dir():
    raise FileNotFoundError(
        f"Project root not found: "
        f"{PROJECT_ROOT}"
    )

if not TRAIN_RECORD_PATH.is_file():
    raise FileNotFoundError(
        f"Train records not found: "
        f"{TRAIN_RECORD_PATH}"
    )

if not TEST_RECORD_PATH.is_file():
    raise FileNotFoundError(
        f"Test records not found: "
        f"{TEST_RECORD_PATH}"
    )

if TRAIN_RECORD_PATH.stat().st_size == 0:
    raise RuntimeError(
        f"Empty train records: "
        f"{TRAIN_RECORD_PATH}"
    )

if TEST_RECORD_PATH.stat().st_size == 0:
    raise RuntimeError(
        f"Empty test records: "
        f"{TEST_RECORD_PATH}"
    )

RUN_TIMESTAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

RESULT_DIR = (
    RESULT_BASE_DIR
    / (
        "fastopic_fast_k30_"
        f"seeds_42_43_44_{RUN_TIMESTAMP}"
    )
)

MODEL_DIR = RESULT_DIR / "models"
ARRAY_DIR = RESULT_DIR / "arrays"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARRAY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMBED_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================================
# 5. GPU AND REPRODUCIBILITY
# ======================================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. "
        "Colab 런타임을 NVIDIA T4로 설정하세요."
    )

GPU_NAME = torch.cuda.get_device_name(0)

GPU_MEMORY_GIB = (
    torch.cuda.get_device_properties(0)
    .total_memory
    / 1024**3
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if hasattr(
    torch.backends.cuda.matmul,
    "allow_tf32",
):
    torch.backends.cuda.matmul.allow_tf32 = False

if hasattr(
    torch.backends.cudnn,
    "allow_tf32",
):
    torch.backends.cudnn.allow_tf32 = False

try:
    torch.set_float32_matmul_precision(
        "highest"
    )
except Exception:
    pass


def set_seed(seed):
    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


set_seed(42)

print("=" * 100)
print(
    "FASTopic FAST 3-SEED CPC BENCHMARK"
)
print("=" * 100)
print(f"GPU                  : {GPU_NAME}")
print(f"GPU memory           : {GPU_MEMORY_GIB:.2f} GiB")
print(f"PyTorch              : {torch.__version__}")
print(f"FASTopic             : {package_version('fastopic')}")
print(f"TopMost              : {package_version('topmost')}")
print(f"Seeds                : {SEEDS}")
print(f"Topics               : {NUM_TOPICS}")
print(f"Vocabulary           : {VOCAB_SIZE:,}")
print(f"Training epochs      : {FAST_TOPIC_EPOCHS}")
print(f"Learning rate        : {FAST_TOPIC_LR}")
print(f"Requested mode       : {TRAINING_MODE}")
print(f"Low-memory batch     : {LOW_MEMORY_BATCH_SIZE:,}")
print(f"Embedding model      : {EMBED_MODEL}")
print(f"Embedding batch      : {INITIAL_EMBED_BATCH_SIZE}")
print(f"Normalize embeddings : {NORMALIZE_EMBEDDINGS}")
print(f"Train records        : {TRAIN_RECORD_PATH}")
print(f"Test records         : {TEST_RECORD_PATH}")
print(f"Embedding cache      : {EMBED_CACHE_DIR}")
print(f"Result directory     : {RESULT_DIR}")
print("=" * 100)

if "T4" not in GPU_NAME.upper():
    print(
        f"[WARNING] 현재 GPU는 T4가 아닙니다: "
        f"{GPU_NAME}"
    )


# ======================================================================================
# 6. LOAD RECORDS
# ======================================================================================

def load_pickle_records(path):
    with open(
        path,
        "rb",
    ) as file:
        obj = pickle.load(
            file
        )

    if isinstance(obj, list):
        records = obj

    elif isinstance(obj, tuple):
        records = list(obj)

    elif isinstance(obj, dict):
        records = None

        for key in [
            "records",
            "data",
            "items",
            "patents",
        ]:
            if (
                key in obj
                and isinstance(
                    obj[key],
                    (list, tuple),
                )
            ):
                records = list(
                    obj[key]
                )
                break

        if records is None:
            if all(
                isinstance(value, dict)
                for value in obj.values()
            ):
                records = list(
                    obj.values()
                )
            else:
                raise TypeError(
                    f"{path.name} dictionary "
                    "structure is unsupported."
                )

    else:
        raise TypeError(
            f"Unsupported pickle type: "
            f"{type(obj)}"
        )

    if not records:
        raise RuntimeError(
            f"No records found: {path}"
        )

    if not isinstance(
        records[0],
        dict,
    ):
        raise TypeError(
            "Each record must be a dictionary."
        )

    return records


print("\n[RECORDS] Loading train records...")

train_records = load_pickle_records(
    TRAIN_RECORD_PATH
)

print("[RECORDS] Loading test records...")

test_records = load_pickle_records(
    TEST_RECORD_PATH
)

print(
    f"[RECORDS] Train patents: "
    f"{len(train_records):,}"
)
print(
    f"[RECORDS] Test patents : "
    f"{len(test_records):,}"
)
print(
    f"[SAMPLE KEYS] "
    f"{list(test_records[0].keys())}"
)

if (
    len(test_records)
    != EXPECTED_TEST_PATENTS
):
    raise RuntimeError(
        "Unexpected test patent count.\n"
        f"Expected: {EXPECTED_TEST_PATENTS:,}\n"
        f"Found   : {len(test_records):,}"
    )


# ======================================================================================
# 7. PATENT TEXT EXTRACTION
# ======================================================================================

CLAIM_CONTAINER_KEYS = [
    "claims",
    "claim_texts",
    "claims_text",
    "claim_records",
    "claim_list",
]

CLAIM_TEXT_KEYS = [
    "text",
    "claim_text",
    "clean_text",
    "normalized_text",
    "raw_text",
    "content",
    "sentence",
]

PATENT_TEXT_KEYS = [
    "patent_text",
    "document",
    "doc_text",
    "full_text",
    "text",
    "content",
]


def scalar_to_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(
        value,
        (int, float),
    ):
        return str(value)

    if isinstance(
        value,
        (list, tuple),
    ):
        if (
            value
            and all(
                isinstance(item, str)
                for item in value
            )
        ):
            return " ".join(
                item.strip()
                for item in value
                if item.strip()
            )

    return ""


def extract_claim_item_text(item):
    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        for key in CLAIM_TEXT_KEYS:
            if key in item:
                text = scalar_to_text(
                    item[key]
                )

                if text:
                    return text

        for key in [
            "tokens",
            "words",
        ]:
            if (
                key in item
                and isinstance(
                    item[key],
                    (list, tuple),
                )
            ):
                return " ".join(
                    str(token)
                    for token in item[key]
                )

    return ""


def extract_patent_text(record):
    for key in CLAIM_CONTAINER_KEYS:
        if key not in record:
            continue

        claims_object = record[key]
        claim_texts = []

        if isinstance(
            claims_object,
            (list, tuple),
        ):
            for claim in claims_object:
                text = extract_claim_item_text(
                    claim
                )

                if text:
                    claim_texts.append(
                        text
                    )

        elif isinstance(
            claims_object,
            str,
        ):
            if claims_object.strip():
                claim_texts.append(
                    claims_object.strip()
                )

        elif isinstance(
            claims_object,
            dict,
        ):
            for claim in claims_object.values():
                text = extract_claim_item_text(
                    claim
                )

                if text:
                    claim_texts.append(
                        text
                    )

        if claim_texts:
            return " ".join(
                claim_texts
            )

    for key in PATENT_TEXT_KEYS:
        if key in record:
            text = scalar_to_text(
                record[key]
            )

            if text:
                return text

    fallback_parts = []

    for key in [
        "title",
        "abstract",
        "summary",
    ]:
        if key in record:
            text = scalar_to_text(
                record[key]
            )

            if text:
                fallback_parts.append(
                    text
                )

    return " ".join(
        fallback_parts
    ).strip()


def extract_patent_id(
    record,
    fallback_index,
):
    for key in [
        "patent_id",
        "publication_number",
        "patent_number",
        "document_id",
        "doc_id",
        "id",
    ]:
        if (
            key in record
            and record[key] is not None
        ):
            return str(
                record[key]
            )

    return (
        f"record_{fallback_index:08d}"
    )


print("\n[TEXT] Extracting train documents...")

text_start = time.time()

train_docs = [
    extract_patent_text(record)
    for record in train_records
]

print("[TEXT] Extracting test documents...")

test_docs = [
    extract_patent_text(record)
    for record in test_records
]

train_ids = [
    extract_patent_id(
        record,
        index,
    )
    for index, record
    in enumerate(train_records)
]

test_ids = [
    extract_patent_id(
        record,
        index,
    )
    for index, record
    in enumerate(test_records)
]

empty_train_indices = [
    index
    for index, text
    in enumerate(train_docs)
    if not text.strip()
]

empty_test_indices = [
    index
    for index, text
    in enumerate(test_docs)
    if not text.strip()
]

if empty_train_indices:
    raise RuntimeError(
        f"Empty train documents: "
        f"{len(empty_train_indices):,}. "
        f"First indices: "
        f"{empty_train_indices[:10]}"
    )

if empty_test_indices:
    raise RuntimeError(
        f"Empty test documents: "
        f"{len(empty_test_indices):,}. "
        f"First indices: "
        f"{empty_test_indices[:10]}"
    )

train_word_lengths = np.asarray([
    len(text.split())
    for text in train_docs
])

test_word_lengths = np.asarray([
    len(text.split())
    for text in test_docs
])

print(
    f"[TRAIN TEXT] "
    f"mean={train_word_lengths.mean():.1f}, "
    f"median={np.median(train_word_lengths):.1f}, "
    f"max={train_word_lengths.max():,}"
)

print(
    f"[TEST TEXT]  "
    f"mean={test_word_lengths.mean():.1f}, "
    f"median={np.median(test_word_lengths):.1f}, "
    f"max={test_word_lengths.max():,}"
)

print(
    f"[TEXT TIME] "
    f"{time.time() - text_start:.1f} seconds"
)


# ======================================================================================
# 8. CPC LABEL EXTRACTION
# ======================================================================================

PRIMARY_CPC_KEYS = [
    "primary_cpc",
    "main_cpc",
    "first_cpc",
    "cpc_primary",
    "primary_cpc_code",
]

GENERAL_CPC_KEYS = [
    "cpc_codes",
    "cpc",
    "cpcs",
    "cpc_labels",
    "cpc_subclasses",
    "classifications",
    "classification",
]

CPC_PATTERN = re.compile(
    r"\b([A-HY]\d{2}[A-Z])"
    r"(?:\d+)?(?:/\d+)?\b",
    re.I,
)


def flatten_values(value):
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(
        value,
        (int, float),
    ):
        return [str(value)]

    if isinstance(
        value,
        (list, tuple),
    ):
        output = []

        for item in value:
            output.extend(
                flatten_values(item)
            )

        return output

    if isinstance(value, dict):
        output = []

        for key in [
            "code",
            "cpc",
            "symbol",
            "classification",
            "subclass",
            "label",
            "value",
        ]:
            if key in value:
                output.extend(
                    flatten_values(
                        value[key]
                    )
                )

        if output:
            return output

        for item in value.values():
            output.extend(
                flatten_values(item)
            )

        return output

    return [str(value)]


def find_cpc_code_in_value(value):
    for item in flatten_values(
        value
    ):
        matches = CPC_PATTERN.findall(
            str(item).upper()
        )

        if matches:
            return matches[0].upper()

    return None


def extract_primary_cpc(record):
    for key in PRIMARY_CPC_KEYS:
        if key in record:
            code = find_cpc_code_in_value(
                record[key]
            )

            if code:
                return code

    for key in GENERAL_CPC_KEYS:
        if key in record:
            code = find_cpc_code_in_value(
                record[key]
            )

            if code:
                return code

    for key, value in record.items():
        if "cpc" in str(key).lower():
            code = find_cpc_code_in_value(
                value
            )

            if code:
                return code

    return None


test_primary_cpc = [
    extract_primary_cpc(record)
    for record in test_records
]

valid_cpc_mask = np.asarray([
    code is not None
    for code in test_primary_cpc
])

section_labels = np.asarray([
    code[0] if code else ""
    for code in test_primary_cpc
])

class_labels = np.asarray([
    code[:3] if code else ""
    for code in test_primary_cpc
])

subclass_labels = np.asarray([
    code[:4] if code else ""
    for code in test_primary_cpc
])

number_of_valid_cpc = int(
    valid_cpc_mask.sum()
)

number_of_sections = len(
    set(
        section_labels[
            valid_cpc_mask
        ]
    )
)

number_of_classes = len(
    set(
        class_labels[
            valid_cpc_mask
        ]
    )
)

number_of_subclasses = len(
    set(
        subclass_labels[
            valid_cpc_mask
        ]
    )
)

print("\n[CPC]")
print(
    f"Valid labels : "
    f"{number_of_valid_cpc:,}"
)
print(
    f"Missing      : "
    f"{len(test_records) - number_of_valid_cpc:,}"
)
print(
    f"Section      : "
    f"{number_of_sections}"
)
print(
    f"Class        : "
    f"{number_of_classes}"
)
print(
    f"Subclass     : "
    f"{number_of_subclasses}"
)

if (
    number_of_sections
    != EXPECTED_SECTION_LABELS
    or number_of_classes
    != EXPECTED_CLASS_LABELS
    or number_of_subclasses
    != EXPECTED_SUBCLASS_LABELS
):
    raise RuntimeError(
        "CPC label count mismatch.\n"
        f"Expected: "
        f"{EXPECTED_SECTION_LABELS}/"
        f"{EXPECTED_CLASS_LABELS}/"
        f"{EXPECTED_SUBCLASS_LABELS}\n"
        f"Found: "
        f"{number_of_sections}/"
        f"{number_of_classes}/"
        f"{number_of_subclasses}"
    )


# ======================================================================================
# 9. RECORD FILE SIGNATURES FOR EMBEDDING CACHE
# ======================================================================================

def file_sha256(
    path,
    chunk_size=8 * 1024 * 1024,
):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
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


print("\n[CACHE] Computing record signatures...")

train_record_hash = file_sha256(
    TRAIN_RECORD_PATH
)[:16]

test_record_hash = file_sha256(
    TEST_RECORD_PATH
)[:16]

safe_embedding_model_name = re.sub(
    r"[^A-Za-z0-9_.-]+",
    "_",
    EMBED_MODEL,
)

normalization_tag = (
    "normalized"
    if NORMALIZE_EMBEDDINGS
    else "raw"
)

TRAIN_EMBED_PATH = (
    EMBED_CACHE_DIR
    / (
        f"train_{safe_embedding_model_name}_"
        f"{normalization_tag}_"
        f"{train_record_hash}.npy"
    )
)

TEST_EMBED_PATH = (
    EMBED_CACHE_DIR
    / (
        f"test_{safe_embedding_model_name}_"
        f"{normalization_tag}_"
        f"{test_record_hash}.npy"
    )
)

print(
    f"Train embedding cache: "
    f"{TRAIN_EMBED_PATH}"
)
print(
    f"Test embedding cache : "
    f"{TEST_EMBED_PATH}"
)


# ======================================================================================
# 10. EMBEDDING GENERATION WITH AUTOMATIC OOM FALLBACK
# ======================================================================================

embedding_encoder = None


def get_embedding_encoder():
    global embedding_encoder

    if embedding_encoder is None:
        print(
            f"[EMBED] Loading "
            f"{EMBED_MODEL}"
        )

        embedding_encoder = (
            SentenceTransformer(
                EMBED_MODEL,
                device=DEVICE,
            )
        )

    return embedding_encoder


def encode_with_batch_fallback(
    docs,
    batch_size,
    split_name,
):
    encoder = get_embedding_encoder()

    current_batch_size = int(
        batch_size
    )

    while (
        current_batch_size
        >= MINIMUM_EMBED_BATCH_SIZE
    ):
        try:
            print(
                f"[EMBED] {split_name}: "
                f"{len(docs):,} documents, "
                f"batch={current_batch_size}"
            )

            embeddings = encoder.encode(
                docs,
                batch_size=current_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=(
                    NORMALIZE_EMBEDDINGS
                ),
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            return (
                embeddings,
                current_batch_size,
            )

        except torch.cuda.OutOfMemoryError:
            print(
                f"[EMBED OOM] "
                f"batch={current_batch_size}"
            )

            clear_memory()

            current_batch_size //= 2

    raise RuntimeError(
        "Embedding generation failed even "
        f"with batch={MINIMUM_EMBED_BATCH_SIZE}."
    )


def load_or_create_embeddings(
    docs,
    output_path,
    split_name,
):
    if output_path.is_file():
        embeddings = np.load(
            output_path,
            mmap_mode=None,
        )

        if (
            embeddings.ndim == 2
            and embeddings.shape[0]
            == len(docs)
        ):
            print(
                f"[EMBED CACHE HIT] "
                f"{split_name}: "
                f"shape={embeddings.shape}"
            )

            return np.asarray(
                embeddings,
                dtype=np.float32,
            )

        print(
            f"[EMBED CACHE INVALID] "
            f"{output_path}"
        )

        output_path.unlink(
            missing_ok=True
        )

    embeddings, used_batch_size = (
        encode_with_batch_fallback(
            docs,
            INITIAL_EMBED_BATCH_SIZE,
            split_name,
        )
    )

    temporary_path = (
        output_path.with_suffix(
            ".tmp.npy"
        )
    )

    np.save(
        temporary_path,
        embeddings,
    )

    os.replace(
        temporary_path,
        output_path,
    )

    print(
        f"[EMBED SAVED] "
        f"{output_path}"
    )
    print(
        f"[EMBED INFO] "
        f"shape={embeddings.shape}, "
        f"batch={used_batch_size}"
    )

    return embeddings


embedding_start = time.time()

train_embeddings = (
    load_or_create_embeddings(
        train_docs,
        TRAIN_EMBED_PATH,
        "TRAIN",
    )
)

test_embeddings = (
    load_or_create_embeddings(
        test_docs,
        TEST_EMBED_PATH,
        "TEST",
    )
)

if embedding_encoder is not None:
    del embedding_encoder
    embedding_encoder = None
    clear_memory()

embedding_seconds = (
    time.time()
    - embedding_start
)

if (
    train_embeddings.ndim != 2
    or test_embeddings.ndim != 2
):
    raise RuntimeError(
        "Embedding arrays must be 2D."
    )

if (
    train_embeddings.shape[1]
    != test_embeddings.shape[1]
):
    raise RuntimeError(
        "Train/test embedding dimension mismatch."
    )

print(
    f"[EMBEDDINGS] "
    f"Train={train_embeddings.shape}, "
    f"Test={test_embeddings.shape}"
)
print(
    f"[EMBED TIME] "
    f"{embedding_seconds / 60:.2f} minutes"
)


# ======================================================================================
# 11. PREPROCESS TRAIN DOCUMENTS ONCE
# ======================================================================================

print("\n" + "=" * 100)
print(
    "FASTopic PREPROCESSING — ONE TIME ONLY"
)
print("=" * 100)

preprocess_start = time.time()

preprocess_builder = Preprocess(
    vocab_size=VOCAB_SIZE,
    min_doc_count=MIN_DOC_COUNT,
    max_doc_freq=MAX_DOC_FREQ,
    stopwords="English",
    keep_num=False,
    keep_alphanum=False,
    min_length=3,
    seed=42,
    verbose=True,
)

cached_preprocessed_data = (
    preprocess_builder.preprocess(
        train_docs
    )
)

if not isinstance(
    cached_preprocessed_data,
    dict,
):
    raise TypeError(
        "Preprocess.preprocess() must "
        "return a dictionary."
    )

if (
    "train_bow"
    not in cached_preprocessed_data
):
    raise KeyError(
        "train_bow not found in "
        "preprocessing output."
    )

if (
    "vocab"
    not in cached_preprocessed_data
):
    raise KeyError(
        "vocab not found in "
        "preprocessing output."
    )

cached_train_bow = (
    cached_preprocessed_data[
        "train_bow"
    ]
)

cached_vocab = list(
    cached_preprocessed_data[
        "vocab"
    ]
)

if (
    cached_train_bow.shape[0]
    != len(train_docs)
):
    raise RuntimeError(
        "BOW document count mismatch."
    )

if (
    cached_train_bow.shape[1]
    != len(cached_vocab)
):
    raise RuntimeError(
        "BOW vocabulary dimension mismatch."
    )

preprocess_seconds = (
    time.time()
    - preprocess_start
)

print(
    f"[PREPROCESS] BOW shape : "
    f"{cached_train_bow.shape}"
)
print(
    f"[PREPROCESS] Vocabulary: "
    f"{len(cached_vocab):,}"
)
print(
    f"[PREPROCESS] Time      : "
    f"{preprocess_seconds / 60:.2f} minutes"
)


class FixedFASTopicPreprocess:

    def __init__(
        self,
        train_bow,
        vocab,
    ):
        self.train_bow = (
            train_bow
        )
        self.vocab = list(
            vocab
        )

    def preprocess(
        self,
        docs,
    ):
        if (
            len(docs)
            != self.train_bow.shape[0]
        ):
            raise RuntimeError(
                "Document count does not match "
                "cached train BOW."
            )

        return {
            "train_bow": (
                self.train_bow.copy()
            ),
            "vocab": list(
                self.vocab
            ),
        }


fixed_preprocess = (
    FixedFASTopicPreprocess(
        cached_train_bow,
        cached_vocab,
    )
)


# ======================================================================================
# 12. AUTOMATIC TRAINING MODE SELECTION
# ======================================================================================

estimated_dense_bow_bytes = (
    len(train_docs)
    * len(cached_vocab)
    * 4
)

estimated_dense_bow_gib = (
    estimated_dense_bow_bytes
    / 1024**3
)

if TRAINING_MODE == "full":
    active_low_memory = False

elif TRAINING_MODE == "low_memory":
    active_low_memory = True

elif TRAINING_MODE == "auto":
    active_low_memory = (
        estimated_dense_bow_gib
        > MAX_ESTIMATED_DENSE_BOW_GIB_FOR_FULL_MODE
    )

else:
    raise ValueError(
        "TRAINING_MODE must be "
        "'auto', 'full', or 'low_memory'."
    )

print("\n" + "=" * 100)
print("TRAINING MODE")
print("=" * 100)
print(
    f"Estimated dense BOW : "
    f"{estimated_dense_bow_gib:.2f} GiB"
)
print(
    f"Mode                : "
    f"{'LOW-MEMORY' if active_low_memory else 'FULL-MEMORY'}"
)

if active_low_memory:
    print(
        f"Batch size          : "
        f"{LOW_MEMORY_BATCH_SIZE:,}"
    )
else:
    print(
        "Batch size          : "
        "Entire training corpus"
    )

print("=" * 100)


# ======================================================================================
# 13. CPC METRICS
# ======================================================================================

def predicted_cluster_purity(
    true_labels,
    predicted_clusters,
):
    true_labels = np.asarray(
        true_labels
    )

    predicted_clusters = np.asarray(
        predicted_clusters
    )

    total_correct = 0

    for cluster_id in np.unique(
        predicted_clusters
    ):
        mask = (
            predicted_clusters
            == cluster_id
        )

        _, counts = np.unique(
            true_labels[mask],
            return_counts=True,
        )

        total_correct += int(
            counts.max()
        )

    return float(
        total_correct
        / len(true_labels)
    )


def inverse_label_purity(
    true_labels,
    predicted_clusters,
):
    true_labels = np.asarray(
        true_labels
    )

    predicted_clusters = np.asarray(
        predicted_clusters
    )

    total_correct = 0

    for label in np.unique(
        true_labels
    ):
        mask = (
            true_labels == label
        )

        _, counts = np.unique(
            predicted_clusters[mask],
            return_counts=True,
        )

        total_correct += int(
            counts.max()
        )

    return float(
        total_correct
        / len(true_labels)
    )


def evaluate_level(
    true_labels,
    predicted_clusters,
):
    return {
        "pur_p": (
            predicted_cluster_purity(
                true_labels,
                predicted_clusters,
            )
        ),
        "pur_a": (
            inverse_label_purity(
                true_labels,
                predicted_clusters,
            )
        ),
        "nmi": float(
            normalized_mutual_info_score(
                true_labels,
                predicted_clusters,
                average_method="arithmetic",
            )
        ),
    }


def evaluate_all_cpc_levels(
    predicted_clusters,
):
    predictions = np.asarray(
        predicted_clusters
    )[valid_cpc_mask]

    return {
        "section": evaluate_level(
            section_labels[
                valid_cpc_mask
            ],
            predictions,
        ),
        "class": evaluate_level(
            class_labels[
                valid_cpc_mask
            ],
            predictions,
        ),
        "subclass": evaluate_level(
            subclass_labels[
                valid_cpc_mask
            ],
            predictions,
        ),
    }


# ======================================================================================
# 14. SAVE CONFIGURATION AND ENVIRONMENT
# ======================================================================================

package_versions = {}

for package_name in [
    "fastopic",
    "topmost",
    "sentence-transformers",
    "transformers",
    "scikit-learn",
    "numpy",
    "pandas",
    "torch",
]:
    package_versions[
        package_name
    ] = package_version(
        package_name
    )

package_versions[
    "cuda"
] = torch.version.cuda

package_versions[
    "gpu"
] = GPU_NAME

configuration = {
    "created_at": (
        datetime.now().isoformat()
    ),
    "project_root": str(
        PROJECT_ROOT
    ),
    "train_record_path": str(
        TRAIN_RECORD_PATH
    ),
    "test_record_path": str(
        TEST_RECORD_PATH
    ),
    "train_record_sha256": (
        train_record_hash
    ),
    "test_record_sha256": (
        test_record_hash
    ),
    "train_patents": len(
        train_docs
    ),
    "test_patents": len(
        test_docs
    ),
    "valid_test_cpc_labels": (
        number_of_valid_cpc
    ),
    "seeds": SEEDS,
    "num_topics": (
        NUM_TOPICS
    ),
    "vocab_size_requested": (
        VOCAB_SIZE
    ),
    "vocab_size_actual": len(
        cached_vocab
    ),
    "min_doc_count": (
        MIN_DOC_COUNT
    ),
    "max_doc_freq": (
        MAX_DOC_FREQ
    ),
    "epochs": (
        FAST_TOPIC_EPOCHS
    ),
    "learning_rate": (
        FAST_TOPIC_LR
    ),
    "requested_training_mode": (
        TRAINING_MODE
    ),
    "actual_low_memory": (
        active_low_memory
    ),
    "low_memory_batch_size": (
        LOW_MEMORY_BATCH_SIZE
        if active_low_memory
        else None
    ),
    "estimated_dense_bow_gib": (
        estimated_dense_bow_gib
    ),
    "embedding_model": (
        EMBED_MODEL
    ),
    "embedding_dimension": int(
        train_embeddings.shape[1]
    ),
    "normalize_embeddings": (
        NORMALIZE_EMBEDDINGS
    ),
    "embedding_cache_dir": str(
        EMBED_CACHE_DIR
    ),
    "device": (
        DEVICE
    ),
    "gpu": (
        GPU_NAME
    ),
    "gpu_memory_gib": (
        GPU_MEMORY_GIB
    ),
    "document_unit": (
        "patent"
    ),
    "document_text": (
        "concatenated claims"
    ),
    "cpc_rule": (
        "primary CPC if available; "
        "otherwise first CPC"
    ),
    "cpc_used_in_training": False,
    "test_used_for_training": False,
    "preprocessing_reused_across_seeds": True,
    "save_models": (
        SAVE_MODELS
    ),
    "save_train_theta": (
        SAVE_TRAIN_THETA
    ),
    "package_versions": (
        package_versions
    ),
}

with open(
    RESULT_DIR / "configuration.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        configuration,
        file,
        indent=2,
        ensure_ascii=False,
    )

with open(
    RESULT_DIR / "package_versions.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        package_versions,
        file,
        indent=2,
        ensure_ascii=False,
    )

label_audit = pd.DataFrame({
    "patent_id": (
        test_ids
    ),
    "primary_cpc": (
        test_primary_cpc
    ),
    "section": (
        section_labels
    ),
    "class": (
        class_labels
    ),
    "subclass": (
        subclass_labels
    ),
    "valid_cpc": (
        valid_cpc_mask
    ),
})

label_audit.to_csv(
    RESULT_DIR
    / "test_cpc_label_audit.csv",
    index=False,
)


# ======================================================================================
# 15. FASTopic MODEL FACTORY
# ======================================================================================

def build_fastopic_model(
    use_low_memory,
):
    return FASTopic(
        num_topics=NUM_TOPICS,
        preprocess=fixed_preprocess,
        num_top_words=25,
        device=DEVICE,
        normalize_embeddings=(
            NORMALIZE_EMBEDDINGS
        ),
        doc_embed_model=(
            EMBED_MODEL
        ),
        DT_alpha=3.0,
        TW_alpha=2.0,
        theta_temp=1.0,
        low_memory=(
            use_low_memory
        ),
        low_memory_batch_size=(
            LOW_MEMORY_BATCH_SIZE
            if use_low_memory
            else None
        ),
        verbose=True,
        log_interval=10,
    )


def is_out_of_memory_error(error):
    if isinstance(
        error,
        torch.cuda.OutOfMemoryError,
    ):
        return True

    return (
        "out of memory"
        in str(error).lower()
    )


# ======================================================================================
# 16. TRAIN AND EVALUATE THREE SEEDS
# ======================================================================================

all_seed_results = []
all_rows = []

total_start = time.time()

actual_low_memory = (
    active_low_memory
)

for seed_index, seed in enumerate(
    SEEDS,
    start=1,
):
    print("\n" + "=" * 100)
    print(
        f"FASTopic SEED {seed} — "
        f"{seed_index}/{len(SEEDS)}"
    )
    print("=" * 100)

    set_seed(
        seed
    )
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    seed_start = time.time()

    model = build_fastopic_model(
        actual_low_memory
    )

    print(
        f"[TRAIN] seed={seed}, "
        f"documents={len(train_docs):,}, "
        f"epochs={FAST_TOPIC_EPOCHS}, "
        f"lr={FAST_TOPIC_LR}, "
        f"mode="
        f"{'low_memory' if actual_low_memory else 'full'}"
    )

    torch.set_grad_enabled(
        True
    )

    try:
        with torch.inference_mode(False):
            with torch.enable_grad():
                (
                    top_words,
                    train_theta,
                ) = model.fit_transform(
                    train_docs,
                    epochs=(
                        FAST_TOPIC_EPOCHS
                    ),
                    learning_rate=(
                        FAST_TOPIC_LR
                    ),
                    preset_doc_embeddings=(
                        train_embeddings
                    ),
                )

    except Exception as training_error:
        # 첫 seed의 full-memory OOM이면 자동으로 low-memory로 다시 시작
        if (
            seed_index == 1
            and not actual_low_memory
            and is_out_of_memory_error(
                training_error
            )
        ):
            print(
                "\n[FULL-MEMORY OOM]"
            )
            print(
                "T4 메모리에 전체 BOW가 들어가지 않아 "
                f"low-memory batch "
                f"{LOW_MEMORY_BATCH_SIZE:,}으로 "
                "자동 전환합니다."
            )

            del model
            clear_memory()

            actual_low_memory = True

            configuration[
                "actual_low_memory"
            ] = True

            configuration[
                "low_memory_batch_size"
            ] = LOW_MEMORY_BATCH_SIZE

            configuration[
                "full_memory_oom_fallback"
            ] = True

            with open(
                RESULT_DIR
                / "configuration.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    configuration,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            set_seed(
                seed
            )

            model = build_fastopic_model(
                True
            )

            with torch.inference_mode(False):
                with torch.enable_grad():
                    (
                        top_words,
                        train_theta,
                    ) = model.fit_transform(
                        train_docs,
                        epochs=(
                            FAST_TOPIC_EPOCHS
                        ),
                        learning_rate=(
                            FAST_TOPIC_LR
                        ),
                        preset_doc_embeddings=(
                            train_embeddings
                        ),
                    )

        else:
            raise

    print(
        f"[TEST] Transforming "
        f"{len(test_docs):,} test patents"
    )

    with torch.inference_mode():
        test_theta = model.transform(
            doc_embeddings=(
                test_embeddings
            )
        )

    train_theta = np.asarray(
        train_theta,
        dtype=np.float32,
    )

    test_theta = np.asarray(
        test_theta,
        dtype=np.float32,
    )

    beta = np.asarray(
        model.get_beta(),
        dtype=np.float32,
    )

    if (
        test_theta.shape
        != (
            len(test_docs),
            NUM_TOPICS,
        )
    ):
        raise RuntimeError(
            "Unexpected test theta shape: "
            f"{test_theta.shape}"
        )

    if not np.isfinite(
        test_theta
    ).all():
        raise RuntimeError(
            f"Seed {seed}: "
            "test theta contains NaN/Inf."
        )

    predicted_topics = (
        test_theta.argmax(
            axis=1
        )
    )

    metrics = (
        evaluate_all_cpc_levels(
            predicted_topics
        )
    )

    active_topics = int(
        np.unique(
            predicted_topics
        ).size
    )

    topic_counts = np.bincount(
        predicted_topics,
        minlength=NUM_TOPICS,
    )

    max_topic_share = float(
        topic_counts.max()
        / topic_counts.sum()
    )

    elapsed_seconds = (
        time.time()
        - seed_start
    )

    peak_gpu_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    seed_result = {
        "seed": seed,
        "training_mode": (
            "low_memory"
            if actual_low_memory
            else "full"
        ),
        "metrics": metrics,
        "active_topics": (
            active_topics
        ),
        "max_topic_share": (
            max_topic_share
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "peak_gpu_gib": (
            peak_gpu_gib
        ),
        "train_theta_shape": list(
            train_theta.shape
        ),
        "test_theta_shape": list(
            test_theta.shape
        ),
        "beta_shape": list(
            beta.shape
        ),
    }

    all_seed_results.append(
        seed_result
    )

    for level in [
        "section",
        "class",
        "subclass",
    ]:
        all_rows.append({
            "seed": seed,
            "level": level,
            "pur_p": (
                metrics[level][
                    "pur_p"
                ]
            ),
            "pur_a": (
                metrics[level][
                    "pur_a"
                ]
            ),
            "nmi": (
                metrics[level][
                    "nmi"
                ]
            ),
            "active_topics": (
                active_topics
            ),
            "max_topic_share": (
                max_topic_share
            ),
            "training_mode": (
                "low_memory"
                if actual_low_memory
                else "full"
            ),
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "peak_gpu_gib": (
                peak_gpu_gib
            ),
        })

    # ----------------------------------------------------------------------------------
    # Save arrays
    # ----------------------------------------------------------------------------------

    if SAVE_TRAIN_THETA:
        np.save(
            ARRAY_DIR
            / (
                f"seed_{seed}"
                "_train_theta.npy"
            ),
            train_theta,
        )

    np.save(
        ARRAY_DIR
        / (
            f"seed_{seed}"
            "_test_theta.npy"
        ),
        test_theta,
    )

    np.save(
        ARRAY_DIR
        / (
            f"seed_{seed}"
            "_beta.npy"
        ),
        beta,
    )

    np.save(
        ARRAY_DIR
        / (
            f"seed_{seed}"
            "_test_topic_assignments.npy"
        ),
        predicted_topics.astype(
            np.int16
        ),
    )

    # ----------------------------------------------------------------------------------
    # Save top words
    # ----------------------------------------------------------------------------------

    top_word_rows = []

    for topic_id, words in enumerate(
        top_words
    ):
        top_word_rows.append({
            "seed": seed,
            "topic_id": (
                topic_id
            ),
            "top_words": (
                words
            ),
        })

    pd.DataFrame(
        top_word_rows
    ).to_csv(
        RESULT_DIR
        / f"seed_{seed}_top_words.csv",
        index=False,
    )

    # ----------------------------------------------------------------------------------
    # Save model
    # ----------------------------------------------------------------------------------

    if SAVE_MODELS:
        model_path = (
            MODEL_DIR
            / f"fastopic_seed_{seed}.zip"
        )

        try:
            model.save(
                str(model_path)
            )

            print(
                f"[MODEL SAVED] "
                f"{model_path}"
            )

        except Exception as save_error:
            print(
                "[WARNING] model.save failed: "
                f"{save_error}"
            )

            try:
                torch.save(
                    {
                        "seed": seed,
                        "model_state_dict": (
                            model.model.state_dict()
                        ),
                        "vocab": (
                            model.vocab
                        ),
                        "num_topics": (
                            NUM_TOPICS
                        ),
                        "embedding_model": (
                            EMBED_MODEL
                        ),
                        "configuration": (
                            configuration
                        ),
                    },
                    MODEL_DIR
                    / (
                        f"fastopic_seed_{seed}"
                        "_state.pt"
                    ),
                )

            except Exception as state_error:
                print(
                    "[WARNING] State save failed: "
                    f"{state_error}"
                )

    with open(
        RESULT_DIR
        / f"seed_{seed}_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            seed_result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "-" * 100)
    print(
        f"[SEED {seed} RESULT]"
    )
    print("-" * 100)
    print(
        "Section : "
        f"Pur_p={metrics['section']['pur_p']:.4f} | "
        f"Pur_a={metrics['section']['pur_a']:.4f} | "
        f"NMI={metrics['section']['nmi']:.4f}"
    )
    print(
        "Class   : "
        f"Pur_p={metrics['class']['pur_p']:.4f} | "
        f"Pur_a={metrics['class']['pur_a']:.4f} | "
        f"NMI={metrics['class']['nmi']:.4f}"
    )
    print(
        "Subclass: "
        f"Pur_p={metrics['subclass']['pur_p']:.4f} | "
        f"Pur_a={metrics['subclass']['pur_a']:.4f} | "
        f"NMI={metrics['subclass']['nmi']:.4f}"
    )
    print(
        f"Active topics={active_topics}/{NUM_TOPICS} | "
        f"max share={max_topic_share:.2%}"
    )
    print(
        f"Mode="
        f"{'low_memory' if actual_low_memory else 'full'} | "
        f"time={elapsed_seconds / 60:.2f} min | "
        f"GPU peak={peak_gpu_gib:.2f} GiB"
    )
    print("-" * 100)

    del model
    del train_theta
    del test_theta
    del beta
    del predicted_topics

    clear_memory()


# ======================================================================================
# 17. AGGREGATE THREE-SEED RESULTS
# ======================================================================================

seed_df = pd.DataFrame(
    all_rows
)

seed_df.to_csv(
    RESULT_DIR
    / "fastopic_seed_level_results.csv",
    index=False,
)

summary_rows = []

for level in [
    "section",
    "class",
    "subclass",
]:
    level_df = seed_df[
        seed_df["level"] == level
    ]

    summary_row = {
        "level": level
    }

    for metric in [
        "pur_p",
        "pur_a",
        "nmi",
    ]:
        values = level_df[
            metric
        ].to_numpy(
            dtype=float
        )

        summary_row[
            f"{metric}_mean"
        ] = float(
            values.mean()
        )

        summary_row[
            f"{metric}_std"
        ] = float(
            values.std(
                ddof=1
            )
        )

        summary_row[
            f"{metric}_min"
        ] = float(
            values.min()
        )

        summary_row[
            f"{metric}_max"
        ] = float(
            values.max()
        )

    summary_rows.append(
        summary_row
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    RESULT_DIR
    / "fastopic_3seed_summary.csv",
    index=False,
)

configuration[
    "actual_low_memory"
] = actual_low_memory

configuration[
    "actual_training_mode"
] = (
    "low_memory"
    if actual_low_memory
    else "full"
)

final_result = {
    "configuration": (
        configuration
    ),
    "seed_results": (
        all_seed_results
    ),
    "summary": (
        summary_rows
    ),
    "timing": {
        "embedding_seconds": (
            embedding_seconds
        ),
        "preprocessing_seconds": (
            preprocess_seconds
        ),
        "total_elapsed_seconds": (
            time.time()
            - total_start
        ),
    },
}

with open(
    RESULT_DIR
    / "fastopic_3seed_complete_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_result,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ======================================================================================
# 18. FINAL REPORT
# ======================================================================================

def summary_value(
    level,
    metric,
    statistic="mean",
):
    row = summary_df[
        summary_df["level"] == level
    ].iloc[0]

    return float(
        row[
            f"{metric}_{statistic}"
        ]
    )


print("\n")
print("=" * 100)
print(
    "FASTopic FAST 3-SEED FINAL CPC ALIGNMENT"
)
print("=" * 100)

for level_title, level in [
    ("SECTION", "section"),
    ("CLASS", "class"),
    ("SUBCLASS", "subclass"),
]:
    print(
        f"\n[{level_title}]"
    )

    print(
        "Pur_p = "
        f"{summary_value(level, 'pur_p'):.4f} "
        "± "
        f"{summary_value(level, 'pur_p', 'std'):.4f}"
    )

    print(
        "Pur_a = "
        f"{summary_value(level, 'pur_a'):.4f} "
        "± "
        f"{summary_value(level, 'pur_a', 'std'):.4f}"
    )

    print(
        "NMI   = "
        f"{summary_value(level, 'nmi'):.4f} "
        "± "
        f"{summary_value(level, 'nmi', 'std'):.4f}"
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
        latex_values.append(
            summary_value(
                level,
                metric,
            )
        )

latex_row = (
    "FASTopic "
    + " & ".join(
        f"{value:.4f}"
        for value in latex_values
    )
    + r" \\"
)

latex_row_with_std = (
    "FASTopic "
    + " & ".join(
        (
            f"${summary_value(level, metric):.4f}"
            f"\\pm"
            f"{summary_value(level, metric, 'std'):.4f}$"
        )
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
    RESULT_DIR
    / "latex_row_mean_only.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(
        latex_row + "\n"
    )

with open(
    RESULT_DIR
    / "latex_row_mean_std.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(
        latex_row_with_std + "\n"
    )

print("\n" + "-" * 100)
print("[LATEX ROW — MEAN ONLY]")
print(
    latex_row
)

print("\n[LATEX ROW — MEAN ± STD]")
print(
    latex_row_with_std
)

print("\n[PER-SEED RESULTS]")

display(
    seed_df[[
        "seed",
        "level",
        "pur_p",
        "pur_a",
        "nmi",
        "active_topics",
        "max_topic_share",
        "training_mode",
        "elapsed_seconds",
        "peak_gpu_gib",
    ]]
)

print("\n[3-SEED SUMMARY]")

display(
    summary_df
)


# ======================================================================================
# 19. BASIC RESULT VALIDATION
# ======================================================================================

warnings_found = []

for seed_result in all_seed_results:
    seed = seed_result[
        "seed"
    ]

    active_topics = seed_result[
        "active_topics"
    ]

    max_topic_share = seed_result[
        "max_topic_share"
    ]

    if active_topics < 20:
        warnings_found.append(
            f"Seed {seed}: "
            f"only {active_topics}/{NUM_TOPICS} "
            "topics are active."
        )

    if max_topic_share > 0.40:
        warnings_found.append(
            f"Seed {seed}: "
            f"maximum topic share is "
            f"{max_topic_share:.2%}."
        )

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
            value = seed_result[
                "metrics"
            ][level][metric]

            if not np.isfinite(
                value
            ):
                warnings_found.append(
                    f"Seed {seed}: "
                    f"{level}/{metric} "
                    "is not finite."
                )

print("\n" + "=" * 100)
print("FINAL STATUS")
print("=" * 100)

if warnings_found:
    print(
        "[WARNING] Evaluation completed, "
        "but the following checks require review:"
    )

    for warning in warnings_found:
        print(
            f"  - {warning}"
        )

else:
    print(
        "[PASS] All three seeds completed."
    )
    print(
        "[PASS] No severe topic collapse was detected."
    )

print(
    f"Training mode : "
    f"{'low_memory' if actual_low_memory else 'full'}"
)

print(
    f"Total elapsed : "
    f"{(time.time() - total_start) / 60:.2f} minutes"
)

print(
    f"Result dir    : "
    f"{RESULT_DIR}"
)

print(
    f"Main JSON     : "
    f"{RESULT_DIR / 'fastopic_3seed_complete_results.json'}"
)

print(
    f"Summary CSV   : "
    f"{RESULT_DIR / 'fastopic_3seed_summary.csv'}"
)

print(
    f"Seed CSV      : "
    f"{RESULT_DIR / 'fastopic_seed_level_results.csv'}"
)

print("=" * 100)
