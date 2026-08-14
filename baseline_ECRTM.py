# ======================================================================================
# ECRTM 3-SEED PATENT CPC BENCHMARK
# ONE-CELL GOOGLE COLAB — NVIDIA T4
#
# Model  : ECRTM (ICML 2023, official TopMost implementation)
# Seeds  : 42, 43, 44
# Topics : 30
# Eval   : Patent-level CPC Section / Class / Subclass
# ======================================================================================

# --------------------------------------------------------------------------------------
# 0. Environment and packages
# --------------------------------------------------------------------------------------
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
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

warnings.filterwarnings("ignore")

def install_required_packages():
    packages = [
        ("topmost", "topmost"),
        ("gensim", "gensim"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]

    missing = []

    for module_name, pip_name in packages:
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)

    if missing:
        print("[INSTALL]", " ".join(missing))
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            *missing,
        ])

install_required_packages()

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from tqdm.auto import tqdm
from sklearn.metrics import normalized_mutual_info_score

from topmost import ECRTM
from topmost.preprocess.preprocess import Preprocess

# --------------------------------------------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------------------------------------------
SEEDS = [42, 43, 44]
NUM_TOPICS = 30

VOCAB_SIZE = 8000
MIN_DOC_COUNT = 10
MAX_DOC_FREQ = 0.70

# Official ECRTM-style training
EPOCHS = 500
BATCH_SIZE = 200
LEARNING_RATE = 0.002
LR_STEP_SIZE = 125
LR_GAMMA = 0.5

# Official TopMost ECRTM defaults
ENCODER_UNITS = 200
DROPOUT = 0.0
EMBED_SIZE = 200
BETA_TEMP = 0.2
WEIGHT_LOSS_ECR = 100.0
SINKHORN_ALPHA = 20.0
SINKHORN_MAX_ITER = 1000

CHECKPOINT_INTERVAL = 25
INFERENCE_BATCH_SIZE = 512
NUM_TOP_WORDS = 25

EXPECTED_TEST_PATENTS = 9881
EXPECTED_SECTIONS = 9
EXPECTED_CLASSES = 121
EXPECTED_SUBCLASSES = 466

DEVICE = torch.device("cuda:0")

# --------------------------------------------------------------------------------------
# 2. GPU and reproducibility
# --------------------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. Colab 런타임 유형을 NVIDIA T4 GPU로 설정하세요."
    )

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GIB = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = False

if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = False

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

print("=" * 100)
print("ECRTM 3-SEED CPC BENCHMARK")
print("=" * 100)
print(f"GPU                     : {GPU_NAME}")
print(f"GPU memory              : {GPU_MEMORY_GIB:.2f} GiB")
print(f"PyTorch                 : {torch.__version__}")
print(f"Device                  : {DEVICE}")
print(f"Seeds                   : {SEEDS}")
print(f"Topics                  : {NUM_TOPICS}")
print(f"Vocabulary              : {VOCAB_SIZE:,}")
print(f"Epochs                  : {EPOCHS}")
print(f"Batch size              : {BATCH_SIZE}")
print(f"Learning rate           : {LEARNING_RATE}")
print(f"LR step/gamma           : {LR_STEP_SIZE} / {LR_GAMMA}")
print(f"Encoder units           : {ENCODER_UNITS}")
print(f"Embedding               : GloVe 200d")
print(f"Beta temperature        : {BETA_TEMP}")
print(f"ECR weight              : {WEIGHT_LOSS_ECR}")
print(f"Sinkhorn alpha          : {SINKHORN_ALPHA}")
print(f"Sinkhorn max iterations : {SINKHORN_MAX_ITER}")
print(f"Mixed precision         : False")
print(f"Checkpoint interval     : {CHECKPOINT_INTERVAL}")
print("=" * 100)

# --------------------------------------------------------------------------------------
# 3. Verified Google Drive mount
# --------------------------------------------------------------------------------------
from google.colab import drive

def contains_test_records(project_root):
    project_root = Path(project_root)

    if not project_root.exists():
        return False

    preferred = (
        project_root
        / "data"
        / "processed"
        / "test_records.pkl"
    )

    if preferred.is_file():
        return True

    try:
        return next(
            project_root.rglob("test_records.pkl"),
            None,
        ) is not None
    except Exception:
        return False

existing_candidates = [
    Path("/content/depth_ot_evaluation_drive/MyDrive/depth_ot_patent"),
    Path("/content/ecrtm_drive/MyDrive/depth_ot_patent"),
    Path("/content/fastopic_drive/MyDrive/depth_ot_patent"),
    Path("/content/gdrive/MyDrive/depth_ot_patent"),
    Path("/content/drive/MyDrive/depth_ot_patent"),
]

PROJECT_ROOT = None
MY_DRIVE = None

for candidate in existing_candidates:
    if contains_test_records(candidate):
        PROJECT_ROOT = candidate
        MY_DRIVE = candidate.parents[1]
        print(f"[VERIFIED EXISTING MOUNT] {candidate}")
        break

if PROJECT_ROOT is None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DRIVE_MOUNT = Path(f"/content/ecrtm_drive_{stamp}")

    print(f"[MOUNT] Google Drive → {DRIVE_MOUNT}")
    drive.mount(str(DRIVE_MOUNT), force_remount=False)

    MY_DRIVE = DRIVE_MOUNT / "MyDrive"

    if not MY_DRIVE.is_dir():
        raise RuntimeError(
            f"MyDrive를 찾을 수 없습니다: {MY_DRIVE}"
        )

    direct_project = MY_DRIVE / "depth_ot_patent"

    if contains_test_records(direct_project):
        PROJECT_ROOT = direct_project
    else:
        print("[SEARCH] MyDrive에서 depth_ot_patent 프로젝트 검색")

        candidates = []

        for test_path in MY_DRIVE.rglob("test_records.pkl"):
            for parent in test_path.parents:
                if parent.name == "depth_ot_patent":
                    candidates.append(parent)
                    break

        unique_candidates = []
        seen = set()

        for candidate in candidates:
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        if not unique_candidates:
            raise FileNotFoundError(
                "MyDrive에서 test_records.pkl이 포함된 "
                "depth_ot_patent 프로젝트를 찾지 못했습니다."
            )

        PROJECT_ROOT = unique_candidates[0]

if not contains_test_records(PROJECT_ROOT):
    raise RuntimeError(
        f"프로젝트 검증 실패: {PROJECT_ROOT}"
    )

print(f"[PROJECT ROOT VERIFIED] {PROJECT_ROOT}")

# --------------------------------------------------------------------------------------
# 4. File discovery
# --------------------------------------------------------------------------------------
def find_required_file(filename, preferred_paths=None):
    preferred_paths = preferred_paths or []

    for path in preferred_paths:
        path = Path(path)
        if path.is_file():
            print(f"[FOUND] {filename}: {path}")
            return path

    matches = sorted(
        p for p in PROJECT_ROOT.rglob(filename)
        if p.is_file()
    )

    if len(matches) == 1:
        print(f"[FOUND] {filename}: {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        processed = [
            p for p in matches
            if "data/processed" in str(p).replace("\\", "/")
        ]

        selected = processed[0] if processed else matches[0]

        print(f"[WARNING] {filename} 후보가 여러 개입니다.")
        for path in matches:
            print("  ", path)
        print(f"[SELECTED] {selected}")

        return selected

    # 프로젝트 경로 밖의 같은 Drive 검색
    drive_matches = sorted(
        p for p in MY_DRIVE.rglob(filename)
        if p.is_file()
    )

    if drive_matches:
        selected = drive_matches[0]
        print(f"[FOUND IN MYDRIVE] {filename}: {selected}")
        return selected

    raise FileNotFoundError(
        f"필수 파일을 찾을 수 없습니다: {filename}"
    )

TRAIN_RECORD_PATH = find_required_file(
    "train_records.pkl",
    [
        PROJECT_ROOT / "data" / "processed" / "train_records.pkl",
        PROJECT_ROOT / "data" / "processed" / "records" / "train_records.pkl",
    ],
)

TEST_RECORD_PATH = find_required_file(
    "test_records.pkl",
    [
        PROJECT_ROOT / "data" / "processed" / "test_records.pkl",
        PROJECT_ROOT / "data" / "processed" / "records" / "test_records.pkl",
    ],
)

if TRAIN_RECORD_PATH.stat().st_size == 0:
    raise RuntimeError(f"빈 파일입니다: {TRAIN_RECORD_PATH}")

if TEST_RECORD_PATH.stat().st_size == 0:
    raise RuntimeError(f"빈 파일입니다: {TEST_RECORD_PATH}")

# 고정 폴더를 사용해 Colab이 중단돼도 자동 resume
RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "ecrtm_k30_glove200_seeds_42_43_44"
)

CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
ARRAY_DIR = RESULT_DIR / "arrays"
TOPIC_DIR = RESULT_DIR / "topics"

# 전처리 데이터는 Drive에 저장하여 새 런타임에서도 재사용
CACHE_DIR = (
    PROJECT_ROOT
    / "cache"
    / "ecrtm_k30_trainonly_vocab8000"
)

for directory in [
    RESULT_DIR,
    CHECKPOINT_DIR,
    ARRAY_DIR,
    TOPIC_DIR,
    CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 100)
print("VERIFIED ECRTM PATHS")
print("=" * 100)
print(f"MyDrive       : {MY_DRIVE}")
print(f"Project root  : {PROJECT_ROOT}")
print(f"Train records : {TRAIN_RECORD_PATH}")
print(f"Test records  : {TEST_RECORD_PATH}")
print(f"Result dir    : {RESULT_DIR}")
print(f"Cache dir     : {CACHE_DIR}")
print("=" * 100)

# --------------------------------------------------------------------------------------
# 5. Record loading
# --------------------------------------------------------------------------------------
def load_pickle_records(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        records = obj

    elif isinstance(obj, tuple):
        records = list(obj)

    elif isinstance(obj, dict):
        records = None

        for key in ["records", "data", "items", "patents"]:
            if key in obj and isinstance(obj[key], (list, tuple)):
                records = list(obj[key])
                break

        if records is None:
            if all(isinstance(v, dict) for v in obj.values()):
                records = list(obj.values())
            else:
                raise TypeError(
                    f"해석할 수 없는 dictionary 구조: "
                    f"{list(obj.keys())[:30]}"
                )
    else:
        raise TypeError(
            f"지원되지 않는 pickle 타입: {type(obj)}"
        )

    if not records:
        raise RuntimeError(f"레코드가 비어 있습니다: {path}")

    if not isinstance(records[0], dict):
        raise TypeError(
            f"각 record는 dict여야 합니다: {type(records[0])}"
        )

    return records

print("[LOAD] Train records")
train_records = load_pickle_records(TRAIN_RECORD_PATH)

print("[LOAD] Test records")
test_records = load_pickle_records(TEST_RECORD_PATH)

print(f"[RECORDS] Train patents: {len(train_records):,}")
print(f"[RECORDS] Test patents : {len(test_records):,}")
print(f"[SAMPLE KEYS] {list(test_records[0].keys())}")

if len(test_records) != EXPECTED_TEST_PATENTS:
    print(
        f"[WARNING] Expected test patents={EXPECTED_TEST_PATENTS:,}, "
        f"found={len(test_records):,}"
    )

# --------------------------------------------------------------------------------------
# 6. Patent claim text extraction
# --------------------------------------------------------------------------------------
CLAIM_KEYS = [
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
]

def scalar_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple)):
        if all(isinstance(x, str) for x in value):
            return " ".join(
                x.strip() for x in value if x.strip()
            )

    return ""

def claim_item_text(item):
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        for key in CLAIM_TEXT_KEYS:
            if key in item:
                text = scalar_text(item[key])
                if text:
                    return text

        for key in ["tokens", "words"]:
            if key in item and isinstance(item[key], (list, tuple)):
                return " ".join(str(x) for x in item[key])

    return ""

def extract_patent_text(record):
    for key in CLAIM_KEYS:
        if key not in record:
            continue

        value = record[key]
        texts = []

        if isinstance(value, (list, tuple)):
            for item in value:
                text = claim_item_text(item)
                if text:
                    texts.append(text)

        elif isinstance(value, dict):
            for item in value.values():
                text = claim_item_text(item)
                if text:
                    texts.append(text)

        elif isinstance(value, str):
            texts.append(value.strip())

        if texts:
            return " ".join(texts)

    for key in [
        "patent_text",
        "document",
        "doc_text",
        "full_text",
        "text",
        "content",
    ]:
        if key in record:
            text = scalar_text(record[key])
            if text:
                return text

    pieces = []

    for key in ["title", "abstract", "summary"]:
        if key in record:
            text = scalar_text(record[key])
            if text:
                pieces.append(text)

    return " ".join(pieces)

def extract_patent_id(record, index):
    for key in [
        "patent_id",
        "publication_number",
        "patent_number",
        "document_id",
        "doc_id",
        "id",
    ]:
        if key in record and record[key] is not None:
            return str(record[key])

    return f"record_{index:08d}"

print("[TEXT] Extracting train patent documents")
train_docs = [
    extract_patent_text(record)
    for record in train_records
]

print("[TEXT] Extracting test patent documents")
test_docs = [
    extract_patent_text(record)
    for record in test_records
]

test_ids = [
    extract_patent_id(record, i)
    for i, record in enumerate(test_records)
]

empty_train = [
    i for i, text in enumerate(train_docs)
    if not text.strip()
]
empty_test = [
    i for i, text in enumerate(test_docs)
    if not text.strip()
]

if empty_train:
    raise RuntimeError(
        f"빈 train documents={len(empty_train):,}, "
        f"sample={empty_train[:10]}"
    )

if empty_test:
    raise RuntimeError(
        f"빈 test documents={len(empty_test):,}, "
        f"sample={empty_test[:10]}"
    )

train_lengths = np.asarray([
    len(text.split()) for text in train_docs
])
test_lengths = np.asarray([
    len(text.split()) for text in test_docs
])

print(
    f"[TRAIN TEXT] mean={train_lengths.mean():.1f}, "
    f"median={np.median(train_lengths):.1f}, "
    f"max={train_lengths.max():,}"
)
print(
    f"[TEST TEXT] mean={test_lengths.mean():.1f}, "
    f"median={np.median(test_lengths):.1f}, "
    f"max={test_lengths.max():,}"
)

# --------------------------------------------------------------------------------------
# 7. CPC extraction
# --------------------------------------------------------------------------------------
CPC_PATTERN = re.compile(
    r"\b([A-HY]\d{2}[A-Z])(?:\d+)?(?:/\d+)?\b",
    re.I,
)

def flatten_values(value):
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, (int, float)):
        return [str(value)]

    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(flatten_values(item))
        return output

    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(flatten_values(item))
        return output

    return [str(value)]

def find_cpc(value):
    for item in flatten_values(value):
        matches = CPC_PATTERN.findall(str(item).upper())
        if matches:
            return matches[0].upper()

    return None

def extract_cpc(record):
    for key in [
        "primary_cpc",
        "main_cpc",
        "first_cpc",
        "cpc_primary",
        "primary_cpc_code",
    ]:
        if key in record:
            code = find_cpc(record[key])
            if code:
                return code

    # 이미 전처리된 subclass field 우선 사용 가능
    if "subclass" in record:
        code = find_cpc(record["subclass"])
        if code:
            return code

    for key in [
        "cpc_codes",
        "cpc",
        "cpcs",
        "cpc_labels",
        "classifications",
    ]:
        if key in record:
            code = find_cpc(record[key])
            if code:
                return code

    return None

test_cpc = [
    extract_cpc(record)
    for record in test_records
]

valid_cpc_mask = np.asarray([
    code is not None for code in test_cpc
])

section_labels = np.asarray([
    code[0] if code else ""
    for code in test_cpc
])
class_labels = np.asarray([
    code[:3] if code else ""
    for code in test_cpc
])
subclass_labels = np.asarray([
    code[:4] if code else ""
    for code in test_cpc
])

num_sections = len(set(section_labels[valid_cpc_mask]))
num_classes = len(set(class_labels[valid_cpc_mask]))
num_subclasses = len(set(subclass_labels[valid_cpc_mask]))

print(f"[CPC] Valid={valid_cpc_mask.sum():,}")
print(
    f"[CPC LABEL COUNTS] Section={num_sections}, "
    f"Class={num_classes}, Subclass={num_subclasses}"
)

if (
    num_sections != EXPECTED_SECTIONS
    or num_classes != EXPECTED_CLASSES
    or num_subclasses != EXPECTED_SUBCLASSES
):
    raise RuntimeError(
        "기존 표의 CPC label count와 일치하지 않습니다.\n"
        f"Expected={EXPECTED_SECTIONS}/{EXPECTED_CLASSES}/"
        f"{EXPECTED_SUBCLASSES}\n"
        f"Found={num_sections}/{num_classes}/{num_subclasses}"
    )

# 레코드 원본은 더 이상 필요하지 않음
del train_records
del test_records
gc.collect()

# --------------------------------------------------------------------------------------
# 8. Train-only vocabulary and sparse BoW
# --------------------------------------------------------------------------------------
TRAIN_BOW_PATH = CACHE_DIR / "train_bow.npz"
TEST_BOW_PATH = CACHE_DIR / "test_bow.npz"
VOCAB_PATH = CACHE_DIR / "vocabulary.json"
GLOVE_PATH = CACHE_DIR / "glove200_embeddings.npy"
CACHE_INFO_PATH = CACHE_DIR / "cache_information.json"

def cache_is_valid():
    required = [
        TRAIN_BOW_PATH,
        TEST_BOW_PATH,
        VOCAB_PATH,
    ]

    if not all(path.exists() for path in required):
        return False

    try:
        train_bow = sp.load_npz(TRAIN_BOW_PATH)
        test_bow = sp.load_npz(TEST_BOW_PATH)

        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        return (
            train_bow.shape == (len(train_docs), VOCAB_SIZE)
            and test_bow.shape == (len(test_docs), VOCAB_SIZE)
            and len(vocab) == VOCAB_SIZE
        )
    except Exception:
        return False

if cache_is_valid():
    print("[BOW CACHE HIT] Loading train/test sparse matrices")

    train_bow = sp.load_npz(TRAIN_BOW_PATH).tocsr().astype(np.float32)
    test_bow = sp.load_npz(TEST_BOW_PATH).tocsr().astype(np.float32)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

else:
    print("[PREPROCESS] Building train-only vocabulary and BoW")
    print(
        f"Vocabulary={VOCAB_SIZE}, min_df={MIN_DOC_COUNT}, "
        f"max_df={MAX_DOC_FREQ}"
    )

    preprocess = Preprocess(
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

    # 중요: test 문서를 전달하지 않으므로 vocabulary는 train-only
    train_rst = preprocess.preprocess(
        train_docs,
        pretrained_WE=False,
    )

    vocab = list(train_rst["vocab"])
    train_bow = train_rst["train_bow"].tocsr().astype(np.float32)

    if len(vocab) != VOCAB_SIZE:
        raise RuntimeError(
            f"Vocabulary size mismatch: "
            f"expected={VOCAB_SIZE}, found={len(vocab)}"
        )

    print("[PREPROCESS] Parsing test documents with fixed train vocabulary")

    _, test_bow = preprocess.parse(
        test_docs,
        vocab,
    )
    test_bow = test_bow.tocsr().astype(np.float32)

    sp.save_npz(
        TRAIN_BOW_PATH,
        train_bow,
        compressed=True,
    )
    sp.save_npz(
        TEST_BOW_PATH,
        test_bow,
        compressed=True,
    )

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    cache_info = {
        "created_at": datetime.now().isoformat(),
        "train_records": str(TRAIN_RECORD_PATH),
        "test_records": str(TEST_RECORD_PATH),
        "train_shape": list(train_bow.shape),
        "test_shape": list(test_bow.shape),
        "vocab_size": len(vocab),
        "min_doc_count": MIN_DOC_COUNT,
        "max_doc_freq": MAX_DOC_FREQ,
        "vocabulary_train_only": True,
    }

    with open(
        CACHE_INFO_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cache_info, f, indent=2, ensure_ascii=False)

    del train_rst
    del preprocess
    gc.collect()

if train_bow.shape != (len(train_docs), VOCAB_SIZE):
    raise RuntimeError(
        f"Unexpected train BoW shape: {train_bow.shape}"
    )

if test_bow.shape != (len(test_docs), VOCAB_SIZE):
    raise RuntimeError(
        f"Unexpected test BoW shape: {test_bow.shape}"
    )

print(f"[BOW] Train shape: {train_bow.shape}")
print(f"[BOW] Test shape : {test_bow.shape}")
print(
    f"[BOW] Train average tokens: "
    f"{train_bow.sum() / train_bow.shape[0]:.2f}"
)
print(
    f"[BOW] Test average tokens : "
    f"{test_bow.sum() / test_bow.shape[0]:.2f}"
)

# 원문은 BoW 생성 후 필요 없음
del train_docs
del test_docs
gc.collect()

# --------------------------------------------------------------------------------------
# 9. GloVe 200d embeddings
# --------------------------------------------------------------------------------------
def load_or_build_glove(vocab):
    if GLOVE_PATH.exists():
        embeddings = np.load(GLOVE_PATH)

        if embeddings.shape == (len(vocab), EMBED_SIZE):
            print(
                f"[GLOVE CACHE HIT] {GLOVE_PATH}, "
                f"shape={embeddings.shape}"
            )
            return embeddings.astype(np.float32)

        print(
            f"[GLOVE CACHE INVALID] {embeddings.shape}"
        )

    print("[GLOVE] Downloading/loading glove-wiki-gigaword-200")
    print("[GLOVE] 최초 실행 시 약 250 MiB 다운로드가 필요합니다.")

    import gensim.downloader as api

    glove = api.load("glove-wiki-gigaword-200")

    embeddings = np.zeros(
        (len(vocab), EMBED_SIZE),
        dtype=np.float32,
    )

    found = 0

    for index, word in enumerate(
        tqdm(vocab, desc="Mapping GloVe embeddings")
    ):
        if word in glove.key_to_index:
            embeddings[index] = glove[word]
            found += 1

    np.save(GLOVE_PATH, embeddings)

    print(
        f"[GLOVE] Found={found:,}/{len(vocab):,} "
        f"({found / len(vocab):.2%})"
    )
    print(f"[GLOVE SAVED] {GLOVE_PATH}")

    del glove
    gc.collect()

    return embeddings

pretrained_word_embeddings = load_or_build_glove(vocab)

if pretrained_word_embeddings.shape != (VOCAB_SIZE, EMBED_SIZE):
    raise RuntimeError(
        f"Unexpected GloVe shape: "
        f"{pretrained_word_embeddings.shape}"
    )

# --------------------------------------------------------------------------------------
# 10. CPC metrics
# --------------------------------------------------------------------------------------
def predicted_cluster_purity(true_labels, predicted_clusters):
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(predicted_clusters)

    correct = 0

    for cluster in np.unique(predicted_clusters):
        mask = predicted_clusters == cluster
        _, counts = np.unique(
            true_labels[mask],
            return_counts=True,
        )
        correct += int(counts.max())

    return float(correct / len(true_labels))

def inverse_label_purity(true_labels, predicted_clusters):
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(predicted_clusters)

    correct = 0

    for label in np.unique(true_labels):
        mask = true_labels == label
        _, counts = np.unique(
            predicted_clusters[mask],
            return_counts=True,
        )
        correct += int(counts.max())

    return float(correct / len(true_labels))

def evaluate_level(true_labels, predicted_clusters):
    return {
        "pur_p": predicted_cluster_purity(
            true_labels,
            predicted_clusters,
        ),
        "pur_a": inverse_label_purity(
            true_labels,
            predicted_clusters,
        ),
        "nmi": float(
            normalized_mutual_info_score(
                true_labels,
                predicted_clusters,
                average_method="arithmetic",
            )
        ),
    }

def evaluate_cpc(predicted_clusters):
    pred = np.asarray(predicted_clusters)[valid_cpc_mask]

    return {
        "section": evaluate_level(
            section_labels[valid_cpc_mask],
            pred,
        ),
        "class": evaluate_level(
            class_labels[valid_cpc_mask],
            pred,
        ),
        "subclass": evaluate_level(
            subclass_labels[valid_cpc_mask],
            pred,
        ),
    }

# --------------------------------------------------------------------------------------
# 11. Sparse batch utilities
# --------------------------------------------------------------------------------------
def sparse_batch_to_gpu(csr_matrix, indices):
    dense = csr_matrix[indices].toarray()
    tensor = torch.from_numpy(dense).float()
    return tensor.to(DEVICE, non_blocking=True)

@torch.no_grad()
def infer_theta(model, csr_matrix, batch_size=512):
    model.eval()

    outputs = []

    for start in tqdm(
        range(0, csr_matrix.shape[0], batch_size),
        desc="Theta inference",
        leave=False,
    ):
        end = min(start + batch_size, csr_matrix.shape[0])
        indices = np.arange(start, end)

        batch = sparse_batch_to_gpu(
            csr_matrix,
            indices,
        )

        theta = model.get_theta(batch)
        outputs.append(
            theta.detach().cpu().numpy().astype(np.float32)
        )

        del batch
        del theta

    return np.concatenate(outputs, axis=0)

def get_beta_numpy(model):
    model.eval()

    with torch.no_grad():
        return (
            model.get_beta()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

def get_top_words(beta, vocab, num_words=25):
    vocab_array = np.asarray(vocab)

    rows = []

    for topic_id, distribution in enumerate(beta):
        indices = np.argsort(distribution)[::-1][:num_words]
        words = vocab_array[indices].tolist()

        rows.append({
            "topic_id": topic_id,
            "top_words": " ".join(words),
        })

    return rows

# --------------------------------------------------------------------------------------
# 12. Configuration
# --------------------------------------------------------------------------------------
configuration = {
    "model": "ECRTM",
    "official_source": "TopMost",
    "created_at": datetime.now().isoformat(),
    "project_root": str(PROJECT_ROOT),
    "train_records": str(TRAIN_RECORD_PATH),
    "test_records": str(TEST_RECORD_PATH),
    "train_patents": int(train_bow.shape[0]),
    "test_patents": int(test_bow.shape[0]),
    "valid_test_cpc": int(valid_cpc_mask.sum()),
    "num_topics": NUM_TOPICS,
    "seeds": SEEDS,
    "vocab_size": VOCAB_SIZE,
    "min_doc_count": MIN_DOC_COUNT,
    "max_doc_freq": MAX_DOC_FREQ,
    "vocabulary_train_only": True,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "lr_step_size": LR_STEP_SIZE,
    "lr_gamma": LR_GAMMA,
    "encoder_units": ENCODER_UNITS,
    "dropout": DROPOUT,
    "pretrained_word_embeddings": "glove-wiki-gigaword-200",
    "embedding_size": EMBED_SIZE,
    "beta_temp": BETA_TEMP,
    "weight_loss_ecr": WEIGHT_LOSS_ECR,
    "sinkhorn_alpha": SINKHORN_ALPHA,
    "sinkhorn_max_iter": SINKHORN_MAX_ITER,
    "mixed_precision": False,
    "gpu": GPU_NAME,
    "document_unit": "patent",
    "document_text": "concatenated claims",
    "cpc_rule": "primary CPC or first CPC",
    "cpc_in_training": False,
    "cpc_in_model_selection": False,
    "checkpoint_interval": CHECKPOINT_INTERVAL,
}

with open(
    RESULT_DIR / "configuration.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        configuration,
        f,
        indent=2,
        ensure_ascii=False,
    )

label_audit = pd.DataFrame({
    "patent_id": test_ids,
    "primary_cpc": test_cpc,
    "section": section_labels,
    "class": class_labels,
    "subclass": subclass_labels,
    "valid": valid_cpc_mask,
})

label_audit.to_csv(
    RESULT_DIR / "test_cpc_label_audit.csv",
    index=False,
)

# --------------------------------------------------------------------------------------
# 13. Three-seed training with automatic resume
# --------------------------------------------------------------------------------------
all_seed_rows = []
all_seed_results = []

total_start = time.time()

for seed_number, seed in enumerate(SEEDS, start=1):
    print("\n" + "=" * 100)
    print(f"ECRTM SEED {seed} — {seed_number}/{len(SEEDS)}")
    print("=" * 100)

    metrics_path = RESULT_DIR / f"seed_{seed}_metrics.json"
    checkpoint_path = CHECKPOINT_DIR / f"seed_{seed}_latest.pt"
    final_model_path = CHECKPOINT_DIR / f"seed_{seed}_final.pt"

    # 이미 완료된 seed는 다시 학습하지 않음
    if metrics_path.exists() and final_model_path.exists():
        print(f"[SKIP] Seed {seed}는 이미 완료됐습니다.")

        with open(metrics_path, "r", encoding="utf-8") as f:
            seed_result = json.load(f)

        all_seed_results.append(seed_result)

        for level in ["section", "class", "subclass"]:
            all_seed_rows.append({
                "seed": seed,
                "level": level,
                "pur_p": seed_result["metrics"][level]["pur_p"],
                "pur_a": seed_result["metrics"][level]["pur_a"],
                "nmi": seed_result["metrics"][level]["nmi"],
                "active_topics": seed_result["active_topics"],
                "max_topic_share": seed_result["max_topic_share"],
                "elapsed_seconds": seed_result["elapsed_seconds"],
                "peak_gpu_gib": seed_result["peak_gpu_gib"],
            })

        continue

    set_seed(seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = ECRTM(
        vocab_size=VOCAB_SIZE,
        num_topics=NUM_TOPICS,
        en_units=ENCODER_UNITS,
        dropout=DROPOUT,
        pretrained_WE=pretrained_word_embeddings.copy(),
        embed_size=EMBED_SIZE,
        beta_temp=BETA_TEMP,
        weight_loss_ECR=WEIGHT_LOSS_ECR,
        sinkhorn_alpha=SINKHORN_ALPHA,
        sinkhorn_max_iter=SINKHORN_MAX_ITER,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )

    start_epoch = 1
    previous_elapsed = 0.0

    # 자동 resume
    if checkpoint_path.exists():
        print(f"[RESUME] {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        previous_elapsed = float(
            checkpoint.get("elapsed_seconds", 0.0)
        )

        print(
            f"[RESUME] Start epoch={start_epoch}, "
            f"previous elapsed={previous_elapsed / 3600:.2f}h"
        )

    seed_start = time.time()

    num_train = train_bow.shape[0]
    num_batches = int(np.ceil(num_train / BATCH_SIZE))

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()

        permutation = np.random.permutation(num_train)

        epoch_totals = defaultdict(float)
        epoch_start = time.time()

        progress = tqdm(
            range(0, num_train, BATCH_SIZE),
            total=num_batches,
            desc=f"Seed {seed} Epoch {epoch:03d}/{EPOCHS}",
            leave=False,
        )

        for start in progress:
            indices = permutation[start:start + BATCH_SIZE]

            batch = sparse_batch_to_gpu(
                train_bow,
                indices,
            )

            optimizer.zero_grad(set_to_none=True)

            # Sinkhorn 안정성을 위해 FP32
            output = model(batch)
            loss = output["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss: seed={seed}, epoch={epoch}, "
                    f"batch_start={start}, loss={loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size_actual = len(indices)

            for key, value in output.items():
                epoch_totals[key] += (
                    float(value.detach().cpu())
                    * batch_size_actual
                )

            progress.set_postfix({
                "loss": f"{float(loss.detach().cpu()):.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "gpu": (
                    f"{torch.cuda.memory_allocated() / (1024**3):.1f}G"
                ),
            })

            del batch
            del output
            del loss

        scheduler.step()

        epoch_elapsed = time.time() - epoch_start

        mean_loss = epoch_totals["loss"] / num_train
        mean_tm = epoch_totals.get("loss_TM", 0.0) / num_train
        mean_ecr = epoch_totals.get("loss_ECR", 0.0) / num_train

        print(
            f"[SEED {seed} EPOCH {epoch:03d}/{EPOCHS}] "
            f"loss={mean_loss:.6f} | "
            f"TM={mean_tm:.6f} | "
            f"ECR={mean_ecr:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"time={epoch_elapsed / 60:.1f}m"
        )

        if (
            epoch % CHECKPOINT_INTERVAL == 0
            or epoch == EPOCHS
        ):
            elapsed = (
                previous_elapsed
                + time.time()
                - seed_start
            )

            temp_path = checkpoint_path.with_suffix(".tmp.pt")

            torch.save({
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "elapsed_seconds": elapsed,
                "configuration": configuration,
                "vocab": vocab,
            }, temp_path)

            os.replace(temp_path, checkpoint_path)

            print(
                f"[CHECKPOINT] Seed={seed}, epoch={epoch}: "
                f"{checkpoint_path}"
            )

    total_seed_elapsed = (
        previous_elapsed
        + time.time()
        - seed_start
    )

    # ----------------------------------------------------------------------------------
    # Final inference
    # ----------------------------------------------------------------------------------
    print(f"[INFERENCE] Seed {seed} train theta")
    train_theta = infer_theta(
        model,
        train_bow,
        INFERENCE_BATCH_SIZE,
    )

    print(f"[INFERENCE] Seed {seed} test theta")
    test_theta = infer_theta(
        model,
        test_bow,
        INFERENCE_BATCH_SIZE,
    )

    beta = get_beta_numpy(model)
    predicted_topics = test_theta.argmax(axis=1)

    metrics = evaluate_cpc(predicted_topics)

    topic_counts = np.bincount(
        predicted_topics,
        minlength=NUM_TOPICS,
    )

    active_topics = int(
        np.count_nonzero(topic_counts)
    )
    max_topic_share = float(
        topic_counts.max() / topic_counts.sum()
    )

    peak_gpu_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    # Arrays
    np.save(
        ARRAY_DIR / f"seed_{seed}_train_theta.npy",
        train_theta.astype(np.float32),
    )
    np.save(
        ARRAY_DIR / f"seed_{seed}_test_theta.npy",
        test_theta.astype(np.float32),
    )
    np.save(
        ARRAY_DIR / f"seed_{seed}_beta.npy",
        beta.astype(np.float32),
    )
    np.save(
        ARRAY_DIR / f"seed_{seed}_test_assignments.npy",
        predicted_topics.astype(np.int16),
    )

    # Topic words
    topic_rows = get_top_words(
        beta,
        vocab,
        NUM_TOP_WORDS,
    )

    pd.DataFrame(topic_rows).to_csv(
        TOPIC_DIR / f"seed_{seed}_top_words.csv",
        index=False,
    )

    seed_result = {
        "seed": seed,
        "metrics": metrics,
        "active_topics": active_topics,
        "max_topic_share": max_topic_share,
        "elapsed_seconds": total_seed_elapsed,
        "peak_gpu_gib": peak_gpu_gib,
        "train_theta_shape": list(train_theta.shape),
        "test_theta_shape": list(test_theta.shape),
        "beta_shape": list(beta.shape),
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            seed_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Final model
    torch.save({
        "seed": seed,
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "configuration": configuration,
        "vocab": vocab,
        "metrics": metrics,
    }, final_model_path)

    all_seed_results.append(seed_result)

    for level in ["section", "class", "subclass"]:
        all_seed_rows.append({
            "seed": seed,
            "level": level,
            "pur_p": metrics[level]["pur_p"],
            "pur_a": metrics[level]["pur_a"],
            "nmi": metrics[level]["nmi"],
            "active_topics": active_topics,
            "max_topic_share": max_topic_share,
            "elapsed_seconds": total_seed_elapsed,
            "peak_gpu_gib": peak_gpu_gib,
        })

    print("-" * 100)
    print(f"[SEED {seed} FINAL CPC ALIGNMENT]")
    print(
        f"Section : Pur_p={metrics['section']['pur_p']:.4f} | "
        f"Pur_a={metrics['section']['pur_a']:.4f} | "
        f"NMI={metrics['section']['nmi']:.4f}"
    )
    print(
        f"Class   : Pur_p={metrics['class']['pur_p']:.4f} | "
        f"Pur_a={metrics['class']['pur_a']:.4f} | "
        f"NMI={metrics['class']['nmi']:.4f}"
    )
    print(
        f"Subclass: Pur_p={metrics['subclass']['pur_p']:.4f} | "
        f"Pur_a={metrics['subclass']['pur_a']:.4f} | "
        f"NMI={metrics['subclass']['nmi']:.4f}"
    )
    print(
        f"Active topics={active_topics}/{NUM_TOPICS} | "
        f"max share={max_topic_share:.2%} | "
        f"GPU peak={peak_gpu_gib:.2f} GiB | "
        f"time={total_seed_elapsed / 3600:.2f}h"
    )
    print("-" * 100)

    del model
    del optimizer
    del scheduler
    del train_theta
    del test_theta
    del beta
    del predicted_topics

    gc.collect()
    torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------
# 14. Aggregate results
# --------------------------------------------------------------------------------------
seed_df = pd.DataFrame(all_seed_rows)

seed_df.to_csv(
    RESULT_DIR / "ecrtm_seed_level_results.csv",
    index=False,
)

summary_rows = []

for level in ["section", "class", "subclass"]:
    level_df = seed_df[
        seed_df["level"] == level
    ]

    row = {"level": level}

    for metric in ["pur_p", "pur_a", "nmi"]:
        values = level_df[metric].to_numpy(dtype=float)

        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=1))
        row[f"{metric}_min"] = float(values.min())
        row[f"{metric}_max"] = float(values.max())

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    RESULT_DIR / "ecrtm_3seed_summary.csv",
    index=False,
)

complete_result = {
    "configuration": configuration,
    "seed_results": all_seed_results,
    "summary": summary_rows,
    "total_elapsed_seconds": time.time() - total_start,
}

with open(
    RESULT_DIR / "ecrtm_3seed_complete_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        complete_result,
        f,
        indent=2,
        ensure_ascii=False,
    )

# --------------------------------------------------------------------------------------
# 15. Final report and LaTeX row
# --------------------------------------------------------------------------------------
def summary_value(level, metric, statistic="mean"):
    row = summary_df[
        summary_df["level"] == level
    ].iloc[0]

    return float(row[f"{metric}_{statistic}"])

print("\n" + "=" * 100)
print("ECRTM 3-SEED FINAL CPC ALIGNMENT")
print("=" * 100)

for title, level in [
    ("SECTION", "section"),
    ("CLASS", "class"),
    ("SUBCLASS", "subclass"),
]:
    print(f"\n[{title}]")

    for metric_title, metric in [
        ("Pur_p", "pur_p"),
        ("Pur_a", "pur_a"),
        ("NMI", "nmi"),
    ]:
        mean = summary_value(level, metric)
        std = summary_value(level, metric, "std")

        print(
            f"{metric_title:<5} = {mean:.4f} ± {std:.4f}"
        )

latex_mean = (
    "ECRTM "
    + " & ".join(
        f"{summary_value(level, metric):.4f}"
        for level in ["section", "class", "subclass"]
        for metric in ["pur_p", "pur_a", "nmi"]
    )
    + r" \\"
)

latex_mean_std = (
    "ECRTM "
    + " & ".join(
        f"${summary_value(level, metric):.4f}"
        f"\\pm{summary_value(level, metric, 'std'):.4f}$"
        for level in ["section", "class", "subclass"]
        for metric in ["pur_p", "pur_a", "nmi"]
    )
    + r" \\"
)

with open(
    RESULT_DIR / "latex_row_mean.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_mean + "\n")

with open(
    RESULT_DIR / "latex_row_mean_std.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_mean_std + "\n")

print("\n" + "-" * 100)
print("[LATEX ROW — MEAN]")
print(latex_mean)

print("\n[LATEX ROW — MEAN ± STD]")
print(latex_mean_std)

print("\n[PER-SEED RESULTS]")
display(seed_df)

print("\n[3-SEED SUMMARY]")
display(summary_df)

print("\n" + "=" * 100)
print("[COMPLETE]")
print(f"Result directory: {RESULT_DIR}")
print(
    "Main JSON      : "
    f"{RESULT_DIR / 'ecrtm_3seed_complete_results.json'}"
)
print(
    "Summary CSV    : "
    f"{RESULT_DIR / 'ecrtm_3seed_summary.csv'}"
)
print(
    "Seed CSV       : "
    f"{RESULT_DIR / 'ecrtm_seed_level_results.csv'}"
)
print("=" * 100)
