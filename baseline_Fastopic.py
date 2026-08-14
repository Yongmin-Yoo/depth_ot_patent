# ======================================================================================
# FASTopic 3-SEED PATENT CPC BENCHMARK — ONE-CELL COLAB / NVIDIA T4
# Seeds: 42, 43, 44 | Topics: 30 | Train-only fitting | Test-only evaluation
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
from pathlib import Path
from datetime import datetime

# --------------------------------------------------------------------------------------
# 0. Environment
# --------------------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

warnings.filterwarnings("ignore")

def install_if_needed():
    required = [
        ("fastopic", "fastopic==1.0.1"),
        ("topmost", "topmost"),
        ("sentence_transformers", "sentence-transformers"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
    ]
    missing = []
    for module_name, pip_name in required:
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)

    if missing:
        print("[INSTALL]", " ".join(missing))
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "--upgrade", *missing
        ])

install_if_needed()

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import normalized_mutual_info_score
from sentence_transformers import SentenceTransformer
from fastopic import FASTopic
from topmost import Preprocess

# --------------------------------------------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------------------------------------------
SEEDS = [42, 43, 44]
NUM_TOPICS = 30

VOCAB_SIZE = 8000
MIN_DOC_COUNT = 10
MAX_DOC_FREQ = 0.70

FAST_TOPIC_EPOCHS = 200
FAST_TOPIC_LR = 0.002

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 128

LOW_MEMORY_BATCH_SIZE = 4096
NORMALIZE_EMBEDDINGS = False

EXPECTED_TEST_PATENTS = 9881
EXPECTED_SECTION_LABELS = 9
EXPECTED_CLASS_LABELS = 121
EXPECTED_SUBCLASS_LABELS = 466

DEVICE = "cuda"

# --------------------------------------------------------------------------------------
# 2. GPU validation and reproducibility
# --------------------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. Colab 런타임 유형을 NVIDIA T4 GPU로 설정하세요."
    )

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GIB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

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
print("FASTopic 3-SEED CPC BENCHMARK")
print("=" * 100)
print(f"GPU                  : {GPU_NAME}")
print(f"GPU memory           : {GPU_MEMORY_GIB:.2f} GiB")
print(f"PyTorch              : {torch.__version__}")
print(f"Seeds                : {SEEDS}")
print(f"Topics               : {NUM_TOPICS}")
print(f"Vocabulary           : {VOCAB_SIZE:,}")
print(f"Training epochs      : {FAST_TOPIC_EPOCHS}")
print(f"Learning rate        : {FAST_TOPIC_LR}")
print(f"Embedding model      : {EMBED_MODEL}")
print(f"Low-memory batch     : {LOW_MEMORY_BATCH_SIZE:,}")
print(f"Normalize embeddings : {NORMALIZE_EMBEDDINGS}")
print("=" * 100)

# --------------------------------------------------------------------------------------
# 3–4. Verified Google Drive mount and robust input/output discovery
# --------------------------------------------------------------------------------------
from google.colab import drive
from pathlib import Path
from datetime import datetime
import os

def contains_test_records(project_root):
    """폴더 존재 여부가 아니라 실제 test_records.pkl 존재 여부로 Drive를 검증."""
    project_root = Path(project_root)

    if not project_root.exists():
        return False

    preferred = project_root / "data" / "processed" / "test_records.pkl"
    if preferred.is_file():
        return True

    try:
        return next(project_root.rglob("test_records.pkl"), None) is not None
    except Exception:
        return False

# 이미 정상적으로 마운트된 위치만 후보로 인정
existing_candidates = [
    Path("/content/depth_ot_evaluation_drive/MyDrive/depth_ot_patent"),
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

# 검증된 기존 mount가 없으면 완전히 새로운 빈 위치에 마운트
if PROJECT_ROOT is None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DRIVE_MOUNT = Path(f"/content/fastopic_drive_{stamp}")

    print(f"[MOUNT] Google Drive → {DRIVE_MOUNT}")
    drive.mount(str(DRIVE_MOUNT), force_remount=False)

    MY_DRIVE = DRIVE_MOUNT / "MyDrive"

    if not MY_DRIVE.is_dir():
        raise RuntimeError(
            f"Google Drive의 MyDrive를 찾을 수 없습니다: {MY_DRIVE}"
        )

    direct_project = MY_DRIVE / "depth_ot_patent"

    if contains_test_records(direct_project):
        PROJECT_ROOT = direct_project
    else:
        print("[SEARCH] MyDrive 전체에서 실제 depth_ot_patent 프로젝트를 검색합니다.")

        project_candidates = []
        for test_path in MY_DRIVE.rglob("test_records.pkl"):
            parent_candidates = [
                p for p in test_path.parents
                if p.name == "depth_ot_patent"
            ]
            project_candidates.extend(parent_candidates)

        # 중복 제거
        unique_projects = []
        seen = set()

        for project in project_candidates:
            project_string = str(project.resolve())
            if project_string not in seen:
                seen.add(project_string)
                unique_projects.append(project)

        if len(unique_projects) == 1:
            PROJECT_ROOT = unique_projects[0]

        elif len(unique_projects) > 1:
            print("[WARNING] 프로젝트 후보가 여러 개입니다:")
            for i, candidate in enumerate(unique_projects):
                print(f"  [{i}] {candidate}")

            # 기본적으로 정확한 MyDrive/depth_ot_patent 우선
            exact = [
                p for p in unique_projects
                if p == direct_project
            ]
            PROJECT_ROOT = exact[0] if exact else unique_projects[0]

            print(f"[SELECTED] {PROJECT_ROOT}")

        else:
            raise FileNotFoundError(
                f"{MY_DRIVE}에서 test_records.pkl을 포함한 "
                "depth_ot_patent 프로젝트를 찾지 못했습니다."
            )

if PROJECT_ROOT is None or not contains_test_records(PROJECT_ROOT):
    raise RuntimeError(
        f"실제 프로젝트 검증에 실패했습니다: {PROJECT_ROOT}"
    )

print(f"[PROJECT ROOT VERIFIED] {PROJECT_ROOT}")

# --------------------------------------------------------------------------------------
# Required file discovery
# --------------------------------------------------------------------------------------
def find_required_file(filename, preferred_paths=None):
    preferred_paths = preferred_paths or []

    # 1. 알려진 표준 경로 우선
    for path in preferred_paths:
        path = Path(path)
        if path.is_file():
            print(f"[FOUND] {filename}: {path}")
            return path

    # 2. 프로젝트 안에서 정확한 파일명 검색
    matches = sorted(
        p for p in PROJECT_ROOT.rglob(filename)
        if p.is_file()
    )

    if len(matches) == 1:
        print(f"[FOUND] {filename}: {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        print(f"[WARNING] {filename} 후보가 여러 개입니다:")
        for i, path in enumerate(matches):
            print(f"  [{i}] {path}")

        # data/processed 아래 경로 우선
        processed_matches = [
            p for p in matches
            if "data/processed" in str(p).replace("\\", "/")
        ]

        selected = (
            processed_matches[0]
            if processed_matches
            else matches[0]
        )

        print(f"[SELECTED] {selected}")
        return selected

    # 3. 프로젝트 경로가 예상과 다른 경우 MyDrive 전체 검색
    print(f"[SEARCH] MyDrive 전체에서 {filename} 검색 중...")

    drive_matches = sorted(
        p for p in MY_DRIVE.rglob(filename)
        if p.is_file()
    )

    if len(drive_matches) == 1:
        print(f"[FOUND IN MYDRIVE] {filename}: {drive_matches[0]}")
        return drive_matches[0]

    if len(drive_matches) > 1:
        print(f"[WARNING] MyDrive에 {filename} 후보가 여러 개입니다:")
        for i, path in enumerate(drive_matches):
            print(f"  [{i}] {path}")

        project_matches = [
            p for p in drive_matches
            if "depth_ot_patent" in p.parts
        ]

        selected = (
            project_matches[0]
            if project_matches
            else drive_matches[0]
        )

        print(f"[SELECTED] {selected}")
        return selected

    # 진단용 유사 파일 출력
    similar_files = sorted(
        p for p in MY_DRIVE.rglob("*record*.pkl")
        if p.is_file()
    )

    diagnostic = "\n".join(
        f"  - {path}" for path in similar_files[:50]
    )

    raise FileNotFoundError(
        f"{filename}을 실제 Google Drive에서 찾지 못했습니다.\n"
        f"검색한 MyDrive: {MY_DRIVE}\n"
        f"발견된 유사 record 파일:\n"
        f"{diagnostic if diagnostic else '  없음'}"
    )

TRAIN_RECORD_PATH = find_required_file(
    "train_records.pkl",
    preferred_paths=[
        PROJECT_ROOT / "data" / "processed" / "train_records.pkl",
        PROJECT_ROOT / "data" / "processed" / "records" / "train_records.pkl",
        PROJECT_ROOT / "records" / "train_records.pkl",
    ],
)

TEST_RECORD_PATH = find_required_file(
    "test_records.pkl",
    preferred_paths=[
        PROJECT_ROOT / "data" / "processed" / "test_records.pkl",
        PROJECT_ROOT / "data" / "processed" / "records" / "test_records.pkl",
        PROJECT_ROOT / "records" / "test_records.pkl",
    ],
)

# 실제 파일 크기로 한 번 더 검증
if TRAIN_RECORD_PATH.stat().st_size == 0:
    raise RuntimeError(f"Train records 파일이 비어 있습니다: {TRAIN_RECORD_PATH}")

if TEST_RECORD_PATH.stat().st_size == 0:
    raise RuntimeError(f"Test records 파일이 비어 있습니다: {TEST_RECORD_PATH}")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / f"fastopic_k30_seeds_42_43_44_{RUN_TIMESTAMP}"
)
MODEL_DIR = RESULT_DIR / "models"
ARRAY_DIR = RESULT_DIR / "arrays"

# 임베딩 cache는 로컬 Colab에 저장해 Drive I/O 병목 감소
CACHE_DIR = Path("/content/fastopic_embedding_cache")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARRAY_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 100)
print("VERIFIED FASTopic PATHS")
print("=" * 100)
print(f"Actual MyDrive : {MY_DRIVE}")
print(f"Project root   : {PROJECT_ROOT}")
print(f"Train records  : {TRAIN_RECORD_PATH}")
print(f"  Size         : {TRAIN_RECORD_PATH.stat().st_size / (1024**2):.2f} MiB")
print(f"Test records   : {TEST_RECORD_PATH}")
print(f"  Size         : {TEST_RECORD_PATH.stat().st_size / (1024**2):.2f} MiB")
print(f"Result dir     : {RESULT_DIR}")
print(f"Local cache    : {CACHE_DIR}")
print("=" * 100)
print("[PASS] 실제 Google Drive와 train/test records를 확인했습니다.")

# --------------------------------------------------------------------------------------
# 5. Load records
# --------------------------------------------------------------------------------------
def load_pickle_records(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        records = obj

    elif isinstance(obj, tuple):
        records = list(obj)

    elif isinstance(obj, dict):
        for key in ["records", "data", "items", "patents"]:
            if key in obj and isinstance(obj[key], (list, tuple)):
                records = list(obj[key])
                break
        else:
            # patent_id -> record 형태
            if all(isinstance(v, dict) for v in obj.values()):
                records = list(obj.values())
            else:
                raise TypeError(
                    f"{path.name}의 dictionary 구조를 자동 해석할 수 없습니다. "
                    f"Top-level keys={list(obj.keys())[:30]}"
                )
    else:
        raise TypeError(f"지원되지 않는 pickle 타입: {type(obj)}")

    if len(records) == 0:
        raise RuntimeError(f"레코드가 비어 있습니다: {path}")

    if not isinstance(records[0], dict):
        raise TypeError(
            f"각 record가 dict여야 합니다. 현재 타입={type(records[0])}"
        )

    return records

train_records = load_pickle_records(TRAIN_RECORD_PATH)
test_records = load_pickle_records(TEST_RECORD_PATH)

print(f"[RECORDS] Train patents: {len(train_records):,}")
print(f"[RECORDS] Test patents : {len(test_records):,}")
print(f"[SAMPLE KEYS] {list(test_records[0].keys())}")

if len(test_records) != EXPECTED_TEST_PATENTS:
    print(
        f"[WARNING] 기존 평가의 test patents={EXPECTED_TEST_PATENTS:,}와 "
        f"현재 값={len(test_records):,}이 다릅니다."
    )

# --------------------------------------------------------------------------------------
# 6. Patent text extraction
#    우선순위: claims/claim_texts -> patent_text/text -> title+abstract
# --------------------------------------------------------------------------------------
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

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (list, tuple)):
        if value and all(isinstance(x, str) for x in value):
            return " ".join(x.strip() for x in value if x.strip())
        return ""

    return ""

def extract_claim_item_text(item):
    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        for key in CLAIM_TEXT_KEYS:
            if key in item:
                text = scalar_to_text(item[key])
                if text:
                    return text

        for key in ["tokens", "words"]:
            if key in item and isinstance(item[key], (list, tuple)):
                return " ".join(str(x) for x in item[key])

    return ""

def extract_patent_text(record):
    # 1. claim들을 연결
    for key in CLAIM_CONTAINER_KEYS:
        if key not in record:
            continue

        claims_obj = record[key]
        claim_texts = []

        if isinstance(claims_obj, (list, tuple)):
            for claim in claims_obj:
                text = extract_claim_item_text(claim)
                if text:
                    claim_texts.append(text)

        elif isinstance(claims_obj, str):
            claim_texts.append(claims_obj.strip())

        elif isinstance(claims_obj, dict):
            for claim in claims_obj.values():
                text = extract_claim_item_text(claim)
                if text:
                    claim_texts.append(text)

        if claim_texts:
            return " ".join(claim_texts)

    # 2. 이미 결합된 patent text
    for key in PATENT_TEXT_KEYS:
        if key in record:
            text = scalar_to_text(record[key])
            if text:
                return text

    # 3. title + abstract fallback
    pieces = []
    for key in ["title", "abstract", "summary"]:
        if key in record:
            text = scalar_to_text(record[key])
            if text:
                pieces.append(text)

    return " ".join(pieces).strip()

def extract_patent_id(record, fallback_index):
    for key in [
        "patent_id", "publication_number", "patent_number",
        "document_id", "doc_id", "id"
    ]:
        if key in record and record[key] is not None:
            return str(record[key])

    return f"record_{fallback_index:08d}"

print("[TEXT] Extracting train patent documents...")
train_docs = [extract_patent_text(r) for r in train_records]

print("[TEXT] Extracting test patent documents...")
test_docs = [extract_patent_text(r) for r in test_records]

train_ids = [
    extract_patent_id(record, i)
    for i, record in enumerate(train_records)
]
test_ids = [
    extract_patent_id(record, i)
    for i, record in enumerate(test_records)
]

empty_train = [i for i, text in enumerate(train_docs) if not text.strip()]
empty_test = [i for i, text in enumerate(test_docs) if not text.strip()]

if empty_train:
    raise RuntimeError(
        f"Train에서 빈 문서가 {len(empty_train):,}개 발견됐습니다. "
        f"첫 index={empty_train[:10]}, sample keys={list(train_records[empty_train[0]].keys())}"
    )

if empty_test:
    raise RuntimeError(
        f"Test에서 빈 문서가 {len(empty_test):,}개 발견됐습니다. "
        f"첫 index={empty_test[:10]}, sample keys={list(test_records[empty_test[0]].keys())}"
    )

train_word_lengths = np.asarray([len(x.split()) for x in train_docs])
test_word_lengths = np.asarray([len(x.split()) for x in test_docs])

print(
    f"[TRAIN TEXT] mean={train_word_lengths.mean():.1f}, "
    f"median={np.median(train_word_lengths):.1f}, "
    f"max={train_word_lengths.max():,} words"
)
print(
    f"[TEST TEXT]  mean={test_word_lengths.mean():.1f}, "
    f"median={np.median(test_word_lengths):.1f}, "
    f"max={test_word_lengths.max():,} words"
)

# --------------------------------------------------------------------------------------
# 7. Primary CPC extraction
#    primary/first CPC code를 Section/Class/Subclass로 변환
# --------------------------------------------------------------------------------------
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

CPC_PATTERN = re.compile(r"\b([A-HY]\d{2}[A-Z])(?:\d+)?(?:/\d+)?\b", re.I)

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
        # CPC object에서 흔히 사용하는 우선 필드
        output = []
        for key in [
            "code", "cpc", "symbol", "classification",
            "subclass", "label", "value"
        ]:
            if key in value:
                output.extend(flatten_values(value[key]))

        if output:
            return output

        for item in value.values():
            output.extend(flatten_values(item))
        return output

    return [str(value)]

def find_cpc_code_in_value(value):
    for item in flatten_values(value):
        matches = CPC_PATTERN.findall(str(item).upper())
        if matches:
            return matches[0].upper()
    return None

def extract_primary_cpc(record):
    # 명시적 primary CPC 우선
    for key in PRIMARY_CPC_KEYS:
        if key in record:
            code = find_cpc_code_in_value(record[key])
            if code:
                return code

    # 없으면 저장 순서상 첫 번째 CPC 사용
    for key in GENERAL_CPC_KEYS:
        if key in record:
            code = find_cpc_code_in_value(record[key])
            if code:
                return code

    # key 이름이 예상과 다를 경우 CPC 문자열 검색
    for key, value in record.items():
        if "cpc" in str(key).lower():
            code = find_cpc_code_in_value(value)
            if code:
                return code

    return None

test_primary_cpc = [extract_primary_cpc(record) for record in test_records]
valid_cpc_mask = np.asarray([code is not None for code in test_primary_cpc])

missing_cpc = int((~valid_cpc_mask).sum())
print(f"[CPC] Valid test labels  : {valid_cpc_mask.sum():,}")
print(f"[CPC] Missing test labels: {missing_cpc:,}")

if valid_cpc_mask.sum() == 0:
    raise RuntimeError(
        "Test record에서 CPC 코드를 찾지 못했습니다. "
        f"Sample keys={list(test_records[0].keys())}"
    )

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

num_sections = len(set(section_labels[valid_cpc_mask]))
num_classes = len(set(class_labels[valid_cpc_mask]))
num_subclasses = len(set(subclass_labels[valid_cpc_mask]))

print(
    f"[CPC LABEL COUNTS] Section={num_sections}, "
    f"Class={num_classes}, Subclass={num_subclasses}"
)

if (
    num_sections != EXPECTED_SECTION_LABELS
    or num_classes != EXPECTED_CLASS_LABELS
    or num_subclasses != EXPECTED_SUBCLASS_LABELS
):
    raise RuntimeError(
        "기존 표의 CPC label count와 일치하지 않습니다.\n"
        f"Expected: Section={EXPECTED_SECTION_LABELS}, "
        f"Class={EXPECTED_CLASS_LABELS}, "
        f"Subclass={EXPECTED_SUBCLASS_LABELS}\n"
        f"Found: Section={num_sections}, "
        f"Class={num_classes}, "
        f"Subclass={num_subclasses}\n"
        "CPC 추출 규칙이 달라진 상태에서 실행하면 기존 표와 직접 비교할 수 없습니다."
    )

print("[CPC SAMPLE]")
shown = 0
for pid, code in zip(test_ids, test_primary_cpc):
    if code:
        print(
            f"  {pid}: {code} -> "
            f"Section={code[0]}, Class={code[:3]}, Subclass={code[:4]}"
        )
        shown += 1
        if shown >= 5:
            break

# --------------------------------------------------------------------------------------
# 8. Cache document embeddings
#    3개 seed 모두 같은 deterministic embeddings 사용
# --------------------------------------------------------------------------------------
def corpus_signature(ids, docs):
    h = hashlib.sha256()
    h.update(str(len(ids)).encode("utf-8"))

    sample_indices = list(range(min(20, len(ids))))
    sample_indices += list(range(max(0, len(ids) - 20), len(ids)))

    for idx in sample_indices:
        h.update(str(ids[idx]).encode("utf-8", errors="ignore"))
        h.update(docs[idx][:1000].encode("utf-8", errors="ignore"))

    return h.hexdigest()[:16]

train_signature = corpus_signature(train_ids, train_docs)
test_signature = corpus_signature(test_ids, test_docs)

safe_model_name = EMBED_MODEL.replace("/", "_")

TRAIN_EMBED_PATH = (
    CACHE_DIR
    / f"train_embeddings_{safe_model_name}_{train_signature}.npy"
)
TEST_EMBED_PATH = (
    CACHE_DIR
    / f"test_embeddings_{safe_model_name}_{test_signature}.npy"
)

def load_or_encode_documents(docs, output_path, split_name):
    if output_path.exists():
        embeddings = np.load(output_path, mmap_mode=None)

        if embeddings.shape[0] == len(docs):
            print(
                f"[EMBED CACHE HIT] {split_name}: "
                f"{output_path} shape={embeddings.shape}"
            )
            return np.asarray(embeddings, dtype=np.float32)

        print(
            f"[EMBED CACHE INVALID] {split_name}: "
            f"expected rows={len(docs)}, found={embeddings.shape[0]}"
        )

    print(f"[EMBED] Loading {EMBED_MODEL}")
    encoder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    print(f"[EMBED] Encoding {split_name}: {len(docs):,} patents")
    embeddings = encoder.encode(
        docs,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    temp_path = output_path.with_suffix(".tmp.npy")
    np.save(temp_path, embeddings)
    os.replace(temp_path, output_path)

    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"[EMBED SAVED] {output_path} "
        f"shape={embeddings.shape}, dtype={embeddings.dtype}"
    )

    return embeddings

train_embeddings = load_or_encode_documents(
    train_docs, TRAIN_EMBED_PATH, "TRAIN"
)
test_embeddings = load_or_encode_documents(
    test_docs, TEST_EMBED_PATH, "TEST"
)

if train_embeddings.ndim != 2 or test_embeddings.ndim != 2:
    raise RuntimeError("Embedding array가 2차원이 아닙니다.")

if train_embeddings.shape[1] != test_embeddings.shape[1]:
    raise RuntimeError(
        f"Train/test embedding dimension 불일치: "
        f"{train_embeddings.shape} vs {test_embeddings.shape}"
    )

print(
    f"[EMBEDDINGS] Train={train_embeddings.shape}, "
    f"Test={test_embeddings.shape}"
)

# --------------------------------------------------------------------------------------
# 9. CPC metrics
# --------------------------------------------------------------------------------------
def predicted_cluster_purity(true_labels, predicted_clusters):
    """
    Pur_p:
    각 predicted cluster에서 가장 많은 true label의 수를 합산 / N
    """
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(predicted_clusters)

    total_correct = 0

    for cluster_id in np.unique(predicted_clusters):
        mask = predicted_clusters == cluster_id
        labels, counts = np.unique(true_labels[mask], return_counts=True)
        total_correct += int(counts.max())

    return float(total_correct / len(true_labels))

def inverse_label_purity(true_labels, predicted_clusters):
    """
    Pur_a:
    각 true label에서 가장 많은 predicted cluster의 수를 합산 / N
    Majority one-cluster baseline이면 1.0이 됨.
    """
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(predicted_clusters)

    total_correct = 0

    for label in np.unique(true_labels):
        mask = true_labels == label
        clusters, counts = np.unique(
            predicted_clusters[mask], return_counts=True
        )
        total_correct += int(counts.max())

    return float(total_correct / len(true_labels))

def evaluate_level(true_labels, predicted_clusters):
    return {
        "pur_p": predicted_cluster_purity(
            true_labels, predicted_clusters
        ),
        "pur_a": inverse_label_purity(
            true_labels, predicted_clusters
        ),
        "nmi": float(normalized_mutual_info_score(
            true_labels,
            predicted_clusters,
            average_method="arithmetic",
        )),
    }

def evaluate_all_cpc_levels(predicted_clusters):
    pred = np.asarray(predicted_clusters)[valid_cpc_mask]

    return {
        "section": evaluate_level(
            section_labels[valid_cpc_mask], pred
        ),
        "class": evaluate_level(
            class_labels[valid_cpc_mask], pred
        ),
        "subclass": evaluate_level(
            subclass_labels[valid_cpc_mask], pred
        ),
    }

# --------------------------------------------------------------------------------------
# 10. Save experiment configuration before training
# --------------------------------------------------------------------------------------
config = {
    "created_at": datetime.now().isoformat(),
    "project_root": str(PROJECT_ROOT),
    "train_record_path": str(TRAIN_RECORD_PATH),
    "test_record_path": str(TEST_RECORD_PATH),
    "train_patents": len(train_docs),
    "test_patents": len(test_docs),
    "valid_test_cpc_labels": int(valid_cpc_mask.sum()),
    "seeds": SEEDS,
    "num_topics": NUM_TOPICS,
    "vocab_size": VOCAB_SIZE,
    "min_doc_count": MIN_DOC_COUNT,
    "max_doc_freq": MAX_DOC_FREQ,
    "epochs": FAST_TOPIC_EPOCHS,
    "learning_rate": FAST_TOPIC_LR,
    "embedding_model": EMBED_MODEL,
    "embedding_dimension": int(train_embeddings.shape[1]),
    "normalize_embeddings": NORMALIZE_EMBEDDINGS,
    "low_memory": True,
    "low_memory_batch_size": LOW_MEMORY_BATCH_SIZE,
    "device": DEVICE,
    "gpu": GPU_NAME,
    "gpu_memory_gib": GPU_MEMORY_GIB,
    "document_unit": "patent",
    "document_text": "concatenated claims",
    "cpc_rule": "primary CPC if available; otherwise first CPC",
    "expected_cpc_counts": {
        "section": EXPECTED_SECTION_LABELS,
        "class": EXPECTED_CLASS_LABELS,
        "subclass": EXPECTED_SUBCLASS_LABELS,
    },
    "cpc_used_in_training": False,
    "cpc_used_in_model_selection": False,
    "test_used_for_training": False,
}

with open(RESULT_DIR / "configuration.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# Test label audit 저장
label_audit = pd.DataFrame({
    "patent_id": test_ids,
    "primary_cpc": test_primary_cpc,
    "section": section_labels,
    "class": class_labels,
    "subclass": subclass_labels,
    "valid_cpc": valid_cpc_mask,
})
label_audit.to_csv(
    RESULT_DIR / "test_cpc_label_audit.csv",
    index=False
)

# --------------------------------------------------------------------------------------
# 11. Train and evaluate FASTopic for three seeds
# --------------------------------------------------------------------------------------
all_seed_results = []
all_rows = []

total_start = time.time()

for seed_index, seed in enumerate(SEEDS, start=1):
    print("\n" + "=" * 100)
    print(f"FASTopic SEED {seed} — {seed_index}/{len(SEEDS)}")
    print("=" * 100)

    set_seed(seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seed_start = time.time()

    # 각 seed에서 동일한 train-only preprocessing 규칙 사용
    preprocess = Preprocess(
        vocab_size=VOCAB_SIZE,
        min_doc_count=MIN_DOC_COUNT,
        max_doc_freq=MAX_DOC_FREQ,
        stopwords="English",
        keep_num=False,
        keep_alphanum=False,
        min_length=3,
        seed=seed,
        verbose=False,
    )

    model = FASTopic(
        num_topics=NUM_TOPICS,
        preprocess=preprocess,
        num_top_words=25,
        device=DEVICE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        doc_embed_model=EMBED_MODEL,
        # 공식 기본 hyperparameters를 명시적으로 고정
        DT_alpha=3.0,
        TW_alpha=2.0,
        theta_temp=1.0,
        low_memory=True,
        low_memory_batch_size=LOW_MEMORY_BATCH_SIZE,
        verbose=True,
        log_interval=10,
    )

    print(
        f"[TRAIN] seed={seed}, documents={len(train_docs):,}, "
        f"epochs={FAST_TOPIC_EPOCHS}"
    )

    # preset embeddings를 사용하므로 seed마다 Transformer embedding을 재계산하지 않음
    top_words, train_theta = model.fit_transform(
        train_docs,
        epochs=FAST_TOPIC_EPOCHS,
        learning_rate=FAST_TOPIC_LR,
        preset_doc_embeddings=train_embeddings,
    )

    print(f"[TEST] Transforming {len(test_docs):,} test patents")

    test_theta = model.transform(
        doc_embeddings=test_embeddings
    )

    train_theta = np.asarray(train_theta, dtype=np.float32)
    test_theta = np.asarray(test_theta, dtype=np.float32)
    beta = np.asarray(model.get_beta(), dtype=np.float32)

    if test_theta.shape != (len(test_docs), NUM_TOPICS):
        raise RuntimeError(
            f"Unexpected test theta shape: {test_theta.shape}"
        )

    predicted_topics = test_theta.argmax(axis=1)
    metrics = evaluate_all_cpc_levels(predicted_topics)

    active_topics = int(np.unique(predicted_topics).size)
    topic_counts = np.bincount(
        predicted_topics,
        minlength=NUM_TOPICS
    )
    max_topic_share = float(topic_counts.max() / topic_counts.sum())

    elapsed_seconds = time.time() - seed_start
    peak_gpu_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    seed_result = {
        "seed": seed,
        "metrics": metrics,
        "active_topics": active_topics,
        "max_topic_share": max_topic_share,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_gib": peak_gpu_gib,
        "train_theta_shape": list(train_theta.shape),
        "test_theta_shape": list(test_theta.shape),
        "beta_shape": list(beta.shape),
    }
    all_seed_results.append(seed_result)

    for level in ["section", "class", "subclass"]:
        row = {
            "seed": seed,
            "level": level,
            "pur_p": metrics[level]["pur_p"],
            "pur_a": metrics[level]["pur_a"],
            "nmi": metrics[level]["nmi"],
            "active_topics": active_topics,
            "max_topic_share": max_topic_share,
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_gib": peak_gpu_gib,
        }
        all_rows.append(row)

    # Arrays
    np.save(
        ARRAY_DIR / f"seed_{seed}_train_theta.npy",
        train_theta
    )
    np.save(
        ARRAY_DIR / f"seed_{seed}_test_theta.npy",
        test_theta
    )
    np.save(
        ARRAY_DIR / f"seed_{seed}_beta.npy",
        beta
    )
    np.save(
        ARRAY_DIR / f"seed_{seed}_test_topic_assignments.npy",
        predicted_topics.astype(np.int16)
    )

    # Top words
    top_word_rows = []
    for topic_id, words in enumerate(top_words):
        top_word_rows.append({
            "seed": seed,
            "topic_id": topic_id,
            "top_words": words,
        })

    pd.DataFrame(top_word_rows).to_csv(
        RESULT_DIR / f"seed_{seed}_top_words.csv",
        index=False
    )

    # 공식 FASTopic 저장 형식
    model_path = MODEL_DIR / f"fastopic_seed_{seed}.zip"
    try:
        model.save(str(model_path))
        print(f"[MODEL SAVED] {model_path}")
    except Exception as save_error:
        print(f"[WARNING] 공식 model.save 실패: {save_error}")

        # 최소 복구용 state 저장
        torch.save(
            {
                "seed": seed,
                "model_state_dict": model.model.state_dict(),
                "vocab": model.vocab,
                "num_topics": NUM_TOPICS,
                "embedding_model": EMBED_MODEL,
                "config": config,
            },
            MODEL_DIR / f"fastopic_seed_{seed}_state.pt",
        )

    with open(
        RESULT_DIR / f"seed_{seed}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(seed_result, f, indent=2, ensure_ascii=False)

    print("-" * 100)
    print(f"[SEED {seed} RESULT]")
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
        f"time={elapsed_seconds / 60:.1f} min | "
        f"GPU peak={peak_gpu_gib:.2f} GiB"
    )
    print("-" * 100)

    del model
    del preprocess
    del train_theta
    del test_theta
    del beta
    del predicted_topics

    gc.collect()
    torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------
# 12. Aggregate mean and sample standard deviation
# --------------------------------------------------------------------------------------
seed_df = pd.DataFrame(all_rows)
seed_df.to_csv(
    RESULT_DIR / "fastopic_seed_level_results.csv",
    index=False
)

summary_rows = []

for level in ["section", "class", "subclass"]:
    level_df = seed_df[seed_df["level"] == level]

    summary_row = {"level": level}

    for metric in ["pur_p", "pur_a", "nmi"]:
        values = level_df[metric].to_numpy(dtype=float)

        summary_row[f"{metric}_mean"] = float(values.mean())
        summary_row[f"{metric}_std"] = float(values.std(ddof=1))
        summary_row[f"{metric}_min"] = float(values.min())
        summary_row[f"{metric}_max"] = float(values.max())

    summary_rows.append(summary_row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    RESULT_DIR / "fastopic_3seed_summary.csv",
    index=False
)

final_result = {
    "configuration": config,
    "seed_results": all_seed_results,
    "summary": summary_rows,
    "total_elapsed_seconds": time.time() - total_start,
}

with open(
    RESULT_DIR / "fastopic_3seed_complete_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)

# --------------------------------------------------------------------------------------
# 13. Final report and LaTeX-ready row
# --------------------------------------------------------------------------------------
def summary_value(level, metric, stat="mean"):
    row = summary_df[summary_df["level"] == level].iloc[0]
    return float(row[f"{metric}_{stat}"])

print("\n" + "=" * 100)
print("FASTopic 3-SEED FINAL CPC ALIGNMENT")
print("=" * 100)

for level_title, level in [
    ("SECTION", "section"),
    ("CLASS", "class"),
    ("SUBCLASS", "subclass"),
]:
    print(f"\n[{level_title}]")
    print(
        f"Pur_p = {summary_value(level, 'pur_p'):.4f} "
        f"± {summary_value(level, 'pur_p', 'std'):.4f}"
    )
    print(
        f"Pur_a = {summary_value(level, 'pur_a'):.4f} "
        f"± {summary_value(level, 'pur_a', 'std'):.4f}"
    )
    print(
        f"NMI   = {summary_value(level, 'nmi'):.4f} "
        f"± {summary_value(level, 'nmi', 'std'):.4f}"
    )

latex_values = []

for level in ["section", "class", "subclass"]:
    for metric in ["pur_p", "pur_a", "nmi"]:
        latex_values.append(summary_value(level, metric))

latex_row = (
    "FASTopic "
    + " & ".join(f"{value:.4f}" for value in latex_values)
    + r" \\"
)

latex_row_with_std = (
    "FASTopic "
    + " & ".join(
        f"${summary_value(level, metric):.4f}"
        f"\\pm{summary_value(level, metric, 'std'):.4f}$"
        for level in ["section", "class", "subclass"]
        for metric in ["pur_p", "pur_a", "nmi"]
    )
    + r" \\"
)

with open(
    RESULT_DIR / "latex_row_mean_only.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_row + "\n")

with open(
    RESULT_DIR / "latex_row_mean_std.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_row_with_std + "\n")

print("\n" + "-" * 100)
print("[LATEX ROW — MEAN ONLY]")
print(latex_row)

print("\n[LATEX ROW — MEAN ± STD]")
print(latex_row_with_std)

print("\n[PER-SEED RESULTS]")
display(
    seed_df[
        ["seed", "level", "pur_p", "pur_a", "nmi",
         "active_topics", "max_topic_share"]
    ]
)

print("\n[3-SEED SUMMARY]")
display(summary_df)

print("\n" + "=" * 100)
print("[COMPLETE]")
print(f"Total elapsed : {(time.time() - total_start) / 3600:.2f} hours")
print(f"Result dir    : {RESULT_DIR}")
print(f"Main JSON     : {RESULT_DIR / 'fastopic_3seed_complete_results.json'}")
print(f"Summary CSV   : {RESULT_DIR / 'fastopic_3seed_summary.csv'}")
print(f"Seed CSV      : {RESULT_DIR / 'fastopic_seed_level_results.csv'}")
print("=" * 100)
