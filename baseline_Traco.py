# ============================================================
# TopMost/TraCo robust recovery:
#   1) lightweight sparse clone
#   2) automatic ZIP fallback
#   3) broken pip-installed topmost bypass
# No runtime restart required
# ============================================================

import os
import sys
import shutil
import subprocess
import importlib
import inspect
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/bobxwu/TopMost.git"
ZIP_URL = "https://codeload.github.com/bobxwu/TopMost/zip/refs/heads/main"

TOPMOST_DIR = Path("/content/TopMost_official")
ZIP_PATH = Path("/tmp/TopMost-main.zip")

def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if result.stdout:
        print(result.stdout[-5000:])
    return result.returncode == 0

# ------------------------------------------------------------
# 1. Remove incomplete clone and stale imported modules
# ------------------------------------------------------------
print("[CLEAN] Removing incomplete TopMost clone")

if TOPMOST_DIR.exists():
    shutil.rmtree(TOPMOST_DIR, ignore_errors=True)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

for module_name in list(sys.modules):
    if module_name == "topmost" or module_name.startswith("topmost."):
        del sys.modules[module_name]

importlib.invalidate_caches()

# ------------------------------------------------------------
# 2. Lightweight sparse clone using HTTP/1.1
# ------------------------------------------------------------
print("[DOWNLOAD 1/2] Trying lightweight sparse clone...")

clone_ok = run_command([
    "git",
    "-c", "http.version=HTTP/1.1",
    "-c", "core.compression=0",
    "clone",
    "--depth", "1",
    "--filter=blob:none",
    "--sparse",
    REPO_URL,
    str(TOPMOST_DIR)
])

if clone_ok:
    print("[SPARSE CHECKOUT] Downloading only the topmost source directory")

    checkout_ok = run_command([
        "git",
        "-C", str(TOPMOST_DIR),
        "sparse-checkout",
        "set",
        "topmost"
    ])

    clone_ok = clone_ok and checkout_ok

# ------------------------------------------------------------
# 3. ZIP fallback if git clone failed
# ------------------------------------------------------------
expected_traco = (
    TOPMOST_DIR
    / "topmost"
    / "models"
    / "hierarchical"
    / "TraCo"
    / "TraCo.py"
)

if not clone_ok or not expected_traco.is_file():
    print("[DOWNLOAD 2/2] Git clone failed; switching to ZIP fallback")

    if TOPMOST_DIR.exists():
        shutil.rmtree(TOPMOST_DIR, ignore_errors=True)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    download_ok = run_command([
        "curl",
        "-L",
        "--http1.1",
        "--retry", "6",
        "--retry-delay", "3",
        "--retry-all-errors",
        "--connect-timeout", "30",
        "--max-time", "900",
        "-o", str(ZIP_PATH),
        ZIP_URL
    ])

    if not download_ok or not ZIP_PATH.is_file():
        raise RuntimeError(
            "Git clone과 ZIP 다운로드가 모두 실패했습니다. "
            "Colab의 GitHub 연결 문제일 가능성이 큽니다."
        )

    print(
        f"[ZIP] Downloaded: {ZIP_PATH} "
        f"({ZIP_PATH.stat().st_size / 1024**2:.1f} MiB)"
    )

    # Extract only official TopMost Python source, not datasets/tutorials.
    TOPMOST_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = [
            item for item in archive.infolist()
            if (
                item.filename.startswith("TopMost-main/topmost/")
                or item.filename == "TopMost-main/pyproject.toml"
            )
        ]

        if not members:
            raise RuntimeError(
                "다운로드한 ZIP에서 TopMost 소스 파일을 찾지 못했습니다."
            )

        for item in members:
            relative = Path(item.filename).relative_to("TopMost-main")
            destination = TOPMOST_DIR / relative

            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(item) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)

    print("[ZIP] Official TopMost source extracted successfully")

# ------------------------------------------------------------
# 4. Verify required official TraCo files
# ------------------------------------------------------------
required_files = [
    TOPMOST_DIR / "topmost" / "__init__.py",
    TOPMOST_DIR / "topmost" / "models" / "hierarchical" / "TraCo" / "TraCo.py",
    TOPMOST_DIR / "topmost" / "models" / "hierarchical" / "TraCo" / "TPD.py",
    TOPMOST_DIR / "topmost" / "models" / "hierarchical" / "TraCo" / "CDDecoder.py",
    TOPMOST_DIR / "topmost" / "preprocess" / "preprocess.py",
]

missing_files = [str(path) for path in required_files if not path.is_file()]

if missing_files:
    raise RuntimeError(
        "Official TopMost 다운로드가 불완전합니다:\n"
        + "\n".join(missing_files)
    )

print("[VERIFY] Required TraCo files are present")

# ------------------------------------------------------------
# 5. Force Python to use source clone instead of broken pip copy
# ------------------------------------------------------------
official_root = str(TOPMOST_DIR.resolve())

# Remove duplicate path and place official source first.
sys.path = [
    path for path in sys.path
    if str(Path(path or ".").resolve()) != official_root
]
sys.path.insert(0, official_root)

# Remove any half-loaded pip-package modules again.
for module_name in list(sys.modules):
    if module_name == "topmost" or module_name.startswith("topmost."):
        del sys.modules[module_name]

importlib.invalidate_caches()

# ------------------------------------------------------------
# 6. Official imports
# ------------------------------------------------------------
print("[IMPORT] Loading official TopMost source")

import topmost
from topmost import TraCo
from topmost.preprocess.preprocess import Preprocess

actual_topmost_file = Path(topmost.__file__).resolve()
expected_package_root = (TOPMOST_DIR / "topmost").resolve()

if expected_package_root not in actual_topmost_file.parents:
    raise RuntimeError(
        "여전히 site-packages의 잘못된 topmost가 로드되었습니다.\n"
        f"Loaded: {actual_topmost_file}\n"
        f"Expected root: {expected_package_root}"
    )

# ------------------------------------------------------------
# 7. API validation
# ------------------------------------------------------------
signature = inspect.signature(TraCo.__init__)

if "num_topics_list" not in signature.parameters:
    raise RuntimeError(
        "불러온 TraCo API가 공식 hierarchical TraCo API와 다릅니다.\n"
        f"Signature: {signature}"
    )

print("\n" + "=" * 72)
print("[SUCCESS] Official TopMost TraCo is ready")
print("=" * 72)
print(f"TopMost source : {topmost.__file__}")
print(f"TraCo source   : {inspect.getfile(TraCo)}")
print(f"TraCo API      : {signature}")
print(f"Preprocess     : {inspect.getfile(Preprocess)}")
print("=" * 72)
print("이제 원래 TraCo 벤치마크 셀을 다시 실행하면 됩니다.")


# ======================================================================================
# TraCo 3-SEED PATENT CPC BENCHMARK
# ONE-CELL GOOGLE COLAB — NVIDIA T4/L4
#
# Model     : TraCo (AAAI 2024, official TopMost implementation)
# Hierarchy : 10 -> 50 -> 200 topics
# Seeds     : 42, 43, 44
#
# Matched CPC evaluation:
#   Section  <- hierarchy level 0 (10 topics)
#   Class    <- hierarchy level 1 (50 topics)
#   Subclass <- hierarchy level 2 (200 topics)
#
# Training:
#   - train-only vocabulary
#   - CPC labels are never used for training or model selection
#   - test documents are only transformed after training
#   - automatic checkpoint/resume
# ======================================================================================

# --------------------------------------------------------------------------------------
# 0. Environment and package installation
# --------------------------------------------------------------------------------------
import os
import sys
import gc
import re
import json
import time
import pickle
import random
import shutil
import inspect
import warnings
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

warnings.filterwarnings("ignore")


def install_packages():
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
    ]

    missing = []

    for module_name, package_name in required:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)

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

    # TraCo가 포함된 TopMost인지 확인
    install_topmost = False

    try:
        import topmost
        from topmost import TraCo

        signature = inspect.signature(TraCo.__init__)

        if "num_topics_list" not in signature.parameters:
            install_topmost = True

    except Exception:
        install_topmost = True

    if install_topmost:
        print("[INSTALL] Official TopMost GitHub version with TraCo")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--no-cache-dir",
            "git+https://github.com/BobXWu/TopMost.git",
        ])


install_packages()

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from tqdm.auto import tqdm
from sklearn.metrics import normalized_mutual_info_score

from topmost import TraCo
from topmost.preprocess.preprocess import Preprocess


# --------------------------------------------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------------------------------------------
SEEDS = [42, 43, 44]

# Official TraCo hierarchy from coarse to fine
NUM_TOPICS_LIST = [10, 50, 200]

# CPC granularity -> hierarchy level
MATCHED_LEVEL_MAP = {
    "section": 0,
    "class": 1,
    "subclass": 2,
}

VOCAB_SIZE = 8000
MIN_DOC_COUNT = 10
MAX_DOC_FREQ = 0.70

# Official TraCo configuration
EPOCHS = 200
BATCH_SIZE = 200
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

NUM_TOP_WORDS = 25
INFERENCE_BATCH_SIZE = 512

CHECKPOINT_INTERVAL = 10
GRADIENT_CLIP_NORM = 5.0

EXPECTED_TEST_PATENTS = 9881
EXPECTED_SECTIONS = 9
EXPECTED_CLASSES = 121
EXPECTED_SUBCLASSES = 466

STRICT_EXPECTED_CPC_COUNTS = True

DEVICE = torch.device("cuda:0")


# --------------------------------------------------------------------------------------
# 2. GPU validation and reproducibility
# --------------------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 없습니다. Colab 런타임 유형을 T4 또는 L4 GPU로 설정하세요."
    )

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GIB = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = False

if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = False

try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


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
print("TraCo 3-SEED PATENT CPC BENCHMARK")
print("=" * 100)
print(f"GPU                     : {GPU_NAME}")
print(f"GPU memory              : {GPU_MEMORY_GIB:.2f} GiB")
print(f"PyTorch                 : {torch.__version__}")
print(f"Device                  : {DEVICE}")
print(f"Seeds                   : {SEEDS}")
print(f"Topic hierarchy         : {NUM_TOPICS_LIST}")
print(f"Vocabulary              : {VOCAB_SIZE:,}")
print(f"Epochs                  : {EPOCHS}")
print(f"Batch size              : {BATCH_SIZE}")
print(f"Learning rate           : {LEARNING_RATE}")
print(f"Encoder units           : {ENCODER_UNITS}")
print(f"Embedding size          : {EMBED_SIZE}")
print(f"Beta temperature        : {BETA_TEMP}")
print(f"TPD weight              : {WEIGHT_LOSS_TPD}")
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
    Path("/content/traco_drive/MyDrive/depth_ot_patent"),
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DRIVE_MOUNT = Path(
        f"/content/traco_drive_{timestamp}"
    )

    print(f"[MOUNT] Google Drive -> {DRIVE_MOUNT}")

    drive.mount(
        str(DRIVE_MOUNT),
        force_remount=False,
    )

    MY_DRIVE = DRIVE_MOUNT / "MyDrive"

    if not MY_DRIVE.is_dir():
        raise RuntimeError(
            f"MyDrive를 찾을 수 없습니다: {MY_DRIVE}"
        )

    direct_project = (
        MY_DRIVE / "depth_ot_patent"
    )

    if contains_test_records(direct_project):
        PROJECT_ROOT = direct_project

    else:
        print(
            "[SEARCH] MyDrive에서 test_records.pkl을 포함한 "
            "depth_ot_patent 프로젝트 검색"
        )

        project_candidates = []

        for test_path in MY_DRIVE.rglob(
            "test_records.pkl"
        ):
            for parent in test_path.parents:
                if parent.name == "depth_ot_patent":
                    project_candidates.append(parent)
                    break

        unique_candidates = []
        seen = set()

        for candidate in project_candidates:
            key = str(candidate.resolve())

            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        if not unique_candidates:
            raise FileNotFoundError(
                f"{MY_DRIVE}에서 test_records.pkl을 포함한 "
                "depth_ot_patent 프로젝트를 찾지 못했습니다."
            )

        if len(unique_candidates) > 1:
            print("[WARNING] 프로젝트 후보가 여러 개입니다:")

            for index, candidate in enumerate(
                unique_candidates
            ):
                print(f"  [{index}] {candidate}")

        exact_candidates = [
            candidate
            for candidate in unique_candidates
            if candidate == direct_project
        ]

        PROJECT_ROOT = (
            exact_candidates[0]
            if exact_candidates
            else unique_candidates[0]
        )

        print(f"[SELECTED] {PROJECT_ROOT}")

if (
    PROJECT_ROOT is None
    or not contains_test_records(PROJECT_ROOT)
):
    raise RuntimeError(
        f"프로젝트 검증에 실패했습니다: {PROJECT_ROOT}"
    )

print(f"[PROJECT ROOT VERIFIED] {PROJECT_ROOT}")


# --------------------------------------------------------------------------------------
# 4. Required file discovery
# --------------------------------------------------------------------------------------
def find_required_file(
    filename,
    preferred_paths=None,
):
    preferred_paths = preferred_paths or []

    for candidate in preferred_paths:
        candidate = Path(candidate)

        if candidate.is_file():
            print(f"[FOUND] {filename}: {candidate}")
            return candidate

    project_matches = sorted(
        path
        for path in PROJECT_ROOT.rglob(filename)
        if path.is_file()
    )

    if len(project_matches) == 1:
        print(
            f"[FOUND] {filename}: "
            f"{project_matches[0]}"
        )
        return project_matches[0]

    if len(project_matches) > 1:
        print(
            f"[WARNING] {filename} 후보가 여러 개입니다:"
        )

        for index, path in enumerate(
            project_matches
        ):
            print(f"  [{index}] {path}")

        processed_matches = [
            path
            for path in project_matches
            if "data/processed" in str(path).replace(
                "\\",
                "/",
            )
        ]

        selected = (
            processed_matches[0]
            if processed_matches
            else project_matches[0]
        )

        print(f"[SELECTED] {selected}")
        return selected

    print(
        f"[SEARCH] MyDrive 전체에서 {filename} 검색"
    )

    drive_matches = sorted(
        path
        for path in MY_DRIVE.rglob(filename)
        if path.is_file()
    )

    if drive_matches:
        project_named_matches = [
            path
            for path in drive_matches
            if "depth_ot_patent" in path.parts
        ]

        selected = (
            project_named_matches[0]
            if project_named_matches
            else drive_matches[0]
        )

        print(f"[FOUND IN MYDRIVE] {selected}")
        return selected

    raise FileNotFoundError(
        f"필수 파일을 찾지 못했습니다: {filename}"
    )


TRAIN_RECORD_PATH = find_required_file(
    "train_records.pkl",
    [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "train_records.pkl",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "records"
        / "train_records.pkl",

        PROJECT_ROOT
        / "records"
        / "train_records.pkl",
    ],
)

TEST_RECORD_PATH = find_required_file(
    "test_records.pkl",
    [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test_records.pkl",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "records"
        / "test_records.pkl",

        PROJECT_ROOT
        / "records"
        / "test_records.pkl",
    ],
)

if TRAIN_RECORD_PATH.stat().st_size <= 0:
    raise RuntimeError(
        f"Train records 파일이 비어 있습니다: "
        f"{TRAIN_RECORD_PATH}"
    )

if TEST_RECORD_PATH.stat().st_size <= 0:
    raise RuntimeError(
        f"Test records 파일이 비어 있습니다: "
        f"{TEST_RECORD_PATH}"
    )


# Fixed directory: interrupted runs resume automatically
RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "traco_10_50_200_seeds_42_43_44"
)

CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
ARRAY_DIR = RESULT_DIR / "arrays"
TOPIC_DIR = RESULT_DIR / "topics"
HIERARCHY_DIR = RESULT_DIR / "hierarchies"

CACHE_DIR = (
    PROJECT_ROOT
    / "cache"
    / "traco_trainonly_vocab8000"
)

for directory in [
    RESULT_DIR,
    CHECKPOINT_DIR,
    ARRAY_DIR,
    TOPIC_DIR,
    HIERARCHY_DIR,
    CACHE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

print("\n" + "=" * 100)
print("VERIFIED TraCo PATHS")
print("=" * 100)
print(f"MyDrive        : {MY_DRIVE}")
print(f"Project root   : {PROJECT_ROOT}")
print(f"Train records  : {TRAIN_RECORD_PATH}")
print(f"Test records   : {TEST_RECORD_PATH}")
print(f"Result dir     : {RESULT_DIR}")
print(f"Checkpoint dir : {CHECKPOINT_DIR}")
print(f"Cache dir      : {CACHE_DIR}")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 5. Record loading
# --------------------------------------------------------------------------------------
def load_pickle_records(path):
    with open(path, "rb") as handle:
        obj = pickle.load(handle)

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
                records = list(obj[key])
                break

        if records is None:
            if all(
                isinstance(value, dict)
                for value in obj.values()
            ):
                records = list(obj.values())

            else:
                raise TypeError(
                    f"{path.name}의 dictionary 구조를 "
                    "자동 해석할 수 없습니다. "
                    f"keys={list(obj.keys())[:30]}"
                )

    else:
        raise TypeError(
            f"지원되지 않는 pickle 타입: {type(obj)}"
        )

    if not records:
        raise RuntimeError(
            f"레코드가 비어 있습니다: {path}"
        )

    if not isinstance(records[0], dict):
        raise TypeError(
            "각 record는 dictionary여야 합니다. "
            f"현재 타입={type(records[0])}"
        )

    return records


print("[LOAD] Train records")
train_records = load_pickle_records(
    TRAIN_RECORD_PATH
)

print("[LOAD] Test records")
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

if len(test_records) != EXPECTED_TEST_PATENTS:
    print(
        "[WARNING] 기존 평가의 test patent 수와 다릅니다. "
        f"Expected={EXPECTED_TEST_PATENTS:,}, "
        f"Found={len(test_records):,}"
    )


# --------------------------------------------------------------------------------------
# 6. Patent claim text extraction
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
        if value and all(
            isinstance(item, str)
            for item in value
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
                text = scalar_to_text(item[key])

                if text:
                    return text

        for key in ["tokens", "words"]:
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
    # Concatenated claims first
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
                text = extract_claim_item_text(claim)

                if text:
                    claim_texts.append(text)

        elif isinstance(claims_object, dict):
            for claim in claims_object.values():
                text = extract_claim_item_text(claim)

                if text:
                    claim_texts.append(text)

        elif isinstance(claims_object, str):
            if claims_object.strip():
                claim_texts.append(
                    claims_object.strip()
                )

        if claim_texts:
            return " ".join(claim_texts)

    # Already concatenated patent text
    for key in PATENT_TEXT_KEYS:
        if key in record:
            text = scalar_to_text(record[key])

            if text:
                return text

    # Last fallback
    pieces = []

    for key in [
        "title",
        "abstract",
        "summary",
    ]:
        if key in record:
            text = scalar_to_text(record[key])

            if text:
                pieces.append(text)

    return " ".join(pieces).strip()


def extract_patent_id(record, index):
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
            return str(record[key])

    return f"record_{index:08d}"


print("[TEXT] Extracting train patent documents")

train_docs = [
    extract_patent_text(record)
    for record in tqdm(
        train_records,
        desc="Train text",
    )
]

print("[TEXT] Extracting test patent documents")

test_docs = [
    extract_patent_text(record)
    for record in tqdm(
        test_records,
        desc="Test text",
    )
]

train_ids = [
    extract_patent_id(record, index)
    for index, record in enumerate(train_records)
]

test_ids = [
    extract_patent_id(record, index)
    for index, record in enumerate(test_records)
]

empty_train = [
    index
    for index, text in enumerate(train_docs)
    if not text.strip()
]

empty_test = [
    index
    for index, text in enumerate(test_docs)
    if not text.strip()
]

if empty_train:
    raise RuntimeError(
        f"빈 train 문서가 {len(empty_train):,}개 있습니다. "
        f"sample={empty_train[:10]}"
    )

if empty_test:
    raise RuntimeError(
        f"빈 test 문서가 {len(empty_test):,}개 있습니다. "
        f"sample={empty_test[:10]}"
    )

train_lengths = np.asarray([
    len(text.split())
    for text in train_docs
])

test_lengths = np.asarray([
    len(text.split())
    for text in test_docs
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
# 7. Primary CPC extraction
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
        prioritized = []

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
                prioritized.extend(
                    flatten_values(value[key])
                )

        if prioritized:
            return prioritized

        output = []

        for item in value.values():
            output.extend(flatten_values(item))

        return output

    return [str(value)]


def find_cpc_code(value):
    for item in flatten_values(value):
        matches = CPC_PATTERN.findall(
            str(item).upper()
        )

        if matches:
            return matches[0].upper()

    return None


def extract_primary_cpc(record):
    for key in PRIMARY_CPC_KEYS:
        if key in record:
            code = find_cpc_code(record[key])

            if code:
                return code

    if "subclass" in record:
        code = find_cpc_code(
            record["subclass"]
        )

        if code:
            return code

    for key in GENERAL_CPC_KEYS:
        if key in record:
            code = find_cpc_code(record[key])

            if code:
                return code

    for key, value in record.items():
        if "cpc" in str(key).lower():
            code = find_cpc_code(value)

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

CPC_LABELS = {
    "section": section_labels,
    "class": class_labels,
    "subclass": subclass_labels,
}

num_sections = len(
    set(section_labels[valid_cpc_mask])
)
num_classes = len(
    set(class_labels[valid_cpc_mask])
)
num_subclasses = len(
    set(subclass_labels[valid_cpc_mask])
)

print(
    f"[CPC] Valid test labels  : "
    f"{int(valid_cpc_mask.sum()):,}"
)
print(
    f"[CPC] Missing test labels: "
    f"{int((~valid_cpc_mask).sum()):,}"
)
print(
    f"[CPC COUNTS] Section={num_sections}, "
    f"Class={num_classes}, "
    f"Subclass={num_subclasses}"
)

expected_match = (
    num_sections == EXPECTED_SECTIONS
    and num_classes == EXPECTED_CLASSES
    and num_subclasses == EXPECTED_SUBCLASSES
)

if not expected_match:
    message = (
        "기존 표의 CPC category 수와 일치하지 않습니다.\n"
        f"Expected={EXPECTED_SECTIONS}/"
        f"{EXPECTED_CLASSES}/"
        f"{EXPECTED_SUBCLASSES}\n"
        f"Found={num_sections}/"
        f"{num_classes}/"
        f"{num_subclasses}"
    )

    if STRICT_EXPECTED_CPC_COUNTS:
        raise RuntimeError(message)

    print(f"[WARNING] {message}")

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
    index=False,
)


# --------------------------------------------------------------------------------------
# 8. Train-only vocabulary and sparse BoW
# --------------------------------------------------------------------------------------
TRAIN_BOW_PATH = CACHE_DIR / "train_bow.npz"
TEST_BOW_PATH = CACHE_DIR / "test_bow.npz"
VOCAB_PATH = CACHE_DIR / "vocabulary.json"
CACHE_INFO_PATH = CACHE_DIR / "cache_information.json"


def source_signature():
    payload = {
        "train_path": str(TRAIN_RECORD_PATH),
        "train_size": int(
            TRAIN_RECORD_PATH.stat().st_size
        ),
        "test_path": str(TEST_RECORD_PATH),
        "test_size": int(
            TEST_RECORD_PATH.stat().st_size
        ),
        "train_patents": len(train_docs),
        "test_patents": len(test_docs),
        "vocab_size": VOCAB_SIZE,
        "min_doc_count": MIN_DOC_COUNT,
        "max_doc_freq": MAX_DOC_FREQ,
    }

    return payload


def cache_is_valid():
    required = [
        TRAIN_BOW_PATH,
        TEST_BOW_PATH,
        VOCAB_PATH,
        CACHE_INFO_PATH,
    ]

    if not all(path.exists() for path in required):
        return False

    try:
        train_bow_cached = sp.load_npz(
            TRAIN_BOW_PATH
        )
        test_bow_cached = sp.load_npz(
            TEST_BOW_PATH
        )

        with open(
            VOCAB_PATH,
            "r",
            encoding="utf-8",
        ) as handle:
            vocabulary_cached = json.load(handle)

        with open(
            CACHE_INFO_PATH,
            "r",
            encoding="utf-8",
        ) as handle:
            cache_information = json.load(handle)

        return (
            train_bow_cached.shape
            == (len(train_docs), VOCAB_SIZE)
            and test_bow_cached.shape
            == (len(test_docs), VOCAB_SIZE)
            and len(vocabulary_cached)
            == VOCAB_SIZE
            and cache_information.get(
                "source_signature"
            )
            == source_signature()
        )

    except Exception as error:
        print(
            f"[CACHE INVALID] {repr(error)}"
        )
        return False


if cache_is_valid():
    print("[BOW CACHE HIT] Loading cached BoW")

    train_bow = (
        sp.load_npz(TRAIN_BOW_PATH)
        .tocsr()
        .astype(np.float32)
    )

    test_bow = (
        sp.load_npz(TEST_BOW_PATH)
        .tocsr()
        .astype(np.float32)
    )

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8",
    ) as handle:
        vocab = json.load(handle)

else:
    print(
        "[PREPROCESS] Building train-only vocabulary and BoW"
    )
    print(
        f"[PREPROCESS] vocab={VOCAB_SIZE}, "
        f"min_df={MIN_DOC_COUNT}, "
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

    # Vocabulary fitting occurs only on training documents.
    train_result = preprocess.preprocess(
        train_docs,
        pretrained_WE=False,
    )

    vocab = list(train_result["vocab"])

    train_bow = (
        train_result["train_bow"]
        .tocsr()
        .astype(np.float32)
    )

    if len(vocab) != VOCAB_SIZE:
        raise RuntimeError(
            "Vocabulary size mismatch: "
            f"expected={VOCAB_SIZE}, "
            f"found={len(vocab)}"
        )

    print(
        "[PREPROCESS] Parsing test documents "
        "with fixed train vocabulary"
    )

    _, test_bow = preprocess.parse(
        test_docs,
        vocab,
    )

    test_bow = (
        test_bow
        .tocsr()
        .astype(np.float32)
    )

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

    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            vocab,
            handle,
            ensure_ascii=False,
        )

    cache_information = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_signature": source_signature(),
        "train_shape": list(train_bow.shape),
        "test_shape": list(test_bow.shape),
        "vocab_size": len(vocab),
        "vocabulary_train_only": True,
    }

    with open(
        CACHE_INFO_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            cache_information,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    del preprocess
    del train_result
    gc.collect()


if train_bow.shape != (
    len(train_docs),
    VOCAB_SIZE,
):
    raise RuntimeError(
        f"Unexpected train BoW shape: "
        f"{train_bow.shape}"
    )

if test_bow.shape != (
    len(test_docs),
    VOCAB_SIZE,
):
    raise RuntimeError(
        f"Unexpected test BoW shape: "
        f"{test_bow.shape}"
    )

empty_train_bow = np.asarray(
    train_bow.sum(axis=1)
).reshape(-1) <= 0

empty_test_bow = np.asarray(
    test_bow.sum(axis=1)
).reshape(-1) <= 0

print(f"[BOW] Train shape: {train_bow.shape}")
print(f"[BOW] Test shape : {test_bow.shape}")
print(
    f"[BOW] Empty train rows: "
    f"{int(empty_train_bow.sum()):,}"
)
print(
    f"[BOW] Empty test rows : "
    f"{int(empty_test_bow.sum()):,}"
)
print(
    f"[BOW] Train average token count: "
    f"{float(train_bow.sum() / train_bow.shape[0]):.2f}"
)
print(
    f"[BOW] Test average token count : "
    f"{float(test_bow.sum() / test_bow.shape[0]):.2f}"
)

# Raw documents and records are no longer needed.
del train_docs
del test_docs
del train_records
del test_records
del train_lengths
del test_lengths

gc.collect()


# --------------------------------------------------------------------------------------
# 9. CPC alignment metrics
# --------------------------------------------------------------------------------------
def predicted_cluster_purity(
    true_labels,
    predicted_clusters,
):
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(
        predicted_clusters
    )

    correct = 0

    for cluster_id in np.unique(
        predicted_clusters
    ):
        mask = (
            predicted_clusters == cluster_id
        )

        _, counts = np.unique(
            true_labels[mask],
            return_counts=True,
        )

        correct += int(counts.max())

    return float(
        correct / len(true_labels)
    )


def inverse_label_purity(
    true_labels,
    predicted_clusters,
):
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(
        predicted_clusters
    )

    correct = 0

    for label in np.unique(true_labels):
        mask = true_labels == label

        _, counts = np.unique(
            predicted_clusters[mask],
            return_counts=True,
        )

        correct += int(counts.max())

    return float(
        correct / len(true_labels)
    )


def evaluate_level(
    true_labels,
    predicted_clusters,
):
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


def evaluate_matched_cpc(
    predicted_by_hierarchy_level,
):
    output = {}

    for cpc_level in [
        "section",
        "class",
        "subclass",
    ]:
        hierarchy_level = MATCHED_LEVEL_MAP[
            cpc_level
        ]

        predicted = np.asarray(
            predicted_by_hierarchy_level[
                hierarchy_level
            ]
        )[valid_cpc_mask]

        labels = CPC_LABELS[
            cpc_level
        ][valid_cpc_mask]

        output[cpc_level] = evaluate_level(
            labels,
            predicted,
        )

        output[cpc_level][
            "hierarchy_level"
        ] = hierarchy_level

        output[cpc_level][
            "num_topics"
        ] = NUM_TOPICS_LIST[
            hierarchy_level
        ]

    return output


def evaluate_all_cross_levels(
    predicted_by_hierarchy_level,
):
    rows = []

    for hierarchy_level, predicted_all in enumerate(
        predicted_by_hierarchy_level
    ):
        predicted = np.asarray(
            predicted_all
        )[valid_cpc_mask]

        for cpc_level in [
            "section",
            "class",
            "subclass",
        ]:
            labels = CPC_LABELS[
                cpc_level
            ][valid_cpc_mask]

            metrics = evaluate_level(
                labels,
                predicted,
            )

            rows.append({
                "hierarchy_level": hierarchy_level,
                "num_topics": NUM_TOPICS_LIST[
                    hierarchy_level
                ],
                "cpc_level": cpc_level,
                **metrics,
            })

    return rows


# --------------------------------------------------------------------------------------
# 10. Sparse batch, inference and hierarchy utilities
# --------------------------------------------------------------------------------------
def sparse_batch_to_gpu(
    csr_matrix,
    indices,
):
    dense = csr_matrix[
        indices
    ].toarray().astype(
        np.float32,
        copy=False,
    )

    return torch.from_numpy(
        dense
    ).to(
        DEVICE,
        non_blocking=True,
    )


def optimizer_to_device(
    optimizer,
    device,
):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def refresh_transport_plans(model):
    model.eval()

    _, transport_plans = model.TPD(
        model.topic_embeddings_list,
        model.weight_loss_TPD,
    )

    model.transp_list = transport_plans


@torch.no_grad()
def infer_hierarchical_theta(
    model,
    csr_matrix,
    batch_size=512,
    description="Theta inference",
):
    model.eval()
    refresh_transport_plans(model)

    outputs = [
        []
        for _ in NUM_TOPICS_LIST
    ]

    for start in tqdm(
        range(
            0,
            csr_matrix.shape[0],
            batch_size,
        ),
        desc=description,
        leave=False,
    ):
        end = min(
            start + batch_size,
            csr_matrix.shape[0],
        )

        indices = np.arange(
            start,
            end,
        )

        batch = sparse_batch_to_gpu(
            csr_matrix,
            indices,
        )

        theta_list = model.get_theta(batch)

        if isinstance(
            theta_list,
            tuple,
        ):
            theta_list = theta_list[0]

        for level, theta in enumerate(
            theta_list
        ):
            outputs[level].append(
                theta.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
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


@torch.no_grad()
def get_beta_list(model):
    model.eval()

    return [
        beta.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        for beta in model.get_beta()
    ]


@torch.no_grad()
def get_phi_list(model):
    model.eval()
    refresh_transport_plans(model)

    return [
        phi.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        for phi in model.get_phi_list()
    ]


def get_top_word_rows(
    beta_list,
    vocab,
    num_words=25,
):
    vocab_array = np.asarray(vocab)
    rows = []

    for level, beta in enumerate(beta_list):
        for topic_id, topic_scores in enumerate(
            beta
        ):
            indices = np.argsort(
                topic_scores
            )[::-1][:num_words]

            words = vocab_array[
                indices
            ].tolist()

            scores = topic_scores[
                indices
            ].tolist()

            rows.append({
                "hierarchy_level": level,
                "num_topics_at_level": (
                    NUM_TOPICS_LIST[level]
                ),
                "topic_id": topic_id,
                "top_words": " ".join(words),
                "top_scores": " ".join(
                    f"{float(score):.8f}"
                    for score in scores
                ),
            })

    return rows


def build_hierarchy_edge_rows(phi_list):
    """
    phi shape:
        [number of parent topics, number of child topics]

    Each child is assigned to its maximum-weight parent for
    deterministic hierarchy export.
    """

    rows = []

    for parent_level, phi in enumerate(
        phi_list
    ):
        child_level = parent_level + 1

        parent_for_child = phi.argmax(
            axis=0
        )

        for child_topic, parent_topic in enumerate(
            parent_for_child
        ):
            parent_topic = int(parent_topic)

            rows.append({
                "parent_level": parent_level,
                "parent_num_topics": (
                    NUM_TOPICS_LIST[parent_level]
                ),
                "parent_topic": parent_topic,
                "child_level": child_level,
                "child_num_topics": (
                    NUM_TOPICS_LIST[child_level]
                ),
                "child_topic": child_topic,
                "dependency_weight": float(
                    phi[
                        parent_topic,
                        child_topic,
                    ]
                ),
            })

    return rows


# --------------------------------------------------------------------------------------
# 11. Atomic save and RNG state
# --------------------------------------------------------------------------------------
def get_rng_state():
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


def restore_rng_state(state):
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
    payload,
    destination,
):
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, local_name = (
        tempfile.mkstemp(
            prefix="traco_checkpoint_",
            suffix=".pt",
            dir="/content",
        )
    )

    os.close(file_descriptor)

    local_path = Path(local_name)

    drive_temp = destination.with_name(
        destination.name
        + f".{os.getpid()}.tmp"
    )

    try:
        torch.save(
            payload,
            local_path,
        )

        shutil.copy2(
            local_path,
            drive_temp,
        )

        if (
            local_path.stat().st_size
            != drive_temp.stat().st_size
        ):
            raise IOError(
                "Checkpoint copy size mismatch."
            )

        os.replace(
            drive_temp,
            destination,
        )

    finally:
        local_path.unlink(
            missing_ok=True
        )
        drive_temp.unlink(
            missing_ok=True
        )


def atomic_json_save(
    payload,
    destination,
):
    destination = Path(destination)

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
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        os.replace(
            temporary,
            destination,
        )

    finally:
        temporary.unlink(
            missing_ok=True
        )


# --------------------------------------------------------------------------------------
# 12. Configuration save
# --------------------------------------------------------------------------------------
configuration = {
    "model": "TraCo",
    "paper": (
        "On the Affinity, Rationality, and Diversity "
        "of Hierarchical Topic Modeling"
    ),
    "official_implementation": "TopMost",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "project_root": str(PROJECT_ROOT),
    "train_record_path": str(
        TRAIN_RECORD_PATH
    ),
    "test_record_path": str(
        TEST_RECORD_PATH
    ),
    "train_patents": int(
        train_bow.shape[0]
    ),
    "test_patents": int(
        test_bow.shape[0]
    ),
    "valid_test_cpc": int(
        valid_cpc_mask.sum()
    ),
    "seeds": SEEDS,
    "num_topics_list": NUM_TOPICS_LIST,
    "matched_level_map": MATCHED_LEVEL_MAP,
    "vocab_size": VOCAB_SIZE,
    "min_doc_count": MIN_DOC_COUNT,
    "max_doc_freq": MAX_DOC_FREQ,
    "vocabulary_train_only": True,
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
    "sinkhorn_max_iter": (
        SINKHORN_MAX_ITER
    ),
    "gradient_clip_norm": (
        GRADIENT_CLIP_NORM
    ),
    "checkpoint_interval": (
        CHECKPOINT_INTERVAL
    ),
    "inference_batch_size": (
        INFERENCE_BATCH_SIZE
    ),
    "mixed_precision": False,
    "tf32": False,
    "gpu": GPU_NAME,
    "gpu_memory_gib": GPU_MEMORY_GIB,
    "document_unit": "patent",
    "document_text": "concatenated claims",
    "cpc_rule": (
        "primary CPC if available; otherwise first CPC"
    ),
    "cpc_used_in_training": False,
    "cpc_used_in_model_selection": False,
    "test_used_for_training": False,
}

atomic_json_save(
    configuration,
    RESULT_DIR / "configuration.json",
)


# --------------------------------------------------------------------------------------
# 13. Three-seed training with automatic resume
# --------------------------------------------------------------------------------------
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
        f"TraCo SEED {seed} — "
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

    # Completed seed: load metrics and skip training.
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
                "active_topics": (
                    seed_result[
                        "level_diagnostics"
                    ][cpc_level][
                        "active_topics"
                    ]
                ),
                "max_topic_share": (
                    seed_result[
                        "level_diagnostics"
                    ][cpc_level][
                        "max_topic_share"
                    ]
                ),
                "elapsed_seconds": (
                    seed_result[
                        "elapsed_seconds"
                    ]
                ),
                "peak_gpu_gib": (
                    seed_result[
                        "peak_gpu_gib"
                    ]
                ),
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

    set_seed(seed)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = TraCo(
        vocab_size=VOCAB_SIZE,
        num_topics_list=NUM_TOPICS_LIST,
        en_units=ENCODER_UNITS,
        dropout=DROPOUT,
        embed_size=EMBED_SIZE,
        bias_topk=BIAS_TOPK,
        bias_p=BIAS_P,
        beta_temp=BETA_TEMP,
        weight_loss_TPD=WEIGHT_LOSS_TPD,
        sinkhorn_alpha=SINKHORN_ALPHA,
        sinkhorn_max_iter=(
            SINKHORN_MAX_ITER
        ),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    start_epoch = 1
    previous_elapsed = 0.0
    training_history = []

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
        ) != NUM_TOPICS_LIST:
            raise RuntimeError(
                "Checkpoint hierarchy mismatch: "
                f"{checkpoint['num_topics_list']} "
                f"!= {NUM_TOPICS_LIST}"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        model.to(DEVICE)

        optimizer_to_device(
            optimizer,
            DEVICE,
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
            f"next epoch={start_epoch}, "
            f"previous elapsed="
            f"{previous_elapsed / 3600:.2f}h"
        )

    seed_start = time.time()
    num_train = train_bow.shape[0]

    num_batches = int(
        np.ceil(num_train / BATCH_SIZE)
    )

    if start_epoch > EPOCHS:
        print(
            "[RESUME] Training epochs already complete. "
            "Running final inference."
        )

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        model.train()

        permutation = np.random.permutation(
            num_train
        )

        epoch_loss_sum = 0.0
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
        )

        for start in progress:
            indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch = sparse_batch_to_gpu(
                train_bow,
                indices,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(batch)
            loss = output["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite TraCo loss: "
                    f"seed={seed}, "
                    f"epoch={epoch}, "
                    f"batch_start={start}, "
                    f"loss={float(loss.detach().cpu())}"
                )

            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        GRADIENT_CLIP_NORM
                    ),
                    error_if_nonfinite=True,
                )
            )

            if not torch.isfinite(
                gradient_norm
            ):
                raise FloatingPointError(
                    "Non-finite TraCo gradient: "
                    f"seed={seed}, "
                    f"epoch={epoch}, "
                    f"batch_start={start}"
                )

            optimizer.step()

            current_batch_size = len(indices)
            loss_value = float(
                loss.detach().cpu().item()
            )

            epoch_loss_sum += (
                loss_value
                * current_batch_size
            )
            epoch_patents += (
                current_batch_size
            )

            progress.set_postfix({
                "loss": f"{loss_value:.3f}",
                "gpu": (
                    f"{torch.cuda.memory_allocated() / 2**30:.1f}G"
                ),
            })

            del batch
            del output
            del loss

        epoch_elapsed = (
            time.time() - epoch_start
        )

        mean_epoch_loss = (
            epoch_loss_sum
            / max(1, epoch_patents)
        )

        history_record = {
            "epoch": epoch,
            "mean_loss": mean_epoch_loss,
            "elapsed_seconds": (
                epoch_elapsed
            ),
            "learning_rate": optimizer.param_groups[
                0
            ]["lr"],
        }

        training_history.append(
            history_record
        )

        print(
            f"[SEED {seed} EPOCH "
            f"{epoch:03d}/{EPOCHS}] "
            f"loss={mean_epoch_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"time={epoch_elapsed / 60:.1f}m"
        )

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
                "format_version": 1,
                "model": "TraCo",
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
                "training_history": (
                    training_history
                ),
                "elapsed_seconds": (
                    current_elapsed
                ),
                "rng_state": get_rng_state(),
                "configuration": configuration,
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

    # ----------------------------------------------------------------------------------
    # Final inference
    # ----------------------------------------------------------------------------------
    print(
        f"[INFERENCE] Seed {seed} train theta"
    )

    train_theta_list = (
        infer_hierarchical_theta(
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
        infer_hierarchical_theta(
            model=model,
            csr_matrix=test_bow,
            batch_size=INFERENCE_BATCH_SIZE,
            description=(
                f"Seed {seed} test theta"
            ),
        )
    )

    beta_list = get_beta_list(model)
    phi_list = get_phi_list(model)

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

    cross_df = pd.DataFrame(cross_rows)

    cross_df.to_csv(
        cross_metrics_path,
        index=False,
    )

    all_cross_rows.extend(cross_rows)

    level_diagnostics_by_cpc = {}
    raw_level_diagnostics = []

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
            counts.max() / counts.sum()
        )

        diagnostic = {
            "hierarchy_level": level,
            "num_topics": (
                NUM_TOPICS_LIST[level]
            ),
            "active_topics": active_topics,
            "max_topic_share": (
                max_topic_share
            ),
            "topic_counts": (
                counts.tolist()
            ),
        }

        raw_level_diagnostics.append(
            diagnostic
        )

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
        / (1024 ** 3)
    )

    # ----------------------------------------------------------------------------------
    # Save arrays by hierarchy level
    # ----------------------------------------------------------------------------------
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

    # Topic words
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

    # Deterministic hierarchy edges
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

    seed_result = {
        "seed": seed,
        "matched_metrics": matched_metrics,
        "level_diagnostics": (
            level_diagnostics_by_cpc
        ),
        "raw_level_diagnostics": (
            raw_level_diagnostics
        ),
        "elapsed_seconds": (
            total_seed_elapsed
        ),
        "peak_gpu_gib": peak_gpu_gib,
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
        "format_version": 1,
        "model": "TraCo",
        "seed": seed,
        "epoch": EPOCHS,
        "num_topics_list": (
            NUM_TOPICS_LIST
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "configuration": configuration,
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
            f"{title} [L{metrics['hierarchy_level']}, "
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
    del train_theta_list
    del test_theta_list
    del beta_list
    del phi_list
    del predicted_by_level

    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# 14. Aggregate matched-level results
# --------------------------------------------------------------------------------------
matched_seed_df = pd.DataFrame(
    all_matched_rows
)

matched_seed_df = matched_seed_df.sort_values(
    ["seed", "hierarchy_level"]
).reset_index(drop=True)

matched_seed_df.to_csv(
    RESULT_DIR
    / "traco_matched_seed_level_results.csv",
    index=False,
)

cross_level_df = pd.DataFrame(
    all_cross_rows
)

if not cross_level_df.empty:
    cross_level_df = cross_level_df.sort_values(
        [
            "seed",
            "hierarchy_level",
            "cpc_level",
        ]
    ).reset_index(drop=True)

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
            MATCHED_LEVEL_MAP[cpc_level]
        ),
        "num_topics": int(
            NUM_TOPICS_LIST[
                MATCHED_LEVEL_MAP[cpc_level]
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


# --------------------------------------------------------------------------------------
# 15. Final report and LaTeX rows
# --------------------------------------------------------------------------------------
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
        row[f"{metric}_{statistic}"]
    )


print("\n" + "=" * 100)
print("TraCo 3-SEED FINAL MATCHED CPC ALIGNMENT")
print("=" * 100)

for title, cpc_level in [
    ("SECTION", "section"),
    ("CLASS", "class"),
    ("SUBCLASS", "subclass"),
]:
    hierarchy_level = (
        MATCHED_LEVEL_MAP[cpc_level]
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

with open(
    RESULT_DIR / "latex_row_mean.txt",
    "w",
    encoding="utf-8",
) as handle:
    handle.write(latex_mean + "\n")

with open(
    RESULT_DIR / "latex_row_mean_std.txt",
    "w",
    encoding="utf-8",
) as handle:
    handle.write(latex_mean_std + "\n")

print("\n" + "-" * 100)
print("[LATEX ROW — MATCHED LEVEL MEAN]")
print(latex_mean)

print("\n[LATEX ROW — MATCHED LEVEL MEAN ± STD]")
print(latex_mean_std)

print("\n[PER-SEED MATCHED RESULTS]")
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
        ]
    ]
)

print("\n[3-SEED MATCHED SUMMARY]")
display(summary_df)

if not cross_level_df.empty:
    print(
        "\n[ALL HIERARCHY LEVEL × CPC LEVEL RESULTS]"
    )
    display(
        cross_level_df[
            [
                "seed",
                "hierarchy_level",
                "num_topics",
                "cpc_level",
                "pur_p",
                "pur_a",
                "nmi",
            ]
        ]
    )

print("\n" + "=" * 100)
print("[COMPLETE]")
print(
    f"Total elapsed : "
    f"{(time.time() - total_start) / 3600:.2f} hours"
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
print(
    "Cross CSV     : "
    f"{RESULT_DIR / 'traco_all_cross_level_results.csv'}"
)
print(f"Arrays        : {ARRAY_DIR}")
print(f"Topics        : {TOPIC_DIR}")
print(f"Hierarchies   : {HIERARCHY_DIR}")
print(f"Checkpoints   : {CHECKPOINT_DIR}")
print("-" * 100)
print("CPC used in training       : False")
print("CPC used in model selection: False")
print("Test used for training     : False")
print("Vocabulary fitted on train : True")
print("=" * 100)


