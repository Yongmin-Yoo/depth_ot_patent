# ============================================================
# Depth-OT 코드 전용 Drive 저장소 생성
#
# 생성 위치:
# /content/drive/MyDrive/depth_ot_code
#
# 기존 파일은 그대로 두고 복사만 수행
# ============================================================

import os
import sys
import json
import shutil
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# 0. Drive mount
# ------------------------------------------------------------

if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")

# ------------------------------------------------------------
# 1. 경로 설정
# ------------------------------------------------------------

SOURCE_PROJECT = Path(
    "/content/drive/MyDrive/depth_ot_patent"
)

CODE_ROOT = Path(
    "/content/drive/MyDrive/depth_ot_code"
)

RECOVERED_SOURCE = (
    SOURCE_PROJECT /
    "recovered_code"
)

INDIVIDUAL_SOURCE = (
    RECOVERED_SOURCE /
    "individual_cells"
)

# 코드 공개 준비용 구조
DIRECTORIES = [
    CODE_ROOT / "archive" / "recovered_raw",
    CODE_ROOT / "archive" / "recovered_cells",
    CODE_ROOT / "src",
    CODE_ROOT / "scripts",
    CODE_ROOT / "configs",
    CODE_ROOT / "notebooks",
    CODE_ROOT / "experiments" / "claim_pooling",
    CODE_ROOT / "experiments" / "low_hierarchy",
    CODE_ROOT / "docs",
    CODE_ROOT / "environment",
    CODE_ROOT / "manifests",
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

print("=" * 100)
print("DEPTH-OT CODE REPOSITORY BACKUP")
print("=" * 100)
print(f"Source : {SOURCE_PROJECT}")
print(f"Target : {CODE_ROOT}")
print()

# ------------------------------------------------------------
# 2. 안전한 복사 함수
# ------------------------------------------------------------

copied_files = []
missing_files = []
skipped_files = []


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_copy(source, destination):
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        missing_files.append(
            str(source)
        )
        print(f"[MISSING] {source}")
        return False

    if source.is_dir():
        skipped_files.append(
            str(source)
        )
        print(f"[SKIP DIR] {source}")
        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    copied_files.append({
        "source": str(source),
        "destination": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    })

    print(
        f"[COPIED] {source.name}\n"
        f"         -> {destination}"
    )

    return True

# ------------------------------------------------------------
# 3. 복구 ZIP 저장
# ------------------------------------------------------------

recovered_zip = (
    RECOVERED_SOURCE /
    "depth_ot_recovered_core_cells.zip"
)

safe_copy(
    recovered_zip,
    CODE_ROOT /
    "archive" /
    "recovered_raw" /
    recovered_zip.name,
)

# ------------------------------------------------------------
# 4. 개별 복구 셀 저장
# ------------------------------------------------------------

cell_files = sorted(
    INDIVIDUAL_SOURCE.glob(
        "recovered_cell_*.py"
    )
)

if not cell_files:
    print(
        "[WARNING] 개별 recovered cell을 찾지 못했습니다."
    )

for cell_file in cell_files:
    safe_copy(
        cell_file,
        CODE_ROOT /
        "archive" /
        "recovered_cells" /
        cell_file.name,
    )

# ------------------------------------------------------------
# 5. 전체 Colab history 저장
# ------------------------------------------------------------

history_files = sorted(
    RECOVERED_SOURCE.glob(
        "all_colab_input_history_*.py"
    )
)

for history_file in history_files:
    safe_copy(
        history_file,
        CODE_ROOT /
        "archive" /
        "recovered_raw" /
        history_file.name,
    )

# ------------------------------------------------------------
# 6. 핵심 코드를 reference notebook 영역에 별도 복사
#
# Cell 15: claim inference/pooling 핵심 후보
# Cell 13/14: 모델 및 평가 의존 코드 후보
# Cell 17: 후속 pooling 평가 코드
# ------------------------------------------------------------

REFERENCE_CELL_MAP = {
    "recovered_cell_013.py":
        "01_model_evaluation_reference.py",

    "recovered_cell_014.py":
        "02_training_or_inference_reference.py",

    "recovered_cell_015.py":
        "03_claim_pooling_inference_reference.py",

    "recovered_cell_017.py":
        "04_claim_pooling_evaluation_reference.py",
}

for source_name, destination_name in REFERENCE_CELL_MAP.items():
    safe_copy(
        INDIVIDUAL_SOURCE / source_name,
        CODE_ROOT /
        "notebooks" /
        destination_name,
    )

# ------------------------------------------------------------
# 7. 핵심 실험 결과 복사
# ------------------------------------------------------------

POOLING_RESULT_ROOT = (
    SOURCE_PROJECT /
    "results" /
    "depth_ot_v2" /
    "depth_ot_v2_patent_semantic_seed42_20260814_055110"
)

pooling_result_candidates = [
    (
        POOLING_RESULT_ROOT /
        "epoch016_extended_independent_pooling_dev_search" /
        "extended_independent_pooling_dev_ranking.csv"
    ),
    (
        POOLING_RESULT_ROOT /
        "epoch016_extended_independent_pooling_dev_search" /
        "extended_independent_pooling_dev_summary.json"
    ),
    (
        POOLING_RESULT_ROOT /
        "epoch016_claim_pooling_dev_search" /
        "epoch016_claim_pooling_dev_ranking.csv"
    ),
    (
        POOLING_RESULT_ROOT /
        "epoch016_claim_pooling_dev_search" /
        "epoch016_claim_pooling_dev_summary.json"
    ),
]

for source_path in pooling_result_candidates:
    if source_path.exists():
        safe_copy(
            source_path,
            CODE_ROOT /
            "experiments" /
            "claim_pooling" /
            source_path.name,
        )

# low-hierarchy 결과
LOWHIER_RESULT_ROOT = (
    SOURCE_PROJECT /
    "results" /
    "depth_ot_v2" /
    "depth_ot_v2_lowhier005_from_epoch016_seed42"
)

lowhier_candidates = [
    (
        LOWHIER_RESULT_ROOT /
        "dev_cpc_checkpoint_evaluation" /
        "dev_checkpoint_comparison_original16_lowhier17_18_19.csv"
    ),
    (
        LOWHIER_RESULT_ROOT /
        "dev_cpc_checkpoint_evaluation" /
        "depth_ot_v2_lowhier_dev_evaluation_complete.json"
    ),
    (
        LOWHIER_RESULT_ROOT /
        "section6_training_summary.json"
    ),
]

for source_path in lowhier_candidates:
    if source_path.exists():
        safe_copy(
            source_path,
            CODE_ROOT /
            "experiments" /
            "low_hierarchy" /
            source_path.name,
        )

# ------------------------------------------------------------
# 8. 학습 history/config 복사
# ------------------------------------------------------------

LOG_ROOT = (
    SOURCE_PROJECT /
    "logs" /
    "depth_ot_v2"
)

history_candidates = [
    (
        LOG_ROOT /
        "depth_ot_v2_lowhier005_from_epoch016_seed42" /
        "section6_training_history.json"
    ),
    (
        LOG_ROOT /
        "depth_ot_v2_lowhier005_from_epoch016_seed42" /
        "section6_training_history.csv"
    ),
]

for source_path in history_candidates:
    if source_path.exists():
        safe_copy(
            source_path,
            CODE_ROOT /
            "experiments" /
            "low_hierarchy" /
            source_path.name,
        )

# ------------------------------------------------------------
# 9. 공개 코드용 폴더에 README placeholder 생성
# ------------------------------------------------------------

src_readme = CODE_ROOT / "src" / "README.md"

src_readme.write_text(
    """# Depth-OT Source Code

이 디렉터리는 최종 공개용 Python 모듈을 저장하는 공간입니다.

예정 모듈:

- `model.py`: Depth-OT V2 모델
- `data.py`: feature shard 및 patent record loader
- `claim_pooling.py`: claim-level patent pooling
- `cpc_evaluation.py`: CPC alignment 평가
- `utils.py`: 공통 유틸리티

주의:

`archive/recovered_cells`와 `notebooks`의 코드는 복구된 원본 코드입니다.
공개 전에 절대경로, 중복 코드, DEV/TEST 설정을 정리해야 합니다.
""",
    encoding="utf-8",
)

scripts_readme = CODE_ROOT / "scripts" / "README.md"

scripts_readme.write_text(
    """# Depth-OT Execution Scripts

예정 실행 스크립트:

- `train_depth_ot.py`
- `evaluate_dev.py`
- `evaluate_test.py`
- `search_claim_pooling.py`

TEST 코드는 최종 설정이 확정된 후 한 번만 실행하도록 분리합니다.
""",
    encoding="utf-8",
)

configs_readme = CODE_ROOT / "configs" / "README.md"

configs_readme.write_text(
    """# Experiment Configurations

이 디렉터리에는 다음 설정을 저장합니다.

- 모델 hyperparameters
- seed
- dataset split
- checkpoint selection rule
- claim pooling parameters
- CPC evaluation configuration
""",
    encoding="utf-8",
)

# ------------------------------------------------------------
# 10. 현재 환경 저장
# ------------------------------------------------------------

python_info_path = (
    CODE_ROOT /
    "environment" /
    "python_environment.txt"
)

python_info = [
    f"created_at: {datetime.now().isoformat()}",
    f"python_version: {sys.version}",
    f"platform: {platform.platform()}",
]

try:
    import torch

    python_info.extend([
        f"torch_version: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"cuda_version: {torch.version.cuda}",
        (
            "gpu_name: "
            + (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "None"
            )
        ),
    ])

except Exception as e:
    python_info.append(
        f"torch_error: {repr(e)}"
    )

python_info_path.write_text(
    "\n".join(python_info) + "\n",
    encoding="utf-8",
)

requirements_path = (
    CODE_ROOT /
    "environment" /
    "requirements_full.txt"
)

try:
    freeze_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    requirements_path.write_text(
        freeze_result.stdout,
        encoding="utf-8",
    )

    print(
        f"[SAVED] Environment: {requirements_path}"
    )

except Exception as e:
    requirements_path.write_text(
        f"pip freeze failed: {repr(e)}\n",
        encoding="utf-8",
    )

# ------------------------------------------------------------
# 11. 루트 README 생성
# ------------------------------------------------------------

root_readme = CODE_ROOT / "README.md"
