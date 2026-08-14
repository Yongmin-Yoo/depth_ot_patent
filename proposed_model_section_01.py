# ============================================================
# SECTION 1 — DEPTH-OT V2
# Existing Feature Fast Restore, Validation, and Manifest
#
# IMPORTANT:
# - 기존 feature를 재추출하지 않습니다.
# - PLM을 로드하지 않습니다.
# - 기존 shard를 수정하지 않습니다.
# ============================================================

import os
import gc
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from tqdm.auto import tqdm


# ============================================================
# 1.1 Preconditions
# ============================================================

REQUIRED_SECTION1_GLOBALS = [
    "CONFIG",
    "DIRS",
    "DEVICE",
    "FEATURE_RUN_NAME",
    "FEATURE_ROOT",
    "FEATURE_SPLIT_DIRS",
    "RUN_LOG_DIR",
]

missing_section1_globals = [
    name
    for name in REQUIRED_SECTION1_GLOBALS
    if name not in globals()
]

if missing_section1_globals:
    raise RuntimeError(
        "Section 0을 먼저 실행하세요. "
        f"Missing: {missing_section1_globals}"
    )

if not bool(
    CONFIG.reuse_existing_features
):
    raise RuntimeError(
        "CONFIG.reuse_existing_features가 "
        "False입니다. V2는 기존 feature를 "
        "재사용하도록 설정되어야 합니다."
    )


# ============================================================
# 1.2 Configuration
# ============================================================

SECTION1_IO_RETRIES = int(
    CONFIG.io_max_retries
)

SECTION1_RETRY_SECONDS = float(
    CONFIG.io_retry_seconds
)

ALLOWED_SHARD_SUFFIXES = {
    ".pt",
    ".pth",
}

EXPECTED_SHARD_COUNTS = {
    "train": 496,
    "dev": 99,
    "test": 99,
}

SAMPLE_SHARDS_PER_SPLIT = 1

SCHEMA_MAX_DEPTH = 4
SCHEMA_MAX_ITEMS = 5
TENSOR_SAMPLE_VALUES = 4096

SECTION1_MANIFEST_PATH = (
    Path(RUN_LOG_DIR)
    / "section1_feature_manifest.json"
)

SECTION1_SCHEMA_PATH = (
    Path(RUN_LOG_DIR)
    / "section1_feature_schema.json"
)


# ============================================================
# 1.3 Safe loading
# ============================================================

if hasattr(
    torch.load,
    "_depth_ot_original",
):
    SECTION1_ORIGINAL_TORCH_LOAD = (
        torch.load._depth_ot_original
    )
else:
    SECTION1_ORIGINAL_TORCH_LOAD = (
        torch.load
    )


def safe_torch_load_section1(
    path,
    map_location="cpu",
):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"파일이 없습니다: {path}"
        )

    last_error = None

    for attempt in range(
        1,
        SECTION1_IO_RETRIES + 1,
    ):
        try:
            return SECTION1_ORIGINAL_TORCH_LOAD(
                path,
                map_location=map_location,
                weights_only=False,
            )

        except (
            OSError,
            EOFError,
            RuntimeError,
        ) as error:
            last_error = error

            if (
                attempt
                >= SECTION1_IO_RETRIES
            ):
                break

            wait_seconds = (
                SECTION1_RETRY_SECONDS
                * attempt
            )

            print(
                f"[I/O RETRY] "
                f"{attempt}/"
                f"{SECTION1_IO_RETRIES} | "
                f"{path.name} | "
                f"{str(error)[:120]}"
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"Shard를 로드하지 못했습니다: "
        f"{path}"
    ) from last_error


# ============================================================
# 1.4 JSON save
# ============================================================

def atomic_json_save_section1(
    data,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    temporary_path.unlink(
        missing_ok=True
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )

    os.replace(
        temporary_path,
        path,
    )


# ============================================================
# 1.5 Shard discovery
# ============================================================

def discover_feature_shards(
    split_directory,
):
    split_directory = Path(
        split_directory
    )

    if not split_directory.is_dir():
        raise FileNotFoundError(
            f"Feature split directory가 없습니다: "
            f"{split_directory}"
        )

    shard_paths = sorted([
        path
        for path in split_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in ALLOWED_SHARD_SUFFIXES
            and not path.name.endswith(
                ".tmp"
            )
        )
    ])

    if not shard_paths:
        raise FileNotFoundError(
            f"Feature shard가 없습니다: "
            f"{split_directory}"
        )

    return shard_paths


FEATURE_SHARDS = {}

for split_name in [
    "train",
    "dev",
    "test",
]:
    FEATURE_SHARDS[
        split_name
    ] = discover_feature_shards(
        FEATURE_SPLIT_DIRS[
            split_name
        ]
    )


# ============================================================
# 1.6 File-level validation
# ============================================================

def validate_shard_files(
    split_name,
    shard_paths,
):
    records = []
    zero_size_files = []
    total_bytes = 0

    for shard_index, shard_path in enumerate(
        tqdm(
            shard_paths,
            desc=f"Indexing {split_name} shards",
        )
    ):
        try:
            stat = shard_path.stat()
        except OSError as error:
            raise OSError(
                f"Shard stat 실패: "
                f"{shard_path}"
            ) from error

        size_bytes = int(
            stat.st_size
        )

        if size_bytes <= 0:
            zero_size_files.append(
                str(shard_path)
            )

        total_bytes += size_bytes

        records.append({
            "index": int(
                shard_index
            ),
            "file": shard_path.name,
            "path": str(
                shard_path
            ),
            "size_bytes": (
                size_bytes
            ),
            "size_mib": float(
                size_bytes / 2**20
            ),
            "modified_time": float(
                stat.st_mtime
            ),
        })

    if zero_size_files:
        raise IOError(
            "크기가 0인 feature shard가 있습니다: "
            f"{zero_size_files[:10]}"
        )

    expected_count = (
        EXPECTED_SHARD_COUNTS.get(
            split_name
        )
    )

    if (
        expected_count is not None
        and len(shard_paths)
        != expected_count
    ):
        raise RuntimeError(
            f"{split_name} shard count mismatch: "
            f"observed={len(shard_paths)}, "
            f"expected={expected_count}"
        )

    return {
        "split": split_name,
        "directory": str(
            FEATURE_SPLIT_DIRS[
                split_name
            ]
        ),
        "number_of_shards": int(
            len(shard_paths)
        ),
        "total_bytes": int(
            total_bytes
        ),
        "total_gib": float(
            total_bytes / 2**30
        ),
        "minimum_shard_mib": float(
            min(
                record["size_mib"]
                for record in records
            )
        ),
        "maximum_shard_mib": float(
            max(
                record["size_mib"]
                for record in records
            )
        ),
        "shards": records,
    }


split_file_manifests = {}

for split_name, shard_paths in (
    FEATURE_SHARDS.items()
):
    split_file_manifests[
        split_name
    ] = validate_shard_files(
        split_name=split_name,
        shard_paths=shard_paths,
    )


# ============================================================
# 1.7 Schema inspection
# ============================================================

def tensor_schema(
    tensor,
):
    tensor = tensor.detach().cpu()

    result = {
        "type": "torch.Tensor",
        "shape": [
            int(value)
            for value in tensor.shape
        ],
        "dtype": str(
            tensor.dtype
        ),
        "numel": int(
            tensor.numel()
        ),
        "requires_grad": bool(
            tensor.requires_grad
        ),
    }

    if tensor.numel() == 0:
        result.update({
            "sample_finite": True,
            "sample_min": None,
            "sample_max": None,
            "sample_mean": None,
        })

        return result

    flat = tensor.reshape(-1)

    if flat.numel() > TENSOR_SAMPLE_VALUES:
        sample_indices = torch.linspace(
            0,
            flat.numel() - 1,
            steps=TENSOR_SAMPLE_VALUES,
            dtype=torch.long,
        )

        sample = flat[
            sample_indices
        ]
    else:
        sample = flat

    if (
        sample.is_floating_point()
        or sample.is_complex()
    ):
        finite_mask = torch.isfinite(
            sample
        )

        result[
            "sample_finite"
        ] = bool(
            finite_mask.all().item()
        )

        if finite_mask.any():
            finite_sample = (
                sample[
                    finite_mask
                ]
                .float()
            )

            result[
                "sample_min"
            ] = float(
                finite_sample.min().item()
            )

            result[
                "sample_max"
            ] = float(
                finite_sample.max().item()
            )

            result[
                "sample_mean"
            ] = float(
                finite_sample.mean().item()
            )
        else:
            result[
                "sample_min"
            ] = None

            result[
                "sample_max"
            ] = None

            result[
                "sample_mean"
            ] = None

    else:
        result[
            "sample_finite"
        ] = True

        result[
            "sample_min"
        ] = int(
            sample.min().item()
        )

        result[
            "sample_max"
        ] = int(
            sample.max().item()
        )

        result[
            "sample_mean"
        ] = float(
            sample.float().mean().item()
        )

    return result


def describe_object(
    value,
    depth=0,
):
    if depth > SCHEMA_MAX_DEPTH:
        return {
            "type": type(
                value
            ).__name__,
            "truncated": True,
        }

    if torch.is_tensor(value):
        return tensor_schema(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        keys = list(
            value.keys()
        )

        selected_keys = keys[
            :SCHEMA_MAX_ITEMS
        ]

        return {
            "type": "dict",
            "length": int(
                len(value)
            ),
            "keys": [
                str(key)
                for key in keys
            ],
            "items": {
                str(key): describe_object(
                    value[key],
                    depth=depth + 1,
                )
                for key in selected_keys
            },
            "items_truncated": bool(
                len(keys)
                > SCHEMA_MAX_ITEMS
            ),
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        selected_items = value[
            :SCHEMA_MAX_ITEMS
        ]

        return {
            "type": type(
                value
            ).__name__,
            "length": int(
                len(value)
            ),
            "items": [
                describe_object(
                    item,
                    depth=depth + 1,
                )
                for item in selected_items
            ],
            "items_truncated": bool(
                len(value)
                > SCHEMA_MAX_ITEMS
            ),
        }

    if isinstance(
        value,
        np.ndarray,
    ):
        array = np.asarray(
            value
        )

        result = {
            "type": "numpy.ndarray",
            "shape": [
                int(item)
                for item in array.shape
            ],
            "dtype": str(
                array.dtype
            ),
            "size": int(
                array.size
            ),
        }

        if (
            array.size > 0
            and np.issubdtype(
                array.dtype,
                np.number,
            )
        ):
            flat = array.reshape(-1)

            sample = flat[
                :min(
                    flat.size,
                    TENSOR_SAMPLE_VALUES,
                )
            ]

            if np.issubdtype(
                sample.dtype,
                np.floating,
            ):
                result[
                    "sample_finite"
                ] = bool(
                    np.isfinite(
                        sample
                    ).all()
                )

        return result

    if isinstance(
        value,
        Path,
    ):
        return {
            "type": "Path",
            "value": str(
                value
            ),
        }

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        display_value = value

        if (
            isinstance(value, str)
            and len(value) > 300
        ):
            display_value = (
                value[:300]
                + "...[truncated]"
            )

        return {
            "type": type(
                value
            ).__name__,
            "value": (
                display_value
            ),
        }

    return {
        "type": type(
            value
        ).__name__,
        "repr": repr(
            value
        )[:300],
    }


def infer_container_count(
    payload,
):
    if isinstance(
        payload,
        (list, tuple),
    ):
        return int(
            len(payload)
        )

    if isinstance(payload, dict):
        candidate_keys = [
            "records",
            "patents",
            "items",
            "examples",
            "data",
            "features",
        ]

        for key in candidate_keys:
            value = payload.get(
                key
            )

            if isinstance(
                value,
                (list, tuple),
            ):
                return int(
                    len(value)
                )

    return None


# ============================================================
# 1.8 Load one sample shard from each split
# ============================================================

sample_schema_reports = {}

for split_name in [
    "train",
    "dev",
    "test",
]:
    split_samples = []

    sample_paths = FEATURE_SHARDS[
        split_name
    ][
        :SAMPLE_SHARDS_PER_SPLIT
    ]

    for sample_path in sample_paths:
        print(
            f"\n[LOAD SAMPLE] "
            f"{split_name}: "
            f"{sample_path.name}"
        )

        payload = safe_torch_load_section1(
            sample_path,
            map_location="cpu",
        )

        schema = describe_object(
            payload
        )

        inferred_count = (
            infer_container_count(
                payload
            )
        )

        split_samples.append({
            "file": sample_path.name,
            "path": str(
                sample_path
            ),
            "size_mib": float(
                sample_path.stat().st_size
                / 2**20
            ),
            "inferred_container_count": (
                inferred_count
            ),
            "schema": schema,
        })

        del payload
        gc.collect()

    sample_schema_reports[
        split_name
    ] = split_samples


# ============================================================
# 1.9 Cross-split top-level validation
# ============================================================

def top_level_signature(
    schema,
):
    result = {
        "type": schema.get(
            "type"
        ),
    }

    if schema.get("type") == "dict":
        result["keys"] = schema.get(
            "keys",
            [],
        )

    return result


split_signatures = {}

for split_name, reports in (
    sample_schema_reports.items()
):
    split_signatures[
        split_name
    ] = top_level_signature(
        reports[0]["schema"]
    )

reference_signature = (
    split_signatures["train"]
)

for split_name in [
    "dev",
    "test",
]:
    if (
        split_signatures[
            split_name
        ]
        != reference_signature
    ):
        raise RuntimeError(
            "Feature shard top-level schema가 "
            f"split마다 다릅니다: "
            f"train={reference_signature}, "
            f"{split_name}="
            f"{split_signatures[split_name]}"
        )


# ============================================================
# 1.10 Save manifests
# ============================================================

section1_manifest = {
    "created_at": (
        datetime.now().isoformat()
    ),
    "run_name": (
        CONFIG.run_name
    ),
    "model_version": (
        CONFIG.model_version
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "feature_root": str(
        FEATURE_ROOT
    ),
    "reuse_existing_features": True,
    "plm_name": (
        CONFIG.plm_name
    ),
    "feature_storage_dtype": (
        CONFIG.feature_storage_dtype
    ),
    "splits": (
        split_file_manifests
    ),
}

section1_schema_payload = {
    "created_at": (
        datetime.now().isoformat()
    ),
    "run_name": (
        CONFIG.run_name
    ),
    "feature_run_name": (
        FEATURE_RUN_NAME
    ),
    "top_level_signatures": (
        split_signatures
    ),
    "sample_schemas": (
        sample_schema_reports
    ),
}

atomic_json_save_section1(
    section1_manifest,
    SECTION1_MANIFEST_PATH,
)

atomic_json_save_section1(
    section1_schema_payload,
    SECTION1_SCHEMA_PATH,
)

FEATURE_MANIFEST = (
    section1_manifest
)

FEATURE_SCHEMA_REPORT = (
    section1_schema_payload
)


# ============================================================
# 1.11 Compact schema print
# ============================================================

def print_compact_schema(
    value,
    name="root",
    indent=0,
    maximum_depth=3,
):
    prefix = "  " * indent

    if indent > maximum_depth:
        print(
            f"{prefix}{name}: ..."
        )
        return

    if torch.is_tensor(value):
        print(
            f"{prefix}{name}: "
            f"Tensor"
            f"{tuple(value.shape)} "
            f"{value.dtype}"
        )
        return

    if isinstance(value, dict):
        print(
            f"{prefix}{name}: "
            f"dict[{len(value)}]"
        )

        for key, item in list(
            value.items()
        )[:10]:
            print_compact_schema(
                item,
                name=str(key),
                indent=indent + 1,
                maximum_depth=maximum_depth,
            )

        if len(value) > 10:
            print(
                f"{prefix}  ... "
                f"{len(value) - 10} more keys"
            )

        return

    if isinstance(
        value,
        (list, tuple),
    ):
        print(
            f"{prefix}{name}: "
            f"{type(value).__name__}"
            f"[{len(value)}]"
        )

        if len(value) > 0:
            print_compact_schema(
                value[0],
                name="[0]",
                indent=indent + 1,
                maximum_depth=maximum_depth,
            )

        return

    print(
        f"{prefix}{name}: "
        f"{type(value).__name__}"
    )


print("\n" + "=" * 88)
print("SAMPLE SCHEMA — TRAIN")
print("=" * 88)

train_sample_path = (
    FEATURE_SHARDS[
        "train"
    ][0]
)

train_sample_payload = (
    safe_torch_load_section1(
        train_sample_path,
        map_location="cpu",
    )
)

print_compact_schema(
    train_sample_payload,
    name="feature_shard",
    maximum_depth=4,
)

del train_sample_payload
gc.collect()


# ============================================================
# 1.12 Final status
# ============================================================

print("\n" + "=" * 88)
print("SECTION 1 — FAST RESTORE COMPLETED")
print("=" * 88)
print(f"Feature run : {FEATURE_RUN_NAME}")
print(f"Feature root: {FEATURE_ROOT}")

for split_name in [
    "train",
    "dev",
    "test",
]:
    split_manifest = (
        split_file_manifests[
            split_name
        ]
    )

    print(
        f"{split_name:5s}: "
        f"shards="
        f"{split_manifest['number_of_shards']:,}, "
        f"size="
        f"{split_manifest['total_gib']:.2f} GiB"
    )

print(
    f"Total size : "
    f"{sum(
        split_file_manifests[split]['total_gib']
        for split in ['train', 'dev', 'test']
    ):.2f} GiB"
)

print(
    f"Manifest   : "
    f"{SECTION1_MANIFEST_PATH}"
)

print(
    f"Schema     : "
    f"{SECTION1_SCHEMA_PATH}"
)

print("=" * 88)
print(
    "[PASS] 기존 feature를 수정하거나 "
    "재추출하지 않았습니다."
)
print(
    "[NEXT] 위 SAMPLE SCHEMA 출력과 "
    "마지막 요약을 보내주세요."
)
