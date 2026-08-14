# ============================================================
# SECTION 3 — DEPTH-OT V2
# Shard-Aware Dataset, Dynamic Patent Batches, and Collation
# ============================================================

import os
import gc
import json
import math
import time
import random
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm


# ============================================================
# 3.1 Preconditions
# ============================================================

REQUIRED_SECTION3_GLOBALS = [
    "CONFIG",
    "DIRS",
    "DEVICE",
    "FEATURE_SHARDS",
    "RECORDS_BY_SPLIT",
    "RECORD_STATISTICS",
    "VOCAB",
    "TOKEN_TO_ID",
    "tokenize_claim_section2",
    "normalize_claims_section2",
    "normalize_patent_id_section2",
    "normalize_claim_id_section2",
    "RUN_LOG_DIR",
]

missing_section3_globals = [
    name
    for name in REQUIRED_SECTION3_GLOBALS
    if name not in globals()
]

if missing_section3_globals:
    raise RuntimeError(
        "Sections 0–2를 먼저 실행하세요. "
        f"Missing: {missing_section3_globals}"
    )

if len(VOCAB) != int(
    CONFIG.vocab_size
):
    raise RuntimeError(
        "Vocabulary size mismatch: "
        f"{len(VOCAB)} != {CONFIG.vocab_size}"
    )


# ============================================================
# 3.2 Configuration
# ============================================================

PATENTS_PER_BATCH = int(
    CONFIG.patents_per_batch
)

MAXIMUM_CLAIMS_PER_BATCH = int(
    CONFIG.maximum_claims_per_batch
)

MAXIMUM_TOKENS_PER_BATCH = int(
    CONFIG.maximum_tokens_per_batch
)

NUM_WORKERS = int(
    CONFIG.num_workers
)

PIN_MEMORY = bool(
    CONFIG.pin_memory
)

# Existing feature run uses 100 patents per shard.
FEATURE_PATENTS_PER_SHARD = 100

# Tokenizer word counts underestimate PatentSBERT subwords.
ESTIMATED_SUBWORD_MULTIPLIER = 1.35

# Each dataset retains only the most recently used shard in RAM.
IN_MEMORY_SHARD_CACHE_SIZE = 1

SECTION3_MANIFEST_PATH = (
    Path(RUN_LOG_DIR)
    / "section3_dataloader_manifest.json"
)


# ============================================================
# 3.3 JSON utility
# ============================================================

def atomic_json_save_section3(
    data,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.unlink(
        missing_ok=True
    )

    with open(
        temporary,
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
        temporary,
        path,
    )


# ============================================================
# 3.4 Safe shard loading
# ============================================================

if hasattr(
    torch.load,
    "_depth_ot_original",
):
    SECTION3_ORIGINAL_TORCH_LOAD = (
        torch.load._depth_ot_original
    )
else:
    SECTION3_ORIGINAL_TORCH_LOAD = (
        torch.load
    )


def safe_torch_load_section3(
    path,
):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Feature shard가 없습니다: "
            f"{path}"
        )

    last_error = None

    for attempt in range(
        1,
        int(CONFIG.io_max_retries) + 1,
    ):
        try:
            return SECTION3_ORIGINAL_TORCH_LOAD(
                path,
                map_location="cpu",
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
                >= int(CONFIG.io_max_retries)
            ):
                break

            wait_seconds = (
                float(CONFIG.io_retry_seconds)
                * attempt
            )

            print(
                f"[I/O RETRY] "
                f"{attempt}/"
                f"{CONFIG.io_max_retries} | "
                f"{path.name} | "
                f"{str(error)[:100]}"
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"Feature shard를 읽지 못했습니다: "
        f"{path}"
    ) from last_error


# ============================================================
# 3.5 Record metadata
# ============================================================

def normalize_depth_mapping(
    record,
):
    raw_depth = record.get(
        "depth",
        {},
    )

    if not isinstance(
        raw_depth,
        dict,
    ):
        raise TypeError(
            "record['depth'] must be a dict."
        )

    return {
        normalize_claim_id_section2(
            claim_id
        ): int(depth)
        for claim_id, depth
        in raw_depth.items()
    }


def normalize_record_edges(
    record,
):
    edges = record.get(
        "edges",
        [],
    )

    if edges is None:
        return []

    normalized = []

    for edge in edges:
        if (
            not isinstance(
                edge,
                (list, tuple),
            )
            or len(edge) != 2
        ):
            raise ValueError(
                f"Invalid dependency edge: "
                f"{edge}"
            )

        parent_id = (
            normalize_claim_id_section2(
                edge[0]
            )
        )

        child_id = (
            normalize_claim_id_section2(
                edge[1]
            )
        )

        normalized.append(
            (
                parent_id,
                child_id,
            )
        )

    return normalized


def estimate_patent_tokens(
    record,
):
    claims = normalize_claims_section2(
        record
    )

    estimated_tokens = 0

    for text in claims.values():
        word_count = len(
            str(text).split()
        )

        estimated_claim_tokens = int(
            math.ceil(
                word_count
                * ESTIMATED_SUBWORD_MULTIPLIER
            )
        )

        estimated_claim_tokens = max(
            1,
            min(
                estimated_claim_tokens,
                int(CONFIG.max_length),
            ),
        )

        estimated_tokens += (
            estimated_claim_tokens
        )

    return estimated_tokens


# ============================================================
# 3.6 Shard-aware dataset
# ============================================================

class DepthOTV2Dataset(Dataset):

    def __init__(
        self,
        split_name,
        records,
        shard_paths,
    ):
        self.split_name = str(
            split_name
        )

        self.records = list(
            records
        )

        self.shard_paths = [
            Path(path)
            for path in shard_paths
        ]

        self.number_of_records = len(
            self.records
        )

        self.patent_ids = []

        self.record_claim_counts = []
        self.estimated_token_counts = []

        self.record_by_patent_id = {}

        for record_index, record in enumerate(
            tqdm(
                self.records,
                desc=(
                    f"Indexing "
                    f"{self.split_name} records"
                ),
            )
        ):
            patent_id = (
                normalize_patent_id_section2(
                    record[
                        "patent_id"
                    ]
                )
            )

            if patent_id in (
                self.record_by_patent_id
            ):
                raise ValueError(
                    f"Duplicate patent ID: "
                    f"{self.split_name}/"
                    f"{patent_id}"
                )

            claims = normalize_claims_section2(
                record
            )

            self.patent_ids.append(
                patent_id
            )

            self.record_claim_counts.append(
                len(claims)
            )

            self.estimated_token_counts.append(
                estimate_patent_tokens(
                    record
                )
            )

            self.record_by_patent_id[
                patent_id
            ] = record_index

        expected_shards = int(
            math.ceil(
                self.number_of_records
                / FEATURE_PATENTS_PER_SHARD
            )
        )

        if len(self.shard_paths) != expected_shards:
            raise RuntimeError(
                f"{self.split_name} shard count mismatch: "
                f"observed={len(self.shard_paths)}, "
                f"expected={expected_shards}"
            )

        self._shard_cache = OrderedDict()

    def __len__(self):
        return self.number_of_records

    def shard_index_for_record(
        self,
        record_index,
    ):
        if not (
            0
            <= int(record_index)
            < self.number_of_records
        ):
            raise IndexError(
                f"Invalid record index: "
                f"{record_index}"
            )

        return int(
            record_index
            // FEATURE_PATENTS_PER_SHARD
        )

    def record_indices_for_shard(
        self,
        shard_index,
    ):
        start = (
            int(shard_index)
            * FEATURE_PATENTS_PER_SHARD
        )

        end = min(
            start
            + FEATURE_PATENTS_PER_SHARD,
            self.number_of_records,
        )

        return list(
            range(
                start,
                end,
            )
        )

    def _normalize_feature_payload(
        self,
        payload,
        shard_path,
    ):
        if not isinstance(
            payload,
            dict,
        ):
            raise TypeError(
                f"Feature shard must be dict: "
                f"{shard_path}"
            )

        observed_split = payload.get(
            "split"
        )

        if (
            observed_split is not None
            and str(observed_split)
            != self.split_name
        ):
            raise ValueError(
                f"Feature split mismatch: "
                f"{shard_path}, "
                f"observed={observed_split}, "
                f"expected={self.split_name}"
            )

        entries = payload.get(
            "entries"
        )

        patent_ids = payload.get(
            "patent_ids"
        )

        feature_signature = payload.get(
            "feature_signature",
            {},
        )

        if not isinstance(entries, list):
            raise TypeError(
                f"Feature entries must be list: "
                f"{shard_path}"
            )

        if not isinstance(
            patent_ids,
            list,
        ):
            raise TypeError(
                f"Feature patent_ids must be list: "
                f"{shard_path}"
            )

        entries_by_patent = defaultdict(
            dict
        )

        for entry in entries:
            if not isinstance(
                entry,
                dict,
            ):
                raise TypeError(
                    "Feature entry must be dict."
                )

            required_keys = {
                "patent_id",
                "claim_id",
                "token_embeddings",
                "edge_index",
                "edge_weight",
                "num_tokens",
            }

            missing = (
                required_keys
                - set(entry.keys())
            )

            if missing:
                raise KeyError(
                    f"Feature entry missing keys: "
                    f"{sorted(missing)}"
                )

            patent_id = (
                normalize_patent_id_section2(
                    entry[
                        "patent_id"
                    ]
                )
            )

            claim_id = (
                normalize_claim_id_section2(
                    entry[
                        "claim_id"
                    ]
                )
            )

            if claim_id in entries_by_patent[
                patent_id
            ]:
                raise ValueError(
                    f"Duplicate feature claim: "
                    f"{patent_id}/{claim_id}"
                )

            entries_by_patent[
                patent_id
            ][claim_id] = entry

        normalized_patent_ids = [
            normalize_patent_id_section2(
                patent_id
            )
            for patent_id in patent_ids
        ]

        return {
            "entries_by_patent": dict(
                entries_by_patent
            ),
            "patent_ids": (
                normalized_patent_ids
            ),
            "feature_signature": (
                feature_signature
            ),
        }

    def _load_shard(
        self,
        shard_index,
    ):
        shard_index = int(
            shard_index
        )

        if shard_index in self._shard_cache:
            cached = self._shard_cache.pop(
                shard_index
            )

            self._shard_cache[
                shard_index
            ] = cached

            return cached

        shard_path = self.shard_paths[
            shard_index
        ]

        payload = safe_torch_load_section3(
            shard_path
        )

        normalized = (
            self._normalize_feature_payload(
                payload,
                shard_path,
            )
        )

        del payload
        gc.collect()

        self._shard_cache[
            shard_index
        ] = normalized

        while (
            len(self._shard_cache)
            > IN_MEMORY_SHARD_CACHE_SIZE
        ):
            self._shard_cache.popitem(
                last=False
            )

            gc.collect()

        return normalized

    def __getitem__(
        self,
        record_index,
    ):
        record_index = int(
            record_index
        )

        record = self.records[
            record_index
        ]

        patent_id = self.patent_ids[
            record_index
        ]

        shard_index = (
            self.shard_index_for_record(
                record_index
            )
        )

        shard = self._load_shard(
            shard_index
        )

        if patent_id not in (
            shard[
                "entries_by_patent"
            ]
        ):
            raise KeyError(
                "Record/shard patent mismatch. "
                f"split={self.split_name}, "
                f"record_index={record_index}, "
                f"patent_id={patent_id}, "
                f"shard={self.shard_paths[shard_index]}"
            )

        feature_entries = (
            shard[
                "entries_by_patent"
            ][patent_id]
        )

        claims = normalize_claims_section2(
            record
        )

        record_claim_ids = set(
            claims.keys()
        )

        feature_claim_ids = set(
            feature_entries.keys()
        )

        if (
            record_claim_ids
            != feature_claim_ids
        ):
            raise ValueError(
                "Record/feature claim mismatch: "
                f"split={self.split_name}, "
                f"patent={patent_id}, "
                f"record_only="
                f"{sorted(record_claim_ids - feature_claim_ids)[:10]}, "
                f"feature_only="
                f"{sorted(feature_claim_ids - record_claim_ids)[:10]}"
            )

        return {
            "split": self.split_name,
            "record_index": record_index,
            "shard_index": shard_index,
            "patent_id": patent_id,
            "record": record,
            "feature_entries": (
                feature_entries
            ),
        }


# ============================================================
# 3.7 Shard-aware dynamic batch sampler
# ============================================================

class ShardAwarePatentBatchSampler(
    Sampler,
):

    def __init__(
        self,
        dataset,
        patents_per_batch,
        maximum_claims,
        maximum_tokens,
        shuffle,
        seed,
    ):
        self.dataset = dataset

        self.patents_per_batch = int(
            patents_per_batch
        )

        self.maximum_claims = int(
            maximum_claims
        )

        self.maximum_tokens = int(
            maximum_tokens
        )

        self.shuffle = bool(
            shuffle
        )

        self.seed = int(
            seed
        )

        self.epoch = 0

        self.oversized_patents = []

        self._base_shard_batches = (
            self._construct_batches()
        )

        self._number_of_batches = sum(
            len(batches)
            for batches in
            self._base_shard_batches.values()
        )

    def set_epoch(
        self,
        epoch,
    ):
        self.epoch = int(
            epoch
        )

    def _construct_batches(
        self,
    ):
        shard_batches = {}

        for shard_index in range(
            len(
                self.dataset.shard_paths
            )
        ):
            record_indices = (
                self.dataset
                .record_indices_for_shard(
                    shard_index
                )
            )

            batches = []

            current_batch = []
            current_claims = 0
            current_tokens = 0

            for record_index in record_indices:
                claim_count = int(
                    self.dataset
                    .record_claim_counts[
                        record_index
                    ]
                )

                token_count = int(
                    self.dataset
                    .estimated_token_counts[
                        record_index
                    ]
                )

                exceeds_single_limit = (
                    claim_count
                    > self.maximum_claims
                    or token_count
                    > self.maximum_tokens
                )

                if exceeds_single_limit:
                    self.oversized_patents.append({
                        "record_index": (
                            record_index
                        ),
                        "patent_id": (
                            self.dataset
                            .patent_ids[
                                record_index
                            ]
                        ),
                        "claims": (
                            claim_count
                        ),
                        "estimated_tokens": (
                            token_count
                        ),
                    })

                would_exceed = (
                    len(current_batch)
                    >= self.patents_per_batch
                    or (
                        current_batch
                        and (
                            current_claims
                            + claim_count
                            > self.maximum_claims
                        )
                    )
                    or (
                        current_batch
                        and (
                            current_tokens
                            + token_count
                            > self.maximum_tokens
                        )
                    )
                )

                if would_exceed:
                    batches.append(
                        current_batch
                    )

                    current_batch = []
                    current_claims = 0
                    current_tokens = 0

                current_batch.append(
                    record_index
                )

                current_claims += (
                    claim_count
                )

                current_tokens += (
                    token_count
                )

            if current_batch:
                batches.append(
                    current_batch
                )

            shard_batches[
                shard_index
            ] = batches

        return shard_batches

    def __iter__(
        self,
    ):
        generator = random.Random(
            self.seed
            + self.epoch
        )

        shard_indices = list(
            self._base_shard_batches.keys()
        )

        if self.shuffle:
            generator.shuffle(
                shard_indices
            )

        for shard_index in shard_indices:
            batches = [
                list(batch)
                for batch in
                self._base_shard_batches[
                    shard_index
                ]
            ]

            if self.shuffle:
                generator.shuffle(
                    batches
                )

                for batch in batches:
                    generator.shuffle(
                        batch
                    )

            for batch in batches:
                yield batch

    def __len__(
        self,
    ):
        return self._number_of_batches


# ============================================================
# 3.8 BoW construction
# ============================================================

def claim_text_to_bow(
    text,
):
    tokens = tokenize_claim_section2(
        text
    )

    token_ids = [
        TOKEN_TO_ID[token]
        for token in tokens
        if token in TOKEN_TO_ID
    ]

    bow = torch.zeros(
        len(VOCAB),
        dtype=torch.float32,
    )

    if token_ids:
        token_id_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        bow.scatter_add_(
            dim=0,
            index=token_id_tensor,
            src=torch.ones(
                len(token_ids),
                dtype=torch.float32,
            ),
        )

    return bow


# ============================================================
# 3.9 V2 collate function
# ============================================================

def collate_depth_ot_v2(
    patent_items,
):
    if not patent_items:
        raise ValueError(
            "Empty patent batch."
        )

    split_names = {
        item["split"]
        for item in patent_items
    }

    if len(split_names) != 1:
        raise ValueError(
            "Mixed splits in one batch."
        )

    split_name = next(
        iter(split_names)
    )

    shard_indices = {
        int(item["shard_index"])
        for item in patent_items
    }

    if len(shard_indices) != 1:
        raise ValueError(
            "A batch spans multiple feature shards."
        )

    patent_ids = []

    claim_keys = []
    claim_ids = []
    claim_depths = []
    claim_to_patent = []

    token_embeddings_parts = []
    token_edge_index_parts = []
    token_edge_weight_parts = []
    token_to_claim_parts = []

    bow_parts = []

    claim_graph_edges = []
    adjacent_claim_graph_edges = []
    non_adjacent_claim_graph_edges = []

    patent_claim_ptr = [0]
    claim_token_ptr = [0]

    total_tokens = 0
    total_claims = 0

    truncated_claims = 0

    for patent_local_index, item in enumerate(
        patent_items
    ):
        patent_id = item[
            "patent_id"
        ]

        record = item[
            "record"
        ]

        feature_entries = item[
            "feature_entries"
        ]

        claims = normalize_claims_section2(
            record
        )

        depth_mapping = (
            normalize_depth_mapping(
                record
            )
        )

        dependency_edges = (
            normalize_record_edges(
                record
            )
        )

        sorted_claim_ids = sorted(
            claims.keys()
        )

        local_claim_position = {}

        patent_ids.append(
            patent_id
        )

        for claim_id in sorted_claim_ids:
            global_claim_index = (
                total_claims
            )

            local_claim_position[
                claim_id
            ] = global_claim_index

            feature_entry = (
                feature_entries[
                    claim_id
                ]
            )

            token_embeddings = (
                feature_entry[
                    "token_embeddings"
                ]
                .detach()
                .to(
                    dtype=torch.float32
                )
                .contiguous()
            )

            edge_index = (
                feature_entry[
                    "edge_index"
                ]
                .detach()
                .long()
                .contiguous()
            )

            edge_weight = (
                feature_entry[
                    "edge_weight"
                ]
                .detach()
                .float()
                .contiguous()
            )

            number_of_tokens = int(
                token_embeddings.shape[0]
            )

            if number_of_tokens <= 0:
                raise ValueError(
                    f"Claim has no token features: "
                    f"{patent_id}/{claim_id}"
                )

            if (
                int(
                    feature_entry[
                        "num_tokens"
                    ]
                )
                != number_of_tokens
            ):
                raise ValueError(
                    f"num_tokens mismatch: "
                    f"{patent_id}/{claim_id}"
                )

            if edge_index.shape[0] != 2:
                raise ValueError(
                    f"Invalid token edge_index: "
                    f"{patent_id}/{claim_id}"
                )

            if (
                edge_index.shape[1]
                != edge_weight.shape[0]
            ):
                raise ValueError(
                    f"Token edge count mismatch: "
                    f"{patent_id}/{claim_id}"
                )

            if edge_index.numel() > 0:
                if (
                    int(
                        edge_index.min().item()
                    )
                    < 0
                    or int(
                        edge_index.max().item()
                    )
                    >= number_of_tokens
                ):
                    raise IndexError(
                        f"Token edge index out of range: "
                        f"{patent_id}/{claim_id}"
                    )

            token_embeddings_parts.append(
                token_embeddings
            )

            token_edge_index_parts.append(
                edge_index
                + total_tokens
            )

            token_edge_weight_parts.append(
                edge_weight
            )

            token_to_claim_parts.append(
                torch.full(
                    (
                        number_of_tokens,
                    ),
                    global_claim_index,
                    dtype=torch.long,
                )
            )

            total_tokens += (
                number_of_tokens
            )

            claim_token_ptr.append(
                total_tokens
            )

            claim_keys.append(
                (
                    patent_id,
                    claim_id,
                )
            )

            claim_ids.append(
                claim_id
            )

            claim_depths.append(
                int(
                    depth_mapping.get(
                        claim_id,
                        0,
                    )
                )
            )

            claim_to_patent.append(
                patent_local_index
            )

            bow_parts.append(
                claim_text_to_bow(
                    claims[
                        claim_id
                    ]
                )
            )

            if bool(
                feature_entry.get(
                    "truncated",
                    False,
                )
            ):
                truncated_claims += 1

            total_claims += 1

        patent_claim_ptr.append(
            total_claims
        )

        for parent_id, child_id in (
            dependency_edges
        ):
            if (
                parent_id
                not in local_claim_position
                or child_id
                not in local_claim_position
            ):
                raise KeyError(
                    f"Dependency claim missing: "
                    f"{patent_id}/"
                    f"{parent_id}->{child_id}"
                )

            parent_index = (
                local_claim_position[
                    parent_id
                ]
            )

            child_index = (
                local_claim_position[
                    child_id
                ]
            )

            parent_depth = int(
                depth_mapping.get(
                    parent_id,
                    0,
                )
            )

            child_depth = int(
                depth_mapping.get(
                    child_id,
                    0,
                )
            )

            depth_difference = (
                child_depth
                - parent_depth
            )

            if depth_difference <= 0:
                raise ValueError(
                    "Non-increasing dependency edge: "
                    f"{patent_id}/"
                    f"{parent_id}->{child_id}, "
                    f"depths="
                    f"{parent_depth}->{child_depth}"
                )

            edge_pair = (
                parent_index,
                child_index,
            )

            claim_graph_edges.append(
                edge_pair
            )

            if depth_difference == 1:
                adjacent_claim_graph_edges.append(
                    edge_pair
                )
            else:
                non_adjacent_claim_graph_edges.append(
                    edge_pair
                )

    token_embeddings = torch.cat(
        token_embeddings_parts,
        dim=0,
    )

    if token_edge_index_parts:
        token_edge_index = torch.cat(
            token_edge_index_parts,
            dim=1,
        )

        token_edge_weight = torch.cat(
            token_edge_weight_parts,
            dim=0,
        )
    else:
        token_edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

        token_edge_weight = torch.empty(
            (
                0,
            ),
            dtype=torch.float32,
        )

    token_to_claim = torch.cat(
        token_to_claim_parts,
        dim=0,
    )

    bow_counts = torch.stack(
        bow_parts,
        dim=0,
    )

    bow_valid = (
        bow_counts.sum(
            dim=1
        )
        > 0
    )

    bow_normalized = (
        bow_counts
        / bow_counts.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(
            float(CONFIG.bow_epsilon)
        )
    )

    def edges_to_tensor(
        edges,
    ):
        if not edges:
            return torch.empty(
                (
                    2,
                    0,
                ),
                dtype=torch.long,
            )

        return torch.tensor(
            edges,
            dtype=torch.long,
        ).t().contiguous()

    claim_edge_index = (
        edges_to_tensor(
            claim_graph_edges
        )
    )

    adjacent_claim_edge_index = (
        edges_to_tensor(
            adjacent_claim_graph_edges
        )
    )

    non_adjacent_claim_edge_index = (
        edges_to_tensor(
            non_adjacent_claim_graph_edges
        )
    )

    claim_depth = torch.tensor(
        claim_depths,
        dtype=torch.long,
    )

    claim_to_patent_tensor = torch.tensor(
        claim_to_patent,
        dtype=torch.long,
    )

    patent_claim_ptr_tensor = torch.tensor(
        patent_claim_ptr,
        dtype=torch.long,
    )

    claim_token_ptr_tensor = torch.tensor(
        claim_token_ptr,
        dtype=torch.long,
    )

    batch = {
        "split": split_name,
        "shard_index": int(
            next(iter(shard_indices))
        ),
        "patent_ids": patent_ids,
        "claim_keys": claim_keys,
        "claim_ids": torch.tensor(
            claim_ids,
            dtype=torch.long,
        ),
        "num_patents": int(
            len(patent_ids)
        ),
        "num_claims": int(
            total_claims
        ),
        "num_tokens": int(
            total_tokens
        ),

        # Token graph
        "token_embeddings": (
            token_embeddings
        ),
        "token_edge_index": (
            token_edge_index
        ),
        "token_edge_weight": (
            token_edge_weight
        ),
        "token_to_claim": (
            token_to_claim
        ),
        "claim_token_ptr": (
            claim_token_ptr_tensor
        ),

        # Claim-to-patent grouping
        "claim_to_patent": (
            claim_to_patent_tensor
        ),
        "patent_claim_ptr": (
            patent_claim_ptr_tensor
        ),

        # Claim dependency graph
        "claim_depth": claim_depth,
        "claim_edge_index": (
            claim_edge_index
        ),
        "adjacent_claim_edge_index": (
            adjacent_claim_edge_index
        ),
        "non_adjacent_claim_edge_index": (
            non_adjacent_claim_edge_index
        ),

        # BoW
        "bow_counts": bow_counts,
        "bow_normalized": (
            bow_normalized
        ),
        "bow_valid": bow_valid,

        # Diagnostics
        "num_dependency_edges": int(
            claim_edge_index.shape[1]
        ),
        "num_adjacent_edges": int(
            adjacent_claim_edge_index.shape[1]
        ),
        "num_non_adjacent_edges": int(
            non_adjacent_claim_edge_index.shape[1]
        ),
        "num_truncated_claims": int(
            truncated_claims
        ),
    }

    return batch


# ============================================================
# 3.10 Create datasets
# ============================================================

train_dataset = DepthOTV2Dataset(
    split_name="train",
    records=RECORDS_BY_SPLIT[
        "train"
    ],
    shard_paths=FEATURE_SHARDS[
        "train"
    ],
)

dev_dataset = DepthOTV2Dataset(
    split_name="dev",
    records=RECORDS_BY_SPLIT[
        "dev"
    ],
    shard_paths=FEATURE_SHARDS[
        "dev"
    ],
)

test_dataset = DepthOTV2Dataset(
    split_name="test",
    records=RECORDS_BY_SPLIT[
        "test"
    ],
    shard_paths=FEATURE_SHARDS[
        "test"
    ],
)


# ============================================================
# 3.11 Create batch samplers
# ============================================================

train_batch_sampler = (
    ShardAwarePatentBatchSampler(
        dataset=train_dataset,
        patents_per_batch=(
            PATENTS_PER_BATCH
        ),
        maximum_claims=(
            MAXIMUM_CLAIMS_PER_BATCH
        ),
        maximum_tokens=(
            MAXIMUM_TOKENS_PER_BATCH
        ),
        shuffle=True,
        seed=int(CONFIG.seed),
    )
)

dev_batch_sampler = (
    ShardAwarePatentBatchSampler(
        dataset=dev_dataset,
        patents_per_batch=(
            PATENTS_PER_BATCH
        ),
        maximum_claims=(
            MAXIMUM_CLAIMS_PER_BATCH
        ),
        maximum_tokens=(
            MAXIMUM_TOKENS_PER_BATCH
        ),
        shuffle=False,
        seed=int(CONFIG.seed),
    )
)

test_batch_sampler = (
    ShardAwarePatentBatchSampler(
        dataset=test_dataset,
        patents_per_batch=(
            PATENTS_PER_BATCH
        ),
        maximum_claims=(
            MAXIMUM_CLAIMS_PER_BATCH
        ),
        maximum_tokens=(
            MAXIMUM_TOKENS_PER_BATCH
        ),
        shuffle=False,
        seed=int(CONFIG.seed),
    )
)


# ============================================================
# 3.12 Oversized patent validation
# ============================================================

def print_oversized_summary(
    split_name,
    sampler,
):
    oversized = (
        sampler.oversized_patents
    )

    if oversized:
        print(
            f"[WARNING] {split_name}: "
            f"{len(oversized)} oversized patents"
        )

        for record in oversized[:10]:
            print(
                "  ",
                record,
            )
    else:
        print(
            f"[PASS] {split_name}: "
            "no oversized patents"
        )


print_oversized_summary(
    "train",
    train_batch_sampler,
)

print_oversized_summary(
    "dev",
    dev_batch_sampler,
)

print_oversized_summary(
    "test",
    test_batch_sampler,
)


# ============================================================
# 3.13 Create loaders
# ============================================================

def create_depth_ot_loader(
    dataset,
    batch_sampler,
):
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_depth_ot_v2,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=(
            seed_worker
            if NUM_WORKERS > 0
            else None
        ),
        generator=(
            DATALOADER_GENERATOR
        ),
    )


train_loader = create_depth_ot_loader(
    train_dataset,
    train_batch_sampler,
)

dev_loader = create_depth_ot_loader(
    dev_dataset,
    dev_batch_sampler,
)

test_loader = create_depth_ot_loader(
    test_dataset,
    test_batch_sampler,
)


# ============================================================
# 3.14 Batch device transfer
# ============================================================

DEVICE_TENSOR_KEYS = {
    "token_embeddings",
    "token_edge_index",
    "token_edge_weight",
    "token_to_claim",
    "claim_token_ptr",
    "claim_to_patent",
    "patent_claim_ptr",
    "claim_depth",
    "claim_edge_index",
    "adjacent_claim_edge_index",
    "non_adjacent_claim_edge_index",
    "bow_counts",
    "bow_normalized",
    "bow_valid",
    "claim_ids",
}


def move_depth_ot_v2_batch(
    batch,
    device=DEVICE,
):
    moved = {}

    for key, value in batch.items():
        if (
            key in DEVICE_TENSOR_KEYS
            and torch.is_tensor(value)
        ):
            moved[key] = value.to(
                device=device,
                non_blocking=True,
            )
        else:
            moved[key] = value

    return moved


# Compatibility alias for later sections
move_depth_ot_batch = (
    move_depth_ot_v2_batch
)


# ============================================================
# 3.15 Validate one batch from each split
# ============================================================

def validate_batch_structure(
    batch,
    split_name,
):
    required_keys = {
        "token_embeddings",
        "token_edge_index",
        "token_edge_weight",
        "token_to_claim",
        "claim_to_patent",
        "claim_depth",
        "claim_edge_index",
        "adjacent_claim_edge_index",
        "bow_counts",
        "bow_normalized",
        "bow_valid",
        "patent_ids",
        "claim_keys",
    }

    missing = (
        required_keys
        - set(batch.keys())
    )

    if missing:
        raise KeyError(
            f"Batch missing keys: "
            f"{sorted(missing)}"
        )

    number_of_patents = int(
        batch["num_patents"]
    )

    number_of_claims = int(
        batch["num_claims"]
    )

    number_of_tokens = int(
        batch["num_tokens"]
    )

    if (
        batch[
            "token_embeddings"
        ].shape[0]
        != number_of_tokens
    ):
        raise ValueError(
            "Token count mismatch."
        )

    if (
        batch[
            "token_to_claim"
        ].shape[0]
        != number_of_tokens
    ):
        raise ValueError(
            "token_to_claim mismatch."
        )

    if (
        batch[
            "claim_to_patent"
        ].shape[0]
        != number_of_claims
    ):
        raise ValueError(
            "claim_to_patent mismatch."
        )

    if (
        batch[
            "claim_depth"
        ].shape[0]
        != number_of_claims
    ):
        raise ValueError(
            "claim_depth mismatch."
        )

    if (
        batch[
            "bow_counts"
        ].shape
        != (
            number_of_claims,
            len(VOCAB),
        )
    ):
        raise ValueError(
            "BoW shape mismatch."
        )

    if (
        len(
            batch[
                "patent_ids"
            ]
        )
        != number_of_patents
    ):
        raise ValueError(
            "Patent ID count mismatch."
        )

    if (
        len(
            batch[
                "claim_keys"
            ]
        )
        != number_of_claims
    ):
        raise ValueError(
            "Claim key count mismatch."
        )

    if number_of_claims > 0:
        minimum_patent_index = int(
            batch[
                "claim_to_patent"
            ].min().item()
        )

        maximum_patent_index = int(
            batch[
                "claim_to_patent"
            ].max().item()
        )

        if minimum_patent_index != 0:
            raise ValueError(
                "claim_to_patent must start at zero."
            )

        if (
            maximum_patent_index
            != number_of_patents - 1
        ):
            raise ValueError(
                "claim_to_patent does not cover "
                "all patents."
            )

    normalized_sums = (
        batch[
            "bow_normalized"
        ].sum(
            dim=1
        )
    )

    valid = batch[
        "bow_valid"
    ]

    if valid.any():
        if not torch.allclose(
            normalized_sums[
                valid
            ],
            torch.ones_like(
                normalized_sums[
                    valid
                ]
            ),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                "Normalized BoW rows do not "
                "sum to one."
            )

    print(
        f"\n[BATCH PASS] {split_name}"
    )
    print(
        f"  patents          : "
        f"{number_of_patents}"
    )
    print(
        f"  claims           : "
        f"{number_of_claims}"
    )
    print(
        f"  tokens           : "
        f"{number_of_tokens:,}"
    )
    print(
        f"  token edges      : "
        f"{batch['token_edge_index'].shape[1]:,}"
    )
    print(
        f"  dependency edges : "
        f"{batch['claim_edge_index'].shape[1]:,}"
    )
    print(
        f"  adjacent edges   : "
        f"{batch['adjacent_claim_edge_index'].shape[1]:,}"
    )
    print(
        f"  valid BoW claims : "
        f"{int(batch['bow_valid'].sum().item()):,}/"
        f"{number_of_claims:,}"
    )
    print(
        f"  token shape      : "
        f"{tuple(batch['token_embeddings'].shape)}"
    )
    print(
        f"  BoW shape        : "
        f"{tuple(batch['bow_counts'].shape)}"
    )


sample_batches = {}

for split_name, loader in [
    ("train", train_loader),
    ("dev", dev_loader),
    ("test", test_loader),
]:
    sample_batch = next(
        iter(loader)
    )

    validate_batch_structure(
        sample_batch,
        split_name,
    )

    sample_batches[
        split_name
    ] = {
        "num_patents": int(
            sample_batch[
                "num_patents"
            ]
        ),
        "num_claims": int(
            sample_batch[
                "num_claims"
            ]
        ),
        "num_tokens": int(
            sample_batch[
                "num_tokens"
            ]
        ),
        "num_token_edges": int(
            sample_batch[
                "token_edge_index"
            ].shape[1]
        ),
        "num_dependency_edges": int(
            sample_batch[
                "claim_edge_index"
            ].shape[1]
        ),
        "num_adjacent_edges": int(
            sample_batch[
                "adjacent_claim_edge_index"
            ].shape[1]
        ),
        "valid_bow_claims": int(
            sample_batch[
                "bow_valid"
            ].sum().item()
        ),
    }

    del sample_batch
    gc.collect()


# ============================================================
# 3.16 Update inferred dimensions
# ============================================================

first_train_batch = next(
    iter(train_loader)
)

observed_feature_dim = int(
    first_train_batch[
        "token_embeddings"
    ].shape[1]
)

if (
    CONFIG.plm_hidden_dim is not None
    and int(CONFIG.plm_hidden_dim)
    != observed_feature_dim
):
    raise ValueError(
        "PLM hidden dimension mismatch: "
        f"config={CONFIG.plm_hidden_dim}, "
        f"observed={observed_feature_dim}"
    )

CONFIG.plm_hidden_dim = (
    observed_feature_dim
)

if int(
    CONFIG.topic_embedding_dim
) != observed_feature_dim:
    print(
        "[INFO] topic_embedding_dim을 "
        f"{CONFIG.topic_embedding_dim}에서 "
        f"{observed_feature_dim}로 맞춥니다."
    )

    CONFIG.topic_embedding_dim = (
        observed_feature_dim
    )

del first_train_batch
gc.collect()


# ============================================================
# 3.17 Save manifest
# ============================================================

SECTION3_MANIFEST = {
    "created_at": (
        datetime.now().isoformat()
    ),
    "run_name": (
        CONFIG.run_name
    ),
    "feature_run_name": (
        CONFIG.feature_run_name
    ),
    "plm_hidden_dim": int(
        CONFIG.plm_hidden_dim
    ),
    "vocabulary_size": int(
        len(VOCAB)
    ),
    "batch_configuration": {
        "patents_per_batch": (
            PATENTS_PER_BATCH
        ),
        "maximum_claims_per_batch": (
            MAXIMUM_CLAIMS_PER_BATCH
        ),
        "maximum_tokens_per_batch": (
            MAXIMUM_TOKENS_PER_BATCH
        ),
        "estimated_subword_multiplier": (
            ESTIMATED_SUBWORD_MULTIPLIER
        ),
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "shard_aware": True,
    },
    "datasets": {
        "train": {
            "patents": len(
                train_dataset
            ),
            "shards": len(
                train_dataset.shard_paths
            ),
            "batches": len(
                train_loader
            ),
            "oversized_patents": len(
                train_batch_sampler
                .oversized_patents
            ),
        },
        "dev": {
            "patents": len(
                dev_dataset
            ),
            "shards": len(
                dev_dataset.shard_paths
            ),
            "batches": len(
                dev_loader
            ),
            "oversized_patents": len(
                dev_batch_sampler
                .oversized_patents
            ),
        },
        "test": {
            "patents": len(
                test_dataset
            ),
            "shards": len(
                test_dataset.shard_paths
            ),
            "batches": len(
                test_loader
            ),
            "oversized_patents": len(
                test_batch_sampler
                .oversized_patents
            ),
        },
    },
    "sample_batches": (
        sample_batches
    ),
    "batch_keys": sorted(
        DEVICE_TENSOR_KEYS
        | {
            "split",
            "shard_index",
            "patent_ids",
            "claim_keys",
            "num_patents",
            "num_claims",
            "num_tokens",
            "num_dependency_edges",
            "num_adjacent_edges",
            "num_non_adjacent_edges",
            "num_truncated_claims",
        }
    ),
    "cpc_labels_in_training_batch": False,
}

atomic_json_save_section3(
    SECTION3_MANIFEST,
    SECTION3_MANIFEST_PATH,
)


# ============================================================
# 3.18 Final status
# ============================================================

print("\n" + "=" * 88)
print("SECTION 3 — DATASETS AND LOADERS COMPLETED")
print("=" * 88)
print(
    f"PLM hidden dimension : "
    f"{CONFIG.plm_hidden_dim}"
)
print(
    f"Vocabulary size      : "
    f"{len(VOCAB):,}"
)
print(
    f"Patents per batch    : "
    f"{PATENTS_PER_BATCH}"
)
print(
    f"Max claims per batch : "
    f"{MAXIMUM_CLAIMS_PER_BATCH}"
)
print(
    f"Max estimated tokens : "
    f"{MAXIMUM_TOKENS_PER_BATCH:,}"
)
print(
    f"Workers              : "
    f"{NUM_WORKERS}"
)
print(
    f"Train                 : "
    f"patents={len(train_dataset):,}, "
    f"batches={len(train_loader):,}"
)
print(
    f"DEV                   : "
    f"patents={len(dev_dataset):,}, "
    f"batches={len(dev_loader):,}"
)
print(
    f"TEST                  : "
    f"patents={len(test_dataset):,}, "
    f"batches={len(test_loader):,}"
)
print(
    f"Train oversized       : "
    f"{len(train_batch_sampler.oversized_patents)}"
)
print(
    f"DEV oversized         : "
    f"{len(dev_batch_sampler.oversized_patents)}"
)
print(
    f"TEST oversized        : "
    f"{len(test_batch_sampler.oversized_patents)}"
)
print(
    f"Manifest              : "
    f"{SECTION3_MANIFEST_PATH}"
)
print("=" * 88)
print(
    "[PASS] claim_to_patent가 모든 batch에 포함되었습니다."
)
print(
    "[PASS] CPC labels는 학습 batch에 포함되지 않았습니다."
)
print(
    "[PASS] Feature shard는 영구 local cache에 복사하지 않습니다."
)
print(
    "[NEXT] BATCH PASS 출력과 마지막 요약을 보내주세요."
)
