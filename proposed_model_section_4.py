# ============================================================
# SECTION 4 — DEPTH-OT V2 MODEL ARCHITECTURE
# Patent-level semantics + claim-level hierarchy
# ============================================================

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 4.0 Preconditions
# ------------------------------------------------------------

_REQUIRED_GLOBALS = [
    "CONFIG",
    "DEVICE",
    "VOCAB",
    "TOKEN_TO_ID",
    "RUN_LOG_DIR",
]

_missing_globals = [name for name in _REQUIRED_GLOBALS if name not in globals()]
if _missing_globals:
    raise RuntimeError(
        "Section 4 실행 전에 필요한 전역 변수가 없습니다: "
        + ", ".join(_missing_globals)
        + "\nSection 0–3을 먼저 실행하세요."
    )

DEVICE = torch.device(DEVICE)
RUN_LOG_DIR = Path(RUN_LOG_DIR)
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = len(VOCAB)
NUM_TOPICS = int(getattr(CONFIG, "num_topics", 30))
PLM_HIDDEN_DIM = int(getattr(CONFIG, "plm_hidden_dim", 768))

if VOCAB_SIZE != int(getattr(CONFIG, "vocab_size", VOCAB_SIZE)):
    raise RuntimeError(
        f"Vocabulary size mismatch: len(VOCAB)={VOCAB_SIZE}, "
        f"CONFIG.vocab_size={getattr(CONFIG, 'vocab_size', None)}"
    )

if NUM_TOPICS <= 1:
    raise ValueError(f"NUM_TOPICS must be > 1, got {NUM_TOPICS}")

if PLM_HIDDEN_DIM <= 0:
    raise ValueError(f"Invalid PLM hidden dimension: {PLM_HIDDEN_DIM}")


# ------------------------------------------------------------
# 4.1 Architecture configuration
# ------------------------------------------------------------

GRAPH_HIDDEN_DIM = int(getattr(CONFIG, "graph_hidden_dim", 256))
CLAIM_HIDDEN_DIM = int(getattr(CONFIG, "claim_hidden_dim", 256))
TOPIC_EMBEDDING_DIM = int(
    getattr(CONFIG, "decoder_topic_embedding_dim", 256)
)

GRAPH_DROPOUT = float(getattr(CONFIG, "graph_dropout", 0.10))
CLAIM_DROPOUT = float(getattr(CONFIG, "claim_dropout", 0.10))
TOPIC_DROPOUT = float(getattr(CONFIG, "topic_dropout", 0.05))

MIN_LOGVAR = float(getattr(CONFIG, "minimum_log_variance", -8.0))
MAX_LOGVAR = float(getattr(CONFIG, "maximum_log_variance", 4.0))

DECODER_INITIAL_TEMPERATURE = float(
    getattr(CONFIG, "decoder_initial_temperature", 0.20)
)
DECODER_MINIMUM_TEMPERATURE = float(
    getattr(CONFIG, "decoder_minimum_temperature", 0.05)
)
DECODER_MAXIMUM_TEMPERATURE = float(
    getattr(CONFIG, "decoder_maximum_temperature", 2.00)
)

TOPIC_COSINE_MARGIN = float(
    getattr(CONFIG, "topic_cosine_margin", 0.20)
)

ANCHOR_MINIMUM_SEPARATION = float(
    getattr(CONFIG, "anchor_minimum_separation", 0.02)
)

MODEL_EPSILON = 1.0e-8
SECTION4_MANIFEST_PATH = RUN_LOG_DIR / "section4_model_manifest.json"


# ------------------------------------------------------------
# 4.2 General utilities
# ------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _first_present(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    *,
    required: bool = False,
    description: str = "value",
) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]

    if required:
        available = sorted(str(key) for key in mapping.keys())
        raise KeyError(
            f"Could not find {description}. "
            f"Tried keys={list(names)}. "
            f"Available keys={available}"
        )

    return None


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)

    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)

    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]

    return value


def _parameter_count(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def _segment_mean(
    values: torch.Tensor,
    segment_index: torch.Tensor,
    number_of_segments: int,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(
            f"values must be rank 2, got shape={tuple(values.shape)}"
        )

    segment_index = segment_index.long().reshape(-1)

    if values.shape[0] != segment_index.numel():
        raise ValueError(
            "Segment index length mismatch: "
            f"values={values.shape[0]}, index={segment_index.numel()}"
        )

    if number_of_segments <= 0:
        raise ValueError(
            f"number_of_segments must be positive, got {number_of_segments}"
        )

    if segment_index.numel() == 0:
        return values.new_zeros((number_of_segments, values.shape[1]))

    minimum_index = int(segment_index.min().item())
    maximum_index = int(segment_index.max().item())

    if minimum_index < 0 or maximum_index >= number_of_segments:
        raise IndexError(
            f"Segment indices out of range: min={minimum_index}, "
            f"max={maximum_index}, number_of_segments={number_of_segments}"
        )

    sums = values.new_zeros((number_of_segments, values.shape[1]))
    sums.index_add_(0, segment_index, values)

    counts = values.new_zeros((number_of_segments, 1))
    ones = values.new_ones((values.shape[0], 1))
    counts.index_add_(0, segment_index, ones)

    return sums / counts.clamp_min(1.0)


def _segment_sum_and_count(
    values: torch.Tensor,
    segment_index: torch.Tensor,
    number_of_segments: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    segment_index = segment_index.long().reshape(-1)

    sums = values.new_zeros((number_of_segments, values.shape[1]))
    counts = values.new_zeros((number_of_segments, 1))

    if values.shape[0] > 0:
        sums.index_add_(0, segment_index, values)
        counts.index_add_(
            0,
            segment_index,
            values.new_ones((values.shape[0], 1)),
        )

    return sums, counts


# ------------------------------------------------------------
# 4.3 Weighted token graph layer
# ------------------------------------------------------------

class WeightedGraphConvolution(nn.Module):
    """
    Lightweight weighted graph convolution.

    The layer aggregates weighted neighboring token representations and
    combines them with a transformed self representation. It does not
    require torch_geometric.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_dimension = int(input_dimension)
        self.output_dimension = int(output_dimension)

        self.self_projection = nn.Linear(
            self.input_dimension,
            self.output_dimension,
            bias=True,
        )
        self.neighbor_projection = nn.Linear(
            self.input_dimension,
            self.output_dimension,
            bias=False,
        )

        self.layer_norm = nn.LayerNorm(self.output_dimension)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError(
                "node_features must be rank 2, "
                f"got {tuple(node_features.shape)}"
            )

        number_of_nodes = int(node_features.shape[0])

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E], "
                f"got {tuple(edge_index.shape)}"
            )

        edge_index = edge_index.long()

        if edge_index.shape[1] == 0:
            output = self.self_projection(node_features)
            output = self.layer_norm(output)
            output = F.gelu(output)
            return self.dropout(output)

        source = edge_index[0]
        target = edge_index[1]

        minimum_node = int(edge_index.min().item())
        maximum_node = int(edge_index.max().item())

        if minimum_node < 0 or maximum_node >= number_of_nodes:
            raise IndexError(
                f"Token edge index out of range: min={minimum_node}, "
                f"max={maximum_node}, nodes={number_of_nodes}"
            )

        if edge_weight is None:
            edge_weight = node_features.new_ones(source.numel())
        else:
            edge_weight = edge_weight.to(
                device=node_features.device,
                dtype=node_features.dtype,
            ).reshape(-1)

        if edge_weight.numel() != source.numel():
            raise ValueError(
                f"edge_weight length={edge_weight.numel()} does not match "
                f"number of edges={source.numel()}"
            )

        edge_weight = edge_weight.clamp_min(0.0)

        degree = node_features.new_zeros(number_of_nodes)
        degree.index_add_(0, target, edge_weight)
        degree = degree.clamp_min(MODEL_EPSILON)

        normalization = edge_weight * torch.rsqrt(
            degree[source] * degree[target]
        )

        projected_neighbors = self.neighbor_projection(node_features)
        messages = projected_neighbors[source]
        messages = messages * normalization.unsqueeze(-1)

        aggregated = node_features.new_zeros(
            (number_of_nodes, self.output_dimension)
        )
        aggregated.index_add_(0, target, messages)

        output = self.self_projection(node_features) + aggregated
        output = self.layer_norm(output)
        output = F.gelu(output)
        output = self.dropout(output)

        return output


# ------------------------------------------------------------
# 4.4 Token graph encoder
# ------------------------------------------------------------

class TokenGraphEncoder(nn.Module):

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
        )

        self.graph_layer_1 = WeightedGraphConvolution(
            input_dimension=hidden_dimension,
            output_dimension=hidden_dimension,
            dropout=dropout,
        )
        self.graph_layer_2 = WeightedGraphConvolution(
            input_dimension=hidden_dimension,
            output_dimension=hidden_dimension,
            dropout=dropout,
        )

        self.output_norm = nn.LayerNorm(hidden_dimension)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_edge_index: torch.Tensor,
        token_edge_weight: Optional[torch.Tensor],
        token_to_claim: torch.Tensor,
        number_of_claims: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature shards are float16, but model computation is float32.
        token_embeddings = token_embeddings.float()

        hidden = self.input_projection(token_embeddings)

        residual = hidden
        hidden = self.graph_layer_1(
            hidden,
            token_edge_index,
            token_edge_weight,
        )
        hidden = hidden + residual

        residual = hidden
        hidden = self.graph_layer_2(
            hidden,
            token_edge_index,
            token_edge_weight,
        )
        hidden = self.output_norm(hidden + residual)

        claim_representations = _segment_mean(
            values=hidden,
            segment_index=token_to_claim,
            number_of_segments=number_of_claims,
        )

        return claim_representations, hidden


# ------------------------------------------------------------
# 4.5 Bidirectional claim dependency encoder
# ------------------------------------------------------------

class BidirectionalClaimDependencyEncoder(nn.Module):

    def __init__(
        self,
        hidden_dimension: int,
        output_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.context_network = nn.Sequential(
            nn.Linear(hidden_dimension * 3, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dimension, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.GELU(),
        )

        self.residual_projection = (
            nn.Identity()
            if hidden_dimension == output_dimension
            else nn.Linear(hidden_dimension, output_dimension)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        claim_representations: torch.Tensor,
        claim_edge_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        number_of_claims = int(claim_representations.shape[0])
        hidden_dimension = int(claim_representations.shape[1])

        parent_context = claim_representations.new_zeros(
            (number_of_claims, hidden_dimension)
        )
        child_context = claim_representations.new_zeros(
            (number_of_claims, hidden_dimension)
        )

        parent_counts = claim_representations.new_zeros(
            (number_of_claims, 1)
        )
        child_counts = claim_representations.new_zeros(
            (number_of_claims, 1)
        )

        if claim_edge_index is not None and claim_edge_index.numel() > 0:
            if claim_edge_index.ndim != 2 or claim_edge_index.shape[0] != 2:
                raise ValueError(
                    "claim_edge_index must have shape [2, E], "
                    f"got {tuple(claim_edge_index.shape)}"
                )

            claim_edge_index = claim_edge_index.long()
            parent = claim_edge_index[0]
            child = claim_edge_index[1]

            minimum_claim = int(claim_edge_index.min().item())
            maximum_claim = int(claim_edge_index.max().item())

            if minimum_claim < 0 or maximum_claim >= number_of_claims:
                raise IndexError(
                    f"Claim edge index out of range: min={minimum_claim}, "
                    f"max={maximum_claim}, claims={number_of_claims}"
                )

            # Information from parent claims to child claims.
            parent_context.index_add_(
                0,
                child,
                claim_representations[parent],
            )
            parent_counts.index_add_(
                0,
                child,
                claim_representations.new_ones((parent.numel(), 1)),
            )

            # Information from child claims to parent claims.
            child_context.index_add_(
                0,
                parent,
                claim_representations[child],
            )
            child_counts.index_add_(
                0,
                parent,
                claim_representations.new_ones((child.numel(), 1)),
            )

        parent_context = parent_context / parent_counts.clamp_min(1.0)
        child_context = child_context / child_counts.clamp_min(1.0)

        combined = torch.cat(
            [
                claim_representations,
                parent_context,
                child_context,
            ],
            dim=-1,
        )

        contextualized = self.context_network(combined)
        contextualized = contextualized + self.residual_projection(
            claim_representations
        )

        return self.dropout(contextualized)


# ------------------------------------------------------------
# 4.6 Logistic-normal variational topic encoder
# ------------------------------------------------------------

class VariationalTopicEncoder(nn.Module):

    def __init__(
        self,
        input_dimension: int,
        number_of_topics: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.number_of_topics = int(number_of_topics)

        self.hidden_network = nn.Sequential(
            nn.Linear(input_dimension, input_dimension),
            nn.LayerNorm(input_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mean_projection = nn.Linear(
            input_dimension,
            self.number_of_topics,
        )
        self.log_variance_projection = nn.Linear(
            input_dimension,
            self.number_of_topics,
        )

        self.posterior_norm = nn.LayerNorm(self.number_of_topics)

    def forward(
        self,
        claim_representations: torch.Tensor,
        sample: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.hidden_network(claim_representations)

        posterior_mean = self.mean_projection(hidden)
        posterior_log_variance = self.log_variance_projection(hidden)
        posterior_log_variance = posterior_log_variance.clamp(
            min=MIN_LOGVAR,
            max=MAX_LOGVAR,
        )

        if sample:
            standard_deviation = torch.exp(
                0.5 * posterior_log_variance
            )
            noise = torch.randn_like(standard_deviation)
            latent = posterior_mean + noise * standard_deviation
        else:
            latent = posterior_mean

        normalized_latent = self.posterior_norm(latent)
        claim_theta = F.softmax(normalized_latent, dim=-1)

        return (
            claim_theta,
            posterior_mean,
            posterior_log_variance,
            latent,
        )


# ------------------------------------------------------------
# 4.7 Factorized topic-word decoder
# ------------------------------------------------------------

class FactorizedTopicWordDecoder(nn.Module):
    """
    Topic-word logits are generated from trainable topic and word
    embeddings. Topic separation therefore directly changes beta rather
    than regularizing an unrelated auxiliary representation.
    """

    def __init__(
        self,
        number_of_topics: int,
        vocabulary_size: int,
        embedding_dimension: int,
        initial_temperature: float,
    ) -> None:
        super().__init__()

        self.number_of_topics = int(number_of_topics)
        self.vocabulary_size = int(vocabulary_size)
        self.embedding_dimension = int(embedding_dimension)

        self.topic_embeddings = nn.Parameter(
            torch.empty(
                self.number_of_topics,
                self.embedding_dimension,
            )
        )
        self.word_embeddings = nn.Parameter(
            torch.empty(
                self.vocabulary_size,
                self.embedding_dimension,
            )
        )
        self.word_bias = nn.Parameter(
            torch.zeros(self.vocabulary_size)
        )

        initial_temperature = max(
            DECODER_MINIMUM_TEMPERATURE,
            min(
                DECODER_MAXIMUM_TEMPERATURE,
                float(initial_temperature),
            ),
        )

        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(initial_temperature), dtype=torch.float32)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.topic_embeddings)
        nn.init.normal_(
            self.word_embeddings,
            mean=0.0,
            std=1.0 / math.sqrt(self.embedding_dimension),
        )
        nn.init.zeros_(self.word_bias)

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(
            min=DECODER_MINIMUM_TEMPERATURE,
            max=DECODER_MAXIMUM_TEMPERATURE,
        )

    def normalized_topic_embeddings(self) -> torch.Tensor:
        return F.normalize(
            self.topic_embeddings,
            p=2,
            dim=-1,
            eps=MODEL_EPSILON,
        )

    def normalized_word_embeddings(self) -> torch.Tensor:
        return F.normalize(
            self.word_embeddings,
            p=2,
            dim=-1,
            eps=MODEL_EPSILON,
        )

    def topic_word_logits(self) -> torch.Tensor:
        normalized_topics = self.normalized_topic_embeddings()
        normalized_words = self.normalized_word_embeddings()

        logits = torch.matmul(
            normalized_topics,
            normalized_words.transpose(0, 1),
        )
        logits = logits / self.temperature()
        logits = logits + self.word_bias.unsqueeze(0)

        return logits

    def beta(self) -> torch.Tensor:
        return F.softmax(self.topic_word_logits(), dim=-1)

    def forward(
        self,
        topic_proportions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta()

        word_probabilities = torch.matmul(topic_proportions, beta)
        word_probabilities = word_probabilities.clamp_min(MODEL_EPSILON)
        word_probabilities = (
            word_probabilities
            / word_probabilities.sum(dim=-1, keepdim=True).clamp_min(
                MODEL_EPSILON
            )
        )

        log_word_probabilities = torch.log(word_probabilities)

        return log_word_probabilities, beta


# ------------------------------------------------------------
# 4.8 Independent depth anchors
# ------------------------------------------------------------

class IndependentDepthAnchors(nn.Module):
    """
    Each topic has an independent scalar depth coordinate in (0, 1).

    Coordinates are not sorted by topic ID. They are initialized across
    the interval in a random topic order and can move independently.
    """

    def __init__(
        self,
        number_of_topics: int,
        seed: int,
    ) -> None:
        super().__init__()

        self.number_of_topics = int(number_of_topics)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 4_001)

        initial_coordinates = torch.linspace(
            0.10,
            0.90,
            steps=self.number_of_topics,
            dtype=torch.float32,
        )

        permutation = torch.randperm(
            self.number_of_topics,
            generator=generator,
        )
        initial_coordinates = initial_coordinates[permutation]

        initial_coordinates = initial_coordinates.clamp(
            1.0e-4,
            1.0 - 1.0e-4,
        )

        initial_logits = torch.logit(initial_coordinates)

        self.raw_coordinates = nn.Parameter(initial_logits)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_coordinates)

    def sorted_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        coordinates = self.forward()
        return torch.sort(coordinates)


# ------------------------------------------------------------
# 4.9 Complete Depth-OT V2 model
# ------------------------------------------------------------

class DepthOTV2Model(nn.Module):

    def __init__(
        self,
        plm_hidden_dimension: int,
        graph_hidden_dimension: int,
        claim_hidden_dimension: int,
        number_of_topics: int,
        vocabulary_size: int,
        topic_embedding_dimension: int,
        seed: int,
    ) -> None:
        super().__init__()

        self.plm_hidden_dimension = int(plm_hidden_dimension)
        self.graph_hidden_dimension = int(graph_hidden_dimension)
        self.claim_hidden_dimension = int(claim_hidden_dimension)
        self.number_of_topics = int(number_of_topics)
        self.vocabulary_size = int(vocabulary_size)
        self.topic_embedding_dimension = int(topic_embedding_dimension)

        self.token_graph_encoder = TokenGraphEncoder(
            input_dimension=self.plm_hidden_dimension,
            hidden_dimension=self.graph_hidden_dimension,
            dropout=GRAPH_DROPOUT,
        )

        self.claim_dependency_encoder = (
            BidirectionalClaimDependencyEncoder(
                hidden_dimension=self.graph_hidden_dimension,
                output_dimension=self.claim_hidden_dimension,
                dropout=CLAIM_DROPOUT,
            )
        )

        self.variational_topic_encoder = VariationalTopicEncoder(
            input_dimension=self.claim_hidden_dimension,
            number_of_topics=self.number_of_topics,
            dropout=TOPIC_DROPOUT,
        )

        self.topic_word_decoder = FactorizedTopicWordDecoder(
            number_of_topics=self.number_of_topics,
            vocabulary_size=self.vocabulary_size,
            embedding_dimension=self.topic_embedding_dimension,
            initial_temperature=DECODER_INITIAL_TEMPERATURE,
        )

        self.depth_anchors = IndependentDepthAnchors(
            number_of_topics=self.number_of_topics,
            seed=seed,
        )

    # --------------------------------------------------------
    # Batch tensor resolution
    # --------------------------------------------------------

    def resolve_batch_tensors(
        self,
        batch: Mapping[str, Any],
    ) -> Dict[str, Any]:
        token_embeddings = _first_present(
            batch,
            [
                "token_embeddings",
                "node_features",
                "token_features",
            ],
            required=True,
            description="token embeddings",
        )

        token_edge_index = _first_present(
            batch,
            [
                "token_edge_index",
                "graph_edge_index",
                "edge_index",
            ],
            required=True,
            description="token graph edge index",
        )

        token_edge_weight = _first_present(
            batch,
            [
                "token_edge_weight",
                "graph_edge_weight",
                "edge_weight",
            ],
            required=False,
            description="token graph edge weights",
        )

        token_to_claim = _first_present(
            batch,
            [
                "token_to_claim",
                "node_to_claim",
                "token_claim_index",
            ],
            required=True,
            description="token-to-claim index",
        )

        claim_to_patent = _first_present(
            batch,
            [
                "claim_to_patent",
                "claim_patent_index",
            ],
            required=True,
            description="claim-to-patent index",
        )

        claim_edge_index = _first_present(
            batch,
            [
                "claim_edge_index",
                "dependency_edge_index",
                "claim_dependency_edge_index",
            ],
            required=False,
            description="claim dependency edge index",
        )

        claim_depth = _first_present(
            batch,
            [
                "claim_depth",
                "claim_depths",
                "depth",
            ],
            required=False,
            description="claim depth",
        )

        claim_bow = _first_present(
            batch,
            [
                "claim_bow",
                "bow",
                "bows",
                "bag_of_words",
            ],
            required=False,
            description="claim bag-of-words tensor",
        )

        return {
            "token_embeddings": token_embeddings,
            "token_edge_index": token_edge_index,
            "token_edge_weight": token_edge_weight,
            "token_to_claim": token_to_claim,
            "claim_to_patent": claim_to_patent,
            "claim_edge_index": claim_edge_index,
            "claim_depth": claim_depth,
            "claim_bow": claim_bow,
        }

    # --------------------------------------------------------
    # Patent aggregation
    # --------------------------------------------------------

    def aggregate_claim_topics_to_patents(
        self,
        claim_theta: torch.Tensor,
        claim_to_patent: torch.Tensor,
        number_of_patents: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        claim_to_patent = claim_to_patent.long().reshape(-1)

        if claim_theta.shape[0] != claim_to_patent.numel():
            raise ValueError(
                f"claim_theta rows={claim_theta.shape[0]} do not match "
                f"claim_to_patent length={claim_to_patent.numel()}"
            )

        if claim_to_patent.numel() == 0:
            raise ValueError("A batch cannot contain zero claims.")

        inferred_patent_count = int(
            claim_to_patent.max().item()
        ) + 1

        if number_of_patents is None:
            number_of_patents = inferred_patent_count
        else:
            number_of_patents = int(number_of_patents)

        if number_of_patents < inferred_patent_count:
            raise ValueError(
                f"number_of_patents={number_of_patents} is smaller than "
                f"inferred count={inferred_patent_count}"
            )

        patent_sums, patent_claim_counts = _segment_sum_and_count(
            values=claim_theta,
            segment_index=claim_to_patent,
            number_of_segments=number_of_patents,
        )

        patent_theta = patent_sums / patent_claim_counts.clamp_min(1.0)
        patent_theta = patent_theta.clamp_min(MODEL_EPSILON)
        patent_theta = patent_theta / patent_theta.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(MODEL_EPSILON)

        return patent_theta, patent_claim_counts.squeeze(-1)

    # --------------------------------------------------------
    # Decoder and anchor interfaces
    # --------------------------------------------------------

    def get_beta(self) -> torch.Tensor:
        return self.topic_word_decoder.beta()

    def get_topic_word_logits(self) -> torch.Tensor:
        return self.topic_word_decoder.topic_word_logits()

    def get_topic_embeddings(self) -> torch.Tensor:
        return self.topic_word_decoder.topic_embeddings

    def get_anchor_coordinates(self) -> torch.Tensor:
        return self.depth_anchors()

    # --------------------------------------------------------
    # Regularization components used by Section 5
    # --------------------------------------------------------

    def topic_embedding_separation_loss(
        self,
        margin: float = TOPIC_COSINE_MARGIN,
    ) -> torch.Tensor:
        normalized_topics = (
            self.topic_word_decoder.normalized_topic_embeddings()
        )

        cosine_matrix = torch.matmul(
            normalized_topics,
            normalized_topics.transpose(0, 1),
        )

        topic_count = cosine_matrix.shape[0]
        off_diagonal_mask = ~torch.eye(
            topic_count,
            dtype=torch.bool,
            device=cosine_matrix.device,
        )

        off_diagonal_cosines = cosine_matrix[off_diagonal_mask]

        if off_diagonal_cosines.numel() == 0:
            return cosine_matrix.new_zeros(())

        return F.relu(
            off_diagonal_cosines - float(margin)
        ).pow(2).mean()

    def beta_cosine_separation_loss(
        self,
        margin: float = TOPIC_COSINE_MARGIN,
    ) -> torch.Tensor:
        beta = self.get_beta()
        normalized_beta = F.normalize(
            beta,
            p=2,
            dim=-1,
            eps=MODEL_EPSILON,
        )

        cosine_matrix = torch.matmul(
            normalized_beta,
            normalized_beta.transpose(0, 1),
        )

        topic_count = cosine_matrix.shape[0]
        off_diagonal_mask = ~torch.eye(
            topic_count,
            dtype=torch.bool,
            device=cosine_matrix.device,
        )

        off_diagonal_cosines = cosine_matrix[off_diagonal_mask]

        if off_diagonal_cosines.numel() == 0:
            return cosine_matrix.new_zeros(())

        return F.relu(
            off_diagonal_cosines - float(margin)
        ).pow(2).mean()

    def anchor_repulsion_loss(
        self,
        minimum_separation: float = ANCHOR_MINIMUM_SEPARATION,
    ) -> torch.Tensor:
        coordinates = self.get_anchor_coordinates()
        pairwise_distance = torch.abs(
            coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        )

        topic_count = coordinates.numel()
        off_diagonal_mask = ~torch.eye(
            topic_count,
            dtype=torch.bool,
            device=coordinates.device,
        )

        distances = pairwise_distance[off_diagonal_mask]

        if distances.numel() == 0:
            return coordinates.new_zeros(())

        return F.relu(
            float(minimum_separation) - distances
        ).pow(2).mean()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        sample: Optional[bool] = None,
        decode: bool = True,
    ) -> Dict[str, Any]:
        tensors = self.resolve_batch_tensors(batch)

        token_embeddings = tensors["token_embeddings"]
        token_edge_index = tensors["token_edge_index"].long()
        token_edge_weight = tensors["token_edge_weight"]
        token_to_claim = tensors["token_to_claim"].long().reshape(-1)
        claim_to_patent = (
            tensors["claim_to_patent"].long().reshape(-1)
        )
        claim_edge_index = tensors["claim_edge_index"]
        claim_depth = tensors["claim_depth"]
        claim_bow = tensors["claim_bow"]

        number_of_claims = int(claim_to_patent.numel())

        if number_of_claims <= 0:
            raise ValueError("The batch contains zero claims.")

        if token_embeddings.shape[0] != token_to_claim.numel():
            raise ValueError(
                f"token_embeddings rows={token_embeddings.shape[0]} "
                f"do not match token_to_claim={token_to_claim.numel()}"
            )

        if token_to_claim.numel() > 0:
            maximum_claim_index = int(token_to_claim.max().item())
            minimum_claim_index = int(token_to_claim.min().item())

            if (
                minimum_claim_index < 0
                or maximum_claim_index >= number_of_claims
            ):
                raise IndexError(
                    f"token_to_claim indices out of range: "
                    f"min={minimum_claim_index}, "
                    f"max={maximum_claim_index}, "
                    f"claims={number_of_claims}"
                )

        if sample is None:
            sample = bool(self.training)

        base_claim_representations, token_representations = (
            self.token_graph_encoder(
                token_embeddings=token_embeddings,
                token_edge_index=token_edge_index,
                token_edge_weight=token_edge_weight,
                token_to_claim=token_to_claim,
                number_of_claims=number_of_claims,
            )
        )

        claim_representations = self.claim_dependency_encoder(
            claim_representations=base_claim_representations,
            claim_edge_index=claim_edge_index,
        )

        (
            claim_theta,
            posterior_mean,
            posterior_log_variance,
            latent,
        ) = self.variational_topic_encoder(
            claim_representations=claim_representations,
            sample=bool(sample),
        )

        patent_theta, patent_claim_counts = (
            self.aggregate_claim_topics_to_patents(
                claim_theta=claim_theta,
                claim_to_patent=claim_to_patent,
            )
        )

        log_word_probabilities = None
        beta = None

        if decode:
            log_word_probabilities, beta = self.topic_word_decoder(
                claim_theta
            )

        outputs: Dict[str, Any] = {
            # Main topic outputs
            "claim_theta": claim_theta,
            "patent_theta": patent_theta,
            "theta": claim_theta,

            # Variational outputs
            "posterior_mean": posterior_mean,
            "posterior_log_variance": posterior_log_variance,
            "latent": latent,

            # Representations
            "token_representations": token_representations,
            "base_claim_representations": base_claim_representations,
            "claim_representations": claim_representations,

            # Decoder outputs
            "log_word_probabilities": log_word_probabilities,
            "beta": beta,
            "topic_word_distribution": beta,

            # Hierarchy outputs
            "anchor_coordinates": self.get_anchor_coordinates(),
            "claim_depth": claim_depth,
            "claim_edge_index": claim_edge_index,

            # Batch indexing
            "claim_to_patent": claim_to_patent,
            "patent_claim_counts": patent_claim_counts,

            # Targets retained for Section 5
            "claim_bow": claim_bow,
        }

        return outputs


# ------------------------------------------------------------
# 4.10 Instantiate model
# ------------------------------------------------------------

depth_ot_v2_model = DepthOTV2Model(
    plm_hidden_dimension=PLM_HIDDEN_DIM,
    graph_hidden_dimension=GRAPH_HIDDEN_DIM,
    claim_hidden_dimension=CLAIM_HIDDEN_DIM,
    number_of_topics=NUM_TOPICS,
    vocabulary_size=VOCAB_SIZE,
    topic_embedding_dimension=TOPIC_EMBEDDING_DIM,
    seed=int(getattr(CONFIG, "seed", 42)),
).to(DEVICE)

# Compatibility aliases for later sections.
depth_ot_model = depth_ot_v2_model
MODEL = depth_ot_v2_model

total_parameters, trainable_parameters = _parameter_count(
    depth_ot_v2_model
)


# ------------------------------------------------------------
# 4.11 Obtain one Section 3 sample batch
# ------------------------------------------------------------

def _find_section3_sample_batch() -> Tuple[str, Mapping[str, Any]]:
    candidate_names = [
        "FIRST_TRAIN_BATCH",
        "first_train_batch",
        "SAMPLE_TRAIN_BATCH",
        "sample_train_batch",
        "TRAIN_SAMPLE_BATCH",
        "train_sample_batch",
    ]

    for candidate_name in candidate_names:
        candidate = globals().get(candidate_name)
        if isinstance(candidate, Mapping):
            return candidate_name, candidate

    loader_names = [
        "train_loader",
        "TRAIN_LOADER",
        "train_dataloader",
        "TRAIN_DATALOADER",
    ]

    for loader_name in loader_names:
        loader = globals().get(loader_name)
        if loader is not None:
            return f"next(iter({loader_name}))", next(iter(loader))

    raise RuntimeError(
        "Section 3의 train sample batch 또는 train loader를 찾지 "
        "못했습니다. 사용 가능한 batch key와 loader 변수명을 확인하세요."
    )


sample_batch_source, section4_sample_batch = (
    _find_section3_sample_batch()
)

section4_sample_batch_device = _move_to_device(
    section4_sample_batch,
    DEVICE,
)


# ------------------------------------------------------------
# 4.12 Forward-pass validation
# ------------------------------------------------------------

depth_ot_v2_model.eval()

with torch.no_grad():
    section4_outputs_1 = depth_ot_v2_model(
        section4_sample_batch_device,
        sample=False,
        decode=True,
    )

    section4_outputs_2 = depth_ot_v2_model(
        section4_sample_batch_device,
        sample=False,
        decode=True,
    )

claim_theta = section4_outputs_1["claim_theta"]
patent_theta = section4_outputs_1["patent_theta"]
beta = section4_outputs_1["beta"]
anchor_coordinates = section4_outputs_1["anchor_coordinates"]
log_word_probabilities = section4_outputs_1[
    "log_word_probabilities"
]
claim_representations = section4_outputs_1[
    "claim_representations"
]

if claim_theta.ndim != 2 or claim_theta.shape[1] != NUM_TOPICS:
    raise RuntimeError(
        f"Invalid claim_theta shape: {tuple(claim_theta.shape)}"
    )

if patent_theta.ndim != 2 or patent_theta.shape[1] != NUM_TOPICS:
    raise RuntimeError(
        f"Invalid patent_theta shape: {tuple(patent_theta.shape)}"
    )

if beta.shape != (NUM_TOPICS, VOCAB_SIZE):
    raise RuntimeError(
        f"Invalid beta shape: {tuple(beta.shape)}, "
        f"expected {(NUM_TOPICS, VOCAB_SIZE)}"
    )

if log_word_probabilities.shape != (
    claim_theta.shape[0],
    VOCAB_SIZE,
):
    raise RuntimeError(
        "Invalid log_word_probabilities shape: "
        f"{tuple(log_word_probabilities.shape)}"
    )

if anchor_coordinates.shape != (NUM_TOPICS,):
    raise RuntimeError(
        f"Invalid anchor shape: {tuple(anchor_coordinates.shape)}"
    )

for output_name, output_tensor in [
    ("claim_theta", claim_theta),
    ("patent_theta", patent_theta),
    ("beta", beta),
    ("anchor_coordinates", anchor_coordinates),
    ("log_word_probabilities", log_word_probabilities),
    ("claim_representations", claim_representations),
]:
    if not torch.isfinite(output_tensor).all():
        raise RuntimeError(
            f"Non-finite values found in {output_name}"
        )

claim_sum_error = float(
    torch.max(
        torch.abs(
            claim_theta.sum(dim=-1)
            - torch.ones_like(claim_theta[:, 0])
        )
    ).item()
)

patent_sum_error = float(
    torch.max(
        torch.abs(
            patent_theta.sum(dim=-1)
            - torch.ones_like(patent_theta[:, 0])
        )
    ).item()
)

beta_sum_error = float(
    torch.max(
        torch.abs(
            beta.sum(dim=-1)
            - torch.ones_like(beta[:, 0])
        )
    ).item()
)

deterministic_difference = float(
    torch.max(
        torch.abs(
            section4_outputs_1["claim_theta"]
            - section4_outputs_2["claim_theta"]
        )
    ).item()
)

if claim_sum_error > 1.0e-5:
    raise RuntimeError(
        f"claim_theta simplex error too large: {claim_sum_error}"
    )

if patent_sum_error > 1.0e-5:
    raise RuntimeError(
        f"patent_theta simplex error too large: {patent_sum_error}"
    )

if beta_sum_error > 1.0e-5:
    raise RuntimeError(
        f"beta simplex error too large: {beta_sum_error}"
    )

if deterministic_difference > 1.0e-7:
    raise RuntimeError(
        "Evaluation-mode deterministic forward check failed: "
        f"max difference={deterministic_difference}"
    )

topic_embedding_separation = float(
    depth_ot_v2_model
    .topic_embedding_separation_loss()
    .detach()
    .item()
)

beta_separation = float(
    depth_ot_v2_model
    .beta_cosine_separation_loss()
    .detach()
    .item()
)

anchor_repulsion = float(
    depth_ot_v2_model
    .anchor_repulsion_loss()
    .detach()
    .item()
)

normalized_beta = F.normalize(
    beta,
    p=2,
    dim=-1,
    eps=MODEL_EPSILON,
)
beta_cosine_matrix = torch.matmul(
    normalized_beta,
    normalized_beta.transpose(0, 1),
)

off_diagonal_mask = ~torch.eye(
    NUM_TOPICS,
    dtype=torch.bool,
    device=beta.device,
)
initial_beta_max_cosine = float(
    beta_cosine_matrix[off_diagonal_mask].max().item()
)
initial_beta_mean_cosine = float(
    beta_cosine_matrix[off_diagonal_mask].mean().item()
)


# ------------------------------------------------------------
# 4.13 Save architecture manifest
# ------------------------------------------------------------

section4_manifest = {
    "created_at_utc": _utc_now(),
    "run_name": str(
        getattr(CONFIG, "run_name", "depth_ot_v2")
    ),
    "model_version": str(
        getattr(CONFIG, "model_version", "depth_ot_v2")
    ),
    "device": str(DEVICE),
    "sample_batch_source": sample_batch_source,
    "architecture": {
        "plm_hidden_dimension": PLM_HIDDEN_DIM,
        "graph_hidden_dimension": GRAPH_HIDDEN_DIM,
        "claim_hidden_dimension": CLAIM_HIDDEN_DIM,
        "number_of_topics": NUM_TOPICS,
        "vocabulary_size": VOCAB_SIZE,
        "topic_embedding_dimension": TOPIC_EMBEDDING_DIM,
        "graph_layers": 2,
        "graph_dropout": GRAPH_DROPOUT,
        "claim_dropout": CLAIM_DROPOUT,
        "topic_dropout": TOPIC_DROPOUT,
        "decoder_type": "factorized_cosine_topic_word_decoder",
        "decoder_initial_temperature": (
            DECODER_INITIAL_TEMPERATURE
        ),
        "anchor_parameterization": "independent_sigmoid",
        "topic_id_monotonic_anchor_constraint": False,
        "claim_to_patent_aggregation": "unweighted_mean",
        "batch_balanced_sinkhorn": False,
    },
    "parameter_counts": {
        "total": total_parameters,
        "trainable": trainable_parameters,
    },
    "sample_output": {
        "number_of_claims": int(claim_theta.shape[0]),
        "number_of_patents": int(patent_theta.shape[0]),
        "claim_theta_shape": list(claim_theta.shape),
        "patent_theta_shape": list(patent_theta.shape),
        "beta_shape": list(beta.shape),
        "claim_representation_shape": list(
            claim_representations.shape
        ),
        "anchor_shape": list(anchor_coordinates.shape),
        "claim_theta_max_sum_error": claim_sum_error,
        "patent_theta_max_sum_error": patent_sum_error,
        "beta_max_sum_error": beta_sum_error,
        "deterministic_max_difference": (
            deterministic_difference
        ),
    },
    "initial_diagnostics": {
        "topic_embedding_separation_loss": (
            topic_embedding_separation
        ),
        "beta_cosine_separation_loss": beta_separation,
        "anchor_repulsion_loss": anchor_repulsion,
        "beta_mean_off_diagonal_cosine": (
            initial_beta_mean_cosine
        ),
        "beta_max_off_diagonal_cosine": (
            initial_beta_max_cosine
        ),
        "decoder_temperature": float(
            depth_ot_v2_model
            .topic_word_decoder
            .temperature()
            .detach()
            .item()
        ),
        "anchor_minimum": float(
            anchor_coordinates.min().item()
        ),
        "anchor_maximum": float(
            anchor_coordinates.max().item()
        ),
    },
    "training_taxonomy_policy": {
        "cpc_used_in_model": False,
        "cpc_used_in_forward": False,
        "cpc_used_for_training": False,
    },
}

_atomic_json_dump(
    section4_manifest,
    SECTION4_MANIFEST_PATH,
)


# ------------------------------------------------------------
# 4.14 Clean validation tensors and return model to train mode
# ------------------------------------------------------------

del section4_outputs_2
depth_ot_v2_model.train()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ------------------------------------------------------------
# 4.15 Final report
# ------------------------------------------------------------

print("\n" + "=" * 88)
print("SECTION 4 — DEPTH-OT V2 MODEL ARCHITECTURE COMPLETED")
print("=" * 88)
print(f"Device                  : {DEVICE}")
print(f"Sample batch source     : {sample_batch_source}")
print(f"PLM hidden dimension    : {PLM_HIDDEN_DIM}")
print(f"Graph hidden dimension  : {GRAPH_HIDDEN_DIM}")
print(f"Claim hidden dimension  : {CLAIM_HIDDEN_DIM}")
print(f"Topic embedding dim     : {TOPIC_EMBEDDING_DIM}")
print(f"Topics                  : {NUM_TOPICS}")
print(f"Vocabulary              : {VOCAB_SIZE:,}")
print(f"Total parameters        : {total_parameters:,}")
print(f"Trainable parameters    : {trainable_parameters:,}")
print("-" * 88)
print(f"Claim theta             : {tuple(claim_theta.shape)}")
print(f"Patent theta            : {tuple(patent_theta.shape)}")
print(f"Topic-word beta         : {tuple(beta.shape)}")
print(
    f"Claim representation    : "
    f"{tuple(claim_representations.shape)}"
)
print(f"Depth anchors           : {tuple(anchor_coordinates.shape)}")
print("-" * 88)
print(f"Claim theta sum error   : {claim_sum_error:.3e}")
print(f"Patent theta sum error  : {patent_sum_error:.3e}")
print(f"Beta sum error          : {beta_sum_error:.3e}")
print(
    f"Deterministic max diff  : "
    f"{deterministic_difference:.3e}"
)
print("-" * 88)
print(
    f"Initial beta mean cosine: "
    f"{initial_beta_mean_cosine:.6f}"
)
print(
    f"Initial beta max cosine : "
    f"{initial_beta_max_cosine:.6f}"
)
print(
    f"Initial anchor range    : "
    f"[{anchor_coordinates.min().item():.4f}, "
    f"{anchor_coordinates.max().item():.4f}]"
)
print(
    f"Decoder temperature     : "
    f"{depth_ot_v2_model.topic_word_decoder.temperature().item():.4f}"
)
print("-" * 88)
print("Claim-level Sinkhorn    : DISABLED")
print("Patent aggregation      : mean(claim theta)")
print("Anchor ordering         : independent; not tied to topic IDs")
print("CPC in model/training   : False")
print(f"Manifest                : {SECTION4_MANIFEST_PATH}")
print("=" * 88)
print("[PASS] Section 4 model forward validation completed.")
print("[NEXT] 위의 전체 마지막 요약을 보내주세요.")
