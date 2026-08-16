# ==================================================================================================
# DEPTH-OT V2 — LARGE-SCALE THETA-SPACE CLUSTERING + CONSENSUS SEARCH
#
# Input:
#   Epoch 16 + Independent x8 + Depth lambda=0.1 + Confidence gamma=0 DEV theta
#
# Search:
#   - Raw / power / Hellinger / log / CLR representations
#   - PCA dimensions: None, 10, 15, 20, 25
#   - Euclidean KMeans / Spherical KMeans
#   - Diagonal / tied Gaussian Mixture
#   - Multiple random seeds
#   - Label-free cluster alignment and consensus voting
#
# Important:
#   - CPC labels are used only to rank DEV candidates.
#   - CPC labels are never supplied to KMeans, PCA, GMM, or consensus alignment.
#   - Number of clusters is fixed at 30.
#   - TEST data and TEST labels are not accessed.
# ==================================================================================================

from pathlib import Path
from datetime import datetime
import json
import pickle
import time
import warnings
import traceback

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------
# 1. PATHS
# --------------------------------------------------------------------------------------------------

PROJECT_ROOT = Path("/content/drive/MyDrive/depth_ot_patent")

INPUT_ROOT = (
    PROJECT_ROOT
    / "results/depth_ot_v2"
    / "depth_ot_v2_patent_semantic_seed42_20260814_055110"
    / "epoch016_independent_depth_confidence_dev_search"
)

THETA_PATH = INPUT_ROOT / "a08_l010_g000_dev_theta.npy"
RECORDS_PATH = PROJECT_ROOT / "data/processed/dev_records.pkl"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results/depth_ot_v2"
    / "depth_ot_v2_patent_semantic_seed42_20260814_055110"
    / "epoch016_theta_space_clustering_dev_search"
)

CODE_SAVE_PATH = Path(
    "/content/drive/MyDrive/depth_ot_code/scripts/"
    "search_theta_space_clustering_dev.py"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CODE_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")

# --------------------------------------------------------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------------------------------------------------------

EXPECTED_PATENTS = 9855
NUMBER_OF_TOPICS = 30
EPS = 1e-12

EXPECTED_INPUT_NMI = 0.307660
EXPECTED_INPUT_PUR_P = 0.369728
EXPECTED_INPUT_PUR_A = 0.427735
VALIDATION_TOLERANCE = 5e-5

# Representation search.
REPRESENTATIONS = [
    {"name": "raw",       "kind": "power", "q": 1.00},
    {"name": "pow025",    "kind": "power", "q": 0.25},
    {"name": "pow050",    "kind": "power", "q": 0.50},
    {"name": "pow075",    "kind": "power", "q": 0.75},
    {"name": "pow150",    "kind": "power", "q": 1.50},
    {"name": "pow200",    "kind": "power", "q": 2.00},
    {"name": "pow300",    "kind": "power", "q": 3.00},
    {"name": "pow400",    "kind": "power", "q": 4.00},
    {"name": "hellinger", "kind": "hellinger", "q": None},
    {"name": "log",       "kind": "log", "q": None},
    {"name": "clr",       "kind": "clr", "q": None},
]

PCA_DIMENSIONS = [None, 10, 15, 20, 25]
KMEANS_MODES = ["euclidean", "spherical"]

# Broad search seeds.
KMEANS_SEEDS = [17, 42, 73]
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 500

# GMM is more expensive, so one stable seed per broad configuration is used.
RUN_GMM = True
GMM_SEEDS = [42]
GMM_COVARIANCE_TYPES = ["diag", "tied"]
GMM_MAX_ITER = 400
GMM_N_INIT = 3
GMM_REG_COVAR = 1e-5

# GMM on CLR/log without dimensionality reduction can be unstable.
GMM_ALLOWED_PCA = [10, 15, 20, 25]

# Consensus sizes. Candidate labels are aligned without CPC labels.
CONSENSUS_SIZES = [3, 5, 7, 10, 15, 20, 30]

STRONG_TARGET_NMI = 0.315
TEST_RECOMMEND_NMI = 0.312
BORDERLINE_TARGET_NMI = 0.310

MIN_FINAL_PUR_P = 0.369
MIN_FINAL_PUR_A = 0.420

print("=" * 130)
print("DEPTH-OT V2 — LARGE-SCALE THETA-SPACE CLUSTERING SEARCH")
print("=" * 130)
print(f"Theta              : {THETA_PATH}")
print(f"Records            : {RECORDS_PATH}")
print(f"Output             : {OUTPUT_ROOT}")
print(f"Representations    : {len(REPRESENTATIONS)}")
print(f"PCA configurations : {PCA_DIMENSIONS}")
print(f"KMeans modes       : {KMEANS_MODES}")
print(f"KMeans seeds       : {KMEANS_SEEDS}")
print(f"GMM enabled        : {RUN_GMM}")
print(f"Clusters           : {NUMBER_OF_TOPICS}")
print("Split              : DEV ONLY")
print("=" * 130)

if not THETA_PATH.exists():
    raise FileNotFoundError(f"Input theta not found:\n{THETA_PATH}")

if not RECORDS_PATH.exists():
    raise FileNotFoundError(f"DEV records not found:\n{RECORDS_PATH}")

# --------------------------------------------------------------------------------------------------
# 3. LOAD AND VALIDATE THETA
# --------------------------------------------------------------------------------------------------

theta = np.load(THETA_PATH).astype(np.float64)

if theta.shape != (EXPECTED_PATENTS, NUMBER_OF_TOPICS):
    raise RuntimeError(
        f"Theta shape mismatch: expected "
        f"({EXPECTED_PATENTS}, {NUMBER_OF_TOPICS}), got {theta.shape}"
    )

if not np.isfinite(theta).all():
    raise RuntimeError("Theta contains non-finite values.")

if (theta < -1e-8).any():
    raise RuntimeError(f"Theta contains negative values. Minimum={theta.min()}")

theta = np.clip(theta, 0.0, None)
theta /= np.clip(theta.sum(axis=1, keepdims=True), EPS, None)

print("\n[THETA]")
print(f"Shape          : {theta.shape}")
print(f"Minimum        : {theta.min():.10e}")
print(f"Maximum        : {theta.max():.10e}")
print(f"Row-sum minimum: {theta.sum(axis=1).min():.12f}")
print(f"Row-sum maximum: {theta.sum(axis=1).max():.12f}")

# --------------------------------------------------------------------------------------------------
# 4. LOAD CPC LABELS
# --------------------------------------------------------------------------------------------------

with open(RECORDS_PATH, "rb") as file:
    records = pickle.load(file)

if isinstance(records, dict):
    if "records" in records and isinstance(records["records"], (list, tuple)):
        records = records["records"]
    elif "data" in records and isinstance(records["data"], (list, tuple)):
        records = records["data"]
    else:
        records = list(records.values())

records = list(records)

if len(records) != EXPECTED_PATENTS:
    raise RuntimeError(
        f"Record count mismatch: expected {EXPECTED_PATENTS}, got {len(records)}"
    )


def to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def clean_label(value):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]

    value = str(value).strip()

    if not value or value.lower() in {"none", "nan", "null"}:
        return None

    return value


labels = {
    "section": [],
    "class": [],
    "subclass": [],
}

patent_ids = []

for index, record in enumerate(records):
    record = to_dict(record)

    patent_ids.append(str(record.get("patent_id", index)))

    section = clean_label(record.get("section"))
    class_label = clean_label(record.get("class"))
    subclass = clean_label(record.get("subclass"))

    cpc_codes = record.get("cpc_codes")

    if isinstance(cpc_codes, np.ndarray):
        cpc_codes = cpc_codes.tolist()

    if isinstance(cpc_codes, (list, tuple)) and cpc_codes:
        first_code = str(cpc_codes[0]).strip()

        if section is None and len(first_code) >= 1:
            section = first_code[:1]

        if class_label is None and len(first_code) >= 3:
            class_label = first_code[:3]

        if subclass is None and len(first_code) >= 4:
            subclass = first_code[:4]

    labels["section"].append(section)
    labels["class"].append(class_label)
    labels["subclass"].append(subclass)

for level in labels:
    labels[level] = np.asarray(labels[level], dtype=object)

    missing = np.asarray(
        [value is None for value in labels[level]],
        dtype=bool,
    )

    if missing.any():
        raise RuntimeError(
            f"{level} contains {int(missing.sum())} missing labels."
        )

print("\n[CPC LABEL COUNTS]")
for level in ("section", "class", "subclass"):
    print(f"{level:8s}: {len(np.unique(labels[level])):,}")

# --------------------------------------------------------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------------------------------------------------------

def cluster_purity(y_true, y_pred):
    correct = 0

    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        _, counts = np.unique(y_true[mask], return_counts=True)
        correct += int(counts.max())

    return float(correct / len(y_true))


def assignment_purity(y_true, y_pred):
    correct = 0

    for label in np.unique(y_true):
        mask = y_true == label
        _, counts = np.unique(y_pred[mask], return_counts=True)
        correct += int(counts.max())

    return float(correct / len(y_true))


def evaluate_predictions(predictions):
    predictions = np.asarray(predictions, dtype=np.int64)

    result = {}
    pur_p_values = []
    pur_a_values = []
    nmi_values = []

    for level in ("section", "class", "subclass"):
        y_true = labels[level]

        pur_p = cluster_purity(y_true, predictions)
        pur_a = assignment_purity(y_true, predictions)
        nmi = normalized_mutual_info_score(
            y_true,
            predictions,
            average_method="arithmetic",
        )

        result[f"{level}_pur_p"] = float(pur_p)
        result[f"{level}_pur_a"] = float(pur_a)
        result[f"{level}_nmi"] = float(nmi)

        pur_p_values.append(pur_p)
        pur_a_values.append(pur_a)
        nmi_values.append(nmi)

    counts = np.bincount(
        predictions,
        minlength=NUMBER_OF_TOPICS,
    )

    result["mean_pur_p"] = float(np.mean(pur_p_values))
    result["mean_pur_a"] = float(np.mean(pur_a_values))
    result["mean_nmi"] = float(np.mean(nmi_values))
    result["active_topics"] = int((counts > 0).sum())
    result["max_topic_share"] = float(counts.max() / counts.sum())

    return result


baseline_predictions = np.argmax(theta, axis=1)
baseline_metrics = evaluate_predictions(baseline_predictions)

print("\n[INPUT VALIDATION]")
print(f"Mean Pur_p  : {baseline_metrics['mean_pur_p']:.6f}")
print(f"Mean Pur_a  : {baseline_metrics['mean_pur_a']:.6f}")
print(f"Mean NMI    : {baseline_metrics['mean_nmi']:.6f}")
print(f"Section NMI : {baseline_metrics['section_nmi']:.6f}")
print(f"Class NMI   : {baseline_metrics['class_nmi']:.6f}")
print(f"Subclass NMI: {baseline_metrics['subclass_nmi']:.6f}")

if abs(baseline_metrics["mean_nmi"] - EXPECTED_INPUT_NMI) > VALIDATION_TOLERANCE:
    raise RuntimeError(
        "Input theta does not reproduce the expected Mean NMI."
    )

if abs(baseline_metrics["mean_pur_p"] - EXPECTED_INPUT_PUR_P) > VALIDATION_TOLERANCE:
    raise RuntimeError(
        "Input theta does not reproduce the expected Mean Pur_p."
    )

if abs(baseline_metrics["mean_pur_a"] - EXPECTED_INPUT_PUR_A) > VALIDATION_TOLERANCE:
    raise RuntimeError(
        "Input theta does not reproduce the expected Mean Pur_a."
    )

print("[PASS] Input metrics reproduced.")

# --------------------------------------------------------------------------------------------------
# 6. REPRESENTATION FUNCTIONS
# --------------------------------------------------------------------------------------------------

def create_representation(input_theta, specification):
    kind = specification["kind"]

    if kind == "power":
        q = float(specification["q"])
        features = np.power(np.clip(input_theta, EPS, None), q)
        features /= np.clip(features.sum(axis=1, keepdims=True), EPS, None)

    elif kind == "hellinger":
        # Since theta rows sum to one, sqrt(theta) has unit L2 norm.
        features = np.sqrt(np.clip(input_theta, 0.0, None))

    elif kind == "log":
        features = np.log(np.clip(input_theta, EPS, None))

    elif kind == "clr":
        log_theta = np.log(np.clip(input_theta, EPS, None))
        features = log_theta - log_theta.mean(axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown representation kind: {kind}")

    features = np.asarray(features, dtype=np.float64)

    if not np.isfinite(features).all():
        raise RuntimeError(
            f"Representation {specification['name']} contains non-finite values."
        )

    return features


def apply_pca(features, dimension, seed=42):
    if dimension is None:
        return features

    pca = PCA(
        n_components=int(dimension),
        whiten=False,
        svd_solver="full",
        random_state=seed,
    )

    return pca.fit_transform(features)


def prepare_mode(features, mode):
    if mode == "euclidean":
        return np.asarray(features, dtype=np.float64)

    if mode == "spherical":
        return normalize(
            features,
            norm="l2",
            axis=1,
            copy=True,
        ).astype(np.float64)

    raise ValueError(f"Unknown KMeans mode: {mode}")

# --------------------------------------------------------------------------------------------------
# 7. BROAD SEARCH
# --------------------------------------------------------------------------------------------------

rows = []
prediction_store = {}
failures = []

total_kmeans = (
    len(REPRESENTATIONS)
    * len(PCA_DIMENSIONS)
    * len(KMEANS_MODES)
    * len(KMEANS_SEEDS)
)

total_gmm = (
    len(REPRESENTATIONS)
    * len(GMM_ALLOWED_PCA)
    * len(GMM_COVARIANCE_TYPES)
    * len(GMM_SEEDS)
    if RUN_GMM
    else 0
)

print("\n[SEARCH PLAN]")
print(f"KMeans fits : {total_kmeans:,}")
print(f"GMM fits    : {total_gmm:,}")
print(f"Total fits  : {total_kmeans + total_gmm:,}")
print("This search may take a long time on a Colab CPU.")

start_time = time.time()
progress = tqdm(
    total=total_kmeans + total_gmm,
    desc="Theta-space DEV search",
)

for specification in REPRESENTATIONS:
    representation_name = specification["name"]

    try:
        base_features = create_representation(theta, specification)
    except Exception as exc:
        failures.append(
            {
                "stage": "representation",
                "representation": representation_name,
                "error": str(exc),
            }
        )
        continue

    for pca_dimension in PCA_DIMENSIONS:
        pca_name = "none" if pca_dimension is None else str(pca_dimension)

        try:
            reduced_features = apply_pca(
                base_features,
                pca_dimension,
                seed=42,
            )
        except Exception as exc:
            failures.append(
                {
                    "stage": "pca",
                    "representation": representation_name,
                    "pca": pca_name,
                    "error": str(exc),
                }
            )

            # Account for skipped KMeans fits.
            progress.update(len(KMEANS_MODES) * len(KMEANS_SEEDS))

            if RUN_GMM and pca_dimension in GMM_ALLOWED_PCA:
                progress.update(
                    len(GMM_COVARIANCE_TYPES) * len(GMM_SEEDS)
                )

            continue

        # ------------------------------------------------------------------------------------------
        # KMeans and spherical KMeans
        # ------------------------------------------------------------------------------------------

        for mode in KMEANS_MODES:
            clustering_features = prepare_mode(reduced_features, mode)

            for seed in KMEANS_SEEDS:
                variant = (
                    f"{representation_name}_pca{pca_name}_"
                    f"{mode}_kmeans_s{seed}"
                )

                try:
                    model = KMeans(
                        n_clusters=NUMBER_OF_TOPICS,
                        init="k-means++",
                        n_init=KMEANS_N_INIT,
                        max_iter=KMEANS_MAX_ITER,
                        tol=1e-4,
                        random_state=seed,
                        algorithm="lloyd",
                    )

                    predictions = model.fit_predict(
                        clustering_features
                    ).astype(np.int16)

                    metrics = evaluate_predictions(predictions)

                    row = {
                        "variant": variant,
                        "algorithm": "kmeans",
                        "representation": representation_name,
                        "representation_kind": specification["kind"],
                        "power_q": specification.get("q"),
                        "pca_dimension": pca_dimension,
                        "mode": mode,
                        "seed": int(seed),
                        "covariance_type": None,
                        "inertia": float(model.inertia_),
                        "lower_bound": None,
                        **metrics,
                    }

                    row["delta_mean_nmi"] = (
                        metrics["mean_nmi"]
                        - baseline_metrics["mean_nmi"]
                    )
                    row["delta_mean_pur_p"] = (
                        metrics["mean_pur_p"]
                        - baseline_metrics["mean_pur_p"]
                    )
                    row["delta_mean_pur_a"] = (
                        metrics["mean_pur_a"]
                        - baseline_metrics["mean_pur_a"]
                    )

                    rows.append(row)
                    prediction_store[variant] = predictions

                except Exception as exc:
                    failures.append(
                        {
                            "stage": "kmeans",
                            "variant": variant,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=2),
                        }
                    )

                progress.update(1)

        # ------------------------------------------------------------------------------------------
        # Gaussian Mixture
        # ------------------------------------------------------------------------------------------

        if RUN_GMM and pca_dimension in GMM_ALLOWED_PCA:
            gmm_features = np.asarray(reduced_features, dtype=np.float64)

            # Standardize only by global feature scale to avoid extreme log/CLR dimensions.
            feature_mean = gmm_features.mean(axis=0, keepdims=True)
            feature_std = gmm_features.std(axis=0, keepdims=True)
            feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
            gmm_features = (gmm_features - feature_mean) / feature_std

            for covariance_type in GMM_COVARIANCE_TYPES:
                for seed in GMM_SEEDS:
                    variant = (
                        f"{representation_name}_pca{pca_name}_"
                        f"gmm_{covariance_type}_s{seed}"
                    )

                    try:
                        model = GaussianMixture(
                            n_components=NUMBER_OF_TOPICS,
                            covariance_type=covariance_type,
                            tol=1e-3,
                            reg_covar=GMM_REG_COVAR,
                            max_iter=GMM_MAX_ITER,
                            n_init=GMM_N_INIT,
                            init_params="kmeans",
                            random_state=seed,
                        )

                        predictions = model.fit_predict(
                            gmm_features
                        ).astype(np.int16)

                        metrics = evaluate_predictions(predictions)

                        row = {
                            "variant": variant,
                            "algorithm": "gmm",
                            "representation": representation_name,
                            "representation_kind": specification["kind"],
                            "power_q": specification.get("q"),
                            "pca_dimension": pca_dimension,
                            "mode": "standardized",
                            "seed": int(seed),
                            "covariance_type": covariance_type,
                            "inertia": None,
                            "lower_bound": float(model.lower_bound_),
                            **metrics,
                        }

                        row["delta_mean_nmi"] = (
                            metrics["mean_nmi"]
                            - baseline_metrics["mean_nmi"]
                        )
                        row["delta_mean_pur_p"] = (
                            metrics["mean_pur_p"]
                            - baseline_metrics["mean_pur_p"]
                        )
                        row["delta_mean_pur_a"] = (
                            metrics["mean_pur_a"]
                            - baseline_metrics["mean_pur_a"]
                        )

                        rows.append(row)
                        prediction_store[variant] = predictions

                    except Exception as exc:
                        failures.append(
                            {
                                "stage": "gmm",
                                "variant": variant,
                                "error": str(exc),
                                "traceback": traceback.format_exc(limit=2),
                            }
                        )

                    progress.update(1)

progress.close()

if not rows:
    raise RuntimeError("All clustering runs failed.")

# --------------------------------------------------------------------------------------------------
# 8. LABEL-FREE CONSENSUS CLUSTERING
# --------------------------------------------------------------------------------------------------

def align_predictions_to_reference(reference, candidate, number_of_clusters):
    """
    Align cluster IDs using only overlap between two predicted assignments.
    CPC labels are not used.
    """
    contingency = np.zeros(
        (number_of_clusters, number_of_clusters),
        dtype=np.int64,
    )

    np.add.at(
        contingency,
        (candidate.astype(int), reference.astype(int)),
        1,
    )

    candidate_ids, reference_ids = linear_sum_assignment(
        -contingency
    )

    mapping = np.arange(number_of_clusters, dtype=np.int64)

    for candidate_id, reference_id in zip(candidate_ids, reference_ids):
        mapping[candidate_id] = reference_id

    return mapping[candidate.astype(int)]


def consensus_vote(prediction_list, number_of_clusters):
    reference = np.asarray(prediction_list[0], dtype=np.int64)
    aligned = [reference]

    for candidate in prediction_list[1:]:
        aligned_candidate = align_predictions_to_reference(
            reference,
            np.asarray(candidate, dtype=np.int64),
            number_of_clusters,
        )
        aligned.append(aligned_candidate)

    aligned_matrix = np.stack(aligned, axis=0)

    consensus = np.empty(
        aligned_matrix.shape[1],
        dtype=np.int16,
    )

    for patent_index in range(aligned_matrix.shape[1]):
        counts = np.bincount(
            aligned_matrix[:, patent_index],
            minlength=number_of_clusters,
        )
        consensus[patent_index] = int(np.argmax(counts))

    return consensus, aligned_matrix


broad_ranking = pd.DataFrame(rows).sort_values(
    by=[
        "mean_nmi",
        "mean_pur_p",
        "mean_pur_a",
        "active_topics",
    ],
    ascending=[False, False, False, False],
).reset_index(drop=True)

# Select the best seed for each structural configuration before consensus.
structural_columns = [
    "algorithm",
    "representation",
    "pca_dimension",
    "mode",
    "covariance_type",
]

diverse_candidates = (
    broad_ranking
    .sort_values(
        by=["mean_nmi", "mean_pur_p", "mean_pur_a"],
        ascending=[False, False, False],
    )
    .groupby(
        structural_columns,
        dropna=False,
        as_index=False,
        sort=False,
    )
    .first()
    .sort_values(
        by=["mean_nmi", "mean_pur_p", "mean_pur_a"],
        ascending=[False, False, False],
    )
    .reset_index(drop=True)
)

consensus_rows = []

print("\n[CONSENSUS SEARCH]")

for size in CONSENSUS_SIZES:
    actual_size = min(size, len(diverse_candidates))

    if actual_size < 2:
        continue

    member_variants = (
        diverse_candidates
        .head(actual_size)["variant"]
        .tolist()
    )

    member_predictions = [
        prediction_store[variant]
        for variant in member_variants
    ]

    consensus_predictions, aligned_matrix = consensus_vote(
        member_predictions,
        NUMBER_OF_TOPICS,
    )

    metrics = evaluate_predictions(consensus_predictions)

    agreement = np.mean(
        aligned_matrix
        == consensus_predictions[None, :]
    )

    variant = f"consensus_top{actual_size:02d}"

    row = {
        "variant": variant,
        "algorithm": "consensus",
        "representation": "multi",
        "representation_kind": "multi",
        "power_q": None,
        "pca_dimension": None,
        "mode": "label_free_vote",
        "seed": None,
        "covariance_type": None,
        "inertia": None,
        "lower_bound": None,
        "consensus_size": int(actual_size),
        "consensus_agreement": float(agreement),
        "consensus_members": member_variants,
        **metrics,
    }

    row["delta_mean_nmi"] = (
        metrics["mean_nmi"]
        - baseline_metrics["mean_nmi"]
    )
    row["delta_mean_pur_p"] = (
        metrics["mean_pur_p"]
        - baseline_metrics["mean_pur_p"]
    )
    row["delta_mean_pur_a"] = (
        metrics["mean_pur_a"]
        - baseline_metrics["mean_pur_a"]
    )

    consensus_rows.append(row)
    prediction_store[variant] = consensus_predictions

    print(
        f"{variant:20s} | "
        f"NMI={metrics['mean_nmi']:.6f} | "
        f"Pur_p={metrics['mean_pur_p']:.6f} | "
        f"Pur_a={metrics['mean_pur_a']:.6f} | "
        f"agreement={agreement:.4f}"
    )

# Add consensus rows to the complete ranking.
all_rows = rows + consensus_rows
ranking = pd.DataFrame(all_rows)

ranking = ranking.sort_values(
    by=[
        "mean_nmi",
        "mean_pur_p",
        "mean_pur_a",
        "active_topics",
        "max_topic_share",
    ],
    ascending=[False, False, False, False, True],
).reset_index(drop=True)

ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

best_row = ranking.iloc[0].to_dict()
best_variant = str(best_row["variant"])
best_predictions = prediction_store[best_variant]
best_nmi = float(best_row["mean_nmi"])
best_pur_p = float(best_row["mean_pur_p"])
best_pur_a = float(best_row["mean_pur_a"])

# --------------------------------------------------------------------------------------------------
# 9. ROBUST/BALANCED ALTERNATIVE
# --------------------------------------------------------------------------------------------------

balanced_candidates = ranking[
    (ranking["mean_pur_p"] >= MIN_FINAL_PUR_P)
    & (ranking["mean_pur_a"] >= MIN_FINAL_PUR_A)
].copy()

if len(balanced_candidates) > 0:
    balanced_row = balanced_candidates.iloc[0].to_dict()
    balanced_variant = str(balanced_row["variant"])
    balanced_predictions = prediction_store[balanced_variant]
else:
    balanced_row = None
    balanced_variant = None
    balanced_predictions = None

# --------------------------------------------------------------------------------------------------
# 10. DECISION
# --------------------------------------------------------------------------------------------------

purity_preserved = (
    best_pur_p >= MIN_FINAL_PUR_P
    and best_pur_a >= MIN_FINAL_PUR_A
)

if best_nmi >= STRONG_TARGET_NMI and purity_preserved:
    decision = "STRONG_CLUSTERING_CANDIDATE_LOCK_AND_RUN_ONE_TEST"
    run_test = True
elif best_nmi >= TEST_RECOMMEND_NMI and purity_preserved:
    decision = "CLUSTERING_CANDIDATE_LOCK_AND_RUN_ONE_TEST"
    run_test = True
elif best_nmi >= BORDERLINE_TARGET_NMI:
    decision = "BORDERLINE_CLUSTERING_CANDIDATE_REVIEW_BEFORE_TEST"
    run_test = False
else:
    decision = "CLUSTERING_TARGET_NOT_REACHED_PROCEED_TO_CHECKPOINT_ENSEMBLE"
    run_test = False

# --------------------------------------------------------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------------------------------------------------------

ranking_csv_path = OUTPUT_ROOT / "theta_space_clustering_dev_ranking.csv"
summary_json_path = OUTPUT_ROOT / "theta_space_clustering_dev_summary.json"
best_predictions_path = OUTPUT_ROOT / "best_theta_space_dev_predictions.npy"
best_assignments_path = OUTPUT_ROOT / "best_theta_space_dev_assignments.csv"
top_predictions_path = OUTPUT_ROOT / "top50_theta_space_predictions.npz"
failures_path = OUTPUT_ROOT / "theta_space_failures.json"

# Lists cannot be written cleanly to CSV.
ranking_for_csv = ranking.copy()

if "consensus_members" in ranking_for_csv.columns:
    ranking_for_csv["consensus_members"] = ranking_for_csv[
        "consensus_members"
    ].apply(
        lambda value: json.dumps(value)
        if isinstance(value, list)
        else None
    )

ranking_for_csv.to_csv(ranking_csv_path, index=False)

np.save(
    best_predictions_path,
    np.asarray(best_predictions, dtype=np.int16),
)

pd.DataFrame(
    {
        "patent_id": patent_ids,
        "predicted_topic": np.asarray(
            best_predictions,
            dtype=np.int16,
        ),
    }
).to_csv(best_assignments_path, index=False)

# Save the top 50 prediction arrays for later ensemble analysis.
top_prediction_payload = {}

for _, row in ranking.head(50).iterrows():
    variant = str(row["variant"])
    safe_name = (
        variant
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

    if variant in prediction_store:
        top_prediction_payload[safe_name] = np.asarray(
            prediction_store[variant],
            dtype=np.int16,
        )

np.savez_compressed(
    top_predictions_path,
    **top_prediction_payload,
)

if balanced_predictions is not None:
    balanced_predictions_path = (
        OUTPUT_ROOT
        / "best_balanced_theta_space_dev_predictions.npy"
    )

    np.save(
        balanced_predictions_path,
        np.asarray(balanced_predictions, dtype=np.int16),
    )
else:
    balanced_predictions_path = None

with open(failures_path, "w", encoding="utf-8") as file:
    json.dump(
        failures,
        file,
        ensure_ascii=False,
        indent=2,
    )


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    return str(value)


summary = {
    "experiment": "depth_ot_v2_theta_space_clustering_dev",
    "timestamp": datetime.now().isoformat(),
    "split": "DEV_ONLY",
    "test_accessed": False,
    "input_theta_path": str(THETA_PATH),
    "records_path": str(RECORDS_PATH),
    "output_root": str(OUTPUT_ROOT),
    "number_of_patents": EXPECTED_PATENTS,
    "number_of_topics": NUMBER_OF_TOPICS,
    "number_of_successful_candidates": int(len(ranking)),
    "number_of_failed_candidates": int(len(failures)),
    "input_metrics": json_safe(baseline_metrics),
    "best_raw_candidate": json_safe(best_row),
    "best_balanced_candidate": json_safe(balanced_row),
    "selection_thresholds": {
        "strong_target_nmi": STRONG_TARGET_NMI,
        "test_recommend_nmi": TEST_RECOMMEND_NMI,
        "borderline_target_nmi": BORDERLINE_TARGET_NMI,
        "minimum_final_pur_p": MIN_FINAL_PUR_P,
        "minimum_final_pur_a": MIN_FINAL_PUR_A,
    },
    "purity_preserved": bool(purity_preserved),
    "run_test_now": bool(run_test),
    "decision": decision,
    "output_files": {
        "ranking_csv": str(ranking_csv_path),
        "summary_json": str(summary_json_path),
        "best_predictions": str(best_predictions_path),
        "best_assignments": str(best_assignments_path),
        "best_balanced_predictions": (
            str(balanced_predictions_path)
            if balanced_predictions_path is not None
            else None
        ),
        "top50_predictions": str(top_predictions_path),
        "failures_json": str(failures_path),
    },
}

summary_json_path = OUTPUT_ROOT / "theta_space_clustering_dev_summary.json"

with open(summary_json_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=2,
    )

# Auto-save the Colab cell.
try:
    history = get_ipython().user_ns.get("In", [])

    if history:
        with open(CODE_SAVE_PATH, "w", encoding="utf-8") as file:
            file.write(history[-1])

        print(f"\n[CODE SAVED] {CODE_SAVE_PATH}")
except Exception as exc:
    print(f"\n[WARNING] Code auto-save failed: {exc}")

# --------------------------------------------------------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------------------------------------------------------

runtime_minutes = (time.time() - start_time) / 60.0

display_columns = [
    "rank",
    "variant",
    "algorithm",
    "representation",
    "pca_dimension",
    "mode",
    "seed",
    "section_nmi",
    "class_nmi",
    "subclass_nmi",
    "mean_pur_p",
    "mean_pur_a",
    "mean_nmi",
    "delta_mean_nmi",
    "active_topics",
    "max_topic_share",
]

available_display_columns = [
    column
    for column in display_columns
    if column in ranking.columns
]

display_frame = ranking[available_display_columns].head(30).copy()

for column in [
    "section_nmi",
    "class_nmi",
    "subclass_nmi",
    "mean_pur_p",
    "mean_pur_a",
    "mean_nmi",
    "delta_mean_nmi",
    "max_topic_share",
]:
    if column in display_frame.columns:
        display_frame[column] = display_frame[column].apply(
            lambda value: (
                f"{float(value):.6f}"
                if pd.notna(value)
                else ""
            )
        )

print("\n" + "=" * 170)
print("THETA-SPACE CLUSTERING — DEV CPC RANKING")
print("=" * 170)
print(display_frame.to_string(index=False))

print("\n" + "=" * 130)
print("FINAL DECISION")
print("=" * 130)
print(f"Input Mean NMI          : {baseline_metrics['mean_nmi']:.6f}")
print(f"Best variant            : {best_variant}")
print(f"Best algorithm          : {best_row.get('algorithm')}")
print(f"Best representation     : {best_row.get('representation')}")
print(f"Best PCA dimension      : {best_row.get('pca_dimension')}")
print(f"Best mode               : {best_row.get('mode')}")
print(f"Best seed               : {best_row.get('seed')}")
print(f"Best Mean Pur_p         : {best_pur_p:.6f}")
print(f"Best Mean Pur_a         : {best_pur_a:.6f}")
print(f"Best Mean NMI           : {best_nmi:.6f}")
print(f"Delta Mean NMI          : {best_row['delta_mean_nmi']:+.6f}")
print(f"Section NMI             : {best_row['section_nmi']:.6f}")
print(f"Class NMI               : {best_row['class_nmi']:.6f}")
print(f"Subclass NMI            : {best_row['subclass_nmi']:.6f}")
print(f"Active topics           : {int(best_row['active_topics'])}/{NUMBER_OF_TOPICS}")
print(f"Maximum topic share     : {100 * best_row['max_topic_share']:.2f}%")
print(f"Purity preserved        : {purity_preserved}")

if balanced_row is not None:
    print("\nBEST BALANCED CANDIDATE")
    print(f"Variant                 : {balanced_variant}")
    print(f"Mean Pur_p              : {balanced_row['mean_pur_p']:.6f}")
    print(f"Mean Pur_a              : {balanced_row['mean_pur_a']:.6f}")
    print(f"Mean NMI                : {balanced_row['mean_nmi']:.6f}")
    print(f"Section NMI             : {balanced_row['section_nmi']:.6f}")
    print(f"Class NMI               : {balanced_row['class_nmi']:.6f}")
    print(f"Subclass NMI            : {balanced_row['subclass_nmi']:.6f}")

print("\nDECISION")
print(f"Run TEST now            : {run_test}")
print(f"Decision                : {decision}")
print(f"Successful candidates   : {len(ranking):,}")
print(f"Failed candidates       : {len(failures):,}")
print(f"Runtime                  : {runtime_minutes:.2f} minutes")

print("\nOUTPUT FILES")
print(f"Ranking CSV             : {ranking_csv_path}")
print(f"Summary JSON            : {summary_json_path}")
print(f"Best predictions        : {best_predictions_path}")
print(f"Best assignments        : {best_assignments_path}")
print(f"Top-50 predictions      : {top_predictions_path}")
print(f"Failures JSON           : {failures_path}")
print(f"Saved code              : {CODE_SAVE_PATH}")

if decision == "STRONG_CLUSTERING_CANDIDATE_LOCK_AND_RUN_ONE_TEST":
    print("\n[STRONG PASS] Lock this configuration. Prepare exactly one TEST evaluation.")
elif decision == "CLUSTERING_CANDIDATE_LOCK_AND_RUN_ONE_TEST":
    print("\n[PASS] Lock this configuration. Prepare exactly one TEST evaluation.")
elif decision == "BORDERLINE_CLUSTERING_CANDIDATE_REVIEW_BEFORE_TEST":
    print("\n[BORDERLINE] Review purity and hierarchy-level NMI before TEST.")
else:
    print("\n[CONTINUE] Proceed to aligned multi-checkpoint ensemble. Do not run TEST yet.")

print("=" * 130)
print("[PASS] Large-scale DEV theta-space clustering search completed.")
