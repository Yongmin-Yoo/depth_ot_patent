#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone scikit-learn LDA training pipeline.

Features
--------
- Loads train/test count BoW matrices from scipy .npz files.
- Trains LDA independently for multiple random seeds.
- Saves each fitted model, test theta, hard topic assignments,
  top words, diagnostics, and a final summary.
- Reuses completed seed models/results when rerun.
- Does NOT average theta across random seeds because topic indices
  are not semantically aligned across independently trained models.

Example
-------
python run_lda.py \
    --train-bow data/processed/bow_train.npz \
    --test-bow data/processed/bow_test.npz \
    --vocab data/processed/vocab.pkl \
    --output-dir outputs/lda \
    --topics 30 \
    --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.decomposition import LatentDirichletAllocation


# ============================================================
# Atomic saving utilities
# ============================================================
def atomic_save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            obj,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def atomic_save_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "wb") as file:
        np.save(file, array)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def atomic_save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    joblib.dump(obj, temporary_path)
    os.replace(temporary_path, path)


# ============================================================
# Data loading
# ============================================================
def load_sparse_count_matrix(path: Path, name: str) -> sp.csr_matrix:
    if not path.is_file():
        raise FileNotFoundError(f"{name} file not found: {path}")

    print(f"Loading {name}: {path}")

    matrix = sp.load_npz(path).tocsr()
    matrix.sort_indices()

    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D matrix, got shape={matrix.shape}"
        )

    if matrix.data.size:
        if not np.isfinite(matrix.data).all():
            raise FloatingPointError(
                f"{name} contains NaN or Inf values."
            )

        if np.any(matrix.data < 0):
            raise ValueError(
                f"{name} contains negative values. "
                "LDA requires non-negative word counts."
            )

    return matrix


def normalize_vocabulary(vocabulary: Any) -> list[str]:
    if isinstance(vocabulary, np.ndarray):
        vocabulary = vocabulary.tolist()

    if isinstance(vocabulary, (list, tuple)):
        return [str(word) for word in vocabulary]

    if isinstance(vocabulary, dict):
        # {word_id: word}
        if all(
            isinstance(key, (int, np.integer))
            for key in vocabulary
        ):
            return [
                str(vocabulary[key])
                for key in sorted(vocabulary)
            ]

        # {word: word_id}
        if all(
            isinstance(value, (int, np.integer))
            for value in vocabulary.values()
        ):
            return [
                str(word)
                for word, _ in sorted(
                    vocabulary.items(),
                    key=lambda item: int(item[1]),
                )
            ]

        for key in [
            "vocab",
            "vocabulary",
            "words",
            "tokens",
            "id2word",
        ]:
            if key in vocabulary:
                return normalize_vocabulary(vocabulary[key])

    if hasattr(vocabulary, "token2id"):
        return [
            str(word)
            for word, _ in sorted(
                vocabulary.token2id.items(),
                key=lambda item: int(item[1]),
            )
        ]

    raise TypeError(
        "Unsupported vocabulary object type: "
        f"{type(vocabulary)}"
    )


def load_vocabulary(path: Path | None) -> list[str] | None:
    if path is None:
        return None

    if not path.is_file():
        raise FileNotFoundError(
            f"Vocabulary file not found: {path}"
        )

    print(f"Loading vocabulary: {path}")

    suffix = path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            vocabulary = pickle.load(file)

    elif suffix == ".npy":
        vocabulary = np.load(
            path,
            allow_pickle=True,
        )

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            vocabulary = json.load(file)

    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as file:
            vocabulary = [
                line.strip()
                for line in file
                if line.strip()
            ]

    else:
        raise ValueError(
            f"Unsupported vocabulary format: {suffix}"
        )

    return normalize_vocabulary(vocabulary)


# ============================================================
# Topic extraction
# ============================================================
def save_top_words(
    lda: LatentDirichletAllocation,
    vocabulary: list[str],
    path: Path,
    top_n: int,
) -> None:
    rows = []

    for topic_id, topic_weights in enumerate(lda.components_):
        top_indices = np.argsort(topic_weights)[::-1][:top_n]
        top_words = [vocabulary[index] for index in top_indices]

        rows.append({
            "topic_id": int(topic_id),
            "top_words": top_words,
        })

    atomic_save_json(rows, path)


# ============================================================
# Test inference
# ============================================================
def infer_theta_in_batches(
    lda: LatentDirichletAllocation,
    bow_test: sp.csr_matrix,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer test document-topic distributions in batches.

    Empty test documents receive an all-zero theta row rather than
    an arbitrary uniform distribution. Such documents should be
    excluded from clustering evaluation.
    """
    number_of_documents = bow_test.shape[0]
    number_of_topics = lda.n_components

    theta = np.zeros(
        (number_of_documents, number_of_topics),
        dtype=np.float32,
    )

    nonempty_mask = np.asarray(
        bow_test.getnnz(axis=1)
    ).ravel() > 0

    total_batches = (
        number_of_documents + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(0, number_of_documents, batch_size),
        start=1,
    ):
        end = min(
            start + batch_size,
            number_of_documents,
        )

        batch_nonempty = nonempty_mask[start:end]
        local_indices = np.flatnonzero(batch_nonempty)

        if len(local_indices):
            batch_matrix = bow_test[
                start:end
            ][local_indices].tocsr()

            batch_theta = lda.transform(batch_matrix)

            if not np.isfinite(batch_theta).all():
                raise FloatingPointError(
                    f"Non-finite theta detected in batch "
                    f"{start}:{end}"
                )

            theta[
                start + local_indices
            ] = batch_theta.astype(np.float32)

        if (
            batch_number == 1
            or batch_number % 10 == 0
            or end == number_of_documents
        ):
            print(
                f"  inference: {end:,}/"
                f"{number_of_documents:,}"
            )

    valid_theta = theta[nonempty_mask]

    if valid_theta.size:
        row_sums = valid_theta.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-5,
        ):
            maximum_error = float(
                np.max(np.abs(row_sums - 1.0))
            )

            raise ValueError(
                "Theta rows are not normalized. "
                f"Maximum error={maximum_error}"
            )

    return theta, nonempty_mask


# ============================================================
# Train or load one seed
# ============================================================
def run_seed(
    seed: int,
    bow_train: sp.csr_matrix,
    bow_test: sp.csr_matrix,
    vocabulary: list[str] | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_dir = output_dir / "models"
    theta_dir = output_dir / "theta"
    topic_dir = output_dir / "topics"
    diagnostic_dir = output_dir / "diagnostics"

    model_path = model_dir / f"lda_seed{seed}.joblib"
    theta_path = theta_dir / f"lda_theta_test_seed{seed}.npy"
    prediction_path = (
        theta_dir / f"lda_predicted_topic_test_seed{seed}.npy"
    )
    nonempty_mask_path = (
        theta_dir / f"lda_test_nonempty_mask_seed{seed}.npy"
    )
    topic_path = topic_dir / f"lda_top_words_seed{seed}.json"
    diagnostic_path = (
        diagnostic_dir / f"lda_diagnostics_seed{seed}.json"
    )

    training_seconds = 0.0
    inference_seconds = 0.0
    reused_model = False
    reused_theta = False

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if args.resume and model_path.is_file():
        print(f"[LOAD] Existing model: {model_path}")
        lda = joblib.load(model_path)
        reused_model = True

        if lda.n_components != args.topics:
            raise ValueError(
                f"Saved model has K={lda.n_components}, "
                f"but current K={args.topics}."
            )

        if lda.components_.shape[1] != bow_train.shape[1]:
            raise ValueError(
                "Saved model vocabulary dimension does not "
                "match the current BoW matrix."
            )

    else:
        lda = LatentDirichletAllocation(
            n_components=args.topics,
            learning_method="online",
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            learning_decay=args.learning_decay,
            learning_offset=args.learning_offset,
            random_state=seed,
            n_jobs=args.n_jobs,
            evaluate_every=args.evaluate_every,
            verbose=args.verbose,
        )

        print(f"[TRAIN] LDA seed={seed}")
        training_start = time.time()

        lda.fit(bow_train)

        training_seconds = time.time() - training_start

        if not np.isfinite(lda.components_).all():
            raise FloatingPointError(
                "The fitted LDA components contain NaN or Inf."
            )

        atomic_save_joblib(lda, model_path)

        print(
            f"[SAVE] Model: {model_path}\n"
            f"[DONE] Training: "
            f"{training_seconds / 60:.2f} minutes"
        )

    # --------------------------------------------------------
    # Test theta
    # --------------------------------------------------------
    if (
        args.resume
        and not args.force_inference
        and theta_path.is_file()
        and prediction_path.is_file()
        and nonempty_mask_path.is_file()
    ):
        print(f"[LOAD] Existing theta: {theta_path}")

        theta_test = np.load(theta_path)
        predicted_topics = np.load(prediction_path)
        nonempty_mask = np.load(nonempty_mask_path)

        reused_theta = True

    else:
        print(f"[INFERENCE] LDA seed={seed}")
        inference_start = time.time()

        theta_test, nonempty_mask = infer_theta_in_batches(
            lda=lda,
            bow_test=bow_test,
            batch_size=args.inference_batch_size,
        )

        inference_seconds = time.time() - inference_start

        predicted_topics = np.full(
            bow_test.shape[0],
            fill_value=-1,
            dtype=np.int32,
        )

        predicted_topics[nonempty_mask] = theta_test[
            nonempty_mask
        ].argmax(axis=1).astype(np.int32)

        atomic_save_npy(theta_test, theta_path)
        atomic_save_npy(predicted_topics, prediction_path)
        atomic_save_npy(nonempty_mask, nonempty_mask_path)

        print(
            f"[SAVE] Theta: {theta_path}\n"
            f"[DONE] Inference: "
            f"{inference_seconds / 60:.2f} minutes"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    expected_shape = (
        bow_test.shape[0],
        args.topics,
    )

    if theta_test.shape != expected_shape:
        raise ValueError(
            f"Unexpected theta shape: {theta_test.shape}, "
            f"expected={expected_shape}"
        )

    if not np.isfinite(theta_test).all():
        raise FloatingPointError(
            "Saved theta contains NaN or Inf."
        )

    used_topics = int(
        np.unique(
            predicted_topics[predicted_topics >= 0]
        ).size
    )

    # lda.bound_ is the final training perplexity estimate
    training_perplexity = float(lda.bound_)

    diagnostics = {
        "seed": int(seed),
        "status": "success",
        "n_topics": int(args.topics),
        "train_documents": int(bow_train.shape[0]),
        "test_documents": int(bow_test.shape[0]),
        "vocabulary_size": int(bow_train.shape[1]),
        "nonempty_test_documents": int(nonempty_mask.sum()),
        "empty_test_documents": int((~nonempty_mask).sum()),
        "used_hard_assignment_topics": used_topics,
        "n_iter_completed": int(lda.n_iter_),
        "n_batch_iterations": int(lda.n_batch_iter_),
        "training_perplexity": training_perplexity,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "reused_model": bool(reused_model),
        "reused_theta": bool(reused_theta),
        "model_path": str(model_path),
        "theta_path": str(theta_path),
        "prediction_path": str(prediction_path),
    }

    atomic_save_json(diagnostics, diagnostic_path)

    if vocabulary is not None:
        save_top_words(
            lda=lda,
            vocabulary=vocabulary,
            path=topic_path,
            top_n=args.top_words,
        )

    print("\n=== Seed diagnostics ===")
    print(f"Seed                   : {seed}")
    print(f"Completed iterations   : {lda.n_iter_}")
    print(f"Training perplexity    : {training_perplexity:.4f}")
    print(f"Used topics            : {used_topics}/{args.topics}")
    print(f"Empty test documents   : {(~nonempty_mask).sum():,}")
    print(f"Theta shape            : {theta_test.shape}")

    del theta_test
    del predicted_topics
    del nonempty_mask
    del lda
    gc.collect()

    return diagnostics


# ============================================================
# Arguments: Colab + command-line compatible
# ============================================================
def parse_arguments() -> argparse.Namespace:
    default_root = Path(
        "/content/drive/MyDrive/depth_ot_patent"
    )

    parser = argparse.ArgumentParser(
        description="Train multi-seed online LDA."
    )

    parser.add_argument(
        "--train-bow",
        type=Path,
        default=(
            default_root
            / "data"
            / "processed"
            / "bow_train.npz"
        ),
        help="Path to bow_train.npz",
    )

    parser.add_argument(
        "--test-bow",
        type=Path,
        default=(
            default_root
            / "data"
            / "processed"
            / "bow_test.npz"
        ),
        help="Path to bow_test.npz",
    )

    parser.add_argument(
        "--vocab",
        type=Path,
        default=(
            default_root
            / "data"
            / "processed"
            / "vocab.pkl"
        ),
        help="Vocabulary path",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            default_root
            / "topic_model"
            / "lda"
        ),
        help="Output directory",
    )

    parser.add_argument(
        "--topics",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning-decay",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--learning-offset",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--top-words",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--force-inference",
        action="store_true",
    )

    # Colab/Jupyter에서는 자동으로 전달되는 -f kernel.json을 무시
    if "ipykernel" in sys.modules:
        return parser.parse_args(args=[])

    # 일반 .py 실행에서는 command-line 인자 사용
    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.topics <= 1:
        raise ValueError("--topics must be greater than 1.")

    if not 0.5 < args.learning_decay <= 1.0:
        raise ValueError(
            "--learning-decay must be in (0.5, 1.0]."
        )

    bow_train = load_sparse_count_matrix(
        args.train_bow,
        "train BoW",
    )
    bow_test = load_sparse_count_matrix(
        args.test_bow,
        "test BoW",
    )

    if bow_train.shape[1] != bow_test.shape[1]:
        raise ValueError(
            "Train and test vocabulary dimensions differ: "
            f"train={bow_train.shape[1]}, "
            f"test={bow_test.shape[1]}"
        )

    vocabulary = load_vocabulary(args.vocab)

    if (
        vocabulary is not None
        and len(vocabulary) != bow_train.shape[1]
    ):
        raise ValueError(
            "Vocabulary size does not match the BoW matrix: "
            f"vocabulary={len(vocabulary)}, "
            f"features={bow_train.shape[1]}"
        )

    # Empty train documents provide no information.
    train_nonempty_mask = np.asarray(
        bow_train.getnnz(axis=1)
    ).ravel() > 0

    number_of_empty_train_documents = int(
        (~train_nonempty_mask).sum()
    )

    if number_of_empty_train_documents:
        print(
            f"[INFO] Removing "
            f"{number_of_empty_train_documents:,} "
            "empty train documents."
        )

        bow_train_fit = bow_train[
            train_nonempty_mask
        ].tocsr()
    else:
        bow_train_fit = bow_train

    print("\n=== LDA configuration ===")
    print(f"Train shape       : {bow_train_fit.shape}")
    print(f"Test shape        : {bow_test.shape}")
    print(f"Topics            : {args.topics}")
    print(f"Seeds             : {args.seeds}")
    print(f"Max iterations    : {args.max_iter}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Learning decay    : {args.learning_decay}")
    print(f"Learning offset   : {args.learning_offset}")
    print(f"Evaluate every    : {args.evaluate_every}")
    print(f"CPU jobs          : {args.n_jobs}")
    print(f"Output directory  : {args.output_dir}")

    records = []
    successful_seeds = []
    failed_seeds = []

    for seed in args.seeds:
        print("\n" + "=" * 70)
        print(f"LDA SEED {seed}")
        print("=" * 70)

        try:
            diagnostics = run_seed(
                seed=seed,
                bow_train=bow_train_fit,
                bow_test=bow_test,
                vocabulary=vocabulary,
                output_dir=args.output_dir,
                args=args,
            )

            records.append(diagnostics)
            successful_seeds.append(seed)

        except Exception as error:
            error_traceback = traceback.format_exc()

            print(f"[FAILED] seed={seed}: {error}")
            print(error_traceback)

            records.append({
                "seed": int(seed),
                "status": "failed",
                "error": str(error),
                "traceback": error_traceback,
            })

            failed_seeds.append(seed)

    summary = {
        "created_at": datetime.now().isoformat(),
        "status": (
            "success"
            if not failed_seeds
            else "partial_failure"
        ),
        "configuration": {
            "topics": int(args.topics),
            "seeds": [int(seed) for seed in args.seeds],
            "max_iter": int(args.max_iter),
            "batch_size": int(args.batch_size),
            "learning_method": "online",
            "learning_decay": float(args.learning_decay),
            "learning_offset": float(args.learning_offset),
            "evaluate_every": int(args.evaluate_every),
            "n_jobs": int(args.n_jobs),
        },
        "data": {
            "train_bow": str(args.train_bow),
            "test_bow": str(args.test_bow),
            "vocab": (
                str(args.vocab)
                if args.vocab is not None
                else None
            ),
            "train_shape_before_empty_filter": list(
                bow_train.shape
            ),
            "train_shape_used": list(
                bow_train_fit.shape
            ),
            "test_shape": list(bow_test.shape),
            "empty_train_documents_removed":
                number_of_empty_train_documents,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "successful_seeds": successful_seeds,
        "failed_seeds": failed_seeds,
        "runs": records,
        "methodological_note": (
            "Document-topic distributions were not averaged "
            "across seeds because independently fitted LDA "
            "topics are not index-aligned. Evaluate each seed "
            "separately and aggregate scalar metrics using "
            "mean and standard deviation."
        ),
    }

    summary_path = args.output_dir / "lda_summary.json"
    atomic_save_json(summary, summary_path)

    print("\n" + "=" * 70)
    print("LDA RUN FINISHED")
    print("=" * 70)
    print(f"Successful seeds: {successful_seeds}")
    print(f"Failed seeds    : {failed_seeds}")
    print(f"Summary         : {summary_path}")

    if failed_seeds:
        raise RuntimeError(
            f"Some LDA seeds failed: {failed_seeds}"
        )


if __name__ == "__main__":
    main()
