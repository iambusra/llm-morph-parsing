#!/usr/bin/env python3

import os
import json
import argparse

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ============================================================
# HELPERS
# ============================================================


def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# LOAD DATA
# ============================================================


def load_vectors_and_metadata(
    step2_dir,
    vector_type
):

    if vector_type == "endpoint":

        vector_path = os.path.join(
            step2_dir,
            "endpoint_vectors.npy"
        )

    elif vector_type == "ambiguity":

        vector_path = os.path.join(
            step2_dir,
            "ambiguity_vectors.npy"
        )

    else:

        raise ValueError(
            f"Unknown vector_type: {vector_type}"
        )

    metadata_path = os.path.join(
        step2_dir,
        "prefix_metadata.csv"
    )

    vectors = np.load(vector_path)

    metadata = pd.read_csv(
        metadata_path
    )

    return vectors, metadata


# ============================================================
# FILTER TRAIN / TEST CONDITIONS
# ============================================================


def build_train_mask(
    metadata,
    train_location,
    train_cue_type,
    target_type
):

    mask = np.ones(
        len(metadata),
        dtype=bool
    )

    # --------------------------------------------------------
    # Cue-before only training
    # --------------------------------------------------------

    mask &= (
        metadata["location"] == train_location
    )

    # --------------------------------------------------------
    # Cue type filtering
    # --------------------------------------------------------

    if train_cue_type != "both":

        mask &= (
            metadata["type"] == train_cue_type
        )

    # --------------------------------------------------------
    # Ambiguity vectors must exist
    # --------------------------------------------------------

    if target_type == "ambiguity":

        mask &= (
            metadata[
                "ambiguity_vector_available"
            ].astype(str)
            .str.lower()
            == "true"
        )

    return mask



def build_test_mask(
    metadata,
    test_location,
    test_cue_type,
    target_type
):

    mask = np.ones(
        len(metadata),
        dtype=bool
    )

    mask &= (
        metadata["location"] == test_location
    )

    if test_cue_type != "both":

        mask &= (
            metadata["type"] == test_cue_type
        )

    if target_type == "ambiguity":

        mask &= (
            metadata[
                "ambiguity_vector_available"
            ].astype(str)
            .str.lower()
            == "true"
        )

    return mask


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================


def grouped_split(
    metadata,
    train_mask,
    random_state=42,
    test_size=0.25
):

    subset = metadata[
        train_mask
    ].copy()

    groups = subset["row_id"].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    idx = np.arange(len(subset))

    train_idx, test_idx = next(
        splitter.split(
            idx,
            groups=groups
        )
    )

    train_rows = subset.iloc[
        train_idx
    ].copy()

    heldout_rows = subset.iloc[
        test_idx
    ].copy()

    return train_rows, heldout_rows


# ============================================================
# PROBE TRAINING
# ============================================================


def train_probe(
    X_train,
    y_train
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    probe = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    probe.fit(
        X_train_scaled,
        y_train
    )

    return scaler, probe


# ============================================================
# LABELS
# ============================================================


def encode_interpretation_labels(y):

    return np.array(
        [
            1 if x == "nominalizer" else 0
            for x in y
        ]
    )



def encode_cue_labels(y):

    return np.array(
        [
            1 if x == "semantic" else 0
            for x in y
        ]
    )


# ============================================================
# SANITY CHECKS
# ============================================================


def check_no_leakage(
    train_rows,
    test_rows
):

    train_ids = set(
        train_rows["row_id"]
    )

    test_ids = set(
        test_rows["row_id"]
    )

    overlap = train_ids.intersection(
        test_ids
    )

    if len(overlap) > 0:

        raise ValueError(
            f"Row leakage detected: {overlap}"
        )



def check_vector_health(X, split_name):

    if np.isnan(X).any():

        raise ValueError(
            f"NaNs detected in {split_name}"
        )

    if np.isinf(X).any():

        raise ValueError(
            f"Infs detected in {split_name}"
        )


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--step2-dir",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--vector-type",
        choices=[
            "endpoint",
            "ambiguity"
        ],
        required=True
    )

    parser.add_argument(
        "--probe-target",
        choices=[
            "interpretation",
            "cue_type"
        ],
        required=True
    )

    parser.add_argument(
        "--train-cue-type",
        choices=[
            "both",
            "syntactic",
            "semantic"
        ],
        default="both"
    )

    parser.add_argument(
        "--test-cue-type",
        choices=[
            "both",
            "syntactic",
            "semantic"
        ],
        default="both"
    )

    args = parser.parse_args()

    ensure_dir(
        args.output_dir
    )

    # ========================================================
    # LOAD
    # ========================================================

    vectors, metadata = load_vectors_and_metadata(
        step2_dir=args.step2_dir,
        vector_type=args.vector_type
    )

    # ========================================================
    # TRAIN ON CUE-BEFORE
    # TEST ON CUE-AFTER
    # ========================================================

    train_mask = build_train_mask(
        metadata=metadata,
        train_location="before",
        train_cue_type=args.train_cue_type,
        target_type=args.vector_type
    )

    test_mask = build_test_mask(
        metadata=metadata,
        test_location="after",
        test_cue_type=args.test_cue_type,
        target_type=args.vector_type
    )

    train_rows, heldout_rows = grouped_split(
        metadata,
        train_mask
    )

    test_rows = metadata[
        test_mask
    ].copy()

    check_no_leakage(
        train_rows,
        test_rows
    )

    # ========================================================
    # LABELS
    # ========================================================

    if args.probe_target == "interpretation":

        y_train = encode_interpretation_labels(
            train_rows["label"]
        )

        y_test = encode_interpretation_labels(
            test_rows["label"]
        )

    else:

        y_train = encode_cue_labels(
            train_rows["type"]
        )

        y_test = encode_cue_labels(
            test_rows["type"]
        )

    # ========================================================
    # INDEXING
    # ========================================================

    train_idx = train_rows[
        "global_prefix_id"
    ].values

    test_idx = test_rows[
        "global_prefix_id"
    ].values

    n_layers = vectors.shape[1]

    # ========================================================
    # OUTPUT CONTAINERS
    # ========================================================

    score_rows = []

    prediction_rows = []

    split_summary_rows = []

    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    split_summary_rows.append(
        {
            "train_cue_type": args.train_cue_type,
            "test_cue_type": args.test_cue_type,
            "vector_type": args.vector_type,
            "probe_target": args.probe_target,
            "n_train": int(len(train_rows)),
            "n_test": int(len(test_rows)),
            "n_train_sentences": int(
                train_rows["row_id"].nunique()
            ),
            "n_test_sentences": int(
                test_rows["row_id"].nunique()
            ),
        }
    )

    # ========================================================
    # LAYER LOOP
    # ========================================================

    for layer in range(n_layers):

        X_train = vectors[
            train_idx,
            layer,
            :
        ]

        X_test = vectors[
            test_idx,
            layer,
            :
        ]

        check_vector_health(
            X_train,
            f"train_layer_{layer}"
        )

        check_vector_health(
            X_test,
            f"test_layer_{layer}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        scaler, probe = train_probe(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        X_test_scaled = scaler.transform(
            X_test
        )

        probs = probe.predict_proba(
            X_test_scaled
        )

        preds = probe.predict(
            X_test_scaled
        )

        distances = probe.decision_function(
            X_test_scaled
        )

        bal_acc = balanced_accuracy_score(
            y_test,
            preds
        )

        # ----------------------------------------------------
        # COLLAPSE CHECK
        # ----------------------------------------------------

        prediction_entropy = float(
            -np.mean(
                probs[:, 0] * np.log(
                    probs[:, 0] + 1e-12
                )
                +
                probs[:, 1] * np.log(
                    probs[:, 1] + 1e-12
                )
            )
        )

        score_rows.append(
            {
                "layer": layer,
                "balanced_accuracy": bal_acc,
                "prediction_entropy": prediction_entropy,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "vector_type": args.vector_type,
                "probe_target": args.probe_target,
                "train_cue_type": args.train_cue_type,
                "test_cue_type": args.test_cue_type,
            }
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        for i in range(len(test_rows)):

            row = test_rows.iloc[i]

            prediction_rows.append(
                {
                    "global_prefix_id": int(
                        row["global_prefix_id"]
                    ),
                    "row_id": int(
                        row["row_id"]
                    ),
                    "layer": layer,
                    "stage": row["stage"],
                    "tracked_region": row[
                        "tracked_region"
                    ],
                    "label": row["label"],
                    "type": row["type"],
                    "location": row["location"],
                    "prefix_end": int(
                        row["prefix_end"]
                    ),
                    "p_class0": float(
                        probs[i, 0]
                    ),
                    "p_class1": float(
                        probs[i, 1]
                    ),
                    "predicted_class": int(
                        preds[i]
                    ),
                    "signed_distance": float(
                        distances[i]
                    ),
                    "vector_type": args.vector_type,
                    "probe_target": args.probe_target,
                    "train_cue_type": args.train_cue_type,
                    "test_cue_type": args.test_cue_type,
                }
            )

    # ========================================================
    # SAVE
    # ========================================================

    scores_df = pd.DataFrame(
        score_rows
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    split_summary_df = pd.DataFrame(
        split_summary_rows
    )

    scores_df.to_csv(
        os.path.join(
            args.output_dir,
            "probe_scores.csv"
        ),
        index=False
    )

    predictions_df.to_csv(
        os.path.join(
            args.output_dir,
            "probe_predictions.csv"
        ),
        index=False
    )

    split_summary_df.to_csv(
        os.path.join(
            args.output_dir,
            "split_summary.csv"
        ),
        index=False
    )

    summary = {
        "step2_dir": args.step2_dir,
        "vector_type": args.vector_type,
        "probe_target": args.probe_target,
        "train_cue_type": args.train_cue_type,
        "test_cue_type": args.test_cue_type,
        "n_layers": int(n_layers),
        "n_train": int(len(train_rows)),
        "n_test": int(len(test_rows)),
    }

    with open(
        os.path.join(
            args.output_dir,
            "summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nDONE\n")

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()