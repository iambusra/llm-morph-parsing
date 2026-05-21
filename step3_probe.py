#!/usr/bin/env python3

"""
STEP 3 v2
---------

Official probe training pipeline.

Inputs:
    model_step2/

Outputs:
    model_step3/

Main changes vs old Step3:

1.
Repeated train/dev splits
(default: 5)

2.
Evaluation set is fixed.
Only train/dev changes.

3.
Store:
    split
    split_seed

4.
Save:
    train/dev/eval subsets

5.
Save:
    layer_summary.csv

6.
Extensive sanity checks.

This script should be the ONLY official Step3.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
)


# ==========================================================
# FILE HELPERS
# ==========================================================

def ensure_dir(path):
    """
    Create directory if needed.
    Safe to call repeatedly.
    """

    os.makedirs(
        path,
        exist_ok=True,
    )


def save_csv(df, path):

    df.to_csv(
        path,
        index=False,
    )


def save_json(obj, path):

    with open(
        path,
        "w",
        encoding="utf8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data(
    step2_dir,
    vector_type,
):
    """
    Load vectors + metadata.

    endpoint
        endpoint_vectors.npy

    ambiguity
        ambiguity_vectors.npy

    Returns:

        vectors:
            [prefix, layer, hidden]

        metadata:
            one row per prefix
    """

    vector_map = {

        "endpoint":
        "endpoint_vectors.npy",

        "ambiguity":
        "ambiguity_vectors.npy",

    }

    vectors = np.load(

        os.path.join(
            step2_dir,
            vector_map[
                vector_type
            ],
        )

    )

    metadata = pd.read_csv(

        os.path.join(
            step2_dir,
            "prefix_metadata.csv",
        )

    )

    if (
        "global_prefix_id"
        not in metadata.columns
    ):

        metadata = metadata.copy()

        metadata[
            "global_prefix_id"
        ] = np.arange(
            len(metadata)
        )

    return (
        vectors,
        metadata,
    )


# ==========================================================
# BOOLEAN NORMALIZATION
# ==========================================================

def normalize_bool_column(
    series
):

    return (

        series

        .astype(str)

        .str.lower()

        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )

    )


# ==========================================================
# LABELS
# ==========================================================

def encode_labels(
    values,
    target,
):

    values = list(
        values
    )

    if (
        target
        ==
        "interpretation"
    ):

        return np.array(

            [

                1
                if x
                ==
                "nominalizer"

                else 0

                for x

                in values

            ]

        )

    elif (
        target
        ==
        "cue_type"
    ):

        return np.array(

            [

                1
                if x
                ==
                "semantic"

                else 0

                for x

                in values

            ]

        )

    raise ValueError()


def check_binary_labels(y, name):
    unique = sorted(set(y.tolist()))

    if unique != [0, 1]:
        raise ValueError(
            f"{name} must contain both classes 0 and 1. Found: {unique}"
        )




def class_names(
    target
):

    if (
        target
        ==
        "interpretation"
    ):

        return {

            0:
            "negation",

            1:
            "nominalizer",

        }

    elif (
        target
        ==
        "cue_type"
    ):

        return {

            0:
            "syntactic",

            1:
            "semantic",

        }

    raise ValueError()


# ==========================================================
# SUBSET BUILDING
# ==========================================================

def build_train_candidates(
    metadata,
    vector_type,
    train_cue_type,
):
    """
    Training:

        BEFORE only
        FULL SENTENCE only

    This matches original design.
    """

    mask = np.ones(
        len(metadata),
        dtype=bool,
    )

    mask &= (
        metadata.location
        ==
        "before"
    )

    mask &= (
        metadata.stage
        ==
        "full_sentence"
    )

    if (
        train_cue_type
        !=
        "both"
    ):

        mask &= (

            metadata.type

            ==

            train_cue_type

        )

    if (
        vector_type
        ==
        "ambiguity"
    ):

        mask &= (

            normalize_bool_column(

                metadata[
                    "ambiguity_vector_available"
                ]

            )

        )

    return metadata[
        mask
    ].copy()


def build_eval_candidates(
    metadata,
    vector_type,
    test_cue_type,
    test_scope,
):
    """
    Evaluation:

        optionally:
            all checkpoints

        or:
            full sentence
    """

    mask = np.ones(
        len(metadata),
        dtype=bool,
    )

    if (
        test_cue_type
        !=
        "both"
    ):

        mask &= (
            metadata.type
            ==
            test_cue_type
        )

    if (
        test_scope
        ==
        "full_sentence"
    ):

        mask &= (
            metadata.stage
            ==
            "full_sentence"
        )

    elif (
        test_scope
        ==
        "all_checkpoints"
    ):

        pass

    else:

        raise ValueError()

    if (
        vector_type
        ==
        "ambiguity"
    ):

        mask &= (

            normalize_bool_column(

                metadata[
                    "ambiguity_vector_available"
                ]

            )

        )

    return metadata[
        mask
    ].copy()


# ==========================================================
# SPLITS
# ==========================================================

def stratified_grouped_train_split(
    train_candidates,
    target,
    test_size,
    seed,
):
    """
    Stratified + grouped split.

    Why:
    GroupShuffleSplit preserves row_id grouping,
    but it does NOT preserve label balance.

    That caused transfer probes to collapse.

    This function preserves:
    - group integrity by row_id
    - approximate label balance
    """

    if test_size != 0.25:
        raise ValueError(
            "This function currently assumes test_size=0.25, "
            "implemented as 4-fold StratifiedGroupKFold."
        )

    label_col = (
        "label"
        if target == "interpretation"
        else "type"
    )

    y = encode_labels(
        train_candidates[label_col],
        target,
    )

    groups = train_candidates["row_id"].values
    idx = np.arange(len(train_candidates))

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed,
    )

    splits = list(
        splitter.split(
            idx,
            y,
            groups,
        )
    )

    fold = seed % len(splits)

    train_idx, dev_idx = splits[fold]

    train_rows = train_candidates.iloc[train_idx].copy()
    dev_rows = train_candidates.iloc[dev_idx].copy()

    y_train = encode_labels(
        train_rows[label_col],
        target,
    )

    y_dev = encode_labels(
        dev_rows[label_col],
        target,
    )

    check_binary_labels(
        y_train,
        "train labels",
    )

    check_binary_labels(
        y_dev,
        "dev labels",
    )

    return train_rows, dev_rows


# ==========================================================
# PROBE
# ==========================================================

def train_probe(
    X,
    y,
):

    scaler = (
        StandardScaler()
    )

    X = scaler.fit_transform(
        X
    )

    probe = (

        LogisticRegression(

            max_iter=5000,

            class_weight="balanced",

            solver="liblinear",

            random_state=42,

        )

    )

    probe.fit(
        X,
        y,
    )

    return (
        scaler,
        probe,
    )


def predict_with_probe(
    scaler,
    probe,
    X,
):

    X = scaler.transform(
        X
    )

    return (

        probe.predict_proba(
            X
        ),

        probe.predict(
            X
        ),

        probe.decision_function(
            X
        ),

    )


# ==========================================================
# SANITY
# ==========================================================

def check_vectors(
    X,
    name,
):

    if np.isnan(X).any():

        raise ValueError(
            f"NaN in {name}"
        )

    if np.isinf(X).any():

        raise ValueError(
            f"Inf in {name}"
        )


def assert_no_leakage(
    train_rows,
    eval_rows,
):

    overlap = (

        set(
            train_rows.row_id
        )

        &

        set(
            eval_rows.row_id
        )

    )

    assert (

        len(
            overlap
        )
        ==
        0

    ), (
        f"Leakage detected: "
        f"{len(overlap)}"
    )

# ==========================================================
# AUDIT HELPERS
# ==========================================================

def count_subset(df, name):
    """
    Compact subset audit.

    This is saved into sanity_check.json.
    """

    out = {
        "name": name,
        "n_rows": int(len(df)),
        "n_sentences": int(df["row_id"].nunique()),
    }

    for col in [
        "stage",
        "location",
        "type",
        "label",
    ]:

        if col in df.columns:

            out[f"{col}_counts"] = (
                df[col]
                .astype(str)
                .value_counts()
                .to_dict()
            )

    return out


def add_probability_columns(
    out,
    probs,
    target,
):
    """
    Adds human-readable probability columns.

    interpretation:
        p_negation
        p_nominalizer

    cue_type:
        p_syntactic
        p_semantic
    """

    names = class_names(
        target
    )

    out[
        f"p_{names[0]}"
    ] = float(
        probs[0]
    )

    out[
        f"p_{names[1]}"
    ] = float(
        probs[1]
    )

    out["p_class0"] = float(
        probs[0]
    )

    out["p_class1"] = float(
        probs[1]
    )

    return out


def prediction_entropy(
    probs,
):

    eps = 1e-12

    return -np.sum(
        probs
        *
        np.log(
            probs
            +
            eps
        ),
        axis=1,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--step2-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--vector-type",
        choices=[
            "endpoint",
            "ambiguity",
        ],
        required=True,
    )

    parser.add_argument(
        "--probe-target",
        choices=[
            "interpretation",
            "cue_type",
        ],
        required=True,
    )

    parser.add_argument(
        "--train-cue-type",
        choices=[
            "both",
            "syntactic",
            "semantic",
        ],
        default="both",
    )

    parser.add_argument(
        "--test-cue-type",
        choices=[
            "both",
            "syntactic",
            "semantic",
        ],
        default="both",
    )

    parser.add_argument(
        "--test-scope",
        choices=[
            "all_checkpoints",
            "full_sentence",
        ],
        default="all_checkpoints",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=.25,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    ensure_dir(
        args.output_dir
    )

    # ------------------------------------------------------
    # Load vectors + metadata
    # ------------------------------------------------------

    vectors, metadata = load_data(
        step2_dir=args.step2_dir,
        vector_type=args.vector_type,
    )

    n_layers = int(
        vectors.shape[1]
    )

    label_names = class_names(
        args.probe_target
    )

    # ------------------------------------------------------
    # Build global candidate pools
    # ------------------------------------------------------

    train_candidates = build_train_candidates(
        metadata=metadata,
        vector_type=args.vector_type,
        train_cue_type=args.train_cue_type,
    )

    eval_candidates = build_eval_candidates(
        metadata=metadata,
        vector_type=args.vector_type,
        test_cue_type=args.test_cue_type,
        test_scope=args.test_scope,
    )

    if len(train_candidates) == 0:
        raise ValueError(
            "No train candidates."
        )

    if len(eval_candidates) == 0:
        raise ValueError(
            "No eval candidates."
        )

    score_rows = []
    prediction_rows = []
    sanity = {
        "step2_dir": args.step2_dir,
        "output_dir": args.output_dir,
        "vector_type": args.vector_type,
        "probe_target": args.probe_target,
        "train_cue_type": args.train_cue_type,
        "test_cue_type": args.test_cue_type,
        "test_scope": args.test_scope,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_splits": args.n_splits,
        "n_layers": n_layers,
        "candidate_audits": {
            "train_candidates": count_subset(
                train_candidates,
                "train_candidates",
            ),
            "eval_candidates": count_subset(
                eval_candidates,
                "eval_candidates",
            ),
        },
        "split_audits": [],
    }

    # ------------------------------------------------------
    # Repeated train/dev splits
    #
    # Evaluation set is fixed in definition,
    # but train row_ids are removed from eval each split
    # to prevent row_id leakage.
    # ------------------------------------------------------

    for split in range(
        args.n_splits
    ):

        split_seed = (
            args.random_state
            +
            split
        )

        print(
            f"\nSPLIT {split} seed={split_seed}"
        )


        train_rows, dev_rows = stratified_grouped_train_split(
        
            train_candidates=train_candidates,

            target=args.probe_target,

            test_size=args.test_size,

            seed=split_seed,

        )



        train_row_ids = set(
            train_rows["row_id"].tolist()
        )

        eval_rows = (
            eval_candidates[
                ~eval_candidates["row_id"].isin(
                    train_row_ids
                )
            ]
            .copy()
        )

        assert_no_leakage(
            train_rows,
            eval_rows,
        )

        # Save split-specific subsets.
        # These are essential for reproducibility/debugging.
        save_csv(
            train_rows,
            os.path.join(
                args.output_dir,
                f"split_{split}_train.csv",
            ),
        )

        save_csv(
            dev_rows,
            os.path.join(
                args.output_dir,
                f"split_{split}_dev.csv",
            ),
        )

        save_csv(
            eval_rows,
            os.path.join(
                args.output_dir,
                f"split_{split}_eval.csv",
            ),
        )

        sanity[
            "split_audits"
        ].append(
            {
                "split": split,
                "split_seed": split_seed,
                "train": count_subset(
                    train_rows,
                    "train",
                ),
                "dev": count_subset(
                    dev_rows,
                    "dev",
                ),
                "eval": count_subset(
                    eval_rows,
                    "eval",
                ),
            }
        )

        label_col = (
            "label"
            if args.probe_target
            ==
            "interpretation"
            else
            "type"
        )

        y_train = encode_labels(
            train_rows[label_col],
            args.probe_target,
        )

        y_dev = encode_labels(
            dev_rows[label_col],
            args.probe_target,
        )


        y_eval = encode_labels(
            eval_rows[label_col],
            args.probe_target,
        )

        if len(np.unique(y_eval)) < 2:

            print(
                f"\nSKIP split={split} "
                f"because eval contains one class"
            )

            continue

        train_idx = (
            train_rows[
                "global_prefix_id"
            ]
            .astype(int)
            .values
        )


        dev_idx = (
            dev_rows[
                "global_prefix_id"
            ]
            .astype(int)
            .values
        )

        eval_idx = (
            eval_rows[
                "global_prefix_id"
            ]
            .astype(int)
            .values
        )

        # --------------------------------------------------
        # Layer loop
        # --------------------------------------------------

        for layer in range(
            n_layers
        ):

            print(
                f"split={split} layer={layer}",
                end="\r",
            )

            X_train = vectors[
                train_idx,
                layer,
                :
            ]

            X_dev = vectors[
                dev_idx,
                layer,
                :
            ]

            X_eval = vectors[
                eval_idx,
                layer,
                :
            ]

            check_vectors(
                X_train,
                f"split {split} layer {layer} train",
            )

            check_vectors(
                X_dev,
                f"split {split} layer {layer} dev",
            )

            check_vectors(
                X_eval,
                f"split {split} layer {layer} eval",
            )

            scaler, probe = train_probe(
                X_train,
                y_train,
            )

            dev_probs, dev_preds, _ = predict_with_probe(
                scaler,
                probe,
                X_dev,
            )

            eval_probs, eval_preds, eval_dist = predict_with_probe(
                scaler,
                probe,
                X_eval,
            )

            dev_acc = balanced_accuracy_score(
                y_dev,
                dev_preds,
            )

            eval_acc = balanced_accuracy_score(
                y_eval,
                eval_preds,
            )

            pred_counts = (
                pd.Series(
                    eval_preds
                )
                .value_counts(
                    normalize=True,
                )
                .to_dict()
            )

            majority_frac = float(
                max(
                    pred_counts.values()
                )
            )

            cm = confusion_matrix(
                y_eval,
                eval_preds,
                labels=[
                    0,
                    1,
                ],
            )

            score_rows.append(
                {
                    "split": split,
                    "split_seed": split_seed,
                    "layer": layer,
                    "dev_balanced_accuracy": float(
                        dev_acc
                    ),
                    "eval_balanced_accuracy": float(
                        eval_acc
                    ),
                    "n_train": int(
                        len(train_rows)
                    ),
                    "n_dev": int(
                        len(dev_rows)
                    ),
                    "n_eval": int(
                        len(eval_rows)
                    ),
                    "n_train_sentences": int(
                        train_rows["row_id"].nunique()
                    ),
                    "n_dev_sentences": int(
                        dev_rows["row_id"].nunique()
                    ),
                    "n_eval_sentences": int(
                        eval_rows["row_id"].nunique()
                    ),
                    "vector_type": args.vector_type,
                    "probe_target": args.probe_target,
                    "train_cue_type": args.train_cue_type,
                    "test_cue_type": args.test_cue_type,
                    "test_scope": args.test_scope,
                    "majority_prediction_fraction": majority_frac,
                    "collapsed_95": bool(
                        majority_frac >= .95
                    ),
                    "confusion_00": int(
                        cm[0, 0]
                    ),
                    "confusion_01": int(
                        cm[0, 1]
                    ),
                    "confusion_10": int(
                        cm[1, 0]
                    ),
                    "confusion_11": int(
                        cm[1, 1]
                    ),
                }
            )

            entropies = prediction_entropy(
                eval_probs
            )

            for i, row in enumerate(
                eval_rows.itertuples()
            ):

                out = {
                    "split": split,
                    "split_seed": split_seed,
                    "global_prefix_id": int(
                        row.global_prefix_id
                    ),
                    "row_id": int(
                        row.row_id
                    ),
                    "layer": layer,
                    "stage": getattr(
                        row,
                        "stage",
                        "",
                    ),
                    "tracked_region": getattr(
                        row,
                        "tracked_region",
                        "",
                    ),
                    "label": getattr(
                        row,
                        "label",
                        "",
                    ),
                    "type": getattr(
                        row,
                        "type",
                        "",
                    ),
                    "location": getattr(
                        row,
                        "location",
                        "",
                    ),
                    "prefix_end": int(
                        getattr(
                            row,
                            "prefix_end",
                            -1,
                        )
                    ),
                    "is_full_sentence": getattr(
                        row,
                        "is_full_sentence",
                        "",
                    ),
                    "gold_class": int(
                        y_eval[i]
                    ),
                    "gold_label": label_names[
                        int(
                            y_eval[i]
                        )
                    ],
                    "predicted_class": int(
                        eval_preds[i]
                    ),
                    "predicted_label": label_names[
                        int(
                            eval_preds[i]
                        )
                    ],
                    "signed_distance": float(
                        eval_dist[i]
                    ),
                    "entropy": float(
                        entropies[i]
                    ),
                    "vector_type": args.vector_type,
                    "probe_target": args.probe_target,
                    "train_cue_type": args.train_cue_type,
                    "test_cue_type": args.test_cue_type,
                    "test_scope": args.test_scope,
                }

                out = add_probability_columns(
                    out,
                    eval_probs[i],
                    args.probe_target,
                )

                prediction_rows.append(
                    out
                )

    # ------------------------------------------------------
    # Save raw repeated-split outputs
    # ------------------------------------------------------

    scores_df = pd.DataFrame(
        score_rows
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    save_csv(
        scores_df,
        os.path.join(
            args.output_dir,
            "probe_scores.csv",
        ),
    )

    save_csv(
        predictions_df,
        os.path.join(
            args.output_dir,
            "probe_predictions.csv",
        ),
    )

    # ------------------------------------------------------
    # Layer summary across splits
    # ------------------------------------------------------

    layer_summary = (
        scores_df
        .groupby(
            "layer",
            as_index=False,
        )
        .agg(
            mean_dev_balanced_accuracy=(
                "dev_balanced_accuracy",
                "mean",
            ),
            sd_dev_balanced_accuracy=(
                "dev_balanced_accuracy",
                "std",
            ),
            mean_eval_balanced_accuracy=(
                "eval_balanced_accuracy",
                "mean",
            ),
            sd_eval_balanced_accuracy=(
                "eval_balanced_accuracy",
                "std",
            ),
            mean_majority_prediction_fraction=(
                "majority_prediction_fraction",
                "mean",
            ),
            n_splits=(
                "split",
                "nunique",
            ),
        )
    )

    save_csv(
        layer_summary,
        os.path.join(
            args.output_dir,
            "layer_summary.csv",
        ),
    )

    peak_row = (
        layer_summary
        .sort_values(
            "mean_eval_balanced_accuracy",
            ascending=False,
        )
        .iloc[0]
    )

    n_collapsed = int(
        scores_df["collapsed_95"].sum()
    )

    # ------------------------------------------------------
    # Final sanity checks
    # ------------------------------------------------------

    sanity[
        "output_checks"
    ] = {
        "probe_scores_rows": int(
            len(scores_df)
        ),
        "probe_predictions_rows": int(
            len(predictions_df)
        ),

        "expected_score_rows": int(
            scores_df["split"].nunique()
            *
            n_layers
        ),

        "n_collapsed_layers_95_total": n_collapsed,
        "peak_layer": int(
            peak_row["layer"]
        ),
        "peak_mean_eval_balanced_accuracy": float(
            peak_row[
                "mean_eval_balanced_accuracy"
            ]
        ),
    }

    expected_rows = (
        scores_df["split"]
        .nunique()
        *
        n_layers
    )

    if len(scores_df) != expected_rows:

        raise ValueError(
            "Unexpected number of score rows."
        )


    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = {
        "step2_dir": args.step2_dir,
        "output_dir": args.output_dir,
        "vector_type": args.vector_type,
        "probe_target": args.probe_target,
        "train_cue_type": args.train_cue_type,
        "test_cue_type": args.test_cue_type,
        "test_scope": args.test_scope,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_splits": args.n_splits,
        "n_layers": n_layers,
        "peak_layer": int(
            peak_row["layer"]
        ),
        "peak_mean_eval_balanced_accuracy": float(
            peak_row[
                "mean_eval_balanced_accuracy"
            ]
        ),
        "n_collapsed_layers_95_total": n_collapsed,
    }

    save_json(
        summary,
        os.path.join(
            args.output_dir,
            "summary.json",
        ),
    )

    save_json(
        sanity,
        os.path.join(
            args.output_dir,
            "sanity_check.json",
        ),
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()