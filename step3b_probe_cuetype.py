#!/usr/bin/env python3

import os
import json
import argparse
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ============================================================
# HELPERS
# ============================================================

def filter_training_rows(df, train_label_type):

    train_df = df.copy()

    # --------------------------------------------------------
    # FULL SENTENCE ONLY
    # --------------------------------------------------------

    train_df = train_df[
        train_df["stage"] == "full_sentence"
    ]

    # --------------------------------------------------------
    # INTERPRETATION FILTER
    # --------------------------------------------------------

    if train_label_type != "both":

        train_df = train_df[
            train_df["label"] == train_label_type
        ]

    return train_df


def get_label_array(df):

    # syntactic = 0
    # semantic = 1

    return np.where(
        df["type"] == "semantic",
        1,
        0
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
        "--train-label-type",
        choices=[
            "both",
            "negation",
            "nominalizer"
        ],
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================
    # LOAD
    # ========================================================

    vectors_path = os.path.join(
        args.step2_dir,
        "prefix_vectors.npy"
    )

    metadata_path = os.path.join(
        args.step2_dir,
        "prefix_metadata.csv"
    )

    summary_path = os.path.join(
        args.step2_dir,
        "summary.json"
    )

    vectors = np.load(vectors_path).astype(np.float32)

    meta = pd.read_csv(metadata_path)

    with open(summary_path) as f:
        summary = json.load(f)

    n_layers = summary["n_layers"]

    # ========================================================
    # TRAINING SUBSET
    # ========================================================

    train_df = filter_training_rows(
        meta,
        args.train_label_type
    )

    print("\nTraining subset size:")
    print(len(train_df))

    train_indices = train_df[
        "global_prefix_id"
    ].to_numpy()

    y_train_full = get_label_array(
        train_df
    )

    groups = train_df[
        "ambiguous_lemma"
    ].to_numpy()

    # ========================================================
    # OUTPUT CONTAINERS
    # ========================================================

    prob_rows = []

    dist_rows = []

    score_rows = []

    trained_probes = {}

    # ========================================================
    # LOOP OVER LAYERS
    # ========================================================

    for layer in tqdm(range(n_layers)):

        # ----------------------------------------------------
        # LAYER VECTORS
        # ----------------------------------------------------

        X_all = vectors[:, layer, :]

        X_train_full = X_all[
            train_indices
        ]

        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=42
        )

        train_split, test_split = next(

            gss.split(
                X_train_full,
                y_train_full,
                groups=groups
            )
        )

        X_train = X_train_full[
            train_split
        ]

        y_train = y_train_full[
            train_split
        ]

        X_test = X_train_full[
            test_split
        ]

        y_test = y_train_full[
            test_split
        ]

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        X_all_scaled = scaler.transform(
            X_all
        )

        # ----------------------------------------------------
        # PROBE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EVAL
        # ----------------------------------------------------

        y_pred = probe.predict(
            X_test_scaled
        )

        bal_acc = balanced_accuracy_score(
            y_test,
            y_pred
        )

        score_rows.append({

            "layer": layer,

            "balanced_accuracy":
                bal_acc,

            "n_train":
                len(train_split),

            "n_test":
                len(test_split),

            "train_label_type":
                args.train_label_type
        })

        # ----------------------------------------------------
        # STORE PROBE
        # ----------------------------------------------------

        trained_probes[layer] = {

            "probe": probe,

            "scaler": scaler
        }

        # ----------------------------------------------------
        # PROJECT ALL PREFIXES
        # ----------------------------------------------------

        probs = probe.predict_proba(
            X_all_scaled
        )

        distances = probe.decision_function(
            X_all_scaled
        )

        preds = probe.predict(
            X_all_scaled
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        for i in range(len(meta)):

            meta_row = meta.iloc[i]

            prob_rows.append({

                "global_prefix_id":
                    meta_row["global_prefix_id"],

                "layer":
                    layer,

                "p_syntactic":
                    probs[i][0],

                "p_semantic":
                    probs[i][1],

                "predicted_type":
                    preds[i],

                "gold_type":
                    meta_row["type"],

                "train_label_type":
                    args.train_label_type
            })

            dist_rows.append({

                "global_prefix_id":
                    meta_row["global_prefix_id"],

                "layer":
                    layer,

                "signed_distance":
                    distances[i],

                "gold_type":
                    meta_row["type"],

                "train_label_type":
                    args.train_label_type
            })

    # ========================================================
    # DATAFRAMES
    # ========================================================

    prob_df = pd.DataFrame(
        prob_rows
    )

    dist_df = pd.DataFrame(
        dist_rows
    )

    score_df = pd.DataFrame(
        score_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    prob_df.to_csv(

        os.path.join(
            args.output_dir,
            "cue_probe_probs.csv"
        ),

        index=False
    )

    dist_df.to_csv(

        os.path.join(
            args.output_dir,
            "cue_probe_distances.csv"
        ),

        index=False
    )

    score_df.to_csv(

        os.path.join(
            args.output_dir,
            "cue_probe_scores.csv"
        ),

        index=False
    )

    with open(

        os.path.join(
            args.output_dir,
            "cue_trained_probes.pkl"
        ),

        "wb"
    ) as f:

        pickle.dump(
            trained_probes,
            f
        )

    config = {

        "step2_dir":
            args.step2_dir,

        "task":
            "cue_type",

        "train_label_type":
            args.train_label_type,

        "n_layers":
            n_layers,

        "n_prefixes":
            len(meta),

        "training_subset_size":
            len(train_df)
    }

    with open(

        os.path.join(
            args.output_dir,
            "cue_probe_config.json"
        ),

        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    print("\nDONE")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()