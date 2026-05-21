#!/usr/bin/env python3

"""
STEP 3 DIAGNOSTIC PLOTTING SCRIPT
---------------------------------

Purpose:
    Diagnostic plotting for the official Step 3 probe outputs.

This script is intentionally conservative.

It does NOT:
    - invent stage names
    - silently filter rows
    - build one giant raw master dataframe
    - average raw rows directly across models
    - average raw layer numbers across models

It DOES:
    - read actual stage values from probe_predictions.csv
    - aggregate inside each model first
    - aggregate by split before averaging across splits
    - use relative layer for cross-model plots
    - save source CSVs for every plot
    - save counts CSVs for every plot
    - save exclusion logs for milestone filtering
    - save global audit CSVs

Expected input structure:

    qwen_step3/
        ambiguity_cuetype/
        ambiguity_interpretation/
        endpoint_cuetype/
        endpoint_interpretation/
        ...

    tgemma_step3/
        ...

    tllama_step3/
        ...

Each run folder should contain:
    probe_predictions.csv
    probe_scores.csv
    layer_summary.csv
    summary.json
    sanity_check.json

Main plot families:

1.
    Interpretation across layers
    endpoint only
    full_sentence only

2.
    Interpretation across layers
    ambiguity only
    clean milestones only

3.
    Cue type across layers
    endpoint only
    full_sentence only

4.
    Cue type across layers
    ambiguity only
    clean milestones only

5.
    Uncertainty across time
    interpretation only
    best layers only
    based on endpoint_interpretation and ambiguity_interpretation

Author note:
    This is a diagnostic script, not a publication plotting script.
"""

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score


# ==========================================================
# CONFIG
# ==========================================================

MODEL_RUNS = {
    "qwen": {
        "display_name": "Qwen2.5-14B-Instruct",
        "root": "qwen_step3",
    },
    "tgemma": {
        "display_name": "Turkish-Gemma-9b-T1",
        "root": "tgemma_step3",
    },
    "tllama": {
        "display_name": "Turkish-Llama-8b-Instruct-v0.1",
        "root": "tllama_step3",
    },
}


RUN_FOLDERS = {
    "endpoint_interpretation": {
        "folder": "endpoint_interpretation",
        "vector_type": "endpoint",
        "probe_target": "interpretation",
    },
    "ambiguity_interpretation": {
        "folder": "ambiguity_interpretation",
        "vector_type": "ambiguity",
        "probe_target": "interpretation",
    },
    "endpoint_cuetype": {
        "folder": "endpoint_cuetype",
        "vector_type": "endpoint",
        "probe_target": "cue_type",
    },
    "ambiguity_cuetype": {
        "folder": "ambiguity_cuetype",
        "vector_type": "ambiguity",
        "probe_target": "cue_type",
    },
}


OUTPUT_ROOT = Path("step3_diagnostic_plots")


# Minimum number of unique sentences required in each:
#
#     model × split × layer × condition
#
# If a condition falls below this threshold, it is excluded
# and logged.
MIN_SENTENCES_PER_SPLIT_LAYER_CONDITION = 10


# Actual stage values discovered from your Step 3 outputs:
#
#     ambiguity_onset
#     cue_onset
#     post_ambiguity_pre_cue
#     post_cue
#     full_sentence
#
# We DO NOT use "disambiguation_zone" because that is not
# an actual value in the data. Your intended disambiguation
# milestone corresponds to "cue_onset".
#
# Clean milestone rules:
#
# before:
#     ambiguity_onset + tracked_region == ambiguity
#     cue_onset       + tracked_region == cue
#     full_sentence   + tracked_region == post_cue_or_post_ambiguity
#
# after:
#     cue_onset       + tracked_region == cue
#     ambiguity_onset + tracked_region == ambiguity
#     full_sentence   + tracked_region == post_cue_or_post_ambiguity

MILESTONE_RULES = [
    {
        "location": "before",
        "stage": "ambiguity_onset",
        "tracked_region": "ambiguity",
        "milestone": "before_1_ambiguity_onset",
        "milestone_order": 1,
        "milestone_label": "Before: ambiguity onset",
    },
    {
        "location": "before",
        "stage": "cue_onset",
        "tracked_region": "cue",
        "milestone": "before_2_cue_onset",
        "milestone_order": 2,
        "milestone_label": "Before: cue onset",
    },
    {
        "location": "before",
        "stage": "full_sentence",
        "tracked_region": "post_cue_or_post_ambiguity",
        "milestone": "before_3_full_sentence",
        "milestone_order": 3,
        "milestone_label": "Before: full sentence",
    },
    {
        "location": "after",
        "stage": "cue_onset",
        "tracked_region": "cue",
        "milestone": "after_1_cue_onset",
        "milestone_order": 1,
        "milestone_label": "After: cue onset",
    },
    {
        "location": "after",
        "stage": "ambiguity_onset",
        "tracked_region": "ambiguity",
        "milestone": "after_2_ambiguity_onset",
        "milestone_order": 2,
        "milestone_label": "After: ambiguity onset",
    },
    {
        "location": "after",
        "stage": "full_sentence",
        "tracked_region": "post_cue_or_post_ambiguity",
        "milestone": "after_3_full_sentence",
        "milestone_order": 3,
        "milestone_label": "After: full sentence",
    },
]


# ==========================================================
# BASIC HELPERS
# ==========================================================

def ensure_dir(path):
    """
    Create a directory if it does not exist.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def save_csv(df, path):
    """
    Save a DataFrame as CSV with no index.
    """

    path = Path(path)
    ensure_dir(path.parent)

    df.to_csv(
        path,
        index=False,
    )


def save_json(obj, path):
    """
    Save a JSON file.
    """

    path = Path(path)
    ensure_dir(path.parent)

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


def require_columns(df, required_columns, context):
    """
    Fail clearly if a required column is missing.
    """

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {context}: {missing}"
        )


def safe_unique_string(df, col, context):
    """
    Return one unique string value from a column.
    Fail if there are zero or multiple values.
    """

    values = (
        df[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(values) != 1:
        raise ValueError(
            f"{context}: expected exactly one unique value in {col}, "
            f"but found {values}"
        )

    return values[0]


def read_json_if_exists(path):
    """
    Read JSON if present.
    Otherwise return an empty dict.
    """

    path = Path(path)

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf8",
    ) as f:
        return json.load(f)


def get_n_layers(run_dir, predictions_df):
    """
    Get number of layers from summary.json if possible.
    Fall back to max layer + 1.
    """

    summary_path = Path(run_dir) / "summary.json"
    summary = read_json_if_exists(summary_path)

    if "n_layers" in summary:
        return int(summary["n_layers"])

    return int(predictions_df["layer"].max()) + 1


def add_relative_layer(df, n_layers):
    """
    Add layer_relative = layer / (n_layers - 1).

    This is the safe cross-model x-axis.
    """

    out = df.copy()

    if n_layers <= 1:
        out["layer_relative"] = 0.0
    else:
        out["layer_relative"] = (
            out["layer"].astype(float)
            /
            float(n_layers - 1)
        )

    return out


def infer_probability_columns(df, probe_target):
    """
    Find target-specific probability columns.

    For interpretation:
        p_negation
        p_nominalizer

    For cue_type:
        p_syntactic
        p_semantic
    """

    if probe_target == "interpretation":
        class0_col = "p_negation"
        class1_col = "p_nominalizer"

    elif probe_target == "cue_type":
        class0_col = "p_syntactic"
        class1_col = "p_semantic"

    else:
        raise ValueError(
            f"Unknown probe_target: {probe_target}"
        )

    if class0_col not in df.columns:
        class0_col = "p_class0"

    if class1_col not in df.columns:
        class1_col = "p_class1"

    require_columns(
        df,
        [class0_col, class1_col],
        f"probability columns for {probe_target}",
    )

    return class0_col, class1_col


# ==========================================================
# DATA LOADING AND VALIDATION
# ==========================================================

def run_dir_for(model_key, run_key):
    """
    Return the path for one model/run combination.
    """

    model_root = MODEL_RUNS[model_key]["root"]
    run_folder = RUN_FOLDERS[run_key]["folder"]

    return Path(model_root) / run_folder


def load_predictions(model_key, run_key):
    """
    Load one probe_predictions.csv file and validate it.
    """

    run_dir = run_dir_for(
        model_key=model_key,
        run_key=run_key,
    )

    path = run_dir / "probe_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find prediction file: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "split",
        "split_seed",
        "global_prefix_id",
        "row_id",
        "layer",
        "stage",
        "tracked_region",
        "label",
        "type",
        "location",
        "gold_class",
        "gold_label",
        "predicted_class",
        "predicted_label",
        "signed_distance",
        "entropy",
        "vector_type",
        "probe_target",
        "train_cue_type",
        "test_cue_type",
        "test_scope",
        "p_class0",
        "p_class1",
    ]

    require_columns(
        df,
        required,
        str(path),
    )

    expected_vector_type = RUN_FOLDERS[run_key]["vector_type"]
    expected_probe_target = RUN_FOLDERS[run_key]["probe_target"]

    actual_vector_type = safe_unique_string(
        df,
        "vector_type",
        str(path),
    )

    actual_probe_target = safe_unique_string(
        df,
        "probe_target",
        str(path),
    )

    if actual_vector_type != expected_vector_type:
        raise ValueError(
            f"{path}: expected vector_type={expected_vector_type}, "
            f"but found {actual_vector_type}"
        )

    if actual_probe_target != expected_probe_target:
        raise ValueError(
            f"{path}: expected probe_target={expected_probe_target}, "
            f"but found {actual_probe_target}"
        )

    n_layers = get_n_layers(
        run_dir=run_dir,
        predictions_df=df,
    )

    df = df.copy()

    df["model"] = model_key
    df["model_display"] = MODEL_RUNS[model_key]["display_name"]
    df["run_key"] = run_key
    df["run_dir"] = str(run_dir)

    df = add_relative_layer(
        df,
        n_layers=n_layers,
    )

    return df, n_layers


# ==========================================================
# GLOBAL AUDITS
# ==========================================================

def make_discovered_runs_audit():
    """
    Save a table showing which expected run folders/files exist.
    """

    rows = []

    for model_key, model_info in MODEL_RUNS.items():

        for run_key, run_info in RUN_FOLDERS.items():

            run_dir = run_dir_for(
                model_key=model_key,
                run_key=run_key,
            )

            predictions_path = run_dir / "probe_predictions.csv"
            scores_path = run_dir / "probe_scores.csv"
            layer_summary_path = run_dir / "layer_summary.csv"
            summary_path = run_dir / "summary.json"
            sanity_path = run_dir / "sanity_check.json"

            rows.append(
                {
                    "model": model_key,
                    "model_display": model_info["display_name"],
                    "run_key": run_key,
                    "run_dir": str(run_dir),
                    "expected_vector_type": run_info["vector_type"],
                    "expected_probe_target": run_info["probe_target"],
                    "run_dir_exists": run_dir.exists(),
                    "probe_predictions_exists": predictions_path.exists(),
                    "probe_scores_exists": scores_path.exists(),
                    "layer_summary_exists": layer_summary_path.exists(),
                    "summary_json_exists": summary_path.exists(),
                    "sanity_check_json_exists": sanity_path.exists(),
                }
            )

    out = pd.DataFrame(rows)

    save_csv(
        out,
        OUTPUT_ROOT / "audits" / "discovered_runs.csv",
    )

    return out


def make_stage_audits():
    """
    Save global stage audits for all four diagnostic runs.
    """

    stage_rows = []
    location_stage_rows = []
    location_stage_region_rows = []

    for model_key in MODEL_RUNS:

        for run_key in RUN_FOLDERS:

            df, n_layers = load_predictions(
                model_key=model_key,
                run_key=run_key,
            )

            base = {
                "model": model_key,
                "model_display": MODEL_RUNS[model_key]["display_name"],
                "run_key": run_key,
                "vector_type": RUN_FOLDERS[run_key]["vector_type"],
                "probe_target": RUN_FOLDERS[run_key]["probe_target"],
                "n_layers": int(n_layers),
            }

            stage_counts = (
                df
                .groupby(
                    ["stage"],
                    dropna=False,
                )
                .agg(
                    n_rows=("row_id", "size"),
                    n_sentences=("row_id", "nunique"),
                    n_splits=("split", "nunique"),
                    n_layers=("layer", "nunique"),
                )
                .reset_index()
            )

            for row in stage_counts.to_dict("records"):
                stage_rows.append(
                    {
                        **base,
                        **row,
                    }
                )

            location_stage_counts = (
                df
                .groupby(
                    ["location", "stage"],
                    dropna=False,
                )
                .agg(
                    n_rows=("row_id", "size"),
                    n_sentences=("row_id", "nunique"),
                    n_splits=("split", "nunique"),
                    n_layers=("layer", "nunique"),
                )
                .reset_index()
            )

            for row in location_stage_counts.to_dict("records"):
                location_stage_rows.append(
                    {
                        **base,
                        **row,
                    }
                )

            location_stage_region_counts = (
                df
                .groupby(
                    [
                        "location",
                        "stage",
                        "tracked_region",
                    ],
                    dropna=False,
                )
                .agg(
                    n_rows=("row_id", "size"),
                    n_sentences=("row_id", "nunique"),
                    n_splits=("split", "nunique"),
                    n_layers=("layer", "nunique"),
                )
                .reset_index()
            )

            for row in location_stage_region_counts.to_dict("records"):
                location_stage_region_rows.append(
                    {
                        **base,
                        **row,
                    }
                )

    save_csv(
        pd.DataFrame(stage_rows),
        OUTPUT_ROOT / "audits" / "stage_counts_by_run.csv",
    )

    save_csv(
        pd.DataFrame(location_stage_rows),
        OUTPUT_ROOT / "audits" / "location_stage_counts_by_run.csv",
    )

    save_csv(
        pd.DataFrame(location_stage_region_rows),
        OUTPUT_ROOT / "audits" / "location_stage_tracked_region_counts_by_run.csv",
    )


# ==========================================================
# FILTERING
# ==========================================================

def filter_full_sentence(df):
    """
    Keep only full sentence rows.

    Save excluded rows elsewhere in the plot function.
    """

    keep_mask = (
        df["stage"].astype(str)
        ==
        "full_sentence"
    )

    kept = df[keep_mask].copy()
    excluded = df[~keep_mask].copy()

    excluded["exclusion_reason"] = (
        "not_full_sentence"
    )

    return kept, excluded


def add_clean_milestone_labels(df):
    """
    Apply explicit clean milestone rules.

    This does not silently filter. It creates:
        keep_milestone
        milestone
        milestone_order
        milestone_label
        milestone_rule

    Rows that do not match a clean milestone rule remain marked
    as keep_milestone == False.
    """

    out = df.copy()

    out["keep_milestone"] = False
    out["milestone"] = ""
    out["milestone_order"] = np.nan
    out["milestone_label"] = ""
    out["milestone_rule"] = ""

    for rule in MILESTONE_RULES:

        mask = (
            (
                out["location"].astype(str)
                ==
                rule["location"]
            )
            &
            (
                out["stage"].astype(str)
                ==
                rule["stage"]
            )
            &
            (
                out["tracked_region"].astype(str)
                ==
                rule["tracked_region"]
            )
        )

        out.loc[
            mask,
            "keep_milestone",
        ] = True

        out.loc[
            mask,
            "milestone",
        ] = rule["milestone"]

        out.loc[
            mask,
            "milestone_order",
        ] = rule["milestone_order"]

        out.loc[
            mask,
            "milestone_label",
        ] = rule["milestone_label"]

        out.loc[
            mask,
            "milestone_rule",
        ] = (
            f"location={rule['location']} | "
            f"stage={rule['stage']} | "
            f"tracked_region={rule['tracked_region']}"
        )

    kept = (
        out[
            out["keep_milestone"]
        ]
        .copy()
    )

    excluded = (
        out[
            ~out["keep_milestone"]
        ]
        .copy()
    )

    excluded["exclusion_reason"] = (
        "not_clean_milestone"
    )

    return kept, excluded


def apply_min_sentence_threshold(
    summary_by_split,
    condition_cols,
    min_sentences,
):
    """
    Exclude model × split × layer × condition cells with too few sentences.

    This is applied after split-level aggregation.

    Returns:
        kept_summary
        excluded_summary
    """

    require_columns(
        summary_by_split,
        ["n_sentences"],
        "apply_min_sentence_threshold",
    )

    keep_mask = (
        summary_by_split["n_sentences"]
        >=
        min_sentences
    )

    kept = summary_by_split[keep_mask].copy()
    excluded = summary_by_split[~keep_mask].copy()

    excluded["exclusion_reason"] = (
        f"n_sentences_below_{min_sentences}"
    )

    return kept, excluded


# ==========================================================
# METRIC AGGREGATION
# ==========================================================

def compute_binary_counts(group):
    """
    Compute class and prediction counts for one grouped cell.
    """

    gold = group["gold_class"].astype(int)
    pred = group["predicted_class"].astype(int)

    out = {
        "n_rows": int(len(group)),
        "n_sentences": int(group["row_id"].nunique()),
        "gold_class0_count": int((gold == 0).sum()),
        "gold_class1_count": int((gold == 1).sum()),
        "pred_class0_count": int((pred == 0).sum()),
        "pred_class1_count": int((pred == 1).sum()),
    }

    return out


def balanced_accuracy_or_nan(gold, pred):
    """
    Balanced accuracy is only meaningful if gold has both classes.

    If a grouped cell contains only one gold class, return NaN.
    """

    gold = np.asarray(gold).astype(int)
    pred = np.asarray(pred).astype(int)

    if len(np.unique(gold)) < 2:
        return np.nan

    return float(
        balanced_accuracy_score(
            gold,
            pred,
        )
    )


def aggregate_predictions_by_split(
    df,
    condition_cols,
):
    """
    Aggregate predictions within each model/run/split/layer/condition.

    This is the core anti-leakage plotting aggregation.

    It computes:
        - balanced accuracy
        - mean entropy
        - mean absolute signed distance
        - mean max probability
        - class counts
        - prediction counts

    It does NOT average raw rows across splits.
    """

    required = [
        "model",
        "model_display",
        "run_key",
        "split",
        "split_seed",
        "layer",
        "layer_relative",
        "gold_class",
        "predicted_class",
        "entropy",
        "signed_distance",
        "row_id",
        "probe_target",
        "vector_type",
    ]

    require_columns(
        df,
        required + list(condition_cols),
        "aggregate_predictions_by_split",
    )

    probe_target = safe_unique_string(
        df,
        "probe_target",
        "aggregate_predictions_by_split",
    )

    class0_col, class1_col = infer_probability_columns(
        df,
        probe_target=probe_target,
    )

    df = df.copy()

    df["max_probability"] = df[
        [
            class0_col,
            class1_col,
        ]
    ].max(axis=1)

    group_cols = [
        "model",
        "model_display",
        "run_key",
        "vector_type",
        "probe_target",
        "split",
        "split_seed",
        "layer",
        "layer_relative",
    ] + list(condition_cols)

    rows = []

    for keys, group in df.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        counts = compute_binary_counts(group)

        row.update(counts)

        row["balanced_accuracy"] = balanced_accuracy_or_nan(
            group["gold_class"],
            group["predicted_class"],
        )

        row["mean_entropy"] = float(
            group["entropy"].mean()
        )

        row["sd_entropy"] = float(
            group["entropy"].std()
        )

        row["mean_abs_signed_distance"] = float(
            group["signed_distance"]
            .abs()
            .mean()
        )

        row["mean_signed_distance"] = float(
            group["signed_distance"].mean()
        )

        row["mean_max_probability"] = float(
            group["max_probability"].mean()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def average_across_splits(
    summary_by_split,
    condition_cols,
):
    """
    Average split-level summaries within model/layer/condition.

    This is where repeated split estimates become one model-level
    diagnostic curve.

    It computes means and SDs across splits.
    """

    required = [
        "model",
        "model_display",
        "run_key",
        "vector_type",
        "probe_target",
        "layer",
        "layer_relative",
        "split",
        "balanced_accuracy",
        "mean_entropy",
        "mean_abs_signed_distance",
        "mean_max_probability",
        "n_rows",
        "n_sentences",
    ]

    require_columns(
        summary_by_split,
        required + list(condition_cols),
        "average_across_splits",
    )

    group_cols = [
        "model",
        "model_display",
        "run_key",
        "vector_type",
        "probe_target",
        "layer",
        "layer_relative",
    ] + list(condition_cols)

    out = (
        summary_by_split
        .groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            sd_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_entropy=(
                "mean_entropy",
                "mean",
            ),
            sd_entropy_across_splits=(
                "mean_entropy",
                "std",
            ),
            mean_abs_signed_distance=(
                "mean_abs_signed_distance",
                "mean",
            ),
            sd_abs_signed_distance_across_splits=(
                "mean_abs_signed_distance",
                "std",
            ),
            mean_max_probability=(
                "mean_max_probability",
                "mean",
            ),
            sd_max_probability_across_splits=(
                "mean_max_probability",
                "std",
            ),
            mean_n_rows=(
                "n_rows",
                "mean",
            ),
            min_n_rows=(
                "n_rows",
                "min",
            ),
            mean_n_sentences=(
                "n_sentences",
                "mean",
            ),
            min_n_sentences=(
                "n_sentences",
                "min",
            ),
            n_splits=(
                "split",
                "nunique",
            ),
        )
    )

    return out


# ==========================================================
# PLOTTING HELPERS
# ==========================================================

def add_chance_line():
    """
    Add 0.5 chance line for balanced accuracy plots.
    """

    plt.axhline(
        y=0.5,
        linestyle="--",
        linewidth=1,
    )


def save_current_plot(plot_dir, filename_stem):
    """
    Save the current matplotlib figure as PNG and PDF.
    """

    ensure_dir(plot_dir)

    png_path = Path(plot_dir) / f"{filename_stem}.png"
    pdf_path = Path(plot_dir) / f"{filename_stem}.pdf"

    plt.tight_layout()

    plt.savefig(
        png_path,
        dpi=200,
    )

    plt.savefig(
        pdf_path,
    )

    plt.close()


def plot_lines_by_model(
    summary,
    y_col,
    title,
    y_label,
    output_dir,
    filename_stem,
    chance_line=False,
):
    """
    Plot one line per model.

    X-axis:
        layer_relative

    This is used for endpoint full-sentence plots.
    """

    plt.figure(
        figsize=(10, 6),
    )

    for model_key, sub in summary.groupby("model", sort=False):

        sub = sub.sort_values(
            "layer_relative"
        )

        label = sub["model_display"].iloc[0]

        plt.plot(
            sub["layer_relative"],
            sub[y_col],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=label,
        )

    if chance_line:
        add_chance_line()

    plt.xlabel(
        "Relative layer"
    )

    plt.ylabel(
        y_label
    )

    plt.title(
        title
    )

    plt.legend()

    save_current_plot(
        output_dir,
        filename_stem,
    )


def plot_milestone_facets_by_model(
    summary,
    y_col,
    title,
    y_label,
    output_dir,
    filename_stem,
    chance_line=False,
):
    """
    Plot milestone curves.

    One figure with separate panels by model.
    Each panel has one line per milestone.

    This keeps the first diagnostic plots readable.
    """

    models = (
        summary["model"]
        .drop_duplicates()
        .tolist()
    )

    n_models = len(models)

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=1,
        figsize=(11, 4 * n_models),
        sharex=True,
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):

        model_df = (
            summary[
                summary["model"] == model_key
            ]
            .copy()
        )

        model_display = model_df["model_display"].iloc[0]

        for milestone, sub in model_df.groupby(
            "milestone",
            sort=False,
        ):

            sub = sub.sort_values(
                "layer_relative"
            )

            milestone_label = sub["milestone_label"].iloc[0]

            ax.plot(
                sub["layer_relative"],
                sub[y_col],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=milestone_label,
            )

        if chance_line:
            ax.axhline(
                y=0.5,
                linestyle="--",
                linewidth=1,
            )

        ax.set_title(
            model_display
        )

        ax.set_ylabel(
            y_label
        )

        ax.legend(
            fontsize=8,
        )

    axes[-1].set_xlabel(
        "Relative layer"
    )

    fig.suptitle(
        title,
        y=1.01,
    )

    ensure_dir(output_dir)

    plt.tight_layout()

    png_path = Path(output_dir) / f"{filename_stem}.png"
    pdf_path = Path(output_dir) / f"{filename_stem}.pdf"

    plt.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close()


def plot_uncertainty_milestones(
    summary,
    output_dir,
    filename_stem,
):
    """
    Plot uncertainty over clean milestones at selected best layers.

    X-axis:
        milestone order

    Y-axis:
        mean entropy

    One panel per model.

    This is intentionally simple and diagnostic.
    """

    models = (
        summary["model"]
        .drop_duplicates()
        .tolist()
    )

    n_models = len(models)

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=1,
        figsize=(11, 4 * n_models),
        sharex=False,
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):

        model_df = (
            summary[
                summary["model"] == model_key
            ]
            .copy()
        )

        model_display = model_df["model_display"].iloc[0]

        for location, sub_location in model_df.groupby(
            "location",
            sort=False,
        ):

            sub_location = sub_location.sort_values(
                "milestone_order"
            )

            ax.plot(
                sub_location["milestone_order"],
                sub_location["mean_entropy"],
                marker="o",
                linewidth=1.5,
                markersize=5,
                label=location,
            )

        ax.set_title(
            model_display
        )

        ax.set_ylabel(
            "Mean entropy"
        )

        ax.set_xticks(
            sorted(
                model_df["milestone_order"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        # Label with the most compact milestone name available.
        tick_labels = []
        for order in sorted(
            model_df["milestone_order"]
            .dropna()
            .unique()
            .tolist()
        ):

            labels = (
                model_df[
                    model_df["milestone_order"] == order
                ]["milestone_label"]
                .drop_duplicates()
                .tolist()
            )

            # These differ by before/after, so keep the number only
            # on the axis and use the legend for location.
            tick_labels.append(
                str(int(order))
            )

        ax.set_xticklabels(
            tick_labels
        )

        ax.legend(
            title="Location",
            fontsize=8,
        )

    axes[-1].set_xlabel(
        "Milestone order within location"
    )

    fig.suptitle(
        "Interpretation uncertainty across clean milestones at best layers",
        y=1.01,
    )

    ensure_dir(output_dir)

    plt.tight_layout()

    png_path = Path(output_dir) / f"{filename_stem}.png"
    pdf_path = Path(output_dir) / f"{filename_stem}.pdf"

    plt.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# PLOT PIPELINE FUNCTIONS
# ==========================================================

def build_endpoint_full_sentence_plot(
    run_key,
    plot_name,
):
    """
    Build endpoint full-sentence diagnostic plot.

    Used for:
        endpoint_interpretation
        endpoint_cuetype
    """

    plot_dir = OUTPUT_ROOT / plot_name

    all_model_summaries = []
    all_model_split_summaries = []
    all_exclusions = []

    for model_key in MODEL_RUNS:

        df, n_layers = load_predictions(
            model_key=model_key,
            run_key=run_key,
        )

        kept, excluded = filter_full_sentence(df)

        if len(excluded) > 0:
            all_exclusions.append(excluded)

        if len(kept) == 0:
            raise ValueError(
                f"No full_sentence rows for {model_key} {run_key}"
            )

        condition_cols = []

        split_summary = aggregate_predictions_by_split(
            kept,
            condition_cols=condition_cols,
        )

        split_summary_kept, split_summary_excluded = apply_min_sentence_threshold(
            split_summary,
            condition_cols=condition_cols,
            min_sentences=MIN_SENTENCES_PER_SPLIT_LAYER_CONDITION,
        )

        if len(split_summary_excluded) > 0:
            split_summary_excluded = split_summary_excluded.copy()
            split_summary_excluded["model"] = model_key
            split_summary_excluded["run_key"] = run_key
            all_exclusions.append(split_summary_excluded)

        model_summary = average_across_splits(
            split_summary_kept,
            condition_cols=condition_cols,
        )

        all_model_split_summaries.append(split_summary_kept)
        all_model_summaries.append(model_summary)

    source = pd.concat(
        all_model_summaries,
        ignore_index=True,
    )

    counts = pd.concat(
        all_model_split_summaries,
        ignore_index=True,
    )

    save_csv(
        source,
        plot_dir / "source_summary_across_splits.csv",
    )

    save_csv(
        counts,
        plot_dir / "counts_summary_by_split.csv",
    )

    if all_exclusions:
        exclusions = pd.concat(
            all_exclusions,
            ignore_index=True,
            sort=False,
        )
    else:
        exclusions = pd.DataFrame()

    save_csv(
        exclusions,
        plot_dir / "excluded_rows_or_cells.csv",
    )

    probe_target = RUN_FOLDERS[run_key]["probe_target"]

    if probe_target == "interpretation":
        title = (
            "Endpoint interpretation decoding across layers "
            "(full sentence only)"
        )
        filename = "endpoint_interpretation_full_sentence"
    else:
        title = (
            "Endpoint cue-type decoding across layers "
            "(full sentence only)"
        )
        filename = "endpoint_cuetype_full_sentence"

    plot_lines_by_model(
        summary=source,
        y_col="mean_balanced_accuracy",
        title=title,
        y_label="Mean balanced accuracy across splits",
        output_dir=plot_dir,
        filename_stem=filename,
        chance_line=True,
    )


def build_ambiguity_milestone_plot(
    run_key,
    plot_name,
):
    """
    Build ambiguity clean-milestone diagnostic plot.

    Used for:
        ambiguity_interpretation
        ambiguity_cuetype
    """

    plot_dir = OUTPUT_ROOT / plot_name

    all_model_summaries = []
    all_model_split_summaries = []
    all_raw_milestone_exclusions = []
    all_threshold_exclusions = []

    for model_key in MODEL_RUNS:

        df, n_layers = load_predictions(
            model_key=model_key,
            run_key=run_key,
        )

        kept, raw_excluded = add_clean_milestone_labels(df)

        if len(raw_excluded) > 0:
            raw_excluded = raw_excluded.copy()
            raw_excluded["model"] = model_key
            raw_excluded["run_key"] = run_key
            all_raw_milestone_exclusions.append(raw_excluded)

        if len(kept) == 0:
            raise ValueError(
                f"No clean milestone rows for {model_key} {run_key}"
            )

        condition_cols = [
            "location",
            "stage",
            "tracked_region",
            "milestone",
            "milestone_order",
            "milestone_label",
            "milestone_rule",
        ]

        split_summary = aggregate_predictions_by_split(
            kept,
            condition_cols=condition_cols,
        )

        split_summary_kept, split_summary_excluded = apply_min_sentence_threshold(
            split_summary,
            condition_cols=condition_cols,
            min_sentences=MIN_SENTENCES_PER_SPLIT_LAYER_CONDITION,
        )

        if len(split_summary_excluded) > 0:
            split_summary_excluded = split_summary_excluded.copy()
            split_summary_excluded["model"] = model_key
            split_summary_excluded["run_key"] = run_key
            all_threshold_exclusions.append(split_summary_excluded)

        model_summary = average_across_splits(
            split_summary_kept,
            condition_cols=condition_cols,
        )

        all_model_split_summaries.append(split_summary_kept)
        all_model_summaries.append(model_summary)

    source = pd.concat(
        all_model_summaries,
        ignore_index=True,
    )

    counts = pd.concat(
        all_model_split_summaries,
        ignore_index=True,
    )

    source = source.sort_values(
        [
            "model",
            "location",
            "milestone_order",
            "layer",
        ]
    )

    counts = counts.sort_values(
        [
            "model",
            "split",
            "location",
            "milestone_order",
            "layer",
        ]
    )

    save_csv(
        source,
        plot_dir / "source_summary_across_splits.csv",
    )

    save_csv(
        counts,
        plot_dir / "counts_summary_by_split.csv",
    )

    if all_raw_milestone_exclusions:
        raw_exclusions = pd.concat(
            all_raw_milestone_exclusions,
            ignore_index=True,
            sort=False,
        )
    else:
        raw_exclusions = pd.DataFrame()

    if all_threshold_exclusions:
        threshold_exclusions = pd.concat(
            all_threshold_exclusions,
            ignore_index=True,
            sort=False,
        )
    else:
        threshold_exclusions = pd.DataFrame()

    save_csv(
        raw_exclusions,
        plot_dir / "excluded_raw_rows_not_clean_milestones.csv",
    )

    save_csv(
        threshold_exclusions,
        plot_dir / "excluded_split_layer_cells_below_threshold.csv",
    )

    probe_target = RUN_FOLDERS[run_key]["probe_target"]

    if probe_target == "interpretation":
        title = (
            "Ambiguity-vector interpretation decoding across layers "
            "(clean milestones only)"
        )
        filename = "ambiguity_interpretation_clean_milestones"
    else:
        title = (
            "Ambiguity-vector cue-type decoding across layers "
            "(clean milestones only)"
        )
        filename = "ambiguity_cuetype_clean_milestones"

    plot_milestone_facets_by_model(
        summary=source,
        y_col="mean_balanced_accuracy",
        title=title,
        y_label="Mean balanced accuracy across splits",
        output_dir=plot_dir,
        filename_stem=filename,
        chance_line=True,
    )


def choose_best_layers_for_uncertainty():
    """
    Choose best layer per model from endpoint_interpretation full sentence.

    This is intentionally simple and auditable.

    We use the already generated source CSV from:
        interpretation_endpoint_full_sentence

    Best layer criterion:
        maximum mean_balanced_accuracy

    If tied:
        first row after sorting by model, descending accuracy, ascending layer.
    """

    source_path = (
        OUTPUT_ROOT
        /
        "interpretation_endpoint_full_sentence"
        /
        "source_summary_across_splits.csv"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Best-layer source file not found. "
            "Run endpoint interpretation plot first."
        )

    source = pd.read_csv(source_path)

    rows = []

    for model_key, sub in source.groupby("model", sort=False):

        sub = (
            sub
            .sort_values(
                [
                    "mean_balanced_accuracy",
                    "layer",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .copy()
        )

        best = sub.iloc[0].to_dict()

        rows.append(
            {
                "model": model_key,
                "model_display": best["model_display"],
                "best_layer": int(best["layer"]),
                "best_layer_relative": float(best["layer_relative"]),
                "best_mean_balanced_accuracy": float(
                    best["mean_balanced_accuracy"]
                ),
                "selection_source": str(source_path),
                "selection_rule": (
                    "max endpoint_interpretation full_sentence "
                    "mean_balanced_accuracy across splits"
                ),
            }
        )

    out = pd.DataFrame(rows)

    save_csv(
        out,
        OUTPUT_ROOT / "uncertainty_interpretation_best_layers" / "best_layers.csv",
    )

    return out


def build_uncertainty_plot():
    """
    Build uncertainty across time for interpretation only.

    Uses:
        ambiguity_interpretation

    Layer selection:
        best endpoint_interpretation full_sentence layer per model

    Metrics:
        mean entropy
        mean max probability
        mean abs signed distance

    This plot is diagnostic, not final.
    """

    plot_dir = OUTPUT_ROOT / "uncertainty_interpretation_best_layers"

    best_layers = choose_best_layers_for_uncertainty()

    all_model_summaries = []
    all_model_split_summaries = []
    all_raw_milestone_exclusions = []
    all_threshold_exclusions = []

    for model_key in MODEL_RUNS:

        best_layer = int(
            best_layers[
                best_layers["model"] == model_key
            ]["best_layer"]
            .iloc[0]
        )

        df, n_layers = load_predictions(
            model_key=model_key,
            run_key="ambiguity_interpretation",
        )

        df = df[
            df["layer"].astype(int)
            ==
            best_layer
        ].copy()

        kept, raw_excluded = add_clean_milestone_labels(df)

        if len(raw_excluded) > 0:
            raw_excluded = raw_excluded.copy()
            raw_excluded["model"] = model_key
            raw_excluded["run_key"] = "ambiguity_interpretation"
            raw_excluded["best_layer"] = best_layer
            all_raw_milestone_exclusions.append(raw_excluded)

        if len(kept) == 0:
            raise ValueError(
                f"No clean milestone rows for uncertainty plot: {model_key}"
            )

        condition_cols = [
            "location",
            "stage",
            "tracked_region",
            "milestone",
            "milestone_order",
            "milestone_label",
            "milestone_rule",
        ]

        split_summary = aggregate_predictions_by_split(
            kept,
            condition_cols=condition_cols,
        )

        split_summary["best_layer"] = best_layer

        split_summary_kept, split_summary_excluded = apply_min_sentence_threshold(
            split_summary,
            condition_cols=condition_cols,
            min_sentences=MIN_SENTENCES_PER_SPLIT_LAYER_CONDITION,
        )

        if len(split_summary_excluded) > 0:
            split_summary_excluded = split_summary_excluded.copy()
            split_summary_excluded["model"] = model_key
            split_summary_excluded["run_key"] = "ambiguity_interpretation"
            split_summary_excluded["best_layer"] = best_layer
            all_threshold_exclusions.append(split_summary_excluded)

        model_summary = average_across_splits(
            split_summary_kept,
            condition_cols=condition_cols,
        )

        model_summary["best_layer"] = best_layer

        all_model_split_summaries.append(split_summary_kept)
        all_model_summaries.append(model_summary)

    source = pd.concat(
        all_model_summaries,
        ignore_index=True,
    )

    counts = pd.concat(
        all_model_split_summaries,
        ignore_index=True,
    )

    source = source.sort_values(
        [
            "model",
            "location",
            "milestone_order",
        ]
    )

    counts = counts.sort_values(
        [
            "model",
            "split",
            "location",
            "milestone_order",
        ]
    )

    save_csv(
        source,
        plot_dir / "source_summary_across_splits.csv",
    )

    save_csv(
        counts,
        plot_dir / "counts_summary_by_split.csv",
    )

    if all_raw_milestone_exclusions:
        raw_exclusions = pd.concat(
            all_raw_milestone_exclusions,
            ignore_index=True,
            sort=False,
        )
    else:
        raw_exclusions = pd.DataFrame()

    if all_threshold_exclusions:
        threshold_exclusions = pd.concat(
            all_threshold_exclusions,
            ignore_index=True,
            sort=False,
        )
    else:
        threshold_exclusions = pd.DataFrame()

    save_csv(
        raw_exclusions,
        plot_dir / "excluded_raw_rows_not_clean_milestones.csv",
    )

    save_csv(
        threshold_exclusions,
        plot_dir / "excluded_split_layer_cells_below_threshold.csv",
    )

    plot_uncertainty_milestones(
        summary=source,
        output_dir=plot_dir,
        filename_stem="uncertainty_entropy_clean_milestones_best_layers",
    )

    # Also create simple line plots for confidence-style diagnostics.
    for metric, label, filename in [
        (
            "mean_max_probability",
            "Mean max probability",
            "max_probability_clean_milestones_best_layers",
        ),
        (
            "mean_abs_signed_distance",
            "Mean absolute signed distance",
            "abs_signed_distance_clean_milestones_best_layers",
        ),
    ]:

        plot_uncertainty_metric_by_location(
            summary=source,
            metric=metric,
            y_label=label,
            title=label + " across clean milestones at best layers",
            output_dir=plot_dir,
            filename_stem=filename,
        )


def plot_uncertainty_metric_by_location(
    summary,
    metric,
    y_label,
    title,
    output_dir,
    filename_stem,
):
    """
    Plot a generic uncertainty/confidence metric over milestones.

    This is used for:
        mean_max_probability
        mean_abs_signed_distance
    """

    models = (
        summary["model"]
        .drop_duplicates()
        .tolist()
    )

    n_models = len(models)

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=1,
        figsize=(11, 4 * n_models),
        sharex=False,
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):

        model_df = (
            summary[
                summary["model"] == model_key
            ]
            .copy()
        )

        model_display = model_df["model_display"].iloc[0]

        for location, sub_location in model_df.groupby(
            "location",
            sort=False,
        ):

            sub_location = sub_location.sort_values(
                "milestone_order"
            )

            ax.plot(
                sub_location["milestone_order"],
                sub_location[metric],
                marker="o",
                linewidth=1.5,
                markersize=5,
                label=location,
            )

        ax.set_title(
            model_display
        )

        ax.set_ylabel(
            y_label
        )

        ax.set_xticks(
            sorted(
                model_df["milestone_order"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        ax.set_xticklabels(
            [
                str(int(x))
                for x in sorted(
                    model_df["milestone_order"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            ]
        )

        ax.legend(
            title="Location",
            fontsize=8,
        )

    axes[-1].set_xlabel(
        "Milestone order within location"
    )

    fig.suptitle(
        title,
        y=1.01,
    )

    ensure_dir(output_dir)

    plt.tight_layout()

    png_path = Path(output_dir) / f"{filename_stem}.png"
    pdf_path = Path(output_dir) / f"{filename_stem}.pdf"

    plt.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():
    """
    Run all diagnostic plotting steps.
    """

    ensure_dir(
        OUTPUT_ROOT
    )

    ensure_dir(
        OUTPUT_ROOT / "audits"
    )

    print()
    print("=" * 80)
    print("STEP 3 DIAGNOSTIC PLOTTING")
    print("=" * 80)
    print(f"Output root: {OUTPUT_ROOT}")
    print()

    print("1. Auditing expected run folders...")
    discovered = make_discovered_runs_audit()

    missing = discovered[
        ~discovered["probe_predictions_exists"]
    ]

    if len(missing) > 0:
        print()
        print("Missing prediction files:")
        print(
            missing[
                [
                    "model",
                    "run_key",
                    "run_dir",
                    "probe_predictions_exists",
                ]
            ]
        )
        raise FileNotFoundError(
            "At least one expected probe_predictions.csv file is missing."
        )

    print("   Done.")

    print("2. Saving stage audits...")
    make_stage_audits()
    print("   Done.")

    print("3. Plotting endpoint interpretation, full sentence only...")
    build_endpoint_full_sentence_plot(
        run_key="endpoint_interpretation",
        plot_name="interpretation_endpoint_full_sentence",
    )
    print("   Done.")

    print("4. Plotting ambiguity interpretation, clean milestones only...")
    build_ambiguity_milestone_plot(
        run_key="ambiguity_interpretation",
        plot_name="interpretation_ambiguity_clean_milestones",
    )
    print("   Done.")

    print("5. Plotting endpoint cue type, full sentence only...")
    build_endpoint_full_sentence_plot(
        run_key="endpoint_cuetype",
        plot_name="cuetype_endpoint_full_sentence",
    )
    print("   Done.")

    print("6. Plotting ambiguity cue type, clean milestones only...")
    build_ambiguity_milestone_plot(
        run_key="ambiguity_cuetype",
        plot_name="cuetype_ambiguity_clean_milestones",
    )
    print("   Done.")

    print("7. Plotting interpretation uncertainty at best layers...")
    build_uncertainty_plot()
    print("   Done.")

    manifest = {
        "output_root": str(OUTPUT_ROOT),
        "min_sentences_per_split_layer_condition": (
            MIN_SENTENCES_PER_SPLIT_LAYER_CONDITION
        ),
        "model_runs": MODEL_RUNS,
        "run_folders": RUN_FOLDERS,
        "milestone_rules": MILESTONE_RULES,
        "plots_created": [
            "interpretation_endpoint_full_sentence",
            "interpretation_ambiguity_clean_milestones",
            "cuetype_endpoint_full_sentence",
            "cuetype_ambiguity_clean_milestones",
            "uncertainty_interpretation_best_layers",
        ],
        "notes": [
            "All stage-sensitive plots use probe_predictions.csv.",
            "Endpoint plots keep only stage == full_sentence.",
            "Ambiguity milestone plots use explicit location + stage + tracked_region rules.",
            "Rows not matching clean milestone rules are saved in exclusion logs.",
            "Cells below the minimum sentence threshold are saved in threshold exclusion logs.",
            "Cross-model plots use layer_relative, not raw layer numbers.",
            "Aggregation is model to split to layer to condition, then averaged across splits.",
        ],
    }

    save_json(
        manifest,
        OUTPUT_ROOT / "plot_manifest.json",
    )

    print()
    print("=" * 80)
    print("ALL DONE")
    print("=" * 80)
    print()
    print("Main output folders:")
    print(f"  {OUTPUT_ROOT / 'audits'}")
    print(f"  {OUTPUT_ROOT / 'interpretation_endpoint_full_sentence'}")
    print(f"  {OUTPUT_ROOT / 'interpretation_ambiguity_clean_milestones'}")
    print(f"  {OUTPUT_ROOT / 'cuetype_endpoint_full_sentence'}")
    print(f"  {OUTPUT_ROOT / 'cuetype_ambiguity_clean_milestones'}")
    print(f"  {OUTPUT_ROOT / 'uncertainty_interpretation_best_layers'}")
    print()
    print("Read the CSVs before interpreting the plots.")
    print()


if __name__ == "__main__":
    main()