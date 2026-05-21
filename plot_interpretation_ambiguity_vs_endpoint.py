#!/usr/bin/env python3

"""
Interpretation decoding:
Ambiguous-form representation vs sentence-end representation.

Goal:
    Compare where interpretation is decodable:
        1. at the ambiguous form
        2. at sentence end

Inputs:
    qwen_step3/ambiguity_interpretation/probe_predictions.csv
    qwen_step3/endpoint_interpretation/probe_predictions.csv

    tgemma_step3/ambiguity_interpretation/probe_predictions.csv
    tgemma_step3/endpoint_interpretation/probe_predictions.csv

    tllama_step3/ambiguity_interpretation/probe_predictions.csv
    tllama_step3/endpoint_interpretation/probe_predictions.csv

Critical rules:
    1. Ambiguous-form site:
        use ambiguity_interpretation
        keep stage == ambiguity_onset
        keep tracked_region == ambiguity

    2. Sentence-end site:
        use endpoint_interpretation
        keep stage == full_sentence
        keep is_full_sentence == true

    3. Unit of analysis:
        one row per model x split x layer x row_id x site

    4. Exclude ambiguous-final sentences:
        remove row_ids where ambiguity_prefix_end >= sentence_end_prefix_end

    5. Important:
        We do NOT require the same split across ambiguity and endpoint runs.
        Step 3 runs can have different eval row_ids because train rows are
        removed separately per run.

    6. Fair comparison:
        We restrict to row_ids available at both sites at the model level.
        Then each site is aggregated using its own valid splits.

    7. Split by cue position:
        before = Disamb > Amb
        after  = Amb > Disamb

Outputs:
    interpretation_ambiguity_vs_endpoint_plot/
        interpretation_ambiguity_vs_endpoint_layerwise.png
        interpretation_ambiguity_vs_endpoint_layerwise.pdf
        interpretation_ambiguity_vs_endpoint_9bins.png
        interpretation_ambiguity_vs_endpoint_9bins.pdf

        ambiguity_rows_before_dedup.csv
        endpoint_rows_before_dedup.csv
        ambiguity_rows_after_dedup.csv
        endpoint_rows_after_dedup.csv
        combined_rows_used.csv

        excluded_ambiguity_final_rowids.csv
        excluded_rowids_not_available_at_both_sites.csv
        duplicate_audit_ambiguity.csv
        duplicate_audit_endpoint.csv
        problem_duplicates_ambiguity.csv
        problem_duplicates_endpoint.csv

        source_layer_split_site_accuracy.csv
        source_model_layerwise_site_accuracy.csv
        source_split_9bin_site_accuracy.csv
        source_model_9bin_site_accuracy.csv
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score


# ==========================================================
# CONFIG
# ==========================================================

MODEL_CONFIGS = {
    "qwen": {
        "display_name": "Qwen2.5-14B-Instruct",
        "ambiguity_path": "qwen_step3/ambiguity_interpretation/probe_predictions.csv",
        "endpoint_path": "qwen_step3/endpoint_interpretation/probe_predictions.csv",
    },
    "tgemma": {
        "display_name": "Turkish-Gemma-9b-T1",
        "ambiguity_path": "tgemma_step3/ambiguity_interpretation/probe_predictions.csv",
        "endpoint_path": "tgemma_step3/endpoint_interpretation/probe_predictions.csv",
    },
    "tllama": {
        "display_name": "Turkish-Llama-8b-Instruct-v0.1",
        "ambiguity_path": "tllama_step3/ambiguity_interpretation/probe_predictions.csv",
        "endpoint_path": "tllama_step3/endpoint_interpretation/probe_predictions.csv",
    },
}

LOCATION_LABELS = {
    "before": "Disamb > Amb",
    "after": "Amb > Disamb",
}

SITE_LABELS = {
    "ambiguity": "Ambiguous form",
    "endpoint": "Sentence end",
}

OUTPUT_DIR = Path("interpretation_ambiguity_vs_endpoint_plot")


# ==========================================================
# HELPERS
# ==========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_bool(series):
    return (
        series
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def require_columns(df, columns, context):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


def validate_prediction_file(df, expected_vector_type, expected_probe_target, context):
    vector_types = sorted(df["vector_type"].dropna().astype(str).unique().tolist())
    probe_targets = sorted(df["probe_target"].dropna().astype(str).unique().tolist())

    if vector_types != [expected_vector_type]:
        raise ValueError(
            f"{context} should have vector_type={expected_vector_type}, "
            f"but found {vector_types}"
        )

    if probe_targets != [expected_probe_target]:
        raise ValueError(
            f"{context} should have probe_target={expected_probe_target}, "
            f"but found {probe_targets}"
        )


# ==========================================================
# LOAD / FILTER
# ==========================================================

def load_ambiguity_rows(model_key, config):
    """
    Load ambiguous-form representation rows.

    Keep:
        stage == ambiguity_onset
        tracked_region == ambiguity
    """

    path = Path(config["ambiguity_path"])

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = [
        "split",
        "split_seed",
        "row_id",
        "layer",
        "stage",
        "tracked_region",
        "location",
        "prefix_end",
        "gold_class",
        "gold_label",
        "predicted_class",
        "predicted_label",
        "vector_type",
        "probe_target",
    ]

    require_columns(df, required, str(path))

    validate_prediction_file(
        df,
        expected_vector_type="ambiguity",
        expected_probe_target="interpretation",
        context=str(path),
    )

    mask = (
        df["stage"].astype(str).eq("ambiguity_onset")
        &
        df["tracked_region"].astype(str).eq("ambiguity")
        &
        df["location"].astype(str).isin(["before", "after"])
    )

    kept = df[mask].copy()
    excluded = df[~mask].copy()

    kept["model"] = model_key
    kept["model_display"] = config["display_name"]
    kept["site"] = "ambiguity"
    kept["site_label"] = SITE_LABELS["ambiguity"]
    kept["location_label"] = kept["location"].map(LOCATION_LABELS)
    kept["ambiguity_prefix_end"] = kept["prefix_end"].astype(int)

    excluded["model"] = model_key
    excluded["model_display"] = config["display_name"]
    excluded["site"] = "ambiguity"
    excluded["exclusion_reason"] = "not_ambiguity_onset_tracked_region_ambiguity"

    return kept, excluded


def load_endpoint_rows(model_key, config):
    """
    Load sentence-end representation rows.

    Keep:
        stage == full_sentence
        is_full_sentence == true
    """

    path = Path(config["endpoint_path"])

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = [
        "split",
        "split_seed",
        "row_id",
        "layer",
        "stage",
        "is_full_sentence",
        "location",
        "prefix_end",
        "gold_class",
        "gold_label",
        "predicted_class",
        "predicted_label",
        "vector_type",
        "probe_target",
    ]

    require_columns(df, required, str(path))

    validate_prediction_file(
        df,
        expected_vector_type="endpoint",
        expected_probe_target="interpretation",
        context=str(path),
    )

    mask = (
        df["stage"].astype(str).eq("full_sentence")
        &
        normalize_bool(df["is_full_sentence"])
        &
        df["location"].astype(str).isin(["before", "after"])
    )

    kept = df[mask].copy()
    excluded = df[~mask].copy()

    kept["model"] = model_key
    kept["model_display"] = config["display_name"]
    kept["site"] = "endpoint"
    kept["site_label"] = SITE_LABELS["endpoint"]
    kept["location_label"] = kept["location"].map(LOCATION_LABELS)
    kept["sentence_end_prefix_end"] = kept["prefix_end"].astype(int)

    excluded["model"] = model_key
    excluded["model_display"] = config["display_name"]
    excluded["site"] = "endpoint"
    excluded["exclusion_reason"] = "not_full_sentence"

    return kept, excluded


# ==========================================================
# DUPLICATE AUDITS
# ==========================================================

def audit_duplicates(df, site_name):
    """
    Each site should contribute once per:

        model x split x layer x row_id

    Duplicates are allowed only if they are identical enough.
    Problem duplicates are logged and excluded before analysis.
    """

    unit_cols = [
        "model",
        "model_display",
        "split",
        "split_seed",
        "layer",
        "row_id",
    ]

    audit = (
        df
        .groupby(unit_cols, as_index=False)
        .agg(
            n_duplicate_rows=("row_id", "size"),
            n_locations=("location", "nunique"),
            locations=("location", lambda x: "|".join(sorted(x.astype(str).unique()))),
            n_gold_classes=("gold_class", "nunique"),
            gold_classes=("gold_class", lambda x: "|".join(sorted(x.astype(str).unique()))),
            n_predicted_classes=("predicted_class", "nunique"),
            predicted_classes=("predicted_class", lambda x: "|".join(sorted(x.astype(str).unique()))),
            n_prefix_end_values=("prefix_end", "nunique"),
            prefix_end_values=("prefix_end", lambda x: "|".join(sorted(x.astype(str).unique()))),
        )
    )

    problems = audit[
        (audit["n_duplicate_rows"] > 1)
        &
        (
            (audit["n_locations"] > 1)
            |
            (audit["n_gold_classes"] > 1)
            |
            (audit["n_predicted_classes"] > 1)
            |
            (audit["n_prefix_end_values"] > 1)
        )
    ].copy()

    audit["site"] = site_name
    problems["site"] = site_name

    return audit, problems


def exclude_problem_duplicate_units(df, problem_duplicates):
    """
    Remove units where duplicate rows disagree.

    Unit:
        model x split x layer x row_id
    """

    if len(problem_duplicates) == 0:
        return df.copy(), pd.DataFrame()

    unit_cols = [
        "model",
        "split",
        "layer",
        "row_id",
    ]

    problem_units = problem_duplicates[unit_cols].drop_duplicates().copy()
    problem_units["problem_duplicate_unit"] = True

    merged = df.merge(
        problem_units,
        on=unit_cols,
        how="left",
    )

    excluded = merged[
        merged["problem_duplicate_unit"].fillna(False)
    ].copy()

    kept = merged[
        ~merged["problem_duplicate_unit"].fillna(False)
    ].copy()

    kept = kept.drop(columns=["problem_duplicate_unit"])
    excluded["exclusion_reason"] = "problem_duplicate_unit"

    return kept, excluded


def deduplicate_site_rows(df):
    """
    Keep one row per:

        model x split x layer x row_id

    This is done after removing problem duplicate units.
    """

    deduped = (
        df
        .sort_values(["model", "split", "layer", "row_id"])
        .drop_duplicates(
            subset=["model", "split", "layer", "row_id"],
            keep="first",
        )
        .copy()
    )

    return deduped


# ==========================================================
# ROW-ID FILTERING ACROSS SITES
# ==========================================================

def build_valid_rowid_filter(ambiguity_df, endpoint_df):
    """
    Build row_id-level exclusions at the model level.

    We do NOT pair by split, because ambiguity and endpoint Step 3 runs
    can have different eval rows per split.

    We do:
        1. Find row_ids that exist at both sites for each model.
        2. Compare ambiguity_prefix_end and sentence_end_prefix_end.
        3. Exclude row_ids where ambiguity_prefix_end >= sentence_end_prefix_end.

    Returns:
        valid_rowids
        excluded_not_both_sites
        excluded_ambiguity_final
        rowid_prefix_audit
    """

    ambiguity_rowids = (
        ambiguity_df
        .groupby(
            [
                "model",
                "model_display",
                "row_id",
            ],
            as_index=False,
        )
        .agg(
            ambiguity_prefix_end=("ambiguity_prefix_end", "min"),
            ambiguity_prefix_end_max=("ambiguity_prefix_end", "max"),
            ambiguity_location_nunique=("location", "nunique"),
            ambiguity_locations=("location", lambda x: "|".join(sorted(x.astype(str).unique()))),
            ambiguity_gold_nunique=("gold_class", "nunique"),
            ambiguity_gold_classes=("gold_class", lambda x: "|".join(sorted(x.astype(str).unique()))),
        )
    )

    endpoint_rowids = (
        endpoint_df
        .groupby(
            [
                "model",
                "model_display",
                "row_id",
            ],
            as_index=False,
        )
        .agg(
            sentence_end_prefix_end=("sentence_end_prefix_end", "max"),
            sentence_end_prefix_end_min=("sentence_end_prefix_end", "min"),
            endpoint_location_nunique=("location", "nunique"),
            endpoint_locations=("location", lambda x: "|".join(sorted(x.astype(str).unique()))),
            endpoint_gold_nunique=("gold_class", "nunique"),
            endpoint_gold_classes=("gold_class", lambda x: "|".join(sorted(x.astype(str).unique()))),
        )
    )

    rowid_check = ambiguity_rowids.merge(
        endpoint_rowids,
        on=[
            "model",
            "model_display",
            "row_id",
        ],
        how="outer",
        indicator=True,
    )

    excluded_not_both = rowid_check[
        rowid_check["_merge"] != "both"
    ].copy()

    if len(excluded_not_both) > 0:
        excluded_not_both["exclusion_reason"] = "row_id_not_available_at_both_sites"

    both = rowid_check[
        rowid_check["_merge"] == "both"
    ].copy()

    inconsistent = both[
        (
            both["ambiguity_location_nunique"].astype(int) > 1
        )
        |
        (
            both["endpoint_location_nunique"].astype(int) > 1
        )
        |
        (
            both["ambiguity_gold_nunique"].astype(int) > 1
        )
        |
        (
            both["endpoint_gold_nunique"].astype(int) > 1
        )
        |
        (
            both["ambiguity_locations"].astype(str)
            !=
            both["endpoint_locations"].astype(str)
        )
        |
        (
            both["ambiguity_gold_classes"].astype(str)
            !=
            both["endpoint_gold_classes"].astype(str)
        )
    ].copy()

    if len(inconsistent) > 0:
        inconsistent["exclusion_reason"] = "row_id_location_or_gold_inconsistent_across_sites"

    consistent = both.merge(
        inconsistent[
            [
                "model",
                "row_id",
            ]
        ].drop_duplicates().assign(inconsistent_rowid=True),
        on=[
            "model",
            "row_id",
        ],
        how="left",
    )

    consistent = consistent[
        ~consistent["inconsistent_rowid"].fillna(False)
    ].copy()

    excluded_ambiguity_final = consistent[
        consistent["ambiguity_prefix_end"].astype(int)
        >=
        consistent["sentence_end_prefix_end"].astype(int)
    ].copy()

    if len(excluded_ambiguity_final) > 0:
        excluded_ambiguity_final["exclusion_reason"] = (
            "ambiguity_prefix_end_greater_or_equal_sentence_end_prefix_end"
        )

    valid = consistent.merge(
        excluded_ambiguity_final[
            [
                "model",
                "row_id",
            ]
        ].drop_duplicates().assign(ambiguity_final=True),
        on=[
            "model",
            "row_id",
        ],
        how="left",
    )

    valid = valid[
        ~valid["ambiguity_final"].fillna(False)
    ].copy()

    valid_rowids = valid[
        [
            "model",
            "row_id",
        ]
    ].drop_duplicates().copy()

    excluded_all = pd.concat(
        [
            excluded_not_both,
            inconsistent,
        ],
        ignore_index=True,
        sort=False,
    )

    return valid_rowids, excluded_all, excluded_ambiguity_final, rowid_check


def keep_valid_rowids(df, valid_rowids):
    """
    Keep only model x row_id units that passed row-id-level filtering.
    """

    kept = df.merge(
        valid_rowids,
        on=[
            "model",
            "row_id",
        ],
        how="inner",
    )

    return kept


# ==========================================================
# RELATIVE LAYERS
# ==========================================================

def add_relative_layer(df):
    """
    Add model-specific relative layer.

    Important:
        This must be computed per model, not globally.
    """

    df = df.copy()

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()

    if duplicated_columns:
        raise ValueError(
            f"Duplicate column names found before adding relative layer: "
            f"{duplicated_columns}"
        )

    df["layer"] = df["layer"].astype(int)

    df["n_layers_model"] = (
        df
        .groupby("model")["layer"]
        .transform("max")
        .astype(int)
        +
        1
    )

    df["layer_relative"] = (
        df["layer"]
        /
        (df["n_layers_model"] - 1)
    )

    df.loc[
        df["n_layers_model"] <= 1,
        "layer_relative",
    ] = 0.0

    return df


# ==========================================================
# 9 BINS
# ==========================================================

def assign_9bin(layer_relative):
    """
    9 relative-layer bins:

        Early 1, Early 2, Early 3
        Middle 1, Middle 2, Middle 3
        Late 1, Late 2, Late 3
    """

    x = float(layer_relative)

    if x < 1 / 9:
        return "Early 1", 1
    elif x < 2 / 9:
        return "Early 2", 2
    elif x < 3 / 9:
        return "Early 3", 3
    elif x < 4 / 9:
        return "Middle 1", 4
    elif x < 5 / 9:
        return "Middle 2", 5
    elif x < 6 / 9:
        return "Middle 3", 6
    elif x < 7 / 9:
        return "Late 1", 7
    elif x < 8 / 9:
        return "Late 2", 8
    else:
        return "Late 3", 9


def add_9bin(df):
    df = df.copy()

    labels = []
    orders = []

    for value in df["layer_relative"]:
        label, order = assign_9bin(value)
        labels.append(label)
        orders.append(order)

    df["layer_bin"] = labels
    df["layer_bin_order"] = orders

    return df


# ==========================================================
# ACCURACY
# ==========================================================

def compute_layer_accuracy_by_split_site(df):
    """
    Compute balanced accuracy per:

        model x location x site x split x layer
    """

    rows = []

    group_cols = [
        "model",
        "model_display",
        "location",
        "location_label",
        "site",
        "site_label",
        "split",
        "split_seed",
        "layer",
        "layer_relative",
        "n_layers_model",
    ]

    for keys, group in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))

        gold = group["gold_class"].astype(int)
        pred = group["predicted_class"].astype(int)

        if len(gold.unique()) < 2:
            acc = float("nan")
        else:
            acc = float(balanced_accuracy_score(gold, pred))

        row.update(
            {
                "balanced_accuracy": acc,
                "n_sentences": int(group["row_id"].nunique()),
                "gold_class0_count": int((gold == 0).sum()),
                "gold_class1_count": int((gold == 1).sum()),
                "pred_class0_count": int((pred == 0).sum()),
                "pred_class1_count": int((pred == 1).sum()),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def average_layer_accuracy_across_splits(layer_split_df):
    """
    Average regular layerwise accuracy across splits.

    Input:
        one row per model x location x site x split x layer

    Output:
        one row per model x location x site x layer
    """

    if len(layer_split_df) == 0:
        return pd.DataFrame()

    return (
        layer_split_df
        .groupby(
            [
                "model",
                "model_display",
                "location",
                "location_label",
                "site",
                "site_label",
                "layer",
                "layer_relative",
                "n_layers_model",
            ],
            as_index=False,
        )
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            n_splits=("split", "nunique"),
            mean_n_sentences=("n_sentences", "mean"),
            min_n_sentences=("n_sentences", "min"),
            mean_gold_class0_count=("gold_class0_count", "mean"),
            mean_gold_class1_count=("gold_class1_count", "mean"),
            mean_pred_class0_count=("pred_class0_count", "mean"),
            mean_pred_class1_count=("pred_class1_count", "mean"),
        )
    )


def compute_9bin_accuracy(layer_split_df):
    """
    9-bin aggregation.

    model x location x site x split x layer accuracy
    -> assign layer bin
    -> average layers inside split/bin
    -> average bins across splits
    """

    if len(layer_split_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = add_9bin(layer_split_df)

    split_bin = (
        df
        .groupby(
            [
                "model",
                "model_display",
                "location",
                "location_label",
                "site",
                "site_label",
                "split",
                "split_seed",
                "layer_bin",
                "layer_bin_order",
            ],
            as_index=False,
        )
        .agg(
            bin_balanced_accuracy=("balanced_accuracy", "mean"),
            n_layers_in_bin=("layer", "nunique"),
            mean_n_sentences=("n_sentences", "mean"),
            min_n_sentences=("n_sentences", "min"),
        )
    )

    model_bin = (
        split_bin
        .groupby(
            [
                "model",
                "model_display",
                "location",
                "location_label",
                "site",
                "site_label",
                "layer_bin",
                "layer_bin_order",
            ],
            as_index=False,
        )
        .agg(
            mean_balanced_accuracy=("bin_balanced_accuracy", "mean"),
            sd_balanced_accuracy=("bin_balanced_accuracy", "std"),
            n_splits=("split", "nunique"),
            mean_n_layers_in_bin=("n_layers_in_bin", "mean"),
            mean_n_sentences=("mean_n_sentences", "mean"),
            min_n_sentences=("min_n_sentences", "min"),
        )
    )

    return split_bin, model_bin


# ==========================================================
# PLOTS
# ==========================================================

def plot_layerwise(model_layer_df):
    """
    Layerwise plot.

    Layout:
        rows = models
        columns = locations

    Lines:
        Ambiguous form
        Sentence end
    """

    models = list(MODEL_CONFIGS.keys())
    locations = ["before", "after"]

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=len(locations),
        figsize=(15, 12),
        sharex=True,
        sharey=True,
    )

    for row_idx, model_key in enumerate(models):
        for col_idx, location in enumerate(locations):
            ax = axes[row_idx][col_idx]

            sub_panel = model_layer_df[
                (model_layer_df["model"] == model_key)
                &
                (model_layer_df["location"] == location)
            ].copy()

            for site in ["ambiguity", "endpoint"]:
                sub = (
                    sub_panel[
                        sub_panel["site"] == site
                    ]
                    .sort_values("layer_relative")
                    .copy()
                )

                if len(sub) == 0:
                    continue

                ax.plot(
                    sub["layer_relative"],
                    sub["mean_balanced_accuracy"],
                    marker="o",
                    linewidth=1.5,
                    markersize=3,
                    label=SITE_LABELS[site],
                )

            ax.axhline(y=0.5, linestyle="--", linewidth=1)
            ax.set_ylim(0.35, 1.00)

            if row_idx == 0:
                ax.set_title(LOCATION_LABELS[location])

            if col_idx == 0:
                ax.set_ylabel(
                    MODEL_CONFIGS[model_key]["display_name"]
                    + "\nMean balanced accuracy"
                )

            if row_idx == len(models) - 1:
                ax.set_xlabel("Relative layer")

            ax.legend()

    fig.suptitle(
        "Interpretation decoding: ambiguous form vs sentence end\n"
        "Common row_ids by site, ambiguity-final sentences excluded"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "interpretation_ambiguity_vs_endpoint_layerwise.png",
        dpi=200,
    )

    plt.savefig(
        OUTPUT_DIR / "interpretation_ambiguity_vs_endpoint_layerwise.pdf",
    )

    plt.close()


def plot_9bin(model_bin_df):
    """
    9-bin plot.

    Layout:
        rows = models
        columns = locations

    Lines:
        Ambiguous form
        Sentence end
    """

    models = list(MODEL_CONFIGS.keys())
    locations = ["before", "after"]

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=len(locations),
        figsize=(15, 12),
        sharex=True,
        sharey=True,
    )

    for row_idx, model_key in enumerate(models):
        for col_idx, location in enumerate(locations):
            ax = axes[row_idx][col_idx]

            sub_panel = model_bin_df[
                (model_bin_df["model"] == model_key)
                &
                (model_bin_df["location"] == location)
            ].copy()

            for site in ["ambiguity", "endpoint"]:
                sub = (
                    sub_panel[
                        sub_panel["site"] == site
                    ]
                    .sort_values("layer_bin_order")
                    .copy()
                )

                if len(sub) == 0:
                    continue

                ax.plot(
                    sub["layer_bin_order"],
                    sub["mean_balanced_accuracy"],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    label=SITE_LABELS[site],
                )

            ax.axhline(y=0.5, linestyle="--", linewidth=1)
            ax.set_ylim(0.35, 1.00)

            ax.set_xticks(list(range(1, 10)))
            ax.set_xticklabels(
                ["E1", "E2", "E3", "M1", "M2", "M3", "L1", "L2", "L3"]
            )

            if row_idx == 0:
                ax.set_title(LOCATION_LABELS[location])

            if col_idx == 0:
                ax.set_ylabel(
                    MODEL_CONFIGS[model_key]["display_name"]
                    + "\nMean balanced accuracy"
                )

            if row_idx == len(models) - 1:
                ax.set_xlabel("Relative layer bin")

            ax.legend()

    fig.suptitle(
        "Interpretation decoding by relative-layer bin: ambiguous form vs sentence end\n"
        "Common row_ids by site, ambiguity-final sentences excluded"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "interpretation_ambiguity_vs_endpoint_9bins.png",
        dpi=200,
    )

    plt.savefig(
        OUTPUT_DIR / "interpretation_ambiguity_vs_endpoint_9bins.pdf",
    )

    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    all_ambiguity = []
    all_endpoint = []
    all_excluded_ambiguity_filter = []
    all_excluded_endpoint_filter = []

    print()
    print("Loading ambiguity and endpoint interpretation predictions...")

    for model_key, config in MODEL_CONFIGS.items():
        print(f"  {model_key}")

        ambiguity_rows, excluded_ambiguity = load_ambiguity_rows(
            model_key,
            config,
        )

        endpoint_rows, excluded_endpoint = load_endpoint_rows(
            model_key,
            config,
        )

        all_ambiguity.append(ambiguity_rows)
        all_endpoint.append(endpoint_rows)
        all_excluded_ambiguity_filter.append(excluded_ambiguity)
        all_excluded_endpoint_filter.append(excluded_endpoint)

    ambiguity_df = pd.concat(all_ambiguity, ignore_index=True)
    endpoint_df = pd.concat(all_endpoint, ignore_index=True)

    excluded_ambiguity_filter_df = pd.concat(
        all_excluded_ambiguity_filter,
        ignore_index=True,
    )

    excluded_endpoint_filter_df = pd.concat(
        all_excluded_endpoint_filter,
        ignore_index=True,
    )

    print()
    print("Auditing duplicates...")

    duplicate_audit_ambiguity, problem_duplicates_ambiguity = audit_duplicates(
        ambiguity_df,
        site_name="ambiguity",
    )

    duplicate_audit_endpoint, problem_duplicates_endpoint = audit_duplicates(
        endpoint_df,
        site_name="endpoint",
    )

    print(f"  ambiguity rows before dedup: {len(ambiguity_df)}")
    print(f"  endpoint rows before dedup: {len(endpoint_df)}")
    print(f"  problem ambiguity duplicate units: {len(problem_duplicates_ambiguity)}")
    print(f"  problem endpoint duplicate units: {len(problem_duplicates_endpoint)}")

    ambiguity_no_problem, excluded_problem_ambiguity = exclude_problem_duplicate_units(
        ambiguity_df,
        problem_duplicates_ambiguity,
    )

    endpoint_no_problem, excluded_problem_endpoint = exclude_problem_duplicate_units(
        endpoint_df,
        problem_duplicates_endpoint,
    )

    print()
    print("Deduplicating site rows...")

    ambiguity_deduped = deduplicate_site_rows(ambiguity_no_problem)
    endpoint_deduped = deduplicate_site_rows(endpoint_no_problem)

    print(f"  ambiguity rows after dedup: {len(ambiguity_deduped)}")
    print(f"  endpoint rows after dedup: {len(endpoint_deduped)}")

    print()
    print("Building valid row_id filter...")

    (
        valid_rowids,
        excluded_not_both_or_inconsistent,
        excluded_ambiguity_final,
        rowid_prefix_audit,
    ) = build_valid_rowid_filter(
        ambiguity_deduped,
        endpoint_deduped,
    )

    print(f"  valid model-row_id units: {len(valid_rowids)}")
    print(f"  row_ids excluded for not both sites or inconsistency: {len(excluded_not_both_or_inconsistent)}")
    print(f"  row_ids excluded because ambiguous form is final: {len(excluded_ambiguity_final)}")

    print()
    print("Applying row_id filter to both sites...")

    ambiguity_valid = keep_valid_rowids(
        ambiguity_deduped,
        valid_rowids,
    )

    endpoint_valid = keep_valid_rowids(
        endpoint_deduped,
        valid_rowids,
    )

    combined_rows = pd.concat(
        [
            ambiguity_valid,
            endpoint_valid,
        ],
        ignore_index=True,
        sort=False,
    )

    combined_rows = add_relative_layer(combined_rows)

    print(f"  ambiguity rows used: {len(ambiguity_valid)}")
    print(f"  endpoint rows used: {len(endpoint_valid)}")
    print(f"  combined rows used: {len(combined_rows)}")

    print()
    print("Computing layerwise accuracy...")

    layer_split_accuracy = compute_layer_accuracy_by_split_site(
        combined_rows
    )

    model_layer_accuracy = average_layer_accuracy_across_splits(
        layer_split_accuracy
    )

    print()
    print("Computing 9-bin accuracy...")

    split_9bin_accuracy, model_9bin_accuracy = compute_9bin_accuracy(
        layer_split_accuracy
    )

    print()
    print("Saving CSVs...")

    ambiguity_df.to_csv(
        OUTPUT_DIR / "ambiguity_rows_before_dedup.csv",
        index=False,
    )

    endpoint_df.to_csv(
        OUTPUT_DIR / "endpoint_rows_before_dedup.csv",
        index=False,
    )

    ambiguity_deduped.to_csv(
        OUTPUT_DIR / "ambiguity_rows_after_dedup.csv",
        index=False,
    )

    endpoint_deduped.to_csv(
        OUTPUT_DIR / "endpoint_rows_after_dedup.csv",
        index=False,
    )

    ambiguity_valid.to_csv(
        OUTPUT_DIR / "ambiguity_rows_used.csv",
        index=False,
    )

    endpoint_valid.to_csv(
        OUTPUT_DIR / "endpoint_rows_used.csv",
        index=False,
    )

    combined_rows.to_csv(
        OUTPUT_DIR / "combined_rows_used.csv",
        index=False,
    )

    excluded_ambiguity_filter_df.to_csv(
        OUTPUT_DIR / "excluded_ambiguity_filter_rows.csv",
        index=False,
    )

    excluded_endpoint_filter_df.to_csv(
        OUTPUT_DIR / "excluded_endpoint_filter_rows.csv",
        index=False,
    )

    excluded_problem_ambiguity.to_csv(
        OUTPUT_DIR / "excluded_problem_duplicates_ambiguity.csv",
        index=False,
    )

    excluded_problem_endpoint.to_csv(
        OUTPUT_DIR / "excluded_problem_duplicates_endpoint.csv",
        index=False,
    )

    duplicate_audit_ambiguity.to_csv(
        OUTPUT_DIR / "duplicate_audit_ambiguity.csv",
        index=False,
    )

    duplicate_audit_endpoint.to_csv(
        OUTPUT_DIR / "duplicate_audit_endpoint.csv",
        index=False,
    )

    problem_duplicates_ambiguity.to_csv(
        OUTPUT_DIR / "problem_duplicates_ambiguity.csv",
        index=False,
    )

    problem_duplicates_endpoint.to_csv(
        OUTPUT_DIR / "problem_duplicates_endpoint.csv",
        index=False,
    )

    valid_rowids.to_csv(
        OUTPUT_DIR / "valid_model_rowids.csv",
        index=False,
    )

    rowid_prefix_audit.to_csv(
        OUTPUT_DIR / "rowid_prefix_audit.csv",
        index=False,
    )

    excluded_not_both_or_inconsistent.to_csv(
        OUTPUT_DIR / "excluded_rowids_not_available_at_both_sites_or_inconsistent.csv",
        index=False,
    )

    excluded_ambiguity_final.to_csv(
        OUTPUT_DIR / "excluded_ambiguity_final_rowids.csv",
        index=False,
    )

    layer_split_accuracy.to_csv(
        OUTPUT_DIR / "source_layer_split_site_accuracy.csv",
        index=False,
    )

    model_layer_accuracy.to_csv(
        OUTPUT_DIR / "source_model_layerwise_site_accuracy.csv",
        index=False,
    )

    split_9bin_accuracy.to_csv(
        OUTPUT_DIR / "source_split_9bin_site_accuracy.csv",
        index=False,
    )

    model_9bin_accuracy.to_csv(
        OUTPUT_DIR / "source_model_9bin_site_accuracy.csv",
        index=False,
    )

    print()
    print("Plotting layerwise plot...")

    plot_layerwise(model_layer_accuracy)

    print("Plotting 9-bin plot...")

    plot_9bin(model_9bin_accuracy)

    print()
    print("Done.")
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    print("Plots:")
    print(OUTPUT_DIR / "interpretation_ambiguity_vs_endpoint_layerwise.png")
    print(OUTPUT_DIR / "interpretation_ambiguity_vs_endpoint_9bins.png")
    print()
    print("Main source CSVs:")
    print(OUTPUT_DIR / "source_model_layerwise_site_accuracy.csv")
    print(OUTPUT_DIR / "source_model_9bin_site_accuracy.csv")
    print()
    print("Important audits:")
    print(OUTPUT_DIR / "combined_rows_used.csv")
    print(OUTPUT_DIR / "valid_model_rowids.csv")
    print(OUTPUT_DIR / "excluded_ambiguity_final_rowids.csv")
    print(OUTPUT_DIR / "excluded_rowids_not_available_at_both_sites_or_inconsistent.csv")
    print(OUTPUT_DIR / "problem_duplicates_ambiguity.csv")
    print(OUTPUT_DIR / "problem_duplicates_endpoint.csv")


if __name__ == "__main__":
    main()