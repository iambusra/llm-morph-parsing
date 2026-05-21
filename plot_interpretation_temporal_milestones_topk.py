#!/usr/bin/env python3

"""
Endpoint interpretation decoding over temporal milestones.

Goal:
    Track how interpretation-label accuracy evolves over sentence time.

Important:
    This script uses endpoint_interpretation only.

Why:
    We want interpretation accuracy at specific prefixes:
        - ambiguous-form prefix
        - end of disambiguation zone
        - sentence end

    Therefore, the representation is the current prefix endpoint,
    not the ambiguous-token vector.

Inputs:
    qwen_step3/endpoint_interpretation/probe_predictions.csv
    tgemma_step3/endpoint_interpretation/probe_predictions.csv
    tllama_step3/endpoint_interpretation/probe_predictions.csv

Temporal conditions:

    location == "after"
        Amb > Disamb

        x-axis:
            1. Ambiguous form
            2. Disamb end
            3. Sentence end

    location == "before"
        Disamb > Amb

        x-axis:
            1. Disamb end
            2. Ambiguous form
            3. Sentence end

Milestone definitions:

    Ambiguous form:
        tracked_region == ambiguity
        choose max prefix_end per model x split x layer x row_id

    Disamb end:
        tracked_region == cue
        choose max prefix_end per model x split x layer x row_id

    Sentence end:
        stage == full_sentence
        is_full_sentence == true
        choose max prefix_end per model x split x layer x row_id

Exclusions:

    Amb > Disamb:
        If Disamb end has the same prefix_end as sentence end,
        exclude the Disamb end datapoint.

    Disamb > Amb:
        If Ambiguous form has the same prefix_end as sentence end,
        exclude the Ambiguous form datapoint.

Layer handling:

    1. Compute layerwise accuracy first.
    2. Select top-k layers per model using sentence-end accuracy.
    3. Keep top-k layers.
    4. Average accuracy across top-k layers within each split.
    5. Average across splits.

Outputs:
    endpoint_interpretation_temporal_milestones_topk_plot/
        endpoint_interpretation_temporal_topk.png
        endpoint_interpretation_temporal_topk.pdf

        rows_loaded.csv
        milestone_candidates_before_selection.csv
        milestone_rows_selected_before_exclusion.csv
        milestone_rows_used.csv
        excluded_rows_no_sentence_end.csv
        excluded_rows_temporal_overlap.csv

        source_layer_split_milestone_accuracy.csv
        all_layer_reference_scores.csv
        topk_layers.csv
        source_topk_layer_rows.csv
        source_split_topk_milestone_accuracy.csv
        source_model_topk_milestone_accuracy.csv
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
        "path": "qwen_step3/endpoint_interpretation/probe_predictions.csv",
    },
    "tgemma": {
        "display_name": "Turkish-Gemma-9b-T1",
        "path": "tgemma_step3/endpoint_interpretation/probe_predictions.csv",
    },
    "tllama": {
        "display_name": "Turkish-Llama-8b-Instruct-v0.1",
        "path": "tllama_step3/endpoint_interpretation/probe_predictions.csv",
    },
}

LOCATION_LABELS = {
    "before": "Disamb > Amb",
    "after": "Amb > Disamb",
}

TOP_K_LAYERS = 5

OUTPUT_DIR = Path("endpoint_interpretation_temporal_milestones_topk_plot")


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


def validate_prediction_file(df, context):
    vector_types = sorted(df["vector_type"].dropna().astype(str).unique().tolist())
    probe_targets = sorted(df["probe_target"].dropna().astype(str).unique().tolist())

    if vector_types != ["endpoint"]:
        raise ValueError(f"{context} should be endpoint vectors, found {vector_types}")

    if probe_targets != ["interpretation"]:
        raise ValueError(f"{context} should be interpretation target, found {probe_targets}")


def balanced_accuracy_or_nan(gold, pred):
    gold = gold.astype(int)
    pred = pred.astype(int)

    if len(gold.unique()) < 2:
        return float("nan")

    return float(balanced_accuracy_score(gold, pred))


# ==========================================================
# LOAD DATA
# ==========================================================

def load_model(model_key, config):
    path = Path(config["path"])

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
        "is_full_sentence",
        "gold_class",
        "gold_label",
        "predicted_class",
        "predicted_label",
        "vector_type",
        "probe_target",
    ]

    require_columns(df, required, str(path))
    validate_prediction_file(df, str(path))

    df = df.copy()

    df["model"] = model_key
    df["model_display"] = config["display_name"]
    df["location_label"] = df["location"].map(LOCATION_LABELS)
    df["prefix_end"] = df["prefix_end"].astype(int)
    df["layer"] = df["layer"].astype(int)

    return df


# ==========================================================
# MILESTONE EXTRACTION
# ==========================================================

def milestone_order(location, milestone):
    """
    Temporal order differs by cue position.

    before:
        Disamb end -> Ambiguous form -> Sentence end

    after:
        Ambiguous form -> Disamb end -> Sentence end
    """

    if location == "before":
        order = {
            "disamb_end": 1,
            "ambiguity": 2,
            "sentence_end": 3,
        }

    elif location == "after":
        order = {
            "ambiguity": 1,
            "disamb_end": 2,
            "sentence_end": 3,
        }

    else:
        raise ValueError(f"Unexpected location: {location}")

    return order[milestone]


def make_milestone_candidates(df):
    """
    Build raw milestone candidate rows.

    We do not use "cue_onset" as the end of the cue.

    Disamb end is defined as:
        tracked_region == cue
        max prefix_end within the cue region
    """

    parts = []

    # Ambiguous-form prefix.
    amb = df[
        df["tracked_region"].astype(str).eq("ambiguity")
        &
        df["location"].astype(str).isin(["before", "after"])
    ].copy()

    amb["milestone"] = "ambiguity"
    amb["milestone_label"] = "Ambiguous form"

    # End of disambiguation zone.
    cue = df[
        df["tracked_region"].astype(str).eq("cue")
        &
        df["location"].astype(str).isin(["before", "after"])
    ].copy()

    cue["milestone"] = "disamb_end"
    cue["milestone_label"] = "Disamb end"

    # Sentence end.
    sent = df[
        df["stage"].astype(str).eq("full_sentence")
        &
        normalize_bool(df["is_full_sentence"])
        &
        df["location"].astype(str).isin(["before", "after"])
    ].copy()

    sent["milestone"] = "sentence_end"
    sent["milestone_label"] = "Sentence end"

    parts.extend([amb, cue, sent])

    candidates = pd.concat(parts, ignore_index=True, sort=False)

    candidates["location_label"] = candidates["location"].map(LOCATION_LABELS)

    candidates["milestone_order"] = candidates.apply(
        lambda row: milestone_order(
            row["location"],
            row["milestone"],
        ),
        axis=1,
    )

    return candidates


def select_one_prefix_per_milestone(candidates):
    """
    For each:
        model x split x layer x row_id x location x milestone

    choose the row with maximum prefix_end.

    This gives exactly one prefix endpoint per sentence per milestone.
    """

    unit_cols = [
        "model",
        "model_display",
        "split",
        "split_seed",
        "layer",
        "row_id",
        "location",
        "location_label",
        "milestone",
        "milestone_label",
        "milestone_order",
    ]

    candidates = candidates.copy()

    candidates = candidates.sort_values(
        unit_cols + ["prefix_end"]
    )

    selected = (
        candidates
        .groupby(
            unit_cols,
            as_index=False,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    return selected


# ==========================================================
# TEMPORAL EXCLUSIONS
# ==========================================================

def apply_temporal_exclusions(selected):
    """
    Exclude bad temporal datapoints.

    Need sentence-end prefix_end for each:
        model x split x layer x row_id x location

    Then:
        Amb > Disamb:
            exclude Disamb end if it is sentence-final.

        Disamb > Amb:
            exclude Ambiguous form if it is sentence-final.
    """

    selected = selected.copy()

    unit_cols = [
        "model",
        "model_display",
        "split",
        "split_seed",
        "layer",
        "row_id",
        "location",
        "location_label",
    ]

    sentence_end = (
        selected[
            selected["milestone"] == "sentence_end"
        ][
            unit_cols + ["prefix_end"]
        ]
        .rename(
            columns={
                "prefix_end": "sentence_end_prefix_end",
            }
        )
        .copy()
    )

    merged = selected.merge(
        sentence_end,
        on=unit_cols,
        how="left",
    )

    no_sentence_end = merged[
        merged["sentence_end_prefix_end"].isna()
    ].copy()

    if len(no_sentence_end) > 0:
        no_sentence_end["exclusion_reason"] = "no_sentence_end_for_unit"

    kept = merged[
        ~merged["sentence_end_prefix_end"].isna()
    ].copy()

    kept["sentence_end_prefix_end"] = kept["sentence_end_prefix_end"].astype(int)

    overlap_mask = (
        (
            kept["location"].astype(str).eq("after")
            &
            kept["milestone"].astype(str).eq("disamb_end")
            &
            (
                kept["prefix_end"].astype(int)
                >=
                kept["sentence_end_prefix_end"].astype(int)
            )
        )
        |
        (
            kept["location"].astype(str).eq("before")
            &
            kept["milestone"].astype(str).eq("ambiguity")
            &
            (
                kept["prefix_end"].astype(int)
                >=
                kept["sentence_end_prefix_end"].astype(int)
            )
        )
    )

    overlap_excluded = kept[overlap_mask].copy()

    if len(overlap_excluded) > 0:
        overlap_excluded["exclusion_reason"] = "milestone_same_as_or_after_sentence_end"

    final_kept = kept[~overlap_mask].copy()

    return final_kept, no_sentence_end, overlap_excluded


# ==========================================================
# ACCURACY
# ==========================================================

def compute_layer_split_milestone_accuracy(rows):
    """
    Compute balanced accuracy per:
        model x location x milestone x split x layer
    """

    group_cols = [
        "model",
        "model_display",
        "location",
        "location_label",
        "milestone",
        "milestone_label",
        "milestone_order",
        "split",
        "split_seed",
        "layer",
    ]

    out_rows = []

    for keys, group in rows.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))

        gold = group["gold_class"].astype(int)
        pred = group["predicted_class"].astype(int)

        row.update(
            {
                "balanced_accuracy": balanced_accuracy_or_nan(gold, pred),
                "n_sentences": int(group["row_id"].nunique()),
                "gold_class0_count": int((gold == 0).sum()),
                "gold_class1_count": int((gold == 1).sum()),
                "pred_class0_count": int((pred == 0).sum()),
                "pred_class1_count": int((pred == 1).sum()),
            }
        )

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def choose_topk_layers(layer_split_accuracy):
    """
    Choose top-k layers per model.

    Reference:
        sentence-end interpretation accuracy,
        pooled across location and split.

    This is intentionally simple and auditable.
    """

    sentence_end = layer_split_accuracy[
        layer_split_accuracy["milestone"] == "sentence_end"
    ].copy()

    layer_scores = (
        sentence_end
        .groupby(
            [
                "model",
                "model_display",
                "layer",
            ],
            as_index=False,
        )
        .agg(
            reference_accuracy=("balanced_accuracy", "mean"),
            reference_sd=("balanced_accuracy", "std"),
            reference_n_splits=("split", "nunique"),
            reference_mean_n_sentences=("n_sentences", "mean"),
        )
    )

    topk_rows = []

    for model_key, sub in layer_scores.groupby("model", sort=False):
        sub = (
            sub
            .sort_values(
                ["reference_accuracy", "layer"],
                ascending=[False, True],
            )
            .head(TOP_K_LAYERS)
            .copy()
        )

        sub["topk_rank"] = range(1, len(sub) + 1)

        topk_rows.append(sub)

    topk = pd.concat(topk_rows, ignore_index=True)

    return topk, layer_scores


def average_topk_layers(layer_split_accuracy, topk_layers):
    """
    Keep only top-k layers.

    Then:
        average layer accuracies inside each split and milestone
        average split-level values across splits
    """

    topk_keys = topk_layers[
        [
            "model",
            "layer",
            "topk_rank",
        ]
    ].copy()

    topk_layer_rows = layer_split_accuracy.merge(
        topk_keys,
        on=[
            "model",
            "layer",
        ],
        how="inner",
    )

    split_topk = (
        topk_layer_rows
        .groupby(
            [
                "model",
                "model_display",
                "location",
                "location_label",
                "milestone",
                "milestone_label",
                "milestone_order",
                "split",
                "split_seed",
            ],
            as_index=False,
        )
        .agg(
            topk_balanced_accuracy=("balanced_accuracy", "mean"),
            n_topk_layers_used=("layer", "nunique"),
            mean_n_sentences=("n_sentences", "mean"),
            min_n_sentences=("n_sentences", "min"),
        )
    )

    model_topk = (
        split_topk
        .groupby(
            [
                "model",
                "model_display",
                "location",
                "location_label",
                "milestone",
                "milestone_label",
                "milestone_order",
            ],
            as_index=False,
        )
        .agg(
            mean_topk_balanced_accuracy=("topk_balanced_accuracy", "mean"),
            sd_topk_balanced_accuracy=("topk_balanced_accuracy", "std"),
            n_splits=("split", "nunique"),
            mean_n_topk_layers_used=("n_topk_layers_used", "mean"),
            mean_n_sentences=("mean_n_sentences", "mean"),
            min_n_sentences=("min_n_sentences", "min"),
        )
    )

    return topk_layer_rows, split_topk, model_topk


# ==========================================================
# PLOT
# ==========================================================

def plot_temporal_topk(model_topk):
    """
    Two panels:
        Disamb > Amb
        Amb > Disamb

    Lines:
        one line per model
    """

    locations = ["before", "after"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
        sharey=True,
    )

    for ax, location in zip(axes, locations):
        panel = model_topk[
            model_topk["location"] == location
        ].copy()

        for model_key in MODEL_CONFIGS:
            sub = (
                panel[
                    panel["model"] == model_key
                ]
                .sort_values("milestone_order")
                .copy()
            )

            if len(sub) == 0:
                continue

            ax.plot(
                sub["milestone_order"],
                sub["mean_topk_balanced_accuracy"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=MODEL_CONFIGS[model_key]["display_name"],
            )

        ax.axhline(
            y=0.5,
            linestyle="--",
            linewidth=1,
        )

        ax.set_ylim(0.35, 1.00)

        if location == "before":
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(
                [
                    "Disamb end",
                    "Ambiguous form",
                    "Sentence end",
                ],
                rotation=20,
            )
        else:
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(
                [
                    "Ambiguous form",
                    "Disamb end",
                    "Sentence end",
                ],
                rotation=20,
            )

        ax.set_title(LOCATION_LABELS[location])
        ax.set_xlabel("Temporal milestone")
        ax.legend(fontsize=8)

    axes[0].set_ylabel(
        f"Mean balanced accuracy across top-{TOP_K_LAYERS} layers and splits"
    )

    fig.suptitle(
        "Endpoint interpretation decoding over temporal milestones\n"
        "Prefix-end representations"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "endpoint_interpretation_temporal_topk.png",
        dpi=200,
    )

    plt.savefig(
        OUTPUT_DIR / "endpoint_interpretation_temporal_topk.pdf",
    )

    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    all_loaded = []

    print()
    print("Loading endpoint-interpretation predictions...")

    for model_key, config in MODEL_CONFIGS.items():
        print(f"  {model_key}: {config['path']}")

        df = load_model(model_key, config)

        all_loaded.append(df)

    loaded = pd.concat(
        all_loaded,
        ignore_index=True,
        sort=False,
    )

    print()
    print("Building milestone candidates...")

    candidates = make_milestone_candidates(loaded)

    print(f"  candidate rows: {len(candidates)}")

    print()
    print("Selecting one prefix per milestone...")

    selected = select_one_prefix_per_milestone(candidates)

    print(f"  selected rows before temporal exclusions: {len(selected)}")

    print()
    print("Applying temporal exclusions...")

    used_rows, excluded_no_sentence_end, excluded_overlap = apply_temporal_exclusions(
        selected
    )

    print(f"  rows used: {len(used_rows)}")
    print(f"  rows excluded because sentence end was missing: {len(excluded_no_sentence_end)}")
    print(f"  rows excluded because milestone overlapped sentence end: {len(excluded_overlap)}")

    print()
    print("Computing layer-split-milestone accuracy...")

    layer_split_accuracy = compute_layer_split_milestone_accuracy(
        used_rows
    )

    print(f"  layer-split accuracy rows: {len(layer_split_accuracy)}")

    print()
    print(f"Choosing top-{TOP_K_LAYERS} layers per model...")

    topk_layers, all_layer_scores = choose_topk_layers(
        layer_split_accuracy
    )

    print(topk_layers[["model", "layer", "reference_accuracy", "topk_rank"]])

    print()
    print("Averaging across top-k layers and splits...")

    topk_layer_rows, split_topk, model_topk = average_topk_layers(
        layer_split_accuracy,
        topk_layers,
    )

    print()
    print("Saving CSVs...")

    loaded.to_csv(
        OUTPUT_DIR / "rows_loaded.csv",
        index=False,
    )

    candidates.to_csv(
        OUTPUT_DIR / "milestone_candidates_before_selection.csv",
        index=False,
    )

    selected.to_csv(
        OUTPUT_DIR / "milestone_rows_selected_before_exclusion.csv",
        index=False,
    )

    used_rows.to_csv(
        OUTPUT_DIR / "milestone_rows_used.csv",
        index=False,
    )

    excluded_no_sentence_end.to_csv(
        OUTPUT_DIR / "excluded_rows_no_sentence_end.csv",
        index=False,
    )

    excluded_overlap.to_csv(
        OUTPUT_DIR / "excluded_rows_temporal_overlap.csv",
        index=False,
    )

    layer_split_accuracy.to_csv(
        OUTPUT_DIR / "source_layer_split_milestone_accuracy.csv",
        index=False,
    )

    all_layer_scores.to_csv(
        OUTPUT_DIR / "all_layer_reference_scores.csv",
        index=False,
    )

    topk_layers.to_csv(
        OUTPUT_DIR / "topk_layers.csv",
        index=False,
    )

    topk_layer_rows.to_csv(
        OUTPUT_DIR / "source_topk_layer_rows.csv",
        index=False,
    )

    split_topk.to_csv(
        OUTPUT_DIR / "source_split_topk_milestone_accuracy.csv",
        index=False,
    )

    model_topk.to_csv(
        OUTPUT_DIR / "source_model_topk_milestone_accuracy.csv",
        index=False,
    )

    print()
    print("Plotting...")

    plot_temporal_topk(model_topk)

    print()
    print("Done.")
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    print("Main plot:")
    print(OUTPUT_DIR / "endpoint_interpretation_temporal_topk.png")
    print()
    print("Important CSVs:")
    print(OUTPUT_DIR / "milestone_rows_used.csv")
    print(OUTPUT_DIR / "excluded_rows_temporal_overlap.csv")
    print(OUTPUT_DIR / "topk_layers.csv")
    print(OUTPUT_DIR / "source_model_topk_milestone_accuracy.csv")


if __name__ == "__main__":
    main()