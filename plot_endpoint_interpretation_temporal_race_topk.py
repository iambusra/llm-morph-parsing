#!/usr/bin/env python3

"""
Endpoint interpretation race plots over temporal milestones.

Goal:
    Show how the two interpretation probabilities compete over time:

        p_negation
        p_nominalizer

This is a race plot:
    Which interpretation has higher probability at each temporal milestone?

Important:
    This script uses endpoint_interpretation only.

Why:
    We want probabilities at specific prefixes:
        - ambiguous-form prefix
        - end of disambiguation zone
        - sentence end

    Therefore, the representation is the current prefix endpoint.

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

    1. Compute interpretation accuracy per layer first.
    2. Select top-k layers per model using sentence-end interpretation accuracy.
    3. Keep top-k layers.
    4. Compute p_negation and p_nominalizer per:
           model x gold_label x location x milestone x split x layer
    5. Average across top-k layers within split.
    6. Average across splits.

Plot layout:
    One figure per model.

    Rows:
        gold label
            gold = negation
            gold = nominalizer

    Columns:
        cue position
            Disamb > Amb
            Amb > Disamb

    Lines:
        p_negation
        p_nominalizer

Outputs:
    endpoint_interpretation_temporal_race_topk_plot/
        race_qwen.png
        race_tgemma.png
        race_tllama.png

        race_qwen.pdf
        race_tgemma.pdf
        race_tllama.pdf

        rows_loaded.csv
        milestone_candidates_before_selection.csv
        milestone_rows_selected_before_exclusion.csv
        milestone_rows_used.csv
        excluded_rows_no_sentence_end.csv
        excluded_rows_temporal_overlap.csv

        source_layer_split_accuracy_for_topk.csv
        all_layer_reference_scores.csv
        topk_layers.csv
        source_layer_split_race.csv
        source_topk_layer_race_rows.csv
        source_split_topk_race.csv
        source_model_topk_race.csv
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

PROBABILITY_LABELS = {
    "mean_topk_p_negation": "p_negation",
    "mean_topk_p_nominalizer": "p_nominalizer",
}

TOP_K_LAYERS = 5

OUTPUT_DIR = Path("endpoint_interpretation_temporal_race_topk_plot")


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
        "p_class0",
        "p_class1",
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
    df["gold_class"] = df["gold_class"].astype(int)

    # Step 3 interpretation encoding:
    #   class 0 = negation
    #   class 1 = nominalizer
    df["p_negation"] = df["p_class0"].astype(float)
    df["p_nominalizer"] = df["p_class1"].astype(float)

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

    Disamb end is NOT cue_onset.

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
# TOP-K LAYER SELECTION
# ==========================================================

def compute_layer_split_accuracy_for_topk(rows):
    """
    Compute interpretation accuracy per:
        model x location x milestone x split x layer

    This is used only for selecting top-k layers.
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

    This intentionally matches the temporal accuracy script.
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


# ==========================================================
# RACE PROBABILITY AGGREGATION
# ==========================================================

def compute_layer_split_race(rows):
    """
    Compute p_negation and p_nominalizer per:
        model x gold_label x location x milestone x split x layer

    Important:
        This averages over sentence rows within each split/layer/milestone.
    """

    group_cols = [
        "model",
        "model_display",
        "gold_label",
        "gold_class",
        "location",
        "location_label",
        "milestone",
        "milestone_label",
        "milestone_order",
        "split",
        "split_seed",
        "layer",
    ]

    out = (
        rows
        .groupby(
            group_cols,
            as_index=False,
            sort=False,
        )
        .agg(
            mean_p_negation=("p_negation", "mean"),
            sd_p_negation=("p_negation", "std"),
            mean_p_nominalizer=("p_nominalizer", "mean"),
            sd_p_nominalizer=("p_nominalizer", "std"),
            n_sentences=("row_id", "nunique"),
            n_rows=("row_id", "size"),
        )
    )

    return out


def average_race_topk_layers(layer_split_race, topk_layers):
    """
    Keep only top-k layers.

    Then:
        1. average probabilities across top-k layers within split/milestone
        2. average across splits
    """

    topk_keys = topk_layers[
        [
            "model",
            "layer",
            "topk_rank",
        ]
    ].copy()

    topk_layer_rows = layer_split_race.merge(
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
                "gold_label",
                "gold_class",
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
            topk_mean_p_negation=("mean_p_negation", "mean"),
            topk_mean_p_nominalizer=("mean_p_nominalizer", "mean"),
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
                "gold_label",
                "gold_class",
                "location",
                "location_label",
                "milestone",
                "milestone_label",
                "milestone_order",
            ],
            as_index=False,
        )
        .agg(
            mean_topk_p_negation=("topk_mean_p_negation", "mean"),
            sd_topk_p_negation=("topk_mean_p_negation", "std"),
            mean_topk_p_nominalizer=("topk_mean_p_nominalizer", "mean"),
            sd_topk_p_nominalizer=("topk_mean_p_nominalizer", "std"),
            n_splits=("split", "nunique"),
            mean_n_topk_layers_used=("n_topk_layers_used", "mean"),
            mean_n_sentences=("mean_n_sentences", "mean"),
            min_n_sentences=("min_n_sentences", "min"),
        )
    )

    return topk_layer_rows, split_topk, model_topk


# ==========================================================
# PLOTTING
# ==========================================================

def plot_race_for_model(model_key, model_topk):
    """
    One figure per model.

    Rows:
        gold label

    Columns:
        Disamb > Amb
        Amb > Disamb

    Lines:
        p_negation
        p_nominalizer
    """

    model_df = model_topk[
        model_topk["model"] == model_key
    ].copy()

    if len(model_df) == 0:
        return

    gold_table = (
        model_df[
            [
                "gold_label",
                "gold_class",
            ]
        ]
        .drop_duplicates()
        .sort_values("gold_class")
    )

    gold_labels = gold_table["gold_label"].tolist()

    locations = ["before", "after"]

    fig, axes = plt.subplots(
        nrows=len(gold_labels),
        ncols=2,
        figsize=(14, 5 * len(gold_labels)),
        sharey=True,
    )

    if len(gold_labels) == 1:
        axes = [axes]

    for row_idx, gold_label in enumerate(gold_labels):
        for col_idx, location in enumerate(locations):
            ax = axes[row_idx][col_idx]

            panel = model_df[
                (model_df["gold_label"] == gold_label)
                &
                (model_df["location"] == location)
            ].copy()

            panel = panel.sort_values("milestone_order")

            if len(panel) == 0:
                continue

            ax.plot(
                panel["milestone_order"],
                panel["mean_topk_p_negation"],
                marker="o",
                linewidth=2,
                markersize=6,
                label="p_negation",
            )

            ax.plot(
                panel["milestone_order"],
                panel["mean_topk_p_nominalizer"],
                marker="o",
                linewidth=2,
                markersize=6,
                label="p_nominalizer",
            )

            ax.axhline(
                y=0.5,
                linestyle="--",
                linewidth=1,
            )

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

            ax.set_title(f"Gold: {gold_label} | {LOCATION_LABELS[location]}")
            ax.set_xlabel("Temporal milestone")
            ax.set_ylim(0.0, 1.0)
            ax.legend(fontsize=8)

            if col_idx == 0:
                ax.set_ylabel(
                    f"Mean probability across top-{TOP_K_LAYERS} layers and splits"
                )

    fig.suptitle(
        f"Endpoint interpretation race plot over temporal milestones\n"
        f"{MODEL_CONFIGS[model_key]['display_name']}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"race_{model_key}.png",
        dpi=200,
    )

    plt.savefig(
        OUTPUT_DIR / f"race_{model_key}.pdf",
    )

    plt.close()


def plot_all_models(model_topk):
    for model_key in MODEL_CONFIGS:
        plot_race_for_model(
            model_key=model_key,
            model_topk=model_topk,
        )


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
    print("Computing layer-split accuracy for top-k layer selection...")

    layer_split_accuracy = compute_layer_split_accuracy_for_topk(
        used_rows
    )

    print(f"  layer-split accuracy rows: {len(layer_split_accuracy)}")

    print()
    print(f"Choosing top-{TOP_K_LAYERS} layers per model using sentence-end accuracy...")

    topk_layers, all_layer_scores = choose_topk_layers(
        layer_split_accuracy
    )

    print(topk_layers[["model", "layer", "reference_accuracy", "topk_rank"]])

    print()
    print("Computing layer-split race probabilities...")

    layer_split_race = compute_layer_split_race(
        used_rows
    )

    print(f"  layer-split race rows: {len(layer_split_race)}")

    print()
    print("Averaging race probabilities across top-k layers and splits...")

    topk_layer_race_rows, split_topk_race, model_topk_race = (
        average_race_topk_layers(
            layer_split_race,
            topk_layers,
        )
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
        OUTPUT_DIR / "source_layer_split_accuracy_for_topk.csv",
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

    layer_split_race.to_csv(
        OUTPUT_DIR / "source_layer_split_race.csv",
        index=False,
    )

    topk_layer_race_rows.to_csv(
        OUTPUT_DIR / "source_topk_layer_race_rows.csv",
        index=False,
    )

    split_topk_race.to_csv(
        OUTPUT_DIR / "source_split_topk_race.csv",
        index=False,
    )

    model_topk_race.to_csv(
        OUTPUT_DIR / "source_model_topk_race.csv",
        index=False,
    )

    print()
    print("Plotting race plots...")

    plot_all_models(model_topk_race)

    print()
    print("Done.")
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    print("Main plots:")
    for model_key in MODEL_CONFIGS:
        print(OUTPUT_DIR / f"race_{model_key}.png")
    print()
    print("Important CSVs:")
    print(OUTPUT_DIR / "milestone_rows_used.csv")
    print(OUTPUT_DIR / "excluded_rows_temporal_overlap.csv")
    print(OUTPUT_DIR / "topk_layers.csv")
    print(OUTPUT_DIR / "source_model_topk_race.csv")


if __name__ == "__main__":
    main()