#!/usr/bin/env python3

"""
Plot 1:
Endpoint interpretation accuracy across layers.

Goal:
    One simple diagnostic plot.

Question:
    At sentence end, how accurately can the probe decode interpretation
    across layers?

Input:
    qwen_step3/endpoint_interpretation/probe_predictions.csv
    tgemma_step3/endpoint_interpretation/probe_predictions.csv
    tllama_step3/endpoint_interpretation/probe_predictions.csv

Important:
    We use probe_predictions.csv, not layer_summary.csv.

Why:
    probe_predictions.csv lets us explicitly filter to full_sentence rows.
    This avoids accidentally mixing stages.

Aggregation:
    model
    -> split
    -> layer
    -> balanced accuracy
    -> average across splits

Output:
    endpoint_interpretation_accuracy_plot/
        endpoint_interpretation_accuracy_across_layers.png
        endpoint_interpretation_accuracy_across_layers.pdf
        source_split_layer_accuracy.csv
        source_model_layer_accuracy.csv
        counts_full_sentence_rows.csv
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

OUTPUT_DIR = Path("endpoint_interpretation_accuracy_plot")


# ==========================================================
# HELPERS
# ==========================================================

def ensure_dir(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def require_columns(df, required_columns, context):
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{context} is missing required columns: {missing}"
        )


def normalize_bool(series):
    return (
        series
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def load_and_filter_model(model_key, config):
    """
    Load one model's endpoint interpretation predictions.

    Keep only full sentence rows.
    """

    path = Path(config["path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "split",
        "split_seed",
        "row_id",
        "layer",
        "stage",
        "is_full_sentence",
        "gold_class",
        "predicted_class",
        "vector_type",
        "probe_target",
    ]

    require_columns(
        df,
        required_columns,
        str(path),
    )

    # Validate that this is the right run.
    vector_types = sorted(
        df["vector_type"].dropna().astype(str).unique().tolist()
    )

    probe_targets = sorted(
        df["probe_target"].dropna().astype(str).unique().tolist()
    )

    if vector_types != ["endpoint"]:
        raise ValueError(
            f"{path} should be endpoint, but found vector_type={vector_types}"
        )

    if probe_targets != ["interpretation"]:
        raise ValueError(
            f"{path} should be interpretation, but found probe_target={probe_targets}"
        )

    # Primary full sentence filter:
    # Use actual stage value.
    full_sentence_mask = (
        df["stage"].astype(str)
        ==
        "full_sentence"
    )

    # Secondary check:
    # If is_full_sentence exists and is meaningful, require it too.
    if "is_full_sentence" in df.columns:
        full_sentence_mask = (
            full_sentence_mask
            &
            normalize_bool(df["is_full_sentence"])
        )

    kept = df[full_sentence_mask].copy()
    excluded = df[~full_sentence_mask].copy()

    if len(kept) == 0:
        raise ValueError(
            f"No full_sentence rows found for {model_key} at {path}"
        )

    kept["model"] = model_key
    kept["model_display"] = config["display_name"]

    excluded["model"] = model_key
    excluded["model_display"] = config["display_name"]
    excluded["exclusion_reason"] = "not_full_sentence"

    return kept, excluded


def compute_split_layer_accuracy(df):
    """
    Compute balanced accuracy separately for each:

        model x split x layer

    Relative layer is computed within each model, not globally.
    """

    rows = []

    group_cols = [
        "model",
        "model_display",
        "split",
        "split_seed",
        "layer",
    ]

    # Correct: model-specific number of layers.
    n_layers_by_model = (
        df
        .groupby("model")["layer"]
        .max()
        .astype(int)
        .add(1)
        .to_dict()
    )

    for keys, group in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))

        model = row["model"]
        layer = int(row["layer"])
        n_layers = int(n_layers_by_model[model])

        gold = group["gold_class"].astype(int)
        pred = group["predicted_class"].astype(int)

        gold_classes = sorted(gold.unique().tolist())

        if len(gold_classes) < 2:
            acc = float("nan")
        else:
            acc = balanced_accuracy_score(gold, pred)

        if n_layers <= 1:
            layer_relative = 0.0
        else:
            layer_relative = layer / float(n_layers - 1)

        row.update(
            {
                "n_layers_model": n_layers,
                "layer_relative": layer_relative,
                "balanced_accuracy": acc,
                "n_rows": int(len(group)),
                "n_sentences": int(group["row_id"].nunique()),
                "gold_class0_count": int((gold == 0).sum()),
                "gold_class1_count": int((gold == 1).sum()),
                "pred_class0_count": int((pred == 0).sum()),
                "pred_class1_count": int((pred == 1).sum()),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)
    
def average_across_splits(split_layer_df):
    """
    Average layer accuracy across repeated splits.
    """

    out = (
        split_layer_df
        .groupby(
            [
                "model",
                "model_display",
                "layer",
                "layer_relative",
            ],
            as_index=False,
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
            n_splits=(
                "split",
                "nunique",
            ),
            mean_n_rows=(
                "n_rows",
                "mean",
            ),
            mean_n_sentences=(
                "n_sentences",
                "mean",
            ),
            min_n_sentences=(
                "n_sentences",
                "min",
            ),
        )
    )

    return out


def make_counts(df):
    """
    Save simple counts after filtering to full sentence.
    """

    counts = (
        df
        .groupby(
            [
                "model",
                "model_display",
                "split",
                "layer",
            ],
            as_index=False,
        )
        .agg(
            n_rows=("row_id", "size"),
            n_sentences=("row_id", "nunique"),
            gold_class0_count=("gold_class", lambda x: int((x.astype(int) == 0).sum())),
            gold_class1_count=("gold_class", lambda x: int((x.astype(int) == 1).sum())),
            pred_class0_count=("predicted_class", lambda x: int((x.astype(int) == 0).sum())),
            pred_class1_count=("predicted_class", lambda x: int((x.astype(int) == 1).sum())),
        )
    )

    return counts


def plot_accuracy(model_layer_df):
    """
    Make one simple plot:
        x = relative layer
        y = mean balanced accuracy
        one line per model
    """

    plt.figure(figsize=(10, 6))

    for model_key, sub in model_layer_df.groupby("model", sort=False):
        sub = sub.sort_values("layer_relative")

        label = sub["model_display"].iloc[0]

        plt.plot(
            sub["layer_relative"],
            sub["mean_balanced_accuracy"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=label,
        )

    plt.axhline(
        y=0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Relative layer")
    plt.ylabel("Mean balanced accuracy across splits")
    plt.title("Endpoint interpretation accuracy across layers")
    plt.legend()
    plt.tight_layout()

    png_path = OUTPUT_DIR / "endpoint_interpretation_accuracy_across_layers.png"
    pdf_path = OUTPUT_DIR / "endpoint_interpretation_accuracy_across_layers.pdf"

    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    all_kept = []
    all_excluded = []

    print()
    print("Loading endpoint interpretation predictions...")

    for model_key, config in MODEL_CONFIGS.items():
        print(f"  {model_key}: {config['path']}")

        kept, excluded = load_and_filter_model(
            model_key,
            config,
        )

        all_kept.append(kept)
        all_excluded.append(excluded)

    full_sentence_df = pd.concat(
        all_kept,
        ignore_index=True,
    )

    excluded_df = pd.concat(
        all_excluded,
        ignore_index=True,
    )

    print()
    print("Computing split-layer balanced accuracy...")

    split_layer_accuracy = compute_split_layer_accuracy(
        full_sentence_df,
    )

    model_layer_accuracy = average_across_splits(
        split_layer_accuracy,
    )

    counts = make_counts(
        full_sentence_df,
    )

    print()
    print("Saving CSV files...")

    full_sentence_df.to_csv(
        OUTPUT_DIR / "full_sentence_rows_used.csv",
        index=False,
    )

    excluded_df.to_csv(
        OUTPUT_DIR / "excluded_not_full_sentence_rows.csv",
        index=False,
    )

    split_layer_accuracy.to_csv(
        OUTPUT_DIR / "source_split_layer_accuracy.csv",
        index=False,
    )

    model_layer_accuracy.to_csv(
        OUTPUT_DIR / "source_model_layer_accuracy.csv",
        index=False,
    )

    counts.to_csv(
        OUTPUT_DIR / "counts_full_sentence_rows.csv",
        index=False,
    )

    print()
    print("Plotting...")

    plot_accuracy(
        model_layer_accuracy,
    )

    print()
    print("Done.")
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    print("Main plot:")
    print(
        OUTPUT_DIR
        /
        "endpoint_interpretation_accuracy_across_layers.png"
    )
    print()
    print("Main source CSV:")
    print(
        OUTPUT_DIR
        /
        "source_model_layer_accuracy.csv"
    )
    print()


if __name__ == "__main__":
    main()