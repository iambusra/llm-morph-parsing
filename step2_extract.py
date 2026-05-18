#!/usr/bin/env python3

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# HELPERS
# ============================================================

def load_model_and_tokenizer(model_name):

    print(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    return tokenizer, model


def tokenize_sentence(tokenizer, sentence):

    enc = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=False
    )

    return enc["input_ids"][0]


def extract_prefix_vectors_for_sentence(
    model,
    input_ids,
    prefix_ends
):

    vectors = []

    device = model.device

    for prefix_end in prefix_ends:

        prefix_ids = input_ids[: prefix_end + 1].unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(
                input_ids=prefix_ids,
                output_hidden_states=True,
                use_cache=False
            )

        # hidden_states includes embedding layer at index 0.
        # We exclude embeddings and keep transformer layers only.
        hidden_states = outputs.hidden_states[1:]

        endpoint_vectors = []

        for layer_h in hidden_states:

            # shape: [batch, seq_len, hidden_size]
            endpoint_vec = layer_h[0, -1, :].detach().cpu().numpy()

            endpoint_vectors.append(endpoint_vec)

        # shape: [n_layers, hidden_size]
        vectors.append(np.stack(endpoint_vectors, axis=0))

    # shape: [n_prefixes_for_sentence, n_layers, hidden_size]
    return np.stack(vectors, axis=0)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        required=True
    )

    parser.add_argument(
        "--step1-dir",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================
    # LOAD STEP 1 FILES
    # ========================================================

    aligned_path = os.path.join(
        args.step1_dir,
        "aligned_examples.csv"
    )

    prefix_path = os.path.join(
        args.step1_dir,
        "prefix_metadata.csv"
    )

    if not os.path.exists(aligned_path):
        raise FileNotFoundError(f"Missing: {aligned_path}")

    if not os.path.exists(prefix_path):
        raise FileNotFoundError(f"Missing: {prefix_path}")

    aligned = pd.read_csv(aligned_path)
    prefix_meta = pd.read_csv(prefix_path)

    # stable global prefix index
    prefix_meta = prefix_meta.copy()
    prefix_meta["global_prefix_id"] = np.arange(len(prefix_meta))

    # ========================================================
    # LOAD MODEL
    # ========================================================

    tokenizer, model = load_model_and_tokenizer(args.model_name)

    all_vectors = []
    enriched_prefix_rows = []

    # ========================================================
    # EXTRACT
    # ========================================================

    print("\nExtracting prefix endpoint vectors...")

    for _, ex in tqdm(
        aligned.iterrows(),
        total=len(aligned)
    ):

        row_id = ex["row_id"]
        sentence = ex["sentence"]

        ex_prefix = prefix_meta[
            prefix_meta["row_id"] == row_id
        ].copy()

        ex_prefix = ex_prefix.sort_values("prefix_end")

        if len(ex_prefix) == 0:
            continue

        input_ids = tokenize_sentence(
            tokenizer,
            sentence
        )

        expected_n_tokens = int(ex["n_tokens"])

        if len(input_ids) != expected_n_tokens:
            raise ValueError(
                f"Token count mismatch for row_id={row_id}: "
                f"Step1 n_tokens={expected_n_tokens}, "
                f"Step2 n_tokens={len(input_ids)}"
            )

        prefix_ends = ex_prefix["prefix_end"].tolist()

        vectors = extract_prefix_vectors_for_sentence(
            model=model,
            input_ids=input_ids,
            prefix_ends=prefix_ends
        )

        all_vectors.append(vectors)

        # ----------------------------------------------------
        # Add sentence-level metadata to prefix rows
        # ----------------------------------------------------

        for _, p in ex_prefix.iterrows():

            row_dict = p.to_dict()

            for col in aligned.columns:
                if col not in row_dict:
                    row_dict[col] = ex[col]

            enriched_prefix_rows.append(row_dict)

    # ========================================================
    # CONCAT
    # ========================================================

    prefix_vectors = np.concatenate(
        all_vectors,
        axis=0
    )

    enriched_prefix_meta = pd.DataFrame(
        enriched_prefix_rows
    )

    if len(enriched_prefix_meta) != prefix_vectors.shape[0]:
        raise ValueError(
            "Metadata/vector mismatch: "
            f"{len(enriched_prefix_meta)} metadata rows vs "
            f"{prefix_vectors.shape[0]} vectors"
        )

    # ========================================================
    # SAVE
    # ========================================================

    vectors_path = os.path.join(
        args.output_dir,
        "prefix_vectors.npy"
    )

    metadata_path = os.path.join(
        args.output_dir,
        "prefix_metadata.csv"
    )

    summary_path = os.path.join(
        args.output_dir,
        "summary.json"
    )

    np.save(vectors_path, prefix_vectors)

    enriched_prefix_meta.to_csv(
        metadata_path,
        index=False
    )

    summary = {
        "model_name": args.model_name,
        "step1_dir": args.step1_dir,
        "n_examples": int(len(aligned)),
        "n_prefixes": int(prefix_vectors.shape[0]),
        "n_layers": int(prefix_vectors.shape[1]),
        "hidden_size": int(prefix_vectors.shape[2]),
        "vectors_file": vectors_path,
        "metadata_file": metadata_path
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()