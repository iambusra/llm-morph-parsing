#!/usr/bin/env python3

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


def load_json_list(x):

    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    return json.loads(x)


def get_device(model):

    return next(model.parameters()).device


def parse_torch_dtype(dtype_name):

    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "bfloat16":
        return torch.bfloat16

    if dtype_name == "float32":
        return torch.float32

    raise ValueError(
        f"Unsupported dtype: {dtype_name}"
    )


def parse_numpy_dtype(dtype_name):

    if dtype_name == "float16":
        return np.float16

    if dtype_name == "float32":
        return np.float32

    raise ValueError(
        f"Unsupported save dtype: {dtype_name}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model_and_tokenizer(
    model_name,
    torch_dtype_name
):

    print(f"\nLoading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    print(f"Loading model: {model_name}")

    torch_dtype = parse_torch_dtype(
        torch_dtype_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    return tokenizer, model


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_sentence(
    tokenizer,
    sentence
):

    enc = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=False
    )

    input_ids = enc["input_ids"][0]

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.tolist()
    )

    return input_ids, tokens


def validate_tokenization(
    row_id,
    step1_tokens,
    step2_tokens
):

    if len(step1_tokens) != len(step2_tokens):

        raise ValueError(
            f"Token count mismatch row_id={row_id}: "
            f"{len(step1_tokens)} vs "
            f"{len(step2_tokens)}"
        )

    mismatches = []

    for i, (a, b) in enumerate(
        zip(step1_tokens, step2_tokens)
    ):

        if a != b:

            mismatches.append(
                (i, a, b)
            )

    if len(mismatches) > 0:

        raise ValueError(
            f"Token mismatch row_id={row_id}. "
            f"First mismatches: {mismatches[:10]}"
        )


# ============================================================
# TRACKED REGION
# ============================================================

def infer_tracked_region(
    prefix_end,
    amb_start,
    amb_end,
    cue_start,
    cue_end
):

    # ambiguity token currently visible
    if amb_start <= prefix_end <= amb_end:
        return "ambiguity"

    # cue token currently visible
    if cue_start <= prefix_end <= cue_end:
        return "cue"

    # before both
    if prefix_end < amb_start and prefix_end < cue_start:
        return "pre_target"

    # after both
    if prefix_end > amb_end and prefix_end > cue_end:
        return "post_cue_or_post_ambiguity"

    # after ambiguity but before cue
    if prefix_end > amb_end and prefix_end < cue_start:
        return "post_ambiguity_pre_cue"

    # after cue but before ambiguity
    if prefix_end > cue_end and prefix_end < amb_start:
        return "post_cue_pre_ambiguity"

    return "other"


# ============================================================
# VECTOR EXTRACTION
# ============================================================

def extract_prefix_vectors(
    model,
    prefix_ids,
    prefix_end,
    amb_tok_start,
    amb_tok_end,
    save_dtype
):
    """
    Returns:
        endpoint_vectors
        ambiguity_vectors
        ambiguity_available

    Shapes:
        [n_layers, hidden_size]
    """

    device = get_device(model)

    prefix_ids = prefix_ids.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=prefix_ids,
            output_hidden_states=True,
            use_cache=False
        )

    # exclude embeddings
    hidden_states = outputs.hidden_states[1:]

    endpoint_vectors = []

    ambiguity_vectors = []

    ambiguity_available = (
        prefix_end >= amb_tok_end
    )

    for layer_h in hidden_states:

        # ====================================================
        # ENDPOINT VECTOR
        # ====================================================

        endpoint_vec = layer_h[
            0,
            -1,
            :
        ]

        endpoint_vectors.append(
            endpoint_vec.detach()
            .float()
            .cpu()
            .numpy()
        )

        # ====================================================
        # AMBIGUITY VECTOR
        # ====================================================

        if ambiguity_available:

            amb_vec = layer_h[
                0,
                amb_tok_start:amb_tok_end + 1,
                :
            ].mean(dim=0)

            ambiguity_vectors.append(
                amb_vec.detach()
                .float()
                .cpu()
                .numpy()
            )

        else:

            nan_vec = np.full(
                shape=(layer_h.shape[-1],),
                fill_value=np.nan,
                dtype=np.float32
            )

            ambiguity_vectors.append(
                nan_vec
            )

    endpoint_vectors = np.stack(
        endpoint_vectors,
        axis=0
    ).astype(save_dtype)

    ambiguity_vectors = np.stack(
        ambiguity_vectors,
        axis=0
    ).astype(save_dtype)

    return (
        endpoint_vectors,
        ambiguity_vectors,
        ambiguity_available
    )


# ============================================================
# VECTOR SUMMARY
# ============================================================

def summarize_vectors(
    name,
    arr
):

    finite_mask = np.isfinite(arr)

    out = {
        f"{name}_shape": list(arr.shape),
        f"{name}_dtype": str(arr.dtype),
        f"{name}_n_nan": int(
            np.isnan(arr).sum()
        ),
        f"{name}_n_inf": int(
            np.isinf(arr).sum()
        ),
        f"{name}_finite_fraction": float(
            finite_mask.mean()
        ),
    }

    if finite_mask.any():

        vals = arr[finite_mask]

        out.update(
            {
                f"{name}_finite_min":
                    float(vals.min()),
                f"{name}_finite_max":
                    float(vals.max()),
                f"{name}_finite_mean":
                    float(vals.mean()),
                f"{name}_finite_std":
                    float(vals.std()),
            }
        )

    return out


# ============================================================
# NORM SUMMARY
# ============================================================

def build_norm_summary(
    vectors,
    vector_type
):

    rows = []

    n_prefixes, n_layers, _ = vectors.shape

    for layer in range(n_layers):

        layer_vecs = vectors[:, layer, :]

        norms = np.linalg.norm(
            layer_vecs,
            axis=1
        )

        finite = np.isfinite(norms)

        if finite.any():

            rows.append(
                {
                    "vector_type": vector_type,
                    "layer": layer,
                    "n_prefixes": int(n_prefixes),
                    "n_finite": int(
                        finite.sum()
                    ),
                    "n_nan_or_inf": int(
                        (~finite).sum()
                    ),
                    "norm_mean": float(
                        norms[finite].mean()
                    ),
                    "norm_std": float(
                        norms[finite].std()
                    ),
                    "norm_min": float(
                        norms[finite].min()
                    ),
                    "norm_max": float(
                        norms[finite].max()
                    ),
                }
            )

        else:

            rows.append(
                {
                    "vector_type": vector_type,
                    "layer": layer,
                    "n_prefixes": int(n_prefixes),
                    "n_finite": 0,
                    "n_nan_or_inf": int(
                        (~finite).sum()
                    ),
                    "norm_mean": np.nan,
                    "norm_std": np.nan,
                    "norm_min": np.nan,
                    "norm_max": np.nan,
                }
            )

    return rows


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

    parser.add_argument(
        "--torch-dtype",
        choices=[
            "float16",
            "bfloat16",
            "float32"
        ],
        default="float16"
    )

    parser.add_argument(
        "--save-dtype",
        choices=[
            "float16",
            "float32"
        ],
        default="float32"
    )

    args = parser.parse_args()

    ensure_dir(
        args.output_dir
    )

    save_dtype = parse_numpy_dtype(
        args.save_dtype
    )

    # ========================================================
    # LOAD STEP1 FILES
    # ========================================================

    aligned_path = os.path.join(
        args.step1_dir,
        "aligned_examples.csv"
    )

    prefix_path = os.path.join(
        args.step1_dir,
        "prefix_metadata.csv"
    )

    aligned = pd.read_csv(
        aligned_path
    )

    prefix_meta = pd.read_csv(
        prefix_path
    )

    prefix_meta = prefix_meta.copy()

    if "global_prefix_id" not in prefix_meta.columns:

        prefix_meta["global_prefix_id"] = np.arange(
            len(prefix_meta)
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    tokenizer, model = load_model_and_tokenizer(
        args.model_name,
        args.torch_dtype
    )

    # ========================================================
    # OUTPUT CONTAINERS
    # ========================================================

    endpoint_chunks = []

    ambiguity_chunks = []

    metadata_rows = []

    audit_rows = []

    skipped_rows = []

    # ========================================================
    # MAIN EXTRACTION LOOP
    # ========================================================

    print("\nExtracting vectors...")

    for _, ex in tqdm(
        aligned.iterrows(),
        total=len(aligned)
    ):

        row_id = int(
            ex["row_id"]
        )

        sentence = ex["sentence"]

        try:

            ex_prefix = prefix_meta[
                prefix_meta["row_id"] == row_id
            ].copy()

            ex_prefix = ex_prefix.sort_values(
                "prefix_end"
            )

            if len(ex_prefix) == 0:

                skipped_rows.append(
                    {
                        "row_id": row_id,
                        "reason": "no_prefix_rows"
                    }
                )

                continue

            # ------------------------------------------------
            # TOKENIZE
            # ------------------------------------------------

            input_ids, step2_tokens = tokenize_sentence(
                tokenizer,
                sentence
            )

            step1_tokens = load_json_list(
                ex["tokens_json"]
            )

            validate_tokenization(
                row_id=row_id,
                step1_tokens=step1_tokens,
                step2_tokens=step2_tokens
            )

            # ------------------------------------------------
            # SPANS
            # ------------------------------------------------

            amb_tok_start = int(
                ex["amb_tok_start"]
            )

            amb_tok_end = int(
                ex["amb_tok_end"]
            )

            cue_tok_start = int(
                ex["cue_tok_start"]
            )

            cue_tok_end = int(
                ex["cue_tok_end"]
            )

            row_endpoint_vectors = []

            row_ambiguity_vectors = []

            # =================================================
            # PREFIX LOOP
            # =================================================

            for _, p in ex_prefix.iterrows():

                prefix_end = int(
                    p["prefix_end"]
                )

                prefix_ids = input_ids[
                    : prefix_end + 1
                ]

                (
                    endpoint_vecs,
                    ambiguity_vecs,
                    ambiguity_available
                ) = extract_prefix_vectors(
                    model=model,
                    prefix_ids=prefix_ids,
                    prefix_end=prefix_end,
                    amb_tok_start=amb_tok_start,
                    amb_tok_end=amb_tok_end,
                    save_dtype=save_dtype
                )

                row_endpoint_vectors.append(
                    endpoint_vecs
                )

                row_ambiguity_vectors.append(
                    ambiguity_vecs
                )

                tracked_region = infer_tracked_region(
                    prefix_end=prefix_end,
                    amb_start=amb_tok_start,
                    amb_end=amb_tok_end,
                    cue_start=cue_tok_start,
                    cue_end=cue_tok_end
                )

                tracked_token = step2_tokens[
                    prefix_end
                ]

                row_dict = p.to_dict()

                for col in aligned.columns:

                    if col not in row_dict:
                        row_dict[col] = ex[col]

                row_dict["tracked_token_index"] = prefix_end

                row_dict["tracked_token_text"] = tracked_token

                row_dict["tracked_region"] = tracked_region

                row_dict["ambiguity_vector_available"] = (
                    ambiguity_available
                )

                metadata_rows.append(
                    row_dict
                )

                audit_rows.append(
                    {
                        "global_prefix_id":
                            int(
                                row_dict[
                                    "global_prefix_id"
                                ]
                            ),
                        "row_id": row_id,
                        "prefix_end": prefix_end,
                        "stage": row_dict["stage"],
                        "tracked_token_index":
                            prefix_end,
                        "tracked_token_text":
                            tracked_token,
                        "tracked_region":
                            tracked_region,
                        "ambiguity_vector_available":
                            ambiguity_available,
                        "prefix_text":
                            row_dict.get(
                                "prefix_text",
                                ""
                            ),
                    }
                )

            # =================================================
            # STACK
            # =================================================

            row_endpoint_vectors = np.stack(
                row_endpoint_vectors,
                axis=0
            )

            row_ambiguity_vectors = np.stack(
                row_ambiguity_vectors,
                axis=0
            )

            endpoint_chunks.append(
                row_endpoint_vectors
            )

            ambiguity_chunks.append(
                row_ambiguity_vectors
            )

        except Exception as e:

            skipped_rows.append(
                {
                    "row_id": row_id,
                    "reason": repr(e)
                }
            )

    # ========================================================
    # CONCAT
    # ========================================================

    if len(endpoint_chunks) == 0:

        raise RuntimeError(
            "No vectors extracted."
        )

    endpoint_vectors = np.concatenate(
        endpoint_chunks,
        axis=0
    )

    ambiguity_vectors = np.concatenate(
        ambiguity_chunks,
        axis=0
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    skipped_df = pd.DataFrame(
        skipped_rows
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    if len(metadata_df) != endpoint_vectors.shape[0]:

        raise ValueError(
            "Endpoint vector / metadata mismatch."
        )

    if len(metadata_df) != ambiguity_vectors.shape[0]:

        raise ValueError(
            "Ambiguity vector / metadata mismatch."
        )

    # ========================================================
    # SUMMARIES
    # ========================================================

    endpoint_summary = summarize_vectors(
        "endpoint_vectors",
        endpoint_vectors
    )

    ambiguity_summary = summarize_vectors(
        "ambiguity_vectors",
        ambiguity_vectors
    )

    norm_rows = []

    norm_rows.extend(
        build_norm_summary(
            endpoint_vectors,
            "endpoint"
        )
    )

    norm_rows.extend(
        build_norm_summary(
            ambiguity_vectors,
            "ambiguity"
        )
    )

    norm_df = pd.DataFrame(
        norm_rows
    )

    tracked_region_summary = (
        audit_df.groupby(
            [
                "tracked_region",
                "ambiguity_vector_available"
            ]
        )
        .size()
        .reset_index(name="n_prefixes")
    )

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        os.path.join(
            args.output_dir,
            "endpoint_vectors.npy"
        ),
        endpoint_vectors
    )

    np.save(
        os.path.join(
            args.output_dir,
            "ambiguity_vectors.npy"
        ),
        ambiguity_vectors
    )

    # backward compatibility
    np.save(
        os.path.join(
            args.output_dir,
            "prefix_vectors.npy"
        ),
        endpoint_vectors
    )

    metadata_df.to_csv(
        os.path.join(
            args.output_dir,
            "prefix_metadata.csv"
        ),
        index=False
    )

    audit_df.to_csv(
        os.path.join(
            args.output_dir,
            "extraction_audit.csv"
        ),
        index=False
    )

    norm_df.to_csv(
        os.path.join(
            args.output_dir,
            "vector_norm_summary.csv"
        ),
        index=False
    )

    tracked_region_summary.to_csv(
        os.path.join(
            args.output_dir,
            "tracked_region_summary.csv"
        ),
        index=False
    )

    skipped_df.to_csv(
        os.path.join(
            args.output_dir,
            "skipped_rows.csv"
        ),
        index=False
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "model_name": args.model_name,
        "step1_dir": args.step1_dir,
        "torch_dtype": args.torch_dtype,
        "save_dtype": args.save_dtype,
        "n_examples_step1": int(
            len(aligned)
        ),
        "n_examples_skipped": int(
            len(skipped_df)
        ),
        "n_prefixes": int(
            endpoint_vectors.shape[0]
        ),
        "n_layers": int(
            endpoint_vectors.shape[1]
        ),
        "hidden_size": int(
            endpoint_vectors.shape[2]
        ),
    }

    summary.update(
        endpoint_summary
    )

    summary.update(
        ambiguity_summary
    )

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