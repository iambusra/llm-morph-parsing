#!/usr/bin/env python3

import os
import json
import argparse

import pandas as pd

from transformers import AutoTokenizer


# ============================================================
# HELPERS
# ============================================================

def find_char_span(sentence, target):

    start = sentence.find(target)

    if start == -1:
        return None, None

    end = start + len(target)

    return start, end


def char_to_token_span(offsets, char_start, char_end):

    token_start = None
    token_end = None

    for idx, (s, e) in enumerate(offsets):

        if s <= char_start < e:
            token_start = idx

        if s < char_end <= e:
            token_end = idx

    return token_start, token_end


def assign_stage(
    prefix_end,
    amb_start,
    amb_end,
    cue_start,
    cue_end,
    full_end
):

    if prefix_end < amb_start:
        return "pre_ambiguity"

    if amb_start <= prefix_end < amb_end:
        return "ambiguity_onset"

    cue_before = cue_start < amb_start

    if cue_before:

        if prefix_end < cue_start:
            return "pre_cue"

        if cue_start <= prefix_end < cue_end:
            return "cue_onset"

        if cue_end <= prefix_end < amb_start:
            return "post_cue_pre_ambiguity"

    else:

        if amb_end <= prefix_end < cue_start:
            return "post_ambiguity_pre_cue"

        if cue_start <= prefix_end < cue_end:
            return "cue_onset"

    if prefix_end >= cue_end:

        if prefix_end == full_end:
            return "full_sentence"

        return "post_cue"

    return "unknown"


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
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--data-file",
        default="data/negnom_data_1to1.xlsx"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    print("Loading data...")

    df = pd.read_excel(args.data_file)

    aligned_rows = []
    prefix_rows = []
    debug_rows = []

    # ========================================================
    # LOOP
    # ========================================================

    for row_id, row in df.iterrows():

        sentence = str(row["sentence"])

        ambiguous = str(row["ambiguous_lemma"])

        cue_start_lemma = str(row["cue_start_lemma"])
        cue_end_lemma = str(row["cue_end_lemma"])

        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        enc = tokenizer(
            sentence,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        token_ids = enc["input_ids"]

        offsets = enc["offset_mapping"]

        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # ----------------------------------------------------
        # AMBIGUOUS SPAN
        # ----------------------------------------------------

        amb_char_start, amb_char_end = find_char_span(
            sentence,
            ambiguous
        )

        if amb_char_start is None:

            print(f"WARNING: ambiguous form not found: {row_id}")

            continue

        amb_tok_start, amb_tok_end = char_to_token_span(
            offsets,
            amb_char_start,
            amb_char_end
        )

        # ----------------------------------------------------
        # CUE SPAN
        # ----------------------------------------------------

        cue_char_start, _ = find_char_span(
            sentence,
            cue_start_lemma
        )

        _, cue_char_end = find_char_span(
            sentence,
            cue_end_lemma
        )

        if cue_char_start is None or cue_char_end is None:

            print(f"WARNING: cue not found: {row_id}")

            continue

        cue_char_end += len(cue_end_lemma)

        cue_tok_start, cue_tok_end = char_to_token_span(
            offsets,
            cue_char_start,
            cue_char_end
        )

        # ----------------------------------------------------
        # STORE EXAMPLE
        # ----------------------------------------------------

        aligned_rows.append({

            "row_id": row_id,

            "sentence": sentence,

            "label": row["label"],

            "type": row["type"],

            "location": row["location"],

            "ambiguous_lemma": ambiguous,

            "cue_start_lemma": cue_start_lemma,

            "cue_end_lemma": cue_end_lemma,

            "amb_char_start": amb_char_start,
            "amb_char_end": amb_char_end,

            "cue_char_start": cue_char_start,
            "cue_char_end": cue_char_end,

            "amb_tok_start": amb_tok_start,
            "amb_tok_end": amb_tok_end,

            "cue_tok_start": cue_tok_start,
            "cue_tok_end": cue_tok_end,

            "n_tokens": len(tokens)
        })

        # ----------------------------------------------------
        # PREFIXES
        # ----------------------------------------------------

        for prefix_end in range(len(tokens)):

            stage = assign_stage(
                prefix_end=prefix_end,
                amb_start=amb_tok_start,
                amb_end=amb_tok_end,
                cue_start=cue_tok_start,
                cue_end=cue_tok_end,
                full_end=len(tokens) - 1
            )

            prefix_rows.append({

                "row_id": row_id,

                "prefix_end": prefix_end,

                "stage": stage,

                "is_full_sentence":
                    prefix_end == len(tokens) - 1
            })

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        debug_rows.append({

            "row_id": row_id,

            "sentence": sentence,

            "tokens": tokens,

            "offsets": offsets,

            "ambiguous_lemma": ambiguous,

            "cue_start_lemma": cue_start_lemma,

            "cue_end_lemma": cue_end_lemma
        })

    # ========================================================
    # SAVE
    # ========================================================

    aligned_df = pd.DataFrame(aligned_rows)

    prefix_df = pd.DataFrame(prefix_rows)

    aligned_df.to_csv(
        os.path.join(
            args.output_dir,
            "aligned_examples.csv"
        ),
        index=False
    )

    prefix_df.to_csv(
        os.path.join(
            args.output_dir,
            "prefix_metadata.csv"
        ),
        index=False
    )

    with open(
        os.path.join(
            args.output_dir,
            "token_debug.jsonl"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        for row in debug_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    summary = {

        "model_name": args.model_name,

        "n_examples": len(aligned_df),

        "n_prefixes": len(prefix_df)
    }

    with open(
        os.path.join(
            args.output_dir,
            "summary.json"
        ),
        "w"
    ) as f:

        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(summary)


if __name__ == "__main__":
    main()