#!/usr/bin/env python3

import os
import re
import json
import argparse
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# HELPERS
# ============================================================

def clean_cell(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


# ============================================================
# FIND ALL WHOLE-WORD MATCHES
# ============================================================

def find_all_spans_ci(
    sentence,
    target
):
    """
    Returns ALL whole-word matches.

    Output:
        [
            (char_start, char_end),
            ...
        ]
    """

    if not target:
        return []

    pattern = r"\b{}\b".format(
        re.escape(target)
    )

    matches = list(
        re.finditer(
            pattern,
            sentence,
            flags=re.IGNORECASE
        )
    )

    results = []

    for m in matches:

        results.append(
            (
                m.start(),
                m.end()
            )
        )

    return results


# ============================================================
# CHARACTER -> TOKEN SPAN
# ============================================================

def char_to_token_span(
    offsets,
    char_start,
    char_end
):
    """
    Maps character span to inclusive token span.
    """

    token_indices = []

    for idx, (tok_s, tok_e) in enumerate(offsets):

        if tok_e <= char_start:
            continue

        if tok_s >= char_end:
            continue

        token_indices.append(idx)

    if len(token_indices) == 0:
        return None, None

    return (
        token_indices[0],
        token_indices[-1]
    )


# ============================================================
# TOKEN TEXT
# ============================================================

def span_token_text(
    tokens,
    start,
    end
):

    if start is None or end is None:
        return ""

    return " ".join(
        tokens[start:end + 1]
    )


# ============================================================
# PREFIX TEXT
# ============================================================

def prefix_text_from_offsets(
    sentence,
    offsets,
    prefix_end
):

    _, char_end = offsets[prefix_end]

    return sentence[:char_end]


# ============================================================
# LOCATION
# ============================================================

def compute_location(
    cue_tok_start,
    amb_tok_start
):

    if cue_tok_start < amb_tok_start:
        return "before"

    return "after"


# ============================================================
# STAGES
# ============================================================

def assign_stage(
    prefix_end,
    amb_start,
    amb_end,
    cue_start,
    cue_end,
    full_end
):

    if prefix_end == full_end:
        return "full_sentence"

    cue_before = (
        cue_start < amb_start
    )

    # ========================================================
    # BEFORE
    # ========================================================

    if cue_before:

        if prefix_end < cue_start:
            return "pre_ambiguity"

        if cue_start <= prefix_end <= cue_end:
            return "cue_onset"

        if cue_end < prefix_end < amb_start:
            return "post_cue_pre_ambiguity"

        if amb_start <= prefix_end <= amb_end:
            return "ambiguity_onset"

        if prefix_end > amb_end:
            return "post_cue"

    # ========================================================
    # AFTER
    # ========================================================

    else:

        if prefix_end < amb_start:
            return "pre_ambiguity"

        if amb_start <= prefix_end <= amb_end:
            return "ambiguity_onset"

        if amb_end < prefix_end < cue_start:
            return "post_ambiguity_pre_cue"

        if cue_start <= prefix_end <= cue_end:
            return "cue_onset"

        if prefix_end > cue_end:
            return "post_cue"

    return "unknown"


# ============================================================
# RESOLVE BEST ALIGNMENT
# ============================================================

def resolve_best_alignment(
    sentence,
    offsets,
    ambiguous_lemma,
    cue_start_lemma,
    cue_end_lemma,
    annotated_location
):
    """
    Finds best ambiguity/cue pairing.
    """

    # --------------------------------------------------------
    # ALL AMBIGUOUS MATCHES
    # --------------------------------------------------------

    ambiguous_matches = find_all_spans_ci(
        sentence,
        ambiguous_lemma
    )

    if len(ambiguous_matches) == 0:
        return None

    # --------------------------------------------------------
    # ALL CUE START MATCHES
    # --------------------------------------------------------

    cue_start_matches = find_all_spans_ci(
        sentence,
        cue_start_lemma
    )

    if len(cue_start_matches) == 0:
        return None

    # --------------------------------------------------------
    # BUILD FULL CUE SPANS
    # --------------------------------------------------------

    cue_spans = []

    # SAME START/END
    if cue_start_lemma.lower() == cue_end_lemma.lower():

        cue_spans = cue_start_matches

    else:

        cue_end_matches = find_all_spans_ci(
            sentence,
            cue_end_lemma
        )

        for cue_s_start, cue_s_end in cue_start_matches:

            for cue_e_start, cue_e_end in cue_end_matches:

                if cue_e_start >= cue_s_end:

                    cue_spans.append(
                        (
                            cue_s_start,
                            cue_e_end
                        )
                    )

    if len(cue_spans) == 0:
        return None

    # --------------------------------------------------------
    # BUILD CANDIDATES
    # --------------------------------------------------------

    candidates = []

    for amb_char_start, amb_char_end in ambiguous_matches:

        amb_tok_start, amb_tok_end = char_to_token_span(
            offsets,
            amb_char_start,
            amb_char_end
        )

        if amb_tok_start is None:
            continue

        for cue_char_start, cue_char_end in cue_spans:

            cue_tok_start, cue_tok_end = char_to_token_span(
                offsets,
                cue_char_start,
                cue_char_end
            )

            if cue_tok_start is None:
                continue

            computed_location = compute_location(
                cue_tok_start,
                amb_tok_start
            )

            # ------------------------------------------------
            # Must match annotation
            # ------------------------------------------------

            if computed_location != annotated_location:
                continue

            # ------------------------------------------------
            # Overlap
            # ------------------------------------------------

            overlap = not (
                cue_tok_end < amb_tok_start
                or
                cue_tok_start > amb_tok_end
            )

            # ------------------------------------------------
            # Distance
            # ------------------------------------------------

            token_distance = abs(
                cue_tok_start - amb_tok_start
            )

            candidates.append(
                {
                    "amb_char_start": amb_char_start,
                    "amb_char_end": amb_char_end,
                    "cue_char_start": cue_char_start,
                    "cue_char_end": cue_char_end,
                    "amb_tok_start": amb_tok_start,
                    "amb_tok_end": amb_tok_end,
                    "cue_tok_start": cue_tok_start,
                    "cue_tok_end": cue_tok_end,
                    "token_distance": token_distance,
                    "overlap": overlap,
                }
            )

    if len(candidates) == 0:
        return None

    # --------------------------------------------------------
    # Prefer non-overlapping
    # --------------------------------------------------------

    non_overlap = [
        c for c in candidates
        if not c["overlap"]
    ]

    if len(non_overlap) > 0:
        candidates = non_overlap

    # --------------------------------------------------------
    # Prefer closest
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x["token_distance"]
    )

    return candidates[0]


# ============================================================
# WRITE HUMAN AUDIT
# ============================================================

def write_human_audit(
    output_path,
    audit_blocks
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for block in audit_blocks:

            f.write(block)
            f.write("\n\n")


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

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # ========================================================
    # DATA
    # ========================================================

    print("Loading dataset...")

    df = pd.read_excel(
        args.data_file
    )

    aligned_rows = []

    prefix_rows = []

    token_rows = []

    skipped_rows = []

    mismatch_rows = []

    audit_blocks = []

    # ========================================================
    # MAIN LOOP
    # ========================================================

    for row_id, row in df.iterrows():

        try:

            sentence = clean_cell(
                row["sentence"]
            )

            label = clean_cell(
                row["label"]
            )

            cue_type = clean_cell(
                row["type"]
            )

            annotated_location = clean_cell(
                row["location"]
            )

            ambiguous_lemma = clean_cell(
                row["ambiguous_lemma"]
            )

            cue_start_lemma = clean_cell(
                row["cue_start_lemma"]
            )

            cue_end_lemma = clean_cell(
                row["cue_end_lemma"]
            )

            lemma_split = clean_cell(
                row.get(
                    "lemma_split",
                    ""
                )
            )

            morphological_parsing = clean_cell(
                row.get(
                    "morphological_parsing",
                    ""
                )
            )

            # ------------------------------------------------
            # TOKENIZE
            # ------------------------------------------------

            enc = tokenizer(
                sentence,
                return_offsets_mapping=True,
                add_special_tokens=False
            )

            token_ids = enc["input_ids"]

            offsets = enc["offset_mapping"]

            tokens = tokenizer.convert_ids_to_tokens(
                token_ids
            )

            n_tokens = len(tokens)

            # ------------------------------------------------
            # ALIGNMENT
            # ------------------------------------------------

            alignment = resolve_best_alignment(
                sentence=sentence,
                offsets=offsets,
                ambiguous_lemma=ambiguous_lemma,
                cue_start_lemma=cue_start_lemma,
                cue_end_lemma=cue_end_lemma,
                annotated_location=annotated_location
            )

            if alignment is None:

                skipped_rows.append(
                    {
                        "row_id": row_id,
                        "reason": "alignment_failed",
                        "sentence": sentence,
                    }
                )

                continue

            # ------------------------------------------------
            # EXTRACT
            # ------------------------------------------------

            amb_char_start = alignment["amb_char_start"]
            amb_char_end = alignment["amb_char_end"]

            cue_char_start = alignment["cue_char_start"]
            cue_char_end = alignment["cue_char_end"]

            amb_tok_start = alignment["amb_tok_start"]
            amb_tok_end = alignment["amb_tok_end"]

            cue_tok_start = alignment["cue_tok_start"]
            cue_tok_end = alignment["cue_tok_end"]

            overlap = alignment["overlap"]

            token_distance = alignment["token_distance"]

            # ------------------------------------------------
            # SANITY ASSERTIONS
            # ------------------------------------------------

            assert amb_char_start < amb_char_end
            assert cue_char_start < cue_char_end

            assert amb_tok_start <= amb_tok_end
            assert cue_tok_start <= cue_tok_end

            assert amb_tok_start >= 0
            assert cue_tok_start >= 0

            assert amb_tok_end < n_tokens
            assert cue_tok_end < n_tokens

            # ------------------------------------------------
            # COMPUTED LOCATION
            # ------------------------------------------------

            computed_location = compute_location(
                cue_tok_start,
                amb_tok_start
            )

            location_matches = (
                computed_location == annotated_location
            )

            if not location_matches:

                mismatch_rows.append(
                    {
                        "row_id": row_id,
                        "sentence": sentence,
                        "annotated_location": annotated_location,
                        "computed_location": computed_location,
                    }
                )

            # ------------------------------------------------
            # EXTRA FEATURES
            # ------------------------------------------------

            cue_length_tokens = (
                cue_tok_end - cue_tok_start + 1
            )

            ambiguity_length_tokens = (
                amb_tok_end - amb_tok_start + 1
            )

            cue_relative_position = (
                cue_tok_start - amb_tok_start
            )

            matched_ambiguous = sentence[
                amb_char_start:amb_char_end
            ]

            matched_cue = sentence[
                cue_char_start:cue_char_end
            ]

            amb_token_text = span_token_text(
                tokens,
                amb_tok_start,
                amb_tok_end
            )

            cue_token_text = span_token_text(
                tokens,
                cue_tok_start,
                cue_tok_end
            )

            # =================================================
            # ALIGNED ROW
            # =================================================

            aligned_rows.append(
                {
                    "row_id": row_id,
                    "sentence": sentence,
                    "label": label,
                    "type": cue_type,
                    "location": annotated_location,
                    "computed_location": computed_location,
                    "location_matches": location_matches,
                    "ambiguous_lemma": ambiguous_lemma,
                    "cue_start_lemma": cue_start_lemma,
                    "cue_end_lemma": cue_end_lemma,
                    "lemma_split": lemma_split,
                    "morphological_parsing": morphological_parsing,
                    "amb_char_start": amb_char_start,
                    "amb_char_end": amb_char_end,
                    "cue_char_start": cue_char_start,
                    "cue_char_end": cue_char_end,
                    "amb_tok_start": amb_tok_start,
                    "amb_tok_end": amb_tok_end,
                    "cue_tok_start": cue_tok_start,
                    "cue_tok_end": cue_tok_end,
                    "cue_length_tokens": cue_length_tokens,
                    "ambiguity_length_tokens": ambiguity_length_tokens,
                    "cue_relative_position": cue_relative_position,
                    "token_distance": token_distance,
                    "overlap": overlap,
                    "matched_ambiguous": matched_ambiguous,
                    "matched_cue": matched_cue,
                    "amb_token_text": amb_token_text,
                    "cue_token_text": cue_token_text,
                    "n_tokens": n_tokens,
                    "tokens_json": json.dumps(
                        tokens,
                        ensure_ascii=False
                    ),
                }
            )

            # =================================================
            # TOKEN AUDIT
            # =================================================

            for tok_idx, (
                tok,
                tok_id,
                (char_s, char_e)
            ) in enumerate(
                zip(
                    tokens,
                    token_ids,
                    offsets
                )
            ):

                token_rows.append(
                    {
                        "row_id": row_id,
                        "token_index": tok_idx,
                        "token": tok,
                        "token_id": tok_id,
                        "char_start": char_s,
                        "char_end": char_e,
                        "text_span": sentence[
                            char_s:char_e
                        ],
                        "in_ambiguous_span":
                            amb_tok_start <= tok_idx <= amb_tok_end,
                        "in_cue_span":
                            cue_tok_start <= tok_idx <= cue_tok_end,
                    }
                )

            # =================================================
            # PREFIX METADATA
            # =================================================

            stage_counter = Counter()

            for prefix_end in range(n_tokens):

                stage = assign_stage(
                    prefix_end=prefix_end,
                    amb_start=amb_tok_start,
                    amb_end=amb_tok_end,
                    cue_start=cue_tok_start,
                    cue_end=cue_tok_end,
                    full_end=n_tokens - 1
                )

                stage_counter[stage] += 1

                prefix_rows.append(
                    {
                        "row_id": row_id,
                        "prefix_end": prefix_end,
                        "prefix_text": prefix_text_from_offsets(
                            sentence,
                            offsets,
                            prefix_end
                        ),
                        "current_token": tokens[prefix_end],
                        "stage": stage,
                        "is_full_sentence":
                            prefix_end == n_tokens - 1,
                        "label": label,
                        "type": cue_type,
                        "location": annotated_location,
                        "computed_location": computed_location,
                        "cue_relative_position":
                            cue_relative_position,
                    }
                )

            # =================================================
            # HUMAN AUDIT
            # =================================================

            indexed_tokens = []

            for i, tok in enumerate(tokens):

                char_s, char_e = offsets[i]

                text_piece = sentence[
                    char_s:char_e
                ]

                indexed_tokens.append(
                    f"{i}: {tok} [{char_s}:{char_e}] '{text_piece}'"
                )

            prefix_lines = []

            for prefix_end in range(n_tokens):

                stage = assign_stage(
                    prefix_end=prefix_end,
                    amb_start=amb_tok_start,
                    amb_end=amb_tok_end,
                    cue_start=cue_tok_start,
                    cue_end=cue_tok_end,
                    full_end=n_tokens - 1
                )

                prefix_text = prefix_text_from_offsets(
                    sentence,
                    offsets,
                    prefix_end
                )

                prefix_lines.append(
                    f"prefix={prefix_end:02d} | "
                    f"stage={stage:<28} | "
                    f"text={prefix_text}"
                )

            audit_block = "\n".join(
                [
                    "=" * 80,
                    f"ROW {row_id}",
                    "",
                    f"LABEL: {label}",
                    f"CUE TYPE: {cue_type}",
                    f"ANNOTATED LOCATION: {annotated_location}",
                    f"COMPUTED LOCATION: {computed_location}",
                    f"LOCATION MATCHES: {location_matches}",
                    "",
                    f"OVERLAP: {overlap}",
                    f"TOKEN DISTANCE: {token_distance}",
                    f"CUE RELATIVE POSITION: {cue_relative_position}",
                    "",
                    f"AMBIGUOUS LEMMA: {ambiguous_lemma}",
                    f"MATCHED AMBIGUOUS: {matched_ambiguous}",
                    f"AMB TOKENS: {amb_tok_start}-{amb_tok_end}",
                    f"AMB TOKEN TEXT: {amb_token_text}",
                    "",
                    f"CUE START LEMMA: {cue_start_lemma}",
                    f"CUE END LEMMA: {cue_end_lemma}",
                    f"MATCHED CUE: {matched_cue}",
                    f"CUE TOKENS: {cue_tok_start}-{cue_tok_end}",
                    f"CUE TOKEN TEXT: {cue_token_text}",
                    "",
                    "STAGE COUNTS:",
                    json.dumps(
                        dict(stage_counter),
                        ensure_ascii=False
                    ),
                    "",
                    "TOKENS:",
                    "\n".join(indexed_tokens),
                    "",
                    "PREFIX TRAJECTORY:",
                    "\n".join(prefix_lines),
                ]
            )

            audit_blocks.append(
                audit_block
            )

        except Exception as e:

            skipped_rows.append(
                {
                    "row_id": row_id,
                    "reason": f"unexpected_error: {repr(e)}",
                }
            )

    # ========================================================
    # SAVE
    # ========================================================

    aligned_df = pd.DataFrame(
        aligned_rows
    )

    prefix_df = pd.DataFrame(
        prefix_rows
    )

    token_df = pd.DataFrame(
        token_rows
    )

    skipped_df = pd.DataFrame(
        skipped_rows
    )

    mismatch_df = pd.DataFrame(
        mismatch_rows
    )

    # --------------------------------------------------------
    # Stage summary
    # --------------------------------------------------------

    if not prefix_df.empty:

        stage_summary_df = (
            prefix_df.groupby(
                [
                    "label",
                    "type",
                    "location",
                    "stage"
                ]
            )
            .size()
            .reset_index(name="n_prefixes")
        )

    else:

        stage_summary_df = pd.DataFrame()

    # --------------------------------------------------------
    # Condition summary
    # --------------------------------------------------------

    if not aligned_df.empty:

        condition_summary_df = (
            aligned_df.groupby(
                [
                    "label",
                    "type",
                    "location",
                    "location_matches"
                ]
            )
            .size()
            .reset_index(name="n_examples")
        )

    else:

        condition_summary_df = pd.DataFrame()

    # ========================================================
    # WRITE FILES
    # ========================================================

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

    token_df.to_csv(
        os.path.join(
            args.output_dir,
            "token_audit.csv"
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

    mismatch_df.to_csv(
        os.path.join(
            args.output_dir,
            "location_mismatches.csv"
        ),
        index=False
    )

    stage_summary_df.to_csv(
        os.path.join(
            args.output_dir,
            "stage_summary.csv"
        ),
        index=False
    )

    condition_summary_df.to_csv(
        os.path.join(
            args.output_dir,
            "condition_summary.csv"
        ),
        index=False
    )

    write_human_audit(
        os.path.join(
            args.output_dir,
            "human_audit.txt"
        ),
        audit_blocks
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "model_name": args.model_name,
        "data_file": args.data_file,
        "n_examples_total": int(len(df)),
        "n_examples_kept": int(len(aligned_df)),
        "n_examples_skipped": int(len(skipped_df)),
        "n_prefixes": int(len(prefix_df)),
        "n_location_mismatches": int(len(mismatch_df)),
    }

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