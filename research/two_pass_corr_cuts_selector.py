#!/usr/bin/env python3
"""Two-pass output-aware correlated pair-ReLU cut selector.

Pass 1: run the verifier WITHOUT correlated cuts; collect
  (a) per-query diagnostic JSONL (must contain `assert_y_true` and
      `worst_rival_ids` + `worst_rival_margins`),
  (b) ReLU trace dump (col_z per layer),
  (c) output HZ dump (Gc, Gb, c).

Selection: for each REQUESTED rival j, compute per-binary score
  score_i(j) = |Gb_out[j, i] - Gb_out[t, i]|
where t is the true label (read from diag). For each ReLU layer's
trace entry, restrict to that layer's binaries (cols `col_z`),
pick top-N local indices. Pairs within a layer = all (i, k)
combinations of the top-N picked.

Pass 2: verifier reads the emitted target JSON via
`ACT_HZ_CORR_PAIR_CUT_TARGET_FILE`. The verifier's cut emitter overrides
its width-score selection with the file's pair list for each ReLU layer.

NO fallback to t=-1: if diag has no `assert_y_true` AND env lacks
`ACT_HZ_TRUE_LABEL`, this script exits with status 2. (The audit
flagged the previous t=-1 fallback as a quality risk; loose-rival
scoring with wrong t gives misleading rankings.)

Usage:
  python research/two_pass_corr_cuts_selector.py \\
    --trace /path/to/relu_trace.json \\
    --diag  /path/to/pass1_diag.jsonl \\
    --out-hz /path/to/out_hz.npz \\
    --target-rivals 53            # OR omit to use diagnostic's worst rivals
    --top-binaries-per-layer 4 \\
    --global-pair-budget 12 \\
    --output /path/to/targets.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _parse_target_rivals(arg: str):
    if not arg:
        return None
    return sorted({int(s.strip()) for s in arg.split(",") if s.strip()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="RELU_TRACE dump JSON path")
    p.add_argument("--diag", required=True, help="Pass 1 diagnostic JSONL path")
    p.add_argument(
        "--out-hz", required=True,
        help="Output HZ dump .npz with keys c, Gc, Gb",
    )
    p.add_argument(
        "--target-rivals", default="",
        help=(
            "Comma-separated rival class indices to target (e.g. '53,70'). "
            "If empty, falls back to diagnostic's `worst_rival_ids` filtered "
            "to those with positive margin and limited by --top-per-rival."
        ),
    )
    p.add_argument(
        "--top-per-rival", type=int, default=4,
        help=(
            "Used only when --target-rivals is empty: how many worst rivals "
            "from the diagnostic to use."
        ),
    )
    p.add_argument(
        "--top-binaries-per-layer", type=int, default=4,
        help=(
            "Per ReLU layer, the number of highest-score binaries to "
            "form pairs from. Pairs = (n choose 2)."
        ),
    )
    p.add_argument(
        "--global-pair-budget", type=int, default=0,
        help=(
            "Optional global cap on emitted pairs across all layers. "
            "0 preserves the old behavior. When positive, candidate pairs "
            "are ranked by --pair-score and only the best N are emitted."
        ),
    )
    p.add_argument(
        "--pair-score",
        choices=("min", "product", "sum", "max"),
        default="product",
        help=(
            "How to rank pair candidates when --global-pair-budget is set. "
            "Scores are computed from the two selected neurons' output-aware "
            "single scores."
        ),
    )
    p.add_argument(
        "--encoding-filter",
        choices=("all", "triangle", "eq_lagr_v8"),
        default="all",
        help=(
            "Restrict target generation to a ReLU encoding. This is useful "
            "for dense-conv experiments where final eq_lagr layers dominate "
            "output scores and can mask earlier triangle-layer experiments."
        ),
    )
    p.add_argument("--output", required=True, help="Output targets.json path")
    args = p.parse_args()

    # ─── Load trace ───
    with open(args.trace) as f:
        trace = json.load(f)

    # ─── Load diag (one record per query in JSONL; take the last) ───
    diag_records = []
    with open(args.diag) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                diag_records.append(json.loads(line))
            except Exception:
                pass
    if not diag_records:
        print("ERROR: no diagnostic records", file=sys.stderr)
        sys.exit(2)
    diag = diag_records[-1]

    # Hard-require y_true. No fallback to t=-1.
    t = None
    if "assert_y_true" in diag and diag["assert_y_true"] is not None:
        t = int(diag["assert_y_true"])
    elif os.environ.get("ACT_HZ_TRUE_LABEL", ""):
        t = int(os.environ["ACT_HZ_TRUE_LABEL"])
    if t is None:
        print(
            "ERROR: diagnostic does not contain assert_y_true and "
            "ACT_HZ_TRUE_LABEL env is not set. Refusing to fall back to t=-1 "
            "which produces misleading pair rankings.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"Using t (true label) = {t}", file=sys.stderr)

    # ─── Resolve target rivals ───
    requested_rivals = _parse_target_rivals(args.target_rivals)
    if requested_rivals is not None:
        target_rivals = requested_rivals
        print(f"Explicit target rivals: {target_rivals}", file=sys.stderr)
    else:
        worst_ids = diag.get("worst_rival_ids", [])
        worst_margins = diag.get("worst_rival_margins", [])
        target_rivals = [
            int(j) for j, m in zip(worst_ids, worst_margins) if m > 0
        ][: args.top_per_rival]
        print(
            f"Diagnostic-derived target rivals: {target_rivals}",
            file=sys.stderr,
        )
    if not target_rivals:
        print("INFO: no rivals to target; emitting empty targets",
              file=sys.stderr)
        Path(args.output).write_text("{}")
        return

    # ─── Load out_hz ───
    out_hz = np.load(args.out_hz)
    Gb_out = out_hz["Gb"]
    Gc_out = out_hz["Gc"]
    n_out, nb_total = Gb_out.shape
    _, ng_total = Gc_out.shape
    print(
        f"out_hz Gb shape={Gb_out.shape}, Gc shape={Gc_out.shape}; n_out={n_out}",
        file=sys.stderr,
    )

    # ─── Per-binary AND per-continuous combined score: max over rivals ───
    # `bin_score` keyed by global binary col; `cont_score` keyed by global
    # continuous col. Both come from the same |Gx_out[j, .] - Gx_out[t, .]|
    # formula — Gb_out for binaries, Gc_out for continuous (used by
    # triangle ReLU layers which add eps generators, not binaries).
    bin_score = np.zeros(nb_total, dtype=np.float64)
    cont_score = np.zeros(ng_total, dtype=np.float64)
    for j in target_rivals:
        if not (0 <= t < n_out and 0 <= j < n_out):
            print(
                f"ERROR: t={t} or rival j={j} out of range [0, {n_out})",
                file=sys.stderr,
            )
            sys.exit(2)
        bin_score = np.maximum(bin_score, np.abs(Gb_out[j] - Gb_out[t]))
        cont_score = np.maximum(cont_score, np.abs(Gc_out[j] - Gc_out[t]))

    # ─── Map each gen col → (layer_counter, local_idx) ───
    # For v8 layers: trace entries carry `col_z` (binary col indices).
    # For triangle layers: trace entries carry `col_eps` (continuous col indices).
    bin_to_layer_local: "dict[int, tuple[int, int]]" = {}
    cont_to_layer_local: "dict[int, tuple[int, int]]" = {}
    for entry in trace:
        enc = entry.get("encoding") or "eq_lagr_v8"
        if args.encoding_filter != "all" and enc != args.encoding_filter:
            continue
        layer_id = int(entry["layer_count"])
        for local_idx, global_col in enumerate(entry.get("col_z", [])):
            bin_to_layer_local[int(global_col)] = (layer_id, int(local_idx))
        for local_idx, global_col in enumerate(entry.get("col_eps", [])):
            cont_to_layer_local[int(global_col)] = (layer_id, int(local_idx))
    print(
        f"Mapped {len(bin_to_layer_local)} binaries + "
        f"{len(cont_to_layer_local)} continuous "
        f"across {len(trace)} ReLU layers",
        file=sys.stderr,
    )

    # ─── Top-N per layer; form pairs.
    # Within a v8 layer, score by bin_score on col_z. Within a triangle
    # layer, score by cont_score on col_eps. Each layer's top-N local
    # indices generate (N choose 2) pairs.
    #
    # IMPORTANT (2026-06-01 fix): we DO NOT filter `s <= 0.0` columns. Filtering
    # out zero-score columns silently drops middle layers where the output-aware
    # score is ≈ 0 (eps generators that don't affect the t vs j Gx_out diff),
    # leaving them without cuts. The iid 8 BREAKTHROUGH originally CERTed
    # because a width-score fallback added cuts to all 7 ReLU layers — once the
    # whitelist mode was fixed, target-file mode dropped 4 of those layers and
    # CERT was lost. We now keep ALL columns; sorting by score descending still
    # picks the most-informative first, and the per-layer top-K guarantees
    # every trace layer gets cuts.
    layer_to_bins: "dict[int, list[tuple[int, float]]]" = {}
    for global_col, (layer_id, local_idx) in bin_to_layer_local.items():
        s = float(bin_score[global_col]) if global_col < nb_total else 0.0
        layer_to_bins.setdefault(layer_id, []).append((local_idx, s))
    for global_col, (layer_id, local_idx) in cont_to_layer_local.items():
        s = float(cont_score[global_col]) if global_col < ng_total else 0.0
        layer_to_bins.setdefault(layer_id, []).append((local_idx, s))

    targets: "dict[str, list[list[int]]]" = {}
    candidate_pairs: "list[tuple[float, int, list[int]]]" = []
    K = args.top_binaries_per_layer
    for layer_id, bins in layer_to_bins.items():
        bins_sorted = sorted(bins, key=lambda x: -x[1])[:K]
        local_idxs = [b[0] for b in bins_sorted]
        local_scores = {b[0]: float(b[1]) for b in bins_sorted}
        if len(local_idxs) < 2:
            continue
        pairs = []
        for i in range(len(local_idxs)):
            for k in range(i + 1, len(local_idxs)):
                a = int(local_idxs[i])
                b = int(local_idxs[k])
                sa = local_scores[a]
                sb = local_scores[b]
                if args.pair_score == "min":
                    ps = min(sa, sb)
                elif args.pair_score == "sum":
                    ps = sa + sb
                elif args.pair_score == "max":
                    ps = max(sa, sb)
                else:
                    ps = sa * sb
                pair = [a, b]
                pairs.append(pair)
                candidate_pairs.append((float(ps), int(layer_id), pair))
        if pairs:
            targets[str(layer_id)] = pairs

    if args.global_pair_budget and args.global_pair_budget > 0:
        targets = {}
        for _, layer_id, pair in sorted(
            candidate_pairs, key=lambda item: -item[0]
        )[: args.global_pair_budget]:
            targets.setdefault(str(layer_id), []).append(pair)

    print(
        f"Generated {sum(len(v) for v in targets.values())} pairs "
        f"across {len(targets)} layers",
        file=sys.stderr,
    )
    print(
        "Layer-pair counts:",
        {k: len(v) for k, v in targets.items()},
        file=sys.stderr,
    )

    Path(args.output).write_text(json.dumps(targets, indent=2))
    print(f"Targets written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
