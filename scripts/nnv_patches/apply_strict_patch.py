#!/usr/bin/env python3
"""Apply the NNV STRICT no-helper patch by string substitution.

Idempotent: checks for a sentinel comment and bails out if already patched.
Creates a .orig backup on first apply. Does NOT use unified-diff context
matching, so it survives upstream cosmetic edits as long as the anchor strings
remain.

See _PATCH_NOTES.md for the scientific-integrity rationale of each edit.
"""

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/data1/Kane/nnv/code/nnv/examples/Submission/VNN_COMP2025/run_vnncomp_instance.m"
)
SENTINEL = "STRICT-MODE PATCH (ACT paper"


# ---- edit 1: header banner + NNV_STRICT detection ----
ANCHOR_HEADER = "t = tic;\nstatus = 2; % unknown (to start with)"
REPLACE_HEADER = (
    "% STRICT-MODE PATCH (ACT paper, 2026-05-27):\n"
    "%   - Gates falsify_single (random + corner sampling) behind env NNV_STRICT_NO_HELPER=1.\n"
    "%   - Refuses cp-star reachability (statistical, not sound) under STRICT mode.\n"
    "%   - Default behavior (unset env) is identical to upstream competition entry.\n"
    "%   See /data1/Kane/ACT/scripts/nnv_patches/_PATCH_NOTES.md for the full rationale.\n"
    "\n"
    "t = tic;\n"
    "status = 2; % unknown (to start with)\n"
    "\n"
    "% STRICT: env-gated mode switch (defined ONCE, used in all subsequent guards)\n"
    "NNV_STRICT = ~isempty(getenv('NNV_STRICT_NO_HELPER')) && strcmp(getenv('NNV_STRICT_NO_HELPER'), '1');"
)


# ---- edit 2: cp-star refusal right after load_vnncomp_network ----
# The anchor is the line that pulls property fields out, which immediately
# follows the load_vnncomp_network call.
ANCHOR_AFTER_LOAD = "prop = property.prop; % output spec to verify"
REPLACE_AFTER_LOAD = (
    "prop = property.prop; % output spec to verify\n"
    "\n"
    "% STRICT: reject benchmarks whose only configured reach method is cp-star\n"
    "% (cp-star is conformal prediction, not sound formal verification).\n"
    "if NNV_STRICT\n"
    "    only_cpstar = ~isempty(reachOptionsList);\n"
    "    for strict_i = 1:length(reachOptionsList)\n"
    "        if ~strcmp(reachOptionsList{strict_i}.reachMethod, 'cp-star')\n"
    "            only_cpstar = false; break;\n"
    "        end\n"
    "    end\n"
    "    if only_cpstar\n"
    "        tTime = toc(t);\n"
    "        fid = fopen(outputfile, 'w');\n"
    "        fprintf(fid, 'unsupported_strict\\n');\n"
    "        fclose(fid);\n"
    "        disp('STRICT: only cp-star configured for this benchmark; refusing (not sound).');\n"
    "        status = 4;\n"
    "        return\n"
    "    end\n"
    "end"
)


# ---- edit 3: gate the entire falsification block ----
# We wrap the existing if/elseif chain that calls falsify_single. The anchor
# is the opening "Choose how to falsify" comment.
ANCHOR_FALSIFY_OPEN = (
    "% Choose how to falsify based on vnnlib file\n"
    "if ~isa(lb, \"cell\") && length(prop) == 1 % one input, one output "
)
REPLACE_FALSIFY_OPEN = (
    "% Choose how to falsify based on vnnlib file\n"
    "% STRICT: skip the entire falsification block — random sampling + lb/ub\n"
    "% corner evaluation in falsify_single are helpers (see _PATCH_NOTES.md).\n"
    "counterEx = nan;\n"
    "if NNV_STRICT\n"
    "    % falsification disabled; counterEx stays NaN, reachability decides.\n"
    "else\n"
    "if ~isa(lb, \"cell\") && length(prop) == 1 % one input, one output "
)


# The closing brace of the falsification if/elseif chain. We add 'end' so the
# outer "if NNV_STRICT ... else ... end" wrapper is balanced.
ANCHOR_FALSIFY_CLOSE = (
    "else\n"
    "    warning(\"Working on adding support to other vnnlib properties\");\n"
    "end\n"
    "\n"
    "cEX_time = toc(t);"
)
REPLACE_FALSIFY_CLOSE = (
    "else\n"
    "    warning(\"Working on adding support to other vnnlib properties\");\n"
    "end\n"
    "end % STRICT bypass of falsification block\n"
    "\n"
    "cEX_time = toc(t);"
)


# ---- edit 4: reject cp-star fallback paths (e.g. linearizenn matlab2nnv catch) ----
# Inserted right before the "if status == 2 && ~quickRun" reachability block.
ANCHOR_BEFORE_REACH = (
    "if status == 2 && ~quickRun % no counterexample found and supported for reachability (otherwise, skip step 3 and write results)"
)
REPLACE_BEFORE_REACH = (
    "% STRICT: per-benchmark code paths may pick cp-star as a *fallback* (e.g.\n"
    "% linearizenn does so when matlab2nnv fails on the loaded network). Refuse it.\n"
    "if NNV_STRICT && status == 2 && ~isempty(reachOptionsList) ...\n"
    "        && strcmp(reachOptionsList{1}.reachMethod, 'cp-star')\n"
    "    tTime = toc(t);\n"
    "    fid = fopen(outputfile, 'w');\n"
    "    fprintf(fid, 'unsupported_strict\\n');\n"
    "    fclose(fid);\n"
    "    disp('STRICT: cp-star fallback path triggered; refusing (not sound).');\n"
    "    status = 4;\n"
    "    return\n"
    "end\n"
    "\n"
    "if status == 2 && ~quickRun % no counterexample found and supported for reachability (otherwise, skip step 3 and write results)"
)


EDITS = [
    ("header banner + NNV_STRICT detect", ANCHOR_HEADER, REPLACE_HEADER),
    ("cp-star primary refusal",            ANCHOR_AFTER_LOAD, REPLACE_AFTER_LOAD),
    ("falsification block open guard",     ANCHOR_FALSIFY_OPEN, REPLACE_FALSIFY_OPEN),
    ("falsification block close guard",    ANCHOR_FALSIFY_CLOSE, REPLACE_FALSIFY_CLOSE),
    ("cp-star fallback refusal",           ANCHOR_BEFORE_REACH, REPLACE_BEFORE_REACH),
]


def apply(target: Path, dry_run: bool, verbose: bool) -> int:
    if not target.is_file():
        print(f"ERROR: target not found: {target}", file=sys.stderr)
        return 1

    text = target.read_text()

    if SENTINEL in text:
        print(f"[apply] {target.name}: STRICT patch already present — nothing to do.")
        return 0

    # Pre-check: every anchor must appear exactly once.
    missing = []
    ambiguous = []
    for name, anchor, _ in EDITS:
        count = text.count(anchor)
        if count == 0:
            missing.append((name, anchor))
        elif count > 1:
            ambiguous.append((name, count))
    if missing or ambiguous:
        print("ERROR: patch cannot be applied — anchors do not match upstream.", file=sys.stderr)
        for name, anchor in missing:
            print(f"  - MISSING anchor for '{name}'. Expected snippet:", file=sys.stderr)
            for line in anchor.splitlines():
                print(f"      | {line}", file=sys.stderr)
        for name, count in ambiguous:
            print(f"  - AMBIGUOUS anchor for '{name}' (found {count} times)", file=sys.stderr)
        return 2

    # Apply substitutions in order.
    patched = text
    for name, anchor, replacement in EDITS:
        patched = patched.replace(anchor, replacement, 1)
        if verbose:
            print(f"[apply] edit applied: {name}")

    if dry_run:
        print(f"[apply] --dry-run: {len(EDITS)} edits would apply cleanly to {target.name}.")
        return 0

    backup = target.with_suffix(target.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[apply] backup created: {backup}")

    target.write_text(patched)
    print(f"[apply] STRICT patch applied to {target} ({len(EDITS)} edits).")
    print(f"[apply] verify with: grep -n 'NNV_STRICT' {target}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                    help=f"path to run_vnncomp_instance.m (default: {DEFAULT_TARGET})")
    ap.add_argument("--dry-run", action="store_true", help="check anchors and exit without writing")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each edit as applied")
    args = ap.parse_args()
    sys.exit(apply(args.target, args.dry_run, args.verbose))


if __name__ == "__main__":
    main()
