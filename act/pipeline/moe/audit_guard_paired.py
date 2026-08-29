# ===- act/pipeline/moe/audit_guard_paired.py - Paired Guard Audit ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Paired support/no-support audit for confirmatory gate branches."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from scipy.stats import binomtest, fisher_exact, spearmanr

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256


SOLVED = {"certified", "falsified"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_guard_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    branches = [
        branch
        for row in rows
        for branch in (row.get("gate") or {}).get("branches", [])
    ]
    cells = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    runtime_differences: list[float] = []
    eliminated: list[int] = []
    transitions: list[int] = []
    for branch in branches:
        no_support = branch.get("matched_no_support_status") in SOLVED
        support = branch.get("branch_status") in SOLVED
        key = (
            "n11" if no_support and support
            else "n10" if no_support
            else "n01" if support
            else "n00"
        )
        cells[key] += 1
        runtime_differences.append(
            float(branch["solve_time"])
            - float(branch["matched_no_support_solve_seconds"])
        )
        eliminated.append(
            int(branch["guard_accounting"]["binary_eliminated"])
        )
        transitions.append(1 if key == "n01" else -1 if key == "n10" else 0)

    discordant = cells["n01"] + cells["n10"]
    mcnemar_p = (
        float(binomtest(cells["n01"], discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    fisher_table = [
        [
            sum(e > 0 and t == 1 for e, t in zip(eliminated, transitions)),
            sum(e > 0 and t != 1 for e, t in zip(eliminated, transitions)),
        ],
        [
            sum(e == 0 and t == 1 for e, t in zip(eliminated, transitions)),
            sum(e == 0 and t != 1 for e, t in zip(eliminated, transitions)),
        ],
    ]
    fisher = fisher_exact(fisher_table)
    spearman = spearmanr(eliminated, transitions)
    return {
        "branches": len(branches),
        "table": {
            "no_support_unsolved_support_unsolved_n00": cells["n00"],
            "no_support_unsolved_support_solved_n01": cells["n01"],
            "no_support_solved_support_unsolved_n10": cells["n10"],
            "no_support_solved_support_solved_n11": cells["n11"],
        },
        "support_only_solved": cells["n01"],
        "no_support_only_solved": cells["n10"],
        "net_solved_gain": cells["n01"] - cells["n10"],
        "exact_mcnemar_two_sided_p": mcnemar_p,
        "median_support_minus_no_support_solve_seconds": (
            statistics.median(runtime_differences)
            if runtime_differences else None
        ),
        "binary_elimination_transition_association": {
            "fisher_table_elimination_by_support_only": fisher_table,
            "fisher_exact_two_sided_p": float(fisher.pvalue),
            "fisher_odds_ratio": float(fisher.statistic),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "interpretation": "secondary association, not a causal runtime claim",
        },
    }


def audit(parent: Path, output: Path) -> dict[str, Any]:
    parent = _inside(parent, WRITE_ROOT)
    output = _inside(output, WRITE_ROOT)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    result = {
        "parent": str(parent),
        "parent_sha256": _sha256(parent),
        **paired_guard_statistics(_load_jsonl(parent)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.parent), Path(args.output)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
