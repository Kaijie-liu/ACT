"""Reconcile the two frozen mini-survey screens without changing the protocol."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ADJUDICATIONS: dict[int, dict[str, Any]] = {
    43: {
        "decision": "INCLUDE",
        "rationale": (
            "The public full text defines input-dependent MoE routing and a "
            "randomized-smoothing certificate for the final classifier, and "
            "reports a certified radius. Sparse methodological detail is an "
            "extraction finding, not an eligibility exclusion."
        ),
        "evidence": "Section 3.6-3.7 and reported radius 0.48 on p. 160",
        "source": "https://zenodo.org/records/18454317",
    },
    140: {
        "decision": "INCLUDE",
        "rationale": (
            "Theorem 4.1 states a bounded-perturbation guarantee for an "
            "input-dependent assembled classifier. The fact that the learned "
            "mixing network is evaluated only by attacks is retained as an "
            "instantiation/semantic-alignment finding rather than used to "
            "exclude the paper family."
        ),
        "evidence": "Theorem 4.1 and Sections 4-5",
        "source": "https://arxiv.org/abs/2301.12554",
    },
    154: {
        "decision": "INCLUDE",
        "rationale": (
            "The routed cache selects an expert only inside a bounded routing "
            "ball, and Theorem 3 supplies a uniform routed-output approximation "
            "error bound. It is retained under the protocol's broad "
            "robustness-or-safety wording and must be stratified as output "
            "approximation rather than classification robustness."
        ),
        "evidence": "Definitions 3-4, routing-radius construction, and Theorem 3",
        "source": "https://arxiv.org/abs/2605.04069",
    },
    178: {
        "decision": "INCLUDE",
        "rationale": (
            "The complete hard-routed policy, environment dynamics, bounded "
            "initial-state set, and finite-horizon safety property are encoded "
            "in SMT and evaluated. The protocol explicitly includes bounded "
            "input safety, not only adversarial classification radii."
        ),
        "evidence": "Section 5 SMT translation and Section 6 verification results",
        "source": "https://arxiv.org/abs/1906.06717",
    },
    193: {
        "decision": "DUPLICATE",
        "rationale": "Alternate preprint deposits have the same normalized title.",
        "merged_into_record_index": 194,
    },
    250: {
        "decision": "DUPLICATE",
        "rationale": "Versioned releases belong to the same repository artifact family.",
        "merged_into_record_index": 251,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _a_label(row: dict[str, Any]) -> str:
    if row["decision"] == "ADVANCE_FULLTEXT":
        return "INCLUDE"
    if row.get("primary_exclusion_code") == "E_DUPLICATE_VERSION":
        return "DUPLICATE"
    return "EXCLUDE"


def _agreement(left: list[str], right: list[str]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("review vectors must be non-empty and equally sized")
    labels = sorted(set(left) | set(right))
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / count
    expected = sum(
        (left.count(label) / count) * (right.count(label) / count)
        for label in labels
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    confusion = Counter(zip(left, right, strict=True))
    return {
        "records": count,
        "labels": labels,
        "raw_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "confusion": {
            f"A={a}|B={b}": value for (a, b), value in sorted(confusion.items())
        },
    }


def reconcile(a_path: Path, b_path: Path) -> dict[str, Any]:
    a_payload = json.loads(a_path.read_text(encoding="utf-8"))
    b_payload = json.loads(b_path.read_text(encoding="utf-8"))
    a_rows = a_payload["records"]
    b_rows = b_payload["decisions"]
    if len(a_rows) != len(b_rows):
        raise ValueError("reviewers did not receive the same frozen corpus")

    a_labels, b_labels = [], []
    disagreements: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for index, (a_row, b_row) in enumerate(zip(a_rows, b_rows, strict=True)):
        if b_row["record_index"] != index or a_row["title"] != b_row["title"]:
            raise ValueError(f"review corpus identity mismatch at record {index}")
        a_label, b_label = _a_label(a_row), b_row["decision"]
        a_labels.append(a_label)
        b_labels.append(b_label)
        if a_label == b_label:
            final = b_label
            resolution = "CONSENSUS"
        else:
            if index not in ADJUDICATIONS:
                raise ValueError(f"unadjudicated disagreement at record {index}")
            final = ADJUDICATIONS[index]["decision"]
            resolution = "WRITTEN_ADJUDICATION"
            disagreements.append(
                {
                    "record_index": index,
                    "title": b_row["title"],
                    "reviewer_a": a_label,
                    "reviewer_b": b_label,
                    **ADJUDICATIONS[index],
                }
            )
        final_rows.append(
            {
                "record_index": index,
                "title": b_row["title"],
                "decision": final,
                "resolution": resolution,
            }
        )

    observed_disagreements = {row["record_index"] for row in disagreements}
    if observed_disagreements != set(ADJUDICATIONS):
        raise ValueError("adjudication table contains stale or missing records")
    binary_a = ["INCLUDE" if value == "INCLUDE" else "NOT_INCLUDE" for value in a_labels]
    binary_b = ["INCLUDE" if value == "INCLUDE" else "NOT_INCLUDE" for value in b_labels]
    final_counts = Counter(row["decision"] for row in final_rows)
    included = [row for row in final_rows if row["decision"] == "INCLUDE"]
    return {
        "schema_version": 1,
        "status": "PARTIAL_SOURCE_EXECUTION_RECONCILED_NO_PREVALENCE_CLAIM",
        "cutoff_date": "2026-08-29",
        "independent_reviewers": 2,
        "reviewer_inputs": {
            "reviewer_a": {"path": str(a_path), "sha256": _sha256(a_path)},
            "reviewer_b": {"path": str(b_path), "sha256": _sha256(b_path)},
        },
        "three_class_agreement": _agreement(a_labels, b_labels),
        "binary_eligibility_agreement": _agreement(binary_a, binary_b),
        "written_adjudications": disagreements,
        "final_flow": {
            "retrieved_records": 480,
            "frozen_exact_title_families": 321,
            "additional_duplicate_versions": final_counts["DUPLICATE"],
            "unique_families": 321 - final_counts["DUPLICATE"],
            "title_abstract_excluded": 309,
            "full_text_assessed": 9,
            "full_text_excluded": 1,
            "included": final_counts["INCLUDE"],
        },
        "included_records": included,
        "final_decision_counts": dict(sorted(final_counts.items())),
        "authors_contacted": False,
        "claim_limit": (
            "Upstream retrieval remains partial for several preregistered "
            "sources and snowballing is incomplete. These counts describe only "
            "the frozen partial corpus and cannot support ecosystem prevalence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.reviewer_a, args.reviewer_b)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
