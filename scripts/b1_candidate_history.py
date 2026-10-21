"""Reconstruct the bounded B1/B4 history union; no model/data/solver imports.

Only the named metadata sources are read. This is not a global history scan,
clean-correct selection, or permission to run the prospective candidates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "base": ("docs/reuse_supervised_v1_freeze.json", "56eaf29df33a18c87acf79ed8b55fe7aba51e6cc3c5b2690e205cc2b004c4b3b"),
    "base_inventory": ("data/moe/results/reuse_supervised_selection_20260916_v1/exclusions.json", "a354be7569566d59dc04fc5654ccb3d6a523ca4bb64fc3854d4d065b9a1aaead"),
    "rter": ("act/pipeline/moe/results/baseline/icml2025_rt_er_b2_seed0_r3_audit.json", "5b744c5db3fd0ec45156aaafaa214cf8c28220c6e351fcf69097dc9de9cfadcd"),
    "conversion": ("data/moe/results/icml2025_rt_er_b2_seed0_r3/prepare.json", "bec773268cc6de903f00e0a259016bc32c5754448068faaab85346db059e524a"),
    "router": ("data/moe/results/icml2025_rt_er/router_init_figures_k20_20260830_r2/manifest.json", "8f51af419d2525bb400b43f324f15f515f367b6c0f8e3ce9566f206f587e0712"),
    "rome": ("configs/recent_moe/rome_multinorm_batch_r1.json", "40d6a348769aef5682c0bd4c4de3b470b64340685a9ceef56629897773b24920"),
    "metamoe": ("docs/metamoe_checked_small_freeze_20260923_r1.json", "c8e81880465050f11ce05b0cd0d0eb5adb3a04854117ec778798841197bcb262"),
    "dual_rs": ("docs/b1_dual_rs_history_20261021.json", "312d448c0c181d61b2a9ad43482dd39f51fe5541658b3c9860f38b0675cba0bf"),
    "multiseed_selection": ("act/pipeline/moe/configs/experiment1_multiseed_selection_r1.json", "86eef2005651401ebb3e2ddd827d9a8e5f713720375d5a387221601cf5611ae4"),
    "multiseed_audit": ("data/moe/results/experiment1_multiseed_selection_r1_audit.json", "839d586ffa388463934b566b38ceb0d3b09f7624980acad06583eee0a0461abb"),
    "multiseed_code": ("act/pipeline/moe/audit_experiment1_multiseed_selection.py", "3a3045f77dd8a0c952fdee2730d053ab06fc7dd2d4ea3b4e8228924d3453056b"),
    "confirmation": ("act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json", "b4167d8ab501df06570bfb49ad646263962c79736c2252f1398dc0499d3a7ee6"),
}
DATASET_HASH = "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identities(values):
    values = list(values)
    require(all(type(i) is int and 0 <= i < 10000 for i in values), "test identity range/type")
    return set(values)


def ranges(values):
    """Lossless, inclusive intervals, preserving singleton identities."""
    result = []
    for value in sorted(identities(values)):
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def audit():
    objects, bindings, total = {}, {}, 0
    for key, (name, expected) in SOURCES.items():
        path = ROOT / name
        size = path.stat().st_size
        require(size <= 2 * 2**20, "individual metadata cap")
        total += size
        require(total <= 4 * 2**20, "cumulative metadata cap")
        with path.open("rb") as stream:
            raw = stream.read(2 * 2**20 + 1)
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == expected,
                "saved metadata identity: " + name)
        bindings[key] = dict(path=name, sha256=expected, bytes=size)
        objects[key] = raw.decode() if name.endswith(".py") else json.loads(raw)

    base, inventory = objects["base"], objects["base_inventory"]
    old = identities(base["excluded_indices"])
    listed = identities(i for row in inventory for i in row["indices"])
    require(len(old) == 902 and listed <= old, "historical exclusion inventory")
    require(base["exclusion_inventory"]["sha256"] == bindings["base_inventory"]["sha256"],
            "base/inventory binding")
    require(base["dataset"]["sha256"] == DATASET_HASH, "base dataset")
    confirmation = objects["confirmation"]
    require(confirmation["dataset"]["raw_test_batch_sha256"] == DATASET_HASH
            and confirmation["dataset"]["split"] == "test", "confirmation dataset")
    require(DATASET_HASH in objects["rome"]["files"].values(), "RoME dataset binding")

    records, excluded = [], set()

    def add(key, field, values, meaning, dataset="CIFAR10"):
        ids = identities(values)
        records.append(dict(source=key, field=field, dataset=dataset, split="test",
                            usage=meaning, count=len(ids), inclusive_ranges=ranges(ids)))
        if dataset == "CIFAR10":
            excluded.update(ids)

    add("base", "excluded_indices", old, "DECLARED_HISTORICAL_EXCLUSIONS")
    add("base", "samples + scanned_clean_only", (r["dataset_index"] for name in
        ("samples", "scanned_clean_only") for r in base[name]), "LATER_SELECTION_EXPOSURE")
    require(objects["rter"]["cohort"]["ordered_test_indices"] == "0--999", "RT-ER cohort")
    add("rter", "cohort.ordered_test_indices", range(1000), "CLEAN_AND_ATTACK_COHORT")
    cross = objects["conversion"]["b3_cohort_cross_identity_agreement"]
    for field in ("indices", "full_clean_10000_prediction_difference_indices",
                  "full_clean_10000_route_difference_indices"):
        add("conversion", "b3_cohort_cross_identity_agreement." + field, cross[field],
            "EXPLICIT_CONVERSION_CHECK_OR_SELECTED_ANOMALY")
    add("router", "maximum_route_boundary.sample_index",
        [objects["router"]["maximum_route_boundary"]["sample_index"]], "SELECTED_ROUTER_EXTREMUM")
    add("rome", "indices", objects["rome"]["indices"], "FROZEN_ATTACK_QUEUE_INCLUDING_FAILURES")
    for dataset in ("CIFAR10", "MNIST"):
        add("metamoe", "scanned_prefix_lengths." + dataset,
            range(objects["metamoe"]["scanned_prefix_lengths"][dataset]),
            "AUTHOR_COMPARISON_SELECTION_PREFIX", dataset)
        add("metamoe", "requests[].{dataset,index}",
            (r["index"] for r in objects["metamoe"]["requests"] if r["dataset"] == dataset),
            "REGISTERED_AUTHOR_COMPARISON_REQUESTS", dataset)
    add("dual_rs", "control_exclusion.indices", objects["dual_rs"]["control_exclusion"]["indices"],
        "EXPLICIT_SELECTOR_TEST_CONTROL")
    selection, saved = objects["multiseed_selection"], objects["multiseed_audit"]
    selected = [r["dataset_index"] for r in selection["samples"]]
    require(saved["status"] == "PASS" and saved["manifest_sha256"] == bindings["multiseed_selection"]["sha256"]
            and selected == saved["recorded_indices"] == saved["reconstructed_indices"]
            and len(selected) == 40 and 1000 <= min(selected) <= max(selected) < 1256,
            "multiseed selection audit")
    # Manual source review: hash-bound code forwards the entire first256 batch
    # before its40-selection stop. This is not256 verification endpoints.
    add("multiseed_code", "lines87-107; bound PASS multiseed_audit", range(1000, 1256),
        "SOURCE_DERIVED_CLEAN_SELECTION_EXPOSURE")
    add("confirmation", "samples + scanned_clean_only", (r["dataset_index"] for name in
        ("samples", "scanned_clean_only") for r in confirmation[name]), "SEALED_CONFIRMATION_SELECTION")
    require({98, 4088, 4096, 4098, 4099} <= excluded, "explicit sealed identities retained")
    ordered = sorted(excluded)
    digest = hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()
    require(len(ordered) == 1855 and digest ==
            "83e0cbe69f444827054d73d90459522ebe4069d753ec891ddaf100675aa65d60", "reviewed union")
    return dict(
        schema="B1_B4_KNOWN_HISTORY_UNION_V1", status="PROSPECTIVE_NOT_EXECUTION_FREEZE",
        dataset=dict(name="CIFAR10", split="test", raw_test_batch_sha256=DATASET_HASH,
                     identity_basis="saved metadata; raw data not opened by this audit"),
        sources=bindings, provenance=records, metadata_bytes_read=total,
        base_inventory=dict(entries=len(inventory), union_count=len(listed),
                            base_additional_indices=sorted(old - listed)),
        excluded_count=len(ordered), excluded_inclusive_ranges=ranges(ordered),
        excluded_sorted_json_sha256=digest,
        explicitly_sealed_indices=[98, 4088, 4096, 4098, 4099],
        sealed_reason="current goal; exclusion is not permission to reopen other sealed confirmation rows",
        prospective_candidates=[i for i in range(10000) if i not in excluded][:12],
        candidate_rule="first12 ascending outside named history union; not clean-correct selected",
        models_from_saved_metadata=confirmation["models"],
        disclosures=dict(full_test_telemetry=True,
            dual_rs_fixed_recipe_test_evaluation=objects["dual_rs"]["fixed_recipe_aggregate_test_use"],
            meaning="previous evaluation is disclosed, not treated as verification of10000 endpoints"),
        limits=["Union covers named sources, not unrecorded human inspection or global history completeness.",
                "Prior family classification remains scoped to named directories; no prefix-wide inference.",
                "Dataset/index mapping and historical source review remain assumptions of this metadata audit.",
                "No checkpoint, dataset, previous candidate predictions or route bounds are loaded to select candidates.",
                "Does not authorize real selection, reopen sealed objects or alter pending resource decisions."],
        real_model_calls=0, native_solver_calls=0,
    )


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, separators=(",", ":")))
