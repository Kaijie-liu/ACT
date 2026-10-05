#!/usr/bin/env python3
"""Freeze the formal 543-row unsolved structure partition.

The two verdict CSVs and all benchmark assets are read-only inputs.  The
generator verifies the complete five-file composite authority before parsing
any verdict, derives cohorts only from ONNX graph structure, and publishes
with atomic exclusive creation below this isolated experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import subprocess
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping

import onnx


FORMAT_VERSION = "neural_hz_formal_unsolved_structure_partition_v1"
SIGNATURE_VERSION = "onnx_operator_signature_v1"
ROW_IDENTITY_VERSION = "formal_composite_row_identity_v1"
CLASSIFICATION_RULE_VERSION = "formal_structure_af_v1"
EXPERIMENT_ROOT = Path(__file__).resolve().parent

DEFAULT_HYZOR_ROOT = Path("/data1/Kane/HyZor")
DEFAULT_BENCHMARK_ROOT = Path(
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
)

EXPECTED_BENCHMARK_GIT_COMMIT = "8b7b811b78ce6a329dc96f04ae6652da3c245948"
EXPECTED_INSTANCES_CSV_HASHES = {
    "safenlp_2024": "2c6a1a33ef085eabb0431595214aaedd5357e52fbbf86b564418f5accec0b15d",
    "sat_relu": "90bbb03e841e7e3633306be70be79c6f7bf89da42a22988d418e51e966535423",
    "malbeware": "35e9959231643ce518440b5700515401d695af546747a898f6e4df0cdb7014b0",
    "metaroom_2023": "ec34936f5c305b7520df5db0ed41d5d4784108c172e6f9b3404c4e7af1b6bfde",
    "acasxu_2023": "741436aeaaa2c4f0d8673218d656b49af7f4e9e2fb57f31f867d4a1c8aa39fab",
    "linearizenn_2024": "3db2928f7545df8ef276670f448c93cf692a0742a4a4223aeaf617a4ce20caf9",
    "relusplitter": "b05fcf943f9b4e4b883ff76a35af2f396a611f985b6045e60052501e498b8644",
    "dist_shift_2023": "e39285afd439f6add18e35879e7b3b22f9a0726ac6a56fa781855679dffe9ba4",
    "tllverifybench_2023": "61e2d6104d99240cb2a3314be2ca79111863fcce9765cffa139856f64c30ca41",
    "cgan_2023": "80dc39ac24fbaf6a01c3deb5c80459592bb5e4e7dcd48f0c70314f9f2b69ae7c",
    "cersyve": "cfeec4f50ebdf30226b336d6e818b8378dd61914dd989d296f97e5355dc02c4f",
    "cora_2024": "06703d1fcfa0bc42f3fa8a5118b792f2219e38c89ddd47a6dd6692a4e8624645",
    "vit_2023": "c6a6326577c2bf5b5d860f19ee4b04c6bfd075050341cbbe5a8c9f05b2576fba",
}

AUTHORITY_FILES = {
    "formal_verdict_table": {
        "relative_path": "VMCAI_2027___Kaijie_Guanqin/tables/verdict.tex",
        "sha256": "de13942115a2a1ba79eb3af82a4c435a4919492c044d174cfdef6be1c41429d0",
        "role": "per-family formal verdict vector",
    },
    "composite_generator": {
        "relative_path": "VIT13_PAPER_UPDATE_20260828/generate_figures.py",
        "sha256": "971f8c8c31a40a5c2b732338d9219572c9350e05b61ff4a496a3d4efe15222b0",
        "role": "composite headline generator asserting 1,870",
    },
    "twelve_family_overlay": {
        "relative_path": "DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv",
        "sha256": "05ab7e4b09c3285bb8d5f1d09ae2876fee7a5933b68c0a6af54b8e6bf951544a",
        "role": "2,213-row strict 12-family source",
    },
    "vit_strict_rows": {
        "relative_path": "vit_hz_legacy1_100s_20260826/consolidated_strict_100s.csv",
        "sha256": "8d52519c65345c04639a2abc345be462c3e62f8edc810215151013321431c8c9",
        "role": "200-row strict-wall ViT source",
    },
    "composite_summary": {
        "relative_path": "VIT13_PAPER_UPDATE_20260828/summary.json",
        "sha256": "90b7f405b140add3c9461cb22cd5e483decc989261a36df45026e968c0921934",
        "role": "aggregate formal summary",
    },
}

FAMILY_ORDER = (
    "safenlp",
    "sat_relu",
    "malbeware",
    "metaroom",
    "acasxu",
    "linearizenn",
    "relusplitter",
    "dist_shift",
    "tllverify",
    "cgan",
    "cersyve",
    "cora",
    "vit",
)

SOURCE_FAMILY_TO_CANONICAL = {
    "safenlp_2024": "safenlp",
    "sat_relu": "sat_relu",
    "malbeware": "malbeware",
    "metaroom_2023": "metaroom",
    "acasxu_2023": "acasxu",
    "linearizenn_2024": "linearizenn",
    "relusplitter": "relusplitter",
    "dist_shift_2023": "dist_shift",
    "tllverifybench_2023": "tllverify",
    "cgan_2023": "cgan",
    "cersyve": "cersyve",
    "cora_2024": "cora",
    "vit_2023": "vit",
}

EXPECTED_FAMILY_VECTOR = {
    "safenlp": (1080, 432, 647, 1, 0),
    "sat_relu": (100, 50, 50, 0, 0),
    "malbeware": (150, 131, 19, 0, 0),
    "metaroom": (100, 94, 1, 0, 5),
    "acasxu": (186, 86, 34, 62, 4),
    "linearizenn": (60, 39, 1, 18, 2),
    "relusplitter": (220, 43, 2, 98, 77),
    "dist_shift": (72, 63, 7, 0, 2),
    "tllverify": (32, 5, 12, 15, 0),
    "cgan": (21, 5, 8, 8, 0),
    "cersyve": (12, 5, 6, 1, 0),
    "cora": (180, 20, 20, 9, 131),
    "vit": (200, 90, 0, 57, 53),
}

EXPECTED_COHORT_VECTOR = {
    "A": {"UNKNOWN": 79, "TIMEOUT": 49, "unsolved": 128},
    "B": {"UNKNOWN": 15, "TIMEOUT": 0, "unsolved": 15},
    "C": {"UNKNOWN": 95, "TIMEOUT": 168, "unsolved": 263},
    "D": {"UNKNOWN": 19, "TIMEOUT": 2, "unsolved": 21},
    "E": {"UNKNOWN": 59, "TIMEOUT": 53, "unsolved": 112},
    "F": {"UNKNOWN": 2, "TIMEOUT": 2, "unsolved": 4},
}

COHORT_NAMES = {
    "A": "Conv/ConvTranspose-ReLU sparse frontier",
    "B": "TLL symmetric/signed ReLU",
    "C": "plain dense FC-ReLU",
    "D": "shared-ancestor Add/Concat/skip",
    "E": "attention plus residual",
    "F": "smooth nonlinear tail",
}

SMOOTH_OPS = frozenset(
    {"Sigmoid", "Tanh", "Gelu", "Erf", "Sin", "Cos", "Exp", "Log", "Softplus"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_timeout(raw: str) -> str:
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid timeout {raw!r}") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"timeout must be finite and positive: {raw!r}")
    rendered = format(value.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _relative_path(raw: str, *, field: str) -> Path:
    value = raw.strip()
    if not value:
        raise ValueError(f"{field} is empty")
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"{field} must be a contained relative path: {raw!r}")
    normalized = Path(*[part for part in path.parts if part not in ("", ".")])
    if not normalized.parts:
        raise ValueError(f"{field} is empty after normalization")
    return normalized


def _contained_file(root: Path, relative: Path, *, field: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes root: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"{field} is not a regular file: {relative}")
    return candidate


def verify_authority(hyzor_root: Path) -> dict[str, dict[str, object]]:
    root = hyzor_root.resolve(strict=True)
    records: dict[str, dict[str, object]] = {}
    for key, expected in AUTHORITY_FILES.items():
        relative = Path(str(expected["relative_path"]))
        path = _contained_file(root, relative, field=f"authority {key}")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(
                f"authority hash mismatch for {key}: {actual} != {expected['sha256']}"
            )
        records[key] = {
            "relative_path": relative.as_posix(),
            "sha256": actual,
            "size_bytes": int(path.stat().st_size),
            "role": expected["role"],
        }
    return records


def verify_benchmark_provenance(benchmark_root: Path) -> dict[str, object]:
    root = benchmark_root.resolve(strict=True)
    repository = Path(
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve(strict=True)
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_BENCHMARK_GIT_COMMIT:
        raise ValueError(
            f"benchmark repository commit mismatch: {commit} != {EXPECTED_BENCHMARK_GIT_COMMIT}"
        )
    relative_root = root.relative_to(repository)
    relevant_paths = [
        (relative_root / source_family).as_posix()
        for source_family in EXPECTED_INSTANCES_CSV_HASHES
    ]
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            *relevant_paths,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("formal benchmark family trees have tracked modifications")

    instances_records: dict[str, dict[str, object]] = {}
    for source_family, expected_hash in EXPECTED_INSTANCES_CSV_HASHES.items():
        relative = Path(source_family) / "instances.csv"
        path = _contained_file(root, relative, field=f"{source_family} instances.csv")
        payload = path.read_bytes()
        actual = sha256_bytes(payload)
        if actual != expected_hash:
            raise ValueError(
                f"instances.csv hash mismatch for {source_family}: {actual} != {expected_hash}"
            )
        instances_records[source_family] = {
            "relative_path": relative.as_posix(),
            "size_bytes": len(payload),
            "sha256": actual,
        }
    return {
        "git_commit": commit,
        "relevant_tracked_tree_clean": True,
        "instances_csv": instances_records,
    }


def _operator_signature(payload: bytes, *, source: str) -> dict[str, object]:
    model = onnx.load_model_from_string(payload)
    for tensor in model.graph.initializer:
        if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError(f"external tensor data is not covered by model hash: {source}")

    counts = Counter(node.op_type for node in model.graph.node)
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    constant_outputs = {
        output
        for node in model.graph.node
        if node.op_type == "Constant"
        for output in node.output
    }
    runtime_values = {value.name for value in model.graph.input}
    runtime_values.update(
        output
        for node in model.graph.node
        if node.op_type != "Constant"
        for output in node.output
    )
    dynamic_merges = {"Add": 0, "Concat": 0, "MatMul": 0}
    for node in model.graph.node:
        if node.op_type not in dynamic_merges:
            continue
        dynamic_inputs = sum(
            name in runtime_values
            and name not in initializer_names
            and name not in constant_outputs
            for name in node.input
        )
        if dynamic_inputs >= 2:
            dynamic_merges[node.op_type] += 1

    payload: dict[str, object] = {
        "format_version": SIGNATURE_VERSION,
        "ir_version": int(model.ir_version),
        "opset_imports": [
            {"domain": item.domain, "version": int(item.version)}
            for item in sorted(model.opset_import, key=lambda item: item.domain)
        ],
        "node_count": len(model.graph.node),
        "op_counts": {key: counts[key] for key in sorted(counts)},
        "dynamic_merge_counts": dynamic_merges,
    }
    payload["signature_sha256"] = sha256_json(payload)
    return payload


def _is_tll_signature(op_counts: Mapping[str, int]) -> bool:
    nonzero = {name for name, count in op_counts.items() if int(count)}
    relu = int(op_counts.get("Relu", 0))
    matmul = int(op_counts.get("MatMul", 0))
    add = int(op_counts.get("Add", 0))
    return (
        relu >= 1
        and nonzero <= {"Add", "MatMul", "Relu"}
        and matmul == add == 2 * relu + 2
    )


def matching_cohorts(signature: Mapping[str, object]) -> list[str]:
    """Return the one disjoint A--F class selected only from graph structure."""

    op_counts = {
        str(key): int(value)
        for key, value in dict(signature["op_counts"]).items()
    }
    dynamic = {
        str(key): int(value)
        for key, value in dict(signature["dynamic_merge_counts"]).items()
    }
    attention = op_counts.get("Softmax", 0) > 0 or dynamic.get("MatMul", 0) > 0
    smooth = any(op_counts.get(name, 0) > 0 for name in SMOOTH_OPS)
    tll = _is_tll_signature(op_counts)
    shared_branch = dynamic.get("Add", 0) > 0 or dynamic.get("Concat", 0) > 0
    convolutional = (
        op_counts.get("Conv", 0) > 0 or op_counts.get("ConvTranspose", 0) > 0
    ) and op_counts.get("Relu", 0) > 0
    dense_relu = (
        op_counts.get("Relu", 0) > 0
        and (op_counts.get("Gemm", 0) > 0 or op_counts.get("MatMul", 0) > 0)
    )

    # The exclusions make the six predicates mutually exclusive rather than
    # hiding overlap behind a family/model menu.  E precedes smooth ops inside
    # attention graphs; F precedes Conv in smooth cGAN tails.
    predicates = {
        "E": attention,
        "F": not attention and smooth,
        "B": not attention and not smooth and tll,
        "D": not attention and not smooth and not tll and shared_branch,
        "A": (
            not attention
            and not smooth
            and not tll
            and not shared_branch
            and convolutional
        ),
        "C": (
            not attention
            and not smooth
            and not tll
            and not shared_branch
            and not convolutional
            and dense_relu
        ),
    }
    return sorted(key for key, matched in predicates.items() if matched)


def classify_signature(signature: Mapping[str, object]) -> str:
    matches = matching_cohorts(signature)
    if len(matches) != 1:
        raise ValueError(f"operator signature has {len(matches)} A-F matches: {matches}")
    return matches[0]


def _normalized_verdict(raw: str, *, vit: bool) -> str:
    value = raw.strip().upper()
    if vit and value == "CERTIFIED":
        value = "CERT"
    if value not in {"CERT", "ADV", "UNKNOWN", "TIMEOUT"}:
        raise ValueError(f"unsupported formal verdict: {raw!r}")
    return value


def _verified_authority_text(
    path: Path, authority_record: Mapping[str, object], *, label: str
) -> str:
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    if actual != authority_record["sha256"]:
        raise ValueError(f"{label} changed after composite authority verification")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc


def _read_source_rows(
    hyzor_root: Path, authority: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    root = hyzor_root.resolve(strict=True)
    rows: list[dict[str, object]] = []
    overlay_path = root / str(authority["twelve_family_overlay"]["relative_path"])
    overlay_text = _verified_authority_text(
        overlay_path, authority["twelve_family_overlay"], label="12-family overlay"
    )
    with io.StringIO(overlay_text, newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"benchmark", "iid", "onnx", "vnnlib", "csv_timeout", "raw_verdict"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("12-family overlay schema is incomplete")
        for source_row_index, raw in enumerate(reader):
            source_family = raw["benchmark"].strip()
            if source_family not in SOURCE_FAMILY_TO_CANONICAL or source_family == "vit_2023":
                raise ValueError(f"unexpected overlay family {source_family!r}")
            rows.append(
                {
                    "authority_key": "twelve_family_overlay",
                    "authority_sha256": authority["twelve_family_overlay"]["sha256"],
                    "source_row_index": source_row_index,
                    "source_record_sha256": sha256_json(dict(raw)),
                    "source_family": source_family,
                    "family": SOURCE_FAMILY_TO_CANONICAL[source_family],
                    "source_iid_raw": raw["iid"].strip(),
                    "model_raw": raw["onnx"],
                    "spec_raw": raw["vnnlib"],
                    "timeout_raw": raw["csv_timeout"],
                    "verdict": _normalized_verdict(raw["raw_verdict"], vit=False),
                }
            )

    vit_path = root / str(authority["vit_strict_rows"]["relative_path"])
    vit_text = _verified_authority_text(
        vit_path, authority["vit_strict_rows"], label="ViT strict source"
    )
    with io.StringIO(vit_text, newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"iid", "onnx", "vnnlib", "timeout_sec", "strict_status"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("ViT strict source schema is incomplete")
        for source_row_index, raw in enumerate(reader):
            rows.append(
                {
                    "authority_key": "vit_strict_rows",
                    "authority_sha256": authority["vit_strict_rows"]["sha256"],
                    "source_row_index": source_row_index,
                    "source_record_sha256": sha256_json(dict(raw)),
                    "source_family": "vit_2023",
                    "family": "vit",
                    "source_iid_raw": raw["iid"].strip(),
                    "model_raw": raw["onnx"],
                    "spec_raw": raw["vnnlib"],
                    "timeout_raw": raw["timeout_sec"],
                    "verdict": _normalized_verdict(raw["strict_status"], vit=True),
                }
            )
    return rows


def _family_summary(source_rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in source_rows:
        counts[str(row["family"])][str(row["verdict"])] += 1
    summary: dict[str, dict[str, int]] = {}
    for family in FAMILY_ORDER:
        current = counts[family]
        record = {
            "total": sum(current.values()),
            "cert": current["CERT"],
            "validated_adv": current["ADV"],
            "solved": current["CERT"] + current["ADV"],
            "unknown": current["UNKNOWN"],
            "timeout": current["TIMEOUT"],
            "remaining": current["UNKNOWN"] + current["TIMEOUT"],
        }
        expected = EXPECTED_FAMILY_VECTOR[family]
        observed = (
            record["total"],
            record["cert"],
            record["validated_adv"],
            record["unknown"],
            record["timeout"],
        )
        if observed != expected:
            raise ValueError(f"formal family vector mismatch for {family}: {observed} != {expected}")
        summary[family] = record
    if set(counts) != set(FAMILY_ORDER):
        raise ValueError(f"unexpected formal family set: {sorted(counts)}")
    return summary


def _asset_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _model_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "operator_signature": _operator_signature(payload, source=str(path)),
    }


def build_manifest(hyzor_root: Path, benchmark_root: Path) -> dict[str, object]:
    authority = verify_authority(hyzor_root)
    benchmark_provenance = verify_benchmark_provenance(benchmark_root)
    source_rows = _read_source_rows(hyzor_root, authority)
    if len(source_rows) != 2413:
        raise ValueError(f"composite source has {len(source_rows)} rows, expected 2,413")

    identities = [(str(row["family"]), str(row["source_iid_raw"])) for row in source_rows]
    if len(set(identities)) != len(identities):
        duplicate = next(identity for identity in identities if identities.count(identity) > 1)
        raise ValueError(f"duplicate family/iid authority row: {duplicate}")

    family_summary = _family_summary(source_rows)
    summary_path = hyzor_root.resolve(strict=True) / str(
        authority["composite_summary"]["relative_path"]
    )
    published_summary = json.loads(
        _verified_authority_text(
            summary_path,
            authority["composite_summary"],
            label="composite summary",
        )
    )
    expected_totals = {"ADV": 807, "CERT": 1063, "TIMEOUT": 274, "UNKNOWN": 269}
    if published_summary.get("totals") != expected_totals:
        raise ValueError("composite summary totals do not match the locked baseline")

    bench_root = benchmark_root.resolve(strict=True)
    model_cache: dict[Path, dict[str, object]] = {}
    spec_cache: dict[Path, dict[str, object]] = {}
    instances: list[dict[str, object]] = []
    family_position = {family: index for index, family in enumerate(FAMILY_ORDER)}

    for source in source_rows:
        verdict = str(source["verdict"])
        if verdict not in {"UNKNOWN", "TIMEOUT"}:
            continue
        source_family = str(source["source_family"])
        family_root = _contained_file(
            bench_root,
            Path(source_family) / "instances.csv",
            field=f"{source_family} instances.csv",
        ).parent
        model_relative = _relative_path(str(source["model_raw"]), field="model path")
        spec_relative = _relative_path(str(source["spec_raw"]), field="spec path")
        model_path = _contained_file(family_root, model_relative, field="model path")
        spec_path = _contained_file(family_root, spec_relative, field="spec path")

        if model_path not in model_cache:
            model_cache[model_path] = _model_record(model_path)
        if spec_path not in spec_cache:
            spec_cache[spec_path] = _asset_record(spec_path)
        model = model_cache[model_path]
        spec = spec_cache[spec_path]
        signature = model["operator_signature"]
        matches = matching_cohorts(signature)
        if len(matches) != 1:
            raise ValueError(
                f"unsolved row {source['family']}:{source['source_iid_raw']} has A-F matches {matches}"
            )
        cohort = matches[0]
        op_counts = dict(signature["op_counts"])
        a_reach = None
        if cohort == "A":
            a_reach = (
                "convtranspose_extension"
                if int(op_counts.get("ConvTranspose", 0)) > 0
                else "current_direct_implicit_conv2d"
            )

        try:
            source_iid = int(str(source["source_iid_raw"]))
        except ValueError as exc:
            raise ValueError(f"non-integer authority iid {source['source_iid_raw']!r}") from exc
        timeout_seconds = _canonical_timeout(str(source["timeout_raw"]))
        row_identity = f"{source['family']}:{source_iid}"
        identity_payload = {
            "format_version": ROW_IDENTITY_VERSION,
            "authority_sha256": source["authority_sha256"],
            "authority_data_row_index": source["source_row_index"],
            "family": source["family"],
            "source_family": source_family,
            "source_iid": source_iid,
            "source_record_sha256": source["source_record_sha256"],
            "model_sha256": model["sha256"],
            "spec_sha256": spec["sha256"],
            "timeout_seconds": timeout_seconds,
        }
        instances.append(
            {
                "row_identity": row_identity,
                "row_identity_sha256": sha256_json(identity_payload),
                "family": source["family"],
                "verdict": verdict,
                "cohort": cohort,
                "cohort_name": COHORT_NAMES[cohort],
                "a_reach_scope": a_reach,
                "source": {
                    "authority_key": source["authority_key"],
                    "authority_data_row_index": source["source_row_index"],
                    "source_record_sha256": source["source_record_sha256"],
                    "source_family": source_family,
                    "source_iid": source_iid,
                    "timeout_seconds": timeout_seconds,
                },
                "model": {
                    "relative_path": (Path(source_family) / model_relative).as_posix(),
                    "size_bytes": model["size_bytes"],
                    "sha256": model["sha256"],
                    "operator_signature": signature,
                },
                "spec": {
                    "relative_path": (Path(source_family) / spec_relative).as_posix(),
                    "size_bytes": spec["size_bytes"],
                    "sha256": spec["sha256"],
                },
            }
        )

    instances.sort(
        key=lambda row: (
            family_position[str(row["family"])],
            int(dict(row["source"])["source_iid"]),
        )
    )

    status_counts = Counter(str(row["verdict"]) for row in instances)
    cohort_counts: dict[str, dict[str, int]] = {}
    family_cohort: dict[str, dict[str, dict[str, int]]] = {}
    for cohort in "ABCDEF":
        rows = [row for row in instances if row["cohort"] == cohort]
        statuses = Counter(str(row["verdict"]) for row in rows)
        cohort_counts[cohort] = {
            "UNKNOWN": statuses["UNKNOWN"],
            "TIMEOUT": statuses["TIMEOUT"],
            "unsolved": len(rows),
        }
        family_cohort[cohort] = {}
        for family in FAMILY_ORDER:
            subset = [row for row in rows if row["family"] == family]
            if not subset:
                continue
            substatuses = Counter(str(row["verdict"]) for row in subset)
            family_cohort[cohort][family] = {
                "UNKNOWN": substatuses["UNKNOWN"],
                "TIMEOUT": substatuses["TIMEOUT"],
                "unsolved": len(subset),
            }

    a_direct = sum(
        row["a_reach_scope"] == "current_direct_implicit_conv2d" for row in instances
    )
    a_extension = sum(
        row["a_reach_scope"] == "convtranspose_extension" for row in instances
    )

    # Recheck the repository guard after hashing/parsing every selected asset,
    # so a concurrent tracked-tree change cannot straddle the census silently.
    if verify_benchmark_provenance(benchmark_root) != benchmark_provenance:
        raise ValueError("benchmark provenance changed during manifest construction")

    manifest: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "classification_rule_version": CLASSIFICATION_RULE_VERSION,
        "generator": {
            "relative_path": Path(__file__).resolve().relative_to(EXPERIMENT_ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "onnx_parser_version": onnx.__version__,
        "source_path_policy": {
            "hyzor": "authority paths are relative to the supplied read-only HyZor root",
            "benchmark": "asset paths are relative to the supplied read-only benchmarks root",
            "absolute_roots_omitted_for_mount_independence": True,
        },
        "composite_authority": authority,
        "benchmark_provenance": benchmark_provenance,
        "formal_baseline": {
            "total": 2413,
            "cert": 1063,
            "validated_adv": 807,
            "solved": 1870,
            "unknown": 269,
            "timeout": 274,
            "remaining": 543,
        },
        "family_summary": family_summary,
        "classification": {
            "reporting_order": list("ABCDEF"),
            "exclusive_precedence": ["E", "F", "B", "D", "A", "C"],
            "cohort_names": COHORT_NAMES,
            "selector_inputs": [
                "ONNX operator multiset",
                "dynamic Add/Concat/MatMul counts",
            ],
            "forbidden_selector_inputs": [
                "family",
                "model path or content hash",
                "iid",
                "property",
                "verdict",
            ],
        },
        "cohort_summary": cohort_counts,
        "cohort_family_matrix": family_cohort,
        "a_scope_proof": {
            "current_direct_implicit_conv2d": a_direct,
            "convtranspose_extension": a_extension,
            "total_A": a_direct + a_extension,
        },
        "coverage_proof": {
            "composite_source_rows": len(source_rows),
            "excluded_solved_rows": len(source_rows) - len(instances),
            "included_unsolved_rows": len(instances),
            "unknown": status_counts["UNKNOWN"],
            "timeout": status_counts["TIMEOUT"],
            "unassigned_rows": 0,
            "multiply_assigned_rows": 0,
            "unique_row_identity_count": len({row["row_identity"] for row in instances}),
            "unique_row_identity_sha256_count": len(
                {row["row_identity_sha256"] for row in instances}
            ),
        },
        "instances": instances,
    }
    validate_manifest(manifest, require_payload_hash=False)
    manifest["manifest_payload_sha256"] = sha256_json(manifest)
    validate_manifest(manifest, require_payload_hash=True)
    return manifest


def validate_manifest(
    manifest: Mapping[str, object], *, require_payload_hash: bool = True
) -> None:
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unexpected structure-manifest format")
    if manifest.get("classification_rule_version") != CLASSIFICATION_RULE_VERSION:
        raise ValueError("unexpected structure classification rule")
    if manifest.get("onnx_parser_version") != onnx.__version__:
        raise ValueError("manifest ONNX parser version mismatch")
    expected_source_path_policy = {
        "hyzor": "authority paths are relative to the supplied read-only HyZor root",
        "benchmark": "asset paths are relative to the supplied read-only benchmarks root",
        "absolute_roots_omitted_for_mount_independence": True,
    }
    if dict(manifest["source_path_policy"]) != expected_source_path_policy:
        raise ValueError("manifest source path policy mismatch")
    if require_payload_hash:
        payload = dict(manifest)
        claimed = payload.pop("manifest_payload_sha256", None)
        if not isinstance(claimed, str) or sha256_json(payload) != claimed:
            raise ValueError("structure-manifest payload hash mismatch")

    authority = dict(manifest["composite_authority"])
    for key, expected in AUTHORITY_FILES.items():
        record = dict(authority[key])
        if (
            record.get("relative_path") != expected["relative_path"]
            or record.get("sha256") != expected["sha256"]
        ):
            raise ValueError(f"manifest composite authority mismatch for {key}")

    benchmark_provenance = dict(manifest["benchmark_provenance"])
    if (
        benchmark_provenance.get("git_commit") != EXPECTED_BENCHMARK_GIT_COMMIT
        or benchmark_provenance.get("relevant_tracked_tree_clean") is not True
    ):
        raise ValueError("manifest benchmark Git provenance mismatch")
    instances_csv = dict(benchmark_provenance["instances_csv"])
    if set(instances_csv) != set(EXPECTED_INSTANCES_CSV_HASHES):
        raise ValueError("manifest benchmark family source set mismatch")
    for source_family, expected_hash in EXPECTED_INSTANCES_CSV_HASHES.items():
        record = dict(instances_csv[source_family])
        if (
            record.get("relative_path") != f"{source_family}/instances.csv"
            or record.get("sha256") != expected_hash
        ):
            raise ValueError(f"manifest instances.csv provenance mismatch for {source_family}")

    generator = dict(manifest["generator"])
    if (
        generator.get("relative_path") != Path(__file__).resolve().relative_to(EXPERIMENT_ROOT).as_posix()
        or generator.get("sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("manifest generator provenance mismatch")
    expected_baseline = {
        "total": 2413,
        "cert": 1063,
        "validated_adv": 807,
        "solved": 1870,
        "unknown": 269,
        "timeout": 274,
        "remaining": 543,
    }
    if dict(manifest["formal_baseline"]) != expected_baseline:
        raise ValueError("manifest formal baseline mismatch")

    family_summary = dict(manifest["family_summary"])
    if set(family_summary) != set(FAMILY_ORDER):
        raise ValueError("manifest family summary set mismatch")
    for family in FAMILY_ORDER:
        current = dict(family_summary[family])
        expected = EXPECTED_FAMILY_VECTOR[family]
        total, cert, validated_adv, unknown, timeout = expected
        expected_record = {
            "total": total,
            "cert": cert,
            "validated_adv": validated_adv,
            "solved": cert + validated_adv,
            "unknown": unknown,
            "timeout": timeout,
            "remaining": unknown + timeout,
        }
        if current != expected_record:
            raise ValueError(f"manifest family vector mismatch for {family}")

    classification = dict(manifest["classification"])
    if (
        classification.get("reporting_order") != list("ABCDEF")
        or classification.get("exclusive_precedence") != ["E", "F", "B", "D", "A", "C"]
        or dict(classification.get("cohort_names", {})) != COHORT_NAMES
        or classification.get("selector_inputs")
        != ["ONNX operator multiset", "dynamic Add/Concat/MatMul counts"]
        or classification.get("forbidden_selector_inputs")
        != ["family", "model path or content hash", "iid", "property", "verdict"]
    ):
        raise ValueError("manifest classification metadata mismatch")

    rows = list(manifest["instances"])
    if len(rows) != 543:
        raise ValueError(f"manifest has {len(rows)} unsolved rows, expected 543")
    row_ids = [str(row["row_identity"]) for row in rows]
    row_hashes = [str(row["row_identity_sha256"]) for row in rows]
    if len(set(row_ids)) != len(rows) or len(set(row_hashes)) != len(rows):
        raise ValueError("manifest row identities are not unique")
    family_position = {family: index for index, family in enumerate(FAMILY_ORDER)}
    expected_order = sorted(
        rows,
        key=lambda row: (
            family_position[str(row["family"])],
            int(dict(row["source"])["source_iid"]),
        ),
    )
    if rows != expected_order:
        raise ValueError("manifest rows are not in canonical family/iid order")

    statuses = Counter(str(row["verdict"]) for row in rows)
    if statuses != Counter({"UNKNOWN": 269, "TIMEOUT": 274}):
        raise ValueError(f"manifest verdict vector mismatch: {statuses}")
    cohorts: dict[str, Counter] = defaultdict(Counter)
    family_unsolved: dict[str, Counter] = defaultdict(Counter)
    authority_row_keys: set[tuple[str, int]] = set()
    matrix: dict[str, dict[str, dict[str, int]]] = {
        cohort: {} for cohort in "ABCDEF"
    }
    for row in rows:
        family = str(row["family"])
        source = dict(row["source"])
        model = dict(row["model"])
        spec = dict(row["spec"])
        expected_row_identity = f"{family}:{int(source['source_iid'])}"
        if row["row_identity"] != expected_row_identity:
            raise ValueError(f"row identity label mismatch for {row['row_identity']}")
        authority_key = str(source["authority_key"])
        expected_source_family = next(
            key
            for key, canonical in SOURCE_FAMILY_TO_CANONICAL.items()
            if canonical == family
        )
        expected_authority_key = (
            "vit_strict_rows" if family == "vit" else "twelve_family_overlay"
        )
        if (
            source["source_family"] != expected_source_family
            or authority_key != expected_authority_key
        ):
            raise ValueError(f"source family/authority mismatch for {row['row_identity']}")
        authority_row_key = (authority_key, int(source["authority_data_row_index"]))
        if authority_row_key in authority_row_keys:
            raise ValueError(f"duplicate authority row index for {row['row_identity']}")
        authority_row_keys.add(authority_row_key)
        identity_payload = {
            "format_version": ROW_IDENTITY_VERSION,
            "authority_sha256": dict(authority[authority_key])["sha256"],
            "authority_data_row_index": source["authority_data_row_index"],
            "family": family,
            "source_family": source["source_family"],
            "source_iid": source["source_iid"],
            "source_record_sha256": source["source_record_sha256"],
            "model_sha256": model["sha256"],
            "spec_sha256": spec["sha256"],
            "timeout_seconds": source["timeout_seconds"],
        }
        if sha256_json(identity_payload) != row["row_identity_sha256"]:
            raise ValueError(f"row identity hash mismatch for {row['row_identity']}")

        signature = dict(model["operator_signature"])
        signature_payload = dict(signature)
        claimed_signature_hash = signature_payload.pop("signature_sha256", None)
        if sha256_json(signature_payload) != claimed_signature_hash:
            raise ValueError(f"operator signature hash mismatch for {row['row_identity']}")
        matches = matching_cohorts(signature)
        if matches != [row["cohort"]]:
            raise ValueError(f"cohort mismatch for {row['row_identity']}: {matches}")
        cohort = str(row["cohort"])
        if row["cohort_name"] != COHORT_NAMES[cohort]:
            raise ValueError(f"cohort name mismatch for {row['row_identity']}")
        verdict = str(row["verdict"])
        cohorts[cohort][verdict] += 1
        family_unsolved[family][verdict] += 1

        op_counts = dict(signature["op_counts"])
        if cohort == "A":
            expected_scope = (
                "convtranspose_extension"
                if int(op_counts.get("ConvTranspose", 0)) > 0
                else "current_direct_implicit_conv2d"
            )
            if row["a_reach_scope"] != expected_scope:
                raise ValueError(f"A reach scope mismatch for {row['row_identity']}")
        elif row["a_reach_scope"] is not None:
            raise ValueError(f"non-A row has A reach scope: {row['row_identity']}")

    for cohort, expected in EXPECTED_COHORT_VECTOR.items():
        observed = {
            "UNKNOWN": cohorts[cohort]["UNKNOWN"],
            "TIMEOUT": cohorts[cohort]["TIMEOUT"],
            "unsolved": sum(cohorts[cohort].values()),
        }
        if observed != expected:
            raise ValueError(f"cohort vector mismatch for {cohort}: {observed}")

    for family in FAMILY_ORDER:
        current = dict(family_summary[family])
        if (
            family_unsolved[family]["UNKNOWN"] != current["unknown"]
            or family_unsolved[family]["TIMEOUT"] != current["timeout"]
        ):
            raise ValueError(f"family unsolved rows mismatch for {family}")

    for cohort in "ABCDEF":
        for family in FAMILY_ORDER:
            subset = [
                row
                for row in rows
                if row["cohort"] == cohort and row["family"] == family
            ]
            if not subset:
                continue
            substatuses = Counter(str(row["verdict"]) for row in subset)
            matrix[cohort][family] = {
                "UNKNOWN": substatuses["UNKNOWN"],
                "TIMEOUT": substatuses["TIMEOUT"],
                "unsolved": len(subset),
            }
    if dict(manifest["cohort_family_matrix"]) != matrix:
        raise ValueError("cohort/family matrix mismatch")

    expected_cohort_summary = {
        cohort: dict(EXPECTED_COHORT_VECTOR[cohort]) for cohort in "ABCDEF"
    }
    if dict(manifest["cohort_summary"]) != expected_cohort_summary:
        raise ValueError("cohort summary mismatch")

    a_direct = sum(
        row["cohort"] == "A"
        and row["a_reach_scope"] == "current_direct_implicit_conv2d"
        for row in rows
    )
    a_extension = sum(
        row["cohort"] == "A"
        and row["a_reach_scope"] == "convtranspose_extension"
        for row in rows
    )
    if (a_direct, a_extension) != (124, 4):
        raise ValueError(f"A reach split mismatch: {(a_direct, a_extension)}")
    expected_a_scope = {
        "current_direct_implicit_conv2d": 124,
        "convtranspose_extension": 4,
        "total_A": 128,
    }
    if dict(manifest["a_scope_proof"]) != expected_a_scope:
        raise ValueError("A scope proof mismatch")

    expected_coverage = {
        "composite_source_rows": 2413,
        "excluded_solved_rows": 1870,
        "included_unsolved_rows": 543,
        "unknown": 269,
        "timeout": 274,
        "unassigned_rows": 0,
        "multiply_assigned_rows": 0,
        "unique_row_identity_count": 543,
        "unique_row_identity_sha256_count": 543,
    }
    if dict(manifest["coverage_proof"]) != expected_coverage:
        raise ValueError("coverage proof mismatch")


def render_manifest(manifest: Mapping[str, object]) -> bytes:
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _require_output_below(output: Path, allowed_root: Path) -> Path:
    root = allowed_root.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    candidate = parent / output.name
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output must remain below {root}: {output}") from exc
    return candidate


def exclusive_atomic_write(
    output: Path, payload: bytes, *, allowed_root: Path = EXPERIMENT_ROOT
) -> None:
    target = _require_output_below(output, allowed_root)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyzor-root", type=Path, default=DEFAULT_HYZOR_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.hyzor_root, args.benchmark_root)
    payload = render_manifest(manifest)
    if args.output is not None:
        exclusive_atomic_write(args.output, payload)
        print(json.dumps({
            "output": str(args.output.resolve()),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_sha256": manifest["manifest_payload_sha256"],
            "rows": len(manifest["instances"]),
        }, sort_keys=True))
        return

    existing = _require_output_below(args.verify, EXPERIMENT_ROOT)
    if existing.read_bytes() != payload:
        raise SystemExit("existing manifest differs from deterministic rebuild")
    print(json.dumps({
        "verified": str(existing.resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_sha256": manifest["manifest_payload_sha256"],
        "rows": len(manifest["instances"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
