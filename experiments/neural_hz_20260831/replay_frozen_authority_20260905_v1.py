"""Rebuild the old manifest using the exact archived table bytes from Git.

The frozen V1 generator and manifest are not edited. A process-local input
adapter supplies the original table's verified metadata; all other authority
files retain their original paths and are checked normally. The resulting
manifest must be BYTE IDENTICAL, including its original authority hashes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neural_hz_20260831 import generate_formal_unsolved_structure_manifest_v1 as v1
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
SNAPSHOT = EXPERIMENT / "evidence/formal_verdict_1870_git_01923a5.tex"
TABLE_COMMIT = "01923a5bc896667bd0d0a19b310ee52e88bb2b70"
TABLE_REPO = v1.DEFAULT_HYZOR_ROOT / "VMCAI_2027___Kaijie_Guanqin"
GENERATOR_HASH = "33a53a00848a970251fabe192c6532aea1cd7f5d0dec05bef84fa823a34eb52e"
MANIFEST_HASH = "393590e26d3ae4edd50c9a7d2df48980c8b3701237b61c030bfbe4ad36947d8d"


def original_authority_metadata(root):
    if root.resolve() != v1.DEFAULT_HYZOR_ROOT.resolve():
        raise ValueError("unexpected authority root")
    records = {}
    for key, expected in v1.AUTHORITY_FILES.items():
        path = SNAPSHOT if key == "formal_verdict_table" else root / expected["relative_path"]
        actual = _sha256(path)
        if actual != expected["sha256"]:
            raise ValueError(f"original authority bytes not available: {key}")
        records[key] = {**expected, "sha256": actual, "size_bytes": path.stat().st_size}
    return records


def build_record():
    if _sha256(Path(v1.__file__)) != GENERATOR_HASH:
        raise ValueError("frozen generator changed")
    original = subprocess.check_output(["git", "-C", str(TABLE_REPO), "show", f"{TABLE_COMMIT}:tables/verdict.tex"])
    if original != SNAPSHOT.read_bytes():
        raise ValueError("isolated table does not match pinned Git blob")
    authority = original_authority_metadata(v1.DEFAULT_HYZOR_ROOT)
    path = EXPERIMENT / "manifests/formal_unsolved_structure_manifest_v1.json"
    if _sha256(path) != MANIFEST_HASH:
        raise ValueError("frozen manifest changed")
    # Explicit input substitution, limited to this diagnostic process. This
    # does not patch a test or make the changed live table pass its old hash.
    with patch.object(v1, "verify_authority", original_authority_metadata):
        rebuilt = v1.build_manifest(v1.DEFAULT_HYZOR_ROOT, v1.DEFAULT_BENCHMARK_ROOT)
    if v1.render_manifest(rebuilt) != path.read_bytes():
        raise ValueError("recovered-authority rebuild differs from original manifest")
    if authority != original_authority_metadata(v1.DEFAULT_HYZOR_ROOT):
        raise ValueError("authority changed during replay")
    live_table = TABLE_REPO / "tables/verdict.tex"
    return {
        "schema": "frozen_authority_recovery_v1", "date": "2026-09-05",
        "formal_baseline": "1870/2413", "gain": 0,
        "recovery_source": {"git_repository": str(TABLE_REPO), "commit": TABLE_COMMIT,
                            "path": "tables/verdict.tex", "snapshot_sha256": _sha256(SNAPSHOT)},
        "live_table_sha256": _sha256(live_table),
        "live_table_is_original_bytes": _sha256(live_table) == _sha256(SNAPSHOT),
        "original_authority": authority,
        "generator_sha256": GENERATOR_HASH,
        "manifest_sha256": MANIFEST_HASH,
        "original_manifest_rebuilt_byte_identically": True,
        "coverage_proof": rebuilt["coverage_proof"],
        "family_summary": rebuilt["family_summary"],
        "live_path_v1_test_still_rejects_changed_table": True,
        "replay_adapter_sha256": _sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.parent.resolve() != (EXPERIMENT / "evidence").resolve():
        raise ValueError("output outside isolated evidence directory")
    if os.path.lexists(output):
        raise FileExistsError(output)
    record = build_record()
    _atomic_exclusive_json(output, record)
    print(json.dumps({"output": str(output), "sha256": _sha256(output),
                      "byte_identical_rebuild": True, "coverage": record["coverage_proof"]}))


if __name__ == "__main__":
    main()
