"""One frozen, four-worker TLL qualification with exclusive per-job records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neural_hz_20260831 import shadow_worker
from experiments.neural_hz_20260831 import generate_formal_unsolved_structure_manifest_v1 as authority
from experiments.neural_hz_20260831.replay_frozen_authority_20260905_v1 import original_authority_metadata
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
BENCH_ROOT = authority.DEFAULT_BENCHMARK_ROOT
SPEC_ROOT = Path("/data1/Kane/HyZor/RQ4_REPRESENTATION_ABLATION_20260822/vnnlib_v2")
FAMILY = "tllverifybench_2023"
ARM = "signed_cancellation_width_adaptive"
EXTRA_CODE = (
    ROOT / "act/pipeline/verification/torch2act.py",
    ROOT / "act/pipeline/verification/batchnorm_graph.py",
    Path(__file__), EXPERIMENT / "TLL_CURRENT_SOURCE_REPLAY_PREREG_20260905.md",
)


def _spec_equivalent(original, converted):
    def content(text):
        return " ".join(line.split(";", 1)[0].strip() for line in text.splitlines()).strip()
    old, new = content(original), content(converted)
    declarations = re.findall(r"\(declare-const ([XY]_\d+) Real\)", old)
    if declarations != ["X_0", "X_1", "Y_0"]:
        raise ValueError("unexpected TLL original declaration schema")
    old = re.sub(r"\(declare-const [XY]_\d+ Real\)", "", old)
    old = re.sub(r"\b([XY])_(\d+)\b", r"\1[\2]", old)
    header = r"\(vnnlib-version 2\.0\)\s*\(declare-network N\s*\(declare-input X Real \[2\]\)\s*\(declare-output Y Real \[1\]\)\s*\)"
    new, count = re.subn(header, "", new, count=1)
    if count != 1 or " ".join(old.split()) != " ".join(new.split()):
        raise ValueError("converted TLL constraints differ from frozen original")


def preflight():
    csv = BENCH_ROOT / FAMILY / "instances.csv"
    if _sha256(csv) != authority.EXPECTED_INSTANCES_CSV_HASHES[FAMILY]:
        raise ValueError("TLL universe changed")
    rows = [line.split(",") for line in csv.read_text().splitlines() if line.strip()]
    if len(rows) != 32:
        raise ValueError("TLL universe is not 32 rows")
    baseline = {
        int(row["source_iid_raw"]): row["verdict"]
        for row in authority._read_source_rows(
            authority.DEFAULT_HYZOR_ROOT,
            original_authority_metadata(authority.DEFAULT_HYZOR_ROOT),
        ) if row["family"] == "tllverify"
    }
    if Counter(baseline.values()) != {"CERT": 5, "ADV": 12, "UNKNOWN": 15}:
        raise ValueError("formal TLL baseline changed")
    jobs = []
    for iid, row in enumerate(rows):
        model = BENCH_ROOT / FAMILY / row[0]
        original = BENCH_ROOT / FAMILY / row[1]
        converted = SPEC_ROOT / FAMILY / row[1]
        _spec_equivalent(original.read_text(), converted.read_text())
        jobs.append({"iid": iid, "baseline": baseline[iid],
                     "prior_candidate_solved": iid not in {9, 15, 16},
                     "assets": {str(p): _sha256(p) for p in (model, original, converted)}})
    return {"schema": "tll_current_source_qualification_v1", "date": "2026-09-05",
            "formal_baseline": "1870/2413", "gain": 0, "family": FAMILY,
            "arm": ARM, "representation": "sparse", "solver_timeout_s": 45,
            "supervisor_wall_limit_s": 240, "memory_gb": 16, "concurrency": 4,
            "numeric_threads": 1, "bn_repair_enabled": False,
            "provenance": shadow_worker._provenance(ROOT),
            "extra_code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in EXTRA_CODE},
            "instances_csv_sha256": _sha256(csv), "jobs": jobs}


def run_one(directory, job, frozen):
    iid = job["iid"]
    output = directory / f"iid{iid:02d}.json"
    log = directory / f"iid{iid:02d}.log"
    record = {"iid": iid, "formal_gain": 0, "counted_verdict": "ERROR"}
    started = time.monotonic()
    command = [sys.executable, str(EXPERIMENT / "shadow_worker.py"), FAMILY, str(iid),
               "--arm", ARM, "--representation", "sparse", "--act-root", str(ROOT),
               "--bench-root", str(BENCH_ROOT), "--vnnlib-root", str(SPEC_ROOT),
               "--solver-timeout", "45", "--memory-gb", "16", "--output", str(output)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        if any(_sha256(Path(p)) != h for p, h in job["assets"].items()):
            raise ValueError("asset changed before execution")
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=240, check=False)
        record["exit_code"] = result.returncode
        if result.returncode != 0:
            raise ValueError("worker returned nonzero")
        data = json.loads(output.read_text())
        if (data.get("iid"), data.get("bench"), data.get("arm"), data.get("provenance")) != (
            iid, FAMILY, ARM, frozen["provenance"]
        ):
            raise ValueError("worker provenance mismatch")
        verdict = data.get("verdict")
        validations = data.get("concrete_validations", [])
        invalid = sum(item.get("valid") is not True for item in validations)
        if invalid or (verdict == "ADV" and not any(
            all(item.get(key) is True for key in ("valid", "input_ok", "violates"))
            for item in validations
        )):
            record["invalid_adv"] = True
            raise ValueError("missing or invalid concrete witness")
        if verdict in {"CERT", "ADV"} and data.get("output_hz_exact") is not True:
            raise ValueError("solved result lacks exact terminal HZ")
        if verdict not in {"CERT", "ADV", "UNKNOWN", "ERROR"}:
            raise ValueError("unexpected verdict")
        record.update(counted_verdict=verdict, result_sha256=_sha256(output),
                      verify_s=data.get("verify_s"), worker_wall_s=data.get("worker_wall_s"),
                      concrete_validation_count=len(validations))
    except subprocess.TimeoutExpired:
        record.update(counted_verdict="TIMEOUT", error="supervisor_wall_limit")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["supervisor_wall_s"] = time.monotonic() - started
    record["command"] = command
    _atomic_exclusive_json(directory / f"iid{iid:02d}.exit.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.absolute()
    if directory.parent.resolve() != (EXPERIMENT / "results").resolve():
        raise ValueError("new directory must be directly under isolated results")
    if os.path.lexists(directory):
        raise FileExistsError(directory)
    frozen = preflight()
    directory.mkdir()
    _atomic_exclusive_json(directory / "preregistered.json", frozen)
    started = time.monotonic()
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(run_one, directory, job, frozen) for job in frozen["jobs"]]
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            print(json.dumps({"completed": len(records), "iid": record["iid"],
                              "verdict": record["counted_verdict"], "error": record.get("error")}), flush=True)
    by_id = {r["iid"]: r for r in records}
    solved = {iid for iid, r in by_id.items() if r["counted_verdict"] in {"CERT", "ADV"}}
    formal = {j["iid"] for j in frozen["jobs"] if j["baseline"] in {"CERT", "ADV"}}
    retained = {j["iid"] for j in frozen["jobs"] if j["prior_candidate_solved"]}
    drift = shadow_worker._provenance(ROOT) != frozen["provenance"] or any(
        _sha256(ROOT / p) != h for p, h in frozen["extra_code_sha256"].items()
    ) or any(_sha256(Path(p)) != h for j in frozen["jobs"] for p, h in j["assets"].items())
    summary = {"schema": frozen["schema"], "formal_baseline": "1870/2413", "formal_gain": 0,
               "completed": len(records), "counts": dict(Counter(r["counted_verdict"] for r in records)),
               "lost_formal_solved": sorted(formal - solved), "lost_prior_candidate_solved": sorted(retained - solved),
               "newly_solved_vs_formal": sorted(solved - formal), "provenance_drift": drift,
               "invalid_adv": sum(bool(r.get("invalid_adv")) for r in records),
               "qualification_passed": len(records) == 32 and formal <= solved and retained <= solved and not drift
                   and not any(r.get("invalid_adv") for r in records),
               "elapsed_s": time.monotonic() - started, "controlled_speed_claim": False,
               "records": sorted(records, key=lambda r: r["iid"])}
    _atomic_exclusive_json(directory / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
