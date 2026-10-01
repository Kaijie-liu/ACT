"""Archive the frozen small CPU controls, or independently recheck saved bounds.

Uses a new directory every time. No native LP/MILP, GPU, model or data load.
The control harness is not the later hard-budget production supervisor.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "configs/hz_batch_support_controls_20261001.json"
PROTOCOL_SHA = "dd83bec8cbcfae785cc222dccaf350aa40f6aec2b79b2a0bf140272857dee125"
FILES = [PROTOCOL, "docs/hz_batch_support_design_20261001.md",
         "act/back_end/moe/batched_support.py", "act/back_end/moe/check_batched_support.py",
         "act/back_end/moe/test_batched_support.py", "scripts/run_hz_batch_support_controls.py",
         "act/back_end/solver/hz_lp_export.py", "act/back_end/solver/check_hz_lp_export.py",
         "act/back_end/solver/lp_certificate.py", "act/back_end/solver/solver_hz.py",
         "scoped_source/rowwise_bound.py", "scoped_source/rowwise_native.py"]
OBSERVATIONS = {"guarded_two_sides_positive", "guarded_nonpositive", "equality_coupled",
                "private_binary_relaxation", "constant_objective", "batch_differential_reference",
                "partial_candidate_rejected"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def load(path):
    return json.loads(path.read_text())


def check_protocol():
    if sha(ROOT / PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("frozen protocol changed")
    protocol = load(ROOT / PROTOCOL)
    for relative, expected in protocol["frozen_dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("frozen dependency changed: " + relative)


def expected_tests():
    tree = ast.parse((ROOT / "act/back_end/moe/test_batched_support.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BatchedSupportTests")
    return sorted("act.back_end.moe.test_batched_support.BatchedSupportTests." + node.name
                  for node in cls.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))


def check_inventory(execution, summary, bindings, observations, anchors):
    required = expected_tests()
    if set(bindings) != set(FILES): raise ValueError("implementation inventory incomplete")
    if (execution["protocol_sha256"] != PROTOCOL_SHA or execution["tests"] != required
            or summary["tests"] != len(required)
            or [row["test"] for row in summary["outcomes"]] != required):
        raise ValueError("fixed control inventory incomplete")
    if set(observations) != OBSERVATIONS or set(anchors) != OBSERVATIONS:
        raise ValueError("required observation/anchor inventory incomplete")
    for record in (execution, summary):
        if any(record[key] != 0 for key in ("native_solves", "gpu_executions", "real_requests")):
            raise ValueError("control execution scope changed")


def run(destination):
    check_protocol()
    if not destination.is_relative_to(ROOT.parent / "baseline_runs"):
        raise ValueError("control archive must be a new baseline_runs directory")
    destination.mkdir(parents=True, exist_ok=False)
    bindings = {}
    for name in FILES:
        target = destination / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        bindings[name] = sha(target)
    save(destination / "implementation.json", bindings)
    start = time.monotonic()
    from act.back_end.moe.test_batched_support import BatchedSupportTests
    import torch
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BatchedSupportTests)
    names = [test.id() for test in suite]
    save(destination / "execution.json", {
        "protocol_sha256": PROTOCOL_SHA, "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": sha(destination / "implementation.json"),
        "tests": names, "python": sys.version, "torch": torch.__version__,
        "environment_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "native_solves": 0, "gpu_executions": 0, "real_requests": 0,
        "hard_budget_supervision": False})
    with (destination / "events.jsonl").open("x") as events, (destination / "tests.log").open("x") as log:
        class Result(unittest.TextTestResult):
            outcomes = []

            def startTest(self, test):
                self.started = time.monotonic()
                self.outcome = "PASS"
                events.write(json.dumps({"event": "START", "test": test.id()}) + "\n")
                events.flush()
                super().startTest(test)

            def addError(self, test, err):
                self.outcome = "ERROR"
                super().addError(test, err)

            def addFailure(self, test, err):
                self.outcome = "FAIL"
                super().addFailure(test, err)

            def stopTest(self, test):
                item = {"test": test.id(), "status": self.outcome, "seconds": time.monotonic() - self.started}
                self.outcomes.append(item)
                events.write(json.dumps({"event": "TERMINAL", **item}) + "\n")
                events.flush()
                super().stopTest(test)

        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    save(destination / "observations.json", BatchedSupportTests.observations)
    from scoped_source.rowwise_bound import identity
    anchors = {name: {"batch_sha256": identity(item["batch"]), "candidate_sha256": identity(item["candidates"])}
               for name, item in BatchedSupportTests.observations.items()}
    save(destination / "anchors.json", anchors)
    summary = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "outcomes": result.outcomes,
               "observations_sha256": sha(destination / "observations.json"), "anchors_sha256": sha(destination / "anchors.json"),
               "seconds_including_imports_tests_and_observation_publication": time.monotonic() - start,
               "scope": "finite synthetic CPU support controls, not speed or MoE proof",
               "native_solves": 0, "gpu_executions": 0, "real_requests": 0}
    save(destination / "summary.json", summary)
    print(json.dumps({"root": str(destination), "status": summary["status"], "tests": summary["tests"],
                      "failures": summary["failures"], "errors": summary["errors"]}))
    return 0 if result.wasSuccessful() else 1


def audit(destination):
    """No candidate algorithm runs; every saved accepted lower bound is rechecked."""
    check_protocol()
    from act.back_end.moe.check_batched_support import check_batch, validated_records
    from scoped_source.rowwise_bound import check_bound, identity
    from fractions import Fraction
    start = time.monotonic()
    summary, execution = load(destination / "summary.json"), load(destination / "execution.json")
    bindings = load(destination / "implementation.json")
    observations, anchors = load(destination / "observations.json"), load(destination / "anchors.json")
    check_inventory(execution, summary, bindings, observations, anchors)
    for name, digest in bindings.items():
        if sha(destination / "implementation" / name) != digest or sha(ROOT / name) != digest:
            raise ValueError("changed executed/current implementation: " + name)
    if execution["implementation_sha256"] != sha(destination / "implementation.json"):
        raise ValueError("implementation manifest binding")
    if summary["status"] != "PASS" or summary["failures"] or summary["errors"]:
        raise ValueError("controls did not pass; preserve failed archive")
    if execution["tests"] != [o["test"] for o in summary["outcomes"]] or any(o["status"] != "PASS" for o in summary["outcomes"]):
        raise ValueError("control roster incomplete")
    for name in ("observations", "anchors"):
        if sha(destination / (name + ".json")) != summary[name + "_sha256"]:
            raise ValueError("saved evidence identity: " + name)
    events = [json.loads(line) for line in (destination / "events.jsonl").read_text().splitlines()]
    expected_events = []
    for terminal in summary["outcomes"]:
        expected_events.extend([{"event": "START", "test": terminal["test"]}, {"event": "TERMINAL", **terminal}])
    if events != expected_events:
        raise ValueError("control terminal event roster mismatch")
    checked = {}
    for name, item in observations.items():
        batch, candidates = item["batch"], item["candidates"]
        anchor = anchors[name]
        if identity(batch) != anchor["batch_sha256"] or identity(candidates) != anchor["candidate_sha256"]:
            raise ValueError("external saved anchor mismatch")
        deadline = time.monotonic() + 300
        if "expected_rejection" in item:
            if (name != "partial_candidate_rejected" or item["required"] != len(batch["queries"])
                    or item["received"] != len(candidates["entries"]) or item["required"] != 4
                    or item["received"] != 3 or item["complete"] is not False):
                raise ValueError("partial evidence denominator/status changed")
            try:
                check_batch(batch, candidates, expected_batch_sha256=anchor["batch_sha256"], deadline=deadline)
            except ValueError as exc:
                if str(exc) != item["expected_rejection"]: raise
            else:
                raise ValueError("partial evidence incorrectly accepted")
            checked[name] = {"status": "REJECTED_AS_REQUIRED", "required": item["required"], "received": item["received"]}
            continue
        if name == "partial_candidate_rejected": raise ValueError("partial evidence missing rejection")
        accepted = check_batch(batch, candidates, expected_batch_sha256=anchor["batch_sha256"], deadline=deadline)
        if accepted != {k: v for k, v in item["accepted"].items() if k != "cost_seconds"}:
            raise ValueError("saved reception changed")
        # These are experiment metadata checks, separate from lower-bound soundness.
        if any(candidates[k] != v for k, v in {"algorithm": "projected_dual_subgradient_multiobjective_v1",
                                               "iterations": 128, "dtype": "float64", "device": "cpu"}.items()):
            raise ValueError("candidate execution metadata")
        records = validated_records(batch, expected_batch_sha256=anchor["batch_sha256"], deadline=deadline)
        for (query, lp), entry in zip(records, candidates["entries"]):
            zero = {"lp_sha256": identity(lp), "inequality_dual": [0] * len(lp["b"]),
                    "equality_dual": [0] * len(lp["h"]), "claimed_lower_bound": entry["zero_candidate_lower_bound"]}
            checked_zero = check_bound(lp, zero, deadline=deadline)
            if Fraction(checked_zero["checked_lower_bound"]) != Fraction(entry["zero_candidate_lower_bound"]):
                raise ValueError("incorrect zero-candidate metadata")
            if entry["id"] != query["id"]: raise ValueError("saved candidate order")
        checked[name] = {"status": accepted["status"], "bounds": accepted["results"],
                         "cost_seconds": item["accepted"]["cost_seconds"]}
    return {"schema": "HZ_BATCH_SUPPORT_CONTROL_ARCHIVE_V1", "status": "PASS",
            "root": str(destination), "protocol_sha256": PROTOCOL_SHA,
            "implementation_sha256": execution["implementation_sha256"],
            "summary_sha256": sha(destination / "summary.json"), "tests": summary["tests"],
            "checked": checked, "audit_seconds": time.monotonic() - start,
            "control_seconds": summary["seconds_including_imports_tests_and_observation_publication"],
            "new_solves_in_audit": 0, "gpu_executions": 0, "real_requests": 0,
            "hard_budget_supervision": False, "new_complete_moe_certificates": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.check:
        report = audit(args.root.resolve())
        if args.report:
            path = args.report.resolve()
            if path.parent != ROOT / "docs": raise ValueError("compact report must be under repo docs")
            save(path, report)
        print(json.dumps(report))
        return 0
    if args.report: raise ValueError("report requires independent audit")
    return run(args.root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
