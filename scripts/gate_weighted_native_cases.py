"""Frozen M1b literal CPU objects; twelve calls, no trained checkpoints.

Imported only inside the bounded native worker AFTER the execution freeze.
"""
import copy
from fractions import Fraction
import json
from pathlib import Path
import time

from scoped_proof.io import save, sha
from scoped_source.hz_request_store import compact, identity, require, POLICY

CONFIG = "act/pipeline/moe/configs/route_complexity_reuse_v1.json"
LITERALS = {
    "A": {"router_w": [[0., 0.], [0., 0.]], "router_b": [0., 0.],
          "expert_w": [[[0., 0.]] * 3] * 2,
          "expert_b": [[0., 1., -2.], [3., 0., -2.]], "expected": "SAFE"},
    "B": {"router_w": [[0., 0.]] * 3, "router_b": [0.] * 3,
          "expert_w": [[[0., 0.]] * 2] * 3,
          "expert_b": [[-.2, 0.], [1., 0.], [2., 0.]], "expected": "SAFE"},
    "C": {"router_w": [[1., 0.], [0., 0.]], "router_b": [0., 0.],
          "expert_w": [[[1., 0.], [0., 0.]]] * 2,
          "expert_b": [[-.45, 0.], [-.45, 0.]], "expected": "UNSAFE"},
}
# Alternate enabled order, no result-based reorder or extra solver calls.
CALLS = [(case, method, enabled) for case in ("A", "B", "C")
         for method in ("staged", "monolithic_f0")
         for enabled in ((False, True) if method == "staged" else (True, False))]
FAULTS = ["default_deadline", "rational_capacity", "source_error_not_resource",
          "counter_partial_and_invalid", "missing_obligation",
          "replay_binding", "reception_deadline_binding", "reception_ledger_reset"]


def build(case):
    import torch
    from torch import nn
    from act.back_end.moe.model import OutputLevelMoE
    from act.back_end.moe.schema import OutputLevelMoESpec, GateKind
    literal = LITERALS[case]

    def linear(weights, biases):
        layer = nn.Linear(2, len(biases), dtype=torch.float64, device="cpu")
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weights, dtype=torch.float64))
            layer.bias.copy_(torch.tensor(biases, dtype=torch.float64))
        return nn.Sequential(nn.Flatten(start_dim=1), layer)

    return OutputLevelMoE(
        linear(literal["router_w"], literal["router_b"]),
        [linear(w, b) for w, b in zip(literal["expert_w"], literal["expert_b"])],
        OutputLevelMoESpec(num_experts=len(literal["expert_b"]), top_k=2,
                           gate=GateKind.SELECTED_SOFTMAX)).eval()


def run(root, run_dir, deadline, phase_deadline, tick):
    import torch
    from act.util.device_manager import initialize_device
    from act.pipeline.moe.staged_verifier import verify_staged_linf, write_evidence_package
    from act.pipeline.moe.audit_staged_evidence import audit_evidence_package, _audit_safe_structure
    from act.pipeline.moe.gate_native_contract import (
        GateWork, GateCapacity, F0Calls, terminal, validate_counts, require_continuation)
    from act.pipeline.moe.request_budget import RequestBudget, BudgetExhausted
    from act.pipeline.moe.gate_weighted_facts import GateFacts
    torch.set_num_threads(1)
    initialize_device("cpu", "float64")
    work = GateWork(deadline)
    records, faults = [], []
    result = dict(status="FAIL", calls=records, faults=faults, work_counts=work.counts,
                  literal_sha256=identity(LITERALS), phase_deadline=phase_deadline)
    first_on_package = None
    first_on_evidence = None

    def reject(name, action, kind):
        tick()
        begin = dict(work.counts)
        try:
            action()
        except kind as exc:
            faults.append(dict(name=name, status="PASS", rejected_as=type(exc).__name__,
                               work_before=begin, work_after=dict(work.counts)))
        else:
            raise AssertionError("fault was accepted: " + name)
        save(run_dir / f"fault_{len(faults):02d}.json", faults[-1])

    try:
        # No numerical call in these finite interface faults.
        reject("default_deadline", lambda: GateFacts([], {}, RequestBudget(
            300, started=time.monotonic()-301)), BudgetExhausted)
        reject("rational_capacity", lambda: work.exact(Fraction(1 << POLICY["work_bits"])),
               GateCapacity)
        require(terminal(GateCapacity("control"))["status"] == "UNKNOWN"
                and terminal(BudgetExhausted("control"))["status"] == "TIMEOUT",
                "typed terminal mapping")
        reject("source_error_not_resource", lambda: terminal(ValueError("binding")), ValueError)
        counters = F0Calls()
        counters.attempted()
        require(counters.record() == dict(attempted=1, returned=0, status0_returned=0),
                "unreturned call must remain attempted")
        reject("counter_partial_and_invalid",
               lambda: validate_counts(dict(attempted=0, returned=1, status0_returned=0)), ValueError)
        for ordinal, (case, method, enabled) in enumerate(CALLS):
            tick()
            require(time.monotonic() < phase_deadline, "native phase cutoff")
            call_start = time.monotonic()
            before = dict(work.counts)
            net = build(case)
            cfg = json.loads((root / CONFIG).read_text())
            cfg.update(comparison_method=method, gate_native_contract=True)
            cfg["f0"]["gate_weighted_facts"] = enabled
            center = torch.full((1, 2), .5, dtype=torch.float64)
            literal_binding = dict(kind="FROZEN_SYNTHETIC_LITERAL", case=case,
                                   sha256=identity(LITERALS[case]))
            package = run_dir / f"call_{ordinal:02d}"
            save(run_dir / f"begin_{ordinal:02d}.json",
                 dict(ordinal=ordinal, case=case, method=method, enabled=enabled,
                      deadline=deadline, started=call_start, literal=literal_binding))
            report = verify_staged_linf(
                net, center, .1, cfg, expected_clean_prediction=0,
                checkpoint_identity=literal_binding, budget_started_at=deadline-300,
                gate_work=work)
            verified = time.monotonic()
            tick()
            require(len(compact(report.evidence)) < 2**20
                    and all(t.numel() <= 2 for t in report.request_tensors.values())
                    and (report.witness is None or report.witness.numel() <= 2),
                    "fixed package size/shape")
            write_evidence_package(report, package)
            serialized = time.monotonic()
            tick()
            audit = audit_evidence_package(package, replay_unsafe=True, replay_model=net,
                                           gate_work=work, deadline=deadline)
            tick()
            record = dict(
                ordinal=ordinal, case=case, method=method, enabled=enabled,
                requested_seconds=300, shared_batch_deadline=deadline,
                request_id=report.evidence["request_id"], status=report.status,
                reason=report.reason, decision_tier=report.evidence["verdict"]["decision_tier"],
                actual_path=report.evidence["route_complexity_schedule"]["selected_path"],
                f0_calls=report.evidence["f0_calls"], package=str(package),
                evidence_sha256=sha(package / "evidence.json"),
                manifest_sha256=sha(package / "manifest.json"),
                verify_seconds=verified-call_start, serialization_seconds=serialized-verified,
                audit_seconds=time.monotonic()-serialized,
                total_through_audit_seconds=time.monotonic()-call_start,
                work_before=before, work_after=dict(work.counts),
                work_delta={k: work.counts[k]-before[k] for k in before}, audit=audit)
            records.append(record)
            save(run_dir / f"end_{ordinal:02d}.json", record)
            require(audit["status"] == "PASS", "full package audit failed")
            require(report.status == LITERALS[case]["expected"], "analytic terminal mismatch")
            require(report.evidence["route_coverage"]["feasible_route_sets"] ==
                    ([[0, 1], [0, 2], [1, 2]] if case == "B" else [[0, 1]]),
                    "complete tie-legal pair coverage")
            if enabled and first_on_package is None:
                first_on_package, first_on_evidence = package, report.evidence

        # Same source, one-factor differential. No requirement of speedup/extra SAFE.
        for case in LITERALS:
            for method in ("staged", "monolithic_f0"):
                pair = [r for r in records if r["case"] == case and r["method"] == method]
                require(len(pair) == 2 and pair[0]["status"] == pair[1]["status"], "terminal loss")
                off, on = sorted(pair, key=lambda r: r["enabled"])
                require(on["f0_calls"]["attempted"] <= off["f0_calls"]["attempted"],
                        "unexpected extra joint F0 calls")
        mutated = copy.deepcopy(first_on_evidence)
        mutated["tier2"]["property_rows"].pop()
        issues = []
        _audit_safe_structure(mutated, issues, gate_work=work)
        require(bool(issues), "missing obligation was accepted")
        faults.append(dict(name="missing_obligation", status="PASS", issues=issues))
        save(run_dir / "fault_05.json", faults[-1])
        # Trusted source, wrong state: auditor must reject; it must not execute a loader
        # named by a package. No extra verification/solver call.
        wrong = build("C")
        audited = audit_evidence_package(first_on_package, replay_model=wrong,
                                        gate_work=work, deadline=deadline)
        require(audited["status"] == "FAIL" and "trusted replay model state mismatch" in audited["issues"],
                "replay binding not rejected")
        faults.append(dict(name="replay_binding", status="PASS", issues=audited["issues"]))
        save(run_dir / "fault_06.json", faults[-1])
        reject("reception_deadline_binding", lambda: audit_evidence_package(
            first_on_package, gate_work=work, deadline=deadline-1), ValueError)
        # Deliberately invalid zero-ledger fixture: never used for real checking,
        # immediately rejected; does not issue a second execution allowance.
        reject("reception_ledger_reset",
               lambda: require_continuation(GateWork(deadline), work.counts), ValueError)
        require([f["name"] for f in faults] == FAULTS, "fixed fault coverage")
        tick()
        result["status"] = "PASS"
    except BaseException as exc:
        result["error"] = repr(exc)
    result.update(work_counts=dict(work.counts), completed_calls=len(records),
                  completed_faults=len(faults))
    return result
