"""Finite no-training controls for scoped interval reuse and exact LP checking."""
import argparse
import json
from pathlib import Path

import torch

from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
from act.back_end.solver.lp_certificate import check, propose
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _git_value
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.staged_verifier import verify_staged_linf, write_evidence_package


def run(output):
    if _git_value("branch", "--show-current") != "feat/moe-route-verification" or _git_value("status", "--porcelain"):
        raise RuntimeError("clean feature branch required")
    output = _inside(output, WRITE_ROOT)
    output.mkdir(exist_ok=False)
    model = build_output_moe(OutputMoEFactoryConfig(
        input_shape=(2,), num_classes=3, num_experts=2, top_k=2,
        gate=GateKind.SELECTED_SOFTMAX, router_hidden=(), expert_hidden=(), seed=7,
    )).cpu().double().eval()
    with torch.no_grad():
        model.router[1].weight.zero_()
        model.router[1].bias.zero_()
        for expert, bias in zip(model.experts, ((0., 1., -2.), (3., 0., -2.))):
            expert[1].weight.zero_()
            expert[1].bias.copy_(torch.tensor(bias))
    records = {}
    for enabled in (False, True):
        config = json.loads((PROJECT_ROOT / "act/pipeline/moe/configs/staged_verifier_v1.json").read_text())
        config["scoped_proof_reuse"] = enabled
        report = verify_staged_linf(model, torch.full((1, 2), .5, dtype=torch.float64), .1, config)
        name = "reuse" if enabled else "reference"
        package = output / name
        write_evidence_package(report, package)
        audit = audit_evidence_package(package)
        save(output / f"{name}.audit.json", audit)
        properties = [r for pair in report.evidence["tier2"]["pairs"] for r in pair["property_rows"]]
        records[name] = {"status": report.status, "audit": audit,
                         "f0_solved_rows": sum(r["solver_bound_kind"] != "scoped_tier1_interval" for r in properties),
                         "reused_rows": report.evidence["tier2"]["reused_property_count"]}
        if audit["status"] != "PASS" or report.status != "SAFE":
            raise RuntimeError("control failed; retained artifacts")
    lp = {"c": [1, 1], "lower": [0, 0], "upper": [1, 1],
          "A": [[-1, 0]], "b": [-.5], "E": [[0, 1]], "h": [.25]}
    certificate = propose(lp)
    checked = check(lp, certificate)
    if checked["checked_lower_bound"] != "3/4":
        raise RuntimeError("analytic LP control disagrees")
    if (records["reference"]["f0_solved_rows"], records["reuse"]["f0_solved_rows"],
            records["reuse"]["reused_rows"]) != (2, 1, 1):
        raise RuntimeError("unexpected reuse counts")
    result = {"classification": "ANALYTIC_CONTROLS_NOT_OFFICIAL_SCALE_EFFECTIVENESS",
              "git_head": _git_value("rev-parse", "HEAD"), "records": records,
              "lp": lp, "certificate": certificate, "checked": checked,
              "scope": "Toy scoped-reuse consistency and exact supplied-LP bound; no independent HZ/MILP or deployment proof."}
    save(output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
