import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
from act.pipeline.moe.staged_verifier import verify_staged_linf, write_evidence_package
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package, _audit_safe_structure, _audit_schedule
from act.pipeline.moe.request_budget import RequestBudget, BudgetExhausted
from act.pipeline.moe import route_complexity_schedule as scheduler
from act.pipeline.moe import paired_monolithic


def config(method="staged"):
    value = json.loads((Path(__file__).parent / "configs/route_complexity_reuse_v1.json").read_text())
    value["comparison_method"] = method
    return value


def model(biases):
    net = build_output_moe(OutputMoEFactoryConfig(input_shape=(2,), num_classes=len(biases[0]),
        num_experts=len(biases), top_k=2, gate=GateKind.SELECTED_SOFTMAX,
        router_hidden=(), expert_hidden=(), seed=7)).cpu().double().eval()
    with torch.no_grad():
        net.router[1].weight.zero_(); net.router[1].bias.zero_()
        for expert, values in zip(net.experts, biases):
            expert[1].weight.zero_(); expert[1].bias.copy_(torch.tensor(values))
    return net


def run(net, method="staged", **kwargs):
    return verify_staged_linf(net, torch.full((1, 2), .5, dtype=torch.float64), .1, config(method), **kwargs)


class BudgetTests(unittest.TestCase):
    def test_live_remaining_and_never_zero(self):
        now = [10.]
        b = RequestBudget(10, clock=lambda: now[0])
        self.assertEqual(b.limit("first", cap=8), 8)
        now[0] += 6
        self.assertEqual(b.limit("second", obligations=2), 2)
        self.assertEqual(b.limit("slice", until=17), 1)
        now[0] = 20
        with self.assertRaises(BudgetExhausted): b.limit("expired")

    def test_loading_charged_and_invalid_config_rejected(self):
        import time
        report = run(model(((2., 0.), (3., 0.))), budget_started_at=time.monotonic()-301)
        self.assertEqual(report.status, "TIMEOUT")
        self.assertFalse(report.evidence["route_coverage"]["coverage_complete"])
        for field, value in [("total_seconds", float("inf")), ("multi_pair_tier1_fraction", 1)]:
            cfg = config(); cfg["route_complexity_schedule"][field] = value
            with self.assertRaises(ValueError):
                verify_staged_linf(model(((2., 0.), (3., 0.))), torch.zeros(1,2), .1, cfg)


class ScheduleTests(unittest.TestCase):
    def audit(self, report):
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as root:
            p = Path(root)/"package"
            write_evidence_package(report, p)
            result = audit_evidence_package(p)
        self.assertEqual(result["issues"], [], result)

    def test_single_pair_keeps_variable_weighted_obligations(self):
        net = model(((0., 1., -2.), (3., 0., -2.)))
        with patch.object(scheduler, "_solve_output", side_effect=AssertionError("single-pair Tier1 solve")):
            a = run(net); b = run(net, "monolithic_f0")
        self.assertEqual((a.status,b.status), ("SAFE","SAFE"))
        self.assertEqual(a.evidence["route_complexity_schedule"]["selected_path"], "SINGLE_PAIR_DIRECT")
        self.assertEqual(a.evidence["verdict"]["decision_tier"], "MONOLITHIC_F0")
        self.assertEqual(a.evidence["tier1"]["branches"], b.evidence["tier1"]["branches"])
        self.assertEqual(a.evidence["proof_reuse"]["available_fact_count"], b.evidence["proof_reuse"]["available_fact_count"])
        rows = a.evidence["tier2"]["property_rows"]
        self.assertEqual(rows[0]["pair_count"], 1)
        self.assertEqual(rows[1]["pair_count"], 0)
        self.audit(a); self.audit(b)

    def test_tie_all_pairs_and_partial_monolithic_partition(self):
        net = model(((-.2,0.), (1.,0.), (2.,0.)))
        a = run(net); b = run(net, "monolithic_f0")
        self.assertEqual((a.status,b.status), ("SAFE","SAFE"))
        self.assertEqual(a.evidence["route_coverage"]["feasible_route_sets"], [[0,1],[0,2],[1,2]])
        self.assertEqual(a.evidence["route_complexity_schedule"]["selected_path"], "MULTI_PAIR_STAGED")
        row = b.evidence["tier2"]["property_rows"][0]
        self.assertEqual(row["pair_count"], 2)
        self.assertEqual(row["coverage_partition"]["solved_pairs"], [[0,1],[0,2]])
        self.assertEqual([v["pair"] for v in row["coverage_partition"]["reused"]], [[1,2]])
        # Common facts match across arms, even though one arm subsequently solves experts.
        self.assertEqual([v["proof_output_bounds"] for v in a.evidence["tier1"]["branches"]],
                         [v["proof_output_bounds"] for v in b.evidence["tier1"]["branches"]])
        self.audit(a); self.audit(b)
        for mutation in ("omit", "duplicate", "property", "bound", "path", "budget"):
            e = copy.deepcopy(b.evidence); r = e["tier2"]["property_rows"][0]
            if mutation == "omit": r["coverage_partition"]["solved_pairs"].pop()
            elif mutation == "duplicate": r["coverage_partition"]["solved_pairs"].append([1,2])
            elif mutation == "property": r["coverage_partition"]["reused"][0]["proof"]["property_index"] = 9
            elif mutation == "bound": r["minimum"] = 100.
            elif mutation == "path": e["route_complexity_schedule"]["selected_path"] = "SINGLE_PAIR_DIRECT"
            else: e["route_complexity_schedule"]["budget"]["events"][0]["granted_seconds"] = 9999.
            issues=[]; _audit_safe_structure(e,issues); _audit_schedule(e,issues)
            self.assertTrue(issues,mutation)

    def test_all_reused_skips_monolithic_propagation(self):
        with patch.object(paired_monolithic, "shared_input_pair_propagation", side_effect=AssertionError("unneeded")):
            report = run(model(((2.,0.),(3.,0.),(4.,0.))), "monolithic_f0")
        self.assertEqual(report.status,"SAFE")
        self.assertEqual(report.evidence["tier2"]["property_rows"][0]["pair_count"],0)
        self.audit(report)

    def test_incomplete_route_analysis_does_not_invoke_output_verification(self):
        original = scheduler.analyze_topk_sets
        def incomplete(*args, **kw): return replace(original(*args, **kw), exact=False)
        with patch.object(scheduler,"analyze_topk_sets",side_effect=incomplete), patch.object(
            scheduler,"_solve_output",side_effect=AssertionError("should not solve")):
            report=run(model(((2.,0.),(3.,0.))))
        self.assertEqual(report.status,"UNKNOWN")
        self.assertFalse(report.evidence["tier2"]["invoked"])
        self.assertFalse(report.evidence["route_coverage"]["coverage_complete"])

    def test_budget_during_weighted_construction_cannot_produce_safe(self):
        with patch.object(paired_monolithic,"build_weighted_top2_f0",side_effect=BudgetExhausted("control")):
            report=run(model(((0.,1.),(3.,0.))))
        self.assertEqual(report.status,"TIMEOUT")
        self.assertTrue(report.evidence["tier2"]["partial_rows_censored"])
        self.assertGreater(report.evidence["tier2"]["elapsed_seconds"],0)
        self.audit(report)

    def test_weighted_witness_replayed_and_returns_immediately(self):
        net=model(((0.,0.),(0.,0.)))
        with torch.no_grad():
            for expert in net.experts:
                expert[1].weight[0,0]=1.
                expert[1].bias[0]=-.45
        for method in ("staged","monolithic_f0"):
            report=run(net,method)
            self.assertEqual(report.status,"UNSAFE")
            self.assertIsNotNone(report.witness)
            x=report.witness.reshape(1,2)
            self.assertTrue(bool(((x>=.4)&(x<=.6)).all()))
            self.assertNotEqual(int(net(x).argmax(1).item()), report.evidence["request"]["clean_prediction"])
            self.assertTrue(report.evidence["verdict"]["full_model_witness_valid"])

    def test_tier1_witness_short_circuits_remaining_experts(self):
        net=model(((0.,0.),(0.,0.),(0.,0.)))
        with torch.no_grad():
            for expert in net.experts:
                expert[1].weight[0,0]=1.
                expert[1].bias[0]=-.45
        with patch.object(scheduler,"_solve_output",wraps=scheduler._solve_output) as solve:
            report=run(net)
        self.assertEqual(report.status,"UNSAFE")
        self.assertEqual(solve.call_count,1)
        self.assertFalse(report.evidence["tier2"]["invoked"])

    def test_common_fact_exhaustion_and_query_deadline_stop(self):
        original=scheduler._propagate_component
        calls=[]
        def abort_second(*a,**kw):
            calls.append(1)
            if len(calls)==2: raise BudgetExhausted("common_fact_test")
            return original(*a,**kw)
        with patch.object(scheduler,"_propagate_component",side_effect=abort_second):
            report=run(model(((2.,0.),(3.,0.))))
        self.assertEqual(report.status,"TIMEOUT")
        self.assertFalse(report.evidence["route_complexity_schedule"]["common_fact_prelude_complete"])
        self.assertFalse(report.evidence["tier2"]["invoked"])
        self.audit(report)


if __name__ == "__main__": unittest.main()
