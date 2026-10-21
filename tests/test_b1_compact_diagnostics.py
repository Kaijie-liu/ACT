"""Same-solve numerical observations; fixed arrays, no optimizer/model queries."""
import copy
import hashlib
import json
import time
import unittest
from types import SimpleNamespace as NS

import numpy as np
import scipy.sparse as sp

from act.pipeline.moe.request_diagnostics import (
    RequestDiagnostics, DeferredCapture, compact_observation,
)
from act.pipeline.moe.request_budget import RequestBudget
from scoped_source.development_work import DevelopmentWork, POLICY
from scoped_source.native_capture import CaptureError, CaptureDeadline, CaptureCapacity


def fixture():
    def source(c, gc, gb):
        return NS(c=np.array(c, dtype=float), Gc=sp.csr_matrix(gc, dtype=float),
                  Gb=sp.csr_matrix(gb, dtype=float), n_cont=1, n_bin=1, n_out=len(c), frame_id=17)
    hz = source([2, 0, 1, 0], [[2], [0], [-1], [0]], [[3], [0], [5], [0]])
    router = source([0, 0, 0], [[1], [-1], [0]], [[2], [-2], [0]])
    support = NS(exact=True, lower_status=(0,), upper_status=(0,), solver_gap=(0.,), solves=2, elapsed=.01)
    encoding = NS(pair=(0, 1), property_row=(1., -1.), property_constant=.25,
        margin_bounds=(-4., 4.), bounds=NS(lambda_lower=.2, lambda_upper=.8,
            difference_lower=-5., difference_upper=5., product_lower=-2., product_upper=2.),
        margin_support=support, difference_support=support, output_hz=NS(frame_id=17))
    context = dict(path="pair", joint=NS(output_hz=hz, a_rows=(0, 1), b_rows=(2, 3)), router=router)
    model = NS(n_cont=3, n_bin=1, value_center=np.array([-1.25]))
    result = NS(x=np.array([.25, 0., -.5, .75]), fun=2., mip_dual_bound=-3.,
                status=1, mip_gap=.4, mip_node_count=12)
    objective = np.array([2., 1., 3., 4.])
    kwargs = dict(model=model, row=0, objective=objective, result=result)
    decision = NS(status="UNKNOWN", reason="LIMIT", minimum=-4.3, candidate_objective=.75,
                  elapsed=.01, solver_status=1, active_pair=(0, 1))
    return encoding, context, kwargs, decision


class CompactControls(unittest.TestCase):
    def setUp(self):
        self.work = DevelopmentWork(time.monotonic()+60, policy=POLICY)

    def observe(self, method="hz", change=None):
        e, ctx, kwargs, decision = fixture()
        holder = NS(scope={"pairs": [[0, 1]]}, encodings=[e], contexts=[ctx], pending=(method, kwargs))
        if change:
            change(holder, kwargs, decision)
        return compact_observation(holder, decision, {"valid": False}, self.work)

    def test_pair_offset_and_binary_layout(self):
        r = self.observe()
        self.assertEqual(r["raw_candidate_objective"], .75)
        self.assertEqual(r["fun_with_offset"], .75)
        self.assertEqual(r["mip_dual_bound_with_offset"], -4.25)
        self.assertEqual(r["decision"]["minimum"], -4.3)
        p = r["point"]
        self.assertEqual((p["b"], p["d"], p["m"], p["lambda_value"], p["w"]),
                         (3.5, .75, 2.5, .5, -1.))
        self.assertEqual(p["b_plus_w"], 2.5)
        self.assertEqual(p["f0_lowering_point_gap"], 1.75)
        self.assertEqual(p["product_gap"], -1.375)
        self.assertIsNone(r["dual_residual_correction"])
        self.assertGreater(sum(self.work.counts.values()), 0)

    def test_negative_margin_and_constant_ranges(self):
        def change(h, k, d):
            k["result"].x[0] = -1
            k["result"].x[3] = 0
            h.encodings[0].bounds.lambda_lower = .25
            h.encodings[0].bounds.lambda_upper = .25
            h.encodings[0].bounds.product_lower = -.5
            h.encodings[0].bounds.product_upper = -.5
        p = self.observe(change=change)["point"]
        self.assertEqual(p["m"], -6.)
        self.assertEqual(p["lambda_value"], .25)
        self.assertEqual(p["w"], -.5)
        self.assertAlmostEqual(p["sigma"], 1/(1+np.exp(6)))

    def mono(self, selector=1., active=(0, 2)):
        e, context, kwargs, decision = fixture()
        second = copy.deepcopy(e); second.pair = (0, 2)
        wrong = copy.deepcopy(context); wrong["router"].c[:] = 999
        # Branch zero has a different width; it is NOT the active source.
        local0 = NS(n_cont=2, n_bin=0, n_var=2)
        local1 = NS(n_cont=3, n_bin=1, n_var=4, value_center=np.array([-1.25]),
                    value_matrix=sp.csr_matrix([[2., 1., 3., 4.]]))
        objective = np.array([7., 0., 2., 1., 3., 4., 0., -1.25])
        model = NS(objective=objective, selector_offset=6, local_offsets=[0, 2], lowered=[local0, local1])
        result = NS(x=np.array([1e-5, 0., .25, 0., -.5, .75, 0., selector]),
                    status=1, fun=2., mip_dual_bound=-3., mip_gap=.4, mip_node_count=12)
        holder = NS(scope={"pairs": [[0, 1], [0, 2]]}, encodings=[e, second],
                    contexts=[wrong, context], pending=("monolithic", dict(model=model, result=result)))
        decision.active_pair = active
        return holder, decision

    def test_active_second_branch_and_global_objective(self):
        h, d = self.mono()
        r = compact_observation(h, d, None, self.work)
        self.assertAlmostEqual(r["raw_candidate_objective"], .75007)
        self.assertEqual(r["point"]["native_local_objective"], .75)
        self.assertEqual(r["point"]["pair"], [0, 2])
        self.assertEqual(r["point"]["m"], 1.25)
        self.assertEqual(r["point"]["b"], 3.5)

    def test_nonunit_selector_not_rounded(self):
        for selector in (.99999999, .5):
            h, d = self.mono(selector=selector)
            r = compact_observation(h, d, None, self.work)
            self.assertEqual(r["point_missing"], "NONUNIT_SELECTOR_NOT_PROJECTED")
            self.assertIsNone(r["point"])
            self.assertIsNotNone(r["raw_candidate_objective"])

    def test_missing_active_is_not_argmax(self):
        h, d = self.mono(active=None)
        r = compact_observation(h, d, None, self.work)
        self.assertEqual(r["point_missing"], "NO_ACCEPTED_ACTIVE_BRANCH")

    def test_missing_and_nonfinite_candidate(self):
        for value, reason in ((None, "NO_NATIVE_CANDIDATE"), (np.zeros(3), "CANDIDATE_SHAPE_MISMATCH"),
                              (np.array([0., 0., np.nan, 0.]), "NONFINITE_CANDIDATE")):
            r = self.observe(change=lambda h, k, d: setattr(k["result"], "x", value))
            self.assertEqual(r["point_missing"], reason)
            self.assertIsNone(r["point"])
            json.dumps(r, allow_nan=False)

    def test_scope_mismatch_rejected(self):
        with self.assertRaises(CaptureError):
            self.observe(change=lambda h, k, d: setattr(h.contexts[0]["router"], "frame_id", 99))

    def observer(self, *, numerical=True, limit=0, sink=None):
        o = RequestDiagnostics(work=self.work, byte_limit=8*2**20, publish=sink,
                               numerical_all=numerical, full_capture_limit=limit)
        ident = dict(config_sha256="b"*64, model_state={"sha256": "c"*64},
                     center={}, lower={}, upper={}, epsilon=.1, property={})
        o.bind("a"*64, ident, RequestBudget(60, started=self.work.deadline-60))
        o.routes([(0, 1)], [([1., -1.], .25), ([1., -1.], .25)], complete=True)
        return o

    def test_all_returns_even_after_first_unknown_and_release(self):
        published = []
        def sink(kind, raw, digest):
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            published.append((kind, json.loads(raw)))
        o = self.observer(sink=sink)
        for i in range(2):
            e, ctx, k, d = fixture()
            h = o.attempt(pairs=[e.pair], index=i, encodings=[e], contexts=[ctx])
            h.hz(**k)
            o.returned(d, replay_valid=False, replay={"valid": False, "prediction": 0, "topk_set": [0, 1]})
            self.assertIsNone(h.pending)
            self.assertIsNone(h.contexts)
            k["result"].x[:] = 123
        summary = o.finish("UNKNOWN")
        self.assertEqual(summary["observed_f0_calls"], 2)
        self.assertEqual(summary["native_calls"], 2)
        self.assertEqual([r["state"] for r in summary["obligations"]], ["RETURNED", "RETURNED"])
        self.assertEqual(o.observations[0]["point"]["m"], 2.5)
        self.assertEqual(sum(k == "attempt" for k, _ in published), 2)
        self.assertEqual(sum(k == "observation" for k, _ in published), 2)
        self.assertNotIn("observations", summary)  # references, not quadratic replication

    def test_first_ordinals_not_replaced_after_missing_source(self):
        o = self.observer(limit=1)
        for i in range(2):
            e, ctx, _, d = fixture()
            h = o.attempt(pairs=[e.pair], index=i, encodings=[e], contexts=[ctx])
            h.failure("SCIPY_UNAVAILABLE", native_called=False)
            o.returned(d, replay_valid=False)
        self.assertEqual(len(o.full_capture_events), 1)
        self.assertEqual(o.full_capture_events[0]["property_index"], 0)
        self.assertEqual(o.full_capture_events[0]["status"], "SOURCE_UNAVAILABLE")
        self.assertEqual(len(o.observations), 2)

    def test_default_preserves_first_only_contract(self):
        o = self.observer(numerical=False, limit=1)
        e, ctx, _, d = fixture()
        h = o.attempt(pairs=[e.pair], index=0, encodings=[e], contexts=[ctx])
        h.failure("SCIPY_UNAVAILABLE", native_called=False)
        o.returned(d, replay_valid=False)
        self.assertIsNone(o.attempt(pairs=[e.pair], index=1, encodings=[e], contexts=[ctx]))
        summary = o.finish("UNKNOWN")
        self.assertNotIn("numerical_all", summary)
        self.assertEqual(o.observations, {})

    def test_replay_mismatch_and_deadline(self):
        o = self.observer()
        e, ctx, k, d = fixture()
        h = o.attempt(pairs=[e.pair], index=0, encodings=[e], contexts=[ctx]); h.hz(**k)
        with self.assertRaises(CaptureError):
            o.returned(d, replay_valid=False, replay={"valid": True})
        self.assertIsNone(h.pending)
        self.work.deadline = time.monotonic()-1
        with self.assertRaises(CaptureDeadline):
            o.finish("UNKNOWN")

    def test_capacity_and_interrupted_attempt_stay_explicit(self):
        o = self.observer()
        e, ctx, _, _ = fixture()
        o.attempt(pairs=[e.pair], index=0, encodings=[e], contexts=[ctx])
        summary = o.finish("TIMEOUT")
        self.assertEqual(summary["obligations"][0]["state"], "CENSORED")
        self.assertEqual(summary["observed_f0_calls"], 0)
        o.ledger.byte_limit = o.ledger.used
        with self.assertRaises(CaptureCapacity):
            o.emit()


class NativeIntegrationControls(unittest.TestCase):
    """Two complete synthetic calls only; no checkpoint, dataset or GPU."""

    def test_two_numeric_paths_and_first_two_obligations(self):
        from pathlib import Path
        import torch
        from act.util.device_manager import initialize_device
        from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
        from act.pipeline.moe.staged_verifier import verify_staged_linf
        initialize_device("cpu", "float64")
        torch.set_num_threads(1)
        model = build_output_moe(OutputMoEFactoryConfig(input_shape=(2,), num_classes=10,
            num_experts=3, top_k=2, gate=GateKind.SELECTED_SOFTMAX,
            router_hidden=(), expert_hidden=(), seed=7)).cpu().double().eval()
        with torch.no_grad():
            model.router[1].weight.zero_(); model.router[1].bias.zero_()
            for expert, first in zip(model.experts, (-1., 1., 1.)):
                expert[1].weight.zero_()
                expert[1].bias.copy_(torch.tensor([first, 0.]+[-2.]*8, dtype=torch.float64))
        center = torch.full((1, 2), .5, dtype=torch.float64)
        cfg = json.loads((Path(__file__).resolve().parents[1]/
            "act/pipeline/moe/configs/route_complexity_reuse_v1.json").read_text())
        started = time.monotonic()
        for path, native_calls in (("staged", 2), ("monolithic_f0", 1)):
            config = copy.deepcopy(cfg); config["comparison_method"] = path
            work = DevelopmentWork(started+300, policy=POLICY)
            records = []
            def sink(kind, raw, digest):
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                records.append((kind, json.loads(raw)))
            observer = RequestDiagnostics(work=work, byte_limit=16*2**20, publish=sink,
                numerical_all=True, full_capture_limit=2)
            report = verify_staged_linf(model, center, .1, config,
                budget_started_at=started, diagnostics=observer)
            self.assertEqual(report.status, "UNKNOWN")
            diag = report.evidence["diagnostics"]
            self.assertEqual(len(diag["obligations"]), 27)
            self.assertEqual(diag["registered_f0_calls"], native_calls)
            self.assertEqual(diag["observed_f0_calls"], native_calls)
            self.assertEqual(len(diag["full_capture_events"]), 2)
            self.assertTrue(all(e["status"] == "CAPTURED" for e in diag["full_capture_events"]))
            captured = [v for kind, v in records if kind == "native"]
            self.assertEqual(len(captured), 2)
            self.assertEqual([v["source_metadata"]["selected_pair"] for v in captured], [[0, 1], [0, 2]])
            self.assertEqual([v["source_metadata"]["source_branch_index"] for v in captured],
                             [0, 1] if path == "monolithic_f0" else [0, 0])
            self.assertTrue(all(v["source_metadata"]["guarantee"] ==
                "NUMERICAL_SOURCE_OBSERVATION_NOT_CHECKED_LOWERING" for v in captured))
            observed = [v for kind, v in records if kind == "observation"]
            self.assertEqual(len(observed), native_calls)
            self.assertTrue(all(v["replay"]["valid"] is False for v in observed))
            self.assertTrue(all(v["point"] is not None for v in observed))
            self.assertTrue(all(abs(v["point"]["f0_lowering_point_gap"]) < 1e-12 for v in observed))
            self.assertLess(observer.ledger.used, 16*2**20)


if __name__ == "__main__":
    unittest.main()
