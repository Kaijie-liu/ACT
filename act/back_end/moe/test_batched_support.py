"""Fixed synthetic CPU controls; no data, model, native solver or CUDA work."""
import copy
from fractions import Fraction as F
import math
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.moe import batched_support as api
from act.back_end.moe.check_batched_support import check_batch, validated_records
from act.back_end.solver.solver_hz import SparseHZono
from scoped_source.rowwise_bound import identity


CONTEXT = {"request": "synthetic-fixed-controls", "domain": "factor-box-minus-plus-one",
           "guard": "two-sided-zero"}


def guarded():
    return SparseHZono(c=np.array([0.]), Gc=sp.csr_matrix([[1.]]), Gb=sp.csr_matrix((1, 0)),
                      Ac=sp.csr_matrix((0, 1)), Ab=sp.csr_matrix((0, 0)), b=np.array([]),
                      Auc=sp.csr_matrix([[1.], [-1.]]), Aub=sp.csr_matrix((2, 0)),
                      ub=np.array([0., 0.]), frame_id=314)


def query(key="plus-min", q=(1,), offset="1/4", side="min"):
    return {"id": key, "q": list(q), "offset": offset, "side": side}


def four_queries():
    return [query(sign + "-" + side, (value,), side=side)
            for sign, value in (("plus", 1), ("minus", -1)) for side in ("min", "max")]


class BatchedSupportTests(unittest.TestCase):
    observations = {}

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.end = time.monotonic() + 300

    def prepare(self, hz=None, queries=None, context=None):
        return api.prepare_batch(hz or guarded(), queries or four_queries(),
                                 context=CONTEXT if context is None else context, deadline=self.end)

    def propose(self, batch):
        return api.propose_batch(batch, expected_batch_sha256=identity(batch), deadline=self.end)

    def check(self, batch, candidates, anchor=None):
        return check_batch(batch, candidates, expected_batch_sha256=anchor or identity(batch), deadline=self.end)

    def run_support(self, hz=None, queries=None, name=None):
        package = api.support_batch(hz or guarded(), queries or four_queries(), context=CONTEXT, deadline=self.end)
        if name:
            self.observations[name] = package
        return package

    def test_guarded_two_sides_positive_and_max_offset(self):
        p = self.run_support(name="guarded_two_sides_positive")
        self.assertEqual(len(p["accepted"]["results"]), 4)
        for q, result in zip(p["batch"]["queries"], p["accepted"]["results"]):
            bound = F(result["bound"])
            if result["side"] == "min":
                self.assertGreater(bound, F(1, 10**7))
                self.assertLessEqual(bound, F(1, 4))
            else:
                self.assertGreaterEqual(bound, F(1, 4))
                self.assertLess(bound, F(1, 2))
                self.assertEqual(F(q["constant"]), -F(1, 4))
        self.assertFalse(p["accepted"]["network_or_complete_moe_proof"])
        self.assertFalse(p["accepted"]["hard_budget_supervision"])
        costs = p["accepted"]["cost_seconds"]
        self.assertAlmostEqual(sum(costs[k] for k in ("preparation", "proposal", "checking")), costs["total"])
        self.assertGreater(costs["total"], 0)

    def test_guarded_nonpositive_is_not_unsafe(self):
        p = self.run_support(queries=[query(offset="-1/4")], name="guarded_nonpositive")
        self.assertLessEqual(F(p["accepted"]["results"][0]["bound"]), -F(1, 4))
        self.assertNotIn(p["accepted"]["status"], ("SAFE", "UNSAFE"))

    def test_equality_coupled(self):
        hz = SparseHZono(c=np.zeros(2), Gc=sp.eye(2, format="csr"), Gb=sp.csr_matrix((2, 0)),
                          Ac=sp.csr_matrix([[1., -1.]]), Ab=sp.csr_matrix((1, 0)), b=np.zeros(1))
        p = self.run_support(hz, [query(q=(1, -1))], "equality_coupled")
        bound = F(p["accepted"]["results"][0]["bound"])
        self.assertGreater(bound, F(1, 10**7))
        self.assertLessEqual(bound, F(1, 4))

    def test_private_binary_relaxation_not_shared(self):
        hz = SparseHZono(c=np.zeros(2), Gc=sp.csr_matrix([[1.], [1.]]), Gb=sp.eye(2, format="csr"),
                          Ac=sp.csr_matrix((0, 1)), Ab=sp.csr_matrix((0, 2)), b=np.array([]), frame_id=314)
        p = self.run_support(hz, [query(side, (1, -1), 0, side) for side in ("min", "max")],
                             "private_binary_relaxation")
        self.assertEqual(p["batch"]["base"]["lower"], [-1, -1, -1])
        self.assertEqual(p["accepted"]["n_relaxed_binaries"], 2)
        self.assertEqual([F(r["bound"]) for r in p["accepted"]["results"]], [-2, 2])
        # Both binary columns remain present despite equal shared continuous columns.
        self.assertEqual(p["batch"]["queries"][0]["c"], ["0", "1", "-1"])
        for mutate in (lambda b: b["base"]["lower"].__setitem__(1, 0),
                       lambda b: b.__setitem__("n_relaxed_binaries", 1)):
            damaged = copy.deepcopy(p["batch"])
            mutate(damaged)
            with self.assertRaises(ValueError):
                validated_records(damaged, expected_batch_sha256=identity(damaged), deadline=self.end)

    def test_constant_objective(self):
        p = self.run_support(queries=[query(side, (0,), "1/4", side) for side in ("min", "max")],
                             name="constant_objective")
        self.assertEqual([F(r["bound"]) for r in p["accepted"]["results"]], [F(1, 4)] * 2)

    def test_batch_single_and_reordering(self):
        queries = four_queries()
        p = self.run_support(queries=queries, name="batch_differential_reference")
        by_id = {r["id"]: r for r in p["accepted"]["results"]}
        runs = [self.run_support(queries=[q]) for q in queries]
        runs.append(self.run_support(queries=list(reversed(queries))))
        for other in runs:
            for row in other["accepted"]["results"]:
                reference = by_id[row["id"]]
                self.assertEqual(row["lp_sha256"], reference["lp_sha256"])
                self.assertLessEqual(abs(F(row["bound"]) - F(reference["bound"])), F(1, 10**10))
            self.assertNotEqual(other["accepted"]["batch_sha256"], p["accepted"]["batch_sha256"])

    def test_self_consistent_wrong_source_query_side_or_request(self):
        original = self.prepare(queries=[query()])
        anchor = identity(original)
        for kind in ("source", "q", "offset", "side", "request", "domain", "guard"):
            hz, queries, context = guarded(), [query()], dict(CONTEXT)
            if kind == "source": hz.c[0] = 0.5
            elif kind == "q": queries[0]["q"] = [-1]
            elif kind == "offset": queries[0]["offset"] = "3/4"
            elif kind == "side": queries[0]["side"] = "max"
            else: context[kind] += "-other"
            wrong = self.prepare(hz, queries, context)
            # Fresh internally consistent certificate still cannot answer old request.
            candidates = self.propose(wrong)
            with self.assertRaises(ValueError):
                self.check(wrong, candidates, anchor)
        candidates = self.propose(original)
        for anchor in (None, "", "0" * 64):
            with self.assertRaises(ValueError):
                check_batch(original, candidates, expected_batch_sha256=anchor, deadline=self.end)

    def test_missing_duplicate_and_misassigned_obligations(self):
        batch = self.prepare()
        candidates = self.propose(batch)
        for change in (lambda c: c["entries"].pop(),
                       lambda c: c["entries"].append(copy.deepcopy(c["entries"][0])),
                       lambda c: c["entries"][1].__setitem__("id", c["entries"][0]["id"]),
                       lambda c: c["entries"][0].__setitem__("id", "other-query"),
                       lambda c: c["entries"][0].__setitem__("certificate", c["entries"][1]["certificate"])):
            wrong = copy.deepcopy(candidates)
            change(wrong)
            with self.assertRaises(ValueError): self.check(batch, wrong)
        reordered = copy.deepcopy(candidates)
        reordered["entries"].reverse()
        self.assertEqual(self.check(batch, reordered)["results"], self.check(batch, candidates)["results"])
        wrong = copy.deepcopy(batch)
        wrong["queries"].pop()
        with self.assertRaises(ValueError): self.check(wrong, self.propose(wrong), identity(batch))
        with self.assertRaises(ValueError): self.prepare(queries=[query(), query()])
        # Equal properties with distinct IDs are still separately required obligations.
        p = self.run_support(queries=[query("a"), query("b")])
        self.assertEqual(len(p["accepted"]["results"]), 2)
        partial = copy.deepcopy(candidates)
        partial["entries"].pop()
        self.observations["partial_candidate_rejected"] = {
            "batch": batch, "candidates": partial, "required": len(batch["queries"]),
            "received": len(partial["entries"]), "complete": False,
            "expected_rejection": "incomplete support certificate roster"}

    def test_independent_lowering_rejects_newly_hashed_wrong_coefficients(self):
        batch = self.prepare(queries=[query()])
        for change in (lambda b: b["queries"][0]["c"].__setitem__(0, "2"),
                       lambda b: b["queries"][0].__setitem__("constant", "5"),
                       lambda b: b["base"]["A"]["data"].__setitem__(0, -1),
                       lambda b: b["base"]["b"].__setitem__(0, 1)):
            wrong = copy.deepcopy(batch)
            change(wrong)
            with self.assertRaises(ValueError):
                validated_records(wrong, expected_batch_sha256=identity(wrong), deadline=self.end)

    def test_corrupt_dual_claim_and_exact_residual(self):
        batch = self.prepare(queries=[query()])
        c = self.propose(batch)
        for key, value in (("inequality_dual", [1, 0]), ("inequality_dual", [0]),
                           ("inequality_dual", [math.nan, 0]), ("inequality_dual", [math.inf, 0]),
                           ("equality_dual", [0]), ("claimed_lower_bound", "1"), ("lp_sha256", "bad")):
            wrong = copy.deepcopy(c)
            wrong["entries"][0]["certificate"][key] = value
            with self.assertRaises(ValueError): self.check(batch, wrong)
        c["entries"][0]["certificate"].update(inequality_dual=[0, -0.5], claimed_lower_bound="-1/4")
        accepted = self.check(batch, c)
        self.assertEqual(F(accepted["results"][0]["bound"]), -F(1, 4))

    def test_invalid_csr_nonfinite_shapes_and_zero_dual_rows(self):
        for key in ("c", "ub"):
            hz = guarded()
            getattr(hz, key)[0] = math.nan
            with self.assertRaises(ValueError): self.prepare(hz)
        hz = guarded()
        hz.Gc = sp.csr_matrix(([1., 1.], [0, 0], [0, 2]), shape=(1, 1))
        with self.assertRaises(ValueError): self.prepare(hz)
        hz = guarded()
        hz.Auc.data[0] = math.inf
        with self.assertRaises(ValueError): self.prepare(hz)
        for q in (query(q=(1, 2)), query(side="both"), query(offset=math.inf)):
            with self.assertRaises(ValueError): self.prepare(queries=[q])
        with self.assertRaises(ValueError): self.prepare(context={})
        # Explicit zero/zero-dual rows must still be parsed by original-LP check.
        batch = self.prepare(queries=[query(q=(0,))])
        c = self.propose(batch)
        broken = copy.deepcopy(batch)
        broken["base"]["A"]["indices"][0] = 3
        with self.assertRaises(ValueError):
            validated_records(broken, expected_batch_sha256=identity(broken), deadline=self.end)
        self.assertEqual(self.check(batch, c)["results"][0]["rows_checked"], 2)

    def test_candidate_exception_partial_nonfinite_and_pollution(self):
        batch = self.prepare(queries=[query()])
        for returned in (([], []), ([[math.inf, 0]], [[]])):
            with patch.object(api, "_candidate_columns", return_value=returned):
                with self.assertRaises((ValueError, OverflowError)):
                    self.propose(batch)
        with patch.object(api, "_candidate_columns", side_effect=RuntimeError("controlled proposal fault")):
            with self.assertRaises(RuntimeError): self.run_support(queries=[query()])
        original = api._candidate_columns
        def pollute(*args, **kwargs):
            result = original(*args, **kwargs)
            batch["context"]["guard"] = "mutated"
            return result
        with patch.object(api, "_candidate_columns", side_effect=pollute):
            with self.assertRaises(ValueError): self.propose(batch)

    def test_deadline_before_preparation_check_and_final_reception(self):
        for end in (time.monotonic() - 1, time.monotonic() + 301, math.inf):
            with self.assertRaises((ValueError, TimeoutError)):
                api.support_batch(guarded(), [query()], context=CONTEXT, deadline=end)
        p = self.run_support(queries=[query()])
        with self.assertRaises(TimeoutError):
            check_batch(p["batch"], p["candidates"], expected_batch_sha256=identity(p["batch"]),
                        deadline=time.monotonic() - 1)
        # Force expiry after a valid check: final reception must still fail closed.
        real_clock = api.clock
        expired = [False]
        def clock(end):
            tick = real_clock(end)
            def checked_tick():
                if expired[0]: raise TimeoutError("controlled late reception")
                tick()
            return checked_tick
        original = api.check_batch
        def late(*args, **kwargs):
            result = original(*args, **kwargs)
            expired[0] = True
            return result
        with patch.object(api, "clock", side_effect=clock), patch.object(api, "check_batch", side_effect=late):
            with self.assertRaises(TimeoutError): self.run_support(queries=[query()])

    def test_cuda_rejected_before_any_initialization(self):
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA touched")) as init:
            for device in ("cuda", "cuda:0", "mps"):
                with self.assertRaises(ValueError):
                    api.support_batch(guarded(), [query()], context=CONTEXT, deadline=self.end, device=device)
            init.assert_not_called()

    def test_cpu_allocation_ignores_process_default_device(self):
        # A process-default meta device detects missing explicit CPU allocation
        # without ever initializing or allocating on CUDA.
        with torch.device("meta"):
            p = self.run_support(queries=[query()])
        self.assertEqual(p["candidates"]["device"], "cpu")

    def test_archive_inventory_cannot_omit_controls_or_evidence(self):
        from scripts.run_hz_batch_support_controls import (FILES, OBSERVATIONS, PROTOCOL_SHA,
                                                          check_inventory, expected_tests)
        names = expected_tests()
        scope = {"native_solves": 0, "gpu_executions": 0, "real_requests": 0}
        original = {"execution": {"protocol_sha256": PROTOCOL_SHA, "tests": names, **scope},
                    "summary": {"tests": len(names), "outcomes": [{"test": n} for n in names], **scope},
                    "bindings": dict.fromkeys(FILES, "bound-by-separate-hash-check"),
                    "observations": dict.fromkeys(OBSERVATIONS), "anchors": dict.fromkeys(OBSERVATIONS)}
        check_inventory(**original)
        for key in ("execution", "summary", "bindings", "observations", "anchors"):
            wrong = copy.deepcopy(original)
            if key == "execution": wrong[key]["tests"].pop()
            elif key == "summary": wrong[key]["tests"] -= 1
            else: wrong[key].pop(next(iter(wrong[key])))
            with self.assertRaises(ValueError): check_inventory(**wrong)


if __name__ == "__main__":
    unittest.main()
