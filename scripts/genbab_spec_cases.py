"""Fixed stdlib-only B2 encoding controls; no solver or trained-source calls."""
import copy
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("genbab_spec", ROOT/"act/pipeline/moe/genbab_spec.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

TESTS = (
    "test_guard_orientation", "test_selected_order_not_score_order", "test_ties_included",
    "test_property_offset", "test_and_inside_or_outside", "test_two_expert_empty_guard",
    "test_full_eight_expert_ten_class_roster", "test_dimensions_and_final_class",
    "test_whole_box_scope_separate", "test_binary64_conversion", "test_inexact_cast_rejected",
    "test_all_results_positive", "test_missing_not_positive", "test_duplicate_rejected",
    "test_source_binding_rejected", "test_spec_tampering_rejected", "test_wrong_engine_rejected",
    "test_sat_needs_replay", "test_timeout_and_error", "test_deadline_continues",
    "test_request_roster_tampering", "test_bad_coefficients", "test_alias_independence",
    "test_unstarted_cannot_decide", "test_actual_stage_retained",
)


class Cases(unittest.TestCase):
    deadline = 0

    def request(self, experts=4, classes=3, mode="guarded", properties=None):
        if properties is None:
            properties = [dict(id="margin", q=[1, -1] + [0]*(classes-2), c="1/4")]
        return api.make_request(model_sha256="1"*64, input_sha256="2"*64,
                                graph_sha256="3"*64, experts=experts, classes=classes,
                                properties=properties, mode=mode, deadline=self.deadline)

    def rows(self, request=None):
        return api.clauses(request or self.request(), deadline=self.deadline)

    def receipts(self, request):
        return [dict(request_sha256=api.digest(request), clause_sha256=api.digest(row),
                     engine=api.ENGINE, stage="nonlinear_bab", status="UNSAT")
                for row in self.rows(request)]

    def receive(self, request, receipts):
        return api.receive(request, receipts, deadline=self.deadline)

    def holds(self, row, outputs):
        return all(sum(Q(c)*Q(v) for c,v in zip(coeff,outputs)) <= Q(rhs)
                   for coeff,rhs in zip(row["C"],row["rhs"]))

    def test_guard_orientation(self):
        row = self.rows()[0]  # pair [0,1]
        self.assertTrue(self.holds(row, [-1,0,0, 2,1,0,-1]))
        self.assertFalse(self.holds(row, [-1,0,0, 2,1,3,-1]))

    def test_selected_order_not_score_order(self):
        self.assertTrue(self.holds(self.rows()[0], [-1,0,0, 1,2,0,-1]))

    def test_ties_included(self):
        self.assertTrue(all(self.holds(row, [-1,0,0, 1,1,1,1]) for row in self.rows()))

    def test_property_offset(self):
        row = self.rows()[0]
        self.assertTrue(self.holds(row, [Q(-1,4),0,0, 2,1,0,-1]))
        self.assertFalse(self.holds(row, [0,0,0, 2,1,0,-1]))
        self.assertEqual(row["rhs"][-1], "-1/4")

    def test_and_inside_or_outside(self):
        values = [-1,0,0, 4,3,2,1]
        self.assertEqual(sum(self.holds(row, values) for row in self.rows()), 1)
        self.assertEqual(self.rows()[0]["logic"], "UNSAFE_AND")

    def test_two_expert_empty_guard(self):
        rows = self.rows(self.request(experts=2))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["C"]), 1)

    def test_full_eight_expert_ten_class_roster(self):
        props = [dict(id=str(i), q=[int(j==9)-int(j==i) for j in range(10)], c=0)
                 for i in range(9)]
        rows = self.rows(self.request(experts=8, classes=10, properties=props))
        self.assertEqual(len(rows), 252)
        self.assertEqual(sum(len(r["C"]) for r in rows), 3276)
        self.assertTrue(all(len(c)==18 for r in rows for c in r["C"]))

    def test_dimensions_and_final_class(self):
        req = self.request(experts=3, classes=5,
                           properties=[dict(id="last", q=[-1,0,0,0,1], c=-2)])
        row = self.rows(req)[0]
        self.assertEqual(row["C"][-1], ["-1","0","0","0","1","0","0","0"])
        self.assertEqual(row["rhs"][-1], "2")

    def test_whole_box_scope_separate(self):
        req = self.request(mode="whole_box")
        self.assertTrue(all(len(r["C"])==1 for r in self.rows(req)))
        self.assertEqual(self.receive(req,self.receipts(req))["scope"], "ALL_PAIRS_WHOLE_BOX_SUFFICIENT")
        with self.assertRaises(ValueError):
            self.receive(self.request(),self.receipts(req))

    def test_binary64_conversion(self):
        req = self.request(); row = self.rows(req)[0]
        c,rhs = api.raw_binary64(req,row,deadline=self.deadline)
        self.assertEqual([[str(Q(v)) for v in r] for r in c], row["C"])
        self.assertEqual([str(Q(v)) for v in rhs], row["rhs"])

    def test_inexact_cast_rejected(self):
        req=self.request(properties=[dict(id="third",q=[1,-1,0],c="1/3")])
        row=self.rows(req)[0]
        with self.assertRaises(ValueError): api.raw_binary64(req,row,deadline=self.deadline)

    def test_all_results_positive(self):
        req=self.request(); result=self.receive(req,self.receipts(req))
        self.assertEqual(result["status"],"POSITIVE")
        self.assertFalse(result["formal_SAFE"])

    def test_missing_not_positive(self):
        req=self.request(); result=self.receive(req,self.receipts(req)[:-1])
        self.assertEqual(result["status"],"UNKNOWN")
        self.assertEqual(len(result["missing"]),1)

    def test_duplicate_rejected(self):
        req=self.request(); receipts=self.receipts(req)
        with self.assertRaises(ValueError): self.receive(req,receipts+[receipts[0]])

    def test_source_binding_rejected(self):
        req=self.request(); receipts=self.receipts(req)
        req["input_sha256"]="4"*64
        with self.assertRaises(ValueError): self.receive(req,receipts)

    def test_spec_tampering_rejected(self):
        req=self.request(); row=self.rows(req)[0]
        for key,value in [("logic","UNSAFE_OR"),("pair",[1,0]),("rhs",["0"]*5),
                          ("property_id","foreign")]:
            bad=copy.deepcopy(row); bad[key]=value
            with self.assertRaises(ValueError): api.validate_clause(req,bad,deadline=self.deadline)
        bad=copy.deepcopy(row); bad["C"][0]=[str(-Q(x)) for x in bad["C"][0]]
        with self.assertRaises(ValueError): api.validate_clause(req,bad,deadline=self.deadline)
        self.assertEqual(api.validate_clause(req,row,deadline=self.deadline),api.digest(row))

    def test_wrong_engine_rejected(self):
        req=self.request(); receipts=self.receipts(req); receipts[0]["engine"]="auto"
        with self.assertRaises(ValueError): self.receive(req,receipts)

    def test_sat_needs_replay(self):
        req=self.request(); receipts=self.receipts(req); receipts[0]["status"]="SAT"
        self.assertEqual(self.receive(req,receipts)["status"],"CANDIDATE_REPLAY_REQUIRED")

    def test_timeout_and_error(self):
        req=self.request(); receipts=self.receipts(req)
        for status in ("UNKNOWN","TIMEOUT","ERROR"):
            receipts[0]["status"]=status
            self.assertEqual(self.receive(req,receipts)["status"],status)

    def test_deadline_continues(self):
        req=self.request(); receipts=self.receipts(req)
        with self.assertRaises(TimeoutError): api.receive(req,receipts,deadline=time.monotonic()-1)
        with self.assertRaises(TimeoutError): api.clauses(req,deadline=time.monotonic()-1)

    def test_request_roster_tampering(self):
        req=self.request(); req["pairs"]=req["pairs"][:-1]
        with self.assertRaises(ValueError): self.rows(req)
        req=self.request(); req["target"]="qF+c>=0"
        with self.assertRaises(ValueError): self.rows(req)

    def test_bad_coefficients(self):
        for value in (float("nan"),float("inf"),True,2**4097):
            with self.assertRaises(ValueError): api.number(value)
        with self.assertRaises(ValueError):
            self.request(properties=[dict(id="bad",q=[1],c=0)])
        p=dict(id="dup",q=[1,0,-1],c=0)
        with self.assertRaises(ValueError): self.request(properties=[p,p])

    def test_alias_independence(self):
        req=self.request(properties=[dict(id="a",q=[1,-1,0],c=0),
                                     dict(id="b",q=[1,0,-1],c=0)])
        rows=self.rows(req); before=copy.deepcopy(rows[1]); rows[0]["C"][0][0]="9"
        self.assertEqual(rows[1],before)
        self.assertEqual(self.rows(req)[1],before)

    def test_unstarted_cannot_decide(self):
        req=self.request(); receipts=self.receipts(req); receipts[0]["stage"]="not_started"
        with self.assertRaises(ValueError): self.receive(req,receipts)

    def test_actual_stage_retained(self):
        req=self.request(); receipts=self.receipts(req)
        for r in receipts: r["stage"]="incomplete_only"
        result=self.receive(req,receipts)
        self.assertEqual(result["stages"]["nonlinear_bab"],0)
        self.assertEqual(result["status"],"POSITIVE")


def run(deadline, prefix):
    Cases.deadline=deadline
    records=[]
    for name in TESTS:
        api.tick(deadline)
        result=unittest.TestResult(); started=time.monotonic()
        Cases(name).run(result)
        records.append(dict(name=name,passed=result.wasSuccessful(),
                            seconds=time.monotonic()-started,
                            errors=[detail for _,detail in result.errors+result.failures]))
        prefix(records)
        if not result.wasSuccessful(): break
    return dict(status="PASS" if len(records)==len(TESTS) and all(r["passed"] for r in records)
                else "FAIL", records=records, synthetic=True, native_calls=0,
                mathematical_bound_checked=False, complete_network_proof=False)
