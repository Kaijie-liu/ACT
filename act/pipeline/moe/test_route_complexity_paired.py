import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from act.pipeline.moe.route_complexity_paired import DEFAULT, artifacts, jobs, smoke_gate, counters, common_facts


class PairingTests(unittest.TestCase):
    def test_registered_prefix_and_method_identity(self):
        config=json.loads(DEFAULT.read_text()); selection, methods=artifacts(config)
        self.assertEqual(config["ranks"],list(range(10)))
        self.assertEqual(config["smoke_ranks"],[0])
        self.assertEqual(methods["adaptive"]["numerical_safety"], methods["monolithic"]["numerical_safety"])
        scheduled=jobs(selection,config["ranks"])
        self.assertEqual(len(scheduled),60)
        self.assertEqual(len({j["job_id"] for j in scheduled}),60)
        for model in selection["models"]:
            for arm in methods:
                for position in (0,1):
                    self.assertEqual(sum(j["model"]==model and j["method"]==arm and j["position"]==position for j in scheduled),5)

    def test_no_full_without_completed_smoke(self):
        config=json.loads(DEFAULT.read_text())
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as tmp:
            config["smoke_output"]=tmp
            with self.assertRaisesRegex(ValueError,"completed audited smoke"):
                smoke_gate(DEFAULT,config,"source")

    def test_reject_smoke_from_different_code(self):
        config=json.loads(DEFAULT.read_text())
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as tmp:
            config["smoke_output"]=tmp
            (Path(tmp)/"audit.final.json").write_text('{}')
            (Path(tmp)/"runtime.json").write_text(json.dumps({"smoke":True,"config":config,
                "config_sha256":"wrong","source_sha256":"old"}))
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                smoke_gate(DEFAULT,config,"new")

    def test_reject_budget_mismatch_and_unfrozen_ranks(self):
        cfg=json.loads(DEFAULT.read_text());cfg["budget_seconds"]=301
        with self.assertRaisesRegex(ValueError,"budget mismatch"):artifacts(cfg)
        cfg=json.loads(DEFAULT.read_text());cfg["ranks"]=[1,2]
        with self.assertRaisesRegex(ValueError,"prefix"):artifacts(cfg)

    def test_censored_counter_not_zero(self):
        self.assertIsNone(counters({"tier2":{"partial_rows_censored":True}}))
        self.assertEqual(counters({"tier2":{}}),{"reused_pair_properties":0,"recorded_weighted_query_rows":0})

    def test_common_fact_comparison_ignores_only_execution_ids(self):
        e={"route_complexity_schedule":{"common_fact_prelude_complete":True},
           "route_coverage":{"candidate_experts":[0,1],"feasible_route_sets":[[0,1]]},
           "tier1":{"branches":[{"candidate":0,"proof_output_bounds":{"lower":[0,1],"upper":[1,2]},"source_policy":"common"}]},
           "proof_reuse":{"available_fact_count":1,"frame_id":"a"},"request_id":"a"}
        other=copy.deepcopy(e);other["request_id"]="b";other["proof_reuse"]["frame_id"]="b"
        self.assertEqual(common_facts(e),common_facts(other))
        other["tier1"]["branches"][0]["proof_output_bounds"]["lower"][0]=.1
        self.assertNotEqual(common_facts(e),common_facts(other))
        other["route_complexity_schedule"]["common_fact_prelude_complete"]=False
        self.assertIsNone(common_facts(other))


if __name__=="__main__":unittest.main()
