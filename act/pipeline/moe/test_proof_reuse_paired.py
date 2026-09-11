import json
import unittest

from act.pipeline.moe.proof_reuse_paired import jobs, DEFAULT
from act.pipeline.moe.experiment1 import _sha256


class ReuseScheduleTests(unittest.TestCase):
    def test_frozen_prefix_and_balanced_pairing(self):
        config = json.loads(DEFAULT.read_text())
        self.assertEqual(config["ranks"], list(range(10)))
        from pathlib import Path
        self.assertEqual(_sha256(Path(config["selection"])), config["selection_sha256"])
        selection = json.loads(Path(config["selection"]).read_text())
        records = jobs(selection, config["ranks"])
        self.assertEqual(len(records), 60)
        self.assertEqual(len({j["job_id"] for j in records}), 60)
        for model in selection["models"]:
            for enabled in (False, True):
                for position in (0, 1):
                    self.assertEqual(sum(r["model"] == model and r["reuse"] == enabled and
                                         r["position"] == position for r in records), 5)
        for a, b in zip(records[::2], records[1::2]):
            self.assertEqual((a["model"], a["dataset_index"]), (b["model"], b["dataset_index"]))
            self.assertNotEqual(a["reuse"], b["reuse"])


if __name__ == "__main__":
    unittest.main()
