import copy
import unittest

from act.pipeline.moe.advmoe_architecture_audit import collect
from act.pipeline.moe.audit_advmoe_architecture_audit import validate


class AdvMoeArchitectureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = collect()

    def test_pinned_architecture_and_training_schedule(self):
        issues, replay = validate(self.result)
        self.assertEqual(issues, [])
        self.assertEqual(replay["routed_moe_convolutions"], 16)
        self.assertEqual(replay["unique_router_object_ids"], 1)
        classification = self.result["classification"]
        self.assertFalse(classification["router_is_hidden_state_router"])
        self.assertTrue(classification["deep_route_conditioned_pathway"])
        self.assertFalse(classification["prefix_hz_before_router_applicable"])

    def test_hidden_state_misclassification_is_rejected(self):
        mutated = copy.deepcopy(self.result)
        mutated["classification"]["router_is_hidden_state_router"] = True
        issues, _ = validate(mutated, replay=False)
        self.assertIn("architecture classification changed", issues)

    def test_main_optimizer_router_overlap_is_rejected(self):
        mutated = copy.deepcopy(self.result)
        mutated["dynamic_confirmation"]["training_schedule"][
            "main_optimizer_router_parameter_overlap"
        ] = 1
        issues, _ = validate(mutated, replay=False)
        self.assertIn("main optimizer unexpectedly includes router parameters", issues)

    def test_unlicensed_source_is_not_embedded(self):
        self.assertEqual(self.result["repository"]["license"], "NOT_FOUND")
        self.assertFalse(self.result["artifact_policy"]["source_copied_into_act"])
        for anchor in self.result["anchors"]:
            self.assertNotIn("matched_line", anchor)
            self.assertNotIn("source", anchor)

    def test_literal_tie_semantics_are_not_silently_changed(self):
        architecture = self.result["dynamic_confirmation"]["architecture"]
        self.assertEqual(architecture["literal_equal_score_selected_index"], 0)
        self.assertEqual(
            self.result["classification"]["sound_overapproximation_tie_policy"],
            "ALL_TIED_ROUTES_CONSERVATIVE",
        )


if __name__ == "__main__":
    unittest.main()
