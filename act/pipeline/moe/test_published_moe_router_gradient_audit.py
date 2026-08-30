import copy
import unittest

from act.pipeline.moe.audit_published_moe_router_gradient_audit import validate
from act.pipeline.moe.published_moe_router_gradient_audit import collect


class PublishedMoeRouterGradientAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = collect()

    def test_pinned_sources_form_complete_gradient_chains(self):
        self.assertEqual(validate(self.result), [])
        pipelines = self.result["pipelines"]
        self.assertEqual(
            pipelines["rt_er"]["router_update_class"],
            "RELEASED_TRAINING_PATH_DOES_NOT_UPDATE_ROUTER",
        )
        self.assertEqual(
            pipelines["robust_moe_cnn"]["router_update_class"],
            "TRAINED_BY_EXPLICIT_ROUTER_OBJECTIVE",
        )
        self.assertEqual(
            pipelines["vmoe"]["router_update_class"],
            "TRAINED_END_TO_END_BY_COMBINE_WEIGHTS_AND_AUXILIARY_LOSSES",
        )

    def test_mutated_classification_is_rejected(self):
        mutated = copy.deepcopy(self.result)
        mutated["pipelines"]["robust_moe_cnn"]["router_update_class"] = (
            "RELEASED_TRAINING_PATH_DOES_NOT_UPDATE_ROUTER"
        )
        self.assertIn(
            "robust_moe_cnn: router update classification changed",
            validate(mutated),
        )

    def test_mutated_anchor_line_is_rejected(self):
        mutated = copy.deepcopy(self.result)
        mutated["pipelines"]["vmoe"]["anchors"][0]["line"] += 1
        self.assertTrue(
            any("anchor line changed" in issue for issue in validate(mutated))
        )

    def test_unlicensed_source_is_not_embedded(self):
        record = self.result["pipelines"]["robust_moe_cnn"]
        self.assertEqual(record["license"], "NOT_FOUND")
        self.assertNotIn("source", record)
        for anchor in record["anchors"]:
            self.assertNotIn("matched_line", anchor)


if __name__ == "__main__":
    unittest.main()
