import copy
import unittest

from act.pipeline.moe.advmoe_dependency_audit import collect
from act.pipeline.moe.audit_advmoe_dependency_audit import validate


class AdvMoeDependencyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = collect()

    def test_dependency_spec_does_not_define_exact_environment(self):
        spec = self.result["official_dependency_specification"]
        self.assertFalse(spec["torch_version_pinned"])
        self.assertFalse(spec["cuda_version_pinned"])
        self.assertFalse(spec["python_version_pinned"])
        self.assertEqual(spec["versioned_entries"], ["scipy==1.6.0"])
        self.assertFalse(spec["readme_requested_file_exists"])

    def test_blackwell_model_probe_passes_without_training_claim(self):
        probe = self.result["blackwell_model_only_probe"]
        self.assertEqual(probe["status"], "PASS")
        self.assertEqual(probe["capability"], [12, 0])
        self.assertEqual(probe["shape"], [2, 2])
        classification = self.result["classification"]
        self.assertTrue(classification["model_only_blackwell_compatible_in_act_py312"])
        self.assertFalse(classification["training_entrypoint_runnable_in_act_py312"])

    def test_no_environment_mutation_is_claimed(self):
        classification = self.result["classification"]
        self.assertFalse(classification["installation_performed"])
        self.assertFalse(classification["environment_created"])

    def test_independent_validator_accepts_pinned_result(self):
        self.assertEqual(validate(self.result), [])


if __name__ == "__main__":
    unittest.main()
