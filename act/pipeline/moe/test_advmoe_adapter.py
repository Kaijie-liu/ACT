import unittest

import torch

from act.pipeline.moe.advmoe_adapter import (
    CrownCompatibleAdvMoePath,
    CrownCompatibleAdvMoeRouter,
    adapter_equivalence,
    construct_official_init,
    path_adapter_equivalence,
    specialize_advmoe_path,
    state_dict_sha256,
)


class AdvMoeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.router, cls.moe_type = construct_official_init(seed=1234)
        cls.model.eval()
        cls.router.eval()

    def test_official_init_is_reproducible(self):
        _model, same, _moe = construct_official_init(seed=1234)
        _model, different, _moe = construct_official_init(seed=1235)
        self.assertEqual(state_dict_sha256(self.router), state_dict_sha256(same))
        self.assertNotEqual(
            state_dict_sha256(self.router), state_dict_sha256(different)
        )

    def test_crown_lowering_is_bit_exact_on_registered_domain(self):
        torch.manual_seed(81)
        inputs = torch.cat(
            [
                torch.rand(6, 3, 32, 32),
                torch.zeros(1, 3, 32, 32),
                torch.ones(1, 3, 32, 32),
            ]
        )
        result = adapter_equivalence(self.router, inputs)
        self.assertTrue(result["outputs_equal"])
        self.assertTrue(result["routes_equal"])
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_specialized_deep_paths_match_dynamic_forward(self):
        route_models = {
            route: specialize_advmoe_path(self.model, route, self.moe_type)[0].eval()
            for route in (0, 1)
        }
        torch.manual_seed(91)
        inputs = torch.rand(8, 3, 32, 32)
        with torch.no_grad():
            dynamic = self.model(inputs)
            routes = self.router(inputs).argmax(dim=1)
            for slot in range(len(inputs)):
                specialized = route_models[int(routes[slot])](inputs[slot : slot + 1])
                self.assertTrue(
                    torch.allclose(
                        dynamic[slot : slot + 1], specialized, atol=1e-6, rtol=1e-6
                    )
                )

    def test_lowering_rejects_unregistered_input_shape(self):
        lowered = CrownCompatibleAdvMoeRouter(self.router)
        with self.assertRaises(ValueError):
            lowered.validate_input_shape(torch.zeros(1, 3, 31, 31))

    def test_static_path_lowering_is_within_frozen_tolerance(self):
        specialized, _count = specialize_advmoe_path(
            self.model, 0, self.moe_type
        )
        torch.manual_seed(92)
        inputs = torch.rand(3, 3, 32, 32)
        result = path_adapter_equivalence(specialized, inputs)
        self.assertTrue(result["outputs_close"])
        self.assertTrue(result["predictions_equal"])
        self.assertLessEqual(result["max_abs_error"], 1e-7)

    def test_static_path_lowering_requires_specialization(self):
        with self.assertRaises(ValueError):
            CrownCompatibleAdvMoePath(self.model)


if __name__ == "__main__":
    unittest.main()
