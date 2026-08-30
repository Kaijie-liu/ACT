from __future__ import annotations

import unittest

import numpy as np

from act.back_end.hybridz_tf.test_operator_micro_rlt import _build
from act.back_end.solver.solver_hz import (
    hz_objbound_decide,
    hz_objbound_safe_capability_receipt,
)


def _solve(build, *, threshold: float):
    C = np.asarray([[1.0]], dtype=np.float64)
    t = np.asarray([threshold], dtype=np.float64)
    groups = ((0,),)
    verdict, witness = hz_objbound_decide(
        build.hz,
        C,
        t,
        is_unsafe_linear=False,
        time_limit=2.0,
        base_witness_precheck=False,
        lp_prefilter_fraction=1.0,
        lp_prefilter_max_seconds=1.5,
        gpu_dual_steps=0,
        gpu_dual_time_limit=0.0,
        gpu_dual_row_topk=0,
        safe_row_groups=groups,
        expected_safe_group_count=1,
        safe_group_mixture_grid_bits=0,
    )
    return C, t, groups, verdict, witness


def _capability(build, C, t, groups, *, binary_lp: bool):
    return hz_objbound_safe_capability_receipt(
        build.hz,
        C,
        t,
        is_unsafe_linear=False,
        tol=1e-9,
        require_base_feasible=True,
        base_witness_precheck=False,
        safe_row_groups=groups,
        expected_safe_group_count=1,
        require_binary_relaxation_lp=binary_lp,
    )


class ObjboundSafeCapabilityTests(unittest.TestCase):
    def test_binary_relaxed_lp_safe_issues_live_bound_capability(
        self,
    ) -> None:
        build = _build(64)
        C, t, groups, verdict, witness = _solve(
            build, threshold=0.1
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        receipt = _capability(
            build, C, t, groups, binary_lp=True
        )
        self.assertIsNotNone(receipt)
        self.assertEqual(
            receipt["proof_stage"], "persistent_lp_lagrangian"
        )
        self.assertEqual(
            receipt["base_discharge"], "FEASIBLE_CHECKED"
        )
        self.assertGreater(receipt["remaining_seconds_at_issue"], 0.0)

    def test_live_hz_objective_threshold_group_and_receipt_tamper_reject(
        self,
    ) -> None:
        cases = ("hz", "C", "threshold", "groups", "receipt")
        for case in cases:
            with self.subTest(case=case):
                build = _build(64)
                C, t, groups, verdict, _ = _solve(
                    build, threshold=0.1
                )
                self.assertEqual(verdict, "SAFE")
                check_C = C
                check_t = t
                check_groups = groups
                if case == "hz":
                    build.hz.ub[-1] = np.nextafter(
                        build.hz.ub[-1], np.inf
                    )
                elif case == "C":
                    check_C = C.copy()
                    check_C[0, 0] = np.nextafter(
                        check_C[0, 0], np.inf
                    )
                elif case == "threshold":
                    check_t = t.copy()
                    check_t[0] = np.nextafter(
                        check_t[0], np.inf
                    )
                elif case == "groups":
                    check_groups = ((0, 0),)
                else:
                    build.hz._solver_objbound_safe_receipt = {
                        **build.hz._solver_objbound_safe_receipt,
                        "base_discharge": (
                            "SOUND_PHASE_COVER_MEMBER_V2"
                        ),
                    }
                self.assertIsNone(
                    _capability(
                        build,
                        check_C,
                        check_t,
                        check_groups,
                        binary_lp=False,
                    )
                )

    def test_new_unknown_call_clears_stale_safe_capability(self) -> None:
        build = _build(64)
        C, t, groups, verdict, _ = _solve(
            build, threshold=0.1
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNotNone(
            _capability(build, C, t, groups, binary_lp=False)
        )
        verdict, witness = hz_objbound_decide(
            build.hz,
            C,
            t,
            is_unsafe_linear=False,
            time_limit=0.0,
            base_witness_precheck=False,
            safe_row_groups=groups,
            expected_safe_group_count=1,
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        self.assertIsNone(
            _capability(build, C, t, groups, binary_lp=False)
        )

    def test_cube_safe_is_sound_but_not_binary_lp_attributed(
        self,
    ) -> None:
        build = _build(64)
        C, t, groups, verdict, _ = _solve(
            build, threshold=100.0
        )
        self.assertEqual(verdict, "SAFE")
        general = _capability(
            build, C, t, groups, binary_lp=False
        )
        self.assertIsNotNone(general)
        self.assertEqual(
            general["proof_stage"], "cube_outward_support"
        )
        self.assertIsNone(
            _capability(build, C, t, groups, binary_lp=True)
        )


if __name__ == "__main__":
    unittest.main()
