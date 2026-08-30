#!/usr/bin/env python3
"""Real ``verify_once`` gates for the phase-clique private solver handoff.

The component tests exercise the K4 candidate, materializer, and handoff in
isolation.  These tests deliberately enter through the public verifier with a
real ACT ``Net``, raw TOP1 VNN-LIB source, and live residual selector.  The
second gate corrupts the public materialized HZ only after it has been
consumed, then observes the object passed to the real objective-bound solver.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Layer, Net
from act.back_end import verifier as verifier_module
from act.back_end.hybridz_tf import (
    operator_phase_clique_pipeline as pipeline_module,
)
from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
    validate_consumed_operator_phase_clique_solver_build,
)
from act.back_end.solver import solver_hz as solver_hz_module
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once
from act.front_end.specs import OutKind, OutputSpec
from act.util import device_manager as device_manager_module
from act.util.stats import VerifyStatus


_DTYPE = torch.float64
_DEVICE = torch.device("cpu")
_K4_WEIGHT = torch.tensor(
    [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ],
    dtype=_DTYPE,
    device=_DEVICE,
)
_K4_BIAS = torch.full(
    (4,), -1.5, dtype=_DTYPE, device=_DEVICE
)
_OUTPUT_WEIGHT = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    ],
    dtype=_DTYPE,
    device=_DEVICE,
)


def _dense_params(
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> dict:
    return {
        "weight": weight,
        "weight_pos": weight.clamp(min=0),
        "weight_neg": weight.clamp(max=0),
        "bias": bias,
        "in_features": int(weight.shape[1]),
        "out_features": int(weight.shape[0]),
        "input_shape": (int(weight.shape[1]),),
    }


def _k4_top1_net(*, true_logit: float = 0.75) -> Net:
    """Four mutually exclusive exact ReLUs and three TOP1 logits."""

    input_vars = [0, 1]
    preactivation_vars = [2, 3, 4, 5]
    relu_vars = [6, 7, 8, 9]
    output_vars = [10, 11, 12]
    output_bias = torch.tensor(
        [true_logit, 0.0, 0.0], dtype=_DTYPE, device=_DEVICE
    )
    assertion = OutputSpec(
        kind=OutKind.TOP1_ROBUST,
        y_true=torch.tensor([0], dtype=torch.int64, device=_DEVICE),
    ).encode_linear(
        B=1,
        n_out=3,
        device=_DEVICE,
        dtype=_DTYPE,
    )
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 2), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=input_vars,
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor(
                    [[-1.0, -1.0]], dtype=_DTYPE, device=_DEVICE
                ),
                "ub": torch.tensor(
                    [[1.0, 1.0]], dtype=_DTYPE, device=_DEVICE
                ),
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=2,
            kind="DENSE",
            params=_dense_params(_K4_WEIGHT, _K4_BIAS),
            in_vars=input_vars,
            out_vars=preactivation_vars,
        ),
        Layer(
            id=3,
            kind="RELU",
            params={},
            in_vars=preactivation_vars,
            out_vars=relu_vars,
        ),
        Layer(
            id=4,
            kind="DENSE",
            params=_dense_params(_OUTPUT_WEIGHT, output_bias),
            in_vars=relu_vars,
            out_vars=output_vars,
        ),
        Layer(
            id=5,
            kind="ASSERT",
            params=assertion,
            in_vars=output_vars,
            out_vars=output_vars,
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _backend_config() -> BackendConfig:
    return BackendConfig(
        solver="hybridz",
        device="cpu",
        dtype="float64",
        timeout=30.0,
        hybridz=HybridZConfig(
            timeout=20.0,
            engine="operator_hz_objbound",
            operator_exact_budget=4,
            operator_phase_clique_time_limit=10.0,
            operator_materialize_add=True,
            property_residual_budget=4,
            property_residual_time_limit=4.0,
        ),
    )


def _raw_top1_source() -> str:
    return """
(set-logic QF_LRA)
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 -1))
(assert (<= X_0 1))
(assert (>= X_1 -1))
(assert (<= X_1 1))
(assert (or (<= Y_0 Y_1) (<= Y_0 Y_2)))
""".strip() + "\n"


def _write_raw(directory: str) -> tuple[Path, str]:
    path = Path(directory) / "k4_top1.vnnlib"
    path.write_text(_raw_top1_source(), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _append_contradictory_zero_row(hz) -> None:
    """Poison a public HZ with the exact contradiction ``0 <= -1``."""

    hz.Auc = sp.vstack(
        [hz.Auc, sp.csr_matrix((1, hz.n_cont), dtype=np.float64)],
        format="csr",
    )
    hz.Aub = sp.vstack(
        [hz.Aub, sp.csr_matrix((1, hz.n_bin), dtype=np.float64)],
        format="csr",
    )
    hz.ub = np.concatenate(
        [hz.ub, np.asarray([-1.0], dtype=np.float64)]
    )


def _model_fn(*, true_logit: float):
    output_bias = torch.tensor(
        [true_logit, 0.0, 0.0], dtype=_DTYPE, device=_DEVICE
    )

    def model(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], 2).to(
            device=_DEVICE, dtype=_DTYPE
        )
        hidden = torch.relu(flat @ _K4_WEIGHT.T + _K4_BIAS)
        return hidden @ _OUTPUT_WEIGHT.T + output_bias

    return model


class OperatorPhaseCliqueVerifierIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_default_device = torch.get_default_device()
        self._old_default_dtype = torch.get_default_dtype()
        self._old_device_manager_initialized = (
            device_manager_module._INITIALIZED
        )
        # The test explicitly owns a CPU backend.  Mark the lazy manager as
        # initialized while the gate runs so its first selector query cannot
        # silently replace that choice with auto-detected CUDA.
        device_manager_module._INITIALIZED = True
        torch.set_default_device("cpu")
        torch.set_default_dtype(_DTYPE)
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")

    def tearDown(self) -> None:
        set_solver_mode(None)
        set_transfer_function_mode("interval")
        torch.set_default_device(self._old_default_device)
        torch.set_default_dtype(self._old_default_dtype)
        device_manager_module._INITIALIZED = (
            self._old_device_manager_initialized
        )

    def _verify(self):
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            return verify_once(
                _k4_top1_net(),
                backend_cfg=_backend_config(),
                raw_vnnlib_path=path,
                expected_raw_vnnlib_sha256=source_sha256,
            )[0]

    def test_real_k4_top1_verify_once_materializes_and_certifies(self) -> None:
        result = self._verify()

        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(result.metadata["hz_verdict"], "SAFE")
        receipt = result.metadata[
            "operator_phase_clique_materialization"
        ]
        self.assertEqual(
            receipt["status"], "fresh_verified_k4_clique_materialized"
        )
        self.assertTrue(receipt["materialized"])
        self.assertEqual(receipt["solver_handoff_status"], "issued")
        self.assertEqual(result.metadata["operator_source_n_ub"], 12)
        self.assertEqual(result.metadata["operator_n_ub"], 13)
        self.assertEqual(
            result.metadata["base_feasibility_status"], "FEASIBLE"
        )
        self.assertTrue(result.metadata["all_rivals_covered"])
        self.assertEqual(result.metadata["cube_survivor_rows"], 2)
        self.assertEqual(result.metadata["lp_certified_rows"], 2)
        self.assertEqual(result.metadata["lp_survivor_rows"], 0)

    def test_public_poison_after_consume_cannot_reach_solver_or_stats(
        self,
    ) -> None:
        observed = {}
        real_consume = (
            pipeline_module
            .consume_operator_phase_clique_pipeline_solver_handoff
        )
        real_decide = solver_hz_module.hz_objbound_decide

        def consume_then_poison(source, result, *, deadline):
            private = real_consume(
                source, result, deadline=deadline
            )
            public_hz = result.build.hz
            observed.update(
                {
                    "pipeline_result": result,
                    "private_build": private,
                    "public_hz": public_hz,
                    "private_n_ub_before": private.hz.n_ub,
                    "public_n_ub_before": public_hz.n_ub,
                }
            )
            public_hz.c[0] += 777.0
            _append_contradictory_zero_row(public_hz)
            observed["public_n_ub_after"] = public_hz.n_ub
            return private

        def observe_real_solver(hz, *args, **kwargs):
            observed["solver_hz"] = hz
            verdict, witness = real_decide(hz, *args, **kwargs)
            observed["private_stats"] = getattr(
                hz, "_solver_objbound_stats", None
            )
            return verdict, witness

        with (
            mock.patch.object(
                pipeline_module,
                "consume_operator_phase_clique_pipeline_solver_handoff",
                side_effect=consume_then_poison,
            ),
            mock.patch.object(
                solver_hz_module,
                "hz_objbound_decide",
                side_effect=observe_real_solver,
            ),
        ):
            result = self._verify()

        private = observed["private_build"]
        public_hz = observed["public_hz"]
        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(result.metadata["hz_verdict"], "SAFE")
        self.assertIs(observed["solver_hz"], private.hz)
        self.assertIsNot(observed["solver_hz"], public_hz)
        self.assertEqual(observed["private_n_ub_before"], 13)
        self.assertEqual(observed["public_n_ub_before"], 13)
        self.assertEqual(observed["public_n_ub_after"], 14)
        self.assertEqual(private.hz.n_ub, 13)
        self.assertEqual(result.metadata["operator_n_ub"], 13)
        self.assertIs(
            observed["private_stats"],
            getattr(private.hz, "_solver_objbound_stats"),
        )
        self.assertEqual(
            observed["private_stats"]["lp_status"], "complete"
        )
        self.assertEqual(
            observed["private_stats"]["lp_survivor_rows"], 0
        )
        self.assertEqual(
            result.metadata["base_feasibility_status"],
            observed["private_stats"]["base_feasibility_status"],
        )
        self.assertIsNone(
            getattr(public_hz, "_solver_objbound_stats", None)
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                observed["pipeline_result"], private
            )
        )

    def test_unsafe_decode_and_replay_also_use_private_hz(self) -> None:
        observed = {}
        real_consume = (
            pipeline_module
            .consume_operator_phase_clique_pipeline_solver_handoff
        )
        real_decide = solver_hz_module.hz_objbound_decide
        real_decode = verifier_module._hybridz_witness_input
        model = _model_fn(true_logit=0.25)

        def consume_then_poison(source, result, *, deadline):
            private = real_consume(
                source, result, deadline=deadline
            )
            public_hz = result.build.hz
            observed.update(
                {
                    "pipeline_result": result,
                    "private_build": private,
                    "public_hz": public_hz,
                }
            )
            public_hz.c[0] -= 999.0
            _append_contradictory_zero_row(public_hz)
            return private

        def observe_real_solver(hz, *args, **kwargs):
            observed["solver_hz"] = hz
            return real_decide(hz, *args, **kwargs)

        def observe_real_decode(hz, *args, **kwargs):
            observed["decode_hz"] = hz
            return real_decode(hz, *args, **kwargs)

        def independent_raw_top1_replay(x_batch: torch.Tensor):
            observed["replay_input"] = x_batch.detach().clone()
            output = model(x_batch)
            accepted = bool(
                torch.any(output[:, 1:] >= output[:, :1]).item()
            )
            return {
                "valid_counterexample": accepted,
                "authority": "independent_k4_raw_top1_replay",
            }

        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with (
                mock.patch.object(
                    pipeline_module,
                    "consume_operator_phase_clique_pipeline_solver_handoff",
                    side_effect=consume_then_poison,
                ),
                mock.patch.object(
                    solver_hz_module,
                    "hz_objbound_decide",
                    side_effect=observe_real_solver,
                ),
                mock.patch.object(
                    verifier_module,
                    "_hybridz_witness_input",
                    side_effect=observe_real_decode,
                ),
            ):
                result = verify_once(
                    _k4_top1_net(true_logit=0.25),
                    model_fn=model,
                    counterexample_replay_fn=(
                        independent_raw_top1_replay
                    ),
                    backend_cfg=_backend_config(),
                    raw_vnnlib_path=path,
                    expected_raw_vnnlib_sha256=source_sha256,
                )[0]

        private = observed["private_build"]
        public_hz = observed["public_hz"]
        self.assertEqual(result.status, VerifyStatus.FALSIFIED)
        self.assertEqual(result.metadata["hz_verdict"], "UNSAFE")
        self.assertEqual(
            result.metadata["hz_candidate_decode"],
            "stable_generator_ids",
        )
        self.assertEqual(
            result.metadata["hz_independent_replay"],
            "independent_replay_accepted",
        )
        self.assertTrue(result.metadata["hz_candidate_model_unsafe"])
        self.assertIs(observed["solver_hz"], private.hz)
        self.assertIs(observed["decode_hz"], private.hz)
        self.assertIsNot(observed["decode_hz"], public_hz)
        self.assertEqual(private.hz.n_ub, 13)
        self.assertEqual(public_hz.n_ub, 14)
        self.assertEqual(result.metadata["operator_n_ub"], 13)
        self.assertIsNotNone(result.counterexample)
        self.assertTrue(
            bool(
                torch.all(result.counterexample >= -1.0 - 1.0e-8).item()
                and torch.all(
                    result.counterexample <= 1.0 + 1.0e-8
                ).item()
            )
        )
        replay_output = model(observed["replay_input"])
        self.assertTrue(
            bool(
                torch.any(
                    replay_output[:, 1:] >= replay_output[:, :1]
                ).item()
            )
        )
        self.assertIsNone(
            getattr(public_hz, "_solver_objbound_stats", None)
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                observed["pipeline_result"], private
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
