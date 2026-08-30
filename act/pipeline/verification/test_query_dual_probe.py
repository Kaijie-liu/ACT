#!/usr/bin/env python3
"""Toy-only fail-closed tests for the query-dual Gate-1 probe."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest import mock

import torch

from act.back_end.hybridz_tf.query_dual_pipeline import (
    validate_verified_query_dual_feedback,
)
from act.back_end.hybridz_tf.test_query_dual_operator_integration import (
    _build_live_feedback,
    _residual_two_relu_toy,
)
from act.pipeline.verification import query_dual_probe as probe
from act.util.device_manager import initialize_device


class _FakeSynthesizedModel:
    def __init__(self) -> None:
        self.moves = []

    def to(self, *, device, dtype):
        self.moves.append((device, dtype))
        return self

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())


class QueryDualProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_device = torch.get_default_device()
        cls.original_dtype = torch.get_default_dtype()
        initialize_device("cpu", "float64")
        cls.toy = _residual_two_relu_toy()
        cls.feedback = _build_live_feedback(cls.toy)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_default_device(cls.original_device)
        torch.set_default_dtype(cls.original_dtype)

    @staticmethod
    def _cuda_environment():
        return {
            "status": "tracking",
            "device": "cuda",
            "dtype": "float64",
            "device_index": 0,
            "device_name": "fake-test-cuda",
            "cuda_build": "test",
        }

    @staticmethod
    def _cuda_memory():
        return {
            "status": "measured",
            "device_index": 0,
            "device_name": "fake-test-cuda",
            "peak_allocated_bytes": 101,
            "peak_reserved_bytes": 202,
            "total_bytes": 303,
        }

    def _fixture(self, directory: str):
        root = Path(directory)
        onnx = root / "toy.onnx"
        vnnlib = root / "toy.vnnlib"
        output = root / "receipt.json"
        onnx.write_bytes(b"toy-onnx-input")
        vnnlib.write_bytes(b"toy-vnnlib-input")
        config = probe.QueryDualProbeConfig(
            onnx_path=onnx,
            vnnlib_path=vnnlib,
            target_relu_ids=(7,),
            steps=1,
            time_limit=12.0,
            block_size=1,
            device="cuda",
            output_path=output,
        )
        return config, output

    def _run(
        self,
        config,
        *,
        validator=validate_verified_query_dual_feedback,
        setup_delay=0.0,
        builder_delay=0.0,
        mutate_input=False,
        cuda_memory_reader=None,
        feedback=None,
    ):
        calls = {}
        model = _FakeSynthesizedModel()

        def load(onnx, vnnlib, *, category):
            if setup_delay:
                time.sleep(setup_delay)
            if mutate_input:
                Path(vnnlib).write_bytes(b"mutated-vnnlib-input")
            calls["load"] = (onnx, vnnlib, category)
            return object()

        def synthesize(specs):
            calls["synthesis_count"] = len(specs)
            return {("only",): model}

        def build(net, C, thresholds, **kwargs):
            calls["builder_called_at"] = time.monotonic()
            if builder_delay:
                time.sleep(builder_delay)
            calls["build"] = (net, C.copy(), thresholds.copy(), dict(kwargs))
            return self.feedback if feedback is None else feedback

        def validate(bundle, **kwargs):
            calls["validator_kwargs"] = dict(kwargs)
            return validator(bundle, **kwargs)

        result = probe.run_query_dual_probe(
            config,
            spec_loader=load,
            synthesizer=synthesize,
            converter_factory=lambda moved: SimpleNamespace(
                run=lambda: self.toy.net
            ),
            feedback_builder=build,
            feedback_validator=validate,
            cuda_initializer=lambda device: self._cuda_environment(),
            cuda_memory_reader=(
                self._cuda_memory
                if cuda_memory_reader is None
                else cuda_memory_reader
            ),
        )
        return result, calls, model

    def test_success_is_live_validated_diagnostic_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            config, output = self._fixture(directory)
            result, calls, model = self._run(config)

            self.assertEqual(result["status"], "verified")
            self.assertEqual(
                result["config"]["pipeline_protocol"],
                probe._PIPELINE_V2,
            )
            self.assertIs(result["config"]["stage_quotas"], None)
            self.assertIs(result["promotion_gate"]["applicable"], False)
            self.assertIs(result["promotion_gate"]["pass"], True)
            self.assertIs(result["proof_authority"], False)
            self.assertIs(result["production_dependencies"], False)
            self.assertIs(result["test_only"], True)
            self.assertIs(result["diagnostic_only"], True)
            self.assertIs(result["produces_verdict"], False)
            self.assertIs(result["operator_hz_called"], False)
            self.assertIs(result["hz_solver_called"], False)
            self.assertEqual(
                result["input_sha256_before"],
                result["input_sha256_after"],
            )
            self.assertEqual(
                result["source_sha256_before"],
                result["source_sha256_after"],
            )
            self.assertIs(result["source_integrity_stable"], True)
            self.assertEqual(result["cuda_memory"]["peak_allocated_bytes"], 101)
            self.assertEqual(result["cuda_memory"]["peak_reserved_bytes"], 202)
            self.assertEqual(result["cuda_memory"]["total_bytes"], 303)
            self.assertEqual(
                result["cpu_parallelism"]["transaction_workers"], 1
            )
            self.assertGreaterEqual(
                result["cpu_parallelism"]["torch_intraop_threads"], 1
            )
            self.assertIs(result["cpu_rss"]["pass"], True)
            self.assertGreaterEqual(
                result["cpu_rss"]["transaction_increment_bytes"], 0
            )
            self.assertEqual(result["stages"][0]["target_relu_id"], 7)
            self.assertIn("strict_improvements", result["stages"][0])
            self.assertIn("candidate_seconds", result["stages"][0])
            self.assertIn("independent_replay_seconds", result["stages"][0])
            self.assertEqual(
                result["pipeline_receipt"]["receipt_sha256"],
                self.feedback.receipt["receipt_sha256"],
            )
            self.assertEqual(
                result["property"]["upper_sha256"],
                self.feedback.receipt["property_upper_sha256"],
            )
            self.assertEqual(
                result["property"]["upper_hex"],
                [float(value).hex() for value in self.feedback.property_upper],
            )
            body = dict(result)
            receipt_hash = body.pop("receipt_sha256")
            self.assertEqual(receipt_hash, probe._canonical_sha256(body))
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(calls["synthesis_count"], 1)
            self.assertEqual(calls["load"][2], "query_dual_gate1_probe")
            self.assertIs(calls["build"][0], self.toy.net)
            self.assertEqual(calls["build"][3]["candidate_device"], "cuda")
            self.assertEqual(
                calls["build"][3]["replay_max_workspace_bytes"],
                512 * 1024 * 1024,
            )
            self.assertGreater(calls["build"][3]["deadline"], time.monotonic())
            self.assertIs(
                calls["validator_kwargs"]["require_live_provenance"],
                True,
            )
            self.assertAlmostEqual(
                result["transaction_seconds"],
                result["phase_seconds"]["query_dual_transaction"],
            )
            self.assertAlmostEqual(
                result["validation_seconds"],
                result["phase_seconds"]["live_validation"],
            )
            self.assertEqual(result["transaction_stop_loss_seconds"], 10.5)
            self.assertIs(result["transaction_stop_loss_pass"], True)
            self.assertEqual(result["transaction_hard_limit_seconds"], 12.0)
            self.assertIs(result["transaction_hard_limit_pass"], True)
            self.assertGreaterEqual(
                result["total_seconds"],
                result["transaction_seconds"],
            )
            self.assertEqual(model.moves[0][0], torch.device("cuda"))
            self.assertEqual(model.moves[0][1], torch.float64)

    def test_explicit_v3_route_is_bound_and_wrong_bundle_fails_closed(self):
        with TemporaryDirectory() as directory:
            config, output = self._fixture(directory)
            config = replace(
                config,
                stage_quotas=(1,),
                selector_time_limit=0.75,
                selector_max_adjoint_cells=1234,
                selector_pool_per_rival=7,
            )
            result, calls, _ = self._run(config)

            self.assertEqual(result["status"], "error")
            self.assertEqual(
                result["config"]["pipeline_protocol"],
                probe._PIPELINE_V3,
            )
            self.assertEqual(result["config"]["stage_quotas"], [1])
            self.assertEqual(
                result["failed_stage"], "pipeline_protocol_binding"
            )
            self.assertIn(
                "PIPELINE_PROTOCOL_BINDING", result["error"]["message"]
            )
            kwargs = calls["build"][3]
            self.assertEqual(kwargs["stage_quotas"], (1,))
            self.assertEqual(kwargs["selector_time_limit"], 0.75)
            self.assertEqual(kwargs["selector_max_adjoint_cells"], 1234)
            self.assertEqual(kwargs["selector_pool_per_rival"], 7)
            self.assertNotIn("validator_kwargs", calls)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_live_v3_route_reports_promotion_stop_loss_metrics(self):
        from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
            build_verified_query_dual_feedback_v3,
        )
        from act.back_end.hybridz_tf.test_query_dual_pipeline import (
            _IntervalCandidateSolver,
        )
        from act.back_end.hybridz_tf.test_query_dual_pipeline_v3 import (
            _toy_selector,
        )

        bundle = build_verified_query_dual_feedback_v3(
            self.toy.net,
            self.toy.C,
            self.toy.thresholds,
            target_relu_ids=(7,),
            stage_quotas=(1,),
            steps=1,
            block_size=1,
            replay_chunk_size=1,
            candidate_device="cpu",
            timeout_s=10.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        with TemporaryDirectory() as directory:
            config, output = self._fixture(directory)
            config = replace(config, stage_quotas=(1,))
            result, calls, _ = self._run(config, feedback=bundle)

            self.assertEqual(result["status"], "verified_not_promoted")
            self.assertEqual(
                result["cuda_memory_stop_loss_bytes"],
                int(2.5 * 1024**3),
            )
            gate = result["promotion_gate"]
            self.assertIs(gate["applicable"], True)
            self.assertIs(gate["pass"], False)
            self.assertEqual(gate["target_strict_improvements"], 1)
            self.assertEqual(
                gate["nonempty_target_stages"][0][
                    "selected_target_count"
                ],
                1,
            )
            self.assertIs(
                gate["nonempty_target_stages"][0]["pass"], True
            )
            self.assertGreaterEqual(
                result["property"]["strict_improvements_from_root"], 0
            )
            self.assertEqual(
                result["property"]["root_interval_upper_sha256"],
                probe._array_digest(
                    [
                        float.fromhex(value)
                        for value in result["property"][
                            "root_interval_upper_hex"
                        ]
                    ]
                ),
            )
            self.assertIn("validator_kwargs", calls)
            self.assertEqual(json.loads(output.read_text()), result)

        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            config = replace(config, stage_quotas=(1,))
            over_limit = int(2.5 * 1024**3) + 1
            result, _, _ = self._run(
                config,
                feedback=bundle,
                cuda_memory_reader=lambda: {
                    "status": "measured",
                    "device_index": 0,
                    "device_name": "fake-test-cuda",
                    "peak_allocated_bytes": over_limit,
                    "peak_reserved_bytes": over_limit,
                    "total_bytes": over_limit + 1,
                },
            )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failed_stage"], "cuda_memory")
            self.assertIn(
                "CUDA_MEMORY_STOP_LOSS", result["error"]["message"]
            )

    def test_default_builder_resolution_selects_v3_only_when_requested(self):
        from act.back_end.hybridz_tf import query_dual_pipeline_v3
        from act.back_end.hybridz_tf.test_query_dual_pipeline import (
            _IntervalCandidateSolver,
        )
        from act.back_end.hybridz_tf.test_query_dual_pipeline_v3 import (
            _toy_selector,
        )

        bundle = query_dual_pipeline_v3.build_verified_query_dual_feedback_v3(
            self.toy.net,
            self.toy.C,
            self.toy.thresholds,
            target_relu_ids=(7,),
            stage_quotas=(1,),
            steps=1,
            block_size=1,
            replay_chunk_size=1,
            candidate_device="cpu",
            timeout_s=10.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        calls = {}

        def selected_builder(net, rows, thresholds, **kwargs):
            calls["net"] = net
            calls["kwargs"] = dict(kwargs)
            return bundle

        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            config = replace(config, stage_quotas=(1,))
            model = _FakeSynthesizedModel()
            with mock.patch.object(
                query_dual_pipeline_v3,
                "build_verified_query_dual_feedback_v3",
                side_effect=selected_builder,
            ) as patched:
                result = probe.run_query_dual_probe(
                    config,
                    spec_loader=lambda *args, **kwargs: object(),
                    synthesizer=lambda specs: {("only",): model},
                    converter_factory=lambda moved: SimpleNamespace(
                        run=lambda: self.toy.net
                    ),
                    cuda_initializer=lambda device: self._cuda_environment(),
                    cuda_memory_reader=self._cuda_memory,
                )

            self.assertEqual(result["status"], "verified_not_promoted")
            patched.assert_called_once()
            self.assertIs(calls["net"], self.toy.net)
            self.assertEqual(calls["kwargs"]["stage_quotas"], (1,))
            self.assertEqual(
                result["pipeline_receipt"]["schema"],
                "act.verified_query_dual_feedback.v3",
            )

    def test_v3_quota_validation_and_cli_are_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            normalized = probe._validate_config(
                replace(
                    config,
                    target_relu_ids=(7, 8),
                    stage_quotas=(0, 1),
                    selector_pool_per_rival=1,
                )
            )
            self.assertEqual(normalized.stage_quotas, (0, 1))
            self.assertEqual(
                probe._pipeline_protocol(normalized), probe._PIPELINE_V3
            )
            for bad in ((1, 2), (65,), (True,), (0,)):
                with self.assertRaisesRegex(
                    probe.QueryDualProbeError, "INVALID_V3_QUOTAS"
                ):
                    probe._validate_config(
                        replace(config, stage_quotas=bad)
                    )
            with self.assertRaisesRegex(
                probe.QueryDualProbeError, "selector pool"
            ):
                probe._validate_config(
                    replace(
                        config,
                        stage_quotas=(2,),
                        selector_pool_per_rival=1,
                    )
                )
            with self.assertRaisesRegex(
                probe.QueryDualProbeError, "12s hard limit"
            ):
                probe._validate_config(
                    replace(config, time_limit=12.000001)
                )

        args = probe._build_parser().parse_args(
            [
                "--onnx",
                "toy.onnx",
                "--vnnlib",
                "toy.vnnlib",
                "--targets",
                "10,14,22,40",
                "--v3-quotas",
                "16,8,24,16",
                "--selector-time",
                "0.8",
                "--output",
                "receipt.json",
            ]
        )
        parsed = probe._config_from_args(args)
        self.assertEqual(parsed.stage_quotas, (16, 8, 24, 16))
        self.assertEqual(parsed.selector_time_limit, 0.8)

    def test_overlong_hard_limit_is_an_atomic_error_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            config, output = self._fixture(directory)
            config = replace(
                config, stage_quotas=(1,), time_limit=12.000001
            )
            result, calls, _ = self._run(config)

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failed_stage"], "config")
            self.assertIn("12s hard limit", result["error"]["message"])
            self.assertIs(result["proof_authority"], False)
            self.assertFalse(calls)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_failed_live_validation_is_atomic_non_authority_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            config, output = self._fixture(directory)
            result, _, _ = self._run(
                config,
                validator=lambda *args, **kwargs: False,
            )

            self.assertEqual(result["status"], "error")
            self.assertIs(result["proof_authority"], False)
            self.assertEqual(result["failed_stage"], "live_validation")
            self.assertIn("LIVE_VALIDATION", result["error"]["message"])
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertFalse(
                list(output.parent.glob(f".{output.name}.*.tmp"))
            )

    def test_setup_and_slow_validator_do_not_consume_transaction_budget(self):
        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            config = replace(config, time_limit=0.05)

            def slow_validator(*args, **kwargs):
                time.sleep(0.07)
                return True

            result, calls, _ = self._run(
                config,
                validator=slow_validator,
                setup_delay=0.07,
            )

            self.assertEqual(result["status"], "verified")
            self.assertGreaterEqual(result["setup_seconds"], 0.07)
            self.assertLess(result["transaction_seconds"], 0.05)
            self.assertGreaterEqual(result["validation_seconds"], 0.07)
            self.assertGreater(result["total_seconds"], 0.14)
            remaining_at_builder = (
                calls["build"][3]["deadline"]
                - calls["builder_called_at"]
            )
            self.assertGreater(remaining_at_builder, 0.04)
            self.assertLessEqual(remaining_at_builder, 0.05)

    def test_source_toctou_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            config, output = self._fixture(directory)
            before = {"proof.py": "a" * 64}
            after = {"proof.py": "b" * 64}
            with mock.patch.object(
                probe,
                "_source_hashes",
                side_effect=[before, after],
            ):
                result, _, _ = self._run(
                    config,
                    validator=lambda *args, **kwargs: True,
                )

            self.assertEqual(result["status"], "error")
            self.assertIs(result["proof_authority"], False)
            self.assertEqual(result["failed_stage"], "source_hash_after")
            self.assertIs(result["source_integrity_stable"], False)
            self.assertEqual(result["source_sha256_before"], before)
            self.assertEqual(result["source_sha256_after"], after)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_transaction_stop_loss_uses_builder_time(self) -> None:
        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            with mock.patch.object(
                probe,
                "_TRANSACTION_STOP_LOSS_SECONDS",
                0.0,
            ):
                result, _, _ = self._run(
                    config,
                    validator=lambda *args, **kwargs: True,
                )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failed_stage"], "transaction_stop_loss")
            self.assertGreater(result["transaction_seconds"], 0.0)
            self.assertIs(result["transaction_stop_loss_pass"], False)

    def test_v3_promotion_margin_is_not_the_hard_deadline(self) -> None:
        from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
            build_verified_query_dual_feedback_v3,
        )
        from act.back_end.hybridz_tf.test_query_dual_pipeline import (
            _IntervalCandidateSolver,
        )
        from act.back_end.hybridz_tf.test_query_dual_pipeline_v3 import (
            _toy_selector,
        )

        bundle = build_verified_query_dual_feedback_v3(
            self.toy.net,
            self.toy.C,
            self.toy.thresholds,
            target_relu_ids=(7,),
            stage_quotas=(1,),
            steps=1,
            block_size=1,
            replay_chunk_size=1,
            candidate_device="cpu",
            timeout_s=10.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            config = replace(config, stage_quotas=(1,))
            with mock.patch.object(
                probe, "_TRANSACTION_STOP_LOSS_SECONDS", 0.0
            ):
                result, _, _ = self._run(config, feedback=bundle)

            self.assertEqual(result["status"], "verified_not_promoted")
            self.assertNotIn("failed_stage", result)
            self.assertIs(result["transaction_stop_loss_pass"], False)
            self.assertIs(result["transaction_hard_limit_pass"], True)
            self.assertIs(result["promotion_gate"]["pass"], False)
            self.assertIs(result["promotion_gate"]["hard_limit_pass"], True)

    def test_cuda_memory_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            mismatch = self._cuda_memory()
            mismatch["device_index"] = 1
            result, _, _ = self._run(
                config,
                cuda_memory_reader=lambda: mismatch,
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failed_stage"], "cuda_memory")
            self.assertIs(result["proof_authority"], False)
            self.assertIn("device index differs", result["error"]["message"])

    def test_input_toctou_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            result, _, _ = self._run(config, mutate_input=True)

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failed_stage"], "input_hash_after")
            self.assertIs(result["input_integrity_stable"], False)
            self.assertNotEqual(
                result["input_sha256_before"],
                result["input_sha256_after"],
            )

    def test_builder_return_after_deadline_still_fails(self) -> None:
        with TemporaryDirectory() as directory:
            config, _ = self._fixture(directory)
            config = replace(config, time_limit=0.01)
            result, calls, _ = self._run(config, builder_delay=0.03)

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failed_stage"], "query_dual_transaction")
            self.assertIn("DEADLINE_EXPIRED", result["error"]["message"])
            self.assertGreaterEqual(result["transaction_seconds"], 0.03)
            self.assertNotIn("validator_kwargs", calls)

    def test_dependency_injection_policy_and_forbidden_import_surface(self):
        hooks = [probe._UNSET] * 7
        self.assertIs(probe._production_dependency_policy(*hooks), True)
        for index in range(len(hooks)):
            injected = list(hooks)
            injected[index] = object()
            self.assertIs(
                probe._production_dependency_policy(*injected),
                False,
            )

        source = Path(probe.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "build_operator_hz",
            "verify_once",
            "solver_hz",
        ):
            self.assertNotIn(forbidden, source)

    def test_atomic_no_clobber_overwrite_and_input_alias(self) -> None:
        with TemporaryDirectory() as directory:
            config, output = self._fixture(directory)
            output.write_text("sentinel")
            with self.assertRaisesRegex(
                probe.QueryDualProbeError, "OUTPUT_EXISTS"
            ):
                probe._atomic_json(output, {"new": True}, overwrite=False)
            self.assertEqual(output.read_text(), "sentinel")

            probe._atomic_json(output, {"new": True}, overwrite=True)
            self.assertEqual(json.loads(output.read_text()), {"new": True})

            input_bytes = config.onnx_path.read_bytes()
            alias_config = replace(
                config,
                output_path=config.onnx_path,
                overwrite=True,
            )
            with self.assertRaisesRegex(
                probe.QueryDualProbeError, "OUTPUT_ALIAS"
            ):
                probe._validate_config(alias_config)
            with self.assertRaisesRegex(
                probe.QueryDualProbeError, "OUTPUT_ALIAS"
            ):
                probe._atomic_json(
                    config.onnx_path,
                    {"must_not": "replace input"},
                    overwrite=True,
                    forbidden_paths=(config.onnx_path, config.vnnlib_path),
                )
            self.assertEqual(config.onnx_path.read_bytes(), input_bytes)

    def test_cli_returns_nonzero_for_error_receipt(self) -> None:
        error = {"status": "error", "proof_authority": False}
        with mock.patch.object(probe, "run_query_dual_probe", return_value=error):
            code = probe.main(
                [
                    "--onnx",
                    "toy.onnx",
                    "--vnnlib",
                    "toy.vnnlib",
                    "--targets",
                    "7",
                    "--output",
                    "receipt.json",
                ]
            )
        self.assertEqual(code, 1)
        defaults = probe._build_parser().parse_args(
            [
                "--onnx",
                "toy.onnx",
                "--vnnlib",
                "toy.vnnlib",
                "--targets",
                "7",
                "--output",
                "receipt.json",
            ]
        )
        self.assertEqual(defaults.time_limit, 12.0)
        self.assertIs(defaults.overwrite, False)


if __name__ == "__main__":
    unittest.main()
