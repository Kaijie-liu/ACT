"""Permanent controlled toys for the isolated V5.1b Conv material cache."""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import os
import time
import unittest
from types import MappingProxyType
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51
from act.back_end.hybridz_tf import query_dual_v51b_material_cache as cache


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _deadline(seconds: float = 30.0) -> frozen._Deadline:
    return frozen._Deadline(time.monotonic() + seconds)


def _stage(label: str, cone: int | None) -> cache.ConvMaterialStageUse:
    return cache.ConvMaterialStageUse(_sha(label), cone)


def _conv_layer(
    *,
    layer_id: int = 2,
    predecessor_id: int = 1,
    weight: np.ndarray | None = None,
    input_shape=(2, 2, 2),
    output_shape=(2, 2, 2),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
) -> frozen._FrozenLayer:
    if weight is None:
        weight = np.asarray(
            [
                [[[1.0]], [[-0.25]]],
                [[[0.5]], [[2.0]]],
            ],
            dtype=np.float64,
        )
    weight_array = frozen._immutable_f64_array(
        weight, name="V5.1b toy weight"
    )
    return frozen._FrozenLayer(
        id=layer_id,
        kind="CONV2D",
        preds=(predecessor_id,),
        width=int(np.prod(output_shape)),
        in_vars=(),
        out_vars=(),
        params=MappingProxyType(
            {
                "weight": weight_array,
                "bias_channels": frozen._immutable_f64_array(
                    np.zeros(
                        int(output_shape[0]), dtype=np.float64
                    ),
                    name="V5.1b toy bias",
                ),
                "input_shape": tuple(input_shape),
                "output_shape": tuple(output_shape),
                "stride": tuple(stride),
                "padding": tuple(padding),
                "dilation": tuple(dilation),
                "groups": int(groups),
            }
        ),
    )


def _box(
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> frozen._Box:
    if lower is None:
        lower = np.asarray(
            [-1.0, -0.5, 0.0, -2.0, -0.25, -1.5, 0.0, -0.75],
            dtype=np.float64,
        )
    if upper is None:
        upper = np.asarray(
            [0.75, 1.0, 0.0, 1.5, 2.0, 0.5, 1.25, 0.25],
            dtype=np.float64,
        )
    return frozen._Box(
        lb=frozen._immutable_f64_array(
            lower, name="V5.1b toy lower"
        ),
        ub=frozen._immutable_f64_array(
            upper, name="V5.1b toy upper"
        ),
    )


def _frame(
    *stages: cache.ConvMaterialStageUse,
    label: str = "frame",
    deadline: frozen._Deadline | None = None,
) -> cache.ConvMaterialFrameCandidate:
    return cache.ConvMaterialFrameCandidate(
        frame_content_sha256=_sha(label),
        expected_stage_uses=stages,
        deadline=_deadline() if deadline is None else deadline,
    )


def _admit(
    frame: cache.ConvMaterialFrameCandidate,
    stage: cache.ConvMaterialStageUse,
    *,
    layer: frozen._FrozenLayer | None = None,
    box: frozen._Box | None = None,
    semantics: str = cache.BOX_OUTPUT,
) -> cache.ConvMaterialStageAliasCandidate:
    layer_value = _conv_layer() if layer is None else layer
    return frame.admit(
        stage_use=stage,
        layer=layer_value,
        predecessor_id=layer_value.preds[0],
        predecessor_box=_box() if box is None else box,
        box_semantics=semantics,
    )


def _exact_reverse(
    layer: frozen._FrozenLayer, query: np.ndarray
) -> list[Fraction]:
    return frozen._fraction_conv_reverse(
        [
            Fraction.from_float(float(value))
            for value in np.asarray(query).reshape(-1)
        ],
        layer,
        frozen._TermBudget(100_000),
    )


class QueryDualV51bMaterialCacheTests(unittest.TestCase):
    def test_same_frame_overlapping_stages_reuse_one_physical_core(self):
        first = _stage("first", 4)
        second = _stage("second", 9)
        frame = _frame(first, second)
        alias_a = _admit(frame, first)
        alias_b = _admit(frame, second)
        self.assertIs(_admit(frame, second), alias_b)

        self.assertIsNot(alias_a, alias_b)
        self.assertEqual(
            alias_a.physical_core_content_sha256,
            alias_b.physical_core_content_sha256,
        )
        self.assertEqual(frame.counters["physical_builds"], 1)
        self.assertEqual(
            frame.counters["cross_stage_physical_hits"], 1
        )
        self.assertEqual(frame.counters["stage_aliases"], 2)
        self.assertEqual(frame.counters["admission_full_validations"], 1)

        with mock.patch.object(
            conv_v51,
            "_validate_plan",
            wraps=conv_v51._validate_plan,
        ) as validator:
            committed = frame.commit()
        self.assertEqual(validator.call_count, 1)
        self.assertEqual(committed.physical_builds, 1)
        self.assertEqual(committed.cross_stage_physical_hits, 1)
        self.assertEqual(committed.stage_aliases, 2)
        self.assertEqual(committed.commit_full_validations, 1)
        self.assertFalse(committed.proof_authority)
        self.assertTrue(
            cache.validate_conv_material_commit_candidate(committed)
        )

    def test_public_replay_still_full_validates_and_is_fraction_sound(self):
        stage = _stage("numeric", 3)
        frame = _frame(stage)
        layer = _conv_layer(
            weight=np.asarray(
                [
                    [[[1.0e16]], [[1.0]]],
                    [[[-1.0e16]], [[0.125]]],
                ],
                dtype=np.float64,
            )
        )
        box = _box()
        alias = _admit(frame, stage, layer=layer, box=box)
        coefficient = np.asarray(
            [
                [1.0, -1.0, 0.5, 2.0, -0.25, 1.5, -2.0, 0.75],
                [-1.0, 0.25, 2.0, -0.5, 1.0, -1.5, 0.75, 3.0],
            ],
            dtype=np.float64,
        )
        with mock.patch.object(
            conv_v51,
            "_validate_plan",
            wraps=conv_v51._validate_plan,
        ) as validator:
            result = frame.replay_dense_conv_public(alias, coefficient)
        self.assertEqual(validator.call_count, 1)
        self.assertEqual(
            frame.counters["execution_alias_lookups"], 1
        )

        reference, _ = frozen._conv_reverse_with_error(
            coefficient,
            layer,
            _deadline(),
            frozen._ReplayStats(),
        )
        self.assertEqual(
            [float(value).hex() for value in result.coefficient.reshape(-1)],
            [float(value).hex() for value in reference.reshape(-1)],
        )
        support = np.maximum(np.abs(box.lb), np.abs(box.ub))
        for row in range(coefficient.shape[0]):
            exact = _exact_reverse(layer, coefficient[row])
            required = sum(
                abs(
                    exact[index]
                    - Fraction.from_float(
                        float(result.coefficient[row, index])
                    )
                )
                * Fraction.from_float(float(support[index]))
                for index in range(len(exact))
            )
            self.assertGreaterEqual(
                Fraction.from_float(float(result.scalar_guard[row])),
                required,
            )
        self.assertFalse(result.proof_authority)

    def test_equal_maxabs_different_exact_boxes_do_not_reuse(self):
        first = _stage("box-a", 1)
        second = _stage("box-b", 2)
        frame = _frame(first, second)
        lower_a = -np.ones(8, dtype=np.float64)
        lower_b = -np.asarray(
            [0.5, 1.0, 0.25, 1.0, 0.75, 1.0, 0.125, 1.0],
            dtype=np.float64,
        )
        upper = np.ones(8, dtype=np.float64)
        box_a = _box(lower=lower_a, upper=upper)
        box_b = _box(lower=lower_b, upper=upper)
        self.assertTrue(
            np.array_equal(
                np.maximum(np.abs(box_a.lb), np.abs(box_a.ub)),
                np.maximum(np.abs(box_b.lb), np.abs(box_b.ub)),
            )
        )
        alias_a = _admit(frame, first, box=box_a)
        alias_b = _admit(frame, second, box=box_b)
        self.assertNotEqual(
            alias_a.physical_key_sha256, alias_b.physical_key_sha256
        )
        self.assertEqual(frame.counters["physical_builds"], 2)
        committed = frame.commit()
        self.assertEqual(committed.commit_full_validations, 2)

    def test_pre_and_post_relu_semantics_do_not_reuse(self):
        first = _stage("pre", 1)
        second = _stage("post", 2)
        frame = _frame(first, second)
        alias_a = _admit(
            frame, first, semantics=cache.BOX_RELU_PRE
        )
        alias_b = _admit(
            frame, second, semantics=cache.BOX_RELU_POST
        )
        self.assertNotEqual(
            alias_a.physical_key_sha256, alias_b.physical_key_sha256
        )
        self.assertEqual(frame.counters["physical_builds"], 2)

    def test_second_frame_never_shares_physical_object(self):
        stage = _stage("same-stage", 5)
        frame_a = _frame(stage, label="identical-frame-content")
        frame_b = _frame(stage, label="identical-frame-content")
        alias_a = _admit(frame_a, stage)
        alias_b = _admit(frame_b, stage)
        core_a = frame_a._cores[alias_a.physical_key_sha256]
        core_b = frame_b._cores[alias_b.physical_key_sha256]
        self.assertEqual(
            alias_a.physical_key_sha256, alias_b.physical_key_sha256
        )
        self.assertIsNot(core_a, core_b)
        self.assertIsNot(core_a.plan, core_b.plan)

    def test_weight_geometry_platform_and_implementation_are_in_key(self):
        stage_a = _stage("weight-a", 1)
        stage_b = _stage("weight-b", 2)
        frame = _frame(stage_a, stage_b)
        alias_a = _admit(frame, stage_a)
        changed_weight = _conv_layer(
            weight=np.asarray(
                [
                    [[[1.0]], [[-0.25]]],
                    [[[0.5]], [[2.0000000000000004]]],
                ],
                dtype=np.float64,
            )
        )
        alias_b = _admit(frame, stage_b, layer=changed_weight)
        self.assertNotEqual(
            alias_a.physical_key_sha256, alias_b.physical_key_sha256
        )

        base = dict(
            frame_content_sha256=_sha("f"),
            layer_id=2,
            predecessor_id=1,
            weight_sha256=_sha("w"),
            geometry_sha256=_sha("g"),
            source_lb_sha256=_sha("lb"),
            source_ub_sha256=_sha("ub"),
            box_semantics=cache.BOX_OUTPUT,
            numeric_platform_sha256=_sha("platform"),
            implementation_sha256=_sha("implementation"),
        )
        baseline = cache._json_sha256(cache._physical_key_body(**base))
        for name in (
            "geometry_sha256",
            "numeric_platform_sha256",
            "implementation_sha256",
        ):
            changed = dict(base)
            changed[name] = _sha("changed-" + name)
            self.assertNotEqual(
                baseline,
                cache._json_sha256(cache._physical_key_body(**changed)),
            )

    def test_physical_arrays_have_irrevocable_bytes_backing(self):
        stage = _stage("bytes", 1)
        frame = _frame(stage)
        alias = _admit(frame, stage)
        plan = frame._cores[alias.physical_key_sha256].plan
        arrays = [plan.weight, plan.support]
        for offset in plan.offsets:
            arrays.extend(
                (
                    offset.output_h_indices,
                    offset.output_w_indices,
                    offset.targets,
                    offset.support_flat,
                    offset.channel_support_flat,
                    offset.support_activity_flat,
                )
            )
        self.assertTrue(all(cache._bytes_backed(value) for value in arrays))
        for value in arrays:
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def _mutated_frame(self, label: str):
        first = _stage(label + "-a", 1)
        second = _stage(label + "-b", 2)
        frame = _frame(first, second, label=label)
        alias = _admit(frame, first)
        _admit(frame, second)
        core = frame._cores[alias.physical_key_sha256]
        return frame, core

    def test_plan_and_offset_mutations_fail_closed(self):
        mutations = {
            "plan": lambda plan: object.__setattr__(
                plan, "layer_id", plan.layer_id + 1
            ),
            "offset": lambda plan: object.__setattr__(
                plan.offsets[0],
                "targets",
                conv_v51._immutable_i64(
                    plan.offsets[0].targets[::-1]
                ),
            ),
            "activity": lambda plan: object.__setattr__(
                plan.offsets[0],
                "support_activity_flat",
                conv_v51._immutable_bool(
                    ~plan.offsets[0].support_activity_flat
                ),
            ),
            "support_sum": lambda plan: object.__setattr__(
                plan.offsets[0],
                "support_sum_upper",
                np.nextafter(
                    plan.offsets[0].support_sum_upper, np.inf
                ),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                frame, core = self._mutated_frame("mutation-" + name)
                mutate(core.plan)
                with self.assertRaises(cache.ConvMaterialCacheError):
                    frame.commit()
                with self.assertRaises(cache.ConvMaterialCacheError):
                    frame.commit()

    def test_offset_mutation_fails_alias_validation_and_fast_admit(self):
        stage = _stage("offset-validate", 1)
        frame = _frame(stage)
        alias = _admit(frame, stage)
        core = frame._cores[alias.physical_key_sha256]
        object.__setattr__(
            core.plan.offsets[0],
            "targets",
            conv_v51._immutable_i64(
                core.plan.offsets[0].targets[::-1]
            ),
        )
        self.assertFalse(frame.validate_alias(alias))

        first = _stage("offset-hit-a", 1)
        second = _stage("offset-hit-b", 2)
        hit_frame = _frame(first, second)
        layer = _conv_layer()
        box = _box()
        first_alias = _admit(
            hit_frame, first, layer=layer, box=box
        )
        hit_core = hit_frame._cores[
            first_alias.physical_key_sha256
        ]
        object.__setattr__(
            hit_core.plan.offsets[0],
            "targets",
            conv_v51._immutable_i64(
                hit_core.plan.offsets[0].targets[::-1]
            ),
        )
        with mock.patch.object(
            frozen,
            "_array_digest",
            side_effect=AssertionError("corrupt hit hashed an array"),
        ) as digest, mock.patch.object(
            conv_v51,
            "_validate_plan",
            side_effect=AssertionError(
                "corrupt hit reached full validation"
            ),
        ) as validator:
            with self.assertRaisesRegex(
                cache.ConvMaterialCacheError, "CORE_SEAL_MISMATCH"
            ):
                _admit(
                    hit_frame,
                    second,
                    layer=layer,
                    box=box,
                )
        self.assertEqual(digest.call_count, 0)
        self.assertEqual(validator.call_count, 0)

    def test_alias_copy_transplant_and_mutation_fail_closed(self):
        stage = _stage("alias", 1)
        source = _frame(stage, label="source")
        alias = _admit(source, stage)

        copied = copy.copy(alias)
        self.assertIsNot(copied, alias)
        self.assertFalse(source.validate_alias(copied))

        source_2 = _frame(stage, label="source-2")
        alias_2 = _admit(source_2, stage)
        target = _frame(stage, label="target")
        _admit(target, stage)
        self.assertFalse(target.validate_alias(alias_2))

        mutated_frame = _frame(stage, label="mutated-alias")
        mutated = _admit(mutated_frame, stage)
        object.__setattr__(
            mutated,
            "physical_core_content_sha256",
            _sha("substituted-core"),
        )
        self.assertFalse(mutated_frame.validate_alias(mutated))

    def test_frame_copy_and_container_substitution_fail_closed(self):
        stage = _stage("frame-copy", 1)
        frame = _frame(stage)
        _admit(frame, stage)
        with self.assertRaises(cache.ConvMaterialCacheError):
            copy.copy(frame)
        object.__setattr__(frame, "_cores", {})
        with self.assertRaises(cache.ConvMaterialCacheError):
            frame.commit()

    def test_finite_absolute_deadline_is_mandatory(self):
        stage = _stage("no-deadline", 1)
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError, "INVALID_DEADLINE"
        ):
            _frame(stage, deadline=frozen._Deadline(None))

    def test_expired_deadline_rejects_cache_hit(self):
        first = _stage("deadline-a", 1)
        second = _stage("deadline-b", 2)
        deadline = _deadline()
        frame = _frame(first, second, deadline=deadline)
        _admit(frame, first)
        with mock.patch.object(
            frozen.time,
            "monotonic",
            return_value=float(deadline.end) + 1.0,
        ):
            with self.assertRaises(cache.ConvMaterialCacheTimeout):
                _admit(frame, second)
        self.assertEqual(frame._cores.__len__(), 1)

    def test_forked_pid_cannot_use_parent_alias(self):
        stage = _stage("fork", 1)
        frame = _frame(stage)
        alias = _admit(frame, stage)
        with mock.patch.object(
            cache.os, "getpid", return_value=os.getpid() + 1
        ):
            self.assertFalse(frame.validate_alias(alias))
        self.assertTrue(frame.validate_alias(alias))

    def test_commit_requires_complete_stage_alias_ledger(self):
        first = _stage("ledger-a", 1)
        second = _stage("ledger-b", 2)
        frame = _frame(first, second)
        _admit(frame, first)
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError, "INCOMPLETE_LEDGER"
        ):
            frame.commit()

    def test_copied_or_reconstructed_stage_use_is_rejected(self):
        expected = _stage("sealed-stage", 7)
        copied = copy.copy(expected)
        reconstructed = cache.ConvMaterialStageUse(
            expected.stage_use_sha256, expected.cone_start_lid
        )
        for label, impostor in (
            ("copied", copied),
            ("reconstructed", reconstructed),
        ):
            with self.subTest(label=label):
                frame = _frame(expected, label="stage-" + label)
                with self.assertRaisesRegex(
                    cache.ConvMaterialCacheError,
                    "STAGE_SEAL_MISMATCH",
                ):
                    _admit(frame, impostor)

    def test_mutated_expected_stage_fails_before_commit_receipt(self):
        stage = _stage("mutated-expected", 3)
        frame = _frame(stage)
        _admit(frame, stage)
        object.__setattr__(stage, "cone_start_lid", 99)
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError, "STAGE_SEAL_MISMATCH"
        ):
            frame.commit()

        validate_stage = _stage("mutated-before-validate", 4)
        validate_frame = _frame(validate_stage)
        alias = _admit(validate_frame, validate_stage)
        object.__setattr__(
            validate_stage, "stage_use_sha256", _sha("substituted")
        )
        self.assertFalse(validate_frame.validate_alias(alias))

    def test_mutated_alias_layer_or_predecessor_is_rejected(self):
        for field_name in ("layer_id", "predecessor_id"):
            with self.subTest(field=field_name):
                stage = _stage("alias-" + field_name, 1)
                frame = _frame(stage, label="alias-" + field_name)
                alias = _admit(frame, stage)
                object.__setattr__(
                    alias, field_name, getattr(alias, field_name) + 1
                )
                self.assertFalse(frame.validate_alias(alias))

    def test_registry_deletion_and_counter_changes_fail_closed(self):
        def populated(label):
            first = _stage(label + "-a", 1)
            second = _stage(label + "-b", 2)
            frame = _frame(first, second, label=label)
            _admit(frame, first)
            _admit(frame, second)
            runtime = cache._FRAME_RUNTIMES[frame._nonce]
            return frame, runtime

        mutations = {
            "core": lambda runtime: runtime.cores.pop(
                next(iter(runtime.cores))
            ),
            "core_seal": lambda runtime: runtime.core_seals.pop(
                next(iter(runtime.core_seals))
            ),
            "alias": lambda runtime: runtime.aliases.pop(
                next(iter(runtime.aliases))
            ),
            "alias_seal": lambda runtime: runtime.alias_seals.pop(
                next(iter(runtime.alias_seals))
            ),
            "aliases_by_use": lambda runtime: runtime.aliases_by_use.pop(
                next(iter(runtime.aliases_by_use))
            ),
            "fast_input": lambda runtime: runtime.fast_inputs.pop(
                next(iter(runtime.fast_inputs))
            ),
            "build_counter": lambda runtime: setattr(
                runtime,
                "physical_builds",
                runtime.physical_builds + 1,
            ),
            "hit_counter": lambda runtime: setattr(
                runtime,
                "cross_stage_physical_hits",
                runtime.cross_stage_physical_hits + 1,
            ),
            "alias_counter": lambda runtime: setattr(
                runtime, "alias_mints", runtime.alias_mints + 1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                frame, runtime = populated("deletion-" + label)
                mutate(runtime)
                with self.assertRaisesRegex(
                    cache.ConvMaterialCacheError,
                    "MEMBERSHIP_MISMATCH",
                ):
                    frame.commit()

    def test_fast_input_same_cardinality_substitution_fails_commit(self):
        stage = _stage("fast-seal-substitution", 1)
        frame = _frame(stage)
        _admit(frame, stage)
        runtime = cache._FRAME_RUNTIMES[frame._nonce]
        seal = next(iter(runtime.fast_inputs.values()))
        object.__setattr__(seal, "signature", ("substituted",))
        self.assertEqual(len(runtime.fast_inputs), 1)
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError,
            "FAST_INPUT_SEAL_MISMATCH",
        ):
            frame.commit()

    def test_fast_input_key_redirect_and_replacement_fail_closed(self):
        first = _stage("redirect-a", 1)
        second = _stage("redirect-b", 2)
        third = _stage("redirect-third", 3)
        frame = _frame(first, second, third, label="redirect")
        layer_a = _conv_layer(layer_id=2)
        layer_b = _conv_layer(
            layer_id=3,
            weight=np.asarray(
                [
                    [[[2.0]], [[-0.5]]],
                    [[[1.0]], [[4.0]]],
                ],
                dtype=np.float64,
            ),
        )
        box_a = _box()
        box_b = _box(
            lower=-np.ones(8, dtype=np.float64),
            upper=np.ones(8, dtype=np.float64),
        )
        alias_a = _admit(
            frame, first, layer=layer_a, box=box_a
        )
        alias_b = _admit(
            frame, second, layer=layer_b, box=box_b
        )
        self.assertNotEqual(
            alias_a.physical_key_sha256,
            alias_b.physical_key_sha256,
        )
        runtime = cache._FRAME_RUNTIMES[frame._nonce]
        token_a = cache._fast_input_token(
            layer_a, box_a, layer_a.preds[0], cache.BOX_OUTPUT
        )
        fast_a = runtime.fast_inputs[token_a]
        object.__setattr__(
            fast_a,
            "physical_key_sha256",
            alias_b.physical_key_sha256,
        )
        self.assertFalse(
            cache._fast_trust_admit_once(
                runtime.nonce,
                token_a,
                fast_a,
                snapshot=fast_a.snapshot,
            )
        )
        self.assertFalse(
            hasattr(cache, "_FAST_INPUT_EXTERNAL_SEALS")
        )
        self.assertFalse(
            hasattr(cache, "_make_fast_input_external_seal")
        )
        with mock.patch.object(
            frozen,
            "_array_digest",
            side_effect=AssertionError("redirect hit hashed"),
        ) as digest, mock.patch.object(
            conv_v51,
            "_validate_plan",
            side_effect=AssertionError(
                "redirect hit reached full validation"
            ),
        ) as validator:
            with self.assertRaisesRegex(
                cache.ConvMaterialCacheError,
                "FAST_INPUT_SEAL_MISMATCH",
            ):
                _admit(
                    frame,
                    third,
                    layer=layer_a,
                    box=box_a,
                )
        self.assertEqual(digest.call_count, 0)
        self.assertEqual(validator.call_count, 0)

        for label, replacement in (
            ("copy", lambda value: copy.copy(value)),
            (
                "reconstructed",
                lambda value: cache._FastInputSeal(
                    token=value.token,
                    signature=value.signature,
                    physical_key_sha256=(
                        value.physical_key_sha256
                    ),
                    layer=value.layer,
                    predecessor_box=value.predecessor_box,
                    snapshot=value.snapshot,
                ),
            ),
        ):
            with self.subTest(label=label):
                stage = _stage("replace-" + label, 1)
                replace_frame = _frame(
                    stage, label="replace-" + label
                )
                _admit(replace_frame, stage)
                replace_runtime = cache._FRAME_RUNTIMES[
                    replace_frame._nonce
                ]
                token = next(iter(replace_runtime.fast_inputs))
                original = replace_runtime.fast_inputs[token]
                replace_runtime.fast_inputs[token] = replacement(
                    original
                )
                self.assertEqual(len(replace_runtime.fast_inputs), 1)
                with self.assertRaisesRegex(
                    cache.ConvMaterialCacheError,
                    "FAST_INPUT_SEAL_MISMATCH",
                ):
                    replace_frame.commit()

        external_first = _stage("external-replace-a", 1)
        external_second = _stage("external-replace-b", 2)
        external_frame = _frame(
            external_first,
            external_second,
            label="external-replace",
        )
        external_layer = _conv_layer()
        external_box = _box()
        _admit(
            external_frame,
            external_first,
            layer=external_layer,
            box=external_box,
        )
        external_runtime = cache._FRAME_RUNTIMES[
            external_frame._nonce
        ]
        external_token = next(iter(external_runtime.fast_inputs))
        external_original = external_runtime.fast_inputs[
            external_token
        ]
        external_replacement = copy.copy(external_original)
        self.assertFalse(
            cache._fast_trust_admit_once(
                external_runtime.nonce,
                external_token,
                external_replacement,
                snapshot=external_replacement.snapshot,
            )
        )
        accepted = _admit(
            external_frame,
            external_second,
            layer=external_layer,
            box=external_box,
        )
        self.assertEqual(
            accepted.physical_key_sha256,
            external_original.physical_key_sha256,
        )

    def test_never_registered_forged_first_binding_is_rejected(self):
        stage_b = _stage("first-forgery-b", 1)
        stage_a = _stage("first-forgery-a", 2)
        frame = _frame(
            stage_b, stage_a, label="first-registration-forgery"
        )
        layer_b = _conv_layer(
            layer_id=3,
            weight=np.asarray(
                [
                    [[[2.0]], [[-0.5]]],
                    [[[1.0]], [[4.0]]],
                ],
                dtype=np.float64,
            ),
        )
        box_b = _box(
            lower=-np.ones(8, dtype=np.float64),
            upper=np.ones(8, dtype=np.float64),
        )
        alias_b = _admit(
            frame, stage_b, layer=layer_b, box=box_b
        )

        layer_a = _conv_layer(layer_id=2)
        box_a = _box()
        token_a = cache._fast_input_token(
            layer_a, box_a, layer_a.preds[0], cache.BOX_OUTPUT
        )
        signature_a = cache._fast_input_signature(
            layer_a, box_a, layer_a.preds[0], cache.BOX_OUTPUT
        )
        self.assertIsNotNone(signature_a)
        snapshot_a = cache._snapshot_fast_input(
            layer_a, box_a, layer_a.preds[0], cache.BOX_OUTPUT
        )
        forged = cache._FastInputSeal(
            token=token_a,
            signature=signature_a,
            physical_key_sha256=alias_b.physical_key_sha256,
            layer=snapshot_a.layer,
            predecessor_box=snapshot_a.predecessor_box,
            snapshot=snapshot_a,
        )
        runtime = cache._FRAME_RUNTIMES[frame._nonce]
        runtime.fast_inputs[token_a] = forged
        runtime.fast_input_mints += 1
        self.assertFalse(
            cache._fast_trust_admit_once(
                runtime.nonce,
                token_a,
                forged,
                snapshot=snapshot_a,
            )
        )
        self.assertFalse(
            hasattr(cache, "_fast_trust_register_once")
        )

        with mock.patch.object(
            frozen,
            "_array_digest",
            side_effect=AssertionError("forged hit hashed"),
        ) as digest, mock.patch.object(
            conv_v51,
            "_validate_plan",
            side_effect=AssertionError(
                "forged hit reached full validation"
            ),
        ) as validator:
            with self.assertRaises(cache.ConvMaterialCacheError):
                _admit(
                    frame,
                    stage_a,
                    layer=layer_a,
                    box=box_a,
                )
        self.assertEqual(digest.call_count, 0)
        self.assertEqual(validator.call_count, 0)
        with self.assertRaises(cache.ConvMaterialCacheError):
            frame.commit()

    def test_public_dynamic_box_toctou_is_rejected_before_read(self):
        lower_a = frozen._immutable_f64_array(
            -np.ones(8, dtype=np.float64),
            name="dynamic lower A",
        )
        upper_a = frozen._immutable_f64_array(
            np.ones(8, dtype=np.float64),
            name="dynamic upper A",
        )
        lower_b = frozen._immutable_f64_array(
            -2.0 * np.ones(8, dtype=np.float64),
            name="dynamic lower B",
        )
        upper_b = frozen._immutable_f64_array(
            2.0 * np.ones(8, dtype=np.float64),
            name="dynamic upper B",
        )

        class DynamicBox(frozen._Box):
            reads = 0

            def __getattribute__(self, name):
                if name in {"lb", "ub"}:
                    cls = type(self)
                    index = cls.reads
                    cls.reads += 1
                    use_b = index % 4 in {1, 2}
                    if name == "lb":
                        return lower_b if use_b else lower_a
                    return upper_b if use_b else upper_a
                return super().__getattribute__(name)

        dynamic = object.__new__(DynamicBox)
        stage = _stage("dynamic-box", 1)
        frame = _frame(stage)
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError, "INVALID_LAYER"
        ):
            frame.admit(
                stage_use=stage,
                layer=_conv_layer(),
                predecessor_id=1,
                predecessor_box=dynamic,
                box_semantics=cache.BOX_OUTPUT,
            )
        self.assertEqual(DynamicBox.reads, 0)
        self.assertEqual(len(frame._cores), 0)

    def test_dynamic_mappingproxy_backing_is_rejected_before_read(self):
        base = _conv_layer()

        class DynamicParams(dict):
            reads = 0

            def __getitem__(self, key):
                type(self).reads += 1
                return super().__getitem__(key)

        dynamic_params = DynamicParams(dict(base.params))
        layer = frozen._FrozenLayer(
            id=base.id,
            kind=base.kind,
            preds=base.preds,
            width=base.width,
            in_vars=base.in_vars,
            out_vars=base.out_vars,
            params=MappingProxyType(dynamic_params),
        )
        stage = _stage("dynamic-params", 1)
        frame = _frame(stage)
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError, "FAST_INPUT_BINDING"
        ):
            frame.admit(
                stage_use=stage,
                layer=layer,
                predecessor_id=1,
                predecessor_box=_box(),
                box_semantics=cache.BOX_OUTPUT,
            )
        self.assertEqual(DynamicParams.reads, 0)
        self.assertEqual(len(frame._cores), 0)

    def test_commit_independently_reconstructs_every_fast_key(self):
        first = _stage("commit-key-a", 1)
        second = _stage("commit-key-b", 2)
        frame = _frame(first, second, label="commit-key")
        layer_a = _conv_layer(layer_id=2)
        layer_b = _conv_layer(
            layer_id=3,
            weight=np.asarray(
                [
                    [[[2.0]], [[-0.5]]],
                    [[[1.0]], [[4.0]]],
                ],
                dtype=np.float64,
            ),
        )
        box_a = _box()
        box_b = _box(
            lower=-np.ones(8, dtype=np.float64),
            upper=np.ones(8, dtype=np.float64),
        )
        alias_a = _admit(
            frame, first, layer=layer_a, box=box_a
        )
        alias_b = _admit(
            frame, second, layer=layer_b, box=box_b
        )
        runtime = cache._FRAME_RUNTIMES[frame._nonce]
        token_a = cache._fast_input_token(
            layer_a, box_a, layer_a.preds[0], cache.BOX_OUTPUT
        )
        token_b = cache._fast_input_token(
            layer_b, box_b, layer_b.preds[0], cache.BOX_OUTPUT
        )
        seal_a = runtime.fast_inputs[token_a]
        seal_b = runtime.fast_inputs[token_b]
        for name, value in (
            ("layer", seal_b.layer),
            ("predecessor_box", seal_b.predecessor_box),
            ("signature", seal_b.signature),
            ("physical_key_sha256", alias_b.physical_key_sha256),
        ):
            object.__setattr__(seal_a, name, value)
        self.assertNotEqual(
            alias_a.physical_key_sha256,
            alias_b.physical_key_sha256,
        )
        self.assertFalse(
            cache._fast_trust_commit_validate(
                runtime.nonce, token_a, seal_a
            )
        )
        # Even if the fast MAC-validation call were compromised, commit's
        # independent SHA reconstruction must reject A->B redirection.
        with mock.patch.object(
            cache, "_fast_trust_validate", return_value=True
        ):
            with self.assertRaises(cache.ConvMaterialCacheError):
                frame.commit()

    def test_external_fast_trust_drop_fails_closed(self):
        stage = _stage("external-drop", 1)
        frame = _frame(stage)
        _admit(frame, stage)
        cache._fast_trust_drop(frame._nonce)
        self.assertIsNone(
            cache._fast_trust_create(
                frame._nonce,
                frame_content_sha256=frame.frame_content_sha256,
                numeric_platform_sha256=(
                    frame._numeric_platform_sha256
                ),
                implementation_sha256=(
                    frame._implementation_sha256
                ),
            )
        )
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError,
            "FAST_INPUT_EXTERNAL_SEAL_MISMATCH",
        ):
            frame.commit()

    def test_validate_alias_rechecks_deadline_after_validation(self):
        stage = _stage("delayed-validate", 1)
        deadline = _deadline()
        frame = _frame(stage, deadline=deadline)
        alias = _admit(frame, stage)
        with mock.patch.object(
            frozen.time,
            "monotonic",
            side_effect=[
                float(deadline.end) - 1.0,
                float(deadline.end) + 1.0,
            ],
        ):
            self.assertFalse(frame.validate_alias(alias))

    def test_cache_hit_and_commit_recheck_deadline_after_work(self):
        first = _stage("delayed-hit-a", 1)
        second = _stage("delayed-hit-b", 2)
        deadline = _deadline()
        frame = _frame(first, second, deadline=deadline)
        layer = _conv_layer()
        box = _box()
        _admit(frame, first, layer=layer, box=box)
        with mock.patch.object(
            frozen.time,
            "monotonic",
            side_effect=[
                float(deadline.end) - 1.0,
                float(deadline.end) + 1.0,
            ],
        ):
            with self.assertRaises(cache.ConvMaterialCacheTimeout):
                _admit(frame, second, layer=layer, box=box)

        commit_stage = _stage("delayed-commit", 1)
        commit_deadline = _deadline()
        commit_frame = _frame(
            commit_stage, deadline=commit_deadline
        )
        _admit(commit_frame, commit_stage)
        clock = [float(commit_deadline.end) - 1.0]
        original_validator = conv_v51._validate_plan

        def delayed_validator(*args, **kwargs):
            original_validator(*args, **kwargs)
            clock[0] = float(commit_deadline.end) + 1.0

        with mock.patch.object(
            frozen.time, "monotonic", side_effect=lambda: clock[0]
        ), mock.patch.object(
            conv_v51, "_validate_plan", side_effect=delayed_validator
        ):
            with self.assertRaises(cache.ConvMaterialCacheTimeout):
                commit_frame.commit()

    def test_cache_hit_has_zero_array_digest_and_zero_math_validation(self):
        first = _stage("fast-a", 1)
        second = _stage("fast-b", 2)
        frame = _frame(first, second, label="fast-hit")
        layer = _conv_layer()
        box = _box()
        _admit(frame, first, layer=layer, box=box)
        with mock.patch.object(
            frozen,
            "_array_digest",
            side_effect=AssertionError("cache hit hashed an array"),
        ) as digest, mock.patch.object(
            conv_v51,
            "_validate_plan",
            side_effect=AssertionError("cache hit full-validated"),
        ) as validator:
            _admit(frame, second, layer=layer, box=box)
        self.assertEqual(digest.call_count, 0)
        self.assertEqual(validator.call_count, 0)
        self.assertEqual(frame.counters["physical_builds"], 1)
        self.assertEqual(
            frame.counters["cross_stage_physical_hits"], 1
        )

    def test_six_conv_cores_nineteen_aliases_thirteen_fast_hits(self):
        stages = tuple(_stage(f"topology-{index}", index) for index in range(19))
        frame = _frame(*stages, label="six-core-topology")
        layers = []
        boxes = []
        for index in range(6):
            weight = np.asarray(
                [
                    [[[1.0 + index / 16.0]], [[-0.25]]],
                    [[[0.5]], [[2.0 + index / 32.0]]],
                ],
                dtype=np.float64,
            )
            layers.append(
                _conv_layer(
                    layer_id=2 + index,
                    predecessor_id=1,
                    weight=weight,
                )
            )
            lower = -np.ones(8, dtype=np.float64)
            upper = np.ones(8, dtype=np.float64)
            lower[index] = -(0.25 + index / 16.0)
            boxes.append(_box(lower=lower, upper=upper))

        aliases = []
        for index in range(6):
            aliases.append(
                _admit(
                    frame,
                    stages[index],
                    layer=layers[index],
                    box=boxes[index],
                )
            )
        with mock.patch.object(
            frozen,
            "_array_digest",
            side_effect=AssertionError("topology cache hit hashed"),
        ) as digest, mock.patch.object(
            conv_v51,
            "_validate_plan",
            side_effect=AssertionError(
                "topology cache hit full-validated"
            ),
        ) as validator:
            for index in range(6, 19):
                core_index = index % 6
                aliases.append(
                    _admit(
                        frame,
                        stages[index],
                        layer=layers[core_index],
                        box=boxes[core_index],
                    )
                )
        self.assertEqual(digest.call_count, 0)
        self.assertEqual(validator.call_count, 0)
        self.assertEqual(len({value._nonce for value in aliases}), 19)
        self.assertEqual(frame.counters["physical_builds"], 6)
        self.assertEqual(frame.counters["stage_aliases"], 19)
        self.assertEqual(
            frame.counters["cross_stage_physical_hits"], 13
        )

        with mock.patch.object(
            conv_v51,
            "_validate_plan",
            wraps=conv_v51._validate_plan,
        ) as commit_validator:
            committed = frame.commit()
        self.assertEqual(commit_validator.call_count, 6)
        self.assertEqual(committed.physical_builds, 6)
        self.assertEqual(committed.stage_aliases, 19)
        self.assertEqual(
            committed.cross_stage_physical_hits, 13
        )
        self.assertEqual(committed.commit_full_validations, 6)
        self.assertEqual(committed.execution_alias_lookups, 0)

    def test_unregistered_stage_is_rejected(self):
        expected = _stage("expected", 1)
        unexpected = _stage("unexpected", 2)
        frame = _frame(expected)
        with self.assertRaisesRegex(
            cache.ConvMaterialCacheError, "STAGE_SEAL_MISMATCH"
        ):
            _admit(frame, unexpected)


if __name__ == "__main__":
    unittest.main()
