"""CPU focused tests for the scratch sealed eliminated-target transaction."""

from __future__ import annotations

from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import threading
import time
import unittest
import weakref

import numpy as np

from act.back_end.hybridz_tf import phase_projection_highs_owner as _owner
import scratch_phase_projection_sealed_eliminated_target_transaction as _tx


_SOURCE_BINDING_SHA256 = hashlib.sha256(
    b"sealed-eliminated-target-synthetic-source-v1"
).hexdigest()


def _f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _bool(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.bool_)


def _fraction_up(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    return rounded


def _fixture() -> tuple[dict[str, object], tuple[_tx.EliminatedTargetPartitionInput, ...]]:
    first0 = _f64(
        [
            [1.0, 0.0, 5.0e-13],
            [0.25, -0.5, 0.0],
        ]
    )
    delta0 = _f64(
        [
            [0.0, 0.0],
            [0.2, 0.0],
        ]
    )
    first1 = _f64(
        [
            [-0.75, 0.125, 0.5],
            [0.0, 1.25, -0.25],
        ]
    )
    delta1 = _f64(
        [
            [0.1, 0.0],
            [0.3, -0.4],
        ]
    )
    partitions = (
        _tx.EliminatedTargetPartitionInput(
            partition_id=11,
            first_pre=first0,
            delta_pre=delta0,
            phase_centers=_f64([-0.4, -100.0]),
            target_active=_bool([True, False]),
            row_ids=_i64([0, 1]),
            stream_rows=_i64([7, 13]),
        ),
        _tx.EliminatedTargetPartitionInput(
            partition_id=29,
            first_pre=first1,
            delta_pre=delta1,
            phase_centers=_f64([-0.2, 0.1]),
            target_active=_bool([True, False]),
            row_ids=_i64([2, 3]),
            stream_rows=_i64([2, 19]),
        ),
    )
    kwargs: dict[str, object] = {
        "partitions": partitions,
        "expected_layout": (
            _tx.EliminatedTargetPartitionLayout(
                partition_id=11,
                row_ids=(0, 1),
                stream_rows=(7, 13),
            ),
            _tx.EliminatedTargetPartitionLayout(
                partition_id=29,
                row_ids=(2, 3),
                stream_rows=(2, 19),
            ),
        ),
        "source_binding_sha256": _SOURCE_BINDING_SHA256,
        "change_ordinals": _i64([0, 2]),
        "change_base_active": _bool([False, False]),
        "change_target_active": _bool([True, True]),
        "x_lower": _f64([-1.0, -1.0, -1.0]),
        "x_upper": _f64([1.0, 1.0, 1.0]),
        "first_output": _f64(
            [
                [0.75, -0.5, 0.25],
                [-0.125, 0.625, 0.5],
            ]
        ),
        "delta_output": _f64(
            [
                [0.2, -0.3],
                [0.4, 0.1],
            ]
        ),
        "property_row": _f64([1.25, -0.75]),
        "target_output_center": _f64([0.2, -0.1]),
        "threshold": 0.05,
    }
    return kwargs, partitions


def _build() -> tuple[
    _tx.SealedEliminatedTargetTransaction,
    dict[str, object],
    tuple[_tx.EliminatedTargetPartitionInput, ...],
]:
    kwargs, partitions = _fixture()
    return _tx.build_sealed_eliminated_target_transaction(**kwargs), kwargs, partitions


class SealedEliminatedTargetTransactionTests(unittest.TestCase):
    def test_bitwise_triangular_partition_and_objective_order(self) -> None:
        transaction, kwargs, partitions = _build()
        try:
            expected_u = np.zeros((2, 3), dtype=np.float64)
            expected_u[0] = partitions[0].first_pre[0]
            expected_u[1] = partitions[1].first_pre[0]
            expected_u[1] += (
                partitions[1].delta_pre[0, :1] @ expected_u[:1]
            )
            self.assertTrue(np.array_equal(transaction.expansion, expected_u))

            for snapshot, source in zip(transaction.partitions, partitions):
                target = source.first_pre + np.asarray(
                    source.delta_pre @ expected_u, dtype=np.float64
                )
                sign = np.where(source.target_active, -1.0, 1.0)[:, None]
                expected = target * sign
                observed = snapshot.logical_rows.as_csr().toarray()
                self.assertTrue(np.array_equal(observed, expected))

            target_output = kwargs["first_output"] + np.asarray(
                kwargs["delta_output"] @ expected_u, dtype=np.float64
            )
            expected_objective = np.asarray(
                kwargs["property_row"][None, :] @ target_output,
                dtype=np.float64,
            ).reshape(-1)
            self.assertTrue(
                np.array_equal(transaction.objective_coeff, expected_objective)
            )
            expected_center = float(
                kwargs["property_row"] @ kwargs["target_output_center"] - 0.05
            )
            self.assertEqual(transaction.objective_center, expected_center)
            self.assertTrue(
                np.array_equal(transaction.base_rows.row_ids, _i64([0, 2, 3]))
            )
            self.assertTrue(
                np.array_equal(transaction.partitions[0].keep, _bool([True, False]))
            )
        finally:
            transaction.close()

    def test_nonassociative_objective_fixture_rejects_reassociation(self) -> None:
        scale = 0.0625
        desired_u = _f64(
            [
                [2.0, -2.0],
                [-67108864.0, 9007199254740992.0],
            ]
        ) * scale
        first_output = _f64(
            [
                [-9007199254740992.0, -9007199254740992.0],
                [67108864.0, 1.0000000000000002],
            ]
        ) * scale
        delta_output = _f64(
            [
                [1.0, -1.0],
                [-67108864.0, 1.0000000000000002],
            ]
        )
        property_row = _f64([-2.0, -67108864.0]) * (2.0 ** -9)
        partitions = (
            _tx.EliminatedTargetPartitionInput(
                partition_id=3,
                first_pre=_f64([desired_u[0], -desired_u[1]]),
                delta_pre=np.zeros((2, 2), dtype=np.float64),
                phase_centers=_f64([-1.0, 0.0]),
                target_active=_bool([True, False]),
                row_ids=_i64([0, 1]),
                stream_rows=_i64([5, 9]),
            ),
        )
        transaction = _tx.build_sealed_eliminated_target_transaction(
            partitions=partitions,
            expected_layout=(
                _tx.EliminatedTargetPartitionLayout(
                    partition_id=3,
                    row_ids=(0, 1),
                    stream_rows=(5, 9),
                ),
            ),
            source_binding_sha256=_SOURCE_BINDING_SHA256,
            change_ordinals=_i64([0, 1]),
            change_base_active=_bool([False, True]),
            change_target_active=_bool([True, False]),
            x_lower=_f64([-1.0, -1.0]),
            x_upper=_f64([1.0, 1.0]),
            first_output=first_output,
            delta_output=delta_output,
            property_row=property_row,
            target_output_center=_f64([0.0, 0.0]),
            threshold=0.0,
        )
        try:
            production_order = property_row @ (
                first_output + delta_output @ desired_u
            )
            reassociated = (
                property_row @ first_output
                + (property_row @ delta_output) @ desired_u
            )
            self.assertFalse(np.array_equal(production_order, reassociated))
            self.assertEqual(
                float(production_order[1] - reassociated[1]), -8192.0
            )
            self.assertTrue(
                np.array_equal(transaction.objective_coeff, production_order)
            )
            self.assertFalse(
                np.array_equal(transaction.objective_coeff, reassociated)
            )
        finally:
            transaction.close()

    def test_screen_precedes_exact_tiny_projection(self) -> None:
        transaction, _kwargs, _partitions = _build()
        try:
            rows = transaction.base_rows
            self.assertEqual(rows.logical_nnz, 8)
            self.assertEqual(rows.deleted_tiny_nnz, 1)
            base_position = int(np.flatnonzero(rows.row_ids == 0)[0])
            exact_rhs = Fraction.from_float(-0.4)
            coefficient = Fraction.from_float(-5.0e-13)
            lo = coefficient * Fraction.from_float(-1.0)
            hi = coefficient * Fraction.from_float(1.0)
            expected_upper = _fraction_up(exact_rhs - min(lo, hi))
            self.assertEqual(float(rows.upper[base_position]), expected_upper)
            start = int(rows.indptr[base_position])
            stop = int(rows.indptr[base_position + 1])
            self.assertNotIn(2, rows.indices[start:stop].tolist())
            self.assertEqual(transaction.receipt.base_deleted_tiny_nnz, 1)
        finally:
            transaction.close()

    def test_caller_aba_is_sealed_and_backing_is_bytes_readonly(self) -> None:
        transaction, kwargs, partitions = _build()
        expected_u = transaction.expansion.copy()
        expected_logical = transaction.partitions[0].logical_rows.data.copy()
        partitions[0].first_pre[:] = 777.0
        partitions[0].delta_pre[:] = -888.0
        partitions[0].phase_centers[:] = 999.0
        kwargs["x_lower"][:] = -7.0
        kwargs["first_output"][:] = 12.0
        kwargs["property_row"][:] = 13.0
        transaction.assert_intact()
        self.assertTrue(np.array_equal(transaction.expansion, expected_u))
        self.assertTrue(
            np.array_equal(transaction.partitions[0].logical_rows.data, expected_logical)
        )
        with self.assertRaises(ValueError):
            transaction.expansion.setflags(write=True)
        with self.assertRaises(ValueError):
            transaction.partitions[0].logical_rows.data[0] = 0.0
        transaction.close()

    def test_same_owner_consumes_base_rows_once_on_cpu(self) -> None:
        transaction, kwargs, _partitions = _build()
        owner = _owner.SafeHighsOwner(deadline_monotonic=time.monotonic() + 10.0)
        with owner:
            result = owner.solve_base(
                cost=np.ascontiguousarray(-transaction.objective_coeff),
                column_lower=np.ascontiguousarray(kwargs["x_lower"]),
                column_upper=np.ascontiguousarray(kwargs["x_upper"]),
                rows=transaction.base_rows,
            )
            self.assertIn(
                type(result).__name__,
                {"OptimalSelector", "InfeasibleSelector", "Unresolved"},
            )
            self.assertIn(owner.state, {"BASE_SOLVED", "BASE_FAILED"})
        self.assertEqual(owner.state, "CLOSED")
        transaction.close()

    def test_one_shot_partition_repair_handoff_and_release(self) -> None:
        transaction, kwargs, _partitions = _build()
        original_partitions = transaction.partitions
        repair_parts = (
            _f64([[0.0, 0.0], [0.1, 0.0]]),
            _f64([[0.2, 0.0], [0.3, 0.4]]),
        )
        repair_output = _f64([[0.5, -0.2], [-0.25, 0.75]])
        lease = transaction.begin_repair(
            selected_ordinals=_i64([0, 2]),
            repair_delta_parts=repair_parts,
            repair_delta_output=repair_output,
        )
        self.assertEqual(transaction.state, "REPAIR_LEASED")
        self.assertEqual(lease.state, "ACTIVE")
        lease.assert_intact()
        self.assertEqual(
            lease.partitions[0].logical_rows.content_sha256,
            original_partitions[0].logical_rows.content_sha256,
        )
        self.assertEqual(
            lease.partitions[1].logical_rows.content_sha256,
            original_partitions[1].logical_rows.content_sha256,
        )
        self.assertTrue(
            np.array_equal(lease.selected_base_row_positions, _i64([0, 1]))
        )
        expected_objective = np.asarray(
            kwargs["property_row"][None, :] @ repair_output,
            dtype=np.float64,
        ).reshape(-1)
        self.assertTrue(np.array_equal(lease.objective_delta, expected_objective))
        repair_parts[1][:] = 919.0
        repair_output[:] = -313.0
        lease.assert_intact()
        self.assertEqual(float(lease.partitions[1].repair_delta[1, 1]), 0.4)
        lease.release()
        self.assertEqual(lease.state, "RELEASED")
        self.assertEqual(transaction.state, "CLOSED")
        with self.assertRaises(_tx.SealedEliminatedTargetUnknown):
            _ = lease.partitions
        with self.assertRaises(_tx.SealedEliminatedTargetUnknown):
            _ = transaction.base_rows

    def test_bad_repair_causality_fails_closed_without_retry(self) -> None:
        transaction, _kwargs, _partitions = _build()
        bad_parts = (
            _f64([[1.0, 0.0], [0.0, 0.0]]),
            _f64([[0.0, 0.0], [0.0, 0.0]]),
        )
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "causality"
        ):
            transaction.begin_repair(
                selected_ordinals=_i64([0, 2]),
                repair_delta_parts=bad_parts,
                repair_delta_output=_f64([[1.0, 0.0], [0.0, 1.0]]),
            )
        self.assertEqual(transaction.state, "CLOSED")
        with self.assertRaises(_tx.SealedEliminatedTargetUnknown):
            transaction.begin_repair(
                selected_ordinals=_i64([0]),
                repair_delta_parts=(_f64([[0.0], [0.0]]), _f64([[0.0], [0.0]])),
                repair_delta_output=_f64([[1.0], [1.0]]),
            )

    def test_resource_and_row_id_preflight_fail_before_build(self) -> None:
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "fixed cap"
        ):
            _tx.preflight_eliminated_target_resource_frame(
                _tx.EliminatedTargetResourceFrame(
                    partition_rows=(200_001,),
                    input_columns=1,
                    initial_changes=1,
                    output_rows=1,
                )
            )
        kwargs, partitions = _fixture()
        bad_partition = _tx.EliminatedTargetPartitionInput(
            partition_id=partitions[1].partition_id,
            first_pre=partitions[1].first_pre,
            delta_pre=partitions[1].delta_pre,
            phase_centers=partitions[1].phase_centers,
            target_active=partitions[1].target_active,
            row_ids=_i64([9, 10]),
            stream_rows=partitions[1].stream_rows,
        )
        kwargs["partitions"] = (partitions[0], bad_partition)
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "row order"
        ):
            _tx.build_sealed_eliminated_target_transaction(**kwargs)

    def test_expected_layer_layout_rejects_reorder_and_split(self) -> None:
        kwargs, partitions = _fixture()
        reversed_partition = _tx.EliminatedTargetPartitionInput(
            partition_id=partitions[0].partition_id,
            first_pre=partitions[0].first_pre,
            delta_pre=partitions[0].delta_pre,
            phase_centers=partitions[0].phase_centers,
            target_active=partitions[0].target_active,
            row_ids=partitions[0].row_ids,
            stream_rows=_i64([13, 7]),
        )
        kwargs["partitions"] = (reversed_partition, partitions[1])
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "row order"
        ):
            _tx.build_sealed_eliminated_target_transaction(**kwargs)

    def test_partition_boundary_has_a_real_binary64_consequence(self) -> None:
        first = np.zeros((2, 2), dtype=np.float64)
        delta = _f64(
            [
                [8.411607067912387e-09, 5.469958128571948e-09],
                [1.7132810989197983e-09, 2.182014926213453e-09],
            ]
        )
        expansion = _f64(
            [
                [0.06497790015582014, -0.11361745786535811],
                [0.12179827336798763, -0.09824334070009909],
            ]
        )
        production_layer = first + delta @ expansion
        split_layers = np.vstack(
            [
                first[index : index + 1]
                + delta[index : index + 1] @ expansion
                for index in range(2)
            ]
        )
        self.assertFalse(np.array_equal(production_layer, split_layers))
        self.assertEqual(
            float(production_layer[1, 1]), -4.090270788767245e-10
        )
        self.assertEqual(
            float(split_layers[1, 1]), -4.0902707887672445e-10
        )

        kwargs, partitions = _fixture()
        split0 = _tx.EliminatedTargetPartitionInput(
            partition_id=11,
            first_pre=np.ascontiguousarray(partitions[0].first_pre[:1]),
            delta_pre=np.ascontiguousarray(partitions[0].delta_pre[:1]),
            phase_centers=np.ascontiguousarray(partitions[0].phase_centers[:1]),
            target_active=np.ascontiguousarray(partitions[0].target_active[:1]),
            row_ids=_i64([0]),
            stream_rows=_i64([7]),
        )
        split1 = _tx.EliminatedTargetPartitionInput(
            partition_id=12,
            first_pre=np.ascontiguousarray(partitions[0].first_pre[1:]),
            delta_pre=np.ascontiguousarray(partitions[0].delta_pre[1:]),
            phase_centers=np.ascontiguousarray(partitions[0].phase_centers[1:]),
            target_active=np.ascontiguousarray(partitions[0].target_active[1:]),
            row_ids=_i64([1]),
            stream_rows=_i64([13]),
        )
        kwargs["partitions"] = (split0, split1, partitions[1])
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "expected layout"
        ):
            _tx.build_sealed_eliminated_target_transaction(**kwargs)

    def test_validation_to_seal_hostile_is_rejected(self) -> None:
        kwargs, partitions = _fixture()
        original = _tx._seal_caller_array

        def hostile(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            if name == "partition 1 first preactivation":
                partitions[1].row_ids[:] = [2, 99]
                partitions[1].stream_rows[:] = [2, 2]
                partitions[1].target_active[1] = True
            return original(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )

        _tx._seal_caller_array = hostile
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "row order"
            ):
                _tx.build_sealed_eliminated_target_transaction(**kwargs)
        finally:
            _tx._seal_caller_array = original

    def test_expected_layout_fields_are_captured_once(self) -> None:
        kwargs, partitions = _fixture()
        second_layout = kwargs["expected_layout"][1]
        hostile_second_partition = _tx.EliminatedTargetPartitionInput(
            partition_id=11,
            first_pre=partitions[1].first_pre,
            delta_pre=partitions[1].delta_pre,
            phase_centers=partitions[1].phase_centers,
            target_active=partitions[1].target_active,
            row_ids=partitions[1].row_ids,
            stream_rows=partitions[1].stream_rows,
        )
        kwargs["partitions"] = (partitions[0], hostile_second_partition)
        layout_type = _tx.EliminatedTargetPartitionLayout
        original_getattribute = layout_type.__getattribute__
        reads = [0]

        def hostile_getattribute(self, name):
            if self is second_layout and name == "partition_id":
                reads[0] += 1
                return 29 if reads[0] <= 3 else 11
            return original_getattribute(self, name)

        layout_type.__getattribute__ = hostile_getattribute
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "partition id differs"
            ):
                _tx.build_sealed_eliminated_target_transaction(**kwargs)
        finally:
            layout_type.__getattribute__ = original_getattribute
        self.assertEqual(reads[0], 1)

    def test_snapshot_dtype_changes_are_rejected_at_every_handoff(self) -> None:
        kwargs, partitions = _fixture()
        original_seal = _tx._seal_caller_array

        def partition_dtype_hostile(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            if name == "partition 1 first preactivation":
                partitions[1].row_ids.dtype = np.float64
                partitions[1].row_ids[:] = [2.0, 3.0]
                partitions[1].stream_rows.dtype = np.float64
                partitions[1].stream_rows[:] = [2.0, 19.0]
            return original_seal(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )

        _tx._seal_caller_array = partition_dtype_hostile
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "shape or authority"
            ):
                _tx.build_sealed_eliminated_target_transaction(**kwargs)
        finally:
            _tx._seal_caller_array = original_seal

        kwargs, _partitions = _fixture()

        def top_dtype_hostile(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            if name == "partition 0 first preactivation":
                kwargs["change_ordinals"].dtype = np.float64
                kwargs["change_ordinals"][:] = [0.0, 2.0]
            return original_seal(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )

        _tx._seal_caller_array = top_dtype_hostile
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "shape or authority"
            ):
                _tx.build_sealed_eliminated_target_transaction(**kwargs)
        finally:
            _tx._seal_caller_array = original_seal

        transaction, _kwargs, _partitions = _build()
        selected = _i64([0, 2])

        def repair_dtype_hostile(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            if name == "selected ordinals":
                selected.dtype = np.float64
                selected[:] = [0.0, 2.0]
            return original_seal(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )

        _tx._seal_caller_array = repair_dtype_hostile
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "shape or authority"
            ):
                transaction.begin_repair(
                    selected_ordinals=selected,
                    repair_delta_parts=(
                        _f64([[0.0, 0.0], [0.1, 0.0]]),
                        _f64([[0.2, 0.0], [0.3, 0.4]]),
                    ),
                    repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
                )
        finally:
            _tx._seal_caller_array = original_seal
        self.assertEqual(transaction.state, "CLOSED")

    def test_initial_delta_causality_is_bound(self) -> None:
        kwargs, partitions = _fixture()
        partitions[0].delta_pre[0, 0] = 0.25
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "initial delta.*causality"
        ):
            _tx.build_sealed_eliminated_target_transaction(**kwargs)

    def test_owner_objective_frame_is_checked_before_open(self) -> None:
        kwargs, _partitions = _fixture()
        kwargs["property_row"] = np.ascontiguousarray(
            kwargs["property_row"] * 1.0e-15
        )
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "owner numeric frame"
        ):
            _tx.build_sealed_eliminated_target_transaction(**kwargs)

    def test_repair_uses_immutable_kept_mapping_against_owner_aba(self) -> None:
        transaction, _kwargs, _partitions = _build()
        base_rows = transaction.base_rows
        original_ids = base_rows.row_ids.copy()
        original_seal = _tx._seal_caller_array

        def hostile(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            result = original_seal(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )
            if name == "selected ordinals":
                base_rows.row_ids.setflags(write=True)
                base_rows.row_ids[1] = 1
                base_rows.row_ids.setflags(write=False)
            return result

        _tx._seal_caller_array = hostile
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "omitted"
            ):
                transaction.begin_repair(
                    selected_ordinals=_i64([0, 1]),
                    repair_delta_parts=(
                        _f64([[0.0, 0.0], [0.1, 0.0]]),
                        _f64([[0.2, 0.0], [0.3, 0.4]]),
                    ),
                    repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
                )
        finally:
            _tx._seal_caller_array = original_seal
            base_rows.row_ids.setflags(write=True)
            base_rows.row_ids[:] = original_ids
            base_rows.row_ids.setflags(write=False)
        self.assertEqual(transaction.state, "CLOSED")

    def test_selected_validation_to_seal_hostile_is_rejected(self) -> None:
        transaction, _kwargs, _partitions = _build()
        selected = _i64([0, 2])
        original_seal = _tx._seal_caller_array

        def hostile(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            if name == "selected ordinals":
                selected[:] = [2, 0]
            return original_seal(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )

        _tx._seal_caller_array = hostile
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "selection frame"
            ):
                transaction.begin_repair(
                    selected_ordinals=selected,
                    repair_delta_parts=(
                        _f64([[0.0, 0.0], [0.1, 0.0]]),
                        _f64([[0.2, 0.0], [0.3, 0.4]]),
                    ),
                    repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
                )
        finally:
            _tx._seal_caller_array = original_seal
        self.assertEqual(transaction.state, "CLOSED")

    def test_entry_integrity_failure_closes_and_forbids_retry(self) -> None:
        transaction, _kwargs, _partitions = _build()
        base_rows = transaction.base_rows
        original_ids = base_rows.row_ids.copy()
        base_rows.row_ids.setflags(write=True)
        base_rows.row_ids[0] = 99
        base_rows.row_ids.setflags(write=False)
        try:
            with self.assertRaises(_owner.HighsOwnerUnknown):
                transaction.begin_repair(
                    selected_ordinals=_i64([0, 2]),
                    repair_delta_parts=(
                        _f64([[0.0, 0.0], [0.1, 0.0]]),
                        _f64([[0.2, 0.0], [0.3, 0.4]]),
                    ),
                    repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
                )
        finally:
            base_rows.row_ids.setflags(write=True)
            base_rows.row_ids[:] = original_ids
            base_rows.row_ids.setflags(write=False)
        self.assertEqual(transaction.state, "CLOSED")
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "not open"
        ):
            transaction.begin_repair(
                selected_ordinals=_i64([0, 2]),
                repair_delta_parts=(
                    _f64([[0.0, 0.0], [0.1, 0.0]]),
                    _f64([[0.2, 0.0], [0.3, 0.4]]),
                ),
                repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
            )

    def test_concurrent_repair_calls_cannot_both_get_active_leases(self) -> None:
        transaction, _kwargs, _partitions = _build()
        original_seal = _tx._seal_caller_array
        first_inside = threading.Event()
        release_first = threading.Event()
        second_attempted = threading.Event()
        hook_lock = threading.Lock()
        first_blocked = [False]

        def hostile(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            should_block = False
            if name == "selected ordinals":
                with hook_lock:
                    if not first_blocked[0]:
                        first_blocked[0] = True
                        should_block = True
            if should_block:
                first_inside.set()
                if not release_first.wait(5.0):
                    raise AssertionError("concurrency gate timed out")
            return original_seal(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )

        def invoke(*, second: bool):
            if second:
                second_attempted.set()
            try:
                return transaction.begin_repair(
                    selected_ordinals=_i64([0, 2]),
                    repair_delta_parts=(
                        _f64([[0.0, 0.0], [0.1, 0.0]]),
                        _f64([[0.2, 0.0], [0.3, 0.4]]),
                    ),
                    repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
                )
            except BaseException as exc:
                return exc

        _tx._seal_caller_array = hostile
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(invoke, second=False)
                self.assertTrue(first_inside.wait(5.0))
                second = executor.submit(invoke, second=True)
                self.assertTrue(second_attempted.wait(5.0))
                release_first.set()
                results = (first.result(timeout=5.0), second.result(timeout=5.0))
        finally:
            release_first.set()
            _tx._seal_caller_array = original_seal
        leases = [value for value in results if isinstance(value, _tx.RepairHandoff)]
        errors = [value for value in results if isinstance(value, BaseException)]
        self.assertEqual(len(leases), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(leases[0].state, "ACTIVE")
        leases[0].release()
        self.assertEqual(transaction.state, "CLOSED")

    def test_repair_preflight_counts_resident_plus_repair_authority(self) -> None:
        transaction, _kwargs, _partitions = _build()
        original_cap = _tx._MAX_AUTHORITY_BYTES
        resident = transaction.receipt.conservative_authority_bytes
        combined_with_one = _tx._preflight_repair_resources(
            resident_authority_bytes=1,
            phase_rows=transaction.receipt.phase_rows,
            output_rows=2,
            selected_width=2,
            partition_count=transaction.receipt.partition_count,
        )
        repair_only = combined_with_one - 1
        hostile_cap = max(resident, repair_only) + 1
        self.assertLess(resident, hostile_cap)
        self.assertLess(repair_only, hostile_cap)
        self.assertGreater(resident + repair_only, hostile_cap)
        _tx._MAX_AUTHORITY_BYTES = hostile_cap
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "repair authority"
            ):
                transaction.begin_repair(
                    selected_ordinals=_i64([0, 2]),
                    repair_delta_parts=(
                        _f64([[0.0, 0.0], [0.1, 0.0]]),
                        _f64([[0.2, 0.0], [0.3, 0.4]]),
                    ),
                    repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
                )
        finally:
            _tx._MAX_AUTHORITY_BYTES = original_cap
        self.assertEqual(transaction.state, "CLOSED")

    def test_repair_resource_frame_resists_public_receipt_aba(self) -> None:
        transaction, _kwargs, _partitions = _build()
        receipt = transaction.receipt
        original_phase_rows = receipt.phase_rows
        original_cap = _tx._MAX_AUTHORITY_BYTES
        original_require = _tx._require_array
        original_seal = _tx._seal_caller_array
        actual_combined = _tx._preflight_repair_resources(
            resident_authority_bytes=receipt.conservative_authority_bytes,
            phase_rows=original_phase_rows,
            output_rows=2,
            selected_width=2,
            partition_count=receipt.partition_count,
        )
        forged_combined = _tx._preflight_repair_resources(
            resident_authority_bytes=receipt.conservative_authority_bytes,
            phase_rows=1,
            output_rows=2,
            selected_width=2,
            partition_count=receipt.partition_count,
        )
        self.assertLess(forged_combined, actual_combined)
        _tx._MAX_AUTHORITY_BYTES = forged_combined

        def hostile_require(value, *, dtype, ndim, name):
            result = original_require(value, dtype=dtype, ndim=ndim, name=name)
            if name == "selected ordinals":
                object.__setattr__(receipt, "phase_rows", 1)
            return result

        def hostile_seal(
            value, *, name, expected_dtype, expected_ndim, expected_shape
        ):
            if name == "selected ordinals":
                object.__setattr__(receipt, "phase_rows", original_phase_rows)
            return original_seal(
                value,
                name=name,
                expected_dtype=expected_dtype,
                expected_ndim=expected_ndim,
                expected_shape=expected_shape,
            )

        _tx._require_array = hostile_require
        _tx._seal_caller_array = hostile_seal
        try:
            with self.assertRaisesRegex(
                _tx.SealedEliminatedTargetUnknown, "repair authority"
            ):
                transaction.begin_repair(
                    selected_ordinals=_i64([0, 2]),
                    repair_delta_parts=(
                        _f64([[0.0, 0.0], [0.1, 0.0]]),
                        _f64([[0.2, 0.0], [0.3, 0.4]]),
                    ),
                    repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
                )
        finally:
            object.__setattr__(receipt, "phase_rows", original_phase_rows)
            _tx._require_array = original_require
            _tx._seal_caller_array = original_seal
            _tx._MAX_AUTHORITY_BYTES = original_cap
        receipt.assert_intact()
        self.assertEqual(transaction.state, "CLOSED")

    def test_public_partition_report_cannot_aba_repair_authority(self) -> None:
        transaction, _kwargs, _partitions = _build()
        public_partition = transaction.partitions[0]
        internal_partition = transaction._partitions[0]
        self.assertIsNot(public_partition, internal_partition)
        self.assertIsNot(public_partition.row_ids, internal_partition.row_ids)
        self.assertIsNot(
            public_partition.logical_rows, internal_partition.logical_rows
        )
        self.assertIsNot(
            public_partition.logical_rows.data,
            internal_partition.logical_rows.data,
        )
        object.__setattr__(
            public_partition,
            "row_ids",
            _tx._seal_array(_i64([10, 11])),
        )
        forged_data = _tx._seal_array(public_partition.logical_rows.data.copy())
        forged_data.shape = (1, forged_data.size)
        object.__setattr__(public_partition.logical_rows, "data", forged_data)
        transaction.assert_intact()
        self.assertTrue(
            np.array_equal(transaction._partitions[0].row_ids, _i64([0, 1]))
        )
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "causality"
        ):
            transaction.begin_repair(
                selected_ordinals=_i64([0, 2]),
                repair_delta_parts=(
                    _f64([[1.0, 0.0], [0.0, 0.0]]),
                    _f64([[0.0, 0.0], [0.0, 0.0]]),
                ),
                repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
            )
        self.assertEqual(transaction.state, "CLOSED")

    def test_release_primary_exception_forces_closed_and_is_idempotent(self) -> None:
        transaction, _kwargs, _partitions = _build()
        lease = transaction.begin_repair(
            selected_ordinals=_i64([0, 2]),
            repair_delta_parts=(
                _f64([[0.0, 0.0], [0.1, 0.0]]),
                _f64([[0.2, 0.0], [0.3, 0.4]]),
            ),
            repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
        )
        original_release = transaction._repair_released

        class Sentinel(BaseException):
            pass

        primary = Sentinel("release interrupted before transaction callback")

        def hostile_release(_lease):
            raise primary

        transaction._repair_released = hostile_release
        try:
            try:
                lease.release()
            except BaseException as observed:
                self.assertIs(observed, primary)
            else:
                self.fail("hostile release unexpectedly succeeded")
        finally:
            transaction._repair_released = original_release
        self.assertEqual(lease.state, "RELEASED")
        self.assertEqual(transaction.state, "CLOSED")
        self.assertIsNone(transaction._lease)
        transaction.close()
        lease.release()
        self.assertEqual(transaction.state, "CLOSED")

    def test_concurrent_close_and_release_share_one_cleanup_barrier(self) -> None:
        transaction, _kwargs, _partitions = _build()
        lease = transaction.begin_repair(
            selected_ordinals=_i64([0, 2]),
            repair_delta_parts=(
                _f64([[0.0, 0.0], [0.1, 0.0]]),
                _f64([[0.2, 0.0], [0.3, 0.4]]),
            ),
            repair_delta_output=_f64([[0.5, -0.2], [-0.25, 0.75]]),
        )
        start = threading.Barrier(2)

        def close_transaction() -> None:
            start.wait(timeout=5.0)
            transaction.close()

        def release_lease() -> None:
            start.wait(timeout=5.0)
            lease.release()

        with ThreadPoolExecutor(max_workers=2) as executor:
            close_future = executor.submit(close_transaction)
            release_future = executor.submit(release_lease)
            close_future.result(timeout=5.0)
            release_future.result(timeout=5.0)
        self.assertEqual(lease.state, "RELEASED")
        self.assertEqual(transaction.state, "CLOSED")
        self.assertIsNone(transaction._lease)

    def test_receipt_metadata_is_content_bound(self) -> None:
        transaction, _kwargs, _partitions = _build()
        receipt = transaction.receipt
        receipt.assert_intact()
        object.__setattr__(receipt, "gpu_used", True)
        with self.assertRaisesRegex(
            _tx.SealedEliminatedTargetUnknown, "receipt changed"
        ):
            transaction.assert_intact()
        transaction.close()

    def test_close_releases_partition_authority_and_receipt_is_honest(self) -> None:
        transaction, _kwargs, _partitions = _build()
        partition = transaction._partitions[0]
        reference = weakref.ref(partition)
        receipt = transaction.receipt
        self.assertTrue(receipt.synthetic_only)
        self.assertFalse(receipt.retained_target_pre_container)
        self.assertFalse(receipt.retained_target_output)
        self.assertFalse(receipt.retained_global_logical_csr)
        self.assertTrue(receipt.retained_partitioned_full_logical_rows)
        self.assertTrue(receipt.retained_owner_screened_frozen_rows)
        self.assertFalse(receipt.retained_screened_scipy_wrapper)
        self.assertFalse(receipt.base_auxiliary_normal_form)
        self.assertTrue(receipt.owner_lifecycle_externally_unproven)
        self.assertTrue(receipt.production_repair_global_csr_blocker)
        self.assertFalse(receipt.gpu_used)
        self.assertFalse(receipt.benchmark_run)
        receipt.assert_intact()
        transaction.close()
        del partition
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(transaction.state, "CLOSED")


if __name__ == "__main__":
    unittest.main()
