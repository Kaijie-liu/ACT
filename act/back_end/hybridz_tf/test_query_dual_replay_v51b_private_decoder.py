"""Exact-ABI and semantic gates for the private V5.1b result decoder."""

from __future__ import annotations

import copy
import ctypes
import dis
import gc
import hashlib
import math
import os
import pickle
import platform
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
import unittest
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from unittest import mock

import numpy as np
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as kernel,
)


_KERNEL_SHA256 = (
    "6f72ee6f1f301c9818dcc2ab754346a4cdb08ff072b1b3f18fb3557c55228329"
)
_DENSE_RAW_TAG = b"act.v51b.private.dense-result.v1"
_CONV_RAW_TAG = b"act.v51b.private.conv-result.v1"
_DENSE_DECODED_TAG = b"act.v51b.private.decoded-dense-result.v1"
_CONV_DECODED_TAG = b"act.v51b.private.decoded-conv-result.v1"
_F64_TAG = np.dtype(np.float64).str.encode("ascii")
_BOOL_TAG = np.dtype(np.bool_).str.encode("ascii")


def _end(seconds: float = 120.0) -> float:
    return float(time.monotonic() + seconds)


def _run_isolated(source: str) -> None:
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(decoder.__file__), "../../..")
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if completed.returncode:
        raise AssertionError(
            "isolated probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _implementation_spec(factory, expected_name: str) -> tuple:
    for cell in factory.__closure__ or ():
        value = cell.cell_contents
        if (
            type(value) is tuple
            and len(value) == 7
            and type(value[0]) is types.CodeType
            and value[1] == expected_name
        ):
            return value
    raise AssertionError("immutable implementation spec was not captured")


def _bytes_backed(value: np.ndarray) -> bool:
    if type(value) is not np.ndarray or value.flags.writeable:
        return False
    current = value
    depth = 0
    while type(current) is np.ndarray:
        if current.flags.writeable or depth > 4:
            return False
        current = current.base
        depth += 1
    return type(current) is bytes


def _frame_array(result: tuple, index: int) -> np.ndarray:
    payload, shape, tag = result[index]
    dtype = np.bool_ if tag == _BOOL_TAG else np.float64
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def _replace_item(result: tuple, index: int, value) -> tuple:
    items = list(result)
    items[index] = value
    return tuple(items)


def _replace_frame_array(
    result: tuple, index: int, value: np.ndarray
) -> tuple:
    frame = result[index]
    replacement = (
        np.ascontiguousarray(value).tobytes(order="C"),
        frame[1],
        frame[2],
    )
    return _replace_item(result, index, replacement)


def _mutate_frame(
    result: tuple, index: int, mutator
) -> tuple:
    value = _frame_array(result, index).copy()
    mutator(value)
    return _replace_frame_array(result, index, value)


def _make_dense_raw() -> tuple:
    weight = np.ascontiguousarray(
        np.asarray(
            [[-0.2, 0.5], [1.0, -0.25]], dtype=np.float64
        )
    )
    max_abs = np.ascontiguousarray(
        np.asarray([5.0, 2.0], dtype=np.float64)
    )
    coefficient = np.ascontiguousarray(
        np.asarray(
            [[-0.75, 0.25], [-0.0, 0.0], [1.0, -2.0]],
            dtype=np.float64,
        )
    )
    port = kernel.create_private_numeric_kernel(deadline=_end())
    locator = port.admit_dense(
        weight=weight,
        predecessor_max_abs=max_abs,
        tile_width=1,
    )
    return port.execute_dense(locator, coefficient)


def _make_fraction_dense_raw() -> tuple:
    port = kernel.create_private_numeric_kernel(deadline=_end())
    locator = port.admit_dense(
        weight=np.asarray([[-0.2]], dtype=np.float64),
        predecessor_max_abs=np.asarray([5.0], dtype=np.float64),
        tile_width=1,
    )
    return port.execute_dense(
        locator, np.asarray([[-0.75]], dtype=np.float64)
    )


def _make_fallback_dense_raw() -> tuple:
    eta = np.nextafter(np.float64(0.0), np.float64(math.inf))
    weight = np.ascontiguousarray(
        np.asarray([[eta, 0.0], [0.0, 1.0]], dtype=np.float64)
    )
    max_abs = np.ascontiguousarray(
        np.asarray([1.0, 0.0], dtype=np.float64)
    )
    coefficient = np.ascontiguousarray(
        np.asarray(
            [[eta, 0.0], [0.0, 1.0], [-0.0, 0.0]],
            dtype=np.float64,
        )
    )
    port = kernel.create_private_numeric_kernel(deadline=_end())
    locator = port.admit_dense(
        weight=weight,
        predecessor_max_abs=max_abs,
        tile_width=1,
    )
    return port.execute_dense(locator, coefficient)


def _make_conv_raw() -> tuple:
    weight = np.ascontiguousarray(
        np.asarray(
            [
                [[[0.1, -0.25], [0.5, 1.0]]],
                [[[-0.2, 0.75], [0.125, -0.5]]],
            ],
            dtype=np.float64,
        )
    )
    coefficient = np.ascontiguousarray(
        np.vstack(
            (
                np.copysign(np.zeros(8, dtype=np.float64), -1.0),
                np.linspace(-1.0, 1.0, 8, dtype=np.float64),
                np.ones(8, dtype=np.float64),
            )
        )
    )
    port = kernel.create_private_numeric_kernel(deadline=_end())
    locator = port.admit_conv(
        layer_id=2,
        weight=weight,
        predecessor_lb=-np.ones(9, dtype=np.float64),
        predecessor_ub=np.ones(9, dtype=np.float64),
        input_shape=(1, 3, 3),
        output_shape=(2, 2, 2),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    return port.execute_conv(locator, coefficient)


class PrivateDecoderAcceptedValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dense_raw = _make_dense_raw()
        cls.fallback_raw = _make_fallback_dense_raw()
        cls.conv_raw = _make_conv_raw()

    def test_frozen_kernel_source_digest(self):
        with open(kernel.__file__, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        self.assertEqual(digest, _KERNEL_SHA256)

    def test_dense_direct_return_decodes_to_exact_readonly_tuple(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        actual = port.decode_dense(
            self.dense_raw, expected_rows=3, expected_width=2
        )
        self.assertIs(type(actual), tuple)
        self.assertEqual(actual[0], _DENSE_DECODED_TAG)
        self.assertIs(actual[1], False)
        self.assertFalse(port.proof_authority)
        self.assertEqual(port.schema, decoder.SCHEMA)
        for source_frame, value in zip(self.dense_raw[2:], actual[2:]):
            self.assertIs(type(value), np.ndarray)
            self.assertTrue(_bytes_backed(value))
            self.assertEqual(
                value.tobytes(order="C"), source_frame[0]
            )
            with self.assertRaises(ValueError):
                value.setflags(write=True)
        self.assertEqual(actual[2].shape, (3, 2))
        self.assertEqual(
            [value.shape for value in actual[3:]], [(3,)] * 6
        )

    def test_dense_fallback_and_positive_zero_are_preserved_bitwise(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        actual = port.decode_dense(
            self.fallback_raw, expected_rows=3, expected_width=2
        )
        self.assertTrue(actual[8][0])
        self.assertEqual(
            actual[6].view(np.uint64).tolist(),
            _frame_array(self.fallback_raw, 6)
            .view(np.uint64)
            .tolist(),
        )
        for index in (3, 4, 5, 6):
            self.assertEqual(int(actual[index].view(np.uint64)[2]), 0)

    def test_conv_direct_return_recomputes_directed_zero_sum(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        actual = port.decode_conv(
            self.conv_raw, expected_rows=3, expected_width=9
        )
        self.assertIs(type(actual), tuple)
        self.assertEqual(actual[0], _CONV_DECODED_TAG)
        self.assertIs(actual[1], False)
        self.assertEqual(actual[2].shape, (3, 9))
        self.assertTrue(np.all(actual[7][1:]))
        self.assertTrue(np.all(actual[8][1:]))
        for value in actual[2:]:
            self.assertTrue(_bytes_backed(value))
        self.assertEqual(
            actual[3].view(np.uint64).tolist(),
            _frame_array(self.conv_raw, 3).view(np.uint64).tolist(),
        )
        for index in (3, 4, 5):
            self.assertEqual(int(actual[index].view(np.uint64)[0]), 0)

    def test_decoded_values_are_never_accepted_as_input(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        dense = port.decode_dense(
            self.dense_raw, expected_rows=3, expected_width=2
        )
        conv = port.decode_conv(
            self.conv_raw, expected_rows=3, expected_width=9
        )
        for value, method, rows, width in (
            (dense, port.decode_dense, 3, 2),
            (conv, port.decode_conv, 3, 9),
            (copy.deepcopy(dense), port.decode_dense, 3, 2),
            (
                pickle.loads(pickle.dumps(conv)),
                port.decode_conv,
                3,
                9,
            ),
        ):
            with self.subTest(tag=value[0]):
                with self.assertRaisesRegex(
                    decoder.PrivateNumericDecoderError,
                    "INVALID_ENVELOPE",
                ):
                    method(
                        value,
                        expected_rows=rows,
                        expected_width=width,
                    )


class PrivateDecoderMalformedDenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = _make_dense_raw()
        cls.fraction_raw = _make_fraction_dense_raw()

    def _reject(self, value, code: str = "") -> None:
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        context = self.assertRaises(decoder.PrivateNumericDecoderError)
        with context:
            port.decode_dense(
                value, expected_rows=3, expected_width=2
            )
        if code:
            self.assertEqual(context.exception.code, code)

    def test_envelope_subclasses_authority_length_and_tag_reject(self):
        class TupleSubclass(tuple):
            pass

        class BytesSubclass(bytes):
            pass

        cases = (
            TupleSubclass(self.raw),
            _replace_item(self.raw, 0, BytesSubclass(self.raw[0])),
            _replace_item(self.raw, 0, b"wrong"),
            _replace_item(self.raw, 1, np.bool_(False)),
            self.raw[:-1],
            self.raw + ((),),
        )
        for value in cases:
            with self.subTest(kind=type(value)):
                self._reject(value, "INVALID_ENVELOPE")

    def test_frame_subclasses_shape_tag_and_payload_reject(self):
        class TupleSubclass(tuple):
            pass

        class BytesSubclass(bytes):
            pass

        class IntSubclass(int):
            pass

        frame = self.raw[2]
        shape_subclass = (IntSubclass(3), 2)
        cases = (
            _replace_item(self.raw, 2, TupleSubclass(frame)),
            _replace_item(
                self.raw,
                2,
                (BytesSubclass(frame[0]), frame[1], frame[2]),
            ),
            _replace_item(
                self.raw, 2, (frame[0], shape_subclass, frame[2])
            ),
            _replace_item(
                self.raw,
                2,
                (frame[0], frame[1], BytesSubclass(frame[2])),
            ),
            _replace_item(
                self.raw, 2, (frame[0], frame[1], _BOOL_TAG)
            ),
            _replace_item(
                self.raw, 2, (frame[0][:-1], frame[1], frame[2])
            ),
            _replace_item(
                self.raw, 2, (frame[0], (3, 1), frame[2])
            ),
            _replace_item(self.raw, 2, frame[:2]),
        )
        for value in cases:
            with self.subTest(frame=value[2]):
                self._reject(value, "INVALID_FRAME")

    def test_shape_element_is_rejected_before_foreign_equality(self):
        calls = [0]

        class Probe:
            def __eq__(self, other):
                del other
                calls[0] += 1
                raise AssertionError("foreign equality was called")

            def __le__(self, other):
                del other
                calls[0] += 1
                raise AssertionError("foreign ordering was called")

        frame = self.raw[2]
        value = _replace_item(
            self.raw,
            2,
            (frame[0], (Probe(), 2), frame[2]),
        )
        self._reject(value, "INVALID_FRAME")
        self.assertEqual(calls[0], 0)

    def test_expected_shape_is_exact_and_session_bound(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        for rows, width in (
            (True, 2),
            (np.int64(3), 2),
            (3, np.int64(2)),
            (0, 2),
            (3, 0),
        ):
            with self.subTest(rows=rows, width=width):
                with self.assertRaisesRegex(
                    decoder.PrivateNumericDecoderError,
                    "INVALID_EXPECTATION",
                ):
                    port.decode_dense(
                        self.raw,
                        expected_rows=rows,
                        expected_width=width,
                    )
        for rows, width in ((2, 2), (3, 3)):
            with self.subTest(rows=rows, width=width):
                with self.assertRaisesRegex(
                    decoder.PrivateNumericDecoderError,
                    "INVALID_FRAME",
                ):
                    port.decode_dense(
                        self.raw,
                        expected_rows=rows,
                        expected_width=width,
                    )

    def test_nonfinite_and_negative_fields_reject(self):
        cases = (
            _mutate_frame(
                self.raw, 2, lambda value: value.__setitem__((0, 0), np.nan)
            ),
            _mutate_frame(
                self.raw, 3, lambda value: value.__setitem__(0, np.inf)
            ),
            _mutate_frame(
                self.raw, 4, lambda value: value.__setitem__(0, -1.0)
            ),
            _mutate_frame(
                self.raw, 5, lambda value: value.__setitem__(0, np.nan)
            ),
            _mutate_frame(
                self.raw, 6, lambda value: value.__setitem__(0, -1.0)
            ),
        )
        for value in cases:
            with self.subTest():
                self._reject(value)

    def test_dense_masks_inactive_zero_and_final_bits_reject(self):
        active = _frame_array(self.raw, 7).copy()
        active[0] = ~active[0]
        fallback = _frame_array(self.raw, 8).copy()
        fallback[1] = True
        signed_zero = _frame_array(self.raw, 4).copy()
        signed_zero[1] = -0.0
        final = _frame_array(self.raw, 6).copy()
        final[0] = np.nextafter(final[0], np.float64(math.inf))
        cases = (
            _replace_frame_array(self.raw, 7, active),
            _replace_frame_array(self.raw, 8, fallback),
            _replace_frame_array(self.raw, 4, signed_zero),
            _replace_frame_array(self.raw, 6, final),
        )
        for value in cases:
            with self.subTest():
                self._reject(value, "SEMANTIC_MISMATCH")

    def test_dense_active_row_cannot_drop_all_guards_to_zero(self):
        for zero in (0.0, -0.0):
            value = self.raw
            for index in (4, 5, 6):
                guard = _frame_array(value, index).copy()
                guard[0] = zero
                value = _replace_frame_array(value, index, guard)
            with self.subTest(zero_bits=np.float64(zero).view(np.uint64)):
                self._reject(value, "SEMANTIC_MISMATCH")

    def test_dense_nonfallback_streamed_guard_is_bitwise_wide_guard(self):
        fallback = _frame_array(self.raw, 8)
        row = int(np.flatnonzero(~fallback)[0])
        streamed = _frame_array(self.raw, 5).copy()
        streamed[row] = np.nextafter(
            streamed[row], np.float64(math.inf)
        )
        self._reject(
            _replace_frame_array(self.raw, 5, streamed),
            "SEMANTIC_MISMATCH",
        )

    def test_fraction_counterexample_is_rejected(self):
        exact = (
            Fraction.from_float(-0.75)
            * Fraction.from_float(-0.2)
        )
        nominal = Fraction.from_float(
            float(_frame_array(self.fraction_raw, 2)[0, 0])
        )
        required = abs(exact - nominal) * Fraction(5)
        self.assertGreater(required, 0)

        forged = self.fraction_raw
        for index in (4, 5, 6):
            forged = _replace_frame_array(
                forged, index, np.zeros(1, dtype=np.float64)
            )
        forged = _replace_frame_array(
            forged, 7, np.zeros(1, dtype=np.bool_)
        )
        forged = _replace_frame_array(
            forged, 8, np.zeros(1, dtype=np.bool_)
        )
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        with self.assertRaisesRegex(
            decoder.PrivateNumericDecoderError,
            "SEMANTIC_MISMATCH",
        ):
            port.decode_dense(
                forged, expected_rows=1, expected_width=1
            )


class PrivateDecoderMalformedConvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = _make_conv_raw()

    def _reject(self, value, code: str = "") -> None:
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        context = self.assertRaises(decoder.PrivateNumericDecoderError)
        with context:
            port.decode_conv(
                value, expected_rows=3, expected_width=9
            )
        if code:
            self.assertEqual(context.exception.code, code)

    def test_conv_nonfinite_negative_and_boolean_payload_reject(self):
        boolean_frame = self.raw[6]
        invalid_boolean = (
            bytes([2]) + boolean_frame[0][1:],
            boolean_frame[1],
            boolean_frame[2],
        )
        cases = (
            _mutate_frame(
                self.raw, 2, lambda value: value.__setitem__((0, 0), np.nan)
            ),
            _mutate_frame(
                self.raw, 3, lambda value: value.__setitem__(1, np.inf)
            ),
            _mutate_frame(
                self.raw, 4, lambda value: value.__setitem__(1, -1.0)
            ),
            _mutate_frame(
                self.raw, 5, lambda value: value.__setitem__(1, np.nan)
            ),
            _replace_item(self.raw, 6, invalid_boolean),
        )
        for value in cases:
            with self.subTest():
                self._reject(value)

    def test_conv_component_masks_union_and_positive_zero_reject(self):
        channel_mask = _frame_array(self.raw, 7).copy()
        channel_mask[1] = False
        accumulation_mask = _frame_array(self.raw, 8).copy()
        accumulation_mask[1] = False
        active = _frame_array(self.raw, 6).copy()
        active[1] = False
        signed_channel_zero = _frame_array(self.raw, 4).copy()
        signed_channel_zero[0] = -0.0
        signed_scalar_zero = _frame_array(self.raw, 3).copy()
        signed_scalar_zero[0] = -0.0
        cases = (
            _replace_frame_array(self.raw, 7, channel_mask),
            _replace_frame_array(self.raw, 8, accumulation_mask),
            _replace_frame_array(self.raw, 6, active),
            _replace_frame_array(self.raw, 4, signed_channel_zero),
            _replace_frame_array(self.raw, 3, signed_scalar_zero),
        )
        for value in cases:
            with self.subTest():
                self._reject(value, "SEMANTIC_MISMATCH")

    def test_conv_scalar_bit_change_rejects(self):
        scalar = _frame_array(self.raw, 3).copy()
        scalar[1] = np.nextafter(
            scalar[1], np.float64(math.inf)
        )
        self._reject(
            _replace_frame_array(self.raw, 3, scalar),
            "SEMANTIC_MISMATCH",
        )

    def test_conv_wrong_tag_shape_and_decoded_roundtrip_reject(self):
        wrong_tag = _replace_item(self.raw, 0, _DENSE_RAW_TAG)
        frame = self.raw[2]
        wrong_shape = _replace_item(
            self.raw, 2, (frame[0], (1, 27), frame[2])
        )
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        decoded = port.decode_conv(
            self.raw, expected_rows=3, expected_width=9
        )
        for value in (wrong_tag, wrong_shape, decoded):
            with self.subTest(tag=value[0]):
                self._reject(value)


class PrivateDecoderLifecycleAndDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dense_raw = _make_dense_raw()
        cls.conv_raw = _make_conv_raw()

    def test_port_copy_deepcopy_pickle_and_construction_reject(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        for operation in (
            copy.copy,
            copy.deepcopy,
            lambda value: pickle.dumps(value),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    decoder.PrivateNumericDecoderError,
                    "COPY_FORBIDDEN",
                ):
                    operation(port)
        with self.assertRaisesRegex(
            decoder.PrivateNumericDecoderError,
            "PORT_CONSTRUCTION",
        ):
            type(port)()

    def test_gc_keeps_output_storage_alive_but_not_port(self):
        raw = _make_dense_raw()
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        port_ref = weakref.ref(port)
        output = port.decode_dense(
            raw, expected_rows=3, expected_width=2
        )
        array_ref = weakref.ref(output[2])
        expected = output[2].tobytes(order="C")
        del raw
        gc.collect()
        self.assertEqual(output[2].tobytes(order="C"), expected)
        self.assertTrue(_bytes_backed(output[2]))
        del port
        gc.collect()
        self.assertIsNone(port_ref())
        self.assertIsNotNone(array_ref())
        del output
        gc.collect()
        self.assertIsNone(array_ref())

    def test_invalid_expired_and_closed_deadlines_fail_closed(self):
        for value in (
            1,
            np.float64(_end()),
            math.inf,
            math.nan,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    decoder.PrivateNumericDecoderError,
                    "INVALID_DEADLINE",
                ):
                    decoder.create_private_numeric_result_decoder(
                        deadline=value
                    )
        with self.assertRaisesRegex(
            decoder.PrivateNumericDecoderError,
            "DEADLINE_EXPIRED",
        ):
            decoder.create_private_numeric_result_decoder(
                deadline=float(time.monotonic() - 1.0)
            )
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        port.close()
        port.close()
        self.assertIs(port.proof_authority, False)
        self.assertEqual(port.schema, decoder.SCHEMA)
        with self.assertRaisesRegex(
            decoder.PrivateNumericDecoderError, "CLOSED"
        ):
            port.decode_dense(
                self.dense_raw, expected_rows=3, expected_width=2
            )
        expiring = decoder.create_private_numeric_result_decoder(
            deadline=float(time.monotonic() + 0.05)
        )
        time.sleep(0.06)
        expiring.close()
        expiring.close()

    @unittest.skipUnless(hasattr(os, "fork"), "fork is unavailable")
    def test_port_rejects_forked_process_before_lock(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                try:
                    port.decode_dense(
                        self.dense_raw,
                        expected_rows=3,
                        expected_width=2,
                    )
                except decoder.PrivateNumericDecoderError as exc:
                    payload = exc.code.encode("ascii")
                else:
                    payload = b"ACCEPTED"
                os.write(write_fd, payload)
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        try:
            payload = os.read(read_fd, 128)
        finally:
            os.close(read_fd)
        waited, status = os.waitpid(pid, 0)
        self.assertEqual(waited, pid)
        self.assertEqual(status, 0)
        self.assertEqual(payload, b"FORKED_PROCESS")

    @unittest.skipUnless(hasattr(os, "fork"), "fork is unavailable")
    def test_fork_pid_check_precedes_an_inherited_held_lock(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        method = type(port).decode_dense
        pending = [method]
        operation_lock = None
        seen = set()
        while pending and operation_lock is None:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            closure = {
                name: cell.cell_contents
                for name, cell in zip(
                    current.__code__.co_freevars,
                    current.__closure__ or (),
                )
            }
            operation_lock = closure.get("_operation_lock")
            pending.extend(
                value
                for value in closure.values()
                if type(value) is types.FunctionType
            )
        self.assertIsNotNone(operation_lock)
        locked = threading.Event()
        release = threading.Event()

        def hold_lock():
            with operation_lock:
                locked.set()
                release.wait(10.0)

        worker = threading.Thread(target=hold_lock)
        worker.start()
        self.assertTrue(locked.wait(2.0))
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                try:
                    port.decode_dense(
                        self.dense_raw,
                        expected_rows=3,
                        expected_width=2,
                    )
                except decoder.PrivateNumericDecoderError as exc:
                    payload = exc.code.encode("ascii")
                else:
                    payload = b"ACCEPTED"
                os.write(write_fd, payload)
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        try:
            ready, _, _ = select.select([read_fd], [], [], 3.0)
            if not ready:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                self.fail("forked decoder waited on the inherited lock")
            payload = os.read(read_fd, 128)
            waited, status = os.waitpid(pid, 0)
            self.assertEqual(waited, pid)
            self.assertEqual(status, 0)
            self.assertEqual(payload, b"FORKED_PROCESS")
        finally:
            os.close(read_fd)
            release.set()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())

    def test_fixed_four_thread_concurrent_decodes_are_bit_identical(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )

        def run(index: int):
            if index % 2:
                result = port.decode_dense(
                    self.dense_raw,
                    expected_rows=3,
                    expected_width=2,
                )
            else:
                result = port.decode_conv(
                    self.conv_raw,
                    expected_rows=3,
                    expected_width=9,
                )
            return (
                result[0],
                tuple(value.tobytes(order="C") for value in result[2:]),
                tuple(value.flags.writeable for value in result[2:]),
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(run, range(64)))
        dense = outputs[1]
        conv = outputs[0]
        for index, value in enumerate(outputs):
            expected = dense if index % 2 else conv
            self.assertEqual(value, expected)
            self.assertFalse(any(value[2]))

    def test_public_methods_use_closure_private_port_check(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        port_type = type(port)
        original = port_type._check_self
        calls = [0]

        def changed(*args, **kwargs):
            del args, kwargs
            calls[0] += 1
            raise AssertionError("dynamic port check was called")

        type.__setattr__(port_type, "_check_self", changed)
        try:
            dense = port.decode_dense(
                self.dense_raw,
                expected_rows=3,
                expected_width=2,
            )
            conv = port.decode_conv(
                self.conv_raw,
                expected_rows=3,
                expected_width=9,
            )
        finally:
            type.__setattr__(port_type, "_check_self", original)
        self.assertEqual(calls[0], 0)
        self.assertEqual(dense[0], _DENSE_DECODED_TAG)
        self.assertEqual(conv[0], _CONV_DECODED_TAG)

    def test_captured_decode_ignores_port_method_replacement(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        captured = port.decode_dense
        port_type = type(port)
        original = port_type.decode_dense
        calls = [0]

        def changed(*args, **kwargs):
            del args, kwargs
            calls[0] += 1
            return (b"changed",)

        type.__setattr__(port_type, "decode_dense", changed)
        try:
            result = captured(
                self.dense_raw,
                expected_rows=3,
                expected_width=2,
            )
        finally:
            type.__setattr__(port_type, "decode_dense", original)
        self.assertEqual(calls[0], 0)
        self.assertEqual(result[0], _DENSE_DECODED_TAG)

    def test_forged_port_cannot_close_factory_state(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        forged = object.__new__(type(port))
        with self.assertRaisesRegex(
            decoder.PrivateNumericDecoderError, "PORT_MISMATCH"
        ):
            type(port).close(forged)
        result = port.decode_dense(
            self.dense_raw,
            expected_rows=3,
            expected_width=2,
        )
        self.assertEqual(result[0], _DENSE_DECODED_TAG)

    def test_factory_rejects_persistent_dependency_change_before_call(self):
        calls = [0]

        def changed(*args, **kwargs):
            del args, kwargs
            calls[0] += 1
            raise AssertionError("changed dependency was called")

        with mock.patch.object(np, "nextafter", changed):
            with self.assertRaisesRegex(
                decoder.PrivateNumericDecoderError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                decoder.create_private_numeric_result_decoder(
                    deadline=_end()
                )
        self.assertEqual(calls[0], 0)

    def test_numeric_policy_is_restored_and_overflow_is_normalized(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        baseline_dense = port.decode_dense(
            self.dense_raw, expected_rows=3, expected_width=2
        )
        baseline_conv = port.decode_conv(
            self.conv_raw, expected_rows=3, expected_width=9
        )
        callback_calls = [0]

        def callback(*args):
            del args
            callback_calls[0] += 1
            raise AssertionError("ambient NumPy callback was invoked")

        def snapshot():
            return (np.geterr().copy(), np.geterrcall(), np.getbufsize())

        original = snapshot()
        try:
            for policy in ("ignore", "raise", "warn", "call"):
                if policy == "call":
                    np.seterrcall(callback)
                    np.seterr(all="call")
                else:
                    np.seterr(all=policy)
                before = snapshot()
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    dense = port.decode_dense(
                        self.dense_raw,
                        expected_rows=3,
                        expected_width=2,
                    )
                    conv = port.decode_conv(
                        self.conv_raw,
                        expected_rows=3,
                        expected_width=9,
                    )
                self.assertEqual(
                    tuple(value.tobytes() for value in dense[2:]),
                    tuple(value.tobytes() for value in baseline_dense[2:]),
                )
                self.assertEqual(
                    tuple(value.tobytes() for value in conv[2:]),
                    tuple(value.tobytes() for value in baseline_conv[2:]),
                )
                self.assertEqual(snapshot(), before)

            maximum = np.finfo(np.float64).max
            malformed = self.conv_raw
            for index in (3, 4, 5):
                malformed = _replace_frame_array(
                    malformed,
                    index,
                    np.full(
                        _frame_array(malformed, index).shape,
                        maximum,
                        dtype=np.float64,
                    ),
                )
            for index in (6, 7, 8):
                malformed = _replace_frame_array(
                    malformed,
                    index,
                    np.ones(
                        _frame_array(malformed, index).shape,
                        dtype=np.bool_,
                    ),
                )
            for policy in ("ignore", "warn", "raise"):
                np.seterr(all=policy)
                before = snapshot()
                with self.assertRaises(
                    decoder.PrivateNumericDecoderError
                ) as captured:
                    port.decode_conv(
                        malformed,
                        expected_rows=3,
                        expected_width=9,
                    )
                self.assertEqual(captured.exception.code, "NONFINITE")
                self.assertIs(
                    type(captured.exception.__cause__),
                    FloatingPointError,
                )
                self.assertEqual(snapshot(), before)
        finally:
            np.setbufsize(original[2])
            np.seterrcall(original[1])
            np.seterr(**original[0])
        self.assertEqual(callback_calls[0], 0)

    def test_mutable_python_instrumentation_is_rejected(self):
        _run_isolated(
            r"""
import sys
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as kernel,
)

kport = kernel.create_private_numeric_kernel(
    deadline=time.monotonic() + 30.0
)
locator = kport.admit_dense(
    weight=np.asarray([[-0.2]], dtype=np.float64),
    predecessor_max_abs=np.asarray([5.0], dtype=np.float64),
)
raw = kport.execute_dense(
    locator, np.asarray([[-0.75]], dtype=np.float64)
)
dport = decoder.create_private_numeric_result_decoder(
    deadline=time.monotonic() + 30.0
)

def trace(frame, event, argument):
    del frame, event, argument
    return trace

def profile(frame, event, argument):
    del frame, event, argument

def expect_rejected():
    operations = (
        lambda: decoder.create_private_numeric_result_decoder(
            deadline=time.monotonic() + 30.0
        ),
        lambda: dport.decode_dense(
            raw, expected_rows=1, expected_width=1
        ),
    )
    for operation in operations:
        try:
            operation()
        except decoder.PrivateNumericDecoderError as exc:
            if exc.code != "NUMERIC_PLATFORM":
                raise AssertionError(("instrumentation code", exc.code))
        else:
            raise AssertionError("decoder instrumentation was accepted")

sys.settrace(trace)
try:
    expect_rejected()
finally:
    sys.settrace(None)

sys.setprofile(profile)
try:
    expect_rejected()
finally:
    sys.setprofile(None)

tool_id = 5
sys.monitoring.use_tool_id(tool_id, "act-private-decoder-test")
try:
    expect_rejected()
finally:
    sys.monitoring.free_tool_id(tool_id)

decoded = dport.decode_dense(
    raw, expected_rows=1, expected_width=1
)
if decoded[0] != b"act.v51b.private.decoded-dense-result.v1":
    raise AssertionError("decoder did not recover after instrumentation")
"""
        )

    @unittest.skipUnless(
        platform.machine().lower() in {"x86_64", "amd64"}
        and shutil.which("gcc") is not None,
        "native floating-environment probe requires x86-64 and gcc",
    )
    def test_factory_and_decode_reject_complete_hardware_control_matrix(self):
        _run_isolated(
            r"""
import ctypes
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as kernel,
)

source = r'''
#include <fenv.h>
#include <xmmintrin.h>
#include <fpu_control.h>
unsigned act_get_mxcsr(void) { return _mm_getcsr(); }
void act_set_mxcsr(unsigned value) { _mm_setcsr(value); }
unsigned short act_get_x87(void) {
    fpu_control_t value;
    _FPU_GETCW(value);
    return value;
}
void act_set_x87(unsigned short value) {
    fpu_control_t current = value;
    _FPU_SETCW(current);
}
void act_clear_flags(void) {
    __asm__ __volatile__("fnclex");
    _mm_setcsr(_mm_getcsr() & ~0x3fu);
}
unsigned short act_get_x87_status(void) {
    unsigned short value;
    __asm__ __volatile__("fnstsw %0" : "=am"(value));
    return value;
}
void act_set_x87_pending_unmasked(unsigned bit) {
    fenv_t environment;
    unsigned char *bytes = (unsigned char *)&environment;
    fpu_control_t control;
    fegetenv(&environment);
    bytes[0] |= 0x3fu;
    bytes[4] = (unsigned char)(bit | 0x80u);
    bytes[5] &= 0x7fu;
    fesetenv(&environment);
    _FPU_GETCW(control);
    control &= (fpu_control_t)~bit;
    _FPU_SETCW(control);
}
void act_set_mxcsr_pending_unmasked(unsigned bit) {
    unsigned value = _mm_getcsr();
    value |= 0x1f80u;
    value &= ~(bit << 7);
    value |= bit;
    _mm_setcsr(value);
}
'''
with tempfile.TemporaryDirectory(prefix="act_decoder_fenv_") as directory:
    library_path = pathlib.Path(directory) / "libact_fenv.so"
    subprocess.run(
        [
            "gcc", "-shared", "-fPIC", "-O2", "-x", "c",
            "-o", str(library_path), "-", "-lm",
        ],
        input=source,
        text=True,
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.act_get_mxcsr.restype = ctypes.c_uint
    library.act_set_mxcsr.argtypes = (ctypes.c_uint,)
    library.act_get_x87.restype = ctypes.c_ushort
    library.act_set_x87.argtypes = (ctypes.c_ushort,)
    library.act_clear_flags.argtypes = ()
    library.act_get_x87_status.restype = ctypes.c_ushort
    library.act_set_x87_pending_unmasked.argtypes = (
        ctypes.c_uint,
    )
    library.act_set_mxcsr_pending_unmasked.argtypes = (
        ctypes.c_uint,
    )

    deadline = time.monotonic() + 30.0
    kport = kernel.create_private_numeric_kernel(deadline=deadline)
    dense_locator = kport.admit_dense(
        weight=np.asarray([[-0.2]], dtype=np.float64),
        predecessor_max_abs=np.asarray([5.0], dtype=np.float64),
    )
    dense_raw = kport.execute_dense(
        dense_locator, np.asarray([[-0.75]], dtype=np.float64)
    )
    conv_locator = kport.admit_conv(
        layer_id=0,
        weight=np.asarray([[[[0.25]]]], dtype=np.float64),
        predecessor_lb=np.asarray([-1.0], dtype=np.float64),
        predecessor_ub=np.asarray([1.0], dtype=np.float64),
        input_shape=(1, 1, 1),
        output_shape=(1, 1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    conv_raw = kport.execute_conv(
        conv_locator, np.asarray([[0.5]], dtype=np.float64)
    )
    dport = decoder.create_private_numeric_result_decoder(
        deadline=deadline
    )
    baseline_dense = dport.decode_dense(
        dense_raw, expected_rows=1, expected_width=1
    )
    baseline_conv = dport.decode_conv(
        conv_raw, expected_rows=1, expected_width=1
    )

    old_mxcsr = library.act_get_mxcsr()
    old_x87 = library.act_get_x87()
    modes = (
        ("MXCSR_FTZ", "mx_mode", 0x8000),
        ("MXCSR_DAZ", "mx_mode", 0x0040),
        ("MXCSR_ROUND_DOWN", "mx_mode", 0x2000),
        ("MXCSR_ROUND_UP", "mx_mode", 0x4000),
        ("MXCSR_ROUND_ZERO", "mx_mode", 0x6000),
        ("X87_ROUND_DOWN", "x87_mode", 0x0400),
        ("X87_ROUND_UP", "x87_mode", 0x0800),
        ("X87_ROUND_ZERO", "x87_mode", 0x0C00),
        ("X87_PRECISION_24", "x87_precision", 0x0000),
        ("X87_PRECISION_53", "x87_precision", 0x0200),
    )
    modes += tuple(
        (f"MXCSR_UNMASK_{bit:x}", "mx_unmask", bit)
        for bit in (0x80, 0x100, 0x200, 0x400, 0x800, 0x1000)
    )
    modes += tuple(
        (f"X87_UNMASK_{bit:x}", "x87_unmask", bit)
        for bit in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20)
    )

    operations = (
        lambda: decoder.create_private_numeric_result_decoder(
            deadline=deadline
        ),
        lambda: dport.decode_dense(
            dense_raw, expected_rows=1, expected_width=1
        ),
        lambda: dport.decode_conv(
            conv_raw, expected_rows=1, expected_width=1
        ),
    )

    def set_mode(kind, bits):
        library.act_set_mxcsr(old_mxcsr)
        library.act_set_x87(old_x87)
        library.act_clear_flags()
        if kind == "mx_mode":
            library.act_set_mxcsr(
                (library.act_get_mxcsr() & ~0xE040) | bits
            )
        elif kind == "mx_unmask":
            library.act_set_mxcsr(
                library.act_get_mxcsr() & ~bits
            )
        elif kind == "x87_mode":
            library.act_set_x87((old_x87 & ~0x0C00) | bits)
        elif kind == "x87_precision":
            library.act_set_x87((old_x87 & ~0x0300) | bits)
        else:
            library.act_set_x87(old_x87 & ~bits)

    for name, kind, bits in modes:
        codes = []
        for operation in operations:
            try:
                set_mode(kind, bits)
                expected_state = (
                    library.act_get_mxcsr(),
                    library.act_get_x87(),
                )
                for _attempt in range(2):
                    try:
                        operation()
                    except decoder.PrivateNumericDecoderError as exc:
                        codes.append(exc.code)
                    else:
                        codes.append("RETURNED")
                    observed_state = (
                        library.act_get_mxcsr(),
                        library.act_get_x87(),
                    )
                    if observed_state != expected_state:
                        raise AssertionError(
                            (
                                name,
                                "state changed",
                                expected_state,
                                observed_state,
                            )
                        )
            finally:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)
        if codes != ["NUMERIC_PLATFORM"] * (2 * len(operations)):
            raise AssertionError((name, codes))

    for bit in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20):
        for operation in operations:
            try:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)
                library.act_clear_flags()
                library.act_set_x87_pending_unmasked(bit)
                before_control = library.act_get_x87()
                before_mxcsr = library.act_get_mxcsr()
                before_status = library.act_get_x87_status()
                if (
                    before_control & bit
                    or not before_status & bit
                ):
                    raise AssertionError(
                        ("pending setup failed", bit, before_status)
                    )
                codes = []
                for _attempt in range(2):
                    try:
                        operation()
                    except decoder.PrivateNumericDecoderError as exc:
                        codes.append(exc.code)
                    else:
                        codes.append("RETURNED")
                after_control = library.act_get_x87()
                after_mxcsr = library.act_get_mxcsr()
                after_status = library.act_get_x87_status()
                if (
                    codes != ["NUMERIC_PLATFORM", "NUMERIC_PLATFORM"]
                    or after_control != before_control
                    or after_mxcsr != before_mxcsr
                    or after_status & 0x80FF
                ):
                    raise AssertionError(
                        (
                            "pending rejection",
                            bit,
                            codes,
                            before_control,
                            after_control,
                            before_mxcsr,
                            after_mxcsr,
                            before_status,
                            after_status,
                        )
                    )
            finally:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)

    for bit in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20):
        for operation in operations:
            try:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)
                library.act_clear_flags()
                library.act_set_mxcsr_pending_unmasked(bit)
                before_mxcsr = library.act_get_mxcsr()
                before_control = before_mxcsr & ~0x003F
                if (
                    not before_mxcsr & bit
                    or before_mxcsr & (bit << 7)
                ):
                    raise AssertionError(
                        ("MXCSR pending setup failed", bit, before_mxcsr)
                    )
                codes = []
                for _attempt in range(2):
                    try:
                        operation()
                    except decoder.PrivateNumericDecoderError as exc:
                        codes.append(exc.code)
                    else:
                        codes.append("RETURNED")
                after_mxcsr = library.act_get_mxcsr()
                if (
                    codes != ["NUMERIC_PLATFORM", "NUMERIC_PLATFORM"]
                    or after_mxcsr & ~0x003F != before_control
                    or after_mxcsr & bit
                ):
                    raise AssertionError(
                        (
                            "MXCSR pending rejection",
                            bit,
                            codes,
                            before_mxcsr,
                            after_mxcsr,
                        )
                    )
            finally:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)

    try:
        library.act_set_mxcsr(old_mxcsr)
        library.act_set_x87(old_x87)
        library.act_clear_flags()
        library.act_set_mxcsr(
            library.act_get_mxcsr() & ~0x0100
        )
        expected_state = (
            library.act_get_mxcsr(),
            library.act_get_x87(),
        )
        nested_codes = []
        in_callback = [False]

        def trace_reentry(frame, event, argument):
            del argument
            if (
                event == "line"
                and frame.f_code.co_name == "_check_fenv_control"
                and not in_callback[0]
                and not nested_codes
                and frame.f_locals.get("read_status") == 0
                and "control_word" not in frame.f_locals
            ):
                in_callback[0] = True
                try:
                    try:
                        dport.decode_dense(
                            dense_raw,
                            expected_rows=1,
                            expected_width=1,
                        )
                    except decoder.PrivateNumericDecoderError as exc:
                        nested_codes.append(exc.code)
                    else:
                        nested_codes.append("RETURNED")
                finally:
                    in_callback[0] = False
            return trace_reentry

        outer_code = "RETURNED"
        sys.settrace(trace_reentry)
        try:
            try:
                dport.decode_dense(
                    dense_raw,
                    expected_rows=1,
                    expected_width=1,
                )
            except decoder.PrivateNumericDecoderError as exc:
                outer_code = exc.code
        finally:
            sys.settrace(None)
        second_code = "RETURNED"
        try:
            dport.decode_dense(
                dense_raw,
                expected_rows=1,
                expected_width=1,
            )
        except decoder.PrivateNumericDecoderError as exc:
            second_code = exc.code
        if (
            nested_codes != ["NUMERIC_PLATFORM"]
            or outer_code != "NUMERIC_PLATFORM"
            or second_code != "NUMERIC_PLATFORM"
            or (
                library.act_get_mxcsr(),
                library.act_get_x87(),
            )
            != expected_state
        ):
            raise AssertionError(
                (
                    "trace re-entry",
                    nested_codes,
                    outer_code,
                    second_code,
                    expected_state,
                    library.act_get_mxcsr(),
                    library.act_get_x87(),
                )
            )
    finally:
        sys.settrace(None)
        library.act_set_mxcsr(old_mxcsr)
        library.act_set_x87(old_x87)

    if (
        tuple(value.tobytes() for value in dport.decode_dense(
            dense_raw, expected_rows=1, expected_width=1
        )[2:])
        != tuple(value.tobytes() for value in baseline_dense[2:])
        or tuple(value.tobytes() for value in dport.decode_conv(
            conv_raw, expected_rows=1, expected_width=1
        )[2:])
        != tuple(value.tobytes() for value in baseline_conv[2:])
    ):
        raise AssertionError("floating-environment restoration changed bits")
"""
        )

    def test_native_fenv_reader_state_is_gated_after_factory(self):
        functions = {
            value.__name__: value
            for value in decoder._GatePrimitivesModule
            if isinstance(value, ctypes._CFuncPtr)
        }
        self.assertEqual(set(functions), {"fegetenv", "fesetenv"})
        reader = functions["fegetenv"]
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        raw = _make_dense_raw()
        baseline = port.decode_dense(
            raw, expected_rows=3, expected_width=2
        )
        calls = [0]

        def changed(*args, **kwargs):
            del args, kwargs
            calls[0] += 1
            raise AssertionError("changed native call path was invoked")

        reader.errcheck = changed
        try:
            with self.assertRaises(
                decoder.PrivateNumericDecoderError
            ) as captured:
                port.decode_dense(
                    raw, expected_rows=3, expected_width=2
                )
            self.assertEqual(
                captured.exception.code, "DEPENDENCY_SUBSTITUTION"
            )
        finally:
            del reader.errcheck
        self.assertEqual(calls[0], 0)

        function_type = type(reader)
        self.assertNotIn("__call__", function_type.__dict__)
        function_type.__call__ = changed
        try:
            observed = port.decode_dense(
                raw, expected_rows=3, expected_width=2
            )
        finally:
            del function_type.__call__
        self.assertEqual(calls[0], 0)
        self.assertEqual(
            tuple(value.tobytes() for value in observed[2:]),
            tuple(value.tobytes() for value in baseline[2:]),
        )

    def test_exact_numpy_array_memory_error_is_normalized(self):
        _run_isolated(
            r"""
import importlib
import sys
import time

import numpy as np
from numpy._core import _exceptions as np_exceptions

decoder_name = (
    "act.back_end.hybridz_tf."
    "query_dual_replay_v51b_private_decoder"
)
sys.modules.pop(decoder_name, None)
original = np.isfinite

def raise_array_memory(*args, **kwargs):
    del args, kwargs
    raise np_exceptions._ArrayMemoryError(
        (1024, 1024), np.dtype(np.float64)
    )

np.isfinite = raise_array_memory
try:
    decoder = importlib.import_module(decoder_name)
    port = decoder.create_private_numeric_result_decoder(
        deadline=time.monotonic() + 30.0
    )
    f64_tag = np.dtype(np.float64).str.encode("ascii")
    bool_tag = np.dtype(np.bool_).str.encode("ascii")
    scalar_f64 = (bytes(8), (1,), f64_tag)
    raw = (
        b"act.v51b.private.dense-result.v1",
        False,
        (bytes(8), (1, 1), f64_tag),
        scalar_f64,
        scalar_f64,
        scalar_f64,
        scalar_f64,
        (bytes(1), (1,), bool_tag),
        (bytes(1), (1,), bool_tag),
    )
    try:
        port.decode_dense(
            raw, expected_rows=1, expected_width=1
        )
    except decoder.PrivateNumericDecoderError as exc:
        if (
            exc.code != "RESOURCE_LIMIT"
            or type(exc.__cause__)
            is not np_exceptions._ArrayMemoryError
        ):
            raise AssertionError(
                (
                    "normalization",
                    exc.code,
                    type(exc.__cause__),
                )
            )
    else:
        raise AssertionError("NumPy ArrayMemoryError escaped decoder")
finally:
    np.isfinite = original
    sys.modules.pop(decoder_name, None)
"""
        )

    def test_aggregate_decode_budget_precedes_frame_access(self):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        for operation in (
            lambda: port.decode_dense(
                object(), expected_rows=8193, expected_width=16384
            ),
            lambda: port.decode_conv(
                object(), expected_rows=2049, expected_width=65536
            ),
        ):
            with self.assertRaises(
                decoder.PrivateNumericDecoderError
            ) as captured:
                operation()
            self.assertEqual(captured.exception.code, "RESOURCE_LIMIT")
            self.assertIsNone(captured.exception.__cause__)

    def test_module_subclass_is_rejected_before_dynamic_attribute_call(self):
        _run_isolated(
            r"""
import math
import os
import time
import types
import ctypes

import numpy as np
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)

calls = [0]

class ChangedModule(types.ModuleType):
    def __getattribute__(self, name):
        calls[0] += 1
        raise AssertionError(("changed module attribute called", name))

for module in (ctypes, math, os, np, types, ufunc_config):
    saved_type = type(module)
    module.__class__ = ChangedModule
    error = None
    try:
        try:
            decoder.create_private_numeric_result_decoder(
                deadline=float(time.monotonic() + 30.0)
            )
        except decoder.PrivateNumericDecoderError as exc:
            error = exc
    finally:
        module.__class__ = saved_type
    if error is None or error.code != "DEPENDENCY_SUBSTITUTION":
        raise AssertionError(("factory accepted module subclass", module))

if calls[0] != 0:
    raise AssertionError(("changed module call count", calls[0]))
"""
        )

    def test_pre_factory_dependency_matrix_and_restore_cycle(self):
        _run_isolated(
            r"""
import _thread
import builtins
import ctypes
import math
import os
import time
import types
import weakref

import numpy as np
from numpy._core import _exceptions as np_exceptions
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)

saved_getattr = getattr
saved_setattr = setattr
deadline = time.monotonic() + 120.0
calls = [0]

def changed(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("changed dependency was called")

targets = [
    (builtins, "bytes"),
    (builtins, "int"),
    (builtins, "tuple"),
    (builtins, "type"),
    (builtins, "float"),
    (builtins, "bool"),
    (builtins, "str"),
    (builtins, "dict"),
    (builtins, "object"),
    (builtins, "id"),
    (builtins, "len"),
    (builtins, "range"),
    (builtins, "property"),
    (builtins, "Exception"),
    (builtins, "FloatingPointError"),
    (builtins, "MemoryError"),
    (builtins, "OverflowError"),
    (builtins, "RuntimeError"),
    (builtins, "TypeError"),
    (builtins, "ValueError"),
    (ctypes, "Array"),
    (ctypes, "CDLL"),
    (ctypes, "_CFuncPtr"),
    (ctypes, "c_int"),
    (ctypes, "c_ubyte"),
    (np, "__version__"),
    (np, "frombuffer"),
    (np, "asarray"),
    (np, "ascontiguousarray"),
    (np, "zeros"),
    (np, "isfinite"),
    (np, "minimum"),
    (np, "nextafter"),
    (np, "logical_and"),
    (np, "logical_or"),
    (np, "ndarray"),
    (np, "dtype"),
    (np, "float64"),
    (np, "bool_"),
    (np, "uint64"),
    (np, "longdouble"),
    (np_exceptions, "_ArrayMemoryError"),
    (math, "inf"),
    (math, "isfinite"),
    (os, "getpid"),
    (os, "uname"),
    (time, "monotonic"),
    (_thread, "RLock"),
    (types, "FunctionType"),
    (types, "MappingProxyType"),
    (types, "ModuleType"),
    (weakref, "ref"),
    (ufunc_config, "_extobj_contextvar"),
    (ufunc_config, "_make_extobj"),
]

for owner, name in targets:
    saved = saved_getattr(owner, name)
    saved_setattr(owner, name, changed)
    error = None
    try:
        try:
            decoder.create_private_numeric_result_decoder(
                deadline=deadline
            )
        except decoder.PrivateNumericDecoderError as exc:
            error = exc
    finally:
        saved_setattr(owner, name, saved)
    if error is None or error.code != "DEPENDENCY_SUBSTITUTION":
        raise AssertionError(("factory accepted changed binding", name))

# A complete change-and-restore cycle before construction is harmless.
saved = np.nextafter
np.nextafter = changed
np.nextafter = saved
port = decoder.create_private_numeric_result_decoder(deadline=deadline)
if port.proof_authority is not False:
    raise AssertionError("decoder gained proof authority")
if calls[0] != 0:
    raise AssertionError(("changed call count", calls[0]))
"""
        )

    def test_pre_factory_ufunc_reduce_override_is_rejected_before_call(self):
        _run_isolated(
            r"""
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)

calls = [0]

def changed(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("changed ufunc reduce was called")

for name in ("logical_and", "logical_or"):
    ufunc = getattr(np, name)
    if "reduce" in ufunc.__dict__:
        raise AssertionError((name, "unexpected baseline override"))
    setattr(ufunc, "reduce", changed)
    error = None
    try:
        try:
            decoder.create_private_numeric_result_decoder(
                deadline=time.monotonic() + 30.0
            )
        except decoder.PrivateNumericDecoderError as exc:
            error = exc
    finally:
        delattr(ufunc, "reduce")
    if error is None or error.code != "DEPENDENCY_SUBSTITUTION":
        raise AssertionError((name, "factory accepted reduce override"))

np.logical_or._act_state_probe = object()
try:
    try:
        decoder.create_private_numeric_result_decoder(
            deadline=time.monotonic() + 30.0
        )
    except decoder.PrivateNumericDecoderError as exc:
        if exc.code != "DEPENDENCY_SUBSTITUTION":
            raise AssertionError(("state probe", exc.code))
    else:
        raise AssertionError("factory accepted changed ufunc state")
finally:
    delattr(np.logical_or, "_act_state_probe")

if calls[0] != 0:
    raise AssertionError(("changed reduce call count", calls[0]))
"""
        )

    def test_post_factory_ufunc_reduce_overrides_are_inert(self):
        _run_isolated(
            r"""
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as kernel,
)

deadline = time.monotonic() + 30.0
kport = kernel.create_private_numeric_kernel(deadline=deadline)
locator = kport.admit_dense(
    weight=np.asarray(
        [[-0.2, 0.5], [1.0, -0.25]], dtype=np.float64
    ),
    predecessor_max_abs=np.asarray([5.0, 2.0], dtype=np.float64),
    tile_width=1,
)
raw = kport.execute_dense(
    locator,
    np.asarray(
        [[-0.75, 0.25], [-0.0, 0.0], [1.0, -2.0]],
        dtype=np.float64,
    ),
)
dport = decoder.create_private_numeric_result_decoder(deadline=deadline)
baseline = dport.decode_dense(
    raw, expected_rows=3, expected_width=2
)
baseline_bits = tuple(value.tobytes(order="C") for value in baseline[2:])
calls = [0]

def changed(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("changed ufunc reduce was called")

ufuncs = (np.logical_and, np.logical_or)
try:
    for ufunc in ufuncs:
        setattr(ufunc, "reduce", changed)
    observed = dport.decode_dense(
        raw, expected_rows=3, expected_width=2
    )
finally:
    for ufunc in reversed(ufuncs):
        delattr(ufunc, "reduce")
observed_bits = tuple(
    value.tobytes(order="C") for value in observed[2:]
)
if calls[0] != 0 or observed_bits != baseline_bits:
    raise AssertionError(("post-factory reduce mutation", calls[0]))
"""
        )

    def test_post_factory_persistent_and_aba_changes_are_inert(self):
        _run_isolated(
            r"""
import _thread
import builtins
import ctypes
import math
import os
import time
import types
import weakref

import numpy as np
from numpy._core import _exceptions as np_exceptions
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_decoder as decoder,
)
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as kernel,
)

deadline = time.monotonic() + 120.0
kport = kernel.create_private_numeric_kernel(deadline=deadline)
dense_locator = kport.admit_dense(
    weight=np.asarray(
        [[-0.2, 0.5], [1.0, -0.25]], dtype=np.float64
    ),
    predecessor_max_abs=np.asarray([5.0, 2.0], dtype=np.float64),
    tile_width=1,
)
dense_raw = kport.execute_dense(
    dense_locator,
    np.asarray(
        [[-0.75, 0.25], [-0.0, 0.0], [1.0, -2.0]],
        dtype=np.float64,
    ),
)
dport = decoder.create_private_numeric_result_decoder(
    deadline=deadline
)
baseline = dport.decode_dense(
    dense_raw, expected_rows=3, expected_width=2
)
baseline_bits = tuple(value.tobytes(order="C") for value in baseline[2:])

saved_getattr = getattr
saved_setattr = setattr
calls = [0]

def changed(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("changed dependency was called")

targets = [
    (ctypes, "Array"),
    (ctypes, "CDLL"),
    (ctypes, "_CFuncPtr"),
    (ctypes, "c_int"),
    (ctypes, "c_ubyte"),
    (np, "__version__"),
    (builtins, "bytes"),
    (builtins, "int"),
    (builtins, "tuple"),
    (builtins, "type"),
    (builtins, "len"),
    (builtins, "range"),
    (builtins, "property"),
    (np, "frombuffer"),
    (np, "asarray"),
    (np, "ascontiguousarray"),
    (np, "zeros"),
    (np, "isfinite"),
    (np, "minimum"),
    (np, "nextafter"),
    (np, "logical_and"),
    (np, "logical_or"),
    (np, "ndarray"),
    (np, "float64"),
    (np, "longdouble"),
    (np_exceptions, "_ArrayMemoryError"),
    (math, "isfinite"),
    (os, "getpid"),
    (time, "monotonic"),
    (_thread, "RLock"),
    (types, "MappingProxyType"),
    (weakref, "ref"),
    (ufunc_config, "_extobj_contextvar"),
    (ufunc_config, "_make_extobj"),
]

for owner, name in targets:
    saved = saved_getattr(owner, name)
    saved_setattr(owner, name, changed)
    try:
        current = dport.decode_dense(
            dense_raw, expected_rows=3, expected_width=2
        )
    finally:
        saved_setattr(owner, name, saved)
    bits = tuple(value.tobytes(order="C") for value in current[2:])
    if bits != baseline_bits:
        raise AssertionError(("persistent change altered bits", name))

# One complete change-and-restore cycle before use is also inert.
saved = np.nextafter
np.nextafter = changed
np.nextafter = saved
current = dport.decode_dense(
    dense_raw, expected_rows=3, expected_width=2
)
bits = tuple(value.tobytes(order="C") for value in current[2:])
if bits != baseline_bits:
    raise AssertionError("change-and-restore altered bits")
if calls[0] != 0:
    raise AssertionError(("changed call count", calls[0]))
"""
        )

    def test_reflective_private_build_class_change_is_inert(self):
        implementation = _implementation_spec(
            decoder.create_private_numeric_result_decoder,
            "_create_private_numeric_result_decoder_impl",
        )
        self.assertFalse(
            any(
                type(cell.cell_contents) is types.FunctionType
                for cell in (
                    decoder.create_private_numeric_result_decoder.__closure__
                    or ()
                )
            )
        )
        first = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        implementation_globals = type(first).decode_dense.__globals__
        private_builtins = implementation_globals["__builtins__"]
        self.assertIs(type(private_builtins), dict)
        self.assertEqual(private_builtins, {})
        calls = [0]

        def changed(*args, **kwargs):
            del args, kwargs
            calls[0] += 1
            raise AssertionError("changed private class builder was called")

        private_builtins["__build_class__"] = changed
        try:
            decoded = first.decode_dense(
                _make_dense_raw(), expected_rows=3, expected_width=2
            )
        finally:
            del private_builtins["__build_class__"]
        self.assertEqual(decoded[0], _DENSE_DECODED_TAG)
        self.assertEqual(calls[0], 0)

        original_entry = implementation_globals["__builtins__"]
        implementation_globals["__builtins__"] = {
            "__build_class__": changed
        }
        try:
            second = decoder.create_private_numeric_result_decoder(
                deadline=_end()
            )
            self.assertIs(second.proof_authority, False)
            self.assertIsNot(
                type(second).decode_dense.__globals__,
                implementation_globals,
            )
        finally:
            implementation_globals["__builtins__"] = original_entry
        self.assertEqual(calls[0], 0)
        self.assertIs(type(implementation[0]), types.CodeType)
        self.assertIs(implementation[4], types.FunctionType)
        self.assertIs(implementation[5], dict)

    def test_recursive_decoder_bytecode_has_no_dynamic_global_or_class_load(
        self,
    ):
        port = decoder.create_private_numeric_result_decoder(
            deadline=_end()
        )
        seen = set()
        failures = []

        def visit(code, path):
            if id(code) in seen:
                return
            seen.add(id(code))
            for instruction in dis.get_instructions(code):
                if instruction.opname in {
                    "LOAD_GLOBAL",
                    "STORE_GLOBAL",
                    "DELETE_GLOBAL",
                    "IMPORT_NAME",
                    "IMPORT_FROM",
                    "LOAD_BUILD_CLASS",
                }:
                    failures.append(
                        (
                            path,
                            instruction.offset,
                            instruction.opname,
                            instruction.argrepr,
                        )
                    )
            for constant in code.co_consts:
                if type(constant) is types.CodeType:
                    visit(constant, f"{path}/{constant.co_name}")

        visit(
            decoder.create_private_numeric_result_decoder.__code__,
            "factory",
        )
        implementation = _implementation_spec(
            decoder.create_private_numeric_result_decoder,
            "_create_private_numeric_result_decoder_impl",
        )
        visit(implementation[0], "implementation")
        for name, value in type(port).__dict__.items():
            if type(value) is types.FunctionType:
                visit(value.__code__, f"port/{name}")
            elif type(value) is property:
                visit(value.fget.__code__, f"port/{name}.fget")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
