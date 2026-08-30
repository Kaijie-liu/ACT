"""Soundness, fail-closed, and integrity tests for the outward BOX anchor."""

from __future__ import annotations

import copy
import unittest
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from act.back_end.hybridz_tf.query_dual_box_certifier import (
    QueryDualBoxError,
    QueryDualBoxTimeout,
    certify_query_dual_boxes,
    verify_query_dual_box_certificate,
    verify_query_dual_box_receipt,
)


_ETA = float.fromhex("0x0.0000000000001p-1022")


def _layer(layer_id, kind, width, params=None):
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[int(layer_id) * 1_000_000 + index for index in range(int(width))],
        cache={},
    )


def _net(layers, preds):
    pred_map = {
        int(layer.id): [int(parent) for parent in preds[int(layer.id)]]
        for layer in layers
    }
    succs = {int(layer.id): [] for layer in layers}
    for child, parents in pred_map.items():
        for parent in parents:
            succs[parent].append(child)
    return SimpleNamespace(
        layers=list(layers),
        preds=pred_map,
        succs=succs,
        by_id={int(layer.id): layer for layer in layers},
    )


def _input_pair(width, lower, upper):
    lower = np.asarray(lower, dtype=np.float64).reshape(1, -1)
    upper = np.asarray(upper, dtype=np.float64).reshape(1, -1)
    return (
        _layer(0, "INPUT", width, {"shape": (1, width), "dtype": "float64"}),
        _layer(
            1,
            "INPUT_SPEC",
            width,
            {"kind": "BOX", "lb": lower, "ub": upper},
        ),
    )


def _assert_contains(test, bounds, exact):
    lower = bounds.lb.detach().cpu().numpy().reshape(-1)
    upper = bounds.ub.detach().cpu().numpy().reshape(-1)
    exact = np.asarray(exact, dtype=np.float64).reshape(-1)
    test.assertEqual(lower.shape, exact.shape)
    test.assertTrue(np.all(lower <= exact))
    test.assertTrue(np.all(exact <= upper))


def _cancellation_net():
    inp, spec = _input_pair(3, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    dense = _layer(
        2,
        "DENSE",
        1,
        {
            "weight": np.asarray([[1.0e16, 1.0, -1.0e16]], dtype=np.float64),
            "bias": np.asarray([0.0], dtype=np.float64),
        },
    )
    relu = _layer(3, "RELU", 1)
    assertion = _layer(4, "ASSERT", 1, {"kind": "AUDIT"})
    return _net(
        [inp, spec, dense, relu, assertion],
        {0: [], 1: [0], 2: [1], 3: [2], 4: [3]},
    )


def _residual_net(*, add_bias=None):
    inp, spec = _input_pair(2, [0.25, -0.5], [0.25, -0.5])
    stem = _layer(
        2,
        "DENSE",
        2,
        {
            "weight": np.asarray([[2.0, -1.0], [1.0, 1.0]], dtype=np.float64),
            "bias": np.asarray([0.0, 0.25], dtype=np.float64),
        },
    )
    relu = _layer(3, "RELU", 2)
    left = _layer(
        4,
        "DENSE",
        2,
        {
            "weight": np.asarray([[1.0, 2.0], [-1.0, 1.0]], dtype=np.float64),
            "bias": np.asarray([0.5, 0.0], dtype=np.float64),
        },
    )
    right = _layer(
        5,
        "DENSE",
        2,
        {
            "weight": np.asarray([[0.5, -1.0], [2.0, 0.25]], dtype=np.float64),
            "bias": np.asarray([-0.25, 1.0], dtype=np.float64),
        },
    )
    add_params = {} if add_bias is None else {"bias": add_bias}
    add = _layer(6, "ADD", 2, add_params)
    flatten = _layer(7, "FLATTEN", 2, {"start_dim": 1, "end_dim": -1})
    assertion = _layer(8, "ASSERT", 2, {"kind": "AUDIT"})
    return _net(
        [inp, spec, stem, relu, left, right, add, flatten, assertion],
        {
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [3],
            6: [4, 5],
            7: [6],
            8: [7],
        },
    )


def _subnormal_net():
    inp, spec = _input_pair(1, [1.0], [1.0])
    dense = _layer(
        2,
        "DENSE",
        1,
        {
            "weight": np.asarray([[_ETA]], dtype=np.float64),
            "bias": np.asarray([0.0], dtype=np.float64),
        },
    )
    assertion = _layer(3, "ASSERT", 1, {"kind": "AUDIT"})
    return _net(
        [inp, spec, dense, assertion],
        {0: [], 1: [0], 2: [1], 3: [2]},
    )


def _grouped_conv_net():
    in_shape = (4, 5, 6)
    output_shape = (4, 3, 4)
    width = int(np.prod(in_shape))
    values = (np.arange(width, dtype=np.float64) % 7.0) - 3.0
    inp, spec = _input_pair(width, values, values)
    weight = (np.arange(4 * 2 * 2 * 2, dtype=np.float64) % 5.0) - 2.0
    weight = weight.reshape(4, 2, 2, 2)
    bias = np.asarray([1.0, -2.0, 0.0, 3.0], dtype=np.float64)
    conv = _layer(
        2,
        "CONV2D",
        int(np.prod(output_shape)),
        {
            "weight": weight,
            "bias": bias,
            "input_shape": in_shape,
            "output_shape": output_shape,
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (2, 1),
            "groups": 2,
            "padding_mode": "zeros",
        },
    )
    assertion = _layer(3, "ASSERT", int(np.prod(output_shape)), {"kind": "AUDIT"})
    net = _net(
        [inp, spec, conv, assertion],
        {0: [], 1: [0], 2: [1], 3: [2]},
    )
    expected = F.conv2d(
        torch.from_numpy(values.reshape(1, *in_shape)),
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(2, 1),
        groups=2,
    ).numpy()
    return net, expected


class QueryDualBoxCertifierTests(unittest.TestCase):
    def test_cancellation_is_enclosed_and_relu_uses_preactivation(self):
        net = _cancellation_net()
        certificate = certify_query_dual_boxes(net)
        exact = sum(
            Fraction.from_float(value)
            for value in (1.0e16, 1.0, -1.0e16)
        )
        self.assertEqual(exact, Fraction(1, 1))
        for lid in (2, 3):
            lower = Fraction.from_float(float(certificate.bounds[lid].lb[0, 0]))
            upper = Fraction.from_float(float(certificate.bounds[lid].ub[0, 0]))
            self.assertLessEqual(lower, exact)
            self.assertLessEqual(exact, upper)
        self.assertEqual(certificate.semantics[2], "output")
        self.assertEqual(certificate.semantics[3], "preactivation")
        self.assertTrue(verify_query_dual_box_certificate(certificate, net=net))
        self.assertFalse(certificate.receipt["ordinary_interval_facts_consumed"])

    def test_residual_point_fanout_and_complete_coverage(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        # stem=[1,0], relu=[1,0], left=[1.5,-1], right=[0.25,3]
        expected = np.asarray([1.75, 2.0], dtype=np.float64)
        _assert_contains(self, certificate.bounds[6], expected)
        _assert_contains(self, certificate.bounds[7], expected)
        self.assertEqual(set(certificate.bounds), {1, 2, 3, 4, 5, 6, 7})
        self.assertEqual(certificate.receipt["coverage_count"], 7)

    def test_grouped_strided_padded_dilated_conv(self):
        net, expected = _grouped_conv_net()
        certificate = certify_query_dual_boxes(net, conv_channel_chunk=1)
        _assert_contains(self, certificate.bounds[2], expected)
        self.assertEqual(
            certificate.receipt["numeric_method"]["conv_guard_scope"],
            "independent-per-channel-chunk",
        )

    def test_subnormal_and_ftz_fail_closed(self):
        net = _subnormal_net()
        certificate = certify_query_dual_boxes(net)
        _assert_contains(self, certificate.bounds[2], [_ETA])
        self.assertTrue(certificate.receipt["numeric_platform"]["round_to_nearest_even"])
        self.assertTrue(
            certificate.receipt["numeric_platform"][
                "torch_matmul_gradual_underflow_probe"
            ]
        )
        self.assertTrue(
            certificate.receipt["numeric_platform"][
                "torch_clamp_abs_gradual_underflow_probe"
            ]
        )
        try:
            torch.set_flush_denormal(True)
            with self.assertRaises(QueryDualBoxError) as caught:
                certify_query_dual_boxes(net)
            self.assertEqual(caught.exception.code, "NUMERIC_PLATFORM")
        finally:
            torch.set_flush_denormal(False)

    def test_nonzero_add_bias_is_exactly_rejected(self):
        net = _residual_net(add_bias=np.asarray([_ETA, 0.0], dtype=np.float64))
        with self.assertRaises(QueryDualBoxError) as caught:
            certify_query_dual_boxes(net)
        self.assertEqual(caught.exception.code, "UNSUPPORTED_ADD")
        zero_net = _residual_net(add_bias=np.asarray([-0.0, 0.0]))
        self.assertTrue(certify_query_dual_boxes(zero_net).proof_authority)

    def test_raw_input_and_live_bound_mutations_are_detected(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        original_input_hash = certificate.receipt["hashes"]["input_sha256"]
        net.layers[1].params["lb"][0, 0] = 0.125
        self.assertFalse(verify_query_dual_box_certificate(certificate, net=net))
        changed = certify_query_dual_boxes(net)
        self.assertNotEqual(
            original_input_hash,
            changed.receipt["hashes"]["input_sha256"],
        )
        self.assertTrue(verify_query_dual_box_certificate(changed, net=net))
        changed.bounds[2].lb[0, 0] -= 1.0
        self.assertFalse(verify_query_dual_box_certificate(changed, net=net))

    def test_receipt_pins_tamper_and_malformed_object(self):
        net = _cancellation_net()
        certificate = certify_query_dual_boxes(net)
        hashes = certificate.receipt["hashes"]
        self.assertTrue(
            verify_query_dual_box_receipt(
                certificate.receipt,
                expected_net_sha256=hashes["net_sha256"],
                expected_input_sha256=hashes["input_sha256"],
                expected_bounds_sha256=hashes["bounds_sha256"],
                expected_implementation_sha256=hashes["implementation_sha256"],
            )
        )
        self.assertFalse(
            verify_query_dual_box_receipt(
                certificate.receipt,
                expected_net_sha256="0" * 64,
            )
        )
        forged = copy.deepcopy(dict(certificate.receipt))
        forged["coverage_count"] += 1
        self.assertFalse(verify_query_dual_box_receipt(forged))
        self.assertFalse(verify_query_dual_box_certificate(object()))

    def test_deadline_and_hash_mismatch_fail_closed(self):
        net = _cancellation_net()
        with self.assertRaises(QueryDualBoxTimeout):
            certify_query_dual_boxes(net, timeout_s=0.0)
        with self.assertRaises(QueryDualBoxError) as caught:
            certify_query_dual_boxes(net, expected_net_sha256="0" * 64)
        self.assertEqual(caught.exception.code, "HASH_MISMATCH")


if __name__ == "__main__":
    unittest.main()
