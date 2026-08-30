"""Bitwise CPU gates for DualTF forward-state last-use reclamation."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import tf_forward
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import initialize_device


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _assertion(n_out: int) -> dict:
    return OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.ones(n_out, dtype=DTYPE, device=DEVICE),
        d=torch.tensor([100.0], dtype=DTYPE, device=DEVICE),
    ).encode_linear(B=1, n_out=n_out, device=DEVICE, dtype=DTYPE)


def _input_layers(n_in: int) -> list[Layer]:
    variables = list(range(n_in))
    return [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, n_in), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=variables,
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.full((1, n_in), -1.0, dtype=DTYPE),
                "ub": torch.full((1, n_in), 1.0, dtype=DTYPE),
            },
            in_vars=variables,
            out_vars=variables,
        ),
    ]


def _dense(
    layer_id: int,
    in_vars: list[int],
    out_vars: list[int],
    weight: list[list[float]],
    bias: list[float],
) -> Layer:
    return Layer(
        id=layer_id,
        kind="DENSE",
        params={
            "weight": torch.tensor(weight, dtype=DTYPE, device=DEVICE),
            "bias": torch.tensor(bias, dtype=DTYPE, device=DEVICE),
            "in_features": len(in_vars),
            "out_features": len(out_vars),
        },
        in_vars=in_vars,
        out_vars=out_vars,
    )


def _chain_net() -> Net:
    layers = _input_layers(2)
    layers.extend(
        [
            _dense(
                2,
                [0, 1],
                [2, 3, 4],
                [[1.0, 0.5], [-0.25, 2.0], [0.75, -1.0]],
                [0.125, -0.5, 0.25],
            ),
            Layer(
                id=3,
                kind="RELU",
                params={},
                in_vars=[2, 3, 4],
                out_vars=[5, 6, 7],
            ),
            _dense(
                4,
                [5, 6, 7],
                [8, 9],
                [[1.0, -0.5, 0.25], [-0.75, 0.5, 1.0]],
                [0.0, 0.375],
            ),
            Layer(
                id=5,
                kind="ASSERT",
                params=_assertion(2),
                in_vars=[8, 9],
                out_vars=[8, 9],
            ),
        ]
    )
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _residual_net() -> Net:
    layers = _input_layers(2)
    layers.extend(
        [
            _dense(
                2,
                [0, 1],
                [2, 3],
                [[1.0, 0.25], [-0.5, 1.0]],
                [0.125, -0.25],
            ),
            _dense(
                3,
                [2, 3],
                [4, 5],
                [[0.75, -0.5], [0.25, 1.25]],
                [0.0, 0.125],
            ),
            Layer(
                id=4,
                kind="RELU",
                params={},
                in_vars=[4, 5],
                out_vars=[6, 7],
            ),
            _dense(
                5,
                [6, 7],
                [8, 9],
                [[1.0, -0.25], [0.5, 0.75]],
                [-0.125, 0.25],
            ),
            _dense(
                6,
                [2, 3],
                [10, 11],
                [[0.5, 0.0], [0.0, 0.5]],
                [0.0, 0.0],
            ),
            Layer(
                id=7,
                kind="ADD",
                params={"x_vars": [8, 9], "y_vars": [10, 11]},
                in_vars=[8, 9, 10, 11],
                out_vars=[12, 13],
            ),
            Layer(
                id=8,
                kind="RELU",
                params={},
                in_vars=[12, 13],
                out_vars=[14, 15],
            ),
            Layer(
                id=9,
                kind="ASSERT",
                params=_assertion(2),
                in_vars=[14, 15],
                out_vars=[14, 15],
            ),
        ]
    )
    return Net(
        layers=layers,
        preds={
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [4],
            6: [2],
            7: [5, 6],
            8: [7],
            9: [8],
        },
        succs={
            0: [1],
            1: [2],
            2: [3, 6],
            3: [4],
            4: [5],
            5: [7],
            6: [7],
            7: [8],
            8: [9],
            9: [],
        },
    )


def _repeated_alias_net() -> Net:
    layers = _input_layers(2)
    layers.extend(
        [
            _dense(
                2,
                [0, 1],
                [2, 3],
                [[1.0, -0.5], [0.25, 0.75]],
                [0.125, -0.25],
            ),
            Layer(
                id=3,
                kind="FLATTEN",
                params={},
                in_vars=[2, 3],
                out_vars=[4, 5],
            ),
            Layer(
                id=4,
                kind="ADD",
                params={"x_vars": [4, 5], "y_vars": [4, 5]},
                in_vars=[4, 5, 4, 5],
                out_vars=[6, 7],
            ),
            Layer(
                id=5,
                kind="ASSERT",
                params=_assertion(2),
                in_vars=[6, 7],
                out_vars=[6, 7],
            ),
        ]
    )
    # Layer 4 intentionally consumes layer 3 twice. ``succs`` contains one
    # consumer node, matching the one last-use event after both handler reads.
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3, 3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _assert_bounds_bitwise_equal(
    testcase: unittest.TestCase,
    expected: dict[int, Bounds],
    actual: dict[int, Bounds],
) -> None:
    testcase.assertEqual(set(expected), set(actual))
    for layer_id in sorted(expected):
        for side in ("lb", "ub"):
            expected_tensor = getattr(expected[layer_id], side).contiguous()
            actual_tensor = getattr(actual[layer_id], side).contiguous()
            testcase.assertEqual(expected_tensor.dtype, DTYPE)
            testcase.assertEqual(actual_tensor.dtype, DTYPE)
            testcase.assertEqual(expected_tensor.shape, actual_tensor.shape)
            testcase.assertTrue(
                torch.equal(
                    expected_tensor.view(torch.int64),
                    actual_tensor.view(torch.int64),
                ),
                f"layer {layer_id} {side} changed at the bit level",
            )


class ForwardStateLivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_device("cpu", "float64")

    def _run_bitwise_gate(
        self,
        net: Net,
        *,
        alphas: dict[int, torch.Tensor] | None = None,
    ) -> list[tuple[int, tuple[int, ...], frozenset[int]]]:
        lower = torch.tensor([[-0.75, -0.5]], dtype=DTYPE, device=DEVICE)
        upper = torch.tensor([[0.875, 1.0]], dtype=DTYPE, device=DEVICE)

        # Replacing only the release hook with a no-op reproduces the previous
        # retain-all lifecycle while executing the identical arithmetic path.
        with mock.patch.object(
            tf_forward,
            "_release_consumed_forward_state",
            return_value=None,
        ):
            baseline = tf_forward.compute_forward_bounds(
                net, lower, upper, alphas=alphas
            )

        release = tf_forward._release_consumed_forward_state
        events: list[tuple[int, tuple[int, ...], frozenset[int]]] = []

        def tracked_release(
            layer_id: int,
            preds: list[int],
            remaining_consumers: dict[int, int],
            box_state: dict[int, Bounds],
            lin_state: dict[int, tf_forward.LinearBound],
            frame_dict: dict[int, tf_forward.Frame],
        ) -> None:
            before = set(box_state)
            release(
                layer_id,
                preds,
                remaining_consumers,
                box_state,
                lin_state,
                frame_dict,
            )
            self.assertEqual(set(box_state), set(lin_state))
            self.assertEqual(set(box_state), set(frame_dict))
            events.append(
                (layer_id, tuple(preds), frozenset(before - set(box_state)))
            )

        with mock.patch.object(
            tf_forward,
            "_release_consumed_forward_state",
            side_effect=tracked_release,
        ):
            reclaimed = tf_forward.compute_forward_bounds(
                net, lower, upper, alphas=alphas
            )

        _assert_bounds_bitwise_equal(self, baseline, reclaimed)
        released_ids = [
            released_id
            for _, _, released in events
            for released_id in released
        ]
        self.assertCountEqual(released_ids, [layer.id for layer in net.layers])
        return events

    def test_chain_matches_retain_all_baseline_bitwise(self) -> None:
        events = self._run_bitwise_gate(_chain_net())
        self.assertEqual(events[2][2], frozenset({1}))
        self.assertEqual(events[-1][2], frozenset({4, 5}))

    def test_explicit_relu_alpha_path_matches_bitwise(self) -> None:
        alpha = torch.tensor(
            [[0.125, 0.5, 0.875]], dtype=DTYPE, device=DEVICE
        )
        events = self._run_bitwise_gate(
            _chain_net(), alphas={3: alpha}
        )
        relu_event = next(event for event in events if event[0] == 3)
        self.assertEqual(relu_event[2], frozenset({2}))

    def test_residual_fanout_waits_for_add_consumers_bitwise(self) -> None:
        events = self._run_bitwise_gate(_residual_net())
        stem_release_events = [
            layer_id for layer_id, _, released in events if 2 in released
        ]
        self.assertEqual(len(stem_release_events), 1)
        self.assertIn(stem_release_events[0], {3, 6})
        add_event = next(event for event in events if event[0] == 7)
        self.assertEqual(add_event[2], frozenset({5, 6}))

    def test_repeated_predecessor_and_flatten_alias_are_counted_once(self) -> None:
        events = self._run_bitwise_gate(_repeated_alias_net())
        flatten_event = next(event for event in events if event[0] == 3)
        self.assertEqual(flatten_event[2], frozenset({2}))
        repeated_add_event = next(event for event in events if event[0] == 4)
        self.assertEqual(repeated_add_event[1], (3, 3))
        self.assertEqual(repeated_add_event[2], frozenset({3}))


if __name__ == "__main__":
    unittest.main()
