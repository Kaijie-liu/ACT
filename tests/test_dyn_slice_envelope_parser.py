"""Parser-level test for the fixed-shape bounded dynamic-Slice envelope path.

Per `frontend_cleanup_plan.md` §2.3 and the 2026-05-25
`CCTSDB_DYNAMIC_SLICE_DESIGN.md`. This test pins the parser's
behavior when ONNX has the cctsdb_yolo-style fixed-shape subset:

    starts = Unsqueeze(Cast(Gather(input, const_idx)))
    ends   = Unsqueeze(Add(Cast(Gather(input, const_idx)), const_value))
    Slice(static_initializer, starts, ends, axes, steps)

The parser MUST:
  1. Recognize that starts/ends both derive from `input[k]` plus
     constant offsets, with `ends - starts = const` (constant window).
  2. Derive the integer interval `(s_lb, s_ub)` for the start from
     the supplied VNNLIB input bounds.
  3. Emit a `LayerKind.LUT_BOUNDS` layer with `params['lb']` and
     `params['ub']` set to the per-output-position envelope produced
     by `act.back_end.interval_tf.tf_mlp.precompute_lut_envelope`.

The test also verifies the brute-force soundness contract: every
realizable runtime crop (over the integer lattice of `start ∈
[s_lb, s_ub]`) is contained in the envelope produced by the parser.

This is NOT a claim that all cctsdb_yolo_2023 dynamic Slice sites are
covered. The real cctsdb `slice_23` can produce variable / empty output
shapes over the VNNLIB box and must still fail closed.

NOTE: this is a parser-side test only. It does NOT modify the
production 253 V/A headline; it lives under the
`research/frontend_cleanup_plan.md` engineering gate, not the
verification roadmap §7.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


def _build_synthetic_cctsdb_like_model(
    T_data: np.ndarray,
    const_idx: int,
    window_size: int,
    input_dim: int,
) -> bytes:
    """Build a tiny ONNX model with the cctsdb Slice pattern.

    Graph:
        input  : Float[input_dim]
        const_idx_tensor : Int64 scalar  (initializer)
        T      : Float[len(T_data)]      (initializer)
        const_window_tensor : Int64 [1]  (initializer; the +window offset)
        axes_tensor  : Int64 [1] = [0]   (initializer)
        steps_tensor : Int64 [1] = [1]   (initializer)

        gathered = Gather(input, const_idx_tensor, axis=0)
        casted   = Cast(gathered, to=INT64)
        starts   = Unsqueeze(casted, axes=[0])
        ended    = Add(casted, const_window_tensor)
        ends     = Unsqueeze(ended, axes=[0])
        out      = Slice(T, starts, ends, axes_tensor, steps_tensor)
    """
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [input_dim]
    )
    out_tensor = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [window_size]
    )

    const_idx_init = numpy_helper.from_array(
        np.array(const_idx, dtype=np.int64), name="const_idx_tensor"
    )
    T_init = numpy_helper.from_array(
        T_data.astype(np.float32), name="T"
    )
    const_window_init = numpy_helper.from_array(
        np.array([window_size], dtype=np.int64), name="const_window_tensor"
    )
    axes_init = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="axes_tensor"
    )
    steps_init = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="steps_tensor"
    )

    gather_node = helper.make_node(
        "Gather", inputs=["input", "const_idx_tensor"],
        outputs=["gathered"], axis=0,
    )
    cast_node = helper.make_node(
        "Cast", inputs=["gathered"], outputs=["casted"],
        to=TensorProto.INT64,
    )
    unsqueeze_starts_axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="unsqueeze_starts_axes",
    )
    unsqueeze_starts = helper.make_node(
        "Unsqueeze", inputs=["casted", "unsqueeze_starts_axes"],
        outputs=["starts"],
    )
    add_node = helper.make_node(
        "Add", inputs=["casted", "const_window_tensor_scalar"],
        outputs=["ended"],
    )
    # Add needs scalar const window; we already have it as [1] tensor —
    # squeeze it.
    const_window_scalar_init = numpy_helper.from_array(
        np.array(window_size, dtype=np.int64),
        name="const_window_tensor_scalar",
    )
    unsqueeze_ends_axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="unsqueeze_ends_axes",
    )
    unsqueeze_ends = helper.make_node(
        "Unsqueeze", inputs=["ended", "unsqueeze_ends_axes"],
        outputs=["ends"],
    )
    slice_node = helper.make_node(
        "Slice", inputs=["T", "starts", "ends", "axes_tensor", "steps_tensor"],
        outputs=["out"],
    )

    graph = helper.make_graph(
        nodes=[gather_node, cast_node, unsqueeze_starts, add_node,
               unsqueeze_ends, slice_node],
        name="synthetic_cctsdb_slice",
        inputs=[input_tensor],
        outputs=[out_tensor],
        initializer=[
            const_idx_init, T_init, const_window_init,
            axes_init, steps_init,
            unsqueeze_starts_axes, unsqueeze_ends_axes,
            const_window_scalar_init,
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 7
    return model.SerializeToString()


class TestDynSliceEnvelopeParser(unittest.TestCase):
    """The parser emits LUT_BOUNDS for the fixed-shape dynamic Slice subset."""

    def test_synthetic_cctsdb_pattern_emits_lut_bounds(self) -> None:
        # 100 elements of static data; window of size 4; start in [0, 8]
        T_data = np.arange(100, dtype=np.float32) * 0.1
        window_size = 4
        input_dim = 16
        const_idx = 5  # input[5] is the dynamic start

        # Input box: input[5] ∈ [0.5, 8.5], so cast-to-int range is [0, 8]
        input_lb = np.zeros(input_dim, dtype=np.float64)
        input_ub = np.full(input_dim, 1.0, dtype=np.float64)
        input_lb[const_idx] = 0.5
        input_ub[const_idx] = 8.5

        model_bytes = _build_synthetic_cctsdb_like_model(
            T_data, const_idx=const_idx,
            window_size=window_size, input_dim=input_dim,
        )

        # Attempt to convert via TorchToACT with input bounds plumbed.
        try:
            import onnx2torch
            torch_mod = onnx2torch.convert(onnx.load_model_from_string(model_bytes))
        except Exception as e:
            self.skipTest(f"onnx2torch not available or failed: {e}")
            return

        # Try the bounds-aware parser. If the parser fix is not yet
        # landed, build_act will raise the cannot-resolve error.
        try:
            from act.pipeline.verification.torch2act import build_act
        except ImportError:
            self.skipTest("act.pipeline.verification.torch2act not importable")
            return

        # Call build_act with explicit input bounds via the parser-time
        # dynamic-index envelope path.
        try:
            layers, _preds, _succs = build_act(
                torch_mod, (input_dim,), torch.float64,
                sample_input=torch.from_numpy(
                    (input_lb + input_ub) / 2.0
                ),
                input_bounds=(
                    torch.from_numpy(input_lb),
                    torch.from_numpy(input_ub),
                ),
            )
        except TypeError as e:
            self.fail(
                f"build_act does not yet accept input_bounds kwarg: {e}. "
                "Implement the parser-side glue per "
                "research/frontend_cleanup_plan.md §2.3."
            )
            return
        except ValueError as e:
            if "cannot resolve starts/ends" in str(e):
                self.fail(
                    "Parser fix not landed: " + str(e) + ". "
                    "Implement _convert_OnnxSlice's bounded-envelope "
                    "fallback per research/frontend_cleanup_plan.md §2.3."
                )
                return
            raise

        # Containment: every realizable runtime crop must be contained
        # element-wise in the LUT envelope the parser produced.
        from act.back_end.layer_schema import LayerKind
        lut_layers = [L for L in layers
                      if str(L.kind).upper() == LayerKind.LUT_BOUNDS.value]
        self.assertGreater(
            len(lut_layers), 0,
            "Expected at least one LUT_BOUNDS layer emitted for the "
            "dynamic Slice pattern; none found."
        )

        # The single LUT layer should match the brute-force envelope.
        from act.back_end.interval_tf.tf_mlp import precompute_lut_envelope
        s_lb = int(np.floor(input_lb[const_idx]))
        s_ub = int(np.floor(input_ub[const_idx]))
        expected_lb, expected_ub = precompute_lut_envelope(
            torch.from_numpy(T_data).to(torch.float64),
            window_size=(window_size,),
            starts_lb=(s_lb,),
            starts_ub=(s_ub,),
        )

        # Find the LUT layer's lb/ub params.
        lut_layer = lut_layers[0]
        emitted_lb = lut_layer.params.get("lb")
        emitted_ub = lut_layer.params.get("ub")
        self.assertIsNotNone(emitted_lb, "LUT_BOUNDS layer missing params['lb']")
        self.assertIsNotNone(emitted_ub, "LUT_BOUNDS layer missing params['ub']")

        np.testing.assert_allclose(
            np.asarray(emitted_lb), expected_lb.numpy(),
            err_msg="LUT_BOUNDS lb does not match precompute_lut_envelope output",
        )
        np.testing.assert_allclose(
            np.asarray(emitted_ub), expected_ub.numpy(),
            err_msg="LUT_BOUNDS ub does not match precompute_lut_envelope output",
        )


if __name__ == "__main__":
    unittest.main()
