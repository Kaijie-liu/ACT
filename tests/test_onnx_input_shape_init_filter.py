"""Regression test: get_onnx_input_shape must skip initializers.

History: collins_rul_cnn_2022 full-62 run (2026-05-25) yielded 19/19
ERROR_RuntimeError on NN_rul_full_window_40.onnx after the tf_conv2d
non-square fix landed. Root cause: get_onnx_input_shape returned
``conv_1_W`` shape (the first entry in graph.input) instead of the
true model input ``imageinput``. Older ONNX exporters list weight
initializers alongside placeholders in graph.input, so picking
``graph.input[0]`` blindly mis-identifies the input.

Fix: filter out tensors whose name appears in graph.initializer.

This test builds a tiny ONNX model with an initializer that comes
BEFORE the real input in graph.input order, and asserts the helper
returns the real input shape.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class TestGetOnnxInputShapeSkipsInitializers(unittest.TestCase):
    def setUp(self):
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np

        # Tiny model: imageinput (1,1,40,20) → Conv (1,1,40,20) → output
        weight = numpy_helper.from_array(
            np.zeros((1, 1, 1, 1), dtype=np.float32), name="conv_W"
        )
        # Order matters here: initializer first, then real input. The buggy
        # code path took graph.input[0] and returned conv_W's (1,1,1,1).
        weight_ti = helper.make_tensor_value_info(
            "conv_W", TensorProto.FLOAT, (1, 1, 1, 1)
        )
        image_ti = helper.make_tensor_value_info(
            "imageinput", TensorProto.FLOAT, (1, 1, 40, 20)
        )
        out_ti = helper.make_tensor_value_info(
            "out", TensorProto.FLOAT, (1, 1, 40, 20)
        )
        conv_node = helper.make_node(
            "Conv",
            inputs=["imageinput", "conv_W"],
            outputs=["out"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
        )
        graph = helper.make_graph(
            nodes=[conv_node],
            name="g",
            inputs=[weight_ti, image_ti],   # weight listed BEFORE imageinput
            outputs=[out_ti],
            initializer=[weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.onnx_path = self.tmp / "init_first.onnx"
        onnx.save(model, str(self.onnx_path))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_returns_imageinput_not_initializer(self):
        from act.front_end.vnnlib_loader.onnx_converter import get_onnx_input_shape
        shape = get_onnx_input_shape(self.onnx_path)
        # The buggy version returned (1,1,1,1) (conv_W). The fix must return
        # the real model input.
        self.assertEqual(
            tuple(shape), (1, 1, 40, 20),
            f"helper picked an initializer instead of the model input; got {shape}"
        )

    def test_real_collins_full_window_40(self):
        """Smoke against the real collins model that surfaced the bug."""
        from act.front_end.vnnlib_loader.onnx_converter import get_onnx_input_shape
        p = Path(
            "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/"
            "collins_rul_cnn_2022/onnx/NN_rul_full_window_40.onnx"
        )
        if not p.exists():
            self.skipTest("collins_rul_cnn_2022 ONNX not present")
        shape = get_onnx_input_shape(p)
        self.assertEqual(tuple(shape), (1, 1, 40, 20))


if __name__ == "__main__":
    unittest.main(verbosity=2)
