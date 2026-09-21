"""AvgPool conversion controls only; no certification or solver calls."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-work", type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from onnx2pytorch import ConvertModel
    from onnx import helper, TensorProto
    torch.set_num_threads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    results = []
    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    for padding in [0, 1]:
        for include in [0, 1]:
            node = helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[2, 2],
                                    strides=[1, 1], pads=[padding] * 4, count_include_pad=include)
            size = 3 + 2 * padding
            graph = helper.make_graph([node], "pool", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
                                       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, size, size])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 9
            converted = ConvertModel(model).eval()
            actual = converted(x).detach().numpy()
            session = ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                            providers=["CPUExecutionProvider"])
            expected = session.run(None, {"x": x.numpy()})[0]
            if not np.array_equal(actual, expected):
                raise ValueError("AvgPool ONNX/PyTorch mismatch")
            results.append({"padding": padding, "count_include_pad": include, "exact_probe_match": True})
    paths = list((a.old_work / "onnx").glob("*.onnx"))
    if len(paths) != 1:
        raise ValueError("ambiguous original failed graph")
    saved = onnx.load(paths[0])
    converted = ConvertModel(saved).eval()
    real_x = np.load(a.old_work / "input.npy", allow_pickle=False)
    session = ort.InferenceSession(str(paths[0]), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        actual = converted(torch.from_numpy(real_x)).numpy()
    expected = session.run(None, {session.get_inputs()[0].name: real_x})[0]
    error = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(error) or error >= 1e-4:
        raise ValueError("original graph conversion conformance failed")
    print(json.dumps({"pool_controls": results, "old_failed_graph_conversion_max_error": error,
                      "scope": "finite compatibility probes, not full-domain equivalence or a certificate"}, indent=2))


if __name__ == "__main__":
    main()
