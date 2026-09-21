"""Bounded saved-input conversion diagnosis; no solver, export or gate change."""
import argparse
import copy
import json
from pathlib import Path
import sys

from metamoe_component_control import validate_freeze
from recent_moe_deployment import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    validate_freeze(cfg)
    item = cfg["requests"][1]
    if item["id"] != "mnist_rt_index0":
        raise ValueError("not the fixed failed request")
    work = Path(cfg["output_root"]) / item["id"] / "work"
    for path in [Path(cfg["author_repo"]),
                 Path(cfg["author_repo"]) / "src/Vision_Transformer_Pytorch",
                 Path(cfg["author_repo"]) / "src/Formal_Neural_Network_Verification/alpha-beta-crown"]:
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from export_to_abcrown import SimplifiedWrapper
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = torch.load(Path(cfg["author_repo"]) / item["checkpoint"],
                       map_location="cpu", weights_only=False).eval().model
    x = torch.from_numpy(np.load(work / "input.npy", allow_pickle=False))
    paths = list((work / "onnx").glob("*.onnx"))
    if len(paths) != 1:
        raise ValueError("ambiguous saved graph")
    folded32 = SimplifiedWrapper(copy.deepcopy(model)).eval()
    source64 = copy.deepcopy(model).double()
    folded64 = SimplifiedWrapper(copy.deepcopy(source64)).eval().double()
    init = {v.name: onnx.numpy_helper.to_array(v)
            for v in onnx.load(paths[0]).graph.initializer}
    compared = []
    for name, value in folded32.state_dict().items():
        if name not in init or not np.array_equal(value.detach().numpy(), init[name]):
            raise ValueError(f"stored folded parameter differs: {name}")
        compared.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(paths[0]), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    onnx_y = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    with torch.no_grad():
        outputs = {"original32": model(x).numpy(), "folded32": folded32(x).numpy(),
                   "original64": source64(x.double()).numpy(),
                   "recomputed_fold64": folded64(x.double()).numpy(), "saved_onnx32": onnx_y}
        layer_rows = []
        v = x
        for i in range(1, 5):
            original = getattr(model, f"bn{i}")(getattr(model, f"conv{i}")(v))
            folded = getattr(folded32, f"conv{i}")(v)
            layer_rows.append({"block": i, "shared_original_activation_max_error":
                               float((original - folded).abs().max())})
            v = original.relu()
            if i < 4:
                v = getattr(model, f"pool{i}")(v)
    pairs = [("original32", "folded32"), ("folded32", "saved_onnx32"),
             ("original32", "saved_onnx32"), ("original64", "recomputed_fold64"),
             ("original32", "original64"), ("folded32", "recomputed_fold64")]
    errors = {f"{u}_vs_{v}": float(np.max(np.abs(outputs[u] - outputs[v]))) for u, v in pairs}
    if not all(np.isfinite(v) for v in errors.values()):
        raise ValueError("nonfinite diagnostic")
    print(json.dumps({"config_sha256": sha256(a.config), "request": item,
                      "input_sha256": sha256(work / "input.npy"), "onnx_sha256": sha256(paths[0]),
                      "parameter_names_checked_equal": compared, "errors": errors,
                      "layers": layer_rows, "logits": {k: v.tolist() for k, v in outputs.items()},
                      "frozen_conformance_tolerance": cfg["conformance_tolerance"],
                      "production_dtype_and_acceptance_unchanged": True,
                      "scope": "one saved failed input; finite arithmetic diagnosis, not a proof or retry"}, indent=2))


if __name__ == "__main__":
    main()
