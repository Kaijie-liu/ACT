"""Locate the pre-registered ZERO probe failure; no new verification query."""
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
        raise ValueError("not frozen failed request")
    work = Path(cfg["output_root"]) / item["id"] / "work"
    repo = Path(cfg["author_repo"])
    for path in [repo, repo / "src/Vision_Transformer_Pytorch",
                 repo / "src/Formal_Neural_Network_Verification/alpha-beta-crown"]:
        sys.path.insert(0, str(path))
    import numpy as np
    import torch
    import onnxruntime as ort
    from export_to_abcrown import SimplifiedWrapper
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    original = torch.load(repo / item["checkpoint"], map_location="cpu", weights_only=False).eval().model
    folded = SimplifiedWrapper(copy.deepcopy(original)).eval()
    original64 = copy.deepcopy(original).double()
    folded64 = SimplifiedWrapper(copy.deepcopy(original64)).eval().double()
    x = torch.zeros_like(torch.from_numpy(np.load(work / "input.npy", allow_pickle=False)))
    paths = list((work / "onnx").glob("*.onnx"))
    if len(paths) != 1:
        raise ValueError("ambiguous saved graph")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(paths[0]), sess_options=options, providers=["CPUExecutionProvider"])
    onnx_y = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    with torch.no_grad():
        values = {"original32": original(x).numpy(), "folded32": folded(x).numpy(),
                  "original64": original64(x.double()).numpy(),
                  "folded64": folded64(x.double()).numpy(), "saved_onnx": onnx_y}
    pairs = [("original32", "folded32"), ("folded32", "saved_onnx"),
             ("original32", "saved_onnx"), ("original64", "folded64")]
    errors = {f"{u}_vs_{v}": float(np.max(np.abs(values[u] - values[v]))) for u, v in pairs}
    if not all(np.isfinite(v) for v in errors.values()):
        raise ValueError("nonfinite diagnostic")
    print(json.dumps({"config_sha256": sha256(a.config), "request": item,
                      "onnx_sha256": sha256(paths[0]), "probe": "pre-registered zero tensor",
                      "errors": errors, "logits": {k: v.tolist() for k, v in values.items()},
                      "acceptance_gate_changed": False, "bound_queries": 0,
                      "scope": "finite conversion diagnosis, float64 is diagnostic only"}, indent=2))


if __name__ == "__main__":
    main()
