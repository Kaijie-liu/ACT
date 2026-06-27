# ===- act/pipeline/hybridz_projected_utils.py - Projected HZ helpers --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Package-local helpers for projected exact-HZ portfolio branches."""

from __future__ import annotations

import time
from pathlib import Path


def _resolve_downloaded_instance(bench: str, iid: int) -> tuple[Path, Path]:
    from act.front_end.vnnlib_loader.data_model_loader import list_downloaded_pairs

    matches = [
        item
        for item in list_downloaded_pairs()
        if str(item.get("category")) == str(bench)
        and int(item.get("index", -1)) == int(iid)
    ]
    if not matches:
        raise FileNotFoundError(f"no downloaded VNNLIB instance for {bench} iid={iid}")
    paths = matches[0].get("paths", {})
    onnx_path = Path(str(paths.get("onnx", "")))
    vnnlib_path = Path(str(paths.get("vnnlib", "")))
    if not onnx_path.exists() or not vnnlib_path.exists():
        raise FileNotFoundError(
            f"downloaded VNNLIB paths are missing for {bench} iid={iid}: "
            f"onnx={onnx_path} vnnlib={vnnlib_path}"
        )
    return onnx_path, vnnlib_path


def build_net_and_interval(bench: str, iid: int, device: str):
    """Load one VNNLIB row and run interval propagation through ACT.

    This mirrors the former research-script helper, but uses the main VNNLIB
    root resolver so package code does not depend on local research scripts or
    a hard-coded benchmark checkout.
    """

    import torch

    from act.back_end.core import Bounds as ABounds
    from act.back_end.transfer_functions import (
        get_transfer_function,
        set_transfer_function_mode,
    )
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.verifiable_model import (
        InputLayer,
        InputSpecLayer,
        OutputSpecLayer,
        VerifiableModel,
    )
    from act.front_end.vnnlib_loader.onnx_converter import (
        convert_onnx_to_pytorch,
        get_onnx_input_shape,
    )
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
    from act.pipeline.verification.torch2act import TorchToACT

    onnx_path, vnnlib_path = _resolve_downloaded_instance(bench, iid)
    input_shape = tuple(get_onnx_input_shape(onnx_path))
    pt = convert_onnx_to_pytorch(onnx_path).float().eval()
    lab = LabeledInputTensor(
        tensor=torch.zeros(input_shape, dtype=torch.float32),
        label=torch.tensor([0]),
    )
    queries = parse_vnnlib_queries(vnnlib_path, labeled_tensor=lab)
    if not queries:
        raise RuntimeError(f"no parsed VNNLIB queries for {bench} iid={iid}")

    wrapped = VerifiableModel(
        input_layer=InputLayer(labeled_input=lab, shape=input_shape, dtype=torch.float32),
        input_spec=InputSpecLayer(queries[0][0]),
        model=pt,
        output_spec=OutputSpecLayer(queries[0][1]),
    )
    net = TorchToACT(wrapped).run()

    lb = queries[0][0].lb.detach().cpu().reshape(1, -1).to(torch.float32)
    ub = queries[0][0].ub.detach().cpu().reshape(1, -1).to(torch.float32)
    if device == "cuda" and torch.cuda.is_available():
        lb = lb.cuda()
        ub = ub.cuda()
    ib = ABounds(lb=lb, ub=ub)

    before = {}
    after = {}
    set_transfer_function_mode("interval")
    tf = get_transfer_function()
    started = time.time()
    for layer in net.layers:
        preds = net.preds.get(layer.id, [])
        inb = ib if (layer.id == 0 or not preds) else after[preds[0]].bounds
        before[layer.id] = inb
        after[layer.id] = tf.apply(layer, inb, net, before, after)
    return onnx_path, vnnlib_path, input_shape, queries, net, before, after, time.time() - started


__all__ = [
    "build_net_and_interval",
]
