"""GPU constraint-erased shadow propagation for HZ-style states (N001 probe).

This is an isolated diagnostic, not a verifier.  It propagates the
constraint-erased *shadow* of an exact Hybrid-Zonotope state,

    sigma(H) = { c + G xi : xi in [-1, 1]^k },

through an ONNX graph made of Conv / BatchNormalization / Relu / Add /
Flatten / Gemm nodes.  The exact HZ keeps additional equality/inequality
predicates and binary phase factors; erasing them can only enlarge the set,
so every bound computed on the shadow is a sound bound for the exact HZ
(modulo floating rounding, which this probe does NOT yet control).

Three ReLU parametrisations of the *same* exact HZ ReLU graph are compared:

* ``fresh``  -- the encoding used by the current ACT sparse HZ
  (tf_mlp.sparse_hz_apply_relu_exact): an unstable output is
  ``u/2 - (u/2) xi_2`` with a fresh factor, so its shadow is the box
  ``[0, u]`` and loses all correlation with the pre-activation.
* ``aligned`` -- the projection-aligned encoding proposed in N001:
  ``y = lam x + mu + mu eta`` with ``lam = u / (u - l)``, ``mu = -lam l / 2``.
  Its shadow equals the DeepZ ReLU image; exactness is restored by four
  predicate rows over (x, y, delta) that are not used by the shadow.
* ``interval`` -- plain interval arithmetic (no generators at all).

Generator rows are indexed by a global factor counter so that residual
branches share the latent identity of their common ancestor.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper


@dataclasses.dataclass
class Zono:
    """Shadow zonotope: center c [*S] and generators G [k, *S]."""

    c: torch.Tensor
    G: torch.Tensor  # rows = global factor ids 0..k-1 (zero padded)

    @property
    def k(self) -> int:
        return int(self.G.shape[0])

    def radius(self) -> torch.Tensor:
        return self.G.abs().sum(dim=0)

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius()
        return self.c - r, self.c + r

    def pad_to(self, k: int) -> "Zono":
        if k == self.k:
            return self
        if k < self.k:
            raise ValueError("cannot shrink factor space")
        pad = torch.zeros((k - self.k,) + tuple(self.G.shape[1:]),
                          dtype=self.G.dtype, device=self.G.device)
        return Zono(self.c, torch.cat([self.G, pad], dim=0))


@dataclasses.dataclass
class Box:
    lb: torch.Tensor
    ub: torch.Tensor


class Graph:
    def __init__(self, path: str, device: str, dtype: torch.dtype):
        model = onnx.load(path)
        self.inits: Dict[str, torch.Tensor] = {}
        for t in model.graph.initializer:
            self.inits[t.name] = torch.as_tensor(
                numpy_helper.to_array(t).copy()).to(device=device, dtype=dtype)
        self.nodes = list(model.graph.node)
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name
        self.device = device
        self.dtype = dtype

    @staticmethod
    def attrs(node) -> Dict[str, object]:
        out = {}
        for a in node.attribute:
            out[a.name] = onnx.helper.get_attribute_value(a)
        return out


@dataclasses.dataclass
class ReluStat:
    name: str
    n: int
    stable_neg: int
    stable_pos: int
    unstable: int


def _conv_params(node, graph: Graph):
    a = Graph.attrs(node)
    W = graph.inits[node.input[1]]
    b = graph.inits[node.input[2]] if len(node.input) > 2 else None
    pads = list(a.get("pads", [0, 0, 0, 0]))
    if pads[0] != pads[2] or pads[1] != pads[3]:
        raise NotImplementedError("asymmetric padding")
    return dict(weight=W, bias=b, stride=tuple(a.get("strides", [1, 1])),
                padding=(pads[0], pads[1]),
                dilation=tuple(a.get("dilations", [1, 1])),
                groups=int(a.get("group", 1)))


def _bn_params(node, graph: Graph):
    a = Graph.attrs(node)
    eps = float(a.get("epsilon", 1e-5))
    gamma, beta, mean, var = (graph.inits[n] for n in node.input[1:5])
    scale = gamma / torch.sqrt(var + eps)
    shift = beta - mean * scale
    return scale, shift


def _chunked_conv(G: torch.Tensor, p, chunk: int) -> torch.Tensor:
    outs = []
    for s in range(0, G.shape[0], chunk):
        outs.append(F.conv2d(G[s:s + chunk], p["weight"], None, p["stride"],
                             p["padding"], p["dilation"], p["groups"]))
    return torch.cat(outs, dim=0)


def propagate(graph: Graph, lb: torch.Tensor, ub: torch.Tensor, mode: str,
              chunk: int = 512):
    """Propagate an input box through the graph.

    lb/ub have the model input shape *without* batch (e.g. [3, 32, 32]).
    Returns (output state, list of ReluStat).
    """
    env: Dict[str, object] = {}
    stats: List[ReluStat] = []
    k_global = 0
    if mode == "interval":
        env[graph.input_name] = Box(lb, ub)
    else:
        c = (lb + ub) / 2
        r = (ub - lb) / 2
        nz = torch.nonzero(r.reshape(-1) > 0).reshape(-1)
        k_global = int(nz.numel())
        G = torch.zeros((k_global, r.numel()), dtype=lb.dtype, device=lb.device)
        G[torch.arange(k_global, device=lb.device), nz] = r.reshape(-1)[nz]
        env[graph.input_name] = Zono(c.unsqueeze(0), G.reshape((k_global, 1) + tuple(lb.shape)))
    for node in graph.nodes:
        op = node.op_type
        x = env[node.input[0]] if node.input[0] in env else None
        if op == "Conv":
            p = _conv_params(node, graph)
            if isinstance(x, Box):
                Wp, Wn = p["weight"].clamp(min=0), p["weight"].clamp(max=0)
                conv = lambda t, w: F.conv2d(t, w, None, p["stride"], p["padding"], p["dilation"], p["groups"])
                lo = conv(x.lb, Wp) + conv(x.ub, Wn)
                hi = conv(x.ub, Wp) + conv(x.lb, Wn)
                if p["bias"] is not None:
                    bb = p["bias"].view(1, -1, 1, 1)
                    lo, hi = lo + bb, hi + bb
                env[node.output[0]] = Box(lo, hi)
            else:
                cc = F.conv2d(x.c, p["weight"], p["bias"], p["stride"], p["padding"], p["dilation"], p["groups"])
                GG = _chunked_conv(x.G.reshape((x.k,) + tuple(x.c.shape[1:])), p, chunk)
                env[node.output[0]] = Zono(cc, GG.unsqueeze(1))
        elif op == "BatchNormalization":
            scale, shift = _bn_params(node, graph)
            shp = (1, -1, 1, 1)
            if isinstance(x, Box):
                s = scale.view(shp)
                lo = torch.minimum(x.lb * s, x.ub * s) + shift.view(shp)
                hi = torch.maximum(x.lb * s, x.ub * s) + shift.view(shp)
                env[node.output[0]] = Box(lo, hi)
            else:
                env[node.output[0]] = Zono(x.c * scale.view(shp) + shift.view(shp),
                                           x.G * scale.view((1,) + shp))
        elif op == "Relu":
            if isinstance(x, Box):
                lo, hi = x.lb.clamp(min=0), x.ub.clamp(min=0)
                n = lo.numel()
                stats.append(ReluStat(node.name, n, int((x.ub <= 0).sum()), int((x.lb >= 0).sum()),
                                      int(((x.lb < 0) & (x.ub > 0)).sum())))
                env[node.output[0]] = Box(lo, hi)
            else:
                l, u = x.bounds()
                neg = u <= 0
                pos = l >= 0
                unst = ~(neg | pos)
                stats.append(ReluStat(node.name, int(l.numel()), int(neg.sum()), int(pos.sum()), int(unst.sum())))
                ku = int(unst.sum())
                if mode == "aligned":
                    lam = torch.where(pos, torch.ones_like(u), torch.zeros_like(u))
                    lam_u = u / (u - l)
                    lam = torch.where(unst, lam_u, lam)
                    mu = torch.where(unst, -lam_u * l / 2, torch.zeros_like(u))
                    cc = x.c * lam + mu
                    GG = x.G * lam.unsqueeze(0)
                    newg_val = mu
                elif mode == "fresh":
                    keep = pos.to(x.c.dtype)
                    cc = x.c * keep + torch.where(unst, u / 2, torch.zeros_like(u))
                    GG = x.G * keep.unsqueeze(0)
                    newg_val = torch.where(unst, u / 2, torch.zeros_like(u))
                else:
                    raise ValueError(mode)
                idx = torch.nonzero(unst.reshape(-1)).reshape(-1)
                newG = torch.zeros((ku, l.numel()), dtype=x.G.dtype, device=x.G.device)
                newG[torch.arange(ku, device=x.G.device), idx] = newg_val.reshape(-1)[idx]
                GG = torch.cat([GG, newG.reshape((ku,) + tuple(GG.shape[1:]))], dim=0)
                # global factor ids: the existing rows are 0..x.k-1, new rows
                # must start at k_global; pad x first if another branch grew.
                if x.k != k_global:
                    pad = torch.zeros((k_global - x.k,) + tuple(GG.shape[1:]), dtype=GG.dtype, device=GG.device)
                    GG = torch.cat([GG[:x.k], pad, GG[x.k:]], dim=0)
                k_global += ku
                env[node.output[0]] = Zono(cc, GG)
        elif op == "Add":
            y = env[node.input[1]] if node.input[1] in env else graph.inits.get(node.input[1])
            if isinstance(x, Box):
                if isinstance(y, Box):
                    env[node.output[0]] = Box(x.lb + y.lb, x.ub + y.ub)
                else:
                    env[node.output[0]] = Box(x.lb + y, x.ub + y)
            else:
                if isinstance(y, Zono):
                    k = max(x.k, y.k)
                    xa, ya = x.pad_to(k), y.pad_to(k)
                    env[node.output[0]] = Zono(xa.c + ya.c, xa.G + ya.G)
                else:
                    env[node.output[0]] = Zono(x.c + y, x.G)
        elif op == "Flatten":
            if isinstance(x, Box):
                env[node.output[0]] = Box(x.lb.reshape(1, -1), x.ub.reshape(1, -1))
            else:
                env[node.output[0]] = Zono(x.c.reshape(1, -1), x.G.reshape(x.k, 1, -1))
        elif op == "Gemm":
            a = Graph.attrs(node)
            W = graph.inits[node.input[1]]
            b = graph.inits[node.input[2]] if len(node.input) > 2 else None
            if int(a.get("transB", 0)):
                W = W.t()
            alpha = float(a.get("alpha", 1.0)); beta = float(a.get("beta", 1.0))
            if isinstance(x, Box):
                Wp, Wn = W.clamp(min=0), W.clamp(max=0)
                lo = alpha * (x.lb @ Wp + x.ub @ Wn); hi = alpha * (x.ub @ Wp + x.lb @ Wn)
                if b is not None:
                    lo, hi = lo + beta * b, hi + beta * b
                env[node.output[0]] = Box(lo, hi)
            else:
                cc = alpha * (x.c @ W) + (beta * b if b is not None else 0)
                GG = alpha * (x.G @ W)
                env[node.output[0]] = Zono(cc, GG)
        else:
            raise NotImplementedError(op)
    return env[graph.output_name], stats


_NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)"


def parse_vnnlib_box_top1(path: str, n_in: int):
    """Parse a VNNLIB file with an input box and a disjunctive 'some Y_j >= Y_t'
    unsafe region (the cifar100/tinyimagenet 2024 form)."""
    text = open(path).read()
    lb = np.full(n_in, np.nan)
    ub = np.full(n_in, np.nan)
    for m in re.finditer(r"\(assert\s*\(<=\s*X_(\d+)\s+(" + _NUM + r")\)\)", text):
        ub[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"\(assert\s*\(>=\s*X_(\d+)\s+(" + _NUM + r")\)\)", text):
        lb[int(m.group(1))] = float(m.group(2))
    if np.isnan(lb).any() or np.isnan(ub).any():
        raise ValueError("incomplete input box")
    pairs = re.findall(r"\(>=\s*Y_(\d+)\s+Y_(\d+)\)", text)
    pairs += [(b, a) for a, b in re.findall(r"\(<=\s*Y_(\d+)\s+Y_(\d+)\)", text)]
    trues = {int(t) for _, t in pairs}
    if len(trues) != 1:
        raise ValueError(f"unexpected property form: {trues}")
    t = trues.pop()
    others = sorted({int(j) for j, _ in pairs})
    return lb, ub, t, others


def margins_lower(out, t: int, others: List[int]) -> torch.Tensor:
    """Sound lower bounds of Y_t - Y_j over the shadow (exact arithmetic aside)."""
    if isinstance(out, Box):
        return out.lb.reshape(-1)[t] - out.ub.reshape(-1)[others]
    c = out.c.reshape(-1)
    G = out.G.reshape(out.k, -1)
    d_c = c[t] - c[others]
    d_G = G[:, t:t + 1] - G[:, others]
    return d_c - d_G.abs().sum(dim=0)
