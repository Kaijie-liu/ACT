"""Neural-HZ v2 rigorous engine n009.1 (consolidated, operator registry).

Semantics, invariant INV and rounding discipline: as nhz_sound.py / nhz_sound_mp.py
(float32 generator storage, float64 arithmetic, exact storage rounding added to the
coordinatewise radius e, gamma pads for every float64 operation).  Operators:

  affine (exact):   MatMul/Gemm with constants, Add/Sub, Mul/Div by constants, Conv,
                    ConvTranspose, BatchNormalization, Flatten/Reshape/Transpose/
                    Squeeze/Unsqueeze/Slice/Concat/Gather-free shape folding, ReduceMean,
                    Upsample/Resize (nearest, integer scales), Identity/Dropout, Pad (const 0)
  ReLU:             projection-aligned exact transformer (phases kept)       [gamma exact]
  Sigmoid/Tanh:     DeepZ shadow + rigorous shifted lines                       [lambda]
  MatMul(state, state): DeepZ-style product with exact radius accounting        [sigma/lambda]
  Softmax:          tangent plane in score differences + interval-Hessian remainder,
                    box rows and sum-to-one rows relaxed by the radii           [sigma/lambda]

Exact-LP polishing of intermediate bounds for small LPs as in n008.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from nhz_engine import _attrs
from nhz_sound import SAZ, SRows, TINY, gamma, optimise_nu, sound_eval
from nhz_sound_mp import map_G, radius64
from nhz_sound_mp2 import rigorous_lines, _sf, _sdf, FPAD
from nhz_sound_mp3 import POLISH_CELLS, POLISH_MAX_NEURONS, _highs_duals

V9_VERSION = "n009.2"  # .2: polishing capped (<=100 unstable per layer, <=600 LPs per propagation, first 25% of the budget)
POLISH_LAYER_MAX = 100
POLISH_TOTAL_MAX = 600
POLISH_BUDGET_SHARE = 0.25
F64 = torch.float64


def _is_state(v):
    return isinstance(v, SAZ)


def _up32(v64: torch.Tensor) -> torch.Tensor:
    v32 = v64.float()
    return torch.where(v32.double() < v64, torch.nextafter(v32, torch.full_like(v32, float("inf"))), v32)


class SoundEngineV9:
    def __init__(self, path: str, device: str = "cuda", polish: bool = True):
        self.deadline = None; self.t_start = None; self.polished = 0
        self.model = onnx.load(path)
        self.device = device; self.polish = polish
        self.consts: Dict[str, torch.Tensor] = {}
        for t in self.model.graph.initializer:
            arr = numpy_helper.to_array(t).copy()
            self.consts[t.name] = torch.as_tensor(arr).to(device=device, dtype=F64 if arr.dtype.kind == "f" else torch.int64)
        init = set(self.consts)
        self.input_name = [i.name for i in self.model.graph.input if i.name not in init][0]
        dims = [d.dim_value for d in [i for i in self.model.graph.input if i.name == self.input_name][0].type.tensor_type.shape.dim]
        self.input_shape = tuple(max(1, d) for d in dims)
        self.output_name = self.model.graph.output[0].name
        self.nodes = list(self.model.graph.node)

    # ------------------------------------------------------------------ helpers
    def _state_from(self, v, shape_like):
        """Lift a constant into a zero-generator state of the same factor count."""
        raise NotImplementedError

    def _bounds(self, fc, fG32, fe, idx, rows: SRows, K: int, iters: int, lr: float, rad_all=None):
        sub = fG32[:, idx]
        rad = radius64(sub) if rad_all is None else rad_all[idx]
        g2 = gamma(K + 2, F64); mag = fc[idx].abs() + rad
        u = fc[idx] + rad + fe[idx] + g2 * mag + TINY * K
        l = fc[idx] - rad - fe[idx] - g2 * mag - TINY * K
        if iters > 0 and rows.n_rows and idx.numel():
            A, b = rows.dense(K); A = A.to(self.device); b = b.to(self.device)
            A32, b32 = A.float(), b.float()
            g = sub.t().contiguous().double(); c = fc[idx]
            for sign in (1.0, -1.0):
                nu = optimise_nu((sign * g).float(), A32, b32, iters, lr, (sign * c).float())
                bd = sound_eval(sign * g, sign * c, A, b, nu) + fe[idx]
                if sign > 0:
                    u = torch.minimum(u, bd)
                else:
                    l = torch.maximum(l, -bd)
            R = rows.n_rows
            import time as _t
            in_budget = self.deadline is None or self.t_start is None or \
                (_t.time() - self.t_start) <= POLISH_BUDGET_SHARE * (self.deadline - self.t_start)
            if (self.polish and idx.numel() <= POLISH_LAYER_MAX and K * R <= POLISH_CELLS
                    and self.polished + 2 * idx.numel() <= POLISH_TOTAL_MAX and in_budget):
                self.polished += 2 * int(idx.numel())
                A64 = A.cpu().numpy(); b64 = b.cpu().numpy(); G = g.cpu().numpy()
                for sign in (1.0, -1.0):
                    nu = torch.as_tensor(_highs_duals(A64, b64, G, sign), device=self.device, dtype=F64)
                    bd = sound_eval(sign * g, sign * c, A, b, nu) + fe[idx]
                    if sign > 0:
                        u = torch.minimum(u, bd)
                    else:
                        l = torch.maximum(l, -bd)
        return l, u

    # ------------------------------------------------------------------ propagate
    def propagate(self, lb, ub, iters: int = 300, lr: float = 0.05, log=None):
        dev = self.device
        lb = torch.as_tensor(lb, dtype=F64).reshape(self.input_shape).to(dev)
        ub = torch.as_tensor(ub, dtype=F64).reshape(self.input_shape).to(dev)
        c0 = (lb + ub) / 2; r0 = (ub - lb) / 2
        flat_r = r0.reshape(-1)
        nz = torch.nonzero(flat_r > 0).reshape(-1); K = int(nz.numel())
        G0 = torch.zeros((K, flat_r.numel()), device=dev, dtype=torch.float32)
        G0[torch.arange(K, device=dev), nz] = _up32(flat_r[nz])
        e0 = gamma(2, F64) * (lb.abs() + ub.abs()) + TINY
        env: Dict[str, object] = {self.input_name: SAZ(c0, G0.reshape((K,) + tuple(c0.shape)), e0)}
        self.input_factor_index = nz
        self.rows = SRows(); self.phases = []; self.K = K; self.iters = iters; self.lr = lr; self.polished = 0
        remaining: Dict[str, int] = {}
        for node in self.nodes:
            for n in node.input:
                if n:
                    remaining[n] = remaining.get(n, 0) + 1
        for node in self.nodes:
            ins = [env.get(n, self.consts.get(n)) if n else None for n in node.input]
            a = _attrs(node)
            h = getattr(self, "op_" + node.op_type, None)
            if h is None:
                raise NotImplementedError(node.op_type)
            out = h(node, ins, a)
            env[node.output[0]] = out
            del ins, out
            for n in node.input:
                if n and n in env and n != self.output_name:
                    remaining[n] -= 1
                    if remaining[n] == 0:
                        del env[n]
        res = env[self.output_name]
        return res.pad_to(self.K), self.rows, self.K, self.phases

    # ------------------------------------------------------------------ constant/shape ops
    def op_Constant(self, node, ins, a):
        arr = numpy_helper.to_array(a["value"]).copy()
        return torch.as_tensor(arr).to(device=self.device, dtype=F64 if arr.dtype.kind == "f" else torch.int64)

    def op_Identity(self, node, ins, a):
        return ins[0]

    op_Dropout = op_Identity

    def op_Cast(self, node, ins, a):
        x = ins[0]
        if _is_state(x):
            return x
        to = int(a.get("to", 1))
        return x.to(F64) if to in (1, 10, 11) else x.to(torch.int64)

    def op_Shape(self, node, ins, a):
        x = ins[0]
        shp = x.c.shape if _is_state(x) else x.shape
        return torch.as_tensor(list(shp), device=self.device, dtype=torch.int64)

    def op_Gather(self, node, ins, a):
        x, idx = ins
        if _is_state(x):
            ax = int(a.get("axis", 0)) % x.c.dim()
            ii = idx.reshape(-1).long()
            c = torch.index_select(x.c, ax, ii); G = torch.index_select(x.G, ax + 1, ii); e = torch.index_select(x.e, ax, ii)
            if idx.dim() == 0:
                c = c.squeeze(ax); G = G.squeeze(ax + 1); e = e.squeeze(ax)
            return SAZ(c, G, e)
        out = torch.index_select(x, int(a.get("axis", 0)), idx.reshape(-1).long())
        return out.reshape(tuple(idx.shape)) if idx.dim() == 0 else out

    def op_Unsqueeze(self, node, ins, a):
        x = ins[0]; axes = a.get("axes") or ins[1].tolist()
        for ax in sorted(int(v) for v in axes):
            if _is_state(x):
                x = SAZ(x.c.unsqueeze(ax), x.G.unsqueeze(ax + 1), x.e.unsqueeze(ax))
            else:
                x = x.unsqueeze(ax)
        return x

    def op_Squeeze(self, node, ins, a):
        x = ins[0]
        axes = a.get("axes") or (ins[1].tolist() if len(ins) > 1 and ins[1] is not None else None)
        if _is_state(x):
            dims = sorted([int(v) % x.c.dim() for v in axes], reverse=True) if axes else [i for i in range(x.c.dim()) if x.c.shape[i] == 1][::-1]
            c, G, e = x.c, x.G, x.e
            for d in dims:
                c = c.squeeze(d); G = G.squeeze(d + 1); e = e.squeeze(d)
            return SAZ(c, G, e)
        return x.squeeze() if not axes else x.squeeze(tuple(int(v) for v in axes))

    def op_ConstantOfShape(self, node, ins, a):
        shp = [max(1, int(v)) for v in ins[0].tolist()]
        val = numpy_helper.to_array(a["value"]).reshape(-1)[0] if "value" in a else 0.0
        return torch.full(shp, float(val), device=self.device, dtype=F64)

    def op_Transpose(self, node, ins, a):
        x = ins[0]; perm = list(a["perm"])
        if _is_state(x):
            return SAZ(x.c.permute(perm), x.G.permute([0] + [p + 1 for p in perm]), x.e.permute(perm))
        return x.permute(perm)

    def op_Flatten(self, node, ins, a):
        x = ins[0]; axis = int(a.get("axis", 1))
        if not _is_state(x):
            return x.reshape(int(np.prod(x.shape[:axis])), -1)
        shape = (int(np.prod(x.c.shape[:axis])), -1); nc = x.c.reshape(shape)
        return SAZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)), x.e.reshape(nc.shape))

    def op_Reshape(self, node, ins, a):
        x, shp = ins
        shp = [int(v) for v in shp.tolist()]
        cs = list(x.c.shape if _is_state(x) else x.shape)
        shp = [cs[i] if v == 0 else v for i, v in enumerate(shp)]
        if not _is_state(x):
            return x.reshape(shp)
        nc = x.c.reshape(shp)
        return SAZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)), x.e.reshape(nc.shape))

    def op_Slice(self, node, ins, a):
        x = ins[0]
        if "starts" in a:
            starts, ends = list(a["starts"]), list(a["ends"]); axes = list(a.get("axes", range(len(starts)))); steps = [1] * len(starts)
        else:
            starts, ends = ins[1].tolist(), ins[2].tolist()
            axes = ins[3].tolist() if len(ins) > 3 and ins[3] is not None else list(range(len(starts)))
            steps = ins[4].tolist() if len(ins) > 4 and ins[4] is not None else [1] * len(starts)
        shape = x.c.shape if _is_state(x) else x.shape
        sl = [slice(None)] * len(shape)
        for st, en, ax, sp_ in zip(starts, ends, axes, steps):
            ax = int(ax) % len(shape)
            sl[ax] = slice(int(st), int(min(en, shape[ax])), int(sp_))
        if _is_state(x):
            return SAZ(x.c[tuple(sl)], x.G[(slice(None),) + tuple(sl)], x.e[tuple(sl)])
        return x[tuple(sl)]

    def op_Concat(self, node, ins, a):
        axis = int(a["axis"])
        if not any(_is_state(v) for v in ins):
            return torch.cat([v.to(ins[0].dtype) for v in ins], axis)
        k = max(v.k for v in ins if _is_state(v))
        cs, Gs, es = [], [], []
        for v in ins:
            if _is_state(v):
                v = v.pad_to(k); cs.append(v.c); Gs.append(v.G); es.append(v.e)
            else:
                v = v.to(F64); cs.append(v); Gs.append(torch.zeros((k,) + tuple(v.shape), device=self.device, dtype=torch.float32)); es.append(torch.zeros_like(v))
        ax = axis % cs[0].dim()
        return SAZ(torch.cat(cs, ax), torch.cat(Gs, ax + 1), torch.cat(es, ax))

    def op_ReduceMean(self, node, ins, a):
        x = ins[0]; axes = [int(v) for v in a["axes"]]; kd = bool(a.get("keepdims", 1))
        n = int(np.prod([x.c.shape[v] for v in axes]))
        rx = radius64(x.G)
        G, se = map_G(x.G, lambda g: g.mean([v + 1 for v in axes], keepdim=kd))
        c = x.c.mean(axes, keepdim=kd)
        e = x.e.mean(axes, keepdim=kd) + gamma(n + 1, F64) * (x.c.abs() + rx).mean(axes, keepdim=kd) + se
        return SAZ(c, G, e)

    # ------------------------------------------------------------------ affine ops
    def op_MatMul(self, node, ins, a):
        x, W = ins
        if _is_state(x) and _is_state(W):
            return self._bilinear(x, W)
        if _is_state(x):
            n = W.shape[-2]; Wa = W.abs(); rx = radius64(x.G)
            G, se = map_G(x.G, lambda g: g @ W)
            return SAZ(x.c @ W, G, x.e @ Wa + gamma(n, F64) * ((x.c.abs() + rx) @ Wa) + se + TINY * n)
        if _is_state(W):
            n = x.shape[-1]; xa = x.abs(); rw = radius64(W.G)
            G, se = map_G(W.G, lambda g: x @ g)
            return SAZ(x @ W.c, G, xa @ W.e + gamma(n, F64) * (xa @ (W.c.abs() + rw)) + se + TINY * n)
        return x @ W

    def op_Gemm(self, node, ins, a):
        x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
        if int(a.get("transA", 0)):
            raise NotImplementedError("transA")
        if int(a.get("transB", 0)):
            W = W.t()
        al = float(a.get("alpha", 1.0)); be = float(a.get("beta", 1.0))
        W = W * al
        bias = (bb * be) if bb is not None else torch.zeros(W.shape[1], device=self.device, dtype=F64)
        n = W.shape[0]; Wa = W.abs(); rx = radius64(x.G)
        G, se = map_G(x.G, lambda g: g @ W)
        return SAZ(x.c @ W + bias, G, x.e @ Wa + gamma(n + 3, F64) * ((x.c.abs() + rx) @ Wa + bias.abs()) + se + TINY * n)

    def _addsub(self, x, y, s):
        if _is_state(x) and _is_state(y):
            k = max(x.k, y.k); xa, ya = x.pad_to(k), y.pad_to(k)
            shp = torch.broadcast_shapes(xa.c.shape, ya.c.shape)
            outs, se = [], None
            for st in range(0, k, 256):
                o64 = xa.G[st:st + 256].double().expand((-1,) + tuple(shp)) + s * ya.G[st:st + 256].double().expand((-1,) + tuple(shp))
                o32 = o64.float(); d = (o64 - o32.double()).abs().sum(0)
                se = d if se is None else se + d; outs.append(o32)
            G = torch.cat(outs, 0) if outs else torch.zeros((0,) + tuple(shp), device=self.device, dtype=torch.float32)
            se = (se if se is not None else torch.zeros(shp, device=self.device, dtype=F64)) * (1 + gamma(k + 2, F64))
            return SAZ(xa.c + s * ya.c, G, xa.e + ya.e + gamma(1, F64) * (xa.c.abs() + ya.c.abs()) + se)
        if _is_state(x):
            shp = torch.broadcast_shapes(x.c.shape, y.shape)
            c = (x.c + s * y).reshape(shp)
            return SAZ(c, x.G.expand((x.k,) + tuple(shp)).contiguous(), (x.e + gamma(1, F64) * (x.c.abs() + y.abs())).expand(shp).contiguous())
        if _is_state(y):
            shp = torch.broadcast_shapes(y.c.shape, x.shape)
            G, se = map_G(y.G, lambda g: s * g)
            return SAZ((x + s * y.c).reshape(shp), G.expand((y.k,) + tuple(shp)).contiguous(),
                       (y.e + gamma(1, F64) * (y.c.abs() + x.abs()) + se).expand(shp).contiguous())
        return x + s * y

    def op_Add(self, node, ins, a):
        return self._addsub(ins[0], ins[1], 1.0)

    def op_Sub(self, node, ins, a):
        return self._addsub(ins[0], ins[1], -1.0)

    def _scale(self, x, sc, g):
        sb = torch.broadcast_to(sc, x.c.shape) if torch.is_tensor(sc) else torch.full_like(x.c, float(sc))
        rx = radius64(x.G)
        G, se = map_G(x.G, lambda gg: gg * sb.unsqueeze(0))
        return SAZ(x.c * sb, G, x.e * sb.abs() * (1 + g) + g * sb.abs() * (x.c.abs() + rx) + se)

    def op_Mul(self, node, ins, a):
        x, y = ins
        if _is_state(x) and not _is_state(y):
            return self._scale(x, y, gamma(1, F64))
        if _is_state(y) and not _is_state(x):
            return self._scale(y, x, gamma(1, F64))
        if not _is_state(x) and not _is_state(y):
            return x * y
        return self._bilinear_elementwise(x, y)

    def op_Div(self, node, ins, a):
        x, y = ins
        if _is_state(x) and not _is_state(y):
            return self._scale(x, 1.0 / y, gamma(3, F64))
        if not _is_state(x) and not _is_state(y):
            return x / y
        raise NotImplementedError("Div by a state")

    def _conv_like(self, x, fn, fn_abs, out_c, n_fanin, bias=None):
        rx = radius64(x.G.reshape((x.k,) + tuple(x.c.shape[1:])))
        Gx = x.G.reshape((x.k,) + tuple(x.c.shape[1:]))
        G, se = map_G(Gx, fn)
        mag = fn_abs(x.c.abs() + rx.reshape(x.c.shape))
        if bias is not None:
            mag = mag + bias.abs().view(1, -1, *([1] * (out_c.dim() - 2)))
        ee = fn_abs(x.e) + gamma(n_fanin + 1, F64) * mag + se.reshape(out_c.shape) + TINY * n_fanin
        return SAZ(out_c, G.reshape((G.shape[0],) + tuple(out_c.shape)), ee)

    def op_Conv(self, node, ins, a):
        x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
        pads = list(a.get("pads", [0, 0, 0, 0]))
        if pads[0] != pads[2] or pads[1] != pads[3]:
            raise NotImplementedError("asymmetric padding")
        kw = dict(stride=tuple(a.get("strides", [1, 1])), padding=(pads[0], pads[1]), dilation=tuple(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
        Wa = W.abs(); n = int(W.shape[1] * W.shape[2] * W.shape[3])
        cc = F.conv2d(x.c, W, bb, **kw)
        return self._conv_like(x, lambda g: F.conv2d(g, W, None, **kw), lambda t: F.conv2d(t, Wa, None, **kw), cc, n, bb)

    def op_ConvTranspose(self, node, ins, a):
        x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
        pads = list(a.get("pads", [0, 0, 0, 0]))
        if pads[0] != pads[2] or pads[1] != pads[3]:
            raise NotImplementedError("asymmetric padding")
        op_ = list(a.get("output_padding", [0, 0]))
        kw = dict(stride=tuple(a.get("strides", [1, 1])), padding=(pads[0], pads[1]), output_padding=tuple(op_),
                  dilation=tuple(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
        Wa = W.abs(); n = int(W.shape[0] * W.shape[2] * W.shape[3])
        cc = F.conv_transpose2d(x.c, W, bb, **kw)
        return self._conv_like(x, lambda g: F.conv_transpose2d(g, W, None, **kw), lambda t: F.conv_transpose2d(t, Wa, None, **kw), cc, n, bb)

    def op_BatchNormalization(self, node, ins, a):
        x = ins[0]; gm, bt, mean, var = ins[1:5]
        eps = float(a.get("epsilon", 1e-5))
        s_ = gm / torch.sqrt(var + eps); t_ = bt - mean * s_
        g4 = gamma(4, F64); g2 = gamma(2, F64)
        ds = g4 * s_.abs(); dtt = mean.abs() * ds + g2 * (bt.abs() + (mean * s_).abs())
        shp = (1, -1) + (1,) * (x.c.dim() - 2)
        sv, tv = s_.view(shp), t_.view(shp)
        rx = radius64(x.G); mag = x.c.abs() + rx
        G, se = map_G(x.G, lambda gg: gg * sv.unsqueeze(0))
        ee = x.e * (sv.abs() + ds.view(shp)) + ds.view(shp) * mag + dtt.view(shp) + g2 * (sv.abs() * mag + tv.abs()) + se
        return SAZ(x.c * sv + tv, G, ee)

    def _nearest(self, x, scales):
        sh, sw = int(round(scales[-2])), int(round(scales[-1]))
        if abs(scales[-2] - sh) > 1e-9 or abs(scales[-1] - sw) > 1e-9 or any(abs(s - 1) > 1e-9 for s in scales[:-2]):
            raise NotImplementedError("non-integer or non-spatial upsample")
        f = lambda t: t.repeat_interleave(sh, dim=-2).repeat_interleave(sw, dim=-1)
        return SAZ(f(x.c), f(x.G), f(x.e))

    def op_Upsample(self, node, ins, a):
        x = ins[0]
        if a.get("mode", b"nearest") not in (b"nearest", "nearest"):
            raise NotImplementedError("upsample mode")
        scales = a.get("scales") or ins[1].tolist()
        return self._nearest(x, [float(s) for s in scales])

    def op_Resize(self, node, ins, a):
        x = ins[0]
        if a.get("mode", b"nearest") not in (b"nearest", "nearest"):
            raise NotImplementedError("resize mode")
        scales = ins[2].tolist() if len(ins) > 2 and ins[2] is not None and ins[2].numel() else None
        if scales is None:
            raise NotImplementedError("resize by sizes")
        return self._nearest(x, [float(s) for s in scales])

    # ------------------------------------------------------------------ nonlinear ops
    def op_Relu(self, node, ins, a):
        x = ins[0].pad_to(self.K)
        out, self.K = self._relu(node.name, x, self.K)
        return out

    def _relu(self, name, x, K):
        dev = self.device; rows = self.rows
        shape = x.c.shape
        fc = x.c.reshape(-1); fG = x.G.reshape(K, fc.numel()); fe = x.e.reshape(-1)
        rad_all = radius64(fG)
        allidx = torch.arange(fc.numel(), device=dev)
        l, u = self._bounds(fc, fG, fe, allidx, SRows(), K, 0, self.lr, rad_all)
        unst = (l < 0) & (u > 0)
        if self.iters > 0 and bool(unst.any()) and rows.n_rows:
            idx = torch.nonzero(unst).reshape(-1)
            l2, u2 = self._bounds(fc, fG, fe, idx, rows, K, self.iters, self.lr, rad_all)
            l = l.clone(); u = u.clone(); l[idx] = torch.maximum(l[idx], l2); u[idx] = torch.minimum(u[idx], u2)
        neg = u <= 0; pos = l >= 0; unst = ~(neg | pos)
        idx = torch.nonzero(unst).reshape(-1); m = int(idx.numel())
        lam = pos.to(F64).clone(); lu, uu = l[idx], u[idx]
        lam_u = (uu / (uu - lu)).clamp(0.0, 1.0)
        Mx = torch.maximum((1 - lam_u) * uu, -lam_u * lu) * (1 + gamma(3, F64)) + TINY
        mu = Mx / 2; lam[idx] = lam_u
        yc = fc * lam; yc[idx] = yc[idx] + mu
        yG, se = map_G(fG, lambda g: g * lam.unsqueeze(0))
        g2 = gamma(2, F64)
        ye = fe * lam + g2 * (lam * (fc.abs() + rad_all)) + se
        ye[idx] = ye[idx] + g2 * mu
        ye = torch.where(neg, torch.zeros_like(ye), ye)
        eta = torch.zeros((m, fc.numel()), device=dev, dtype=torch.float32)
        eta[torch.arange(m, device=dev), idx] = _up32(mu)
        yG = torch.cat([yG, eta], 0)
        gx = fG[:, idx].t().contiguous(); cx = fc[idx]; gy = yG[:, idx].t().contiguous(); cy = yc[idx]
        ex, ey = fe[idx], ye[idx]
        gx_full = torch.cat([gx, gx.new_zeros((m, m))], 1)
        rows.add((-gy).double(), (cy + ey).double())
        rows.add((gx_full - gy).double(), (cy - cx + ex + ey).double())
        self.phases.append(dict(layer=name, idx=idx, eta0=K, l=lu, u=uu, gx=gx, cx=cx, gy=gy, cy=cy, ex=ex, ey=ey,
                                row0=rows.n_rows - 2 * m))
        return SAZ(yc.reshape(shape), yG.reshape((K + m,) + tuple(shape)), ye.reshape(shape)), K + m

    def _smooth(self, kind, x):
        K = self.K; x = x.pad_to(K); dev = self.device; rows = self.rows
        shape = x.c.shape
        fc = x.c.reshape(-1); fG = x.G.reshape(K, fc.numel()); fe = x.e.reshape(-1)
        rad_all = radius64(fG); allidx = torch.arange(fc.numel(), device=dev)
        l, u = self._bounds(fc, fG, fe, allidx, SRows(), K, 0, self.lr, rad_all)
        if self.iters > 0 and rows.n_rows:
            l2, u2 = self._bounds(fc, fG, fe, allidx, rows, K, self.iters, self.lr, rad_all)
            l = torch.maximum(l, l2); u = torch.minimum(u, u2)
        u = torch.maximum(u, l); n = fc.numel()
        lam = torch.minimum(_sdf(kind, l), _sdf(kind, u)) * (1 - 1e-9)
        lo = _sf(kind, l) - lam * l; hi = _sf(kind, u) - lam * u
        padf = FPAD * (1 + _sf(kind, l).abs() + _sf(kind, u).abs() + (lam * l).abs() + (lam * u).abs())
        lo = lo - padf; hi = hi + padf
        mu = (lo + hi) / 2; nu = (hi - lo) / 2 * (1 + gamma(2, F64))
        yc = lam * fc + mu
        yG, se = map_G(fG, lambda g: g * lam.unsqueeze(0))
        ye = lam * fe + gamma(3, F64) * (lam * (fc.abs() + rad_all) + mu.abs()) + se
        eta = torch.zeros((n, n), device=dev, dtype=torch.float32)
        eta[torch.arange(n, device=dev), torch.arange(n, device=dev)] = _up32(nu)
        yG = torch.cat([yG, eta], 0); Kn = K + n
        lows, ups = rigorous_lines(kind, l, u)
        gx = torch.cat([fG, fG.new_zeros((n, n))], 0).double(); gy = yG.double()
        for a_, b_ in ups:
            rows.add((gy - gx * a_.unsqueeze(0)).t().contiguous(), b_ - yc + a_ * fc + ye + a_.abs() * fe)
        for a_, b_ in lows:
            rows.add((gx * a_.unsqueeze(0) - gy).t().contiguous(), yc - a_ * fc - b_ + ye + a_.abs() * fe)
        self.K = Kn
        return SAZ(yc.reshape(shape), yG.reshape((Kn,) + tuple(shape)), ye.reshape(shape))

    def op_Sigmoid(self, node, ins, a):
        return self._smooth("Sigmoid", ins[0])

    def op_Tanh(self, node, ins, a):
        return self._smooth("Tanh", ins[0])

    def _bilinear(self, x, y):
        """Rigorous DeepZ-style product z = x @ y of two states (shared factor space)."""
        K = self.K; x = x.pad_to(K); y = y.pad_to(K); dev = self.device
        gx = x.G.double(); gy = y.G.double()                          # transient float64 copies (attention is small)
        cz = x.c @ y.c
        Gz = x.c.unsqueeze(0) @ gy + gx @ y.c.unsqueeze(0)
        m_ = 0.5 * torch.einsum("k...nt,k...tm->...nm", gx, gy)
        absd = torch.einsum("k...nt,k...tm->...nm", gx.abs(), gy.abs())
        rx = gx.abs().sum(0); ry = gy.abs().sum(0)
        rad = (rx @ ry - 0.5 * absd).clamp(min=0)
        t = x.c.shape[-1]
        # true-value error: |x_t y_t - xhat yhat| <= |xhat| e_y + |yhat| e_x + e_x e_y, summed over t
        ax = x.c.abs() + rx; ay = y.c.abs() + ry
        e_true = ax @ y.e + x.e @ ay + x.e @ y.e
        # float64 rounding of every computed quantity (centre, generators, radius), all O(t K) terms
        mag = ax @ ay
        e_round = gamma(t * K + 4, F64) * 4 * mag
        Gz32 = Gz.float(); se = (Gz - Gz32.double()).abs().sum(0)
        rad = rad * (1 + gamma(t * K + 4, F64)) + e_round
        cz = cz + m_
        n_new = int(rad.numel())
        eta = torch.zeros((n_new,) + tuple(cz.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1)
        ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rad.reshape(-1))
        self.K = K + n_new
        return SAZ(cz, torch.cat([Gz32, eta], 0), e_true + e_round + se)

    def _bilinear_elementwise(self, x, y):
        raise NotImplementedError("elementwise state*state")

    def op_Softmax(self, node, ins, a):
        K = self.K; s = ins[0].pad_to(K); dev = self.device; rows = self.rows
        axis = int(a.get("axis", -1)) % s.c.dim()
        perm = [i for i in range(s.c.dim()) if i != axis] + [axis]; inv = [perm.index(i) for i in range(s.c.dim())]
        c = s.c.permute(perm); G = s.G.permute([0] + [p + 1 for p in perm]).double(); e = s.e.permute(perm)
        n = c.shape[-1]; eye = torch.eye(n, device=dev, dtype=F64); mask = 1 - eye
        dc = c.unsqueeze(-2) - c.unsqueeze(-1)
        dG = G.unsqueeze(-2) - G.unsqueeze(-1)
        de = (e.unsqueeze(-2) + e.unsqueeze(-1)) * mask               # radius of the true difference around the form
        dr = dG.abs().sum(0) * (1 + gamma(K + 2, F64))
        dlo, dhi = dc - dr - de, dc + dr + de
        e0 = torch.exp(dc); S0 = e0.sum(-1); p0 = 1.0 / S0
        grad = -e0 / (S0 ** 2).unsqueeze(-1) * mask
        pG = (grad.unsqueeze(0) * dG).sum(-1)
        elo = torch.exp(dlo) * mask + eye; ehi = torch.exp(dhi) * mask + eye
        Tlo = elo.sum(-1); Thi = ehi.sum(-1)
        Hoff = 2 * ehi.unsqueeze(-1) * ehi.unsqueeze(-2) / (Tlo ** 3).unsqueeze(-1).unsqueeze(-1)
        Hdiag = torch.maximum(2 * ehi ** 2 / Tlo.unsqueeze(-1) ** 3, ehi / Tlo.unsqueeze(-1) ** 2)
        Hb = (Hoff * mask.unsqueeze(0) if False else Hoff) * (1 - torch.eye(n, device=dev, dtype=F64)) + torch.diag_embed(Hdiag)
        Hb = Hb * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        delta = torch.maximum(dhi - dc, dc - dlo) * mask
        rem = 0.5 * torch.einsum("...ij,...ijk,...ik->...i", delta, Hb, delta)
        # the true difference = form + offset with |offset| <= de; that part of the tangent term goes to e_p
        ep = (grad.abs() * de).sum(-1)
        padv = 1e-12 * (p0 + rem + (grad.abs() * (dr + de)).sum(-1)) + gamma(n * K + 8, F64) * (p0 + (grad.abs() * dr).sum(-1))
        rem = rem + padv
        plo = (1.0 / Thi) * (1 - 1e-12); phi = (1.0 / Tlo) * (1 + 1e-12)
        n_new = int(rem.numel())
        pG32 = pG.float(); se = (pG - pG32.double()).abs().sum(0)
        eta = torch.zeros((n_new,) + tuple(p0.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1); ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rem.reshape(-1))
        Gp = torch.cat([pG32, eta], 0)
        epp = ep + se
        out = SAZ(p0.permute(inv).contiguous(), Gp.permute([0] + [i + 1 for i in inv]).contiguous(), epp.permute(inv).contiguous())
        Kn = K + n_new; self.K = Kn
        fc = out.c.reshape(-1); fGd = out.G.reshape(Kn, -1).double(); fe = out.e.reshape(-1)
        rows.add(fGd.t().contiguous(), (phi.permute(inv).reshape(-1) - fc + fe))
        rows.add(-fGd.t().contiguous(), (fc - plo.permute(inv).reshape(-1) + fe))
        sc = out.c.sum(axis).reshape(-1); sG = out.G.double().sum(axis + 1).reshape(Kn, -1); se_sum = out.e.sum(axis).reshape(-1)
        pad1 = gamma(n + 2, F64) * (1 + out.c.abs().sum(axis).reshape(-1))
        rows.add(sG.t().contiguous(), 1.0 - sc + se_sum + pad1)
        rows.add(-sG.t().contiguous(), sc - 1.0 + se_sum + pad1)
        return out

    def op_Pad(self, node, ins, a):
        x = ins[0]
        pads = list(a["pads"]) if "pads" in a else ins[1].tolist()
        mode = a.get("mode", b"constant")
        if mode not in (b"constant", "constant"):
            raise NotImplementedError("pad mode")
        val = float(ins[2]) if len(ins) > 2 and ins[2] is not None and ins[2].numel() else float(a.get("value", 0.0))
        if val != 0.0:
            raise NotImplementedError("nonzero pad value")
        d = x.c.dim(); half = len(pads) // 2
        tp = []
        for i in reversed(range(d)):
            tp += [int(pads[i]), int(pads[i + half])]
        if not _is_state(x):
            return F.pad(x, tp)
        return SAZ(F.pad(x.c, tp), F.pad(x.G, tp), F.pad(x.e, tp))
