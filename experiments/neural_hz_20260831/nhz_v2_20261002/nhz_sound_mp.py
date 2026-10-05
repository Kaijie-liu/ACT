"""Neural-HZ v2 sound engine, mixed precision (n006.1).

Same semantics and invariant (INV) as nhz_sound.py (n004), but generator
tensors are STORED in float32 while every operation is COMPUTED in float64 on
factor chunks.  The exact storage rounding of each chunk,
sum_k |G64_k - float64(float32(G64_k))|, is added to the coordinatewise
rounding radius e (the subtraction is exact by Sterbenz; the summation is padded
by gamma).  Centres and radii stay in float64.  Eta generators are rounded
UP to float32 so that the V-function coverage of Lemma 1 is preserved.

Memory is that of float32 generators; precision loss is one float32 storage
rounding per operator instead of the n*u pads that made the pure float32
rigorous mode unusable (LOG N012).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from nhz_engine import _attrs
from nhz_sound import SAZ, SRows, TINY, gamma, optimise_nu, sound_eval

MP_VERSION = "n006.2"  # .2: map_G output shape for K = 0
U32 = 2.0 ** -24
CHUNK = 256


def radius64(G32: torch.Tensor) -> torch.Tensor:
    """sum_k |G_k| of the stored float32 generators, accumulated in float64."""
    r = None
    for s in range(0, G32.shape[0], CHUNK):
        part = G32[s:s + CHUNK].double().abs().sum(0)
        r = part if r is None else r + part
    if r is None:
        return torch.zeros(G32.shape[1:], device=G32.device, dtype=torch.float64)
    return r * (1 + gamma(G32.shape[0] + 2, torch.float64))


def map_G(G32: torch.Tensor, fn, out_shape=None):
    """Apply a float64 linear map fn to every factor row of G32 (chunked);
    return (stored float32 result, exact-plus-padded storage error per coordinate)."""
    outs, err = [], None
    for s in range(0, G32.shape[0], CHUNK):
        o64 = fn(G32[s:s + CHUNK].double())
        o32 = o64.float()
        d = (o64 - o32.double()).abs().sum(0)
        err = d if err is None else err + d
        outs.append(o32)
    if not outs:
        o = fn(G32[:0].double())
        return o.float(), torch.zeros(tuple(o.shape[1:]), device=G32.device, dtype=torch.float64)
    return torch.cat(outs, 0), err * (1 + gamma(G32.shape[0] + 2, torch.float64))


class SoundEngineMP:
    def __init__(self, path: str, device: str = "cuda"):
        self.model = onnx.load(path)
        self.device = device
        self.consts: Dict[str, torch.Tensor] = {}
        for t in self.model.graph.initializer:
            arr = numpy_helper.to_array(t).copy()
            self.consts[t.name] = torch.as_tensor(arr).to(device=device, dtype=torch.float64 if arr.dtype.kind == "f" else torch.int64)
        init_names = set(self.consts)
        self.input_name = [i.name for i in self.model.graph.input if i.name not in init_names][0]
        dims = [d.dim_value for d in [i for i in self.model.graph.input if i.name == self.input_name][0].type.tensor_type.shape.dim]
        self.input_shape = tuple(max(1, d) for d in dims)
        self.output_name = self.model.graph.output[0].name
        self.nodes = list(self.model.graph.node)

    def propagate(self, lb, ub, iters: int = 300, lr: float = 0.05, log=None):
        dev = self.device; f64 = torch.float64
        lb = torch.as_tensor(lb, dtype=f64).reshape(self.input_shape).to(dev)
        ub = torch.as_tensor(ub, dtype=f64).reshape(self.input_shape).to(dev)
        c0 = (lb + ub) / 2; r0 = (ub - lb) / 2
        flat_r = r0.reshape(-1)
        nz = torch.nonzero(flat_r > 0).reshape(-1); K = int(nz.numel())
        G0 = torch.zeros((K, flat_r.numel()), device=dev, dtype=torch.float32)
        r32 = flat_r[nz].float()
        r32 = torch.where(r32.double() < flat_r[nz], torch.nextafter(r32, torch.full_like(r32, float("inf"))), r32)  # round radius up
        G0[torch.arange(K, device=dev), nz] = r32
        e0 = gamma(2, f64) * (lb.abs() + ub.abs()) + TINY
        env: Dict[str, object] = {self.input_name: SAZ(c0, G0.reshape((K,) + tuple(c0.shape)), e0)}
        self.input_factor_index = nz
        rows = SRows(); phases = []
        remaining: Dict[str, int] = {}
        for node in self.nodes:
            for n in node.input:
                if n:
                    remaining[n] = remaining.get(n, 0) + 1
        for node in self.nodes:
            op = node.op_type
            ins = [env.get(n, self.consts.get(n)) if n else None for n in node.input]
            a = _attrs(node)
            if op == "Constant":
                arr = numpy_helper.to_array(a["value"]).copy()
                out = torch.as_tensor(arr).to(device=dev, dtype=f64 if arr.dtype.kind == "f" else torch.int64)
            elif op in ("Identity", "Dropout"):
                out = ins[0]
            elif op == "MatMul":
                x, W = ins
                if not isinstance(x, SAZ) or isinstance(W, SAZ):
                    raise NotImplementedError("MatMul form")
                n = W.shape[0]; Wa = W.abs(); rx = radius64(x.G)
                G, se = map_G(x.G, lambda g: g @ W)
                out = SAZ(x.c @ W, G, x.e @ Wa + gamma(n, f64) * ((x.c.abs() + rx) @ Wa) + se + TINY * n)
            elif op == "Gemm":
                x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
                if int(a.get("transA", 0)) or float(a.get("alpha", 1.0)) != 1.0 or float(a.get("beta", 1.0)) != 1.0:
                    raise NotImplementedError("Gemm form")
                if int(a.get("transB", 0)):
                    W = W.t()
                n = W.shape[0]; Wa = W.abs(); rx = radius64(x.G)
                bias = bb if bb is not None else torch.zeros(W.shape[1], device=dev, dtype=f64)
                G, se = map_G(x.G, lambda g: g @ W)
                out = SAZ(x.c @ W + bias, G, x.e @ Wa + gamma(n + 1, f64) * ((x.c.abs() + rx) @ Wa + bias.abs()) + se + TINY * n)
            elif op in ("Add", "Sub"):
                x, y = ins; s = 1.0 if op == "Add" else -1.0
                if isinstance(x, SAZ) and isinstance(y, SAZ):
                    k = max(x.k, y.k); xa, ya = x.pad_to(k), y.pad_to(k)
                    rx, ry = radius64(xa.G), radius64(ya.G)
                    # float32 add of stored rows: compute in float64, store, track error
                    outs, se = [], None
                    for st in range(0, k, CHUNK):
                        o64 = xa.G[st:st + CHUNK].double() + s * ya.G[st:st + CHUNK].double()
                        o32 = o64.float(); d = (o64 - o32.double()).abs().sum(0)
                        se = d if se is None else se + d; outs.append(o32)
                    G = torch.cat(outs, 0) if outs else xa.G
                    se = (se if se is not None else torch.zeros_like(xa.c)) * (1 + gamma(k + 2, f64))
                    out = SAZ(xa.c + s * ya.c, G, xa.e + ya.e + gamma(1, f64) * (xa.c.abs() + ya.c.abs()) + se)
                elif isinstance(x, SAZ):
                    yb = torch.broadcast_to(y, x.c.shape)
                    out = SAZ(x.c + s * yb, x.G, x.e + gamma(1, f64) * (x.c.abs() + yb.abs()))
                elif isinstance(y, SAZ):
                    xb = torch.broadcast_to(x, y.c.shape)
                    G, se = map_G(y.G, lambda g: s * g)
                    out = SAZ(xb + s * y.c, G, y.e + gamma(1, f64) * (y.c.abs() + xb.abs()) + se)
                else:
                    out = x + s * y
            elif op in ("Mul", "Div"):
                x, y = ins
                if isinstance(x, SAZ) and not isinstance(y, SAZ):
                    sc = y if op == "Mul" else 1.0 / y
                    g = gamma(1 if op == "Mul" else 3, f64)
                    sb = torch.broadcast_to(sc, x.c.shape); rx = radius64(x.G)
                    G, se = map_G(x.G, lambda gg: gg * sb.unsqueeze(0))
                    out = SAZ(x.c * sb, G, x.e * sb.abs() * (1 + g) + g * sb.abs() * (x.c.abs() + rx) + se)
                elif not isinstance(x, SAZ) and not isinstance(y, SAZ):
                    out = x * y if op == "Mul" else x / y
                else:
                    raise NotImplementedError("Mul form")
            elif op == "Conv":
                x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
                pads = list(a.get("pads", [0, 0, 0, 0]))
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    raise NotImplementedError("asymmetric padding")
                kw = dict(stride=tuple(a.get("strides", [1, 1])), padding=(pads[0], pads[1]),
                          dilation=tuple(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
                n = int(W.shape[1] * W.shape[2] * W.shape[3]); Wa = W.abs()
                cc = F.conv2d(x.c, W, bb, **kw)
                rx = radius64(x.G.reshape((x.k,) + tuple(x.c.shape[1:])))
                Gx = x.G.reshape((x.k,) + tuple(x.c.shape[1:]))
                G, se = map_G(Gx, lambda g: F.conv2d(g, W, None, **kw), out_shape=tuple(cc.shape[1:]))
                mag = F.conv2d((x.c.abs() + rx.reshape(x.c.shape)), Wa, None, **kw)
                if bb is not None:
                    mag = mag + bb.abs().view(1, -1, 1, 1)
                ee = F.conv2d(x.e, Wa, None, **kw) + gamma(n + 1, f64) * mag + se.reshape(cc.shape) + TINY * n
                out = SAZ(cc, G.unsqueeze(1) if G.dim() == cc.dim() else G.reshape((G.shape[0],) + tuple(cc.shape)), ee)
            elif op == "BatchNormalization":
                x = ins[0]; gm, bt, mean, var = ins[1:5]
                eps = float(a.get("epsilon", 1e-5))
                s_ = gm / torch.sqrt(var + eps); t_ = bt - mean * s_
                g4 = gamma(4, f64); g2 = gamma(2, f64)
                ds = g4 * s_.abs(); dtt = mean.abs() * ds + g2 * (bt.abs() + (mean * s_).abs())
                shp = (1, -1) + (1,) * (x.c.dim() - 2)
                sv, tv = s_.view(shp), t_.view(shp)
                rx = radius64(x.G); mag = x.c.abs() + rx
                G, se = map_G(x.G, lambda gg: gg * sv.unsqueeze(0))
                ee = x.e * (sv.abs() + ds.view(shp)) + ds.view(shp) * mag + dtt.view(shp) + g2 * (sv.abs() * mag + tv.abs()) + se
                out = SAZ(x.c * sv + tv, G, ee)
            elif op == "Flatten":
                x = ins[0]; axis = int(a.get("axis", 1))
                shape = (int(np.prod(x.c.shape[:axis])), -1); nc = x.c.reshape(shape)
                out = SAZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)), x.e.reshape(nc.shape))
            elif op == "Reshape":
                x, shp = ins
                shp = [int(v) for v in shp.tolist()]; cs = list(x.c.shape)
                shp = [cs[i] if v == 0 else v for i, v in enumerate(shp)]
                nc = x.c.reshape(shp)
                out = SAZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)), x.e.reshape(nc.shape))
            elif op == "Relu":
                x = ins[0].pad_to(K)
                out, K = self._relu(node.name, x, K, rows, phases, iters, lr, log)
            else:
                raise NotImplementedError(op)
            env[node.output[0]] = out
            del ins, out
            for n in node.input:
                if n and n in env and n != self.output_name:
                    remaining[n] -= 1
                    if remaining[n] == 0:
                        del env[n]
        res = env[self.output_name]
        return res.pad_to(K), rows, K, phases

    def _bounds(self, fc, fG32, fe, idx, rows: SRows, K: int, iters: int, lr: float, rad_all=None):
        sub = fG32[:, idx]
        rad = radius64(sub) if rad_all is None else rad_all[idx]
        g2 = gamma(K + 2, torch.float64); mag = fc[idx].abs() + rad
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
        return l, u

    def _relu(self, name, x: SAZ, K: int, rows: SRows, phases, iters, lr, log):
        dev = self.device; f64 = torch.float64
        shape = x.c.shape
        fc = x.c.reshape(-1); fG = x.G.reshape(K, fc.numel()); fe = x.e.reshape(-1)
        rad_all = radius64(fG)
        allidx = torch.arange(fc.numel(), device=dev)
        l, u = self._bounds(fc, fG, fe, allidx, SRows(), K, 0, lr, rad_all)
        unst = (l < 0) & (u > 0)
        n_sh = int(unst.sum())
        if iters > 0 and bool(unst.any()) and rows.n_rows:
            idx = torch.nonzero(unst).reshape(-1)
            l2, u2 = self._bounds(fc, fG, fe, idx, rows, K, iters, lr, rad_all)
            l = l.clone(); u = u.clone()
            l[idx] = torch.maximum(l[idx], l2); u[idx] = torch.minimum(u[idx], u2)
        neg = u <= 0; pos = l >= 0; unst = ~(neg | pos)
        idx = torch.nonzero(unst).reshape(-1); m = int(idx.numel())
        lam = pos.to(f64).clone()
        lu, uu = l[idx], u[idx]
        lam_u = (uu / (uu - lu)).clamp(0.0, 1.0)
        Mx = torch.maximum((1 - lam_u) * uu, -lam_u * lu) * (1 + gamma(3, f64)) + TINY
        mu = Mx / 2
        lam[idx] = lam_u
        yc = fc * lam; yc[idx] = yc[idx] + mu
        yG, se = map_G(fG, lambda g: g * lam.unsqueeze(0))
        g2 = gamma(2, f64)
        ye = fe * lam + g2 * (lam * (fc.abs() + rad_all)) + se
        ye[idx] = ye[idx] + g2 * mu
        ye = torch.where(neg, torch.zeros_like(ye), ye)
        mu32 = mu.float()
        mu32 = torch.where(mu32.double() < mu, torch.nextafter(mu32, torch.full_like(mu32, float("inf"))), mu32)
        eta = torch.zeros((m, fc.numel()), device=dev, dtype=torch.float32)
        eta[torch.arange(m, device=dev), idx] = mu32
        yG = torch.cat([yG, eta], 0)
        gx = fG[:, idx].t().contiguous(); cx = fc[idx]
        gy = yG[:, idx].t().contiguous(); cy = yc[idx]
        ex, ey = fe[idx], ye[idx]
        gx_full = torch.cat([gx, gx.new_zeros((m, m))], 1)
        rows.add((-gy).double(), (cy + ey).double())
        rows.add((gx_full - gy).double(), (cy - cx + ex + ey).double())
        phases.append(dict(layer=name, idx=idx, eta0=K, l=lu, u=uu, gx=gx, cx=cx, gy=gy, cy=cy, ex=ex, ey=ey,
                           row0=rows.n_rows - 2 * m))
        K = K + m
        if log is not None:
            log.append((name, int(fc.numel()), n_sh, m))
        return SAZ(yc.reshape(shape), yG.reshape((K,) + tuple(shape)), ye.reshape(shape)), K
