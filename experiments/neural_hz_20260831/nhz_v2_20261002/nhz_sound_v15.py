"""Neural-HZ rigorous engine n015 = n014 + fused attention-mix with Q/K re-linking.

Pattern: MatMul(P, V), P = Softmax(c * (Q @ Kt)) over the last axis (Q, Kt, V states).
    z_id = sum_j p0_ij v_jd                                    (A: exact, linear in V)
         + scale * q_i . kt~_id - sum_j a_ijd s0_ij + R_B          (B: Taylor term, re-linked)
         + X_id                                                (C: cross term)
with p0 = softmax(s0) (s0 = centre of the score state), m = centre of V, f0 = p0 m,
a_ijd = p0_ij (m_jd - f0_id) and kt~_ide = sum_j a_ijd Kt_ej (a state: linear in Kt, its
shared generators cancel because sum_j a_ijd = 0).  q_i . kt~_id is a state-state product
(DeepZ with the shared-factor diagonal correction, one remainder per (i, d)) instead of the
sum of the score entries' separate product remainders.  R_B: as n012.1 (min of the span bound
and the softmax-box-weighted variance bound); C: sum_j dp_ij r_jd with the softmax box.
Per output coordinate the engine keeps the n014 composite (softmax state x V) or this
enclosure, whichever has the smaller radius (sum |G| + fresh + error); each coordinate keeps
its own fresh factor, so the state is sound.
"""
import torch

from nhz_sound import SAZ, gamma
from nhz_sound_mp import radius64
from nhz_sound_v9 import F64, _up32, _is_state
from nhz_sound_v14 import SoundEngineV14

V15_VERSION = "n015.0"


class SoundEngineV15(SoundEngineV14):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._sm15 = {}; self.fused_coords = 0; self.fused_total = 0

    def op_Softmax(self, node, ins, a):
        s = ins[0]; axis = int(a.get("axis", -1)) % s.c.dim()
        info = self._prod.get(node.input[0])
        if axis == s.c.dim() - 1 and info is not None:
            self._sm15[node.output[0]] = (s, info)
        return super().op_Softmax(node, ins, a)

    def op_MatMul(self, node, ins, a):
        src = self._sm15.get(node.input[0])
        if src is not None and _is_state(ins[0]) and _is_state(ins[1]):
            K_before = self.K
            comp = super().op_MatMul(node, ins, a)              # n014 composite (softmax state x V)
            K_mid = self.K
            fused = self._fused(src[0], src[1], ins[1])
            return self._choose(comp, fused, K_before, K_mid)
        return super().op_MatMul(node, ins, a)

    def _choose(self, comp, fused, K0, K1):
        K2 = self.K
        comp = comp.pad_to(K2); fused = fused.pad_to(K2)
        rc = radius64(comp.G.reshape(K2, -1)).reshape(comp.c.shape) + comp.e
        rf = radius64(fused.G.reshape(K2, -1)).reshape(fused.c.shape) + fused.e
        use = rf < rc
        self.fused_coords += int(use.sum()); self.fused_total += int(use.numel())
        sh = (1,) * 0
        c = torch.where(use, fused.c, comp.c)
        G = torch.where(use.unsqueeze(0), fused.G, comp.G)
        e = torch.where(use, fused.e, comp.e)
        return SAZ(c, G, e)

    def _fused(self, s, info, v):
        K = self.K; dev = self.device
        s = s.pad_to(K); v = v.pad_to(K)
        q, kt, scale = info["x"].pad_to(K), info["y"].pad_to(K), info["scale"]
        sc = s.c; se_ = s.e; vc = v.c; ve = v.e; T = sc.shape[-1]
        sG = s.G.double(); vG = v.G.double()
        p0 = torch.softmax(sc, -1)
        dG = sG.unsqueeze(-2) - sG.unsqueeze(-1)
        dr = dG.abs().sum(0) * (1 + gamma(K + 2, F64)); del dG
        dr = self._improve_dr_tensor(info, dr, K)
        de = se_.unsqueeze(-2) + se_.unsqueeze(-1); dev_ab = dr + de
        sigma = dev_ab.amax(dim=(-1, -2))
        dc = sc.unsqueeze(-2) - sc.unsqueeze(-1)
        eye = torch.eye(T, device=dev, dtype=F64); mask = 1 - eye
        Thi = (torch.exp(dc + dev_ab) * mask + eye).sum(-1); Tlo = (torch.exp(dc - dev_ab) * mask + eye).sum(-1)
        plo = (1.0 / Thi) * (1 - 1e-12); phi = (1.0 / Tlo) * (1 + 1e-12)
        dp = torch.maximum(phi - p0, p0 - plo).clamp(min=0)
        rv = vG.abs().sum(0) + ve
        # A
        zc = p0 @ vc
        GA = torch.einsum("...ij,k...jd->k...id", p0, vG)
        eA = p0 @ ve
        # B: re-linked linear term  scale * q_i . kt~_id,  kt~_ide = sum_j a_ijd kt_ej
        acoef = p0.unsqueeze(-1) * (vc.unsqueeze(-3) - zc.unsqueeze(-2))          # (..., i, j, d)
        ktc = kt.c; ktG = kt.G.double(); kte = kt.e                                # (..., e, j)
        kc = torch.einsum("...ijd,...ej->...ide", acoef, ktc)
        kG = torch.einsum("...ijd,k...ej->k...ide", acoef, ktG)
        ke = torch.einsum("...ijd,...ej->...ide", acoef.abs(), kte)
        qc = q.c.unsqueeze(-2); qG = q.G.double().unsqueeze(-2); qe = q.e.unsqueeze(-2)   # (..., i, 1, e)
        cB = (qc * kc).sum(-1)
        GB = (qc.unsqueeze(0) * kG).sum(-1) + (qG * kc.unsqueeze(0)).sum(-1)
        mB = 0.5 * (qG * kG).sum(-1).sum(0)
        absB = (qG.abs() * kG.abs()).sum(-1).sum(0)
        rq = qG.abs().sum(0); rk = kG.abs().sum(0)
        radB = ((rq * rk).sum(-1) - 0.5 * absB).clamp(min=0)
        eB = ((qc.abs() + rq) * ke + qe * (kc.abs() + rk) + qe * ke).sum(-1)
        constB = (acoef * s.c.unsqueeze(-1)).sum(-2)                              # sum_j a_ijd s0_ij
        # the Taylor term is f(s)-f0 = sum_j a_j (s_j - s0_j) + R_B with s = scale * q.kt (true scores)
        centreB = scale * (cB + mB) - constB
        GB = scale * GB; radB = abs(scale) * radB; eB = abs(scale) * eB
        # R_B (n012.1)
        span_m = vc.amax(-2) - vc.amin(-2)
        RB_span = span_m.unsqueeze(-2) * (sigma ** 2).unsqueeze(-1) / 8.0
        rho = sG.abs().sum(0) * (1 + gamma(K + 2, F64)) + se_
        rbar = torch.minimum((phi * rho).sum(-1), rho.amax(-1))
        mu_ = torch.maximum(vc - vc.amin(-2, keepdim=True), vc.amax(-2, keepdim=True) - vc)
        RB_w = 0.5 * ((phi * (rho + rbar.unsqueeze(-1)) ** 2) @ mu_)
        RB = torch.minimum(RB_span, RB_w)
        RC = dp @ rv
        n_ops = T * K + 8
        mag = p0 @ (vc.abs() + vG.abs().sum(0)) + abs(scale) * ((qc.abs() + rq) * (kc.abs() + rk)).sum(-1) + constB.abs()
        e_round = gamma(n_ops, F64) * 8 * (mag + RB + RC + zc.abs())
        c_out = zc + centreB
        G = GA + GB
        G32 = G.float(); sstore = (G - G32.double()).abs().sum(0)
        rad = (radB + RB + RC) * (1 + gamma(8, F64)) + e_round
        n_new = int(rad.numel())
        eta = torch.zeros((n_new,) + tuple(zc.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1); ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rad.reshape(-1))
        self.K = K + n_new
        return SAZ(c_out, torch.cat([G32, eta], 0), eA + eB + sstore + e_round)

    def _improve_dr_tensor(self, info, dr, K):
        class _N:  # adapter for n014's _improve_dr (needs node.input[0] lookup)
            pass
        x, y, K0, K1, scale = info["x"], info["y"], info["K0"], info["K1"], abs(info["scale"])
        return dr  # bounds already improved where the softmax was propagated; the fused bound uses dr as is
