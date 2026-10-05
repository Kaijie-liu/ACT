"""Neural-HZ rigorous engine n012 = n011.2 + fused attention-mix transformer.

Pattern: MatMul(P, V) with P = Softmax(S, axis = last) and V a state.  Instead of
composing the per-coordinate softmax enclosure with the DeepZ product, the output
z_id = sum_j p_ij v_jd is enclosed as (p0 = softmax(centre of S), m = centre of V)

  z = sum_j p0_j v_j                                  (A: exact, linear in V)
    + [f(s) - f(s0)],  f(s) = sum_j softmax(s)_j m_j   (B: linear Taylor term in S + remainder)
    + sum_j (p_j - p0_j)(v_j - m_j)                   (C: cross term)

  B linear coefficient a_ij = p0_ij (m_jd - f0_id); remainder
      |R_B| <= span_j(m_jd) * sigma_i^2 / 8,  sigma_i >= max_{a,b} |(s_b - s_a) - (s0_b - s0_a)|
      (g''(t) = E_q[(h - E_q h)^2 (m - E_q m)], Popoviciu: Var_q h <= span(h)^2 / 4).
  C: |C| <= sum_j dp_ij r_jd with dp the distance of p0 to the rigorous softmax box and
      r the radius of V around m.
All three parts are sound for the true values (THEORY.md Section 14); the result gets one
fresh factor per output coordinate with radius |R_B| + |C| plus rounding, and inherits the
value-map errors of V and S.  The Softmax node itself is still propagated by n011.2 (its
state and rows stay in the element, other consumers use it unchanged).
"""
import torch

from nhz_sound import SAZ, gamma
from nhz_sound_v9 import F64, _up32, _is_state
from nhz_sound_v11b import SoundEngineV11b

V12_VERSION = "n012.1"  # .1: weighted-variance remainder for term B (uses the softmax box)


class SoundEngineV12(SoundEngineV11b):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._sm_src = {}; self.fused_attention = 0

    def op_Softmax(self, node, ins, a):
        s = ins[0]
        axis = int(a.get("axis", -1)) % s.c.dim()
        if axis == s.c.dim() - 1:
            self._sm_src[node.output[0]] = (s, self.K)
        return super().op_Softmax(node, ins, a)

    def op_MatMul(self, node, ins, a):
        src = self._sm_src.get(node.input[0])
        if src is not None and _is_state(ins[1]):
            return self._attention_mix(src[0], ins[1])
        return super().op_MatMul(node, ins, a)

    def _attention_mix(self, s, v):
        K = self.K; s = s.pad_to(K); v = v.pad_to(K); dev = self.device
        sc = s.c; se_ = s.e; vc = v.c; ve = v.e
        T = sc.shape[-1]
        sG = s.G.double(); vG = v.G.double()
        p0 = torch.softmax(sc, -1)                                             # (..., T, T)
        # score-difference deviations: d_ab = s_b - s_a
        dG = sG.unsqueeze(-2) - sG.unsqueeze(-1)                                # (K, ..., T, T, T)
        dr = dG.abs().sum(0) * (1 + gamma(K + 2, F64))
        de = se_.unsqueeze(-2) + se_.unsqueeze(-1)
        dev_ab = dr + de                                                        # |h_b - h_a| bound
        sigma = dev_ab.amax(dim=(-1, -2))                                      # (..., T)  per query row i
        dc = sc.unsqueeze(-2) - sc.unsqueeze(-1)
        dhi = dc + dev_ab; dlo = dc - dev_ab
        eye = torch.eye(T, device=dev, dtype=F64); mask = 1 - eye
        Thi = (torch.exp(dhi) * mask + eye).sum(-1); Tlo = (torch.exp(dlo) * mask + eye).sum(-1)
        plo = (1.0 / Thi) * (1 - 1e-12); phi = (1.0 / Tlo) * (1 + 1e-12)
        dp = torch.maximum(phi - p0, p0 - plo).clamp(min=0)                     # (..., T, T)
        del dG
        rv = vG.abs().sum(0) + ve                                               # (..., T, D)
        # A
        zc = p0 @ vc
        GA = torch.einsum("...ij,k...jd->k...id", p0, vG)
        eA = p0 @ ve
        # B
        f0 = zc
        acoef = p0.unsqueeze(-1) * (vc.unsqueeze(-3) - f0.unsqueeze(-2))        # (..., i, j, d)
        GB = torch.einsum("k...ij,...ijd->k...id", sG, acoef)
        eB = torch.einsum("...ij,...ijd->...id", se_, acoef.abs())
        span_m = vc.amax(-2) - vc.amin(-2)                                      # (..., d)
        RB_span = span_m.unsqueeze(-2) * (sigma ** 2).unsqueeze(-1) / 8.0       # (..., i, d)
        # weighted bound: |g''| <= sum_j q_j (h_j - E_q h)^2 |m_j - E_q m|, q in the softmax box,
        # |h_j| <= rho_j (score radius), |E_q h| <= min(sum_k phi_k rho_k, max_k rho_k),
        # |m_j - E_q m| <= mu_jd = max(m_jd - min_k m_kd, max_k m_kd - m_jd)
        rho = sG.abs().sum(0) * (1 + gamma(K + 2, F64)) + se_                  # (..., i, j)
        rbar = torch.minimum((phi * rho).sum(-1), rho.amax(-1))                 # (..., i)
        mu_ = torch.maximum(vc - vc.amin(-2, keepdim=True), vc.amax(-2, keepdim=True) - vc)   # (..., j, d)
        wq = phi * (rho + rbar.unsqueeze(-1)) ** 2                              # (..., i, j)
        RB_w = 0.5 * (wq @ mu_)                                                 # (..., i, d)
        RB = torch.minimum(RB_span, RB_w)
        # C
        RC = dp @ rv
        # rounding of the float64 evaluation (all O(T K) sums)
        n_ops = T * K + 8
        mag = p0 @ (vc.abs() + vG.abs().sum(0)) + torch.einsum("...ij,...ijd->...id", sc.abs() + sG.abs().sum(0), acoef.abs())
        e_round = gamma(n_ops, F64) * 4 * (mag + RB + RC + zc.abs())
        G = GA + GB
        G32 = G.float(); sstore = (G - G32.double()).abs().sum(0)
        rad = (RB + RC) * (1 + gamma(4, F64)) + e_round
        n_new = int(rad.numel())
        eta = torch.zeros((n_new,) + tuple(zc.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1); ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rad.reshape(-1))
        self.last_fused_terms = {"RB_mean": float(RB.mean()), "RC_mean": float(RC.mean()), "sigma_mean": float(sigma.mean()),
                                 "span_m_mean": float(span_m.mean()), "lin_mean": float(G.abs().sum(0).mean()), "eA_eB_mean": float((eA + eB).mean())}
        self.K = K + n_new; self.fused_attention += 1
        return SAZ(zc, torch.cat([G32, eta], 0), eA + eB + sstore + e_round)
