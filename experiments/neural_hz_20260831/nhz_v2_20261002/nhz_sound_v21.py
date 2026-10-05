"""Neural-HZ rigorous engine n021 = n020.1 + a row-coupled softmax element (THEORY 16, direction 1 + 3).

Per softmax row (query q), with i* the token of largest centre score:
  (a) the score differences s_j - s_{i*} are bounded through the element's rows by the GPU LP
      (`_bounds`, rigorous), and the bounds are merged into the difference box used by the softmax
      enclosure (box, Hessian remainder, interval choice);
  (b) ratio rows p_j <= exp(U_j) p_{i*} and p_j >= exp(L_j) p_{i*} are added on the softmax output
      value maps (tolerance = output radii + rounding pad).  In this construction the attention
      product references the softmax fresh factors, so these rows reach the terminal LP.
Everything else is n020.1 (domain-only lineage; pooling; LP time share).
"""
import numpy as np
import torch

from nhz_sound import SAZ, gamma
from nhz_sound_v9 import F64, _up32
from nhz_sound_v20 import SoundEngineV20

V21_VERSION = "n021.0"


class SoundEngineV21(SoundEngineV20):
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
        if axis == s.c.dim() - 1:
            dr = self._improve_dr(node, dr, dG, K)
        dlo, dhi = dc - dr - de, dc + dr + de
        # n021: LP-tightened bounds of the differences s_j - s_{i*} (i* = token with the largest centre score,
        # an observable structural choice) through the element's rows; merged symmetrically into dlo / dhi
        lead = c.shape[:-1]; Q = int(np.prod(lead)) if len(lead) else 1
        istar = c.reshape(Q, n).argmax(-1)                                        # (Q,)
        qi = torch.arange(Q, device=dev)
        dc_q = dc.reshape(Q, n, n); dG_q = dG.reshape(K, Q, n, n); de_q = de.reshape(Q, n, n)
        fc_d = dc_q[qi, istar, :].reshape(-1)                                     # (Q*n,)  c_j - c_{i*}
        fG_d64 = dG_q[:, qi, istar, :].reshape(K, -1)                             # (K, Q*n)
        fG_d32 = fG_d64.float(); cast = (fG_d64 - fG_d32.double()).abs().sum(0)
        fe_d = de_q[qi, istar, :].reshape(-1) + cast
        if self.rows.n_rows and self.iters > 0:
            Ld, Ud = self._bounds(fc_d, fG_d32, fe_d, torch.arange(Q * n, device=dev), self.rows, K, self.iters, self.lr)
            Ld = Ld.reshape(Q, n); Ud = Ud.reshape(Q, n)
            dlo_q = dlo.reshape(Q, n, n).clone(); dhi_q = dhi.reshape(Q, n, n).clone()
            dlo_q[qi, istar, :] = torch.maximum(dlo_q[qi, istar, :], Ld); dhi_q[qi, istar, :] = torch.minimum(dhi_q[qi, istar, :], Ud)
            dlo_q[qi, :, istar] = torch.maximum(dlo_q[qi, :, istar], -Ud); dhi_q[qi, :, istar] = torch.minimum(dhi_q[qi, :, istar], -Ld)
            # the diagonal difference is exactly zero
            dlo_q[:, torch.arange(n), torch.arange(n)] = 0.0; dhi_q[:, torch.arange(n), torch.arange(n)] = 0.0
            dlo = dlo_q.reshape(dlo.shape); dhi = dhi_q.reshape(dhi.shape)
            self.sm_lp_tightened = getattr(self, "sm_lp_tightened", 0) + Q * n
        Lrat = dlo.reshape(Q, n, n)[qi, istar, :]; Urat = dhi.reshape(Q, n, n)[qi, istar, :]   # bounds of s_j - s_{i*}
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
        # n011: per-coordinate choice of the smaller sound enclosure (tangent plane + remainder,
        # or the interval [plo, phi] with one fresh factor); each coordinate has its own fresh factor
        r_tan = pG.abs().sum(0) * (1 + gamma(K + 2, F64)) + rem + ep
        r_int = (phi - plo) / 2
        use_int = (rem + ep) >= r_int      # n011.2: switch only where the remainder alone is uninformative
        self.softmax_interval_coords += int(use_int.sum()); self.softmax_coords += int(use_int.numel())
        couple = use_int & ((rem + ep) < r_int)          # tangent relation still informative
        tan_c, tan_G, tan_r = p0.clone(), pG.clone(), (rem + ep).clone()
        p0 = torch.where(use_int, (plo + phi) / 2, p0)
        pG = torch.where(use_int.unsqueeze(0), torch.zeros_like(pG), pG)
        rem = torch.where(use_int, r_int * (1 + gamma(4, F64)) + 1e-300, rem)
        ep = torch.where(use_int, torch.zeros_like(ep), ep)
        n_new = int(rem.numel())
        pG32 = pG.float(); se = (pG - pG32.double()).abs().sum(0)
        eta = torch.zeros((n_new,) + tuple(p0.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1); ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rem.reshape(-1))
        Gp = torch.cat([pG32, eta], 0)
        epp = ep + se
        out = SAZ(p0.permute(inv).contiguous(), Gp.permute([0] + [i + 1 for i in inv]).contiguous(), epp.permute(inv).contiguous())
        Kn = K + n_new; self.K = Kn
        # n021: ratio rows  p_j <= exp(U_j) p_{i*}  and  p_j >= exp(L_j) p_{i*}  (true values), written on the value
        # maps with the output radii as tolerance: couples the softmax coordinates of a row through their factors
        Gq = Gp.double().reshape(Kn, Q, n); cq = p0.reshape(Q, n); eq = epp.reshape(Q, n)
        Gs = Gq[:, qi, istar]; cs = cq[qi, istar]; es = eq[qi, istar]                      # reference token
        eU = torch.exp(Urat) * (1 + 1e-12); eL = torch.exp(Lrat) * (1 - 1e-12)
        padr = gamma(Kn + 6, F64) * (cq.abs() + eU * cs.unsqueeze(-1).abs() + Gq.abs().sum(0) + eU * Gs.abs().sum(0).unsqueeze(-1)) + 1e-300
        offd = torch.ones(Q, n, dtype=torch.bool, device=dev); offd[qi, istar] = False
        A_up = (Gq - eU.unsqueeze(0) * Gs.unsqueeze(-1))                                   # (Kn, Q, n): p_j - eU p_i*
        b_up = -(cq - eU * cs.unsqueeze(-1)) + eq + eU * es.unsqueeze(-1) + padr
        A_lo = (eL.unsqueeze(0) * Gs.unsqueeze(-1) - Gq)                                   # eL p_i* - p_j <= ...
        b_lo = -(eL * cs.unsqueeze(-1) - cq) + eq + eL * es.unsqueeze(-1) + padr
        sel = offd.reshape(-1)
        rows.add(A_up.reshape(Kn, -1)[:, sel].t().contiguous(), b_up.reshape(-1)[sel])
        rows.add(A_lo.reshape(Kn, -1)[:, sel].t().contiguous(), b_lo.reshape(-1)[sel])
        self.sm_ratio_rows = getattr(self, "sm_ratio_rows", 0) + 2 * int(sel.sum())
        fc = out.c.reshape(-1); fGd = out.G.reshape(Kn, -1).double(); fe = out.e.reshape(-1)
        rows.add(fGd.t().contiguous(), (phi.permute(inv).reshape(-1) - fc + fe))
        rows.add(-fGd.t().contiguous(), (fc - plo.permute(inv).reshape(-1) + fe))
        sc = out.c.sum(axis).reshape(-1); sG = out.G.double().sum(axis + 1).reshape(Kn, -1); se_sum = out.e.sum(axis).reshape(-1)
        pad1 = gamma(n + 2, F64) * (1 + out.c.abs().sum(axis).reshape(-1))
        rows.add(sG.t().contiguous(), 1.0 - sc + se_sum + pad1)
        rows.add(-sG.t().contiguous(), sc - 1.0 + se_sum + pad1)
        # coupling rows for interval coordinates:  |y_i - (tan_c_i + tan_G_i^T w)| <= tan_r_i + e_i (+ pads)
        cpl = couple.permute(inv).reshape(-1)
        if bool(cpl.any()):
            ids = torch.nonzero(cpl).reshape(-1)
            tG = tan_G.permute([0] + [i + 1 for i in inv]).reshape(K, -1)[:, ids]           # (K, m)
            tc = tan_c.permute(inv).reshape(-1)[ids]; tr = tan_r.permute(inv).reshape(-1)[ids]
            yG = fGd[:, ids]                                                                 # (Kn, m), y value map
            D = yG.clone(); D[:K] -= tG
            padc = gamma(K + 4, F64) * (fc[ids].abs() + tc.abs() + yG.abs().sum(0) + tG.abs().sum(0)) + 1e-300
            rhs = tr + fe[ids] + padc
            rows.add(D.t().contiguous(), rhs - (fc[ids] - tc))
            rows.add(-D.t().contiguous(), rhs + (fc[ids] - tc))
            self.softmax_coupled = getattr(self, "softmax_coupled", 0) + int(ids.numel())
        return out
