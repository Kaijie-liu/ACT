"""Engine n010.1 = n009 + AveragePool (affine) + MaxPool (exact, via max(a,b) = a + ReLU(b - a)).

MaxPool over a window v_1..v_m is computed as m_1 = v_1, m_{j+1} = m_j + ReLU(v_{j+1} - m_j);
each ReLU uses the projection-aligned exact transformer, so the max keeps exact binary
phases (gamma level) and needs no new transformer.  Windows are formed by unfolding the
state (centre, generators and radius alike).
"""
import torch
import torch.nn.functional as F

from nhz_sound import SAZ, TINY, gamma
from nhz_sound_mp import map_G, radius64
from nhz_sound_v9 import F64, SoundEngineV9

V10_VERSION = "n010.2"  # .2: K = 0 unfold


def _unfold_state(x: SAZ, k, s, p, d):
    C, H, W = x.c.shape[1:]
    Ho = (H + 2 * p[0] - d[0] * (k[0] - 1) - 1) // s[0] + 1; Wo = (W + 2 * p[1] - d[1] * (k[1] - 1) - 1) // s[1] + 1
    L = Ho * Wo
    def uf(t, lead):
        if lead == 0:
            return t.new_zeros((0, C, k[0] * k[1], L))
        t2 = t.reshape((lead, C, H, W))
        u = F.unfold(t2, k, dilation=d, padding=p, stride=s)          # [B, C*kh*kw, L]
        return u.reshape((lead, C, k[0] * k[1], L))
    c = uf(x.c, 1); G = uf(x.G.float() if x.G.dtype != torch.float32 else x.G, x.k); e = uf(x.e, 1)
    return c, G, e


class SoundEngineV10(SoundEngineV9):
    def op_AveragePool(self, node, ins, a):
        x = ins[0]
        k = tuple(a["kernel_shape"]); s = tuple(a.get("strides", k)); pads = list(a.get("pads", [0, 0, 0, 0]))
        if pads[0] != pads[2] or pads[1] != pads[3]:
            raise NotImplementedError("asymmetric padding")
        cip = bool(a.get("count_include_pad", 0)); p = (pads[0], pads[1])
        if (p[0] or p[1]) and not cip:
            raise NotImplementedError("count_include_pad=0 with padding")
        ap = lambda t: F.avg_pool2d(t, k, s, p, count_include_pad=True)
        n = k[0] * k[1]
        rx = radius64(x.G.reshape((x.k,) + tuple(x.c.shape[1:])))
        Gx = x.G.reshape((x.k,) + tuple(x.c.shape[1:]))
        G, se = map_G(Gx, ap)
        cc = ap(x.c)
        ee = ap(x.e) + gamma(n + 2, F64) * ap(x.c.abs() + rx.reshape(x.c.shape)) + se.reshape(cc.shape) + TINY * n
        return SAZ(cc, G.reshape((G.shape[0],) + tuple(cc.shape)), ee)

    def op_MaxPool(self, node, ins, a):
        x = ins[0].pad_to(self.K)
        k = tuple(a["kernel_shape"]); s = tuple(a.get("strides", k)); pads = list(a.get("pads", [0, 0, 0, 0]))
        d = tuple(a.get("dilations", [1, 1]))
        if pads[0] != pads[2] or pads[1] != pads[3] or any(pads):
            raise NotImplementedError("padded MaxPool")   # zero padding would change the max; refuse
        C, H, W = x.c.shape[1:]
        Ho = (H - d[0] * (k[0] - 1) - 1) // s[0] + 1; Wo = (W - d[1] * (k[1] - 1) - 1) // s[1] + 1
        c, G, e = _unfold_state(x, k, s, (0, 0), d)          # [1,C,m,L], [K,C,m,L], [1,C,m,L]
        m = k[0] * k[1]
        cur = SAZ(c[:, :, 0].contiguous(), G[:, :, 0].contiguous().unsqueeze(1), e[:, :, 0].contiguous())
        for j in range(1, m):
            K = self.K
            vj = SAZ(c[:, :, j].contiguous(), G[:, :, j].contiguous().unsqueeze(1), e[:, :, j].contiguous()).pad_to(K)
            diff = self._addsub(vj, cur.pad_to(K), -1.0)
            r_, self.K = self._relu(f"{node.name}_max{j}", diff.pad_to(self.K), self.K)
            cur = self._addsub(cur.pad_to(self.K), r_, 1.0)
        out_shape = (1, C, Ho, Wo)
        return SAZ(cur.c.reshape(out_shape), cur.G.reshape((cur.k,) + out_shape), cur.e.reshape(out_shape))
