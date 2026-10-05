"""Neural-HZ rigorous engine n022 = n021 + LP-tightened remainders for state-state products.

The DeepZ product z = x @ y bounds the quadratic part sum_k,l gx_k gy_l w_k w_l per output by
  rad1 = rx @ ry - 0.5 |gx|.|gy|   with the centre shift 0.5 sum_k gx_k gy_k   (n009),
where rx, ry are the generator l1 radii, i.e. ranges over the latent BOX.  Every true latent point
also satisfies the element's rows, so |gx_t . w| <= rx'_t and |gy_t . w| <= ry'_t with rx', ry' the
LP-tightened ranges (rigorous `_bounds` through the rows), and the quadratic part is bounded by
  rad2 = sum_t rx'_t ry'_t   (no centre shift).
n022 computes both and keeps, per output coordinate, the smaller radius with its own centre; the
value maps (centre, generators) are unchanged except for the shift choice, and each coordinate
keeps its own fresh factor.  Everything else is n021.
"""
import torch

from nhz_sound import SAZ, gamma
from nhz_sound_mp import radius64
from nhz_sound_v9 import F64, _up32
from nhz_sound_v21 import SoundEngineV21

V22_VERSION = "n022.0"


class SoundEngineV22(SoundEngineV21):
    def _lp_radius(self, st, K):
        """Rigorous per-coordinate radius |g . w| over box ∩ rows for every coordinate of state st
        (K factors): max(u - c, c - l) from the LP-tightened bounds of the value map (no error term)."""
        fc = st.c.reshape(-1)
        if K == 0 or fc.numel() == 0:
            return torch.zeros_like(st.c, dtype=F64)
        fG = st.G.reshape(K, -1); fe = torch.zeros_like(fc)
        if self.rows.n_rows == 0 or self.iters <= 0:
            return radius64(fG).reshape(st.c.shape)
        idx = torch.arange(fc.numel(), device=self.device)
        l, u = self._bounds(fc, fG, fe, idx, self.rows, K, self.iters, self.lr)
        r = torch.maximum(u - fc, fc - l).clamp(min=0)
        return torch.minimum(r, radius64(fG)).reshape(st.c.shape)

    def _bilinear(self, x, y):
        K = self.K; x = x.pad_to(K); y = y.pad_to(K); dev = self.device
        gx = x.G.double(); gy = y.G.double()
        cz = x.c @ y.c
        Gz = x.c.unsqueeze(0) @ gy + gx @ y.c.unsqueeze(0)
        m_ = 0.5 * torch.einsum("k...nt,k...tm->...nm", gx, gy)
        absd = torch.einsum("k...nt,k...tm->...nm", gx.abs(), gy.abs())
        rx = gx.abs().sum(0); ry = gy.abs().sum(0)
        rad1 = (rx @ ry - 0.5 * absd).clamp(min=0)
        # n022: LP-tightened ranges of the factors
        rxl = self._lp_radius(x, K) * (1 + gamma(K + 4, F64)); ryl = self._lp_radius(y, K) * (1 + gamma(K + 4, F64))
        rad2 = rxl @ ryl
        t = x.c.shape[-1]
        ax = x.c.abs() + rx; ay = y.c.abs() + ry
        e_true = ax @ y.e + x.e @ ay + x.e @ y.e
        mag = ax @ ay
        e_round = gamma(t * K + 4, F64) * 4 * mag
        use2 = rad2 < rad1
        self.bilinear_lp_coords = getattr(self, "bilinear_lp_coords", 0) + int(use2.sum())
        self.bilinear_coords = getattr(self, "bilinear_coords", 0) + int(use2.numel())
        Gz32 = Gz.float(); se = (Gz - Gz32.double()).abs().sum(0)
        rad = torch.where(use2, rad2, rad1) * (1 + gamma(t * K + 4, F64)) + e_round
        cz = torch.where(use2, cz, cz + m_)
        n_new = int(rad.numel())
        eta = torch.zeros((n_new,) + tuple(cz.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1); ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rad.reshape(-1))
        self.K = K + n_new
        return SAZ(cz, torch.cat([Gz32, eta], 0), e_true + e_round + se)
