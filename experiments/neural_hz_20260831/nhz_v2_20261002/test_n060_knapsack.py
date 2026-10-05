"""N060 test: knapsack_cross is an upper bound of max sum_j |dp_j| r_j over the box with the
sum constraint (checked against scipy linprog on the u/w relaxation and random feasible points)."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v13 import knapsack_cross
from scipy.optimize import linprog
rng = np.random.default_rng(0); worst = 0.0; viol = 0
for trial in range(300):
    T = rng.integers(2, 9); D = rng.integers(1, 4)
    p = rng.dirichlet(np.ones(T)); ph = p + rng.normal(0, 0.05, T)
    lo = np.clip(p - rng.uniform(0, 0.2, T), 0, 1); hi = np.clip(p + rng.uniform(0, 0.2, T), 0, 1)
    A = lo - ph; B = hi - ph; S = 1 - ph.sum(); r = rng.uniform(0, 1, (T, D))
    kb = knapsack_cross(torch.tensor(A)[None], torch.tensor(B)[None], torch.tensor([S]), torch.tensor(r))[0].numpy()
    for d in range(D):
        # LP over u, w: max sum (u+w) r  s.t. sum u - sum w = S, 0<=u<=B+, 0<=w<=A-
        c = -np.concatenate([r[:, d], r[:, d]])
        res = linprog(c, A_eq=np.concatenate([np.ones(T), -np.ones(T)])[None], b_eq=[S],
                      bounds=[(0, max(b, 0)) for b in B] + [(0, max(-a, 0)) for a in A])
        if res.status == 0:
            worst = max(worst, abs(-res.fun - kb[d]))
        # random feasible dp (true p in box): |sum dp r| <= bound
        for _ in range(20):
            q = rng.dirichlet(np.ones(T))
            if np.all(q >= lo) and np.all(q <= hi):
                dp = q - ph
                if np.abs(dp) @ r[:, d] > kb[d] + 1e-9:
                    viol += 1
print("max |greedy - LP|", worst, "random-point violations", viol)
