"""Candidate path v4 = path v3 with the terminal MILP replaced by nhz_terminal_v4 (seed portfolio + early stop).

Original path v3 description:
Candidate path v3 (fixed cascade, identical for every instance).

Differences from path v2 (nhz_path_v2.py), each motivated by a measured failure:
  * terminal objective = epigraph min-slack for multi-atom disjuncts
    (nhz_terminal_v3; LOG N034/N035: incumbents on conjunction boundaries);
  * if the last-layer stage already covers every ReLU layer with phases it
    receives the whole remaining budget (LOG N036: sat_relu stopped at 40 percent);
  * intended to be used with engine n008 (exact-LP polishing of small LPs).
Stages: A propagate + GPU LP tightening; B rigorous epigraph LP per disjunct;
C LP box-maximiser witnesses; D gamma = last ReLU layer MILP; E gamma = all layers.
"""
import time

import numpy as np
import torch

from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from nhz_terminal_v4 import plan_milp_portfolio as plan_milp_highs_v3

PATH_VERSION = "path-v4.3"  # .3: passes the deadline to the engine (bounded polishing)  # v3 + seed portfolio (0-3) + early stop at violation target; .2 callback interrupt


def solve_box_v4(e, sess, iname, spec, lb, ub, deadline, iters=300, final_iters=1000, d_share=0.4):
    n_in = int(np.prod(e.input_shape))
    info = {"path": PATH_VERSION}
    t = time.time()
    if hasattr(e, "deadline"):
        e.deadline = deadline; e.t_start = t
    out, rset, K, phases = e.propagate(lb, ub, iters, 0.05)
    torch.cuda.synchronize(); info["propagate_s"] = time.time() - t
    t = time.time()
    A, b = rset.dense(K); A = A.to(out.c.device); b = b.to(out.c.device)
    objs = [epigraph_objective(out, K, atoms) for atoms in spec.disjuncts]
    ups, cands = [], []
    for g, cc, pad, extra, T in objs:
        ub_, cand = lp_bound_epigraph(A, b, g, cc, extra, final_iters, 0.05)
        ups.append(ub_ + pad); cands.append(cand)
    torch.cuda.synchronize(); info["terminal_lp_s"] = time.time() - t
    open_ids = sorted([i for i, v in enumerate(ups) if v >= -1e-4], key=lambda i: -ups[i])
    n_layers = sum(1 for p in phases if int(p["idx"].numel()))
    info.update({"K": K, "unstable": sum(int(p["idx"].numel()) for p in phases), "relu_layers_with_phases": n_layers,
                 "lp_worst": max(ups), "lp_open": len(open_ids)})
    nzi = e.input_factor_index.cpu().numpy(); center = (lb + ub) / 2; rad = (ub - lb) / 2

    def check(w):
        xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[: nzi.size]
        x64 = np.clip(center + rad * xi, lb, ub); x = x64.astype(np.float32)
        y = sess.run(None, {iname: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
        hit = next((k for k, atoms in enumerate(spec.disjuncts) if all(float(a @ y) <= bb for a, bb in atoms)), None)
        return hit, (x, x64)

    if not open_ids:
        return "CERT", info, None
    for i in open_ids:
        hit, xx = check(cands[i])
        if hit is not None:
            return "ADV", info, ("lp_box_maximiser", i, hit, xx)
    milp_log = {}
    info["milp"] = milp_log
    A_, b_ = rset.dense(K)
    stages = [("D", 1), ("E", n_layers)] if n_layers > 1 else [("D", 1)]
    for i in open_ids:
        g, cc, pad, extra, T = objs[i]
        excluded = False
        for stage, gl in stages:
            remaining = deadline - time.time()
            if remaining <= 1:
                return "TIMEOUT", info, None
            tl = remaining * d_share if (stage == "D" and len(stages) > 1) else remaining
            m = plan_milp_highs_v3(A_, b_, phases, K, g, cc, pad, extra, gl, 99, tl)
            milp_log[f"{i}:{stage}"] = {k: v for k, v in m.items() if k != "incumbent_w"}
            if m["incumbent_w"] is not None:
                hit, xx = check(m["incumbent_w"])
                if hit is not None:
                    return "ADV", info, (f"milp_incumbent_{stage}", i, hit, xx)
            if m["excluded"]:
                excluded = True
                break
        if not excluded:
            return ("TIMEOUT" if time.time() >= deadline - 1 else "UNKNOWN"), info, None
    return "CERT", info, None
