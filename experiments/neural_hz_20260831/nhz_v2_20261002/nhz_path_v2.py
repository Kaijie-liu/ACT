"""Candidate path v2 per input box (fixed cascade, identical for every instance).

  A  aligned sound propagation (mixed-precision engine n006) + GPU LP tightening
  B  rigorous batched terminal LP for every unsafe disjunct
  C  witness candidates: LP box maximisers (S1 semantics, ORT validated)
  D  for open disjuncts (worst first): gamma = last ReLU layer MILP (HiGHS cutoff)
     with at most `d_share` of the remaining budget
  E  if D did not exclude: gamma = all ReLU layers MILP (the exact terminal query)
     with the remaining budget; incumbents are witness candidates
Outcome per box: CERT / ADV / UNKNOWN / TIMEOUT.  No per-instance choices.
"""
import time

import numpy as np
import torch

from nhz_terminal import plan_milp_highs, sound_lp_batch

PATH_VERSION = "path-v2.1"


def solve_box_v2(e, sess, iname, spec, lb, ub, deadline, iters=300, final_iters=1000, d_share=0.4):
    n_in = int(np.prod(e.input_shape))
    info = {"path": PATH_VERSION}
    t = time.time()
    out, rset, K, phases = e.propagate(lb, ub, iters, 0.05)
    torch.cuda.synchronize(); info["propagate_s"] = time.time() - t
    t = time.time()
    lp = sound_lp_batch(out, rset, K, spec.disjuncts, final_iters, 0.05)
    torch.cuda.synchronize(); info["terminal_lp_s"] = time.time() - t
    ups = [d["upper"] for d in lp]
    open_ids = sorted([i for i, v in enumerate(ups) if v >= -1e-4], key=lambda i: -ups[i])
    n_layers = sum(1 for p in phases if int(p["idx"].numel()))
    info.update({"K": K, "unstable": sum(int(p["idx"].numel()) for p in phases), "relu_layers_with_phases": n_layers,
                 "lp_worst": max(ups), "lp_open": len(open_ids)})
    nzi = e.input_factor_index.cpu().numpy(); center = (lb + ub) / 2; rad = (ub - lb) / 2

    def check(w):
        xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[: nzi.size]
        x64 = np.clip(center + rad * xi, lb, ub); x = x64.astype(np.float32)
        y = sess.run(None, {iname: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
        hit = next((k for k, atoms in enumerate(spec.disjuncts) if all(float(a @ y) <= b for a, b in atoms)), None)
        return hit, (x, x64)

    if not open_ids:
        return "CERT", info, None
    for i in open_ids:
        hit, xx = check(lp[i]["cand"])
        if hit is not None:
            return "ADV", info, ("lp_box_maximiser", i, hit, xx)
    milp_log = {}
    info["milp"] = milp_log
    A_, b_ = rset.dense(K)
    for i in open_ids:
        d = lp[i]
        excluded = False
        for stage, gl in (("D", 1), ("E", n_layers)):
            if stage == "E" and n_layers <= 1:
                break
            remaining = deadline - time.time()
            if remaining <= 1:
                return "TIMEOUT", info, None
            tl = remaining * d_share if stage == "D" else remaining
            m = plan_milp_highs(A_, b_, phases, K, d["g"], d["cc"], d["pad"], d["extra"], gl, 99, tl)
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
