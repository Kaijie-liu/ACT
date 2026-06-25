#!/usr/bin/env python
"""HZ full-run worker: verify ONE (benchmark, iid) with the engine's pure config.

Config: hybridz HZ domain, tight_bounds=True (LP-tight [alpha,beta], the #1
precision lever), sigmoid K=2 / tanh K=1 by default, reduce Step-0 always-on,
verdict = hz_certify_spec (scipy LP-relax + exact HiGHS MILP, Gurobi OFF). ORT
is a truth-check only: it may confirm an engine-produced exact witness or
downgrade a bad CERT, but pure mode never lets ORT decide a case that HybridZ
left UNKNOWN. One process per instance => mem-isolated; RLIMIT_AS caps CPU mem
(cpu device) so a runaway cannot kill the host/VS Code. Prints exactly ONE json
result line to stdout.

Multi-query specs (OR-disjunct robustness: malbeware 24 / metaroom 19 / ...):
prop ONCE, certify EVERY disjunct's C/t; instance CERT iff ALL disjuncts are
proved empty (the union of unsafe disjuncts is unreachable). UNKNOWN otherwise.
"""
import os, sys, json, time, argparse, resource
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH_ROOT = Path(os.environ.get(
    "ACT_VNNCOMP_BENCH_ROOT",
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
))

def _hz_solve(hz, C, t, is_uns):
    """LP relaxation (binaries relaxed to [-1,1] = convex hull) of the spec margin in
    the UNSAFE direction. Returns (margin, witness_xi); the witness's continuous block
    maps to a concrete input. is_uns: joint  min s  s.t. C[r]y-s<=t[r] (SAFE iff s>0).
    ALL-rows: max_r(max_y C[r]y - t[r]) (SAFE iff <0). The exact MILP escalation lives
    in hz_objbound_decide; this is LP-only."""
    import numpy as np
    from scipy import sparse as sp
    from scipy.optimize import linprog
    from act.back_end.solver.solver_hz_verdict import _hz_relax_np_sparse
    out_dim = int(hz.c.numel())
    C = np.asarray(C, dtype=np.float64).reshape(-1, out_dim)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size == 1 and C.shape[0] != 1:
        t = np.repeat(t, C.shape[0])
    c, Gc, Gb, A_eq_base, be, A_ub_base, bl = _hz_relax_np_sparse(hz)
    ng, nb = Gc.shape[1], Gb.shape[1]

    if ng + nb == 0:
        return float(np.max(C @ c - t)), np.zeros(0, dtype=np.float64)

    if not is_uns:
        # ALL-rows: maximize each row's C[r]y, keep the worst margin + its witness.
        best_m, best_xi = -np.inf, None
        for r in range(C.shape[0]):
            obj = np.concatenate([C[r] @ Gc, C[r] @ Gb])
            rr = linprog(-obj,
                         A_eq=A_eq_base, b_eq=(be if A_eq_base is not None else None),
                         A_ub=A_ub_base, b_ub=(bl if A_ub_base is not None else None),
                         bounds=[(-1, 1)] * (ng + nb), method="highs")
            if not rr.success:
                return None, None
            m_r = (float(C[r] @ c) - rr.fun) - float(t[r])
            if m_r > best_m:
                best_m, best_xi = m_r, rr.x[:ng + nb]
        return best_m, best_xi

    # is_uns: joint epigraph  min s  s.t. C[r]y - s <= t[r]  (vars [xi_c, xi_b, s]).
    nrow = C.shape[0]; nv = ng + nb + 1
    epi_A = np.zeros((nrow, nv)); epi_b = np.empty(nrow)
    for r in range(nrow):
        epi_A[r, :ng] = C[r] @ Gc; epi_A[r, ng:ng + nb] = C[r] @ Gb
        epi_A[r, ng + nb] = -1.0; epi_b[r] = float(t[r] - C[r] @ c)
    obj = np.zeros(nv); obj[ng + nb] = 1.0
    A_ub_rows = [sp.csr_matrix(epi_A)]; b_ub = [epi_b]
    if A_ub_base is not None:
        A_ub_rows.append(sp.hstack(
            [A_ub_base, sp.csr_matrix((A_ub_base.shape[0], 1))],
            format="csr"))
        b_ub.append(bl)
    Aeq2 = (sp.hstack([A_eq_base, sp.csr_matrix((A_eq_base.shape[0], 1))], format="csr")
            if A_eq_base is not None else None)
    rr = linprog(obj, A_ub=sp.vstack(A_ub_rows, format="csr"), b_ub=np.concatenate(b_ub),
                 A_eq=Aeq2, b_eq=(be if A_eq_base is not None else None),
                 bounds=[(-1, 1)] * (ng + nb) + [(-1e12, 1e12)], method="highs")
    return (float(rr.fun), rr.x[:ng + nb]) if rr.success else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_pos", nargs="?")
    ap.add_argument("iid_pos", nargs="?", type=int)
    ap.add_argument("--bench", dest="bench_opt")
    ap.add_argument("--iid", dest="iid_opt", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cap", type=int, default=200000, help="_HZ_MAX_INPUT_DIM override")
    ap.add_argument("--milp-timeout", type=float, default=15.0)
    ap.add_argument("--mem-gb", type=float, default=10.0, help="RLIMIT_AS cap (cpu device only)")
    ap.add_argument("--cell-budget", type=int, default=0,
                    help="override HybridzTF._hz_cell_budget for controlled carry experiments")
    ap.add_argument("--relu-valid-cuts", action="store_true",
                    help="add redundant exact ReLU graph facets as solver cuts")
    ap.add_argument("--compressed-relu", action="store_true",
                    help="opt-in exact ReLU slack projection; keeps exact eq_lagr semantics")
    ap.add_argument("--ort-samples", type=int, default=4000)
    ap.add_argument("--query-workers", type=int, default=0,
                    help="parallel pure per-query LP/MILP solves after one HZ propagation; 0 uses HZ_QUERY_WORKERS/1")
    ap.add_argument("--sigmoid-k", type=int, default=2,
                    help="piecewise segment count per side for HZ sigmoid (default: 2)")
    ap.add_argument("--tanh-k", type=int, default=1,
                    help="piecewise segment count per side for HZ tanh (default: 1)")
    ap.add_argument("--convex-maxpool", action="store_true",
                    help="use the old convex MaxPool relaxation; default is exact pairwise max")
    ap.add_argument("--no-tight", action="store_true",
                    help="disable tight_bounds (fast box) -- for nets whose LP-tight "
                         "[alpha,beta] pass blows memory with no CERT benefit (dist_shift sigmoid)")
    a = ap.parse_args()
    a.bench = a.bench_opt or a.bench_pos
    a.iid = a.iid_opt if a.iid_opt is not None else a.iid_pos
    if a.bench is None or a.iid is None:
        ap.error("bench/iid must be supplied either positionally or via --bench/--iid")

    res = {"bench": a.bench, "iid": a.iid, "verdict": "ERROR", "margin": None,
           "n_queries": None, "nc": None, "ng": None, "nb": None,
           "time_s": None, "verify_s": None, "device": a.device,
           "gt_cex": None, "p0": False, "err": None}
    t_start = time.time()
    try:
        if a.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # Hard CPU-mem backstop: a runaway prop dies with MemoryError in THIS
            # subprocess instead of ballooning the host (RLIMIT_AS = virtual mem).
            cap = int(a.mem_gb * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")
        sys.path.insert(0, str(REPO))
        import numpy as np, torch
        import act.back_end.solver.solver_hz as S
        S._HAS_GUROBI = False  # P3: engine verdict is scipy/HiGHS only
        import act.back_end.hybridz_tf.hybridz_tf as HZTF
        HZTF.HybridzTF._HZ_MAX_INPUT_DIM = a.cap
        HZTF.HybridzTF._HZ_MAX_AFFINE_DIM = max(a.cap, 8192)
        if a.cell_budget:
            HZTF.HybridzTF._hz_cell_budget = a.cell_budget
        elif a.bench == "dist_shift_2023" and int(a.sigmoid_k) >= 2:
            # K=2 is materially tighter on dist_shift's 784-wide sigmoid, but
            # the default 200M cell cap drops the HZ before the solver can use
            # it. 800M carries the K=2 HZ on the official instances under the
            # normal per-worker memory guard.
            HZTF.HybridzTF._hz_cell_budget = 800_000_000
        from pathlib import Path
        from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch, get_onnx_input_shape
        from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
        from act.front_end.verifiable_model import VerifiableModel, InputLayer, InputSpecLayer, OutputSpecLayer
        from act.front_end.spec_creator_base import LabeledInputTensor
        from act.pipeline.verification.torch2act import TorchToACT
        from act.back_end.transfer_functions import set_transfer_function_mode, get_transfer_function
        from act.back_end.core import Bounds as ABounds
        from act.back_end.solver.solver_hz_verdict import hz_objbound_decide

        dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
        base = BENCH_ROOT / a.bench
        rows = [l.strip().split(",") for l in open(base / "instances.csv") if l.strip()]
        o = base / rows[a.iid][0].replace("./", ""); v = base / rows[a.iid][1].replace("./", "")
        ish = tuple(get_onnx_input_shape(o)); pt = convert_onnx_to_pytorch(o).float().eval()
        lab = LabeledInputTensor(tensor=torch.zeros(ish, dtype=torch.float32), label=torch.tensor([0]))
        queries = parse_vnnlib_queries(v, labeled_tensor=lab)
        res["n_queries"] = len(queries)

        # Build one ACT net per query (cheap: only net[0] is propagated; the rest
        # share the identical pre-ASSERT structure, we read their ASSERT C/t).
        def build(qk):
            return TorchToACT(VerifiableModel(
                input_layer=InputLayer(labeled_input=lab, shape=ish, dtype=torch.float32),
                input_spec=InputSpecLayer(qk[0]), model=pt, output_spec=OutputSpecLayer(qk[1]))).run()
        net0 = build(queries[0])
        ins0 = queries[0][0]
        lb = ins0.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
        ub = ins0.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)

        # Propagate ONCE (tight_bounds = #1 precision lever; K=1, reduce = defaults)
        set_transfer_function_mode("hybridz"); tf = get_transfer_function()
        tf._hz_cache.clear(); tf._cache_net_id = None
        tf._relu_tight_bounds = not a.no_tight
        tf._sigmoid_K = max(1, int(a.sigmoid_k))
        tf._tanh_K = max(1, int(a.tanh_k))
        tf._maxpool_exact = not a.convex_maxpool
        tf._relu_compressed = bool(a.compressed_relu)
        tf._relu_valid_cuts = a.relu_valid_cuts or (
            os.environ.get("HZ_RELU_VALID_CUTS", "").strip().lower()
            in {"1", "true", "yes", "on"})
        ib = ABounds(lb=torch.tensor(lb, dtype=torch.float32, device=dev).reshape(1, -1),
                     ub=torch.tensor(ub, dtype=torch.float32, device=dev).reshape(1, -1))
        # verifier-time = HZ propagation + engine LP/MILP solve only. The witness
        # forward-eval ("find x*") and the ORT soundness audit are NOT the
        # verifier's job (it only decides IF an adv exists) and are excluded.
        _t_prop0 = time.time()
        after = {}
        for L in net0.layers:
            pr = net0.preds.get(L.id, []); inb = ib if (L.id == 0 or not pr) else after[pr[0]].bounds
            after[L.id] = tf.apply(L, inb, net0, {}, after)
        tf._relu_tight_bounds = False
        _verify_s = time.time() - _t_prop0   # accumulates engine solve below
        pid = net0.preds.get(net0.layers[-1].id, [None])[0]
        hz = tf._hz_cache.get(pid)
        res["hz_dropped"] = hz is None
        res["nc"] = int(hz.Ac.shape[0]) if hz is not None else None
        res["ng"] = int(hz.Gc.shape[1]) if hz is not None else None
        res["nb"] = int(hz.Gb.shape[1]) if hz is not None else None

        # If the HZ collapsed to None (e.g. cifar/ResNet wide conv exceeds the cell
        # budget -> interval fallback), no engine evidence is possible -> UNKNOWN.
        # Short-circuit BEFORE the costly multi-query spec build + ORT guard, which
        # are pure wasted wall-time for these instances (the analysis' cifar point).
        if hz is None:
            res["verdict"] = "UNKNOWN"
            res["time_s"] = round(time.time() - t_start, 2)
            res["verify_s"] = round(_verify_s, 2)
            print(json.dumps(res), flush=True)
            return

        center = (lb + ub) / 2.0; rad = (ub - lb) / 2.0
        inhz = next((tf._hz_cache.get(L.id) for L in net0.layers
                     if L.kind.upper() == "INPUT_SPEC"), None)
        full_ids = getattr(inhz, "full_col_ids", None) if inhz is not None else None
        if full_ids is not None:
            in_ids = full_ids.detach().cpu().numpy()
        elif inhz is not None and inhz.col_ids is not None and inhz.col_ids.numel() == lb.size:
            in_ids = inhz.col_ids.detach().cpu().numpy()
        else:
            in_ids = None
        pos = None
        if hz is not None and in_ids is not None and hz.col_ids is not None:
            pos = {int(v): k for k, v in enumerate(hz.col_ids.detach().cpu().numpy())}

        sess = None
        iname = None
        def _ort_sess():
            nonlocal sess, iname
            if sess is None:
                import onnxruntime as ort
                sess = ort.InferenceSession(str(o), providers=["CPUExecutionProvider"])
                iname = sess.get_inputs()[0].name
            return sess, iname
        def _net(x):
            sess, iname = _ort_sess()
            return sess.run(None, {iname: x.reshape(ish).astype(np.float32)})[0].reshape(-1)

        def _spec_from_net(nk):
            AL = nk.layers[-1]
            C = AL.params["C"].detach().cpu().numpy().astype(np.float64)
            t = AL.params["thresholds"].detach().cpu().numpy().astype(np.float64).reshape(-1)
            C = C.reshape(-1, C.shape[-1])
            if t.size == 1 and C.shape[0] != 1:
                t = np.repeat(t, C.shape[0])
            is_uns = "UNSAFE_LINEAR" in str(AL.params.get("kind"))
            return C, t, is_uns

        def _spec_from_output_spec(ospec):
            params = ospec.encode_linear(
                B=1,
                n_out=int(hz.c.numel()),
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            C = params["C"].detach().cpu().numpy().astype(np.float64)
            t = params["thresholds"].detach().cpu().numpy().astype(np.float64).reshape(-1)
            C = C.reshape(-1, C.shape[-1])
            if t.size == 1 and C.shape[0] != 1:
                t = np.repeat(t, C.shape[0])
            is_uns = "UNSAFE_LINEAR" in str(params.get("kind"))
            return C, t, is_uns

        def _spec_from_query(qk):
            try:
                return _spec_from_output_spec(qk[1])
            except Exception:
                return _spec_from_net(build(qk))

        def _is_cex(xi, C, t, is_uns):
            if xi is None or pos is None:
                return False
            xin = np.array([xi[pos[int(v)]] if int(v) in pos else 0.0 for v in in_ids])
            y = _net(center + rad * xin); cy = y @ C.T
            return bool((cy <= t + 1e-9).all()) if is_uns else bool((cy >= t - 1e-9).any())

        specs = []   # (C, t, is_unsafe_linear) per query
        for qk in queries:
            specs.append(_spec_from_query(qk))

        # ---- Unified pure CERTIFY + exact FALSIFY per disjunct.
        # The HZ margin solve in the unsafe direction returns BOTH the margin
        # (sound CERT decision) and an LP witness. With live binaries (nb>0), the
        # LP witness is only a relaxation witness, so pure mode does NOT accept it
        # as ADV; it escalates to the binary-exact MILP. The one exception is a
        # true zonotope (nb=0), where the LP is the exact HybridZ query and a
        # replayed witness is an engine-produced ADV.
        def _solve(C, t, is_uns):
            nonlocal _verify_s
            _ts = time.time()
            out = _hz_solve(hz, C, t, is_uns)
            _verify_s += time.time() - _ts   # engine solve counts; the _is_cex map does not
            return out

        TOL = 1e-9
        adv = False; all_cert = True; min_margin = float("inf"); any_margin = False
        query_workers = int(a.query_workers or os.environ.get("HZ_QUERY_WORKERS", "1") or 1)
        query_workers = max(1, min(query_workers, len(specs)))
        parallel_queries = query_workers > 1 and len(specs) > 1

        def _solve_one_spec(k, C, t, is_uns):
            q_t0 = time.time()
            m_lp, xi_lp = _hz_solve(hz, C, t, is_uns)
            lp_safe = (m_lp is not None) and ((m_lp > TOL) if is_uns else (m_lp < -TOL))
            if lp_safe:
                return {"k": k, "kind": "SAFE", "margin": float(m_lp),
                        "C": C, "t": t, "is_uns": is_uns, "xi": None,
                        "sec": round(time.time() - q_t0, 3)}
            lp_witness_is_engine_exact = (int(hz.Gb.shape[1]) == 0)
            if lp_witness_is_engine_exact:
                return {"k": k, "kind": "ADV_CAND", "margin": m_lp,
                        "C": C, "t": t, "is_uns": is_uns, "xi": xi_lp,
                        "sec": round(time.time() - q_t0, 3)}
            use_mip_start = os.environ.get("HZ_MILP_START", "").strip().lower() in {
                "1", "true", "yes", "on", "lp", "lp_binary", "lp-binary"
            }
            verdict, xi_mi = hz_objbound_decide(
                hz, C, t, is_unsafe_linear=is_uns,
                time_limit=a.milp_timeout,
                mip_start_xi=(xi_lp if use_mip_start else None))
            if verdict == "UNSAFE":
                return {"k": k, "kind": "ADV_CAND", "margin": m_lp,
                        "C": C, "t": t, "is_uns": is_uns, "xi": xi_mi,
                        "sec": round(time.time() - q_t0, 3)}
            if verdict == "SAFE":
                return {"k": k, "kind": "SAFE", "margin": None,
                        "C": C, "t": t, "is_uns": is_uns, "xi": None,
                        "sec": round(time.time() - q_t0, 3)}
            return {"k": k, "kind": "UNKNOWN", "margin": m_lp,
                    "C": C, "t": t, "is_uns": is_uns, "xi": None,
                    "sec": round(time.time() - q_t0, 3)}

        if parallel_queries:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            res["query_workers"] = query_workers
            _ts = time.time()
            q_results = []
            with ThreadPoolExecutor(max_workers=query_workers) as ex:
                futs = [
                    ex.submit(_solve_one_spec, k, C, t, is_uns)
                    for k, (C, t, is_uns) in enumerate(specs)
                ]
                for fut in as_completed(futs):
                    q_results.append(fut.result())
            _verify_s += time.time() - _ts
            for qr in sorted(q_results, key=lambda d: d["k"]):
                if qr["kind"] == "SAFE":
                    if qr["margin"] is not None:
                        min_margin = min(min_margin, qr["margin"]); any_margin = True
                    continue
                if qr["kind"] == "ADV_CAND" and _is_cex(qr["xi"], qr["C"], qr["t"], qr["is_uns"]):
                    adv = True; break
                all_cert = False
            if os.environ.get("HZ_QUERY_TIMINGS", "").strip().lower() in {"1", "true", "yes", "on"}:
                res["query_timings"] = [
                    {"k": int(qr["k"]), "kind": qr["kind"], "sec": qr.get("sec")}
                    for qr in sorted(q_results, key=lambda d: d["k"])
                ]
        else:
            for (C, t, is_uns) in specs:
                if hz is None:
                    all_cert = False; continue
                # LP pass: cheap certify + cheap falsify witness.
                m_lp, xi_lp = _solve(C, t, is_uns)
                lp_safe = (m_lp is not None) and ((m_lp > TOL) if is_uns else (m_lp < -TOL))
                if lp_safe:
                    min_margin = min(min_margin, m_lp); any_margin = True
                    continue
                lp_witness_is_engine_exact = (int(hz.Gb.shape[1]) == 0)
                if lp_witness_is_engine_exact and _is_cex(xi_lp, C, t, is_uns):
                    adv = True; break
                # MILP escalation via OBJBOUND (verdict-only, EXACT, early-stop). HiGHS
                # objective_target/objective_bound stop B&B once the SAFE/UNSAFE sign vs
                # the threshold is proven (mip_rel_gap=1e-9) -> 6x median speedup, verdict
                # identical to the full solve (validated 16/16, P0=0). UNSAFE returns the
                # worst-case witness, still forward-verified by _is_cex before ADV.
                _ts = time.time()
                use_mip_start = os.environ.get("HZ_MILP_START", "").strip().lower() in {
                    "1", "true", "yes", "on", "lp", "lp_binary", "lp-binary"
                }
                verdict, xi_mi = hz_objbound_decide(hz, C, t, is_unsafe_linear=is_uns,
                                                    time_limit=a.milp_timeout,
                                                    mip_start_xi=(xi_lp if use_mip_start else None))
                _verify_s += time.time() - _ts
                if verdict == "UNSAFE":
                    if _is_cex(xi_mi, C, t, is_uns):
                        adv = True; break
                    all_cert = False   # HZ-unsafe but not a real cex -> cannot certify
                elif verdict != "SAFE":  # UNKNOWN (objbound timed out) -> not certified
                    all_cert = False
                # verdict == "SAFE" -> this disjunct's unsafe set provably empty (certified)

        if adv:
            res["verdict"] = "ADV"; res["margin"] = None
        else:
            res["verdict"] = "CERT" if all_cert else "UNKNOWN"
            res["margin"] = (min_margin if any_margin and min_margin != float("inf") else None)
        res["time_s"] = round(time.time() - t_start, 2)   # total wall (drives timeout)
        res["verify_s"] = round(_verify_s, 2)              # verifier engine time only

        # ORT soundness guard (downgrade only): a sample is a real cex iff it lands
        # in ANY disjunct's unsafe set. P0 = engine CERT but a real cex exists.
        try:
            sess, iname = _ort_sess()
            rng = np.random.RandomState(0)
            X = np.vstack([lb, ub, (lb + ub) / 2,
                           rng.uniform(lb, ub, size=(a.ort_samples, lb.size))]).astype(np.float32)
            try:
                Y = sess.run(None, {iname: X.reshape((-1,) + ish[1:])})[0].reshape(X.shape[0], -1)
            except Exception:
                Y = np.stack([sess.run(None, {iname: x.reshape(ish)})[0].reshape(-1) for x in X])
            gt_cex = False
            for (C, t, is_uns) in specs:
                if is_uns:  # disjunct unsafe = all rows C[r]y <= t[r]
                    inu = np.ones(X.shape[0], bool)
                    for r in range(C.shape[0]):
                        inu &= (Y @ C[r] <= t[r] + 1e-9)
                    if inu.any(): gt_cex = True; break
                else:       # TOP1/ALL-rows unsafe = some row C[r]y >= t[r]
                    vio = np.zeros(X.shape[0], bool)
                    for r in range(C.shape[0]):
                        vio |= (Y @ C[r] >= t[r] - 1e-9)
                    if vio.any(): gt_cex = True; break
            res["gt_cex"] = gt_cex
            if res["verdict"] == "CERT" and gt_cex:
                res["p0"] = True; res["verdict"] = "UNKNOWN_P0DOWNGRADE"
        except Exception as e:
            res["gt_cex"] = None; res["err_ort"] = str(e)[:120]
    except MemoryError:
        res["verdict"] = "UNKNOWN"
        res["err"] = "MemoryError (exact-HZ capacity / RLIMIT_AS cap hit)"
        res["time_s"] = round(time.time() - t_start, 2)
    except Exception as e:
        import traceback
        msg = str(e)
        if ("DefaultCPUAllocator" in msg or "Cannot allocate memory" in msg
                or "RLIMIT_AS" in msg):
            res["verdict"] = "UNKNOWN"
            res["err"] = f"{type(e).__name__}: exact-HZ capacity ({msg[:160]})"
        else:
            res["verdict"] = "ERROR"
            res["err"] = f"{type(e).__name__}: {msg[:200]}"
        res["time_s"] = round(time.time() - t_start, 2)
        print(traceback.format_exc()[:1500], file=sys.stderr)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
