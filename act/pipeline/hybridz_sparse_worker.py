#!/usr/bin/env python
"""Opt-in SPARSE-HZ worker (runnable 'hybridz_sparse' mode).

Wraps the packaged sparse-CSR exact-HZ propagation module for the
*representation-drop* conv benches (malbeware/relusplitter/cgan) that the dense
production path drops to UNKNOWN/ERROR. Sound by construction:

  * base HZ feasible + all HZ∩unsafe disjuncts EMPTY   -> CERT  (no ORT needed)
  * any disjunct has an exact-HZ unsafe witness         -> replay that exact
        sparse-HZ MILP witness with ORT; ADV iff confirmed, else UNKNOWN
  * any disjunct times out                              -> UNKNOWN

The sparse path NEVER produces a CERT for an instance with an unverified witness,
and NEVER an ADV without ORT replay. P0-safe. Prints exactly one JSON line.
"""
import os, sys, json, time, re, signal, subprocess, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
PROBE_MODULE = "act.pipeline.hybridz_sparse_exact_probe"
ACTIVE_CHILD_PGIDS = set()


def _kill_active_children():
    for pgid in list(ACTIVE_CHILD_PGIDS):
        try:
            os.killpg(int(pgid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass


def _signal_cleanup(signum, frame):
    _kill_active_children()
    raise SystemExit(128 + int(signum))


signal.signal(signal.SIGTERM, _signal_cleanup)
signal.signal(signal.SIGINT, _signal_cleanup)


def parse_big(s):
    if s is None:
        return None
    s = str(s).strip().lower()
    try:
        if s.endswith("k"):
            return float(s[:-1]) * 1000.0
        return float(s)
    except ValueError:
        return None


def run_sparse(
    bench,
    iid,
    milp_timeout,
    lp_queries,
    cutoff_row,
    check_witness=False,
    compressed_relu=False,
    relu_cuts=False,
    highs_options=None,
    elim_eq_subst=False,
    skip_lp_before_milp=False,
    fbbt_passes=0,
    relax_precheck_timeout=0.0,
    mip_solver="highs",
    compressed_sigmoid=False,
    sigmoid_prune_degenerate=False,
    sigmoid_k=None,
    tanh_k=None,
    scurve_domain_cuts=False,
    scurve_graph_cuts=False,
    scurve_grid="uniform",
    query_indices="",
    connected_presolve=False,
    mip_start="none",
    elim_singletons=True,
    worker_deadline=None,
):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    summary_path = Path(tempfile.gettempdir()) / (
        f"hz_sparse_probe_{os.getpid()}_{bench}_{iid}_{time.time_ns()}.json"
    )
    cmd = [PY, "-m", PROBE_MODULE, "--bench", bench, "--iid", str(iid), "--device", "cpu",
           "--lp-queries", str(lp_queries), "--lp-timeout", "15",
           "--milp-all", "--milp-cutoff", "--milp-timeout", str(milp_timeout),
           "--summary-json", str(summary_path)]
    if query_indices:
        cmd.extend(["--query-indices", str(query_indices)])
    if elim_singletons:
        cmd.append("--elim-singletons")
    if mip_solver:
        cmd.extend(["--mip-solver", str(mip_solver)])
    if connected_presolve:
        cmd.append("--connected-presolve")
    if mip_start and str(mip_start) != "none":
        cmd.extend(["--mip-start", str(mip_start)])
    if check_witness:
        cmd.extend(["--check-witness", "--check-milp-witness-only", "--stop-on-unsafe"])
    if cutoff_row:
        cmd.append("--cutoff-as-row")
    if compressed_relu:
        cmd.append("--compressed-relu")
    if relu_cuts:
        cmd.append("--relu-cuts")
    if elim_eq_subst:
        cmd.append("--elim-eq-subst")
    if skip_lp_before_milp:
        cmd.append("--skip-lp-before-milp")
    if int(fbbt_passes) > 0:
        cmd.extend(["--fbbt-passes", str(int(fbbt_passes))])
    if float(relax_precheck_timeout) > 0.0:
        cmd.extend(["--relax-precheck-timeout", str(float(relax_precheck_timeout))])
    if compressed_sigmoid:
        cmd.append("--compressed-sigmoid")
    if sigmoid_prune_degenerate:
        cmd.append("--sigmoid-prune-degenerate")
    if sigmoid_k is not None:
        cmd.extend(["--sigmoid-k", str(int(sigmoid_k))])
    if tanh_k is not None:
        cmd.extend(["--tanh-k", str(int(tanh_k))])
    if scurve_domain_cuts:
        cmd.append("--scurve-domain-cuts")
    if scurve_graph_cuts:
        cmd.append("--scurve-graph-cuts")
    if scurve_grid and str(scurve_grid) != "uniform":
        cmd.extend(["--scurve-grid", str(scurve_grid)])
    for opt in highs_options or []:
        cmd.extend(["--highs-option", str(opt)])
    if worker_deadline is not None and time.time() >= float(worker_deadline):
        raise subprocess.TimeoutExpired(cmd, 0.0)
    p = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    ACTIVE_CHILD_PGIDS.add(p.pid)
    try:
        timeout = milp_timeout * 26 + 60
        if worker_deadline is not None:
            timeout = min(timeout, max(0.1, float(worker_deadline) - time.time()))
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            p.communicate(timeout=2)
        except Exception:
            pass
        raise
    finally:
        ACTIVE_CHILD_PGIDS.discard(p.pid)
    try:
        probe_summary = json.loads(summary_path.read_text())
    except Exception:
        probe_summary = {}
    try:
        summary_path.unlink(missing_ok=True)
    except Exception:
        pass
    m = re.search(
        r"verdict_summary checked=(\d+) cert=(\d+) hz_unsafe=(\d+) unknown=(\d+)(?: real_adv=(\d+))?",
        out,
    )
    hz = re.search(r"final sparse HZ:.*?ng=([\d.]+k?) nb=([\d.]+k?) nc=([\d.]+k?)", out)
    total_m = re.search(r"interval_hard=\d+/(\d+)", out)
    base_m = re.search(r"base_hz_feasible=(True|False) msg=(.*)", out)
    unsupported_m = re.search(r"EARLY_STOP unsupported_layer ([^\n]+)", out)
    if not m:
        combined = (out or "") + "\n" + (err or "")
        unsupported = re.search(r"NotImplementedError: ([^\n]+)", combined)
        if unsupported:
            return {"checked": 0, "cert": 0, "hz_unsafe": 0, "unknown": 1,
                    "real_adv": 0, "total": None, "hz": None,
                    "base_hz_feasible": None,
                    "base_hz_feas_msg": unsupported.group(1),
                    "unsupported": unsupported.group(1)}
        return None
    chk, cert, uns, unk = map(int, m.groups()[:4])
    real_adv = int(m.group(5) or 0)
    total = int(total_m.group(1)) if total_m else None
    query_audit = []
    for qr in probe_summary.get("query_results") or []:
        ms = qr.get("milp_stats") or {}
        query_audit.append({
            "q": qr.get("q"),
            "verdict": qr.get("verdict"),
            "cert_source": qr.get("cert_source"),
            "real_unsafe": qr.get("real_unsafe"),
            "witness_checked": qr.get("witness_checked"),
            "milp_status": qr.get("milp_status"),
            "milp_margin": qr.get("milp_margin"),
            "milp_sec": qr.get("milp_sec"),
            "milp_solver_status": ms.get("status"),
            "incumbent_validation": ms.get("incumbent_validation"),
        })
    return {"checked": chk, "cert": cert, "hz_unsafe": uns, "unknown": unk,
            "real_adv": real_adv, "total": total, "hz": hz.group(0) if hz else None,
            "n_cont": parse_big(hz.group(1)) if hz else None,
            "n_bin": parse_big(hz.group(2)) if hz else None,
            "n_eq": parse_big(hz.group(3)) if hz else None,
            "base_hz_feasible": (base_m.group(1) == "True") if base_m else None,
            "base_hz_feas_msg": base_m.group(2).strip() if base_m else (unsupported_m.group(1).strip() if unsupported_m else None),
            "unsupported": unsupported_m.group(1).strip() if unsupported_m else None,
            "probe_summary_status": probe_summary.get("instance_status"),
            "query_audit": query_audit}


def is_full_cert(s):
    return (
        s is not None
        and s.get("base_hz_feasible") is True
        and s.get("total") is not None
        and s["checked"] == s["total"]
        and s["cert"] == s["total"]
        and s["hz_unsafe"] == 0
        and s["unknown"] == 0
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_pos", nargs="?")
    ap.add_argument("iid_pos", nargs="?", type=int)
    ap.add_argument("--bench", dest="bench_opt")
    ap.add_argument("--iid", dest="iid_opt", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--milp-timeout", type=float, default=40.0)
    ap.add_argument("--lp-queries", type=int, default=99)
    ap.add_argument("--cutoff-row", action="store_true",
                    help="use cutoff-as-row formulation in the sparse MILP")
    ap.add_argument("--compressed-relu", action="store_true",
                    help="use exact compressed eq_lagr ReLU slack projection in the sparse probe")
    ap.add_argument("--relu-cuts", action="store_true",
                    help="add redundant exact-valid ReLU graph cuts in the sparse probe")
    ap.add_argument("--highs-option", action="append", default=[],
                    help="extra HiGHS option passed to the sparse probe")
    ap.add_argument("--elim-eq-subst", action="store_true",
                    help="use sparse exact equality-substitution presolve in the probe")
    ap.add_argument("--skip-lp-before-milp", action="store_true",
                    help="skip sparse LP precheck and run the exact MILP query directly")
    ap.add_argument("--fbbt-passes", type=int, default=0,
                    help="run N exact FBBT presolve passes in the sparse probe")
    ap.add_argument("--relax-precheck-timeout", type=float, default=0.0,
                    help="run continuous-relaxation EMPTY-only precheck in the sparse probe")
    ap.add_argument("--mip-solver", choices=["highs", "scip"], default="highs",
                    help="MILP backend used by the sparse probe")
    ap.add_argument("--compressed-sigmoid", action="store_true",
                    help="use exact compressed S-curve slack projection in the sparse probe")
    ap.add_argument("--sigmoid-prune-degenerate", action="store_true",
                    help="drop degenerate S-curve segments in compressed mode")
    ap.add_argument("--sigmoid-k", type=int, default=None,
                    help="sigmoid segments per side passed to the sparse probe")
    ap.add_argument("--tanh-k", type=int, default=None,
                    help="tanh segments per side passed to the sparse probe")
    ap.add_argument("--scurve-domain-cuts", action="store_true",
                    help="add exact-valid selected-segment domain/range cuts")
    ap.add_argument("--scurve-graph-cuts", action="store_true",
                    help="add exact-valid selected-segment graph cuts")
    ap.add_argument("--scurve-grid", choices=["uniform", "curvature"], default="uniform",
                    help="S-curve segment grid passed to the sparse probe")
    ap.add_argument("--query-indices", default="",
                    help="fixed comma-separated query/disjunct order passed to the sparse probe")
    ap.add_argument("--connected-presolve", action="store_true",
                    help="keep only the MILP constraint component connected to the active margin")
    ap.add_argument("--mip-start", choices=["none", "lp-round", "lp-binary-round", "base", "base-binary"], default="none",
                    help="MILP start passed through to the sparse probe")
    ap.add_argument("--no-elim-singletons", action="store_true",
                    help="disable the sparse worker's default exact singleton projection")
    ap.add_argument("--worker-timeout", type=float, default=0.0,
                    help="hard wall in seconds for this sparse worker and its child probes")
    a = ap.parse_args()
    a.bench = a.bench_opt or a.bench_pos
    a.iid = a.iid_opt if a.iid_opt is not None else a.iid_pos
    if a.bench is None or a.iid is None:
        ap.error("bench/iid must be supplied either positionally or via --bench/--iid")
    t0 = time.time()
    res = {"bench": a.bench, "iid": a.iid, "verdict": "ERROR", "p0": False,
           "gt_cex": None, "mode": "sparse", "time_s": None, "err": None}
    try:
        worker_deadline = (
            time.time() + float(a.worker_timeout)
            if float(a.worker_timeout or 0.0) > 0.0
            else None
        )

        def call_sparse(*args, **kwargs):
            return run_sparse(*args, **kwargs, worker_deadline=worker_deadline)

        witness_cutoff_row = a.cutoff_row
        if a.bench in {"malbeware", "relusplitter", "metaroom_2023"}:
            # For sparse exact-HZ ADVs on metaroom/malbeware/relusplitter, HiGHS'
            # objective-target mode finds primal unsafe witnesses quickly, while
            # cutoff-row feasibility can stall at root. Keep cutoff-row for the
            # CERT pass below, where it is useful for proving EMPTY.
            witness_cutoff_row = False
        relusplitter_tail_cert_first = (
            a.bench == "relusplitter"
            and int(a.iid) >= 140
            and a.elim_eq_subst
            and a.skip_lp_before_milp
        )
        if relusplitter_tail_cert_first:
            s = call_sparse(
                a.bench,
                a.iid,
                a.milp_timeout,
                a.lp_queries,
                a.cutoff_row,
                check_witness=False,
                compressed_relu=a.compressed_relu,
                relu_cuts=a.relu_cuts,
                highs_options=a.highs_option,
                elim_eq_subst=a.elim_eq_subst,
                skip_lp_before_milp=a.skip_lp_before_milp,
                fbbt_passes=a.fbbt_passes,
                relax_precheck_timeout=a.relax_precheck_timeout,
                mip_solver=a.mip_solver,
                compressed_sigmoid=a.compressed_sigmoid,
                sigmoid_prune_degenerate=a.sigmoid_prune_degenerate,
                sigmoid_k=a.sigmoid_k,
                tanh_k=a.tanh_k,
                scurve_domain_cuts=a.scurve_domain_cuts,
                scurve_graph_cuts=a.scurve_graph_cuts,
                scurve_grid=a.scurve_grid,
                query_indices=a.query_indices,
                connected_presolve=a.connected_presolve,
                mip_start=a.mip_start,
                elim_singletons=not a.no_elim_singletons,
            )
            if is_full_cert(s):
                res.update({"verdict": "CERT", "hz": s["hz"], "sparse": s,
                            "cert_first": True})
                res["time_s"] = round(time.time() - t0, 1)
                print(json.dumps(res), flush=True)
                return
            if s is not None and s.get("hz_unsafe", 0) > 0:
                sw_tail = call_sparse(
                    a.bench,
                    a.iid,
                    a.milp_timeout,
                    a.lp_queries,
                    cutoff_row=False,
                    check_witness=True,
                    compressed_relu=a.compressed_relu,
                    relu_cuts=a.relu_cuts,
                    highs_options=a.highs_option,
                    elim_eq_subst=a.elim_eq_subst,
                    skip_lp_before_milp=a.skip_lp_before_milp,
                    fbbt_passes=a.fbbt_passes,
                    relax_precheck_timeout=a.relax_precheck_timeout,
                    mip_solver=a.mip_solver,
                    compressed_sigmoid=a.compressed_sigmoid,
                    sigmoid_prune_degenerate=a.sigmoid_prune_degenerate,
                    sigmoid_k=a.sigmoid_k,
                    tanh_k=a.tanh_k,
                    scurve_domain_cuts=a.scurve_domain_cuts,
                    scurve_graph_cuts=a.scurve_graph_cuts,
                    scurve_grid=a.scurve_grid,
                    query_indices=a.query_indices,
                    connected_presolve=a.connected_presolve,
                    mip_start=a.mip_start,
                    elim_singletons=not a.no_elim_singletons,
                )
                if sw_tail is not None and sw_tail.get("real_adv", 0) > 0:
                    res.update({"verdict": "ADV", "gt_cex": True,
                                "ort_verified": True, "sparse": sw_tail,
                                "cert_probe": s, "cert_first": True})
                    res["time_s"] = round(time.time() - t0, 1)
                    print(json.dumps(res), flush=True)
                    return
            res.update({"verdict": "UNKNOWN", "sparse": s, "cert_first": True})
            res["time_s"] = round(time.time() - t0, 1)
            print(json.dumps(res), flush=True)
            return
        sw = call_sparse(
            a.bench,
            a.iid,
            a.milp_timeout,
            a.lp_queries,
            witness_cutoff_row,
            check_witness=True,
            compressed_relu=a.compressed_relu,
            relu_cuts=a.relu_cuts,
            highs_options=a.highs_option,
            elim_eq_subst=a.elim_eq_subst,
            skip_lp_before_milp=a.skip_lp_before_milp,
            fbbt_passes=a.fbbt_passes,
            relax_precheck_timeout=a.relax_precheck_timeout,
            mip_solver=a.mip_solver,
            compressed_sigmoid=a.compressed_sigmoid,
            sigmoid_prune_degenerate=a.sigmoid_prune_degenerate,
            sigmoid_k=a.sigmoid_k,
            tanh_k=a.tanh_k,
            scurve_domain_cuts=a.scurve_domain_cuts,
            scurve_graph_cuts=a.scurve_graph_cuts,
            scurve_grid=a.scurve_grid,
            query_indices=a.query_indices,
            connected_presolve=a.connected_presolve,
            mip_start=a.mip_start,
            elim_singletons=not a.no_elim_singletons,
        )
        if sw is not None and sw.get("real_adv", 0) > 0:
            res.update({"verdict": "ADV", "gt_cex": True, "ort_verified": True,
                        "sparse": sw, "witness_probe": sw})
            res["time_s"] = round(time.time() - t0, 1)
            print(json.dumps(res), flush=True)
            return
        if is_full_cert(sw):
            res.update({"verdict": "CERT", "hz": sw["hz"], "sparse": sw,
                        "witness_probe": sw})
            res["time_s"] = round(time.time() - t0, 1)
            print(json.dumps(res), flush=True)
            return
        if (a.bench == "tllverifybench_2023" and sw is not None
                and (sw.get("n_bin") is not None and sw["n_bin"] <= 3000.0)):
            # Some small TLL cases (notably iid10) need HiGHS objective-target
            # search to find a primal exact-HZ witness, while iid11/13 are found
            # by cutoff-row.  Restrict this second profile to small sparse HZs so
            # large TLL instances do not spend another long root-MIP timeout.
            sw_obj = call_sparse(
                a.bench, a.iid, max(float(a.milp_timeout), 120.0), a.lp_queries,
                cutoff_row=False, check_witness=True,
                compressed_relu=a.compressed_relu,
                relu_cuts=a.relu_cuts,
                highs_options=a.highs_option,
                elim_eq_subst=a.elim_eq_subst,
                skip_lp_before_milp=a.skip_lp_before_milp,
                fbbt_passes=a.fbbt_passes,
                relax_precheck_timeout=a.relax_precheck_timeout,
                mip_solver=a.mip_solver,
                compressed_sigmoid=a.compressed_sigmoid,
                sigmoid_prune_degenerate=a.sigmoid_prune_degenerate,
                sigmoid_k=a.sigmoid_k,
                tanh_k=a.tanh_k,
                scurve_domain_cuts=a.scurve_domain_cuts,
                scurve_graph_cuts=a.scurve_graph_cuts,
                scurve_grid=a.scurve_grid,
                query_indices=a.query_indices,
                connected_presolve=a.connected_presolve,
                mip_start=a.mip_start,
                elim_singletons=not a.no_elim_singletons,
            )
            if sw_obj is not None and sw_obj.get("real_adv", 0) > 0:
                res.update({"verdict": "ADV", "gt_cex": True, "ort_verified": True,
                            "sparse": sw_obj, "witness_probe": sw_obj,
                            "witness_probe_cutoff": sw})
                res["time_s"] = round(time.time() - t0, 1)
                print(json.dumps(res), flush=True)
                return
            if sw_obj is not None:
                res["witness_probe_objtarget"] = sw_obj

        s = call_sparse(
            a.bench,
            a.iid,
            a.milp_timeout,
            a.lp_queries,
            a.cutoff_row,
            check_witness=False,
            compressed_relu=a.compressed_relu,
            relu_cuts=a.relu_cuts,
            highs_options=a.highs_option,
            elim_eq_subst=a.elim_eq_subst,
            skip_lp_before_milp=a.skip_lp_before_milp,
            fbbt_passes=a.fbbt_passes,
            relax_precheck_timeout=a.relax_precheck_timeout,
            mip_solver=a.mip_solver,
            compressed_sigmoid=a.compressed_sigmoid,
            sigmoid_prune_degenerate=a.sigmoid_prune_degenerate,
            sigmoid_k=a.sigmoid_k,
            tanh_k=a.tanh_k,
            scurve_domain_cuts=a.scurve_domain_cuts,
            scurve_graph_cuts=a.scurve_graph_cuts,
            scurve_grid=a.scurve_grid,
            query_indices=a.query_indices,
            connected_presolve=a.connected_presolve,
            mip_start=a.mip_start,
            elim_singletons=not a.no_elim_singletons,
        )
        if s is None:
            res["err"] = "sparse-noparse"
        elif is_full_cert(s):
            res.update({"verdict": "CERT", "hz": s["hz"], "sparse": s})  # all EMPTY -> SAFE, sound
        elif s["hz_unsafe"] > 0:
            # The sparse engine found an exact-HZ unsafe point in the non-replay
            # pass, but the replay pass above did not confirm a MILP witness as a
            # real network cex. ORT/dense fallback must not upgrade UNKNOWN here:
            # pure ADV requires the exact sparse-HZ MILP witness itself to replay.
            res.update({"verdict": "UNKNOWN", "ort_verified": False,
                        "sparse": s, "hz_unsafe_unreplayed": True})
        else:
            res.update({"verdict": "UNKNOWN", "sparse": s})
        if sw is not None:
            res["witness_probe"] = sw
    except subprocess.TimeoutExpired:
        res["verdict"] = "TIMEOUT"
    except Exception as e:
        res["verdict"] = "ERROR"; res["err"] = f"{type(e).__name__}:{str(e)[:100]}"
    res["time_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
