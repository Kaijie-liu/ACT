"""Fixed A1 interfaces only; no trained source, scalar proof or effect test.

Top-level imports and receive() are portable standard-library code. Native
imports occur only inside run(), after the frozen supervisor installs limits.
"""
import copy
from fractions import Fraction as F
import hashlib
import time

from scoped_source.hz_request_store import compact, decode, digest, require, Work, rational
from scoped_source.development_work import DevelopmentWork, POLICY
from scoped_source import check_margin_hz_source as check
from scoped_source.check_gate_margin_rows import freeze, MARGIN_PROFILE, source

CASES = ["full_height_ten_class_shared_private_source", "exact_properties_and_objectives",
         "native_candidates_exact_reception", "source_scope_map_direction_faults",
         "property_and_candidate_faults", "malformed_csr_and_live_alias",
         "capacity_work_bits_deadline_and_partial"]
GOALS = ((F(0), F(0)), (F(1, 2), F(-1, 4)), (F(1), F(1, 4)), (F(1, 2), F(0)))
PROPERTIES = [dict(q=[str(F((-1)**i*(i+1), 16)) for i in range(10)], constant="1/8"),
              dict(q=["0"]*8+["1", "-1"], constant="-1/4")]
SCOPE = {k: hashlib.sha256(("A1_SYNTHETIC_20261021_"+k).encode()).hexdigest()
         for k in ("request", "domain", "guard", "model", "input")}


def csr(nr, nc, entries):
    rows = [{} for _ in range(nr)]
    for i, j, v in entries:
        require(j not in rows[i] and 0 <= j < nc and v != 0, "fixture unique coefficient")
        rows[i][j] = float(v)
    data, ids, ptr = [], [], [0]
    for row in rows:
        for j, v in sorted(row.items()):
            ids.append(j); data.append(v)
        ptr.append(len(data))
    return dict(shape=[nr, nc], data=data, indices=ids, indptr=ptr)


def fixtures():
    """Original source descriptors independent of the native joint constructor."""
    sc = 3072
    def hz(c, nc, nb, gc, gb, eqc, eqb, b, lec, leb, ub):
        return dict(c=list(map(float, c)), b=list(map(float, b)), ub=list(map(float, ub)),
                    Gc=csr(len(c), nc, gc), Gb=csr(len(c), nb, gb),
                    Ac=csr(len(b), nc, eqc), Ab=csr(len(b), nb, eqb),
                    Auc=csr(len(ub), nc, lec), Aub=csr(len(ub), nb, leb),
                    frame_id=20261021, exact=True)
    # Router rows occupy entry equality1 / inequality1, not an implicit identity map.
    eqc, eqb, b = [(0, 1, 1), (1, 0, 1)], [], [0, 0]
    lec, leb, ub = [(0, 2, 1), (1, 0, 1)], [(1, 0, 1/4)], [3/4, 1/2]
    entry = hz([1/8]*sc, sc, 1, [(i, i, 1/16) for i in range(sc)], [(0, 0, 1/32)],
               eqc, eqb, b, lec, leb, ub)
    router = hz([1/4, -1/8, -1/4], sc, 1,
                [(0, 0, 1/4), (0, 2, 1/8), (1, 1, 1/2), (2, 2, -1/4)],
                [(0, 0, 1/8), (2, 0, -1/16)], [(0, 0, 1)], [], [0],
                [(0, 0, 1)], [(0, 0, 1/4)], [1/2])
    experts = []
    for side, private in enumerate((1, 2)):
        gc, gb = [], []
        for i in range(10):
            gc.extend([(i, i, (i+1)/32), (i, sc+private-1, (-1)**(i+side)/16)])
            gb.extend([(i, 0, (i+1)/64), (i, private, (-1)**(i+side)/32)])
        experts.append(hz([(i+1+side)/8 for i in range(10)], sc+private, 1+private,
                          gc, gb, eqc+[(2, sc+private-1, 1)], [(2, private, -1/4)], b+[0],
                          lec+[(2, sc+private-1, -1)], leb+[(2, private, 1/2)], ub+[3/4]))
    return dict(entry=entry, a=experts[0], b=experts[1], router=router)


def rejected(fn, text):
    try:
        fn()
    except (ValueError, TimeoutError) as exc:
        require(text in str(exc), "wrong rejection: "+repr(exc))
        return dict(type=type(exc).__name__, reason=str(exc))
    raise AssertionError("fault was accepted: "+text)


def receive(value, work):
    """Externally fix source descriptors, properties and all goals before checking."""
    require(value["status"] == "PASS" and value["policy"] == POLICY,
            "complete successful candidate")
    require([r["name"] for r in value["cases"]] == CASES
            and all(r["status"] == "PASS" for r in value["cases"]), "fixed complete cases")
    bundle = value["source"]
    require(bundle["scope"] == SCOPE and bundle["base"]["context"]["pair"] == [0, 2],
            "frozen synthetic source scope")
    originals = fixtures()
    require(all(bundle["base"]["sources"][k] == v for k, v in originals.items()),
            "frozen original source coefficients")
    require(bundle["base"]["router_eq_rows"] == [1]
            and bundle["base"]["router_le_rows"] == [1], "frozen retained row maps")
    raw = compact(bundle)
    require(digest(raw) == value["source_sha256"], "anchored synthetic source")
    require(len(value["supports"]) == len(PROPERTIES), "all property receipts")
    receipts = []
    for prop, item in zip(PROPERTIES, value["supports"]):
        require(item["view"]["q"] == prop["q"] and item["view"]["constant"] == prop["constant"],
                "frozen complete property")
        receipt = check.check_supports(raw, compact(item["view"]), item["candidates"],
                                      expected_source=value["source_sha256"],
                                      expected_property=item["property_sha256"],
                                      expected_goals=GOALS, work=work)
        require(receipt == item["receipt"], "independent native receipt")
        receipts.append(receipt)
    work.tick()
    return receipts


def direct_objective(hz, prop, t, nu, work):
    """Coordinate sum from native output buffers, independent of b/d/m view logic."""
    q = tuple(F(v) for v in prop["q"])
    weights = tuple(t*v for v in q)+tuple((1-t)*v for v in q)+(nu, -nu)
    center, row = F(prop["constant"]), {}
    for i, weight in enumerate(weights):
        work.visit("row_visits")
        center = work.exact(center+work.exact(weight*F(float(hz.c[i]))))
        for matrix, shift in ((hz.Gc, 0), (hz.Gb, hz.n_cont)):
            for p in range(matrix.indptr[i], matrix.indptr[i+1]):
                work.visit("coefficient_visits")
                j = shift+int(matrix.indices[p])
                row[j] = work.exact(row.get(j, F(0))+work.exact(weight*F(float(matrix.data[p]))))
    return center, {j: v for j, v in row.items() if v}


def run(deadline, publish):
    """One frozen native worker; unexpected failure preserves the completed prefix."""
    work = DevelopmentWork(deadline, policy=POLICY)
    result = dict(status="FAIL", policy=POLICY, deadline=deadline, cases=[], supports=[],
                  native_candidate_calls=0, scalar_aggregate_checked=False, complete_request=False)
    start = time.monotonic()
    def passed(name, **detail):
        work.tick()
        require(name == CASES[len(result["cases"])], "fixed control order")
        result["cases"].append(dict(name=name, status="PASS", detail=detail,
                                    elapsed=time.monotonic()-start))
        publish(result, work)
    try:
        import numpy as np
        import scipy.sparse as sp
        import torch
        import importlib.metadata
        from act.back_end.solver.solver_hz import SparseHZono
        from act.back_end.moe.margin_hz_source import bind_source, propose_supports
        from act.back_end.moe.gate_margin_rows import snapshot
        torch.set_num_threads(1)
        result["native_import_seconds"] = time.monotonic()-start
        result["versions"] = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch")}
        originals = fixtures()
        def native(obj):
            args = dict(obj)
            for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                m = args[key]
                args[key] = sp.csr_matrix((m["data"], m["indices"], m["indptr"]),
                                          shape=m["shape"], dtype=np.float64)
            return SparseHZono(**args)
        live = {k: native(v) for k, v in originals.items()}
        def bind(pair=(0, 2), maps=(1,)):
            return bind_source(entry=live["entry"], expert_a=live["a"], expert_b=live["b"],
                               router=live["router"], pair=pair, scope=SCOPE,
                               router_eq_rows=maps, router_le_rows=(1,), work=work)
        captured = time.monotonic()
        bound = bind()
        result.update(source=decode(bound.payload), source_sha256=bound.sha256,
                      capture_seconds=time.monotonic()-captured)
        require(bound.output_hz.n_out == 22 and bound.output_hz.n_cont == 3075
                and bound.output_hz.n_bin == 4, "full source dimensions")
        passed(CASES[0], classes=10, input_rows=3072, factors=3079, relaxed_binary=4)
        views = []
        for prop in PROPERTIES:
            raw, anchor = check.make_property(bound.payload, expected_source=bound.sha256,
                                               q=prop["q"], constant=prop["constant"], work=work)
            parsed = check.inspect_property(bound.payload, raw, expected_source=bound.sha256,
                                            expected_property=anchor, work=work)
            for t, nu in GOALS:
                require(check.objective(parsed, t, nu, work)
                        == direct_objective(bound.output_hz, prop, t, nu, work), "direct rational projection")
            views.append((raw, anchor))
        passed(CASES[1], properties=2, exact_objectives=8, last_two_classes_retained=True)
        for raw, anchor in views:
            t0 = time.monotonic()
            result["native_candidate_calls"] += 1
            generated = propose_supports(bound, raw, expected_property=anchor, goals=GOALS, work=work)
            result["supports"].append(dict(view=decode(raw), property_sha256=anchor,
                                            candidates=generated["candidates"], receipt=generated["receipt"],
                                            proposal_and_checks_seconds=time.monotonic()-t0))
        passed(CASES[2], candidate_calls=2, columns=8, updates_per_call=128)
        faults = []
        faults.append(rejected(lambda: bind(pair=(2, 0)), "canonical native pair"))
        faults.append(rejected(lambda: bind(maps=(0,)), "coefficient row mismatch"))
        changed = copy.deepcopy(result["source"])
        changed["scope"]["request"] = "0"*64
        raw, sha = freeze(changed, work, profile=MARGIN_PROFILE)
        faults.append(rejected(lambda: check.inspect(raw, expected_source=sha, work=work), "base scope binding"))
        changed = copy.deepcopy(result["source"])
        changed["augmented"]["Gb"]["data"][-1] += 1/16
        raw, sha = freeze(changed, work, profile=MARGIN_PROFILE)
        faults.append(rejected(lambda: check.inspect(raw, expected_source=sha, work=work), "coefficient row mismatch"))
        passed(CASES[3], rejections=faults)
        view, anchor = views[0]
        item = result["supports"][0]
        def accept(candidates, v=view, a=anchor):
            return check.check_supports(bound.payload, v, candidates, expected_source=bound.sha256,
                                        expected_property=a, expected_goals=GOALS, work=work)
        faults = [rejected(lambda: accept(item["candidates"][:-1]), "complete candidates")]
        for field, value, reason in (("lower", "123456789", "residual lower"),
                                     ("t", "1/3", "objective binding")):
            bad = copy.deepcopy(item["candidates"]); bad[0][field] = value
            faults.append(rejected(lambda: accept(bad), reason))
        bad = copy.deepcopy(item["candidates"]); bad[0]["y"][0] = "1"
        faults.append(rejected(lambda: accept(bad), "dual sign"))
        bad = copy.deepcopy(item["candidates"]); bad[0]["z"] = bad[0]["z"][:-1]
        faults.append(rejected(lambda: accept(bad), "complete dual rows"))
        bad = decode(view); bad["rows"]["m"]["constant"] = "9"
        raw, sha = freeze(bad, work, profile=MARGIN_PROFILE)
        faults.append(rejected(lambda: accept(item["candidates"], raw, sha), "property reconstruction"))
        passed(CASES[4], rejections=faults)
        live["a"].Gc.indices[1] = live["a"].Gc.indices[0]
        fault = rejected(bind, "native CSR duplicate")
        live["a"] = native(originals["a"])
        old = bound.output_hz.c[0]; bound.output_hz.c[0] += 1
        alias = rejected(lambda: propose_supports(bound, view, expected_property=anchor,
                                                   goals=GOALS, work=work), "live joint source changed")
        bound.output_hz.c[0] = old
        require(decode(bound.payload) == result["source"], "owned payload alias isolation")
        passed(CASES[5], malformed=fault, alias=alias)
        faults = [rejected(lambda: source(originals["entry"], work), "finite factor capacity"),
                  rejected(lambda: Work(deadline).visit("row_visits", 8000001), "work capacity"),
                  rejected(lambda: work.exact(F(1 << 8192)), "intermediate rational capacity"),
                  rejected(lambda: rational(str(1 << 4096)), "rational capacity"),
                  rejected(lambda: DevelopmentWork(time.monotonic()-1, policy=POLICY), "deadline")]
        # Account-only artificial visits are explicit control load, not algorithm cost.
        work.visit("row_visits", 8000001)
        result["synthetic_accounting_fault_visits"] = 8000001
        faults.append(rejected(lambda: work.continue_from(work.counts), "cannot reset"))
        partial = dict(result, status="PASS", cases=[])
        faults.append(rejected(lambda: receive(partial, work), "fixed complete cases"))
        passed(CASES[6], rejections=faults, legacy_rejects_and_development_accumulates=True)
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    result["work_counts"] = dict(work.counts)
    result["elapsed_before_worker_export"] = time.monotonic()-start
    return result


def observation_controls(deadline):
    """Private reader seam only: no monkeypatch or native-library execution."""
    from scoped_proof import owned_session as owned
    class Reader:
        def __init__(self, missing=True, change=False, terminal=False):
            self.missing, self.change, self.terminal = missing, change, terminal
            self.reads = {}
        def pids(self):
            return (10, 11)
        def read(self, pid, name, cap):
            if name == "status":
                return b"Name:\tcontrol\n" if self.missing and pid == 11 else b"VmRSS:\t12 kB\n"
            self.reads[pid] = self.reads.get(pid, 0)+1
            fields = ["0"]*20
            fields[0] = "Z" if self.terminal else "R"
            fields[3] = "10"
            fields[19] = str(100+(self.change and self.reads[pid] > 1))
            return (str(pid)+" (control) "+" ".join(fields)).encode()
    results = []
    current = owned._observe(10, deadline, Reader(), development_policy=POLICY)
    require(current["live"] == [10, 11] and current["rss"] is None
            and current["known_rss_subtotal"] == 12*1024
            and current["rss_unknown_pids"] == [11], "unknown RSS not zero/exit")
    results.append(dict(name="missing_rss_is_live_unknown", status="PASS"))
    for name, kwargs, needle in (
            ("legacy_missing_rejected", {}, "RSS unavailable"),
            ("identity_change_rejected", dict(development_policy=POLICY), "identity changed")):
        try:
            owned._observe(10, deadline, Reader(change=name.startswith("identity")), **kwargs)
        except owned._ObservationFailure as exc:
            require(needle in str(exc), "observation rejection reason")
        else:
            raise AssertionError(name)
        results.append(dict(name=name, status="PASS"))
    normal = owned._observe(10, deadline, Reader(missing=False), development_policy=POLICY)
    legacy = owned._observe(10, deadline, Reader(missing=False))
    require(all(normal[k] == legacy[k] for k in ("live", "zombies", "rss")), "legacy known RSS differential")
    terminal = owned._observe(10, deadline, Reader(terminal=True), development_policy=POLICY)
    require(terminal["live"] == [] and terminal["zombies"] == [10, 11], "terminal identity")
    try:
        owned._observe(10, time.monotonic()-1, Reader(), development_policy=POLICY)
    except owned._ObservationFailure as exc:
        require(exc.timed_out, "same observation deadline")
    else:
        raise AssertionError("observation deadline")
    results.extend(dict(name=n, status="PASS") for n in
                   ("known_rss_legacy_differential", "terminal_identity", "expired_observation"))
    return results
