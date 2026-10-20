"""Once-only saved-evidence diagnostic, no optimizer or new source/query.

Transfer already checked tilted duals to closed margin slices, blend at local
gate endpoints, and independently recompute the original-coefficient residual.
This is an offline analysis, never online request acceptance.
"""
import time
ENTRY = time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
from fractions import Fraction as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from scoped_source.hz_request_store import compact, require, rational
from scoped_source.development_work import DevelopmentWork, POLICY
from scoped_source import check_margin_hz_source as source
from scoped_proof.io import save
import margin_scalar_cases as cases

RUN = Path("/data1/Kane/MOE/results/margin_slice_transfer_20261021_r1")
FREEZE = ROOT/"docs/margin_slice_transfer_freeze_20261021.json"
CAP = 128*2**20


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def text(q):
    return str(rational(str(q)))


def check_lower(parsed, gate, ys, zs, slab_duals, cell, work):
    """Direct lower on original P plus BOTH exact closed-slab rows.

    Reconstruct objectives from original expert outputs, not the producer's
    conditional value lines. No source row is removed or merged.
    """
    hz, q, c = parsed["joint"], parsed["q"], parsed["constant"]
    weights = tuple(work.exact(gate*v) for v in q)+tuple(work.exact((1-gate)*v) for v in q)+(F(0), F(0))
    lower, residual = source.project(hz, weights, c, work)
    for duals, ck, bk, rhs, signed in ((ys, "Auc", "Aub", "ub", True),
                                      (zs, "Ac", "Ab", "b", False)):
        require(len(duals) == len(hz[rhs]), "complete original dual rows")
        for i, dual in enumerate(duals):
            work.visit("row_visits")
            require(not signed or dual <= 0, "original inequality sign")
            lower = work.exact(lower+work.exact(dual*hz[rhs][i]))
            for key, shift in ((ck, 0), (bk, hz["nc"])):
                for j, value in hz[key][i].items():
                    work.visit("coefficient_visits")
                    index = shift+j
                    residual[index] = work.exact(residual.get(index, F(0))-work.exact(dual*value))
    # Explicit affine constants: m=m0+Mz, Mz<=y-m0, -Mz<=m0-x.
    m0, margin = parsed["rows"]["m"]
    x, y = cell
    require(len(slab_duals) == 2 and all(v <= 0 for v in slab_duals), "both slice dual signs")
    for sign, rhs, dual in ((1, work.exact(y-m0), slab_duals[0]),
                             (-1, work.exact(m0-x), slab_duals[1])):
        work.visit("row_visits")
        lower = work.exact(lower+work.exact(rhs*dual))
        for j, value in margin.items():
            work.visit("coefficient_visits")
            residual[j] = work.exact(residual.get(j, F(0))-work.exact(sign*dual*value))
    norm = F(0)
    for v in residual.values():
        work.visit("coefficient_visits")
        norm = work.exact(norm+abs(v))
    return work.exact(lower-norm), norm


def transfer(parsed, record, work):
    anchors = tuple(source.parameter(v, work) for v in record["plan"]["anchors"])
    require(len(anchors) == 2 and anchors[0] < anchors[1], "frozen two-anchor saved diagnostic")
    l, h = anchors
    candidates = record["supports"]
    rows, gains, losses = [], 0, 0
    for index, old in enumerate(record["cells"]):
        work.visit("row_visits")
        x, y = source.parameter(old["x"], work), source.parameter(old["y"], work)
        selected = []
        for anchor in anchors:
            best = None
            for position, candidate in enumerate(candidates):
                if source.parameter(candidate["t"], work) != anchor:
                    continue
                nu = source.parameter(candidate["nu"], work)
                lower = work.exact(source.parameter(candidate["lower"], work)
                                   +min(work.exact(-nu*x), work.exact(-nu*y)))
                # Strict > makes the first saved support the deterministic tie winner.
                if best is None or lower > best["lower"]:
                    ys = tuple(source.parameter(v, work) for v in candidate["y"])
                    zs = tuple(source.parameter(v, work) for v in candidate["z"])
                    slab = (-nu, F(0)) if nu >= 0 else (F(0), nu)
                    best = dict(position=position, nu=nu, lower=lower, ys=ys, zs=zs, slab=slab)
            require(best is not None, "all anchor candidates")
            checked, norm = check_lower(parsed, anchor, best["ys"], best["zs"], best["slab"], (x, y), work)
            require(checked == best["lower"], "exact anchor-to-slab transfer including offset")
            best["norm"] = norm
            selected.append(best)
        a, b = selected
        endpoints = []
        for saved_gate in old["gate"]:
            g = source.parameter(saved_gate, work)
            weight_a = work.exact(work.exact(h-g)/work.exact(h-l))
            weight_b = work.exact(1-weight_a)
            require(0 <= weight_a <= 1 and 0 <= weight_b <= 1, "nonnegative interpolation")
            def blend(key):
                require(len(a[key]) == len(b[key]), "same original row frame")
                work.visit("coefficient_visits", 2*len(a[key]))
                return tuple(work.exact(work.exact(weight_a*u)+work.exact(weight_b*v))
                             for u, v in zip(a[key], b[key]))
            ys, zs, slab = blend("ys"), blend("zs"), blend("slab")
            lower, norm = check_lower(parsed, g, ys, zs, slab, (x, y), work)
            interpolated = work.exact(work.exact(weight_a*a["lower"])+work.exact(weight_b*b["lower"]))
            cancellation = work.exact(work.exact(weight_a*a["norm"])+work.exact(weight_b*b["norm"])-norm)
            require(cancellation >= 0 and lower == work.exact(interpolated+cancellation),
                    "residual triangle identity")
            endpoints.append(dict(gate=text(g), weights=[text(weight_a), text(weight_b)],
                                  y=list(map(text, ys)), z=list(map(text, zs)),
                                  slice_duals=list(map(text, slab)), lower=text(lower),
                                  scalar_interpolation=text(interpolated), residual_cancellation=text(cancellation)))
        old_lower = source.parameter(old["lower"], work)
        lower = min(source.parameter(v["lower"], work) for v in endpoints)
        require(lower >= old_lower, "slice transfer cannot weaken the saved scalar cell")
        gains += lower > old_lower
        losses += lower < old_lower
        rows.append(dict(index=index, x=text(x), y=text(y), support_indices=[a["position"], b["position"]],
                         selected_tilts=[text(a["nu"]), text(b["nu"])], endpoints=endpoints,
                         original_lower=text(old_lower), transferred_lower=text(lower)))
    aggregate = min(source.parameter(item["transferred_lower"], work) for item in rows)
    original = source.parameter(record["lower"], work)
    require(aggregate >= original, "complete saved grid transfer dominance")
    work.tick()
    return dict(cells=rows, original_lower=text(original), transferred_lower=text(aggregate),
                increase=text(work.exact(aggregate-original)), cell_gains=gains, cell_losses=losses,
                new_candidate_calls=0, optimality_gap="UNKNOWN", complete_request=False,
                grade="CHECKED_TRANSFER_ON_GIVEN_HZ_CONTINUOUS_RELAXATION")


def main(anchor):
    deadline = ENTRY+300
    for name, cap in ((resource.RLIMIT_AS, 8*2**30), (resource.RLIMIT_FSIZE, CAP)):
        resource.setrlimit(name, (cap, cap))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    frozen_raw = FREEZE.read_bytes()
    require(sha(frozen_raw) == anchor, "frozen saved-only diagnostic")
    frozen = json.loads(frozen_raw)
    require(frozen["attempts"] == 1 and frozen["result_directory"] == str(RUN)
            and frozen["records"] == list(cases.MODES) and frozen["new_native_calls"] == 0,
            "fixed saved-only roster")
    require(frozen["archive"] == "docs/margin_scalar_records_20261021.json"
            and frozen["archive"] in frozen["files"], "fixed original archive identity")
    for name, identity in frozen["files"].items():
        require(sha((ROOT/name).read_bytes()) == identity, "frozen saved-only file: "+name)
    RUN.mkdir()
    save(RUN/"attempt.json", dict(freeze_sha256=anchor, entry=ENTRY, deadline=deadline,
                                  status="STARTED_NOT_ACCEPTED", new_native_calls=0))
    work = DevelopmentWork(deadline, policy=POLICY)
    result = dict(status="FAIL", schema="SAVED_MARGIN_SLICE_TRANSFER_V1", freeze_sha256=anchor,
                  offline_only=True, online_status_changed=False, new_native_calls=0, results={},
                  complete_request=False, entry=ENTRY, deadline=deadline)
    try:
        archive = json.loads((ROOT/frozen["archive"]).read_bytes())
        worker = archive["worker"]
        receipts = cases.receive(worker, work)
        parsed = source.inspect_property(compact(worker["source"]), compact(worker["view"]),
                                          expected_source=worker["source_sha256"],
                                          expected_property=worker["property_sha256"], work=work)
        result.update(source_sha256=worker["source_sha256"], property_sha256=worker["property_sha256"],
                      original_receipts=receipts, input_recheck_seconds=time.monotonic()-ENTRY)
        for name in cases.MODES:
            started = time.monotonic()
            value = transfer(parsed, worker["records"][name]["record"], work)
            value["seconds"] = time.monotonic()-started
            result["results"][name] = value
        require(not any(n in sys.modules for n in ("act", "numpy", "scipy", "torch")), "stdlib only analysis")
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    result.update(work_counts=dict(work.counts), elapsed_before_export=time.monotonic()-ENTRY)
    require(2*len(compact(result))+2**20 < CAP, "diagnostic and archive reserve")
    receipt = save(RUN/"result.json", result)
    work.tick()
    print(json.dumps(dict(status=result["status"], receipt=receipt,
                          through_export_seconds=time.monotonic()-ENTRY)), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    require(len(sys.argv) == 3 and sys.argv[1] == "--frozen-saved-analysis", "explicit saved-only invocation")
    raise SystemExit(main(sys.argv[2]))
