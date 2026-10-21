"""Fixed synthetic native-capture controls; no checkpoint or strategy search."""
import hashlib
import json
import math
import time

from scoped_source.native_capture import (
    CaptureError, CaptureDeadline, CaptureCapacity, CaptureLedger,
    canonical, receive, require,
)
from scoped_source.development_work import DevelopmentWork, POLICY

NATIVE_SLOTS = ("pair_off", "pair_on", "mono_off", "mono_on", "empty_off",
                "empty_on", "pair_capture_deadline", "mono_capture_deadline")
FAULTS = (
    "missing_candidate", "wrong_dtype", "source_dimension", "missing_matrix",
    "wrong_offset", "wrong_objective", "missing_branch", "wrong_branch_pair",
    "wrong_scope", "wrong_constant", "reset_deadline", "expired_receive",
    "missing_problem", "wrong_selector_flag", "capture_capacity",
    "second_record_capacity", "metadata_capacity", "malformed_binding",
    "single_capture", "pair_lowering_failure", "mono_lowering_failure",
    "solver_failure_record", "missing_backend_record", "missing_candidate_record",
    "nonfinite_candidate", "owned_aliases", "constant_no_solver",
    "wrong_call_flag", "unicode_metadata_capacity",
)


def scope(path, ordinal=0):
    hashes = {k: hashlib.sha256(("B1_SYNTHETIC_ONLY:"+k).encode()).hexdigest()
              for k in ("request", "config", "model", "domain", "roster")}
    return dict(hashes, path=path, ordinal=ordinal, property_index=1,
                pairs=[[0, 1]] if path == "pair" else [[0, 1], [0, 2]],
                q=[1.0, -1.0]+[0.0]*8, constant=.125)


class DeadlineFault:
    """Local deterministic fault, never replaces the real outer deadline."""
    def __init__(self, work):
        self.base, self.expired = work, False
        self.deadline = work.deadline
    def tick(self):
        self.base.tick()
        if self.expired:
            raise TimeoutError("fixed capture deadline injection")
    def visit(self, kind, amount=1):
        self.tick(); self.base.visit(kind, amount)


def signature(result):
    from dataclasses import fields
    value = {f.name: getattr(result, f.name) for f in fields(result)
             if f.name not in ("elapsed", "candidate_input")}
    candidate = result.candidate_input
    value["candidate_input"] = None if candidate is None else candidate.detach().cpu().tolist()
    return value


def run(deadline, publish):
    # Imports and fixture construction are inside the worker's charged budget.
    import numpy as np
    import scipy.sparse as sp
    from act.back_end.solver.solver_hz import SparseHZono, hz_minimize_output, _lower_hz_milp
    from act.back_end.solver.diagnostic_capture import NativeCapture
    from act.back_end.moe.weighted_top2 import WeightedTop2F0Encoding, solve_weighted_top2_f0
    from act.back_end.moe.monolithic_f0 import solve_monolithic_weighted_top2_f0

    started = time.monotonic()
    work = DevelopmentWork(deadline, policy=POLICY)
    ledger = CaptureLedger(work, 4*2**20)
    result = dict(status="FAIL", deadline=deadline, events=[], comparisons={}, controls=[],
                  records=[], trained_source=False, gpu=False, complete_request=False)
    def add_record(name, capture):
        require(capture.record is not None, "expected complete capture")
        rec = capture.record
        receipt = publish(name+".json", json.loads(rec.payload), 2**20)
        require(receipt["sha256"] == rec.sha256, "saved canonical bytes")
        expected = json.loads(capture.scope_bytes)
        checked = receive(rec.payload, expected_sha256=rec.sha256, expected_scope=expected, work=work)
        result["records"].append(dict(name=name+".json", sha256=rec.sha256,
                                      scope=expected, receipt=checked))
    def control(name, fn, error=None):
        require(name in FAULTS and name not in [r["name"] for r in result["controls"]], "fixed control name")
        begin = time.monotonic()
        if error is None:
            fn(); detail = "accepted as structural diagnostic only"
        else:
            try:
                fn()
            except error as exc:
                detail = type(exc).__name__+":"+str(exc)
            else:
                raise AssertionError("missing expected rejection: "+name)
        result["controls"].append(dict(name=name, status="PASS", detail=detail, seconds=time.monotonic()-begin))
        publish("control_%02d.json" % len(result["controls"]), result["controls"][-1], 4096)
    def fixture(center=2.0, rhs=0.0, pair=(0, 1)):
        # Supplied scalar F0-shaped HZ, not a claim about a real network lowering.
        hz = SparseHZono(np.array([center]), sp.csr_matrix([[.25]]), sp.csr_matrix([[.5]]),
                         sp.csr_matrix([[1.0]]), sp.csr_matrix([[1.0]]), np.array([rhs]), frame_id=21)
        inp = SparseHZono(np.array([0.0]), sp.csr_matrix([[1.0]]), sp.csr_matrix([[0.0]]),
                          sp.csr_matrix([[1.0]]), sp.csr_matrix([[1.0]]), np.array([rhs]), frame_id=21)
        return WeightedTop2F0Encoding(hz, inp, pair, tuple(scope("pair")["q"]), .125,
                                     (-1.0, 1.0), None, None, None, np.zeros((0, 0)), np.zeros(0))
    positive = fixture(); negative = fixture(-.25, pair=(0, 2)); empty = fixture(rhs=3.0)
    captures = {}
    def native(name, encodings, capture=None, fail=None):
        require(len(result["events"]) % 2 == 0 and
                name == NATIVE_SLOTS[len(result["events"])//2], "native slot order/denominator")
        enter = dict(slot=name, phase="ENTER", clock=time.monotonic())
        result["events"].append(enter)
        publish("event_%02d.json" % len(result["events"]), enter, 4096)
        begin = time.monotonic()
        def invoke():
            kwargs = dict(input_shape=(1,), time_limit=min(5.0, max(.001, deadline-time.monotonic())))
            if capture is not None:
                kwargs["diagnostic_capture"] = capture
            if name.startswith("mono"):
                return solve_monolithic_weighted_top2_f0(encodings, **kwargs)
            return solve_weighted_top2_f0(encodings[0], **kwargs)
        if fail is None:
            decision = invoke(); payload = signature(decision)
        else:
            try:
                invoke()
            except fail as exc:
                payload = dict(expected_exception=type(exc).__name__, message=str(exc))
            else:
                raise AssertionError("capture deadline swallowed")
        work.tick()
        leave = dict(slot=name, phase="RETURN", seconds=time.monotonic()-begin, result=payload,
                     capture_seconds_included=0.0 if capture is None else capture.capture_seconds)
        result["events"].append(leave)
        publish("event_%02d.json" % len(result["events"]), leave, 4096)
        return payload
    try:
        for group, enc in (("pair", [positive]), ("mono", [positive, negative]), ("empty", [empty])):
            path = "monolithic" if group == "mono" else "pair"
            cap = NativeCapture(scope=scope(path), ledger=ledger,
                                extra_hz={"original_scalar": enc[0].output_hz},
                                metadata={"fixture": "supplied scalar, no network proof"})
            off = native(group+"_off", enc)
            on = native(group+"_on", enc, cap)
            require(off == on, "default-off/on decision difference: "+group)
            result["comparisons"][group] = dict(equal=True, decision=on)
            add_record(group, cap); captures[group] = cap
        class FaultCapture(NativeCapture):
            def hz(self, **kw):
                self.ledger.work.expired = True
                return super().hz(**kw)
            def monolithic(self, **kw):
                self.ledger.work.expired = True
                return super().monolithic(**kw)
        for path, name, enc in (("pair", "pair_capture_deadline", [positive]),
                                ("monolithic", "mono_capture_deadline", [positive, negative])):
            fault_work = DeadlineFault(work)
            cap = FaultCapture(scope=scope(path), ledger=CaptureLedger(fault_work, 4*2**20))
            native(name, enc, cap, CaptureDeadline)
            require(cap.record is None and cap.event == "CAPTURE_FAILED", "no late capture receipt")
        def mutate(cap, change):
            value = json.loads(cap.record.payload); change(value)
            raw = canonical(value)
            return receive(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                           expected_scope=json.loads(cap.scope_bytes), work=work)
        pair, mono = captures["pair"], captures["mono"]
        mutations = {
            "missing_candidate": (pair, lambda v: v["arrays"].pop("candidate")),
            "wrong_dtype": (pair, lambda v: v["arrays"]["native.integrality"].update(dtype="<f8")),
            "source_dimension": (pair, lambda v: v["sources"]["f0"].update(n_out=2)),
            "missing_matrix": (pair, lambda v: v["matrices"].pop("f0.Gc")),
            "wrong_offset": (pair, lambda v: v["native"].update(objective_offset=7.0.hex())),
            "wrong_objective": (pair, lambda v: v["arrays"].update({"native.objective": v["arrays"]["native.var_ub"]})),
            "missing_branch": (mono, lambda v: v["native"]["branches"].pop()),
            "wrong_branch_pair": (mono, lambda v: v["native"]["branches"][0].update(pair=[1, 2])),
            "wrong_scope": (pair, lambda v: v["scope"].update(request="0"*64)),
            "wrong_constant": (pair, lambda v: v["scope"].update(constant=0.0)),
            "reset_deadline": (pair, lambda v: v.update(deadline=deadline+1)),
            "missing_problem": (pair, lambda v: v.update(native=None)),
            "wrong_call_flag": (pair, lambda v: v.update(native_called=False)),
            "wrong_selector_flag": (mono, lambda v: v["arrays"].update({"native.integrality": v["arrays"]["native.var_lb"]})),
        }
        for name, (cap, fn) in mutations.items():
            control(name, lambda cap=cap, fn=fn: mutate(cap, fn), CaptureError)
        fault_work = DeadlineFault(work); fault_work.expired = True
        control("expired_receive", lambda: receive(pair.record.payload, expected_sha256=pair.record.sha256,
                expected_scope=scope("pair"), work=fault_work), CaptureDeadline)
        def native_stub(cap, x=None):
            model = _lower_hz_milp(positive.output_hz)
            cap.hz(hz=positive.output_hz, input_hz=positive.input_hz, model=model, row=0,
                   objective=model.value_matrix.getrow(0).toarray().reshape(-1),
                   result=dict(status=1, x=x, fun=None, mip_dual_bound=-1.0))
        control("capture_capacity", lambda: native_stub(NativeCapture(scope=scope("pair"),
                ledger=CaptureLedger(work, 100))), CaptureCapacity)
        def second_capacity():
            # First small failure fits the conservative preflight; a pending second
            # preflight exceeds this same ledger. No per-record budget reset.
            small = CaptureLedger(work, 33000+len(canonical(scope("pair"))))
            a = NativeCapture(scope=scope("pair"), ledger=small)
            a.failure("SCIPY_UNAVAILABLE", native_called=False)
            result["control_ledger_capture_bytes"] = small.used
            b = NativeCapture(scope=scope("pair"), ledger=small)
            b.failure("SCIPY_UNAVAILABLE", native_called=False)
        control("second_record_capacity", second_capacity, CaptureCapacity)
        control("metadata_capacity", lambda: NativeCapture(scope=scope("pair"), ledger=ledger,
                metadata={"oversized": "x"*20000}), CaptureError)
        control("unicode_metadata_capacity", lambda: NativeCapture(scope=scope("pair"), ledger=ledger,
                metadata={"oversized": "\U0001f642"*1500}), CaptureError)
        control("malformed_binding", lambda: NativeCapture(scope=scope("pair"), ledger=ledger)
                .validate_encodings([object()]), CaptureError)
        control("single_capture", lambda: pair.failure("SCIPY_UNAVAILABLE", native_called=False), CaptureError)
        def lowering_pair():
            cap = NativeCapture(scope=scope("pair"), ledger=ledger)
            r = hz_minimize_output(object(), 0, input_hz=None, input_shape=None,
                                   time_limit=1, diagnostic_capture=cap)
            require(r.status == "unknown", "pair lowering keeps UNKNOWN")
            add_record("lowering_pair", cap)
        control("pair_lowering_failure", lowering_pair)
        def lowering_mono():
            from dataclasses import replace
            cap = NativeCapture(scope=scope("monolithic"), ledger=ledger)
            try:
                solve_monolithic_weighted_top2_f0([replace(positive, output_hz=object()), negative],
                    input_shape=(1,), time_limit=1, diagnostic_capture=cap)
            except TypeError:
                require(cap.record is not None, "preserved lowering failure record")
                add_record("lowering_mono", cap)
            else:
                raise AssertionError("original monolithic lowering error lost")
        control("mono_lowering_failure", lowering_mono)
        for name, event, called in (("solver_failure_record", "SOLVER_FAILED:RuntimeError", True),
                                    ("missing_backend_record", "SCIPY_UNAVAILABLE", False)):
            def failure_record(name=name, event=event, called=called):
                cap = NativeCapture(scope=scope("pair"), ledger=ledger)
                cap.failure(event, native_called=called); add_record(name, cap)
            control(name, failure_record)
        def missing():
            cap = NativeCapture(scope=scope("pair"), ledger=ledger)
            native_stub(cap); add_record("missing_candidate", cap)
        control("missing_candidate_record", missing)
        raw = np.array([-0.0, np.nan])
        special = NativeCapture(scope=scope("pair"), ledger=ledger)
        def nonfinite():
            native_stub(special, raw); add_record("nonfinite_candidate", special)
            require(json.loads(special.record.payload)["candidate"] == "RAW_UNCHECKED", "raw is not feasible")
        control("nonfinite_candidate", nonfinite)
        def aliases():
            original = special.record.payload
            raw[:] = [np.inf, 77.0]; positive.output_hz.c[:] = 99.0
            require(special.record.payload == original and
                    hashlib.sha256(original).hexdigest() == special.record.sha256, "immutable owned buffers")
            receive(original, expected_sha256=special.record.sha256, expected_scope=scope("pair"), work=work)
        control("owned_aliases", aliases)
        def constant():
            hz = SparseHZono(np.array([.5]), sp.csr_matrix((1, 0)), sp.csr_matrix((1, 0)),
                sp.csr_matrix((0, 0)), sp.csr_matrix((0, 0)), np.zeros(0), frame_id=22)
            cap = NativeCapture(scope=scope("pair"), ledger=ledger)
            off = hz_minimize_output(hz, 0, input_hz=None, input_shape=None, time_limit=1)
            on = hz_minimize_output(hz, 0, input_hz=None, input_shape=None, time_limit=1, diagnostic_capture=cap)
            require(off.status == on.status == "optimal" and off.minimum == on.minimum == .5,
                    "constant decision unchanged")
            require(not json.loads(cap.record.payload)["native_called"], "constant no native call")
            add_record("constant", cap)
        control("constant_no_solver", constant)
        require(len(result["events"]) == 2*len(NATIVE_SLOTS) and
                {c["name"] for c in result["controls"]} == set(FAULTS), "complete frozen denominator")
        result["status"] = "PASS"
    except BaseException as exc:
        result["error"] = repr(exc)
    result.update(work_counts=dict(work.counts), main_ledger_capture_bytes=ledger.used,
                  elapsed_before_export=time.monotonic()-started)
    return result


def replay(root, value, work):
    """Same-deadline stdlib reception of the complete saved record roster."""
    require(value["status"] == "PASS" and value["deadline"] == work.deadline, "worker completion/deadline")
    require([e["slot"] for e in value["events"][::2]] == list(NATIVE_SLOTS)
            and len(value["events"]) == 2*len(NATIVE_SLOTS), "fixed native slots")
    for i, name in enumerate(NATIVE_SLOTS):
        a, b = value["events"][2*i:2*i+2]
        require(a["slot"] == b["slot"] == name and a["phase"] == "ENTER" and b["phase"] == "RETURN"
                and math.isfinite(b["seconds"]) and b["seconds"] >= 0, "paired native events")
    require(len(value["controls"]) == len(FAULTS) and
            {c["name"] for c in value["controls"]} == set(FAULTS)
            and all(c["status"] == "PASS" for c in value["controls"]), "control denominator")
    expected_names = {"pair.json", "mono.json", "empty.json", "lowering_pair.json", "lowering_mono.json",
                      "solver_failure_record.json", "missing_backend_record.json", "missing_candidate.json",
                      "nonfinite_candidate.json", "constant.json"}
    require(len(value["records"]) == len(expected_names) and
            {r["name"] for r in value["records"]} == expected_names, "complete capture roster")
    receipts = []
    for r in value["records"]:
        expected = scope("monolithic" if r["name"] in ("mono.json", "lowering_mono.json") else "pair")
        require(r["scope"] == expected, "independent fixed fixture scope")
        p = root/r["name"]
        require(not p.is_symlink() and p.stat().st_size <= 2**20, "bounded owned record")
        raw = p.read_bytes()
        rec = receive(raw, expected_sha256=r["sha256"], expected_scope=expected, work=work)
        require(rec == r["receipt"], "same structural receipt")
        receipts.append(rec)
    return receipts
