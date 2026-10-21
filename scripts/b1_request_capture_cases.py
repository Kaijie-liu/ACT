"""One frozen request-integration control, not a real-source experiment."""
import hashlib
import json
import time

from scoped_source.native_capture import CaptureError, CaptureDeadline, canonical, receive, require
from scoped_source.development_work import DevelopmentWork, POLICY

REQUESTS = ("staged_off", "staged_on", "monolithic_f0_off", "monolithic_f0_on")
FAULTS = ("different_deadline", "duplicate_inventory", "incomplete_coverage",
          "partial_and_censored", "duplicate_reuse", "wrong_property", "deferred_missing_native",
          "unselected_release", "first_ordinal", "tampered_guard_prefix", "expired_acceptance",
          "publication_error", "post_capture_alias", "noncanonical_prefix", "nonzero_padding",
          "terminal_skip_reason", "deadline_during_publication")


def run(deadline, publish):
    import copy
    from types import SimpleNamespace as NS
    from pathlib import Path
    import torch
    from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
    from act.pipeline.moe.staged_verifier import verify_staged_linf
    from act.pipeline.moe.request_diagnostics import RequestDiagnostics, source_context, constraint_prefix
    from act.pipeline.moe.request_budget import RequestBudget
    from b1_native_capture_cases import DeadlineFault

    work = DevelopmentWork(deadline, policy=POLICY)
    value = dict(status="FAIL", deadline=deadline, requests=[], controls=[], events=[],
                 trained_source=False, gpu=False, real_request_slots=0)
    captures, observers = {}, {}
    source_counts = 0
    def publication(name):
        counter = [0]
        def accept(kind, raw, digest):
            work.tick()
            require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == digest, "owned publication")
            filename = name+"_%03d_%s.json" % (counter[0], kind); counter[0] += 1
            receipt = publish(filename, json.loads(raw), 2**20)
            require(receipt["sha256"] == digest, "published canonical identity")
            captures.setdefault(name, []).append(dict(kind=kind, filename=filename, sha256=digest))
        return accept
    def model():
        net = build_output_moe(OutputMoEFactoryConfig(input_shape=(2,), num_classes=10, num_experts=3,
            top_k=2, gate=GateKind.SELECTED_SOFTMAX, router_hidden=(), expert_hidden=(), seed=7)).cpu().double().eval()
        with torch.no_grad():
            net.router[1].weight.zero_(); net.router[1].bias.zero_()
            for expert, first in zip(net.experts, (-1., 1., 1.)):
                expert[1].weight.zero_()
                expert[1].bias.copy_(torch.tensor([first, 0.]+[-2.]*8, dtype=torch.float64))
        return net
    net = model(); x = torch.full((1, 2), .5, dtype=torch.float64)
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/"act/pipeline/moe/configs/route_complexity_reuse_v1.json").read_text())
    def control(name, fn, error=None):
        require(name in FAULTS and name not in [r["name"] for r in value["controls"]], "fixed control")
        if error is None:
            fn(); detail = "PASS"
        else:
            try:
                fn()
            except error as exc:
                detail = type(exc).__name__+":"+str(exc)
            else:
                raise AssertionError("missing expected rejection: "+name)
        record = dict(name=name, status="PASS", detail=detail)
        value["controls"].append(record)
        publish("fault_%02d.json" % len(value["controls"]), record, 4096)
    try:
        for name in REQUESTS:
            enter = dict(name=name, phase="ENTER", clock=time.monotonic())
            value["events"].append(enter); publish("event_%02d.json" % len(value["events"]), enter, 4096)
            config = copy.deepcopy(cfg)
            config["comparison_method"] = name.rsplit("_", 1)[0]
            observer = RequestDiagnostics(work=work, byte_limit=16*2**20, publish=publication(name)) if name.endswith("_on") else None
            t = time.monotonic()
            report = verify_staged_linf(net, x, .1, config, budget_started_at=deadline-300,
                **({"diagnostics": observer} if observer is not None else {}))
            result = dict(name=name, phase="RETURN", seconds=time.monotonic()-t,
                status=report.status, reason=report.reason, decision_tier=report.evidence["verdict"]["decision_tier"],
                pairs=report.evidence["route_coverage"]["feasible_route_sets"],
                coverage=report.evidence["route_coverage"]["coverage_complete"],
                full_model_witness_valid=report.evidence["verdict"]["full_model_witness_valid"])
            value["events"].append(dict(result)); publish("event_%02d.json" % len(value["events"]), result, 4096)
            require(report.status == "UNKNOWN" and not result["full_model_witness_valid"] and
                    result["pairs"] == [[0, 1], [0, 2], [1, 2]] and result["coverage"], "fixed tied fixture")
            result["evidence_receipt"] = publish(name+"_evidence.json", report.evidence, 2**20)
            if observer is not None:
                observers[name] = observer
                diag = report.evidence["diagnostics"]
                require(len(diag["obligations"]) == 27 and len(diag["roster"]["properties"]) == 9,
                        "full ten-class three-pair denominator")
                require(observer.selected is not None and observer.selected["pair"] == [0, 1]
                        and observer.selected["property_index"] == 0, "first unresolved fixed ordinal")
                source_counts += 1
                states = [r["state"] for r in diag["obligations"]]
                require(states.count("SELECTED") == 1 and states.count("REUSED") == 25
                        and states.count("RETURNED") == 1, "complete selected/reused/residual partition")
                require(diag["registered_f0_calls"] == (2 if name == "staged_on" else 1), "F0 slots vs pair count")
                require(diag["terminal"]["status"] == report.status and observer.selected["source_available"],
                        "final request and source availability")
                record = observer.selected["record"]
                raw = json.loads(record.payload)
                metadata = raw["source_metadata"]
                require(set(raw["sources"]) >= {"upstream_guarded_entry", "upstream_conditioned_router",
                                               "upstream_expert_joint"}, "same invocation upstream sources")
                require(metadata["margin_relax_binaries"] is False and metadata["difference_relax_binaries"] is True
                        and metadata["a_rows"] == list(range(10)) and metadata["b_rows"] == list(range(10,20)),
                        "ranges and full row semantics")
                result["capture_sha256"] = record.sha256
            value["requests"].append(dict(result))
        for off, on in ((0, 1), (2, 3)):
            keys = ("status", "reason", "decision_tier", "pairs", "coverage", "full_model_witness_valid")
            require(all(value["requests"][off][k] == value["requests"][on][k] for k in keys), "off/on request decisions")

        # Source-free finite state controls; no extra model, solve or replay.
        props = [([1., -1.], 0.), ([1., 0.], .25)]
        rid = "a"*64
        ident = dict(config_sha256="b"*64, model_state={"sha256": "c"*64},
                     center={}, lower={}, upper={}, epsilon=.1, property={})
        budget = RequestBudget(300, started=deadline-300)
        def observer(complete=True, sink=None, w=work):
            o = RequestDiagnostics(work=w, byte_limit=4*2**20, publish=sink)
            o.bind(rid, ident, budget)
            o.routes([(0, 1), (0, 2)], props, complete=complete)
            return o
        def enc(pair=(0,1), q=(1.,-1.)):
            return NS(pair=pair, property_row=q, property_constant=0.)
        contexts = [dict(path="pair")]
        control("different_deadline", lambda: RequestDiagnostics(work=work, byte_limit=2**20).bind(
            rid, ident, RequestBudget(299, started=deadline-300)), CaptureError)
        o = observer()
        control("duplicate_inventory", lambda: o.routes([(0,1)], props, complete=True), CaptureError)
        o = observer(False)
        control("incomplete_coverage", lambda: o.attempt(pairs=[(0,1)], index=0, encodings=[enc()], contexts=contexts), CaptureError)
        def partial():
            o = observer(); o.attempt(pairs=[(0,1)], index=0, encodings=[enc()], contexts=contexts)
            o.censor("fixed interrupted invocation")
            r = o.finish("TIMEOUT", "CONTROL")
            require([v["state"] for v in r["obligations"]] == ["CENSORED", "NOT_REACHED", "NOT_REACHED", "NOT_REACHED"],
                    "preserve attempted and not-reached obligations")
        control("partial_and_censored", partial)
        o = observer(); o.reuse((0,1), 0, "TIER1_FACT")
        control("duplicate_reuse", lambda: o.reuse((0,1), 0, "TIER1_FACT"), CaptureError)
        o = observer()
        control("wrong_property", lambda: o.attempt(pairs=[(0,1)], index=0, encodings=[enc(q=(0.,1.))], contexts=contexts), CaptureError)
        o = observer(); o.attempt(pairs=[(0,1)], index=0, encodings=[enc()], contexts=contexts)
        control("deferred_missing_native", lambda: o.returned(NS(status="UNKNOWN"), replay_valid=False), CaptureError)
        def release():
            o = observer(); h = o.attempt(pairs=[(0,1)], index=0, encodings=[enc()], contexts=contexts)
            h.failure("SCIPY_UNAVAILABLE", native_called=False)
            o.returned(NS(status="SAFE"), replay_valid=False)  # state-control stub, not a solver result
            require(h.pending is h.encodings is h.contexts is None and o.selected is None, "unselected references released")
        control("unselected_release", release)
        def first():
            o = observer()
            h = o.attempt(pairs=[(0,1),(0,2)], index=0, encodings=[enc(),enc((0,2))],
                          contexts=[dict(path="monolithic")]*2)
            h.failure("SOLVER_FAILED:RuntimeError", native_called=True)
            o.returned(NS(status="UNKNOWN", active_pair=(0,2)), replay_valid=False)
            require(o.selected["pair"] == [0,1] and o.selected["other_residual_pairs_unseparated"] == [[0,2]],
                    "do not cherry-pick active branch")
            require(not o.selected["source_available"] and
                    o.rows[(0,1),0]["state"] == "SELECTED_SOURCE_UNAVAILABLE", "unavailable source locks ordinal")
            require(o.attempt(pairs=[(0,1)], index=1,
                    encodings=[NS(pair=(0,1),property_row=(1.,0.),property_constant=.25)],
                    contexts=[dict(path="pair")]) is None, "no replacement source after failed first selection")
            o.censor("CONTROL_END")
        control("first_ordinal", first)
        def badprefix():
            # Small synthetic context with inconsistent literal prefix. No solve.
            import numpy as np
            import scipy.sparse as sp
            from act.back_end.solver.solver_hz import SparseHZono
            def hz(no, rhs):
                return SparseHZono(np.zeros(no), sp.csr_matrix((no,1)), sp.csr_matrix((no,0)),
                    sp.csr_matrix([[1.]]), sp.csr_matrix((1,0)), np.array([rhs]), frame_id=7)
            entry, router, base = hz(1,1.), hz(2,0.), hz(4,1.)
            joint = NS(relation_mode="shared_input", output_hz=base, shared_continuous=1, shared_binary=0,
                a_private_continuous=0,b_private_continuous=0,a_private_binary=0,b_private_binary=0,
                a_rows=(0,1), b_rows=(2,3))
            e = NS(gate_margin_evidence=None, output_hz=NS(frame_id=7,n_cont=3,n_bin=0), property_row=(1.,-1.))
            source_context(encoding=e,entry=entry,router=router,joint=joint,
                           router_input=NS(n_eq=0,n_ineq=0),work=work)
        control("tampered_guard_prefix", badprefix, CaptureError)
        fault = DeadlineFault(work); o = observer(w=fault); fault.expired = True
        control("expired_acceptance", lambda: o.finish("UNKNOWN"), CaptureDeadline)
        def bad_sink(*_):
            raise OSError("fixed sink failure")
        control("publication_error", lambda: observer(sink=bad_sink), CaptureError)
        def alias():
            o = observers["staged_on"]; rec = o.selected["record"]
            raw = rec.payload
            with torch.no_grad():
                net.experts[0][1].bias.fill_(77.)
            require(rec.payload == raw and hashlib.sha256(raw).hexdigest() == rec.sha256, "source mutation cannot alter bytes")
            receive(raw, expected_sha256=rec.sha256, expected_scope=o.selected["receipt"]["scope"], work=work)
        control("post_capture_alias", alias)
        def prefix(padding=False):
            import numpy as np
            import scipy.sparse as sp
            # Same exact row, different CSR order and duplicate representation.
            a = sp.csr_matrix((np.array([2., 1., 3.]), np.array([1,0,1]), np.array([0,3])), shape=(1,2))
            b = sp.csr_matrix([[1.,5.,float(padding)]])
            def view(matrix):
                return NS(Ac=matrix, Ab=sp.csr_matrix((1,0)), b=np.array([.5]),
                          Auc=sp.csr_matrix((0,matrix.shape[1])), Aub=sp.csr_matrix((0,0)), ub=np.array([]))
            saved = (a.data.tobytes(), a.indices.tobytes(), a.indptr.tobytes())
            constraint_prefix(view(a), view(b), dict(eq=0,ineq=0), "control", work)
            require(saved == (a.data.tobytes(), a.indices.tobytes(), a.indptr.tobytes()), "prefix check is read only")
        control("noncanonical_prefix", prefix)
        control("nonzero_padding", lambda: prefix(True), CaptureError)
        def skipped():
            for status, source, reason in (("SAFE","TIER1","REQUEST_ACCEPTED_BY_TIER1"),
                                         ("UNSAFE","TIER1","FULL_MODEL_COUNTEREXAMPLE"),
                                         ("UNKNOWN","TIER1","REQUEST_INCOMPLETE")):
                o = observer(); r = o.finish(status, source)
                require(all(v["state"] == "NOT_REACHED" and v["skip_reason"] == reason
                            for v in r["obligations"]), "closure is not a fabricated per-row solve")
        control("terminal_skip_reason", skipped)
        def publication_deadline():
            fault = DeadlineFault(work); o = observer(w=fault)
            def cutoff(*_):
                fault.expired = True
            o.publish = cutoff
            o.finish("UNKNOWN", "CONTROL")
        control("deadline_during_publication", publication_deadline, CaptureDeadline)
        require(len(value["requests"]) == 4 and len(value["controls"]) == len(FAULTS) and source_counts == 2,
                "complete frozen scope")
        value["status"] = "PASS"
    except BaseException as exc:
        value["error"] = repr(exc)
    value.update(work_counts=dict(work.counts), publications=captures,
                 captured_sources=source_counts, observed_complete_synthetic_requests=len(value["requests"]))
    return value
