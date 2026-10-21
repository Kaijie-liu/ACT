"""Default-off B1 obligation tracking and deferred same-invocation capture.

This is observational, not a verifier. The first returned unresolved F0 call is
selected after the existing decision/replay; no extra solve or replay occurs.
"""
import hashlib
import math
from functools import wraps
from fractions import Fraction

from scoped_source.native_capture import (
    CaptureError, CaptureDeadline, CaptureLedger, canonical, require, receive,
)


SCHEMA = "B1_REQUEST_DIAGNOSTICS_V1"


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def observed(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except CaptureError:
            raise
        except TimeoutError as exc:
            raise CaptureDeadline(str(exc)) from exc
        except Exception as exc:
            raise CaptureError("diagnostic interface: "+type(exc).__name__+": "+str(exc)) from exc
    return call


class DeferredCapture:
    """Internal short-lived holder, never passed to an artifact consumer."""
    def __init__(self, owner, scope, encodings, contexts):
        self.owner, self.scope = owner, scope
        self.encodings, self.contexts = encodings, contexts
        self.pending = None

    @observed
    def validate_encodings(self, encodings):
        self.owner.ledger.tick()
        require(len(encodings) == len(self.encodings) and
                all(a is b for a, b in zip(encodings, self.encodings)), "same invocation encodings")

    @observed
    def store(self, method, kwargs):
        self.owner.ledger.tick()
        require(self.pending is None, "one native return per diagnostic call")
        self.pending = (method, kwargs)

    def hz(self, **kwargs):
        self.store("hz", kwargs)

    def monolithic(self, **kwargs):
        self.store("monolithic", kwargs)

    def failure(self, reason, *, native_called):
        self.store("failure", dict(reason=reason, native_called=native_called))

    def clear(self):
        self.pending = self.encodings = self.contexts = None


def constraint_prefix(left, right, offset, prefix, work):
    """Compare exact sums of stored binary64 rows without changing either CSR.

    Column sorting/duplicates need not match. This is a stored-constraint check,
    not a proof of propagation, guard construction, or native lowering.
    """
    def row(matrix, r):
        lo, hi = matrix.indptr[r:r+2]
        work.visit("coefficient_visits", int(hi-lo))
        result = {}
        for column, value in zip(matrix.indices[lo:hi], matrix.data[lo:hi]):
            value = float(value)
            require(math.isfinite(value), prefix+" finite coefficient")
            column = int(column)
            result[column] = work.exact(result.get(column, Fraction(0))+Fraction(value))
        return {k: v for k, v in result.items() if v}
    for kind in ("eq", "ineq"):
        c, b, rhs = ("Ac", "Ab", "b") if kind == "eq" else ("Auc", "Aub", "ub")
        start = offset[kind]
        require(start >= 0 and start+len(getattr(left, rhs)) <= len(getattr(right, rhs)),
                prefix+" row extent")
        for r, bound in enumerate(getattr(left, rhs)):
            work.visit("row_visits")
            other = float(getattr(right, rhs)[start+r])
            require(math.isfinite(float(bound)) and math.isfinite(other)
                    and Fraction(float(bound)) == Fraction(other), prefix+" RHS")
            for key in (c, b):
                lm, rm = getattr(left, key), getattr(right, key)
                require(lm.shape[1] <= rm.shape[1], prefix+" factor prefix")
                require(row(lm, r) == row(rm, start+r), prefix+" coefficients")


def source_context(*, encoding, entry, router, joint, router_input, work):
    """Actual references plus checked stored-prefix maps; no repropagation."""
    from act.pipeline.moe.experiment1f0 import _support_status
    require(joint.relation_mode == "shared_input" and encoding.gate_margin_evidence is None,
            "B1 diagnostic source is shared-input baseline F0")
    base = joint.output_hz
    require(entry.frame_id == router.frame_id == base.frame_id == encoding.output_hz.frame_id,
            "same invocation source frames")
    require(joint.shared_continuous == entry.n_cont and joint.shared_binary == entry.n_bin
            and base.n_cont == entry.n_cont+joint.a_private_continuous+joint.b_private_continuous
            and base.n_bin == entry.n_bin+joint.a_private_binary+joint.b_private_binary,
            "shared/private factor widths")
    classes = len(encoding.property_row)
    require(joint.a_rows == tuple(range(classes)) and
            joint.b_rows == tuple(range(classes, 2*classes)) and base.n_out == 2*classes,
            "complete expert output rows")
    require(encoding.output_hz.n_cont == base.n_cont+2 and encoding.output_hz.n_bin == base.n_bin,
            "F0 original-factor prefix and appended slots")
    offsets = dict(eq=router_input.n_eq, ineq=router_input.n_ineq)
    require(entry.n_eq == offsets["eq"]+router.n_eq and
            entry.n_ineq == offsets["ineq"]+router.n_ineq, "router-to-entry row extents")
    constraint_prefix(router, entry, offsets, "router_to_entry", work)
    constraint_prefix(entry, base, dict(eq=0, ineq=0), "entry_to_joint", work)
    constraint_prefix(base, encoding.output_hz, dict(eq=0, ineq=0), "joint_to_f0", work)
    bounds = encoding.bounds
    ranges = dict(margin=list(encoding.margin_bounds),
                  gate=[bounds.lambda_lower, bounds.lambda_upper],
                  difference=[bounds.difference_lower, bounds.difference_upper],
                  product=[bounds.product_lower, bounds.product_upper])
    require(all(len(v) == 2 and all(math.isfinite(x) for x in v) and v[0] <= v[1]
                for v in ranges.values()), "finite ordered source ranges")
    metadata = dict(schema="B1_STORED_UPSTREAM_CONTEXT_V1", selected_pair=list(encoding.pair),
        relation="shared_input", a_rows=list(joint.a_rows), b_rows=list(joint.b_rows),
        shared=[joint.shared_continuous, joint.shared_binary],
        private_a=[joint.a_private_continuous, joint.a_private_binary],
        private_b=[joint.b_private_continuous, joint.b_private_binary],
        router_to_entry_row_offsets=offsets, router_prefix=[router.n_cont, router.n_bin],
        lambda_slot=base.n_cont, product_slot=base.n_cont+1,
        lambda_center=(bounds.lambda_lower+bounds.lambda_upper)*.5,
        lambda_radius=(bounds.lambda_upper-bounds.lambda_lower)*.5,
        product_center=(bounds.product_lower+bounds.product_upper)*.5,
        product_radius=(bounds.product_upper-bounds.product_lower)*.5,
        ranges=ranges, margin_support=_support_status(encoding.margin_support),
        difference_support=_support_status(encoding.difference_support),
        margin_relax_binaries=False, difference_relax_binaries=True,
        guarantee="STORED_PREFIX_CHECK_NOT_NETWORK_OR_GUARD_PROOF")
    return {"guarded_entry": entry, "conditioned_router": router, "expert_joint": base}, metadata


class RequestDiagnostics:
    """One request; external consumer receives only bounded immutable bytes.

    publish(kind, payload, sha256) must use the same outer budget and cumulative
    artifact cap. It is optional for tests, mandatory for a frozen real worker.
    It must not request model/solver work. Hard-kill missingness stays censored.
    """
    def __init__(self, *, work, byte_limit, publish=None):
        self.ledger = CaptureLedger(work, byte_limit)
        self.publish = publish
        self.binding = None
        self.roster = None
        self.rows = {}
        self.selected = None
        self.pending = None
        self.queries = 0
        self.terminal = None
        self.inventory_upper = 8192

    @observed
    def bind(self, request_id, request_identity, budget):
        self.ledger.tick()
        require(self.binding is None and budget is not None and budget.deadline == self.ledger.work.deadline,
                "diagnostics need same existing request deadline")
        self.binding = dict(request=request_id, config=request_identity["config_sha256"],
            model=request_identity["model_state"]["sha256"],
            domain=identity({k: request_identity[k] for k in ("center", "lower", "upper", "epsilon", "property")}))

    @observed
    def routes(self, pairs, properties, *, complete):
        self.ledger.tick()
        require(self.binding is not None and self.roster is None, "single bound route inventory")
        require(type(complete) is bool and len(pairs) <= 28 and 1 <= len(properties) <= 255,
                "bounded route/property inventory")
        pairs = [list(p) for p in pairs]
        require(all(len(p) == 2 and all(type(v) is int for v in p) and 0 <= p[0] < p[1] for p in pairs)
                and len({tuple(p) for p in pairs}) == len(pairs), "canonical unique route inventory")
        props = [{"q": [float(v) for v in q], "constant": float(c)} for q, c in properties]
        require(all(1 <= len(p["q"]) <= 256 and all(math.isfinite(v) for v in p["q"]+[p["constant"]])
                    for p in props), "complete finite property rows")
        self.roster = dict(pairs=pairs, properties=props, coverage_complete=complete)
        self.inventory_upper = 8192+len(pairs)*len(props)*1024+sum(256+32*len(p["q"]) for p in props)
        self.roster_hash = identity(self.roster)
        for pair in pairs:
            for index in range(len(props)):
                self.rows[tuple(pair), index] = dict(state="NOT_REACHED", query=None, result=None)
        self.emit()

    @observed
    def reuse(self, pair, index, kind):
        self.ledger.tick()
        row = self.rows[tuple(pair), index]
        require(row["state"] == "NOT_REACHED" and kind in ("TIER1_FACT", "GATE_FACT"), "single reuse")
        row.update(state="REUSED", result=kind)

    @observed
    def attempt(self, *, pairs, index, encodings, contexts):
        self.ledger.tick()
        require(self.roster is not None and self.roster["coverage_complete"] and self.pending is None,
                "complete inventory and no pending solve")
        pairs = [tuple(p) for p in pairs]
        require(pairs and len(encodings) == len(pairs) and len(contexts) == len(pairs)
                and [tuple(e.pair) for e in encodings] == pairs, "residual invocation roster")
        prop = self.roster["properties"][index]
        require(all(list(e.property_row) == prop["q"] and e.property_constant == prop["constant"]
                    for e in encodings), "invocation property identity")
        ordinal = self.queries; self.queries += 1
        for p in pairs:
            row = self.rows[p, index]
            require(row["state"] == "NOT_REACHED", "one attempt per obligation")
            row.update(state="ATTEMPTED", query=ordinal)
        path = contexts[0]["path"]
        require(path in ("pair", "monolithic") and (path == "monolithic" or len(pairs) == 1), "native path")
        scoped = dict(self.binding, roster=self.roster_hash, path=path, ordinal=ordinal,
                      property_index=index, pairs=[list(p) for p in pairs], **prop)
        holder = DeferredCapture(self, scoped, encodings, contexts) if self.selected is None else None
        self.pending = (pairs, index, holder)
        self.emit()
        return holder

    @observed
    def returned(self, decision, *, replay_valid):
        self.ledger.tick()
        require(self.pending is not None and type(replay_valid) is bool, "pending native return/replay")
        pairs, index, holder = self.pending
        final = "UNSAFE_REPLAYED" if replay_valid else decision.status
        require(final in ("SAFE", "UNKNOWN", "UNSAFE_REPLAYED"), "returned decision grade")
        try:
            for pair in pairs:
                self.rows[pair, index].update(state="RETURNED", result=final)
            if holder is not None and final == "UNKNOWN":
                require(holder.pending is not None, "native hook return required")
                from act.back_end.solver.diagnostic_capture import NativeCapture
                method, kwargs = holder.pending
                if method == "failure":
                    extra, metadata = {}, {"source_unavailable": True}
                else:
                    context = holder.contexts[0]
                    extra, metadata = source_context(encoding=holder.encodings[0],
                        **{k: context[k] for k in ("entry", "router", "joint", "router_input")},
                        work=self.ledger.work)
                    metadata["residual_pair_scope"] = "FIRST_ORDINAL_NOT_ACTIVE_BRANCH"
                capture = NativeCapture(scope=holder.scope, ledger=self.ledger,
                                        extra_hz=extra, metadata=metadata)
                getattr(capture, method)(**kwargs)
                rec = capture.record
                receipt = receive(rec.payload, expected_sha256=rec.sha256,
                                  expected_scope=holder.scope, work=self.ledger.work)
                self.selected = dict(query=holder.scope["ordinal"], pair=list(pairs[0]), property_index=index,
                    record=rec, receipt=receipt, source_available=method != "failure",
                    other_residual_pairs_unseparated=[list(p) for p in pairs[1:]])
                self.rows[pairs[0], index]["state"] = (
                    "SELECTED" if method != "failure" else "SELECTED_SOURCE_UNAVAILABLE")
                if self.publish is not None:
                    self.publish("native", rec.payload, rec.sha256)
                    self.ledger.tick()
        finally:
            if holder is not None:
                holder.clear()
            self.pending = None
        self.emit()

    def censor(self, reason):
        # Called while handling an existing failure: record in memory even if
        # the deadline has expired, never publish/reaccept it with a new clock.
        if self.pending is not None:
            pairs, index, holder = self.pending
            for p in pairs:
                self.rows[p, index].update(state="CENSORED", result=str(reason)[:160])
            if holder is not None:
                holder.clear()
            self.pending = None

    @observed
    def finish(self, status, decision_source=None):
        self.ledger.tick()
        require(self.terminal is None, "single request terminal")
        self.censor("REQUEST_TERMINAL")
        for row in self.rows.values():
            if row["state"] == "NOT_REACHED":
                row["skip_reason"] = (
                    "REQUEST_ACCEPTED_BY_"+str(decision_source) if status == "SAFE" else
                    "FULL_MODEL_COUNTEREXAMPLE" if status == "UNSAFE" else "REQUEST_INCOMPLETE")
        if self.roster is None:
            self.terminal = dict(status=status, coverage="ROUTE_COVERAGE_INCOMPLETE", decision_source=decision_source)
        else:
            # A Tier-1 request verdict is not a fabricated native per-row solve.
            self.terminal = dict(status=status, decision_source=decision_source, coverage="COMPLETE" if self.roster["coverage_complete"]
                                 else "ROUTE_COVERAGE_INCOMPLETE")
        return self.emit()

    @observed
    def emit(self):
        self.ledger.tick()
        summary = dict(schema=SCHEMA, binding=self.binding, deadline=self.ledger.work.deadline,
            roster=self.roster, roster_sha256=getattr(self, "roster_hash", None),
            obligations=[dict(pair=list(p), property_index=i, **row) for (p, i), row in self.rows.items()],
            selected=None if self.selected is None else {k: v for k, v in self.selected.items() if k != "record"},
            registered_f0_calls=self.queries, terminal=self.terminal,
            guarantee="STRUCTURAL_DIAGNOSTIC_NOT_PROOF")
        # Fixed bounded inventory, check conservative allocation before encoding.
        upper = self.inventory_upper
        self.ledger.reserve(upper)
        raw = canonical(summary)
        require(len(raw) <= upper, "diagnostic inventory encoding bound")
        self.ledger.visit(len(self.rows)+(len(raw)+4095)//4096)
        self.ledger.accept(len(raw))
        if self.publish is not None:
            self.publish("inventory", raw, hashlib.sha256(raw).hexdigest())
            self.ledger.tick()
        return summary
