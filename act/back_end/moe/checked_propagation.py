"""Opt-in given-HZ checked support in the real HybridzTF propagation path.

Finite CPU controls, cooperative deadlines, trusted legacy lowering. Neither a
whole-MoE verifier nor an outward-rounded network transformer. Frozen original
TFs/configuration are not patched, registered globally or changed on import.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import time

import numpy as np
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.moe.batched_support import prepare_batch, propose_batch
from act.back_end.moe.check_batched_support import check_batch
from act.back_end.moe.hz_routing import guarded_input_topk_set
from act.back_end.solver.hz_lp_export import snapshot
from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_fast_bounds
from scoped_source.rowwise_bound import clock, identity


@dataclass(frozen=True)
class CheckedSupportOptions:
    enabled: bool = True
    rows: int = 4
    candidate_seconds: float = 4.0

    def __post_init__(self):
        if (type(self.enabled) is not bool or type(self.rows) is not int
                or not 1 <= self.rows <= 4
                or type(self.candidate_seconds) not in (int, float)
                or not 0 < self.candidate_seconds <= 4):
            raise ValueError("finite checked-propagation options")


def outward(value, side):
    """Enclose an exact rational by one binary64 endpoint, never nearest-only."""
    q = Fraction(value)
    x = float(q)
    if side not in ("min", "max") or not math.isfinite(x):
        raise ValueError("finite support endpoint/side required")
    if side == "min" and Fraction.from_float(x) > q:
        x = math.nextafter(x, -math.inf)
    elif side == "max" and Fraction.from_float(x) < q:
        x = math.nextafter(x, math.inf)
    if not math.isfinite(x):
        raise ValueError("outward endpoint overflow")
    return x


def _value(value):
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, str):
        return str(value)
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("finite CPU network parameter required")
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.detach().reshape(-1).tolist()}
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("finite network array required")
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.reshape(-1).tolist()}
    if type(value) in (list, tuple):
        return [_value(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _value(v) for k, v in value.items()}
    raise ValueError("unsupported network identity parameter: " + type(value).__name__)


def net_snapshot(net):
    if not 1 <= len(net.layers) <= 32:
        raise ValueError("finite network layer capacity")
    if (len({l.id for l in net.layers}) != len(net.layers)
            or set(net.by_id) != {l.id for l in net.layers}
            or any(net.by_id[l.id] is not l for l in net.layers)):
        raise ValueError("network dispatch map differs from bound layer roster")
    return {"layers": [{"id": l.id, "kind": l.kind, "params": _value(l.params),
                        "in_vars": list(l.in_vars), "out_vars": list(l.out_vars)} for l in net.layers],
            "preds": {str(k): list(v) for k, v in net.preds.items()},
            "succs": {str(k): list(v) for k, v in net.succs.items()}}


def _bounds(bounds):
    if any(t.device.type != "cpu" or t.dtype != torch.float64
           or not bool(torch.isfinite(t).all()) for t in (bounds.lb, bounds.ub)):
        raise ValueError("finite CPU/float64 propagation bounds required")
    if bounds.lb.shape != bounds.ub.shape or bounds.lb.ndim < 2 or bounds.lb.shape[0] != 1:
        raise ValueError("one-lane propagation bounds required")
    if bool((bounds.lb > bounds.ub).any()):
        raise ValueError("contradictory propagation bounds")
    return {"lower": bounds.lb.tolist(), "upper": bounds.ub.tolist()}


class CheckedGuardedHybridzTF(HybridzTF):
    """One invocation owns one source/guard/network; instantiate anew to rebind."""

    def __init__(self, config, *, net, input_hz, router_hz, route, expert,
                 request, entry_bounds, deadline, options=None, retain_guard=True):
        self.started = time.monotonic()
        self.tick = clock(deadline)
        self.deadline = deadline
        self.options = options or CheckedSupportOptions()
        if not self.options.enabled and config.guarded_support_enabled:
            raise ValueError("disabled finite control requires native support disabled")
        if config.sparse_resource_policy == "legacy_affine_cells":
            raise ValueError("explicit sparse memory policy required")
        if type(retain_guard) is not bool:
            raise ValueError("explicit guard ablation flag")
        if (type(request) is not str or not request
                or not all(isinstance(h, SparseHZono) for h in (input_hz, router_hz))
                or input_hz.frame_id is None or input_hz.frame_id != router_hz.frame_id):
            raise ValueError("correlated sparse caller source/request required")
        if (type(expert) is not int or not route or any(type(i) is not int for i in route)
                or expert not in route):
            raise ValueError("expert must belong to declared route")
        if input_hz.n_cont > router_hz.n_cont or input_hz.n_bin > router_hz.n_bin:
            raise ValueError("router lost input factor columns")
        # Necessary provenance check only: a caller must still justify common
        # factor meaning. Equal frame labels are not such a proof.
        for left, right, rhs in (("Ac", "Ab", "b"), ("Auc", "Aub", "ub")):
            count = len(getattr(input_hz, rhs))
            if len(getattr(router_hz, rhs)) < count or not np.array_equal(getattr(input_hz, rhs), getattr(router_hz, rhs)[:count]):
                raise ValueError("router lost input constraint prefix")
            for key in (left, right):
                a, b = getattr(input_hz, key), getattr(router_hz, key)
                if (a != b[:count, :a.shape[1]]).nnz or b[:count, a.shape[1]:].nnz:
                    raise ValueError("router changed input constraint prefix")
        self._source_input, self._source_router = input_hz, router_hz
        self._bound_net, self._source_bounds = net, entry_bounds
        source = {"input": snapshot(input_hz), "router": snapshot(router_hz),
                  "network": net_snapshot(net), "entry_bounds": _bounds(entry_bounds)}
        # Require an outer seed for the supplied input HZ. This is a legacy
        # numerical-policy check, not an independent source inclusion proof.
        hb = sparse_hz_fast_bounds(input_hz)
        if (entry_bounds.lb.numel() != input_hz.n_out
                or bool((entry_bounds.lb.reshape(-1) > hb.lb.reshape(-1)).any())
                or bool((entry_bounds.ub.reshape(-1) < hb.ub.reshape(-1)).any())):
            raise ValueError("entry bounds must enclose supplied input fast bounds")
        guarded = guarded_input_topk_set(input_hz, router_hz, route)
        entry = guarded.hz if retain_guard else input_hz
        self.scope = {"schema": "CHECKED_GUARDED_PROPAGATION_V1", "request": request,
                      "source": source, "route": list(guarded.route_set), "expert": expert,
                      "guard_kind": "TIE_LEGAL_UNORDERED_SET" if retain_guard else "GUARD_DISCARDED_OUTER",
                      "entry": snapshot(entry), "config": asdict(config), "options": asdict(self.options)}
        self.scope_sha256 = identity(self.scope)
        self.source_sha256 = identity(source)
        self.entry_sha256 = identity(self.scope["entry"])
        self.events = []
        super().__init__(config)
        super().set_entry_hz(entry)
        self.tick()
        self.construction_seconds = time.monotonic() - self.started

    def set_entry_hz(self, hz):
        raise ValueError("bound invocation cannot be rebound; create a new instance")

    def _validate_scope(self, net):
        self.tick()
        current = {"input": snapshot(self._source_input), "router": snapshot(self._source_router),
                   "network": net_snapshot(net), "entry_bounds": _bounds(self._source_bounds)}
        if (net is not self._bound_net or identity(current) != self.source_sha256
                or identity(self.scope) != self.scope_sha256
                or self._entry_sparse_hz_override is None
                or identity(snapshot(self._entry_sparse_hz_override)) != self.entry_sha256):
            raise ValueError("bound source/router/network/guard changed")
        self.tick()

    def apply(self, L, input_bounds, net, before, after):
        self._validate_scope(net)
        _bounds(input_bounds)
        if not any(L is layer for layer in net.layers):
            raise ValueError("foreign layer outside bound network")
        result = super().apply(L, input_bounds, net, before, after)
        self._validate_scope(net)
        return result

    def _native_fallback(self, hz, lo, hi, event):
        """Same native support interface, admitted afresh before each call.

        Native bounds retain their old numerical contract, not the exact-check
        provenance above. Only the admission seam is exercised in this stage.
        """
        from act.back_end.hybridz_tf.tf_mlp import _guarded_support_query
        stages = []
        event['native_stages'] = stages
        if not self._guarded_support_enabled or not (hz.n_eq + hz.n_ineq):
            return lo, hi
        for count, limit, relaxed in (
                (self._guarded_support_lp_neurons, self._guarded_support_lp_time_limit, True),
                (self._guarded_support_milp_neurons, self._guarded_support_milp_time_limit, False)):
            self.tick()
            remaining = self.deadline-time.monotonic()
            budget = min(limit, max(0., remaining))
            unstable = np.flatnonzero((lo < 0) & (hi > 0))
            if count <= 0 or budget <= 0 or not len(unstable):
                continue
            selected = unstable[np.argsort(np.minimum(-lo[unstable], hi[unstable]), kind='stable')][:count]
            begin = time.monotonic()
            record = {'relax_binaries': relaxed, 'rows': selected.tolist(), 'allowance': budget,
                      'remaining_before_call': remaining, 'completed': False,
                      'guarantee': 'LEGACY_NATIVE_NUMERICAL_POLICY'}
            stages.append(record)
            try:
                support, telemetry = _guarded_support_query(self, hz, selected, time_limit=budget,
                                                            relax_binaries=relaxed)
                self.tick()  # A late LP must not start a fresh MILP.
                lb = support.bounds.lb.reshape(-1).numpy()
                ub = support.bounds.ub.reshape(-1).numpy()
                if (len(lb) != len(selected) or len(ub) != len(selected)
                        or not np.isfinite(lb).all() or not np.isfinite(ub).all()):
                    raise ValueError('malformed native support')
                lo[selected] = np.maximum(lo[selected], lb)
                hi[selected] = np.minimum(hi[selected], ub)
                if np.any(lo > hi):
                    raise ValueError('native support contradicts existing bounds')
                record.update(completed=True, lower_status=list(support.lower_status),
                              upper_status=list(support.upper_status), telemetry=telemetry)
            finally:
                record['seconds'] = time.monotonic()-begin
        return lo, hi

    def _propagate_sparse_hz(self, L, input_bounds, result):
        self.tick()
        hz = self._sparse_hz_cache.get(L.id)
        if L.kind.upper() != "RELU" or hz is None or not self.options.enabled:
            value = super()._propagate_sparse_hz(L, input_bounds, result)
            self.tick()
            return value
        begin = time.monotonic()
        event = {"layer_id": L.id, "scope_sha256": self.scope_sha256,
                 "status": "STARTED", "selected": [], "package": None}
        self.events.append(event)
        original_lp = self._guarded_support_lp_time_limit
        original_mip = self._guarded_support_milp_time_limit
        original_enabled = self._guarded_support_enabled
        try:
            _bounds(input_bounds)
            hb = sparse_hz_fast_bounds(hz)
            lo = torch.maximum(input_bounds.lb.reshape(-1), hb.lb.reshape(-1)).numpy().copy()
            hi = torch.minimum(input_bounds.ub.reshape(-1), hb.ub.reshape(-1)).numpy().copy()
            if np.any(lo > hi):
                raise ValueError("inconsistent preactivation input")
            unstable = np.flatnonzero((lo < 0) & (hi > 0))
            event["fast_unstable"] = len(unstable)
            selected = unstable[np.argsort(np.minimum(-lo[unstable], hi[unstable]), kind="stable")][:self.options.rows]
            event["selected"] = selected.tolist()
            event["status"] = "NO_UNSTABLE_ROWS" if not len(selected) else "NO_CONSTRAINTS"
            if len(selected) and hz.n_eq + hz.n_ineq:
                if not (1 <= hz.n_cont + hz.n_bin <= 128 and hz.n_out <= 128
                        and hz.n_eq + hz.n_ineq <= 256):
                    event["status"] = "CAPACITY_NATIVE_FALLBACK"
                else:
                    local = min(self.deadline, time.monotonic() + self.options.candidate_seconds)
                    source_hash = identity(snapshot(hz))
                    event["preactivation_sha256"] = source_hash
                    queries = [{"id": f"{L.id}:{int(row)}:{side}", "q": [int(j == row) for j in range(hz.n_out)],
                                "offset": 0, "side": side} for row in selected for side in ("min", "max")]
                    intended = tuple((q["id"], q["side"], tuple(q["q"]), Fraction(q["offset"])) for q in queries)
                    context = {"request": self.scope["request"], "domain": self.source_sha256,
                               "guard": self.entry_sha256, "caller_scope": self.scope_sha256, "layer": L.id}
                    try:
                        batch = prepare_batch(hz, queries, context=context, deadline=local)
                        anchor = identity(batch)
                        prepared = time.monotonic()
                        candidates = propose_batch(batch, expected_batch_sha256=anchor, deadline=local)
                        proposed = time.monotonic()
                        accepted = check_batch(batch, candidates, expected_batch_sha256=anchor, deadline=local)
                        checked = time.monotonic()
                        # Revalidate parent-owned source and ordered queries before use.
                        if (batch["source_sha256"] != source_hash or batch["context"] != context
                                or tuple((q["id"], q["side"], tuple(Fraction(v) for v in q["q"]),
                                          Fraction(q["offset"])) for q in batch["queries"]) != intended):
                            raise ValueError("propagation support origin changed")
                        new_lo, new_hi = lo.copy(), hi.copy()
                        for row, lower, upper in zip(selected, accepted["results"][::2], accepted["results"][1::2]):
                            new_lo[row] = max(lo[row], outward(lower["bound"], "min"))
                            new_hi[row] = min(hi[row], outward(upper["bound"], "max"))
                        if np.any(new_lo > new_hi):
                            raise ValueError("checked intersection contradicts existing bounds")
                        clock(local)()
                        lo, hi = new_lo, new_hi
                        event.update(status="CHECKED_SUPPORT_APPLIED", package={"batch": batch,
                                     "candidates": candidates, "accepted": accepted},
                                     prepare_seconds=prepared-begin, proposal_seconds=proposed-prepared,
                                     check_seconds=checked-proposed)
                    except (ValueError, RuntimeError, OverflowError, TimeoutError) as exc:
                        event.update(status="CANDIDATE_REJECTED_NATIVE_FALLBACK", reason=f"{type(exc).__name__}: {exc}")
                    # Source pollution is never demoted to a native fallback.
                    if identity(snapshot(hz)) != source_hash:
                        raise ValueError("preactivation HZ polluted during support")
                    self._validate_scope(self._bound_net)
            event["after_checked_unstable"] = int(((lo < 0) & (hi > 0)).sum())
            remaining = max(0.0, self.deadline - time.monotonic())
            self.tick()
            # One absolute budget, not a fresh full native allotment per stage.
            self._guarded_support_lp_time_limit = min(original_lp, remaining)
            self._guarded_support_milp_time_limit = min(original_mip, max(0.0, remaining-self._guarded_support_lp_time_limit))
            event["native_allowances"] = [self._guarded_support_lp_time_limit, self._guarded_support_milp_time_limit]
            event["remaining_before_native"] = remaining
            lo, hi = self._native_fallback(hz, lo, hi, event)
            self._validate_scope(self._bound_net)
            # Native fallback has already consumed its per-stage remaining
            # allowance. Inherited ReLU must not launch those calls a second time.
            self._guarded_support_enabled = False
            refined = Bounds(torch.from_numpy(lo).reshape(input_bounds.lb.shape),
                             torch.from_numpy(hi).reshape(input_bounds.ub.shape))
            value = super()._propagate_sparse_hz(L, refined, result)
            self.tick()
            out = self._sparse_hz_cache.get(L.id)
            event["output_n_bin"] = None if out is None else out.n_bin
            event["completed"] = True
            return value
        except BaseException as exc:
            event.update(completed=False, terminal_exception=type(exc).__name__)
            raise
        finally:
            self._guarded_support_lp_time_limit = original_lp
            self._guarded_support_milp_time_limit = original_mip
            self._guarded_support_enabled = original_enabled
            event["total_seconds"] = time.monotonic() - begin


def propagate_checked_guarded(net, *, input_hz, router_hz, route, expert,
                              request, entry_bounds, config, deadline,
                              options=None, retain_guard=True):
    """Real ACT analyzer; one bound expert/route. Does not call a property solver."""
    started = time.monotonic()
    tick = clock(deadline)
    from act.back_end.analyze import analyze
    from act.back_end.moe.route_a import _component_output_hz
    import act.back_end.transfer_functions as state
    from act.back_end.verifier import find_entry_layer_id

    tf = CheckedGuardedHybridzTF(config, net=net, input_hz=input_hz, router_hz=router_hz,
                               route=route, expert=expert, request=request, entry_bounds=entry_bounds,
                               deadline=deadline, options=options, retain_guard=retain_guard)
    try:
        previous = state.get_transfer_function()
    except RuntimeError:
        previous = None
    previous_solver = state.get_solver_mode()
    state.set_transfer_function(tf)
    state.set_solver_mode("hybridz")
    try:
        analyze(net, find_entry_layer_id(net), Fact(entry_bounds, ConSet()))
        output = _component_output_hz(net, tf)
        tf._validate_scope(net)
        outcome = {"scope": tf.scope, "scope_sha256": tf.scope_sha256, "events": tf.events,
                   "output": snapshot(output), "construction_seconds": tf.construction_seconds,
                   "status": "PROPAGATED_TRUSTED_LEGACY_LOWERING",
                   "network_or_complete_moe_proof": False, "hard_budget_supervision": False}
    finally:
        tf.clear_entry_hz()
        state.set_transfer_function(previous)
        state.set_solver_mode(previous_solver)
    outcome["total_seconds"] = time.monotonic() - started
    tick()
    return outcome
