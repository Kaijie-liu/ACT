"""Default-off M-B complete-request callbacks, sharing the existing budget.

No new supervisor or numerical engine. Prior guards condition prefix margins;
all selected guards condition output properties. This module does not select
models or admit mechanisms, and structural reception is not an exact proof.
"""
import time
from scoped_proof.io import tick
import gate_decoupling_worker as W
from moe_aware_diagnostics import aggregate_mb, raw_spec


def source(request, prefix, purpose, backend):
    original = {k: request[k] for k in ('model', 'sample', 'selection_root')}
    return W.digest(dict(source=original, prefix=[list(p) for p in prefix],
        purpose=purpose, backend=backend, method='M_B_PRIOR_GUARD_CONDITIONAL_CROWN_V1'))


class ConditionalUpper(dict):
    """Upper bounds valid only on the specified prior-history guard domain."""
    def __init__(self, ordered, bounds):
        super().__init__((pair, -lo) for pair, lo in zip(ordered, bounds['lower']))
        self.guard_support = bounds


class GuardDual:
    def __init__(self, model, tensors, request, backend_identity, *, history_fallback=None,
                 support=None, router_fallback=None, coupled=None):
        from act.back_end.moe.moe_aware_bab import conditional_bounds
        from act.back_end.moe.multilayer_external import router_upper_bounds
        from b2_observed_worker import verify_graph
        self.model, self.tensors, self.request, self.backend = model, tensors, request, backend_identity
        self.support = support or conditional_bounds
        self.router_fallback = router_fallback or router_upper_bounds
        self.coupled = coupled or verify_graph
        self.history_fallback = history_fallback
        self.experts = tuple(len(layer.experts) for layer in model.layers)

    def prefix(self, model, prefix, center, lower, upper, deadline):
        from act.back_end.moe.moe_aware_bab import prefix_objectives
        tick(deadline)
        W.require(model is self.model, 'same guarded model')
        if not prefix:
            # No past guard exists; retain the original cheap root query.
            return self.router_fallback(model, prefix, center, lower, upper, deadline)
        sid = source(self.request, prefix, 'NEGATIVE_NEXT_ROUTER_MARGINS', self.backend)
        graph, ordered = prefix_objectives(model, prefix, source_id=sid)
        bounds = self.support(graph, center, lower, upper, deadline=deadline, use_guards=True, iterations=20)
        aggregate_mb(bounds, sid, [list(p) for p in prefix], 'MB_ON', len(ordered),
            sum(2*(e-2) for e in self.experts[:len(prefix)]), deadline)
        # Negation reverses LOWER(-margin) to an UPPER(margin). The next
        # undecided layer supplies objectives, never assumptions/guards.
        return ConditionalUpper(ordered, bounds)

    def history(self, history, prefix_records, deadline):
        from act.back_end.moe.moe_aware_bab import history_objectives
        from act.back_end.moe.multilayer_external import GuardedHistory
        tick(deadline); began = time.monotonic()
        data = self.tensors; n = len(data['rows'])
        sid = source(self.request, history, 'OUTPUT_PROPERTIES', self.backend)
        graph = history_objectives(self.model, history, data['rows'], data['thresholds'], source_id=sid)
        bounds = self.support(graph, data['center'], data['lower'], data['upper'],
            deadline=deadline, use_guards=True, iterations=20)
        aggregate = aggregate_mb(bounds, sid, [list(p) for p in history], 'MB_ON', n,
                                  sum(2*(e-2) for e in self.experts), deadline)
        del graph
        if aggregate['closed']:
            outcome, attacks = dict(status='verified', actual_raw_spec=raw_spec(n),
                                    evidence_grade='GENBAB_NUMERICAL_FILTER'), None
        elif self.history_fallback is not None:
            outcome, attacks = self.history_fallback(history, prefix_records, deadline)
        else:
            graph = GuardedHistory(self.model, history, data['rows'], data['thresholds']).eval()
            outcome, attacks = self.coupled(graph, data['lower'], data['upper'], n, deadline)
        tick(deadline)
        return dict(outcome, guard_support=bounds, guard_closed=aggregate['closed'],
                    seconds=time.monotonic()-began), attacks


def audit_result(result, request, backend_identity, deadline):
    """All accepted conditional records must bind the same subject and domain."""
    cfg = request['model']['architecture']
    experts = [cfg['experts']]*cfg['layers']
    for record in result['prefix_bounds']:
        tick(deadline); prefix = record['prefix']
        if not prefix:
            W.require('guard_support' not in record
                and record['guarantee'] == 'WHOLE_BOX_CROWN_NUMERICAL_UPPER', 'unchanged root bounds')
            continue
        count = experts[len(prefix)]
        ordered = [(i, j) for i in range(count) for j in range(count) if i != j]
        sid = source(request, prefix, 'NEGATIVE_NEXT_ROUTER_MARGINS', backend_identity)
        bounds = record['guard_support']
        aggregate_mb(bounds, sid, prefix, 'MB_ON', len(ordered),
            sum(2*(e-2) for e in experts[:len(prefix)]), deadline)
        W.require(record['guarantee'] == 'GUARD_CONDITIONAL_CROWN_NUMERICAL_UPPER'
            and record['upper'] == [[i, j, -v] for (i, j), v in zip(ordered, bounds['lower'])],
            'conditional scope and upper/lower direction')
    for record in result['histories']:
        tick(deadline); outcome = record['outcome']; history = record['history']
        sid = source(request, history, 'OUTPUT_PROPERTIES', backend_identity)
        aggregate = aggregate_mb(outcome['guard_support'], sid, history, 'MB_ON',
            cfg['classes']-1, sum(2*(e-2) for e in experts), deadline)
        W.require(outcome['guard_closed'] is aggregate['closed'], 'same complete property aggregation')
        if outcome['guard_closed']:
            W.require(outcome['status'] == 'verified', 'positive guard-dual outcome preserved')
    tick(deadline)
