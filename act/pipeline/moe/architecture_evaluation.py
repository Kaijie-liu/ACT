"""S1 common bound/verification semantics; opt-in, no training or source changes.

The MoE path uses all tie-legal histories or conservative subtree exclusions.
CROWN-IBP is an incomplete evaluation algorithm, not a training label. Its
first unproved history may stop the request as UNKNOWN; no partial SAFE.
"""
import itertools
import math
import time
import torch
from torch import nn
from act.back_end.moe.class_separated_top1 import validate_replay
from act.back_end.moe.genbab_pair import scalar_bad_set_clauses
from act.back_end.moe.multilayer_external import StaticPrefix, verify_histories


def tick(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('S1 shared request deadline')


class Properties(nn.Module):
    def __init__(self, model, rows, thresholds):
        super().__init__()
        if rows.ndim != 2 or thresholds.shape != (len(rows),) or len(rows) == 0:
            raise ValueError('complete nonempty property matrix')
        self.model = model
        self.register_buffer('rows', rows.detach().clone())
        self.register_buffer('bias', -thresholds.detach().clone())

    def forward(self, x):
        return torch.nn.functional.linear(self.model(x), self.rows, self.bias)


def crown_ibp(graph, lower, upper, property_count, deadline):
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    tick(deadline); began = time.monotonic()
    center = (lower + upper)/2
    bounded = BoundedModule(graph, center, device='cpu')
    x = BoundedTensor(center, PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper))
    lo, _ = bounded.compute_bounds(x=(x,), method='CROWN-IBP', bound_upper=False)
    tick(deadline)
    if tuple(lo.shape) != (1, property_count):
        raise ValueError('all property bounds required')
    finite = bool(torch.isfinite(lo).all())
    positive = finite and bool((lo > 0).all())
    return dict(status='verified' if positive else 'unknown', success=positive,
        bounds=[float(x) if math.isfinite(float(x)) else None for x in lo[0]],
        seconds=time.monotonic()-began, actual_raw_spec=scalar_bad_set_clauses(property_count),
        stage='incomplete_only', bound_propagation='CROWN-IBP', complete_verifier=None,
        branching=None, guarantee='CROWN_IBP_NUMERICAL_FILTER_NOT_INDEPENDENT_PROOF'), None


def prefix_crown_ibp(model, prefix, center, lower, upper, deadline):
    """Same incomplete bound method for pruning; no free full-CROWN queries."""
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    tick(deadline)
    graph = StaticPrefix(model, prefix, next_router=True).eval()
    e = len(model.layers[len(prefix)].experts)
    pairs = list(itertools.permutations(range(e), 2))
    rows = center.new_zeros((1, len(pairs), e))
    for k, (a, b) in enumerate(pairs):
        rows[0, k, a], rows[0, k, b] = 1., -1.
    bounded = BoundedModule(graph, center, device='cpu')
    x = BoundedTensor(center, PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper))
    _, upper_bounds = bounded.compute_bounds(x=(x,), C=rows, method='CROWN-IBP', bound_lower=False)
    tick(deadline)
    if tuple(upper_bounds.shape) != (1, len(pairs)):
        raise ValueError('ordered router margin dimensions')
    return {pair: float(upper_bounds[0, k]) for k, pair in enumerate(pairs)}


def dense_configuration():
    import genbab_cpu_ready as backend
    # Same numerical engine, CPU/double/batch/deadline/attack policy. Native
    # default kFSB is the ReLU-specific comparator, not needless sigmoid BaB.
    return dict(backend.CONFIG, **{
        'bab.branching.method': 'kfsb',
        'bab.branching.nonlinear_split.disable': True})


def evaluate(model, tensors, *, family, mode, deadline, publish=lambda _: None,
             complete_configuration=None):
    tick(deadline)
    if family not in ('dense', 'moe') or mode not in ('incomplete', 'complete'):
        raise ValueError('registered S1 family/mode')
    center, lower, upper, rows, thresholds = (tensors[k] for k in
        ('center', 'lower', 'upper', 'rows', 'thresholds'))
    if (center.shape != lower.shape or center.shape != upper.shape or len(center) != 1
            or not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(lower).all())
            or not bool(torch.isfinite(upper).all())
            or not bool(((lower <= center) & (center <= upper)).all())
            or rows.ndim != 2 or len(rows) == 0 or thresholds.shape != (len(rows),)
            or not bool(torch.isfinite(rows).all()) or not bool(torch.isfinite(thresholds).all())):
        raise ValueError('S1 input/property domain')
    with torch.no_grad():
        output = model(center)
    if output.ndim != 2 or output.shape != (1, rows.shape[1]) or not bool(torch.isfinite(output).all()):
        raise ValueError('S1 finite full-model output dimension')
    # Evaluation never requires clean-correct selection. Negative center margins
    # are full-model witnesses, not errors or omitted samples.
    if family == 'moe':
        from b2_observed_worker import verify_graph
        options = {}
        if mode == 'incomplete':
            options.update(verify_graph=crown_ibp, bound_router=prefix_crown_ibp,
                stop_on_unknown=True)
        else:
            if complete_configuration is None:
                options['verify_graph'] = verify_graph
            else:
                def configured_graph(graph, lower, upper, property_count, end):
                    return verify_graph(graph, lower, upper, property_count, end,
                                        configuration=complete_configuration)
                options['verify_graph'] = configured_graph
        result = verify_histories(model, **tensors, pruning=True, deadline=deadline,
            publish=publish, **options)
        if mode == 'incomplete' and result['status'] == 'POSITIVE':
            result['evidence_grade'] = 'CROWN_IBP_NUMERICAL_FILTER'
        for row in result['prefix_bounds']:
            row['bound_method'] = 'CROWN-IBP' if mode == 'incomplete' else 'CROWN'
    else:
        result = dict(status='UNKNOWN', expected_histories=1, verified_histories=0,
            pruned_histories=0, attempted_histories=0, returned_histories=0,
            attempted_prefix_queries=0, prefix_queries=0, histories=[], exclusions=[],
            prefix_bounds=[], complete=False, evidence_grade='NONE', formal_SAFE=False)
        if validate_replay(model, center, lower, upper, rows, thresholds):
            result.update(status='UNSAFE_REPLAYED', witness=center.tolist(), evidence_grade='FULL_MODEL_REPLAY')
        else:
            graph = Properties(model, rows, thresholds).eval()
            result['active'] = dict(phase='history_solver', prefix=[])
            result['attempted_histories'] = 1; publish(result)
            if mode == 'incomplete':
                outcome, attacks = crown_ibp(graph, lower, upper, len(rows), deadline)
            else:
                from b2_observed_worker import verify_graph
                outcome, attacks = verify_graph(graph, lower, upper, len(rows), deadline,
                    configuration=dense_configuration() if complete_configuration is None
                        else complete_configuration)
            tick(deadline)
            result['histories'] = [dict(history=[], outcome=outcome)]
            result['returned_histories'] = 1; result['active'] = None
            if attacks is not None:
                for point in attacks.detach().cpu().double().reshape(-1, *center.shape[1:]):
                    tick(deadline)
                    point = point.reshape_as(center)
                    if validate_replay(model, point, lower, upper, rows, thresholds):
                        result.update(status='UNSAFE_REPLAYED', witness=point.tolist(),
                            evidence_grade='FULL_MODEL_REPLAY')
                        break
            if outcome['status'] == 'verified':
                if result['status'] == 'UNSAFE_REPLAYED':
                    raise ValueError('positive bound conflicts with full-model witness')
                result.update(status='POSITIVE', verified_histories=1, complete=True,
                    evidence_grade='CROWN_IBP_NUMERICAL_FILTER' if mode == 'incomplete'
                        else 'ALPHA_BETA_CROWN_NUMERICAL_FILTER')
            publish(result)
    tick(deadline)
    result['evaluation'] = dict(family=family, mode=mode,
        strict_positive_required=True, clean_correct_prefilter=False,
        source_complete_proof=False)
    if complete_configuration is not None and mode == 'complete':
        result['evaluation']['explicit_complete_configuration'] = dict(complete_configuration)
    return result
