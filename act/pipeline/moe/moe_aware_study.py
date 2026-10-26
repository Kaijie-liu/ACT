"""B4 paired catalog and registered endpoints; no model/solver execution.

Numerical positives are not source-complete SAFE certificates. Input blocks,
not model calls, are the resampling units. Missing calls stay missing.
"""
from collections import Counter
import math
import random
import statistics


MULTILAYER = tuple(f'{family}_seed{s}' for family in
    ('L2E8', 'L3E8', 'L2E16', 'R2E8', 'R3E8') for s in range(3))
SINGLELAYER = tuple(f'L1E8_seed{s}' for s in range(3))
# Two fixed seeds from each original topology; all residual models remain in
# the main comparison. This subset is chosen before Stage5a/B4 outcomes.
SUBSET = tuple(f'{family}_seed{s}' for family in ('L2E8', 'L3E8', 'L2E16') for s in (0, 1))
CAPS = {'main': (600, 50*3600), 'ablation': (360, 30*3600),
        'radius': (240, 20*3600), 'boundary': (120, 10*3600)}
MECHANISMS = ('M-A', 'M-B', 'M-C')
STATES = {'POSITIVE', 'UNSAFE_REPLAYED', 'UNKNOWN', 'TIMEOUT', 'ERROR', 'UNSUPPORTED', 'RESOURCE_LIMIT'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def catalog(indices, components):
    """Four disjoint budgets. Main on rows also serve as ablation references.

    components must be supplied from the later, bound Stage5a decision (M-A is
    the already approved cascade). This pure catalog cannot admit a mechanism.
    """
    indices = tuple(indices); components = tuple(components)
    require(len(indices) == 20 and tuple(sorted(set(indices))) == indices,
            'twenty distinct ascending materialized input identities')
    require(components and components[0] == 'M-A'
            and tuple(m for m in MECHANISMS if m in components) == components,
            'ordered unique registered mechanisms, cascade retained')
    rows = []
    for part, models, radius, arms in (
        ('main', MULTILAYER, (2, 255), ('off', 'on')),
        ('ablation', SUBSET, (2, 255), tuple('minus_'+m for m in components)),
        ('radius', SUBSET, (4, 255), ('off', 'on')),
        ('boundary', SINGLELAYER, (2, 255), ('off', 'on')),
    ):
        for rank, index in enumerate(indices):
            for position, model in enumerate(models):
                offset = (rank+position) % len(arms)
                for arm in arms[offset:]+arms[:offset]:
                    enabled = () if arm == 'off' else tuple(m for m in components if arm != 'minus_'+m)
                    rows.append(dict(part=part, model=model, index=index, rank=rank,
                        epsilon=list(radius), arm=arm, components=list(enabled),
                        request_seconds=300,
                        reference_part='main' if part == 'ablation' else part))
    counts = Counter(r['part'] for r in rows)
    require(all(counts[p] <= n for p, (n, _) in CAPS.items()) and len(rows) <= 1320,
            'independent user budgets; unused ablations do not transfer')
    return rows


def percentile(values, p):
    values = sorted(values)
    require(values and 0 <= p <= 1, 'nonempty quantile')
    x = (len(values)-1)*p; a = math.floor(x); b = math.ceil(x)
    return values[a]+(values[b]-values[a])*(x-a)


def decision(net, rate, interval, cost_ratio, *, complete=True, conflict=False):
    if not complete:
        return 'PENDING_INCOMPLETE_QUEUE'
    if conflict:
        return 'EVIDENCE_CONFLICT_REPORT'
    require(interval is not None and cost_ratio is not None, 'complete decision data')
    # A positive point estimate with an interval containing zero is gray only
    # if neither the prespecified practical-effect nor cost condition fails.
    if net <= 0 or rate < .05 or cost_ratio > 2:
        return 'STOP_REGISTERED_GATE'
    if interval[0] <= 0 <= interval[1]:
        return 'GRAY_REPORT'
    return 'PASS' if interval[0] > 0 else 'STOP_REGISTERED_GATE'


def contrast(rows, models, indices, *, off='off', on='on', seed=20261026, replicates=10000):
    """Terminal metadata must first pass the real request/source/replay auditor.

    Call this with exactly one cohort/radius, never with pooled boundary rows.
    rows: model,index,arm,status,accepted,seconds. No accepted partial positive.
    """
    models = tuple(models); indices = tuple(indices)
    require(models and indices and len(set(models)) == len(models)
            and len(set(indices)) == len(indices), 'fixed unique cohort')
    require(type(replicates) is int and replicates > 0, 'bootstrap replicates')
    by = {}
    for row in rows:
        key = (row['model'], row['index'], row['arm'])
        require(key not in by and key[0] in models and key[1] in indices and key[2] in (off, on),
                'unique row within fixed paired cohort')
        require(row['status'] in STATES and type(row['accepted']) is bool
                and math.isfinite(row['seconds']) and 0 < row['seconds'] <= 300,
                'known terminal and complete bounded cost')
        require(row['status'] not in ('POSITIVE', 'UNSAFE_REPLAYED') or row['accepted'],
                'positive/replayed endpoint requires original online acceptance')
        by[key] = row
    missing = [(m, i, a) for i in indices for m in models for a in (off, on) if (m, i, a) not in by]
    output = dict(denominator=len(models)*len(indices), input_blocks=len(indices),
        fixed_models=list(models), missing=missing, complete=not missing,
        guarantee='GENBAB_NUMERICAL_FILTER_NOT_FORMAL_SAFE',
        states={a: dict(Counter(r['status'] for r in rows if r['arm'] == a)) for a in (off, on)},
        resampling='input blocks carrying every fixed model; no model/arm independent resampling')
    if missing:
        return dict(output, decision='PENDING_INCOMPLETE_QUEUE')
    gained, lost, intersection, unsafe, conflicts = [], [], [], [], []
    model_results = {}
    block_delta = []; paired_seconds = []
    for index in indices:
        net = 0
        for model in models:
            a, b = by[model, index, off], by[model, index, on]
            pa, pb = a['status'] == 'POSITIVE', b['status'] == 'POSITIVE'
            identity = (model, index)
            net += int(pb)-int(pa)
            if pb and not pa: gained.append(identity)
            if pa and not pb: lost.append(identity)
            if pa and pb: intersection.append(identity)
            if 'UNSAFE_REPLAYED' in (a['status'], b['status']): unsafe.append(identity)
            if {a['status'], b['status']} == {'POSITIVE', 'UNSAFE_REPLAYED'}: conflicts.append(identity)
            paired_seconds.append(b['seconds']-a['seconds'])
        block_delta.append(net/len(models))
    rng = random.Random(seed)
    boots = [statistics.mean(rng.choices(block_delta, k=len(indices))) for _ in range(replicates)]
    ci = [percentile(boots, .025), percentile(boots, .975)]
    net = len(gained)-len(lost); rate = net/output['denominator']
    costs = {a: [by[m, i, a]['seconds'] for i in indices for m in models] for a in (off, on)}
    ratio = statistics.median(costs[on])/statistics.median(costs[off])
    for m in models:
        model_results[m] = dict(gained=[i for x, i in gained if x == m],
            lost=[i for x, i in lost if x == m], intersection=[i for x, i in intersection if x == m])
    solved = {'POSITIVE', 'UNSAFE_REPLAYED'}
    solved_gain = [(m, i) for i in indices for m in models
        if by[m, i, on]['status'] in solved and by[m, i, off]['status'] not in solved]
    solved_loss = [(m, i) for i in indices for m in models
        if by[m, i, off]['status'] in solved and by[m, i, on]['status'] not in solved]
    return dict(output, gained=gained, lost=lost, intersection=intersection, net=net,
        positive_rate_delta=rate, bootstrap95=ci, bootstrap_seed=seed, bootstrap_replicates=replicates,
        degenerate_input_differences=len(set(block_delta)) == 1,
        original_replayed_unsafe=unsafe, conflicts=conflicts, by_model=model_results,
        solved_gained=solved_gain, solved_lost=solved_loss,
        cost=dict(median_ratio=ratio, median_paired_delta=statistics.median(paired_seconds),
            means={a: statistics.mean(v) for a, v in costs.items()},
            medians={a: statistics.median(v) for a, v in costs.items()},
            total_seconds={a: sum(v) for a, v in costs.items()},
            scope='all terminals including UNKNOWN/TIMEOUT/ERROR; not uncensored speedup'),
        decision=decision(net, rate, ci, ratio, conflict=bool(conflicts)))
