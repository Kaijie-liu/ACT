"""Frozen S1 paired-input statistics, not a post-result model selector."""
import math
import statistics
from collections import Counter

STATES = ('POSITIVE', 'UNSAFE_REPLAYED', 'UNKNOWN', 'TIMEOUT', 'ERROR', 'NOT_EXECUTED')


def model_summary(rows, expected):
    if len(rows) != expected or len({r['index'] for r in rows}) != expected:
        raise ValueError('complete fixed input denominator')
    if any(r['status'] not in STATES for r in rows):
        raise ValueError('registered verification states')
    counts = Counter(r['status'] for r in rows)
    costs = [r.get('seconds') for r in rows]
    finite = [v for v in costs if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    route = [r['route_candidate_upper'] for r in rows if r.get('route_candidate_upper') is not None]
    clean = [r['clean_correct'] for r in rows if r.get('clean_correct') is not None]
    return dict(denominator=expected, states={s:counts[s] for s in STATES},
        numerical_certified_accuracy=counts['POSITIVE']/expected,
        completed_verification=expected-counts['NOT_EXECUTED'],
        all_request_seconds=sum(finite), missing_cost=expected-len(finite),
        median_request_seconds=statistics.median(finite) if len(finite) == expected else None,
        clean_correct=sum(clean), clean_observed=len(clean), clean_unknown=expected-len(clean),
        route_observed=len(route), route_unknown=expected-len(route),
        numeric_single_candidate=sum(n == 1 for n in route),
        numeric_single_candidate_fraction_of_all=sum(n == 1 for n in route)/expected,
        route_candidate_upper_median=statistics.median(route) if route else None,
        route_interpretation='upper candidate counts from recorded numerical exclusions; not exact legal/flip census')


def paired(models_a, models_b, *, count, resamples=10000, seed=20261029):
    """Three fixed paired seeds; resample INPUT blocks with all seeds retained.

    Time compares the two all-request medians, not only successful requests.
    No interval gate is added to the user's point-estimate S1 screen.
    """
    import numpy as np
    if len(models_a) != 3 or len(models_b) != 3 or resamples <= 0:
        raise ValueError('three fixed seeds and a finite bootstrap')
    expected = set(range(count))
    for rows in models_a+models_b:
        if len(rows) != count or {r['index'] for r in rows} != expected:
            raise ValueError('all paired input identities')
    a = [sorted(rows, key=lambda r:r['index']) for rows in models_a]
    b = [sorted(rows, key=lambda r:r['index']) for rows in models_b]
    pa = np.array([[r['status'] == 'POSITIVE' for r in rows] for rows in a], dtype=float).T
    pb = np.array([[r['status'] == 'POSITIVE' for r in rows] for rows in b], dtype=float).T
    missing = sum(r['status'] == 'NOT_EXECUTED' for groups in (a,b) for rows in groups for r in rows)
    gain = [(i,s) for i in range(count) for s in range(3) if pa[i,s] and not pb[i,s]]
    loss = [(i,s) for i in range(count) for s in range(3) if pb[i,s] and not pa[i,s]]
    result = dict(denominator=count*3, input_blocks=count, fixed_models_per_arm=3,
        rates=dict(moe=float(pa.mean()),dense=float(pb.mean())),
        rate_difference=float((pa-pb).mean()), gained=len(gain),lost=len(loss),
        gained_input_seed=gain,lost_input_seed=loss,positive_intersection=int((pa*pb).sum()),
        not_executed_arm_items=missing,bootstrap_resamples=resamples,bootstrap_seed=seed,
        bootstrap_unit='input with all three paired fixed seeds',
        both_certified_rates_zero=not bool(pa.any() or pb.any()))
    valid_cost = lambda r: type(r.get('seconds')) in (int,float) and math.isfinite(r['seconds']) and r['seconds'] >= 0
    cost_missing = sum(not valid_cost(r) for groups in (a,b) for rows in groups for r in rows)
    result['missing_cost_arm_items'] = cost_missing
    if missing or cost_missing:
        result.update(decision='PENDING_INCOMPLETE_EXECUTION_OR_COST',intervals=None)
        return result
    ta = np.array([[r['seconds'] for r in rows] for rows in a], dtype=float).T
    tb = np.array([[r['seconds'] for r in rows] for rows in b], dtype=float).T
    ma,mb = float(np.median(ta)),float(np.median(tb))
    result['median_seconds'] = dict(moe=ma,dense=mb,difference=ma-mb,ratio=ma/mb if mb>0 else None)
    # Fixed point gate, including a disclosed zero-certificate degeneracy. A
    # passing screen authorizes S2; it does not itself establish the C2 claim.
    result['decision'] = 'ENTER_S2' if result['rate_difference'] >= 0 and ma < mb else 'STOP_S1_MATCH'
    differences=[];time_differences=[];time_ratios=[]
    rng=np.random.default_rng(seed)
    for start in range(0,resamples,100):
        indices=rng.integers(0,count,size=(min(100,resamples-start),count))
        differences.extend((pa-pb)[indices].mean(axis=(1,2)).tolist())
        left=np.median(ta[indices].reshape(len(indices),-1),axis=1)
        right=np.median(tb[indices].reshape(len(indices),-1),axis=1)
        time_differences.extend((left-right).tolist())
        if np.all(right>0):time_ratios.extend((left/right).tolist())
    ci=lambda values:np.quantile(values,[.025,.975]).tolist()
    result['intervals']=dict(rate_difference=ci(differences),median_seconds_difference=ci(time_differences),
        median_seconds_ratio=ci(time_ratios) if len(time_ratios)==resamples else None,
        interpretation='unadjusted descriptive paired-input intervals on these three fixed seed pairs')
    common=np.array([[a[s][i].get('clean_correct') is True and b[s][i].get('clean_correct') is True
        for s in range(3)] for i in range(count)])
    unknown_clean=sum(a[s][i].get('clean_correct') is None or b[s][i].get('clean_correct') is None
        for i in range(count) for s in range(3))
    result['shared_clean_correct']=dict(pairs=int(common.sum()),unknown_pairs=unknown_clean,
        median_moe_seconds=float(np.median(ta[common])) if common.any() else None,
        median_dense_seconds=float(np.median(tb[common])) if common.any() else None,
        interpretation='explanatory stratum only; main denominator and point gate unchanged')
    return result


def s1_decision(total_match, active_match, *, all_queue_items_settled):
    if not all_queue_items_settled or any(r['decision'].startswith('PENDING') for r in (total_match,active_match)):
        return 'PENDING_S1_COMPLETION'
    return 'ENTER_S2' if any(r['decision']=='ENTER_S2' for r in (total_match,active_match)) else 'STOP_S1_RETURN_TO_REDUCTION_PAPER'
