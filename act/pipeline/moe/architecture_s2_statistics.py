"""S2a paired crossed seed/input bootstrap, distinct from S1 fixed-seed CIs."""
import numpy as np

ARMS = ('D-act', 'D-tot', 'M', 'M-reg')
VERIFICATION_STATES = ('POSITIVE','UNSAFE_REPLAYED','UNKNOWN','TIMEOUT','ERROR','NOT_EXECUTED')


def crossed_differences(values, *, resamples=10000, seed=20261031, chunk=100):
    """values: [comparisons, seeds, inputs]. Shared draws for all comparisons.

    Independent deterministic RNG streams make the draws independent of chunk
    size. Every bootstrap replicate draws seed blocks AND common input blocks;
    a model-input cell is never resampled as an independent observation.
    """
    values = np.asarray(values)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('finite comparison/seed/input array')
    if resamples < 1 or chunk < 1:
        raise ValueError('finite positive bootstrap dimensions')
    comparisons, seeds, count = values.shape
    seed_rng, input_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    result = np.empty((resamples, comparisons), dtype=np.float64)
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples-start)
        selected = seed_rng.integers(0, seeds, size=(size, seeds))
        multiplicity = (selected[:, :, None] == np.arange(seeds)).sum(1)
        indices = input_rng.integers(0, count, size=(size, count))
        weighted = np.einsum('bs,ksn->bkn', multiplicity, values, optimize=True)
        sampled = np.take_along_axis(weighted, indices[:, None, :], axis=2)
        result[start:start+size] = sampled.sum(2) / (seeds*count)
    return result


def primary_summary(certified, clean, route_counts, *, resamples=10000,
                    seed=20261031, expected_inputs=10000, expected_seeds=5):
    """Complete primary arrays only; missing observations cannot silently vanish."""
    expected = (len(ARMS), expected_seeds, expected_inputs)
    certified, clean, route_counts = map(np.asarray, (certified, clean, route_counts))
    if (certified.shape != expected or clean.shape != expected or route_counts.shape != expected
            or certified.dtype != np.bool_ or clean.dtype != np.bool_
            or not np.issubdtype(route_counts.dtype, np.integer)):
        raise ValueError('complete paired primary identities/dtypes')
    if (certified & ~clean).any():
        raise ValueError('clean errors cannot be certified')
    if (route_counts[:2] != 0).any() or ((route_counts[2:] < 1) | (route_counts[2:] > 28)).any():
        raise ValueError('dense route sentinel or E8 complete candidate count')
    numerators = certified.sum(axis=(1, 2))
    denominator = expected_seeds*expected_inputs
    m = certified[2].astype(np.int64)
    comparisons = np.stack([m-certified[i].astype(np.int64) for i in (0, 1, 3)])
    samples = crossed_differences(comparisons, resamples=resamples, seed=seed)
    intervals = np.quantile(samples, [.025, .975], axis=0)
    deltas = numerators[2]-numerators[[0, 1, 3]]
    pairs = {}
    for j, i in enumerate((0, 1, 3)):
        pairs[f'M_minus_{ARMS[i]}'] = dict(
            rate_difference=float(deltas[j]/denominator),
            interval95=intervals[:, j].tolist(),
            gained=int((certified[2] & ~certified[i]).sum()),
            lost=int((~certified[2] & certified[i]).sum()),
            intersection=int((certified[2] & certified[i]).sum()))
    single = int((route_counts[2] == 1).sum())
    gates = dict(
        # Integer comparisons avoid a float subtraction moving an exact 2pp
        # or 90% boundary to the wrong side. The registered inequalities stand.
        m_minus_active_at_least_2pp=bool(int(deltas[0])*100 >= 2*denominator),
        m_minus_active_ci_lower_positive=bool(intervals[0, 0] > 0),
        m_minus_total_nonnegative=bool(deltas[1] >= 0),
        m_single_route_at_least_90pct=bool(single*10 >= 9*denominator))
    arms = {}
    for i, name in enumerate(ARMS):
        arms[name] = dict(denominator=denominator, certified=int(numerators[i]),
            certified_accuracy=float(numerators[i]/denominator),
            clean_correct=int(clean[i].sum()), clean_accuracy=float(clean[i].mean()),
            certified_per_seed=certified[i].sum(1).tolist(),
            clean_per_seed=clean[i].sum(1).tolist(),
            conditional_positive_fraction=(float(numerators[i]/clean[i].sum())
                                           if clean[i].any() else None),
            single_route_fraction=float((route_counts[i] == 1).mean()) if i >= 2 else None)
    return dict(arms=arms, comparisons=pairs, gates=gates,
        decision='ENTER_S2B' if all(gates.values()) else 'STOP_S2A_WRITE_RESULTS',
        input_count=expected_inputs, paired_seeds=expected_seeds,
        bootstrap=dict(resamples=resamples, seed=seed, unit='crossed paired seeds and inputs',
                       intervals='unadjusted percentile95; selected configuration held fixed'),
        conditional_interpretation='post-selection descriptive populations differ; not primary or causal',
        evidence_grade='IBP_NUMERICAL_POLICY_NOT_SOURCE_COMPLETE_PROOF')


def crossed_medians(values, *, resamples=10000, seed=20261031, chunk=64):
    """Paired all-request median costs; the same crossed draws as rate CIs."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=3 or min(values.shape)<1 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('complete nonnegative cost arrays')
    if resamples<1 or chunk<1:raise ValueError('positive bootstrap dimensions')
    arms,seeds,count=values.shape
    sr,ir=(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    result=np.empty((resamples,arms))
    for start in range(0,resamples,chunk):
        size=min(chunk,resamples-start)
        si=sr.integers(0,seeds,size=(size,seeds));ii=ir.integers(0,count,size=(size,count))
        selected=values[:,si[:,:,None],ii[:,None,:]]
        result[start:start+size]=np.median(selected,axis=(2,3)).T
    return result


def verification_summary(states, costs, *, expected_inputs, expected_seeds=5,
                         resamples=10000, seed=20261031):
    """Secondary comparisons never rescue or redefine the primary S2a gate.

    NOT_EXECUTED is kept in each planned denominator. Prefix gain/loss lists
    require both methods to have executed, and receive no final CI. TIMEOUT,
    UNKNOWN and ERROR are executed nonpositive outcomes with charged costs.
    """
    states=np.asarray(states);costs=np.asarray(costs,dtype=np.float64)
    shape=(len(ARMS),expected_seeds,expected_inputs)
    if states.shape!=shape or costs.shape!=shape or not np.isin(states,VERIFICATION_STATES).all():
        raise ValueError('complete registered secondary identities and states')
    if np.isinf(costs).any() or (costs[np.isfinite(costs)]<0).any():
        raise ValueError('finite nonnegative measured request costs')
    executed=states!='NOT_EXECUTED';positive=states=='POSITIVE'
    solved=positive|(states=='UNSAFE_REPLAYED')
    if np.isfinite(costs[~executed]).any():raise ValueError('unexecuted calls have no measured runtime')
    complete=bool(executed.all());complete_cost=bool(complete and np.isfinite(costs).all())
    deltas=np.stack([positive[2].astype(int)-positive[i].astype(int) for i in (0,1,3)])
    solved_deltas=np.stack([solved[2].astype(int)-solved[i].astype(int) for i in (0,1,3)])
    samples=crossed_differences(np.concatenate([deltas,solved_deltas]),resamples=resamples,seed=seed) if complete else None
    ci=np.quantile(samples,[.025,.975],axis=0) if samples is not None else None
    medians=crossed_medians(costs,resamples=resamples,seed=seed) if complete_cost else None
    cost_delta=crossed_differences(np.stack([costs[2]-costs[i] for i in (0,1,3)]),
                                  resamples=resamples,seed=seed) if complete_cost else None
    n=expected_seeds*expected_inputs;tables={};pairs={}
    for a,name in enumerate(ARMS):
        measured=costs[a][np.isfinite(costs[a])]
        tables[name]=dict(denominator=n,states={s:int((states[a]==s).sum()) for s in VERIFICATION_STATES},
            numerical_positive_fraction=float(positive[a].mean()),solved_fraction=float(solved[a].mean()),
            positive_per_seed=positive[a].sum(1).tolist(),solved_per_seed=solved[a].sum(1).tolist(),
            summed_request_seconds=float(measured.sum()),
            missing_executed_cost=int((executed[a]&~np.isfinite(costs[a])).sum()),
            median_all_request_seconds=float(np.median(measured)) if len(measured)==n else None,
            mean_all_request_seconds=float(measured.mean()) if len(measured)==n else None)
    for j,i in enumerate((0,1,3)):
        paired=executed[2]&executed[i]
        gained=np.argwhere(paired&positive[2]&~positive[i]).tolist()
        lost=np.argwhere(paired&~positive[2]&positive[i]).tolist()
        pa=dict(denominator=n,paired_executed=int(paired.sum()),not_executed_arm_items=int((~executed[[2,i]]).sum()),
            gained=len(gained),lost=len(lost),gained_seed_input=gained,lost_seed_input=lost,
            positive_intersection=int((paired&positive[2]&positive[i]).sum()),
            rate_difference=float(deltas[j].mean()) if complete else None,
            rate_difference_interval95=ci[:,j].tolist() if complete else None,
            solved_difference=float(solved_deltas[j].mean()) if complete else None,
            solved_difference_interval95=ci[:,j+3].tolist() if complete else None,
            all_request_cost=None)
        if complete_cost:
            ma,mb=np.median(costs[2]),np.median(costs[i])
            ratios=medians[:,2]/medians[:,i] if (medians[:,i]>0).all() else None
            pa['all_request_cost']=dict(median_m=ma,median_comparator=mb,
                median_difference=ma-mb,median_difference_interval95=np.quantile(medians[:,2]-medians[:,i],[.025,.975]).tolist(),
                ratio_of_medians=float(ma/mb) if mb>0 else None,
                ratio_of_medians_interval95=np.quantile(ratios,[.025,.975]).tolist() if ratios is not None else None,
                mean_difference=float((costs[2]-costs[i]).mean()),
                mean_difference_interval95=np.quantile(cost_delta[:,j],[.025,.975]).tolist())
        pairs[f'M_minus_{ARMS[i]}']=pa
    return dict(complete=complete,complete_cost=complete_cost,arms=tables,comparisons=pairs,
        input_count=expected_inputs,paired_seeds=expected_seeds,
        bootstrap=dict(resamples=resamples,seed=seed,unit='crossed paired seeds and inputs',
                       intervals='descriptive unadjusted percentile95; no secondary decision gate'),
        timing='all executed requests, including clean errors, unknowns, timeouts and errors; not solved-only',
        evidence_grade='NUMERICAL_FILTERS_AND_RECORDED_FULL_MODEL_REPLAYS_NOT_SOURCE_COMPLETE_PROOF')
