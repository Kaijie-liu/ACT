"""S2a paired crossed seed/input bootstrap, distinct from S1 fixed-seed CIs."""
import numpy as np

ARMS = ('D-act', 'D-tot', 'M', 'M-reg')


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
