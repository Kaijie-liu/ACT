"""Count-based RS accounting; statistical/numerical, not deterministic HZ SAFE."""
import math


def check_counts(values, total, classes):
    if (not isinstance(values, list) or len(values) != classes or
            any(type(v) is not int or v < 0 for v in values) or sum(values) != total):
        raise ValueError('incomplete/invalid Monte Carlo counts')


def recording_smooth(native_class, n0, n, classes, expected_batch, publish):
    """Instrumentation only: no replacement sampling and no additional draws."""
    class Recorded(native_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.samples = []

        def _sample_noise(self, inputs, num, batch_size):
            expected = n0 if len(self.samples) == 0 else n
            if len(self.samples) > 1 or num != expected or batch_size != expected_batch:
                raise ValueError('native sampling call contract')
            counts = super()._sample_noise(inputs, num, batch_size)
            record = counts.tolist()
            check_counts(record, expected, classes)
            self.samples.append(record)
            publish(len(self.samples), record)
            return counts
    return Recorded


def from_counts(selection, estimation, n0, n, classes, sigma, alpha):
    from scipy.stats import beta, norm
    check_counts(selection, n0, classes)
    check_counts(estimation, n, classes)
    if not (0 < alpha < .5 and math.isfinite(sigma) and sigma > 0):
        raise ValueError('invalid confidence/noise setting')
    # Same first-index tie policy as numpy.argmax, independent of native Smooth.
    chosen = max(range(classes), key=lambda i: selection[i])
    na = estimation[chosen]
    lower = 0. if na == 0 else float(beta.ppf(alpha, na, n - na + 1))
    prediction = -1 if lower < .5 else chosen
    radius = 0. if prediction == -1 else float(sigma * norm.ppf(lower))
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('invalid RS radius')
    return {'selected_class': chosen, 'nA': na, 'p_lower': lower,
            'prediction': prediction, 'radius_l2': radius}


def combine(selector, classifier, sigma_candidates, label):
    i = selector['prediction']
    if i == -1:
        if classifier is not None:
            raise ValueError('classifier must not run after selector abstention')
        prediction, radius, sigma = -1, 0., None
    else:
        if type(i) is not int or not 0 <= i < len(sigma_candidates) or classifier is None:
            raise ValueError('missing/misbound second stage')
        sigma = sigma_candidates[i]
        prediction = classifier['prediction']
        radius = 0. if prediction == -1 else min(selector['radius_l2'], classifier['radius_l2'])
    return {'selected_sigma': sigma, 'prediction': prediction, 'radius_l2': radius,
            'correct': prediction == label,
            'correct_radius_l2': radius if prediction == label else 0.,
            'grade': 'PROBABILISTIC_RS_NATIVE_NUMERICAL', 'deterministic_formal_SAFE': False}


def summarize(receipt, inner):
    accepted = bool(receipt['status'] == 'COMPLETED' and inner and
                    inner.get('status') == 'CERTIFICATION_PILOT_AUDITED')
    return {'status': inner['status'] if accepted else
            ('ERROR' if receipt['status'] == 'COMPLETED' else receipt['status']),
            'accepted': accepted, 'partial_evidence_preserved': not accepted,
            'execution_seconds': receipt['execution_including_preflight_seconds'],
            'total_with_postflight_seconds': receipt['total_with_postflight_seconds'],
            'postflight_in_execution_budget': False,
            'deterministic_formal_SAFE': False}
