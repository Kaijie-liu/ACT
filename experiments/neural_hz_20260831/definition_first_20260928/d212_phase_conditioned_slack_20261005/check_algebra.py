"""Once-only paper arithmetic audit; no domain imports or qualification."""

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from fractions import Fraction as Q


ROOT = Path('/data1/Kane/FSE/ACT')
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/neural_hz_20260831/results/d212_phase_conditioned_slack_20261005_v1'
NAMES = ('1', 'x', 'y', 'q1', 'q2', 'beta1', 'beta2')
PEAK_BITS = 0


def checked(value):
    global PEAK_BITS
    value = Q(value)
    bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
    PEAK_BITS = max(PEAK_BITS, bits)
    if bits > 512:
        raise ArithmeticError('512-bit ceiling exceeded')
    return value


def mul(a, b):
    return checked(checked(a) * checked(b))


def plus(a, b):
    return checked(checked(a) + checked(b))


def div(a, b):
    return checked(checked(a) / checked(b))


def form(name=None, coefficient=1):
    return tuple(checked(coefficient if name == n else 0) for n in NAMES)


def scale(coefficient, value):
    return tuple(mul(coefficient, item) for item in value)


def add(*values):
    result = form()
    for value in values:
        result = tuple(plus(a, b) for a, b in zip(result, value, strict=True))
    return result


def at(value, point):
    result = Q(0)
    for coefficient, argument in zip(value, point, strict=True):
        result = plus(result, mul(coefficient, argument))
    return result


def root_upper(value):
    if any(value[3:]):
        raise ValueError('non-parent symbols remain in root bound')
    return plus(plus(value[0], abs(value[1])), abs(value[2]))


def pair_weights(w, m):
    denominator = plus(1, -mul(m[0], m[1]))
    if denominator <= 0:
        raise ValueError('non-contractive pair')
    v = (max(Q(0), w[0], div(plus(w[0], mul(m[1], w[1])), denominator)),
         max(Q(0), w[1], div(plus(w[1], mul(m[0], w[0])), denominator)))
    residual = (plus(plus(v[0], -mul(m[1], v[1])), -w[0]),
                plus(plus(v[1], -mul(m[0], v[0])), -w[1]))
    if min(residual) < 0:
        raise ValueError('invalid fixed multiplier')
    return v


def algebra():
    one, x, y, q1, q2, beta1, beta2 = (form(n) for n in NAMES)
    f = (add(x, scale(Q(1, 12), y)), add(x, scale(-Q(1, 12), y)))
    qs, betas = (q1, q2), (beta1, beta2)
    c, u, upper_pair = Q(5, 8), Q(13, 12), Q(5, 4)
    r = (add(f[0], scale(-Q(1, 2), f[1])),
         add(f[1], scale(-Q(1, 2), f[0])))
    cap_slacks = (add(scale(Q(1, 2), add(one, scale(-1, x))),
                      scale(Q(1, 8), add(one, scale(-1, y)))),
                  add(scale(Q(1, 2), add(one, scale(-1, x))),
                      scale(Q(1, 8), add(one, y))))
    for i in range(2):
        assert add(scale(c, one), scale(-1, r[i])) == cap_slacks[i]
    t = scale(Q(4, 5), add(one, scale(-1, x)))
    box = root_upper(t)
    intrinsic = plus(1, div(u, upper_pair))
    T = max(Q(1), min(box, intrinsic))
    assert (box, intrinsic, T) == (Q(8, 5), Q(28, 15), Q(8, 5))
    tau = min(Q(1), div(1, mul(2, box)))
    old_scale = add(one, scale(-tau, t))
    lower = plus(1, -mul(tau, box))
    assert old_scale == add(scale(Q(3, 4), one), scale(Q(1, 4), x))
    assert lower == Q(1, 2)
    projected = tuple(add(scale(plus(1, -T), beta), scale(T, one), scale(-1, t))
                      for beta in betas)
    graph, old_rows, new_rows = [], [], []
    for i, j in ((0, 1), (1, 0)):
        qi, qj, bi = qs[i], qs[j], betas[i]
        graph.extend((qi, add(qi, scale(-1, f[i])),
                      add(scale(u, bi), scale(-1, qi)),
                      add(f[i], scale(u, add(one, scale(-1, bi))), scale(-1, qi))))
        old_budget = add(old_scale, scale(-lower, one), scale(lower, bi))
        old_rows.extend((qi, add(scale(upper_pair, bi), scale(-1, qi)),
                         add(scale(upper_pair, old_budget), scale(-1, qi)),
                         add(scale(Q(1, 2), qj), scale(c, bi), scale(-1, qi)),
                         add(scale(Q(1, 2), qj), scale(c, old_budget), scale(-1, qi))))
        new_rows.extend((add(scale(upper_pair, projected[i]), scale(-1, qi)),
                         add(scale(Q(1, 2), qj), scale(c, projected[i]), scale(-1, qi))))
    divisor = plus(u, mul(c, plus(T, -1)))
    h = scale(div(mul(c, u), divisor), add(scale(T, one), scale(-1, t)))
    m = div(mul(u, Q(1, 2)), divisor)
    assert h == scale(Q(13, 35), add(one, x)) and m == Q(13, 35)
    for i, j in ((0, 1), (1, 0)):
        actual = add(h, scale(m, qs[j]), scale(-1, qs[i]))
        identity = add(scale(div(u, divisor), new_rows[2*i+1]),
                       scale(div(mul(c, plus(T, -1)), divisor), graph[4*i+2]))
        assert actual == identity
    point = (Q(1), -Q(1, 2), Q(3, 4), Q(1, 5), Q(0), Q(2, 5), Q(0))
    assert all(-1 <= point[i] <= 1 for i in (1, 2))
    assert all(0 <= point[i] <= u for i in (3, 4))
    assert all(0 <= point[i] <= 1 for i in (5, 6))
    old_values = tuple(at(row, point) for row in graph + old_rows)
    assert len(old_values) == 18 and min(old_values) >= 0
    assert at(graph[3], point) == Q(1, 80)
    assert at(old_rows[4], point) == Q(1, 320)
    assert at(new_rows[1], point) == -Q(1, 10)
    w = (Q(1), -m)
    eps = Q(1, 140)
    remainder = add(scale(-1, h), scale(-eps, one))
    child = add(q1, scale(-m, q2), remainder)
    assert at(child, point) == eps
    assert pair_weights(w, (m, m)) == (Q(1), Q(0))
    new_bound = root_upper(add(h, remainder))
    assert new_bound == -eps
    triangle = scale(Q(1, 2), add(f[0], scale(u, one)))
    old_bounds = [root_upper(add(triangle, remainder))]
    for lo, sc in ((Q(1), one), (lower, old_scale), (lower, old_scale)):
        denominator = plus(mul(c, lo), u)
        hs = tuple(add(scale(div(mul(c, lo), denominator), fi),
                       scale(div(mul(c, u), denominator), sc)) for fi in f)
        mm = div(u, mul(2, denominator))
        v = pair_weights(w, (mm, mm))
        old_bounds.append(root_upper(add(scale(v[0], hs[0]), scale(v[1], hs[1]), remainder)))
    assert old_bounds == [Q(1, 3), Q(309, 5740), Q(37933, 1245580), Q(37933, 1245580)]
    assert min(old_bounds) > 0
    zero_floor_h = scale(Q(5, 16), add(one, x))
    zero_floor_v = pair_weights(w, (Q(1, 2), Q(1, 2)))
    assert zero_floor_v == (Q(38, 35), Q(6, 35))
    ablated_bound = root_upper(add(scale(plus(*zero_floor_v), zero_floor_h), remainder))
    assert ablated_bound == Q(1, 28)
    # Counterexample to extending same-T dominance after inflating B<1 to T=1.
    small_B, tt, bb = Q(3, 4), Q(3, 5), Q(1, 2)
    old_deduction = mul(div(1, mul(2, small_B)), plus(tt, -mul(small_B, plus(1, -bb))))
    new_deduction = max(Q(0), plus(tt, -plus(1, -bb)))
    assert (old_deduction, new_deduction) == (Q(3, 20), Q(1, 10))
    return {'old_row_slacks': list(map(str, old_values)),
            'new_projected_slacks': [str(at(row, point)) for row in new_rows],
            'old_fixed_child_bounds': list(map(str, old_bounds)),
            'new_fixed_child_bound': str(new_bound), 'fake_child_value': str(at(child, point)),
            'zero_floor_only_bound': str(ablated_bound), 'peak_rational_bits': PEAK_BITS,
            'assertions_completed': True}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def provenance(manifest):
    for name, expected in manifest['sha256'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('source drift: ' + name)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)
    assert git('rev-parse', 'HEAD').decode().strip() == manifest['commit']
    assert git('branch', '--show-current').decode().strip() == manifest['branch']
    assert hashlib.sha256(git('diff', '--binary', '--no-ext-diff')).hexdigest() == manifest['tracked_diff_sha256']


def save(name, data):
    with (RUN / name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def timed_out(signum, frame):
    raise TimeoutError('registered 60-second wall limit')


def main():
    started = time.monotonic()
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(60)
    os.sched_setaffinity(0, {0})
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    RUN.mkdir(parents=False, exist_ok=False)
    receipt = {'kind': 'paper_arithmetic_audit_only', 'formal_gain': 0,
               'candidate_executed': False, 'component_qualified': False,
               'native_qualified': False, 'gpu_qualified': False,
               'physical_qualified': False, 'new_benchmark_solves': 0,
               'prior_math_population_replaced': False, 'success': False,
               'pre_provenance_match': False}
    failure, manifest, freeze_digest, result = None, None, None, None
    try:
        if not __debug__:
            raise RuntimeError('assertion-stripping execution is forbidden')
        required_env = {name: '1' for name in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
        required_env['CUDA_VISIBLE_DEVICES'] = ''
        receipt['environment'] = {name: os.environ.get(name) for name in required_env}
        if receipt['environment'] != required_env:
            raise RuntimeError('registered execution environment does not match')
        freeze_digest = digest(HERE / 'freeze.json')
        manifest = json.loads((HERE / 'freeze.json').read_text())
        provenance(manifest)
        receipt['pre_provenance_match'] = True
        result = algebra()
    except BaseException as exc:
        failure = exc
        receipt['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if manifest is not None:
            try:
                provenance(manifest)
                if digest(HERE / 'freeze.json') != freeze_digest:
                    raise RuntimeError('freeze manifest drift')
                receipt['post_provenance_match'] = True
                receipt['pre_post_provenance_match'] = receipt['pre_provenance_match']
            except BaseException as exc:
                receipt['post_provenance_match'] = False
                receipt['pre_post_provenance_match'] = False
                receipt['post_check_failure'] = type(exc).__name__ + ': ' + str(exc)
                if failure is None:
                    failure = exc
            receipt['freeze_sha256'] = freeze_digest
            receipt['branch'] = manifest['branch']
            receipt['commit'] = manifest['commit']
            receipt['tracked_diff_sha256'] = manifest['tracked_diff_sha256']
        receipt['success'] = failure is None and result is not None
        if receipt['success']:
            save('algebra.json', result)
        receipt['wall_seconds'] = time.monotonic() - started
        receipt['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        receipt['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save('exit.json', receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
