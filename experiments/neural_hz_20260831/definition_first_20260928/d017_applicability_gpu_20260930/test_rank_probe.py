"""Five CPU-only tests; importing the candidate must not import Torch/CUDA."""
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d017_applicability_gpu_20260930 import rank_probe as r


def test_prime_and_overflow_bound():
    assert r.PRIME == 32749
    assert r.check_prime_and_bounds(r.MAX_COLUMNS) is True
    assert 6 * (r.PRIME - 1) ** 3 <= r.INT64_MAX
    with pytest.raises(OverflowError):
        r.check_prime_and_bounds(r.INT64_MAX // (r.PRIME - 1) ** 2 + 1)
    with pytest.raises(ValueError):
        r.check_prime_and_bounds(r.MAX_COLUMNS + 1)
    with pytest.raises(ValueError):
        r.check_prime_and_bounds(True)


def test_dyadic_embedding():
    assert r.dyadic_mod(F(1, 8)) * 8 % r.PRIME == 1
    assert r.dyadic_mod(F(-3, 2)) * 2 % r.PRIME == r.PRIME - 3
    assert r.dyadic_mod(F(0)) == 0
    with pytest.raises(ValueError):
        r.dyadic_mod(F(1, r.PRIME))
    with pytest.raises(ValueError):
        r.dyadic_mod(F(1, 3))
    with pytest.raises(ValueError):
        r.dyadic_mod(F(1, 1 << 512))
    with pytest.raises(TypeError):
        r.dyadic_mod(0.125)


def test_determinant_valid_nonzero():
    assert r.det3_mod([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 1
    assert r.det3_mod([[1, 0, 0], [1, 1, 1], [1, 2, 4]]) == 2
    assert r.det3_mod([[1, 0, 0], [0, 1, 0], [1, 1, 0]]) == 0
    with pytest.raises(ValueError):
        r.det3_mod([[r.PRIME, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        r.det3_mod([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    identity = [[F(int(i == j)) for j in range(3)] for i in range(3)]
    answer = r.project_certificate(identity)
    assert answer['status'] == 'rank_at_least_3'
    assert answer['determinant_mod_prime'] == 2
    assert answer['proves_dependence'] is False


def test_zero_projection_is_unknown():
    # Three independent rational rows; the last annihilates degree <= 2.
    rows = [[F(v) for v in row] for row in
            ((1, 0, 0, 0), (0, 1, 0, 0), (-1, 3, -3, 1))]
    answer = r.project_certificate(rows)
    assert answer['status'] == 'unknown'
    assert answer['determinant_mod_prime'] == 0
    assert answer['proves_dependence'] is False
    # Independent rational rows that instead lose rank upon reduction mod p.
    rows = [[F(1), F(0), F(0)], [F(0), F(1), F(0)],
            [F(0), F(0), F(r.PRIME)]]
    assert r.project_certificate(rows)['status'] == 'unknown'
    with pytest.raises(ValueError):
        r.project_certificate([[F(1)], [F(1), F(2)], [F(3)]])


def test_opt_in_required():
    with pytest.raises(RuntimeError, match='default-off'):
        r.gpu_probe()
    with pytest.raises(RuntimeError, match='default-off'):
        r.gpu_probe(enabled=1)
    with pytest.raises(RuntimeError, match='default-off'):
        r.gpu_probe(enabled=False)
