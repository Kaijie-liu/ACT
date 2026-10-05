"""Default-off exact modular determinant fixture; not a source census/verifier.

The CPU projection is a one-sided rational-rank certificate. The CUDA fixture
tests the determinant primitive only; it does not qualify GPU projection or
establish the occurrence/absence of any Neural-HZ structure in real models.
"""
from fractions import Fraction
from itertools import combinations
from math import isqrt


PRIME = 32749
MAX_BITS = 512
MAX_COLUMNS = 4096
INT64_MAX = (1 << 63) - 1
FIXTURE_ROWS = 64
FIXTURE_TRIPLES = FIXTURE_ROWS * (FIXTURE_ROWS - 1) * (FIXTURE_ROWS - 2) // 6


def check_prime_and_bounds(columns=3):
    """Prove this fixed modulus prime and bound every unreduced int64 sum."""
    if type(columns) is not int or columns <= 0:
        raise ValueError('positive integer column count required')
    if columns * (PRIME - 1) ** 2 > INT64_MAX:
        raise OverflowError('projection accumulation exceeds signed int64')
    if 6 * (PRIME - 1) ** 3 > INT64_MAX:
        raise OverflowError('determinant accumulation exceeds signed int64')
    if columns > MAX_COLUMNS:
        raise ValueError('registered projection column cap exceeded')
    if PRIME < 3 or PRIME % 2 == 0 or any(
            PRIME % divisor == 0 for divisor in range(3, isqrt(PRIME) + 1, 2)):
        raise ValueError('fixed modulus is not an odd prime')
    return True


def dyadic_mod(value):
    """Embed an exact bounded dyadic Fraction in the fixed finite field."""
    if type(value) is not Fraction:
        raise TypeError('an exact Fraction is required, not a float')
    numerator, denominator = value.numerator, value.denominator
    if max(numerator.bit_length(), denominator.bit_length()) > MAX_BITS:
        raise ValueError('512-bit rational cap exceeded')
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError('raw source denominator must be a power of two')
    if denominator % PRIME == 0:
        raise ValueError('noninvertible denominator')
    return (numerator % PRIME) * pow(denominator % PRIME, -1, PRIME) % PRIME


def det3_mod(rows):
    """Six-term determinant of canonical residues; no floating arithmetic."""
    if type(rows) not in (tuple, list) or len(rows) != 3:
        raise ValueError('three rows required')
    if any(type(row) not in (tuple, list) or len(row) != 3 for row in rows):
        raise ValueError('three columns required')
    if any(type(v) is not int or not 0 <= v < PRIME for row in rows for v in row):
        raise ValueError('canonical integer residues required')
    a, b, c = rows[0]
    d, e, f = rows[1]
    g, h, i = rows[2]
    return (a * e * i + b * f * g + c * d * h
            - c * e * g - b * d * i - a * f * h) % PRIME


def project_certificate(rows):
    """Nonzero projected minor proves rank >= 3; zero means unknown only.

    P[j] = (1, j mod p, j^2 mod p). A homomorphic image and a linear
    projection cannot increase rank. No structural-dependence conclusion is
    made on zero, and this function does not inspect input-domain supports.
    """
    if type(rows) not in (tuple, list) or len(rows) != 3:
        raise ValueError('three rational rows required')
    if any(type(row) not in (tuple, list) for row in rows):
        raise ValueError('explicit bounded rows required')
    columns = len(rows[0])
    check_prime_and_bounds(columns)
    if any(len(row) != columns for row in rows):
        raise ValueError('row widths differ')
    projection = [(1, j % PRIME, (j * j) % PRIME) for j in range(columns)]
    projected = []
    for row in rows:
        residues = [dyadic_mod(value) for value in row]
        projected.append([sum(residues[j] * projection[j][axis]
                              for j in range(columns)) % PRIME for axis in range(3)])
    determinant = det3_mod(projected)
    return dict(status='rank_at_least_3' if determinant else 'unknown',
                determinant_mod_prime=determinant, projected_rows=projected,
                prime=PRIME, projection='P[j]=(1,j mod p,j^2 mod p)',
                proves_dependence=False)


def gpu_probe(*, enabled=False):
    """Run both fixed fixtures on CUDA, checking EVERY triple on the host.

    The supervisor must set CPU1/AS16GiB, trace from before this call/import,
    enforce elapsed/RSS/traced/device limits, and save terminal failures.
    Import/initialization failure propagates; there is no CPU fallback.
    """
    if enabled is not True:
        raise RuntimeError('GPU fixture is default-off; exact enabled=True required')
    check_prime_and_bounds(3)
    import torch  # Deliberately unreachable during module import and CPU tests.

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no fallback is registered')
    device = torch.device('cuda:0')
    torch.cuda.init()
    # Do not reset peak counters or subtract runtime initialization costs.
    triples = list(combinations(range(FIXTURE_ROWS), 3))
    if len(triples) != FIXTURE_TRIPLES:
        raise RuntimeError('fixture population changed')
    indices = torch.tensor(triples, dtype=torch.int64, device=device)
    if indices.dtype != torch.int64 or tuple(indices.shape) != (FIXTURE_TRIPLES, 3):
        raise RuntimeError('invalid exact index tensor')
    fixture_counts = {}
    for name in ('vandermonde', 'dependent'):
        values = [(1, i, i * i if name == 'vandermonde' else 1 + i)
                  for i in range(FIXTURE_ROWS)]
        if any(type(v) is not int or not 0 <= v < PRIME for row in values for v in row):
            raise RuntimeError('fixture is not canonical int64 residue data')
        table = torch.tensor(values, dtype=torch.int64, device=device)
        if table.dtype != torch.int64 or tuple(table.shape) != (FIXTURE_ROWS, 3):
            raise RuntimeError('invalid exact data tensor')
        selected = table[indices]
        a, b, c = selected[:, 0, 0], selected[:, 0, 1], selected[:, 0, 2]
        d, e, f = selected[:, 1, 0], selected[:, 1, 1], selected[:, 1, 2]
        g, h, i = selected[:, 2, 0], selected[:, 2, 1], selected[:, 2, 2]
        determinants = (a * e * i + b * f * g + c * d * h
                        - c * e * g - b * d * i - a * f * h).remainder(PRIME)
        if determinants.dtype != torch.int64 or tuple(determinants.shape) != (FIXTURE_TRIPLES,):
            raise RuntimeError('invalid determinant output tensor')
        torch.cuda.synchronize(device)
        actual = determinants.cpu().tolist()
        # Independent identities, not the determinant implementation under test:
        # Vandermonde product, or col3 = col1 + col2 for the dependent fixture.
        expected = [((j - i) * (k - i) * (k - j)) % PRIME
                    if name == 'vandermonde' else 0 for i, j, k in triples]
        if (len(actual) != FIXTURE_TRIPLES
                or any(type(v) is not int or not 0 <= v < PRIME for v in actual)
                or actual != expected
                or (name == 'vandermonde' and any(v == 0 for v in actual))):
            raise RuntimeError('exact GPU fixture mismatch; no correctness admission')
        fixture_counts[name] = len(actual)

    torch.cuda.synchronize(device)
    allocated_peak = int(torch.cuda.max_memory_allocated(device))
    reserved_peak = int(torch.cuda.max_memory_reserved(device))
    allocated_current = int(torch.cuda.memory_allocated(device))
    reserved_current = int(torch.cuda.memory_reserved(device))
    if not (0 <= allocated_current <= allocated_peak <= reserved_peak
            and allocated_current <= reserved_current <= reserved_peak):
        raise RuntimeError('inconsistent CUDA allocator accounting')
    # Conservative declared scalar/data-operation bound: <= 1024 per triple
    # per fixture, including triple generation/index packing, 9-entry gather,
    # <= 18 determinant arithmetic operations and their tensor reads/writes,
    # host transfer/reference/check. 1 Mi units cover the 64-row preparation,
    # fixed primality check, shape/ledger/report operations. Runtime setup is
    # measured by the supervisor's wall/physical gates, not disguised as one
    # numerical matrix operation. CUDA allocator storage is paid separately.
    work_upper_bound = 1024 * 2 * FIXTURE_TRIPLES + 1024 ** 2
    host_entries_upper_bound = 64 * (2 * FIXTURE_TRIPLES + FIXTURE_ROWS) + 65536
    retained_entries_upper_bound = host_entries_upper_bound + (reserved_peak + 7) // 8
    if work_upper_bound > 200_000_000 or retained_entries_upper_bound > 64_000_000:
        raise RuntimeError('registered work/retained-entry bound exceeded')
    return dict(correctness=True, fixture_counts=fixture_counts,
                total_checked_determinants=2 * FIXTURE_TRIPLES,
                scope='fixed exact GPU determinant fixtures only; no source census',
                source_census_completed=False, native_HZ_admitted=False,
                formal_gain=0, new_benchmark_solves=0,
                gpu_projection_qualified=False, speedup_claimed=False,
                work_upper_bound=work_upper_bound,
                retained_entries_upper_bound=retained_entries_upper_bound,
                cuda_max_memory_allocated_bytes=allocated_peak,
                cuda_max_memory_reserved_bytes=reserved_peak,
                cuda_memory_allocated_bytes=allocated_current,
                cuda_memory_reserved_bytes=reserved_current,
                cuda_allocator_excludes_driver_context=True)
