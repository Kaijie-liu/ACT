"""Default-off compositional MAIN-box proof transfer, not an HZ rewrite.

An archived receipt authenticates an independent all-row DAG/quotient proof.
It is proof input, not an instance-dependent selection rule. Queries never
read the defining coefficient vector or recompute a norm. Authentication is
explicitly measured/charged; this standalone pass is not a free live append.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import time

import numpy as np

from experiments.neural_hz_20260831.c10_portable_binding_v1 import identity, verify
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Receipt:
    identity_sha256: str
    hz_sha256: str
    proof_sha256: str
    binding_sha256: str
    main_count: int
    old_n_cont: int
    logical_n_cont: int
    seal: str = ''

    def validate(self):
        expected = {'identity_sha256', 'hz_sha256', 'proof_sha256', 'binding_sha256',
            'main_count', 'old_n_cont', 'logical_n_cont', 'seal'}
        if set(vars(self)) != expected:
            raise ValueError('unregistered box receipt payload')
        fields = asdict(self)
        seal = fields.pop('seal')
        if digest(fields) != seal:
            raise ValueError('box receipt changed')


def receipt_from_archive(candidate, proof_bytes, binding_bytes, *, proof_sha256, binding_sha256):
    """Hashes must come from the enclosing preregistered independent manifest.

    No acceptance on a bare exact=True flag or caller-supplied boolean. The
    original proof must establish ALL MAIN boxes, exact quotient rows and maps;
    the portable identity binds every source/operator/content/sharing relation.
    """
    if (hashlib.sha256(proof_bytes).hexdigest() != proof_sha256
            or hashlib.sha256(binding_bytes).hexdigest() != binding_sha256):
        raise ValueError('independent proof/binding artifact hash mismatch')
    proof, binding = json.loads(proof_bytes), json.loads(binding_bytes)
    evidence = proof.get('identity', {})
    original, quotient = evidence.get('original_affine_proof', {}), evidence.get('quotient_proof', {})
    main = candidate.logical_n_cont - candidate.old_n_cont
    required = (
        proof.get('completed') is True,
        evidence.get('status') == 'EXACT_ORIGINAL_DAG_AND_QUOTIENT',
        evidence.get('tagged_physical_lineage_checked') is True,
        original.get('all_redundant_main_and_radix_boxes_proved') is True,
        original.get('all_original_coefficients_exact') is True,
        original.get('all_main_defining_rows_checked') == main,
        quotient.get('status') == 'EXACT_TWO_WAY_CONTINUOUS_QUOTIENT',
        quotient.get('redundant_boxes_proved') is True,
        quotient.get('all_binary_factors_retained') is True,
        quotient.get('original_input_prefix_unchanged') is True,
        binding.get('all_rows_proved') is True,
        binding.get('result_sha256') == proof_sha256,
        binding.get('checkpoint_sha256') == proof.get('checkpoint_sha256'),
        binding.get('identity', {}).get('hz_sha256') == proof.get('hz_sha256'),
    )
    if not all(required):
        raise ValueError('incomplete independent MAIN-box/quotient/lineage proof')
    bound = verify(candidate, binding)['portable_identity']
    fields = dict(identity_sha256=bound['sha256'], hz_sha256=bound['hz_sha256'],
        proof_sha256=proof_sha256, binding_sha256=binding_sha256, main_count=main,
        old_n_cont=candidate.old_n_cont, logical_n_cont=candidate.logical_n_cont)
    return Receipt(**fields, seal=digest(fields))


def classify_direct_main(candidate, index):
    """Constant-size row-metadata query INSIDE a bound transaction only.

    0=already erased alias, 1=direct MAIN with transferred redundant box,
    2=non-direct/radix-root shape (no norm assertion). Does not assert liveness,
    a unique consumer, RHS addition exactness, physical saving or eliminability.
    """
    if type(index) is not int or not 0 <= index < candidate.logical_n_cont - candidate.old_n_cont:
        raise ValueError('query outside MAIN prefix')
    d = int(candidate.eq_roots[candidate.old_n_eq + index])
    if d < 0:
        return 0
    hz, column = candidate.hz, candidate.old_n_cont + index
    begin, end = map(int, hz.Ac.indptr[d:d + 2])
    if begin == end or int(hz.Ac.indices[end - 1]) != column:
        return 2
    # In the authenticated canonical row, last column==j implies every other
    # continuous column<j, hence no radix link (radix columns>=logical_n_cont).
    pivot = float(hz.Ac.data[end - 1])
    if pivot <= 0. or not math.isfinite(pivot) or math.frexp(pivot)[0] != .5:
        raise ValueError('authenticated direct MAIN pivot is not positive dyadic')
    return 1


def transfer(candidate, receipt, *, enabled=False, max_work=256_000_000, observe=None):
    if not enabled:
        return None
    receipt.validate()
    pool = WorkPool(max_work)
    main = candidate.logical_n_cont - candidate.old_n_cont
    nnz = sum(getattr(candidate.hz, key).nnz for key in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'))
    # Same structural validation/authentication tariff, not a free hash pass.
    pool.charge('complete_authentication_and_metadata', 8 * int(nnz) + 32 * main)
    if nnz > 64_000_000:
        raise MemoryError('unchanged input coefficient entry ceiling exceeded')
    started = time.monotonic()
    before = identity(candidate)
    if (receipt.identity_sha256 != before['sha256'] or receipt.hz_sha256 != before['hz_sha256']
            or receipt.old_n_cont != candidate.old_n_cont or receipt.logical_n_cont != candidate.logical_n_cont
            or receipt.main_count != main):
        raise ValueError('box receipt does not bind this exact candidate')
    codes = np.empty(main, np.int8)
    parent_terms = binary_terms = 0
    for index in range(main):
        code = classify_direct_main(candidate, index)
        codes[index] = code
        if code == 1:
            d = int(candidate.eq_roots[candidate.old_n_eq + index])
            parent_terms += int(candidate.hz.Ac.indptr[d + 1] - candidate.hz.Ac.indptr[d] - 1)
            binary_terms += int(candidate.hz.Ab.indptr[d + 1] - candidate.hz.Ab.indptr[d])
    if identity(candidate) != before:
        raise ValueError('candidate mutated during bound transfer transaction')
    report = {'schema': 'c16_compositional_main_box_transfer_v1', 'formal_gain': 0,
        'complete_population': main, 'already_erased_aliases': int(np.count_nonzero(codes == 0)),
        'direct_main_boxes_transferred': int(np.count_nonzero(codes == 1)),
        'nondirect_rows_without_claim': int(np.count_nonzero(codes == 2)),
        'direct_parent_continuous_terms_not_norm_scanned': parent_terms,
        'direct_binary_terms_not_norm_scanned': binary_terms,
        'logical_work_upper': pool.used, 'work_parts': dict(pool.parts),
        'authentication_and_query_elapsed_s': time.monotonic() - started,
        'whole_identity_authentication_executed': True,
        'coefficient_norm_executed_by_transfer': False,
        'receipt_numeric_arrays': 0, 'diagnostic_code_array_bytes': codes.nbytes,
        'candidate_hz_unchanged': True, 'candidate_transformation_constructed': False,
        'solver_executed': False, 'native_ingestion_executed': False,
        'whole_live_path_proved': False, 'live_work_saving_claimed': False,
        'receipt_sha256': receipt.seal, 'candidate_identity_sha256': before['sha256']}
    if observe:
        observe({'event': 'complete_main_box_transfer', **report})
    return report, codes
