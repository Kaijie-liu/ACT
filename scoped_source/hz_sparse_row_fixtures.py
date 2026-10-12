"""Pure fixed fixtures and concrete mutations; no producer/numeric imports.

The eight legacy fixtures/points and two overflow fixtures are byte-compared
with test_hz_binary64 during controls. Audit reconstruction uses this module.
"""
from copy import deepcopy
from fractions import Fraction as F
import sys

from source_enclosure.format import empty, pack, unpack
from scoped_source.check_hz_sparse_rows import REFERENCE, identity

NORMAL = ('identity', 'affine_sum', 'relu_endpoint', 'equality_rhs', 'guard_rhs',
          'subnormal', 'shared_private', 'zero_empty', 'wide_256', 'wide_1')


def owned(h, ci, bi, co=None, bo=None):
    return {'schema': REFERENCE, 'state': pack(h, ci, bi),
            'ownership': {'continuous': co if co is not None else ['input'] * len(ci),
                          'binary': bo if bo is not None else ['router'] * len(bi)}}


def legacy_patterns():
    w = F(1) + F(1, 2**52)
    h = empty(1); h['c'] = [F(1, 4)]; h['Gc'] = [{0: F(1)}]; h['Gb'] = [{0: F(1, 2)}]
    h['Ac'] = [{0: F(1)}]; h['Ab'] = [{0: F(1)}]; h['b'] = [F(0)]
    cases = {'identity': (owned(h, ['x'], ['b']), [(['-1'], ['1']), (['1'], ['-1'])])}
    h = empty(1); h['Gc'] = [{0: w*w, 1: w*w/F(2**96)}]
    cases['affine_sum'] = (owned(h, ['x', 'y'], []), [(['1', '1'], []), (['-1', '-1'], [])])
    h = empty(1); h['c'] = [F(2) + F(1, 2**54)]; h['Gc'] = [{0: F(-1)}]
    cases['relu_endpoint'] = (owned(h, ['x'], []), [(['1'], []), (['-1'], [])])
    rhs = F(1, 2**56) - F(1, 2); h = empty(1); h['Gc'] = [{0: F(1)}]
    h['Ac'] = [{0: F(1)}]; h['Ab'] = [{}]; h['b'] = [rhs]
    cases['equality_rhs'] = (owned(h, ['x'], []), [([str(rhs)], [])])
    bound = F(1) - F(1, 2**54); h = empty(1); h['Gc'] = [{0: F(1)}]
    h['Auc'] = [{0: F(1)}, {0: F(1)}]; h['Aub'] = [{}, {}]; h['ub'] = [bound, -bound]
    cases['guard_rhs'] = (owned(h, ['x'], []), [(['-1'], []), ([str(-bound)], [])])
    tiny = F(1, 2**1075); h = empty(2); h['c'] = [tiny, -tiny]
    h['Gc'] = [{0: tiny}, {0: -tiny}]; h['Auc'] = [{0: tiny}, {0: -tiny}]
    h['Aub'] = [{}, {}]; h['ub'] = [tiny, tiny]; h['Ac'] = [{0: tiny}]; h['Ab'] = [{}]; h['b'] = [F(0)]
    cases['subnormal'] = (owned(h, ['x'], []), [(['0'], [])])
    h = empty(2); h['c'] = [F(1) + F(1, 2**54)] * 2
    h['Gc'] = [{0: F(1), 1: F(1)}, {0: F(1), 2: F(1)}]; h['Gb'] = [{0: F(1)}, {1: F(1)}]
    cases['shared_private'] = (owned(h, ['input/x', 'expert0/a', 'expert1/a'], ['expert0/z', 'expert1/z'],
                                    ['shared', 'expert0', 'expert1'], ['expert0', 'expert1']),
                               [(['0', '1', '-1'], ['1', '-1']), (['1', '-1', '1'], ['-1', '1'])])
    h = empty(1); h['Ac'] = [{}]; h['Ab'] = [{}]; h['b'] = [F(0)]
    cases['zero_empty'] = (owned(h, [], []), [([], [])])
    return cases


def overflow_patterns():
    h = empty(1); h['c'] = [F(2**1024)]; first = owned(h, [], [])
    h = empty(1); h['Auc'] = [{}]; h['Aub'] = [{}]; h['ub'] = [F(sys.float_info.max) + 1]
    return [first, owned(h, [], [])]


def fixture(name):
    if name not in NORMAL:
        raise ValueError('unregistered fixture')
    if not name.startswith('wide_'):
        ref, points = legacy_patterns()[name]
        return ref, points, 'pair/0-1', 256
    w = (F(1) + F(1, 2**52))**2
    row = {j: (-1)**j * w / 2**(j % 3) for j in range(1536)}
    negative = {j: -v for j, v in row.items()}
    h = empty(2); h['c'] = [F(1) + F(1, 2**54), F(0)]
    for key in ('Gc', 'Gb'): h[key] = [dict(row), dict(negative)]
    for key in ('Ac', 'Ab', 'Auc', 'Aub'): h[key] = [dict(row)]
    h['b'] = [F(0)]; h['ub'] = [F(1) - F(1, 2**54)]
    ref = owned(h, [f'input/{j}' for j in range(1536)], [f'expert0/b/{j}' for j in range(1536)],
                ['shared'] * 1536, ['expert0'] * 1536)
    return ref, [(['1'] * 1536, ['1'] * 1536)], 'wide/row', int(name.split('_')[1])


def extend(reference, proof, assignment):
    """Construct exact test points only; not used in inclusion acceptance."""
    s, ci, bi = unpack(reference['state']); t, _, _ = unpack(proof['target']['state'])
    x, z = ([F(v) for v in assignment[k]] for k in ('continuous', 'binary'))
    result = list(x)
    for kind, vec, ck, bk in (('output', 'c', 'Gc', 'Gb'), ('equality', 'b', 'Ac', 'Ab')):
        for i, claim in enumerate(proof[kind + '_errors']):
            rho = F(claim['radius'])
            if not rho: continue
            old = sum((v*x[j] for j, v in s[ck][i].items()), F(0)) + sum((v*z[j] for j, v in s[bk][i].items()), F(0))
            new = sum((v*x[j] for j, v in t[ck][i].items() if j < len(ci)), F(0)) + sum((v*z[j] for j, v in t[bk][i].items()), F(0))
            delta = s[vec][i] + old - t[vec][i] - new if kind == 'output' else t['b'][i] - new
            result.append(delta / rho)
    return {'continuous': list(map(str, result)), 'binary': list(map(str, z))}


def math_view(proof):
    return {k: proof[k] for k in ('target', 'maps', 'output_errors', 'equality_errors')}


def expected_wide(ref):
    """Frozen algebraic target, without calling any rounding/producer routine."""
    ci, bi = ref['state']['continuous_ids'], ref['state']['binary_ids']
    anchor, owner = identity(ref), 'wide/row'
    row = {j: (-1)**j * (F(1) + F(1, 2**51)) / 2**(j % 3) for j in range(1536)}
    h = empty(2); h['c'] = [F(1), F(0)]
    for key in ('Gc', 'Gb'): h[key] = [dict(row), {j: -v for j, v in row.items()}]
    for key in ('Ac', 'Ab', 'Auc', 'Aub'): h[key] = [dict(row)]
    h['b'] = [F(0)]; h['ub'] = [F(1)]
    d = F(7, 2**96); radii = [F(1, 2**54)+d, d, d]
    ids = [f'{owner}/{anchor}/output/0', f'{owner}/{anchor}/output/1', f'{owner}/{anchor}/equality/0']
    h['Gc'][0][1536] = radii[0]; h['Gc'][1][1537] = radii[1]; h['Ac'][0][1538] = radii[2]
    target = owned(h, ci+ids, bi, ['shared']*1536+[owner]*3, ['expert0']*1536)
    return {'target': target, 'maps': {'continuous': list(range(1536)), 'binary': list(range(1536)),
                                      'flat': list(range(1536)) + list(range(1539, 3075))},
            'output_errors': [{'factor': ids[i], 'radius': str(radii[i])} for i in range(2)],
            'equality_errors': [{'factor': ids[2], 'radius': str(d)}]}


def zero_resource_proof(ref):
    from scoped_source.check_hz_sparse_rows import SCHEMA
    ci = ref['state']['continuous_ids']; target = owned(empty(1), ci, [])
    return {'schema': SCHEMA, 'reference_sha256': identity(ref), 'owner': 'resource',
            'target': target, 'target_sha256': identity(target),
            'maps': {'continuous': list(range(3)), 'binary': [], 'flat': list(range(3))},
            'output_errors': [{'factor': None, 'radius': '0'}], 'equality_errors': []}


# Finite concrete mutation roster. Audit reconstructs each from its fixed parent;
# arbitrary rejected objects cannot be substituted for a named control.
MUTATIONS = {
    'source_binding': ('identity', 'external reference binding'),
    'proof_binding': ('identity', 'identity/schema'),
    'owner': ('identity', 'identity/schema'),
    'frame': ('identity', 'original factor'),
    'owner_private': ('shared_private', 'original factor'),
    'alias': ('shared_private', 'identity alias'),
    'binary_offset': ('shared_private', 'factor maps'),
    'bool_map': ('identity', 'factor maps'),
    'missing_output': ('shared_private', 'compensation row inventory'),
    'duplicate_claim': ('shared_private', 'compensation identity'),
    'missing_equality': ('zero_empty', 'canonical CSR shape'),
    'csr_extra': ('identity', 'CSR fields'),
    'csr_duplicate': ('affine_sum', 'CSR coordinate'),
    'csr_pointer': ('identity', 'CSR pointers'),
    'csr_bool': ('identity', 'CSR coordinate'),
    'csr_zero': ('identity', 'zero CSR'),
    'noncanonical': ('identity', 'canonical rational'),
    'nonbinary64': ('identity', 'exact stored binary64'),
    'bool_frame': ('identity', 'frame/exact'),
    'exact_flag': ('identity', 'frame/exact'),
    'output_inward': ('affine_sum', 'inward compensation'),
    'equality_inward': ('equality_rhs', 'inward compensation'),
    'guard_inward': ('guard_rhs', 'inward inequality'),
    'output_sign': ('relu_endpoint', 'inward compensation'),
    'equality_sign': ('equality_rhs', 'inward compensation'),
    'output_error_crossrow': ('shared_private', 'zero layout'),
    'output_error_constraint': ('subnormal', 'zero layout'),
    'equality_error_output': ('equality_rhs', 'zero layout'),
    'equality_error_sign': ('equality_rhs', 'zero layout'),
    'partial_proof': ('identity', 'identity/schema'),
    **{f'target_only_{key}': ('wide_256', 'inward') for key in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')},
}


def mutation(name, observations):
    pattern, reason = MUTATIONS[name]; saved = observations[pattern]
    ref, proof = deepcopy(saved['reference']), deepcopy(saved['proof'])
    expected, owner = identity(ref), saved['owner']; raw = proof['target']['state']['hz']
    if name == 'source_binding': ref['state']['hz']['c'][0] = '2'
    elif name == 'proof_binding': proof['reference_sha256'] = '0' * 64
    elif name == 'owner': proof['owner'] = 'wrong'
    elif name == 'frame': raw['frame_id'] += 1
    elif name == 'owner_private': proof['target']['ownership']['continuous'][1] = 'expert1'
    elif name == 'alias': proof['target']['state']['binary_ids'][1] = 'expert0/z'
    elif name == 'binary_offset': proof['maps']['flat'] = [0, 1, 2, 3, 4]
    elif name == 'bool_map': proof['maps']['continuous'] = [False]
    elif name == 'missing_output': proof['output_errors'].pop()
    elif name == 'duplicate_claim': proof['output_errors'][1] = deepcopy(proof['output_errors'][0])
    elif name == 'missing_equality': raw['b'].clear()
    elif name == 'csr_extra': raw['Gc']['unknown'] = 0
    elif name == 'csr_duplicate': raw['Gc']['indices'][1] = 0
    elif name == 'csr_pointer': raw['Gc']['indptr'][-1] = 0
    elif name == 'csr_bool': raw['Gc']['indices'][0] = False
    elif name == 'csr_zero': raw['Gc']['data'][0] = '0'
    elif name == 'noncanonical': raw['c'][0] = '2/8'
    elif name == 'nonbinary64': raw['c'][0] = '1/3'
    elif name == 'bool_frame': raw['frame_id'] = True
    elif name == 'exact_flag': raw['exact'] = True
    elif name == 'output_inward': proof['output_errors'][0]['radius'] = '0'
    elif name == 'equality_inward': proof['equality_errors'][0]['radius'] = '0'
    elif name == 'guard_inward': raw['ub'][1] = '-1'
    elif name == 'output_sign': raw['c'][0] = '-2'
    elif name == 'equality_sign': raw['b'][0] = '1/2'
    elif name == 'partial_proof': del proof['output_errors']
    elif name.startswith('target_only_'):
        # Remove a SOURCE coordinate and bind that different exact reference.
        # The target coordinate is now target-only and compensation is insufficient.
        key = name[len('target_only_'):]
        h, ci, bi = unpack(ref['state']); del h[key][0][0]; ref['state'] = pack(h, ci, bi)
        newhash = identity(ref)
        proof['reference_sha256'] = newhash
        proof['target']['state']['continuous_ids'] = [v.replace(expected, newhash) for v in proof['target']['state']['continuous_ids']]
        for claims in ('output_errors', 'equality_errors'):
            for claim in proof[claims]:
                if claim['factor']: claim['factor'] = claim['factor'].replace(expected, newhash)
        expected = newhash
    else:
        h, ci, bi = unpack(proof['target']['state'])
        if name == 'output_error_crossrow': h['Gc'][1][3] = F(1)
        elif name == 'output_error_constraint': h['Auc'][0][1] = F(1)
        elif name == 'equality_error_output': h['Gc'][0][1] = F(1)
        elif name == 'equality_error_sign': h['Ac'][0][1] *= -1
        else: raise ValueError('unknown mutation')
        proof['target']['state'] = pack(h, ci, bi)
    proof['target_sha256'] = identity(proof['target'])
    return {'pattern': pattern, 'reference': ref, 'proof': proof, 'expected': expected,
            'owner': owner, 'reason': reason}


RESOURCE_CASES = ('factor_limit', 'row_limit', 'nnz_limit', 'stored_bits', 'intermediate_bits')

FAULT_MESSAGES = {
    'producer_chunk': 'sparse-row deadline: producer_residual_chunk',
    'checker_chunk': 'sparse-row deadline: residual_chunk',
    'producer_final_identity': 'sparse-row deadline: after_final_identity',
    'checker_final_identity': 'sparse-row deadline: after_final_identity',
    'producer_source_mutation': 'reference changed during construction',
    'checker_source_mutation': 'enclosure input changed during checking',
    'checker_proof_mutation': 'enclosure input changed during checking',
    'live_expired': 'sparse-row deadline: start',
}


def resource_case(name):
    h = empty(1); ci = ['x']; bi = []
    if name == 'factor_limit': ci = [f'x/{i}' for i in range(4097)]
    elif name == 'row_limit': h = empty(17)
    elif name == 'stored_bits': h['c'] = [F(2**4096)]
    elif name == 'intermediate_bits':
        # Pairwise coprime Mersenne denominators: gcd(2^a-1,2^b-1)=1.
        # Every stored operand is <=4096 bits; the third exact sum exceeds 8192.
        ci = ['x', 'y', 'z']; h['Gc'] = [{i: F(1, 2**n - 1) for i, n in enumerate((3001, 3011, 3019))}]
    elif name != 'nnz_limit': raise ValueError('unknown resource control')
    ref = owned(h, ci, bi)
    if name == 'nnz_limit':
        ref['state']['hz']['Gc']['data'] = ['1'] * 262145
        # Oversized malformed storage must refuse before coordinate validation.
    return ref
