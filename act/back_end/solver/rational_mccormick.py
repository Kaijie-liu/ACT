"""Direct sparse rational weighted-property LP, before floating F0 lowering.

The supplied joint HZ has outputs [expert_a, expert_b] in one factor frame.
All arithmetic after reading its binary floating-point coefficients is exact.
This is an outer relaxation, NOT an exact linearization of softmax/product.
"""
from fractions import Fraction
from act.back_end.solver.lp_certificate import identity, rational
from act.back_end.solver.sparse_lp_certificate import rows


def csr(items, width):
    result = {'shape': [len(items), width], 'data': [], 'indices': [], 'indptr': [0]}
    for row in items:
        for j, v in sorted(row.items()):
            if v:
                result['indices'].append(j); result['data'].append(str(v))
        result['indptr'].append(len(result['data']))
    return result


def build(source, q, offset, gate, difference):
    nc, nb = source['Gc']['shape'][1], source['Gb']['shape'][1]
    n, outputs = nc + nb, len(source['c'])
    if outputs != 2 * len(q): raise ValueError('joint expert width mismatch')
    q = [rational(x) for x in q]
    def projection(weights, constant=0):
        coeff = [Fraction(0)] * n
        for name, shift, width in [('Gc',0,nc), ('Gb',nc,nb)]:
            for weight, row in zip(weights, rows(source[name],width)):
                for j, v in row: coeff[shift+j] += weight*v
        center = rational(constant) + sum((a*rational(b) for a,b in zip(weights,source['c'])), Fraction(0))
        return center, coeff
    u0, u = projection([Fraction(0)]*len(q)+q, offset)
    d0, d = projection(q+[-x for x in q])
    a,b = map(rational,gate); l,h = map(rational,difference)
    if not 0 <= a <= b <= 1 or l > h: raise ValueError('invalid product rectangle')
    constraints = {}
    for label,left,right in [('E','Ac','Ab'), ('A','Auc','Aub')]:
        left_rows=list(rows(source[left],nc)); right_rows=list(rows(source[right],nb))
        if len(left_rows)!=len(right_rows): raise ValueError('HZ constraint height mismatch')
        constraints[label]=[dict(cr+[(j+nc,v) for j,v in br]) for cr,br in zip(left_rows,right_rows)]
    rhs = [rational(x) for x in source['ub']]
    # Lower: s*d + t*lambda - w <= s*t. Upper: its negative.
    for s,t,sign in [(a,l,1),(b,h,1),(b,l,-1),(a,h,-1)]:
        row = {j:sign*s*v for j,v in enumerate(d) if v}
        row[n]=sign*t; row[n+1]=-sign
        constraints['A'].append(row); rhs.append(sign*s*(t-d0))
    corners=[x*y for x in (a,b) for y in (l,h)]
    lp={'matrix_format':'csr_v1','c':[str(v) for v in u]+['0','1'], 'offset':str(u0),
        'lower':[-1]*n+[str(a),str(min(corners))], 'upper':[1]*n+[str(b),str(max(corners))],
        'A':csr(constraints['A'],n+2),'b':[str(v) for v in rhs],
        'E':csr(constraints['E'],n+2),'h':source['b']}
    return {'schema':'shared_hz_rational_mccormick_v1', 'source':source,
            'source_sha256':identity(source), 'q':[str(v) for v in q], 'offset':str(rational(offset)),
            'gate':[str(a),str(b)],'difference':[str(l),str(h)],
            'relaxation':'BINARY_MINUS_PLUS_ONE_TO_CONTINUOUS_BOX', 'n_relaxed_binaries':nb,
            'u':{'constant':str(u0),'coefficients':[str(v) for v in u]},
            'd':{'constant':str(d0),'coefficients':[str(v) for v in d]}, 'lp':lp}
