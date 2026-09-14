"""Exact finite-box dual checking for canonical CSR constraints (no solver)."""
from fractions import Fraction
from act.back_end.solver.lp_certificate import identity, rational


def rows(matrix, n):
    shape = matrix['shape']; ptr = matrix['indptr']; columns = matrix['indices']; data = matrix['data']
    if (len(shape) != 2 or any(type(v) is not int or v < 0 for v in shape) or shape[1] != n
            or len(ptr) != shape[0]+1 or ptr[0] != 0 or ptr[-1] != len(data) or len(columns) != len(data)
            or any(type(v) is not int for v in ptr+columns)):
        raise ValueError('invalid canonical CSR dimensions')
    for i in range(shape[0]):
        if not 0 <= ptr[i] <= ptr[i+1] <= len(data): raise ValueError('invalid CSR pointer')
        values = []; previous = -1
        for k in range(ptr[i], ptr[i+1]):
            j = columns[k]
            if not previous < j < n: raise ValueError('noncanonical CSR index')
            previous = j; values.append((j, rational(data[k])))
        yield values


def evaluate(lp, certificate):
    if lp.get('matrix_format') != 'csr_v1' or certificate['lp_sha256'] != identity(lp):
        raise ValueError('LP format/identity mismatch')
    c, lo, hi = ([rational(v) for v in lp[k]] for k in ('c','lower','upper'))
    n = len(c)
    if not n or len(lo) != n or len(hi) != n or any(a>b for a,b in zip(lo,hi)):
        raise ValueError('invalid finite box')
    residual = c[:]; bound = rational(lp.get('offset',0))
    for m, rhs, key, signed in (('A','b','inequality_dual',True), ('E','h','equality_dual',False)):
        dual = [rational(v) for v in certificate[key]]
        if lp[m]['shape'][0] != len(lp[rhs]) or len(dual) != len(lp[rhs]):
            raise ValueError('row/dual mismatch')
        for i, row in enumerate(rows(lp[m],n)):
            d = dual[i]
            if signed and d > 0: raise ValueError('invalid inequality dual sign')
            bound += d*rational(lp[rhs][i])
            for j,v in row: residual[j] -= d*v
    bound += sum((r*(a if r >= 0 else b) for r,a,b in zip(residual,lo,hi)), Fraction(0))
    return bound, residual


def check(lp, certificate):
    bound, residual = evaluate(lp, certificate)
    claim = rational(certificate['claimed_lower_bound'])
    if claim > bound: raise ValueError('claimed bound exceeds checked bound')
    return {'status':'CHECKED', 'lp_sha256':identity(lp), 'checked_lower_bound':str(bound),
            'claimed_lower_bound':str(claim), 'residual':[str(r) for r in residual],
            'scope':'Exact rational supplied sparse finite-box LP only; no feasibility, network or MILP proof.'}


def propose(lp, *, time_limit=None):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    def matrix(key):
        m = lp[key]
        list(rows(m,len(lp['c'])))  # fail closed on malformed sparse inputs
        if not m['shape'][0]: return None
        return csr_matrix(([float(rational(v)) for v in m['data']],m['indices'],m['indptr']),shape=m['shape'])
    vec = lambda v: [float(rational(x)) for x in v]
    res = linprog(vec(lp['c']), A_ub=matrix('A'), b_ub=vec(lp['b']) or None,
                  A_eq=matrix('E'), b_eq=vec(lp['h']) or None,
                  bounds=list(zip(vec(lp['lower']),vec(lp['upper']))), method='highs',
                  options={} if time_limit is None else {'time_limit':float(time_limit)})
    if not res.success: raise ValueError('proposal solver did not complete')
    certificate = {'lp_sha256':identity(lp), 'inequality_dual':[min(0.,float(v)) for v in res.ineqlin.marginals],
                   'equality_dual':[float(v) for v in res.eqlin.marginals]}
    certificate['claimed_lower_bound'] = str(evaluate(lp,certificate)[0])
    check(lp,certificate)
    return certificate
