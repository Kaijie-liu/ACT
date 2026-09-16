"""Original sparse proposal with explicit dependencies for timing, not changed math."""
from act.back_end.solver.sparse_lp_certificate import rows, evaluate, check, identity, rational
# Set only in the request-local bound function; no global mutation or eager SciPy import.
linprog = csr_matrix = None

def propose(lp, *, time_limit=None):



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
