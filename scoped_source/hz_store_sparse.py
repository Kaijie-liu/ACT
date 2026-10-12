"""One bounded live SparseHZono workspace from immutable rows, no snapshots."""
from fractions import Fraction as F
import math

from scoped_source.hz_request_store import require


def kind_maps(factors, view):
    return tuple(j for j in sorted(view.factors) if factors[j][1] == 'continuous'), tuple(
        j for j in sorted(view.factors) if factors[j][1] == 'binary')


def instantiate(factors, view, work):
    import numpy as np
    import scipy.sparse as sp
    from act.back_end.solver.solver_hz import SparseHZono
    cids,bids = kind_maps(factors,view)
    def matrix(rows, ids):
        lookup = {j:k for k,j in enumerate(ids)}; data=[]; indices=[]; ptr=[0]
        for row in rows:
            work.visit('row_visits')
            for j,q in row.coefficients:
                work.visit('coefficient_visits')
                if j in lookup:
                    f = float(q); require(math.isfinite(f) and F(f) == q, 'nonbinary live coefficient')
                    data.append(f); indices.append(lookup[j])
            ptr.append(len(data))
        return sp.csr_matrix((data,indices,ptr),shape=(rows.count,len(ids)),dtype=np.float64)
    def vector(rows):
        out = []
        for row in rows:
            work.visit('row_visits'); f = float(row.rhs)
            require(math.isfinite(f) and F(f) == row.rhs, 'nonbinary live RHS')
            out.append(f)
        return np.asarray(out,dtype=np.float64)
    live = SparseHZono(vector(view.output),matrix(view.output,cids),matrix(view.output,bids),
        matrix(view.eq,cids),matrix(view.eq,bids),vector(view.eq),
        matrix(view.ine,cids),matrix(view.ine,bids),vector(view.ine),frame_id=view.frame,exact=False)
    work.tick()
    return live,cids,bids


def match(factors, view, live, cids, bids, work):
    """Read every live CSR/RHS entry independently; no producer serialization."""
    import numpy as np
    require((tuple(cids),tuple(bids)) == kind_maps(factors,view), 'live factor bijection/order')
    require(live.frame_id == view.frame and live.exact is False, 'live frame/enclosure flag')
    for rows,cv,bv,rv in ((view.output,live.Gc,live.Gb,live.c),
                        (view.eq,live.Ac,live.Ab,live.b),
                        (view.ine,live.Auc,live.Aub,live.ub)):
        require(rv.dtype == np.float64 and rv.shape == (rows.count,), 'live RHS shape/dtype')
        for m,ids in ((cv,cids),(bv,bids)):
            require(m.format == 'csr' and m.dtype == np.float64
                    and m.shape == (rows.count,len(ids))
                    and m.indptr.shape == (rows.count+1,)
                    and m.indices.ndim == m.data.ndim == 1
                    and np.issubdtype(m.indptr.dtype,np.integer)
                    and np.issubdtype(m.indices.dtype,np.integer)
                    and int(m.indptr[0]) == 0
                    and int(m.indptr[-1]) == len(m.indices) == len(m.data),
                    'live CSR shape/dtype/canonical')
            require(all(0 <= int(a) <= int(b) <= len(m.data)
                        for a,b in zip(m.indptr[:-1],m.indptr[1:])), 'live CSR pointers')
        for i,row in enumerate(rows):
            work.visit('row_visits')
            require(math.isfinite(float(rv[i])) and F(float(rv[i])) == row.rhs, 'live RHS mismatch')
            observed = {}
            for m,ids in ((cv,cids),(bv,bids)):
                previous = -1
                for k in range(int(m.indptr[i]),int(m.indptr[i+1])):
                    work.visit('coefficient_visits'); value = float(m.data[k]); local = int(m.indices[k])
                    require(math.isfinite(value) and previous < local < len(ids), 'live coefficient/column')
                    previous = local
                    if value: observed[ids[local]] = F(value)
            require(observed == dict(row.coefficients), 'live coefficient mismatch')
    work.tick()
    return dict(outputs=view.output.count,equalities=view.eq.count,inequalities=view.ine.count,
                continuous=len(cids),binary=len(bids))
