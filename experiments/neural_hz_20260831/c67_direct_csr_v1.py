"""Write owned row indices directly into their final exact CSR integer width."""
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c31_prepared_owned_rows_v1 import PreparedOwnedEncoder


def matrices(encoder,rows):
    # No caller array or 'already checked' flag grants the integer range.
    # Original encoder indices are bounded by the shared frame; exact quotient
    # roots are original topological coordinates in that same frame.
    if (type(encoder) is not PreparedOwnedEncoder or not encoder.heads_discarded
            or encoder.pending_uid is not None or encoder.in_auxiliary
            or (rows is not encoder.eq and rows is not encoder.ineq)):
        raise ValueError('complete owned post-emission row population required')
    nc=encoder.nc+len(encoder.def_rows)
    cptr=np.concatenate(([0],np.cumsum([r[0].size for r in rows],dtype=np.int64)))
    bptr=np.concatenate(([0],np.cumsum([r[2].size for r in rows],dtype=np.int64)))
    ceiling=np.iinfo(np.int32).max
    cdtype=np.int32 if max(nc,len(rows),int(cptr[-1]))<=ceiling else np.int64
    bdtype=np.int32 if max(encoder.nb,len(rows),int(bptr[-1]))<=ceiling else np.int64
    def concat(index,dtype):
        # Write into the final type directly. The old full int64 concatenate
        # followed by SciPy's full int32 copy is absent, not merely hidden.
        return np.concatenate([r[index] for r in rows],dtype=dtype,casting='unsafe') if rows else np.empty(0,dtype=dtype)
    ac=sp.csr_matrix((concat(1,np.float64),concat(0,cdtype),cptr),shape=(len(rows),nc))
    ab=sp.csr_matrix((concat(3,np.float64),concat(2,bdtype),bptr),shape=(len(rows),encoder.nb))
    if not ac.has_canonical_format or not ab.has_canonical_format:raise ValueError('noncanonical radix matrix')
    return ac,ab,np.array([r[4] for r in rows],dtype=np.float64)
