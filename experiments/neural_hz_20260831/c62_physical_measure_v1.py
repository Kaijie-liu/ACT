"""Complete known-state numeric owners and nonoverlapping Python metadata."""
import sys
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr,SparseHZAffineTerm
from act.back_end.hybridz_tf.exact_linear_op import CSRLinearOp,DiagonalLinearOp,ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots

OPAQUE=(SparseHZAffineExpr,SparseHZAffineTerm,CSRLinearOp,DiagonalLinearOp,ImplicitConv2DOp)


def numeric_layout(value,pool):
    roots={};seen=set()
    def walk(v):
        if id(v) in seen:return
        pool.charge('c62_complete_numeric_header_walk',16);seen.add(id(v))
        if type(v) is np.ndarray or sp.issparse(v) or type(v) in (SparseHZono,*OPAQUE):
            roots[str(len(roots))]=v
        elif type(v) is dict:
            for k,x in v.items():walk(k);walk(x)
        elif type(v) in (tuple,list):
            for x in v:walk(x)
        elif v is None or type(v) in (str,bytes,int,float,bool) or isinstance(v,np.generic):pass
        else:raise ValueError('unregistered complete-state header type: '+type(v).__name__)
    walk(value);pool.charge('c62_complete_numeric_owner_layout',128*len(roots)+1024)
    return snapshot_partial_csr_owners(WholeStateRoots(active=roots))


def fingerprint(value,layout,pool,*,already_paid=False):
    if not already_paid:pool.charge('c62_complete_state_fingerprint',int(layout.resident_entries)+1024)
    got=collect(SimpleNamespace(),dict(state=value))
    measured=got.measure()
    if (measured.resident_bytes,measured.resident_entries)!=(layout.resident_bytes,layout.resident_entries):
        raise ValueError('complete header and value owner ledgers disagree')
    return got.fingerprint,got.python_shallow_bytes


def metadata(value,pool):
    """Numeric backing payload is already counted; opaque inherited IDs must cancel."""
    seen=set();opaque=set();total=0
    def walk(v,buffer=False):
        nonlocal total
        if id(v) in seen:return
        seen.add(id(v));pool.charge('c62_complete_metadata_header',8)
        size=sys.getsizeof(v)
        if type(v) is np.ndarray:
            total+=size-(v.nbytes if v.flags.owndata else 0)
            if v.base is not None:walk(v.base,True)
        elif type(v) is memoryview:
            total+=size;walk(v.obj,True)
        elif type(v) in (bytes,bytearray):total+=size-(len(v) if buffer else 0)
        elif type(v) in OPAQUE:
            total+=size;opaque.add(id(v))
        elif type(v) is dict:
            total+=size
            for k,x in v.items():walk(k);walk(x)
        elif type(v) in (tuple,list):
            total+=size
            for x in v:walk(x)
        elif type(v) is SparseHZono or sp.isspmatrix_csr(v):
            total+=size;walk(vars(v))
        elif v is None or type(v) in (str,int,float,bool) or isinstance(v,np.generic):total+=size
        else:raise ValueError('unregistered nonnumeric metadata type: '+type(v).__name__)
    walk(value)
    return dict(nonoverlapping_known_metadata_bytes=total,opaque_inherited_ids=sorted(opaque),
                python_allocator_occupancy_not_measured=True)
