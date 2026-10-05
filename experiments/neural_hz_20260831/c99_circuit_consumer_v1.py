"""Explicit circuit-source views and exact phase-coordinate correspondence.

These are offline mathematical components, not source/native admission receipts.
The caller must bind the complete source theorem and original phase theorem.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import SCHEMA, widen
from experiments.neural_hz_20260831.c22_uid_runs_v1 import unpack, LIMIT
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import resolve as old_resolve
from experiments.neural_hz_20260831.c70_native_proof_v1 import PACKET, equal, same_matrix
from experiments.neural_hz_20260831.c30_first_write_v1 import AppendView

PHASE='c99_circuit_phase_correspondence_v1'


@dataclass(frozen=True)
class CircuitSource:
    state: dict

    def __post_init__(self):
        if self.state['schema']!=SCHEMA or self.state['native_or_LIVE_admission']:
            raise ValueError('explicit offline circuit source required')

    def __getattr__(self,name):
        if name in self.state['fields']:return self.state['fields'][name]
        raise AttributeError(name)


def resolve(candidate,uid,*,pool):
    if type(candidate) is not CircuitSource or type(uid) is not int or not 0<=uid<LIMIT:
        raise ValueError('explicit circuit source and UID required')
    pool.charge('c99_circuit_UID_interval',16)
    start=candidate.report['radix_uid_base']+len(candidate.def_rows)
    aux=candidate.state['auxiliary_records']
    if start<=uid<start+len(aux):
        pool.charge('c99_circuit_UID_record',16)
        r=aux[uid-start]
        if int(r[1])!=uid:raise ValueError('circuit UID record differs')
        return False,int(r[0])
    return old_resolve(candidate,uid,pool=pool)


def closed_uid_tables(candidate):
    """Independent complete old-map + new-definition partition; caller pays rows."""
    if type(candidate) is not CircuitSource:raise ValueError('circuit source required')
    eq=np.full(candidate.hz.n_eq,-1,np.int64)
    le=np.full(candidate.hz.n_ineq,-1,np.int64)
    eq[candidate.eq_roots[:candidate.old_n_eq]]=np.arange(candidate.old_n_eq)
    le[candidate.ineq_roots]=candidate.old_n_eq+np.arange(len(le))
    for raw in candidate.uid_slabs:
        uid,first,length=unpack(raw)
        roots=candidate.eq_roots[candidate.old_n_eq+first:candidate.old_n_eq+first+length]
        active=roots>=0;eq[roots[active]]=uid+np.flatnonzero(active)
    eq[candidate.def_rows]=candidate.report['radix_uid_base']+np.arange(len(candidate.def_rows))
    for physical,uid,*_ in candidate.state['auxiliary_records']:
        physical,uid=int(physical),int(uid)
        if not 0<=physical<len(eq) or eq[physical]!=-1:raise ValueError('circuit row overlaps old partition')
        eq[physical]=uid
    if (np.any(eq<0) or np.any(le<0) or np.any(eq>=LIMIT) or np.any(le>=LIMIT)
        or len(set(map(int,np.r_[eq,le])))!=len(eq)+len(le)):
        raise ValueError('incomplete/reused circuit UID partition')
    return eq,le


def inject_phase(source,packet,*,pool,enabled=False):
    """Keep old source slots and inject appended phase slots above new auxiliaries.

All matrix literals/RHS/binaries are shared unchanged; only three continuous
index arrays are copied. The complete original packet stays retained by caller.
"""
    if not enabled:return None
    if type(source) is not CircuitSource:raise ValueError('explicit circuit source required')
    h=source.hz;old=source.state['old_source_n_cont'];delta=h.n_cont-old
    pool.charge('c99_complete_phase_source_binding',256+4*(len(h.c)+2*h.Gc.nnz+2*h.Gb.nnz+2*(h.n_out+1)))
    if (packet['schema']!=PACKET or not packet['offline_only'] or packet['fresh_native_execution']
        or packet['source_n_cont']!=old or packet['source_n_bin']!=h.n_bin
        or packet['old_n_cont']!=source.old_n_cont or packet['old_n_eq']!=source.old_n_eq
        or packet['logical_n_cont']!=source.logical_n_cont or packet['frame_id']!=h.frame_id
        or packet['first_uid']!=source.report['radix_uid_base']+16384
        or delta!=len(source.state['auxiliary_records']) or delta<=0
        or not equal(packet['pre_c'],h.c) or not same_matrix(packet['pre_Gb'],h.Gb)
        or not same_matrix(widen(packet['pre_Gc'],h.n_cont),h.Gc)):
        raise ValueError('complete original phase/circuit source binding differs')
    result=dict(packet)
    width=packet['Gc'].shape[1]
    for name in ('eq_c','le_c','Gc'):
        matrix=packet[name]
        pool.charge('c99_phase_continuous_injection',128+8*int(matrix.nnz))
        if (matrix.shape[1]!=width or width<old or width+delta>=2**31
            or not sp.isspmatrix_csr(matrix) or matrix.indices.dtype!=np.int32
            or not matrix.has_canonical_format):
            raise ValueError('complete original phase index domain differs')
        indices=matrix.indices.copy();indices[indices>=old]+=delta
        result[name]=sp.csr_matrix((matrix.data,indices,matrix.indptr),
            shape=(matrix.shape[0],width+delta),copy=False)
    result.update(schema=PHASE,source_n_cont=h.n_cont,pre_Gc=widen(packet['pre_Gc'],h.n_cont),
        coordinate_injection=(old,delta),original_packet_schema=PACKET)
    return result


def bind_phase(source,original,injected,*,pool,enabled=False):
    """Independently compare every packet field and exact inverse index image."""
    if not enabled:return None
    old,delta=injected['coordinate_injection'];h=source.hz
    pool.charge('c99_phase_correspondence_headers',512)
    changed={'schema','source_n_cont','pre_Gc','eq_c','le_c','Gc'}
    if (injected['schema']!=PHASE or injected['original_packet_schema']!=PACKET
        or original['schema']!=PACKET or old!=source.state['old_source_n_cont']
        or delta!=h.n_cont-old or injected['source_n_cont']!=h.n_cont
        or set(injected)!=set(original)|{'coordinate_injection','original_packet_schema'}
        or not same_matrix(injected['pre_Gc'],h.Gc)):
        raise ValueError('complete phase correspondence header differs')
    for name,value in original.items():
        if name in changed:continue
        other=injected[name]
        if sp.isspmatrix_csr(value):
            pool.charge('c99_complete_unchanged_phase_payload',4*(2*int(value.nnz)+len(value.indptr)))
            good=same_matrix(value,other)
        elif type(value) is np.ndarray:
            pool.charge('c99_complete_unchanged_phase_payload',4*int(value.size));good=equal(value,other)
        else:good=value==other
        if not good:raise ValueError('original phase field changed: '+name)
    for name in ('eq_c','le_c','Gc'):
        a,b=original[name],injected[name]
        pool.charge('c99_complete_inverse_index_image',128+12*int(a.nnz)+4*len(a.indptr))
        inverse=b.indices.astype(np.int64);inverse[inverse>=old+delta]-=delta
        if (b.shape!=(a.shape[0],a.shape[1]+delta) or not equal(a.data,b.data)
            or not equal(a.indptr,b.indptr) or not np.array_equal(a.indices,inverse)
            or np.any((b.indices>=old)&(b.indices<old+delta))):
            raise ValueError('complete inverse phase-coordinate image differs')
    view=AppendView(h,*(injected[k] for k in
        ('eq_c','eq_b','eq_rhs','le_c','le_b','le_rhs','c','Gc','Gb')))
    view.validate_shape(pool)
    return view
