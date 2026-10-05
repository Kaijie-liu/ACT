"""C59's unchanged complete original-EQ binding plus normalized native cost."""
import numpy as np
from fractions import Fraction as F
from experiments.neural_hz_20260831.c59_full_consumer_census_v2 import decode_frame
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word,fraction
from experiments.neural_hz_20260831.c60_normalized_carrier_cost_v1 import CarrierCensus

def assess(saved,packet,*,pool,enabled=False,observe=None):
    if not enabled:return None
    hz=saved['hz'];old=int(saved['old_n_cont']);logical=int(saved['logical_n_cont']);old_eq=int(saved['old_n_eq'])
    if packet['n_cont']!=hz.n_cont or packet['n_bin']!=hz.n_bin or not hz.exact:raise ValueError('original HZ frame differs')
    roots,weights,proof=decode_frame(packet,pool)
    chosen=np.flatnonzero(roots!=np.arange(hz.n_cont));removed=np.zeros(hz.n_eq,bool)
    if np.any(chosen<old) or np.any(chosen>=logical):raise ValueError('old/radix coordinate cannot be removed')
    pool.charge('c59_complete_defining_row_binding',160*len(chosen)+8*hz.n_cont+hz.n_eq+1024)
    proofs=set()
    for slot in chosen:
        index=int(saved['eq_roots'][old_eq+int(slot)-old]);a,b=map(int,hz.Ac.indptr[index:index+2])
        cols=hz.Ac.indices[a:b];values=hz.Ac.data[a:b]
        if (len(cols)!=2 or int(cols[-1])!=slot or not 0<=int(cols[0])<slot
                or hz.Ab.indptr[index]!=hz.Ab.indptr[index+1] or hz.b[index]!=0 or removed[index]
                or values[-1]<=0):raise ValueError('unproved original homogeneous defining EQ')
        parent=int(cols[0]);pivot=native_word(values[-1])
        if pivot[0]!=1 or roots[slot]!=roots[parent]:raise ValueError('positive dyadic defining pivot/shared root differs')
        key=(float(values[0]),float(values[-1]),weights[parent],weights[int(slot)])
        if key not in proofs:
            if F(key[0])*fraction(key[2])+F(key[1])*fraction(key[3])!=0:raise ValueError('inverse fails original defining equation')
            proofs.add(key)
        removed[index]=True
    if observe:observe(dict(event='complete_original_defining_equations_rebound',coordinates=len(chosen),
        original_frame_coordinates=hz.n_cont,independent_equation_keys=len(proofs),inverse_packet_proof=proof,
        complete_C58_original_source_qualification_inherited=True))
    census=CarrierCensus(roots,weights,removed,pool)
    result=census.finish(hz,observe)
    result['fresh_original_defining_rows_proved']=len(chosen)
    result['inverse_packet_equations_checked']=proof
    return result
