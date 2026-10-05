"""Circuit owner UID retags and exact unit/local/circuit inverse composition."""
from dataclasses import dataclass
from fractions import Fraction as F
import numpy as np
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, UID_LIMIT
from experiments.neural_hz_20260831.c73_outer_query_v1 import compile_journal as compile_local
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA as LOCAL
from experiments.neural_hz_20260831.c81_binned_inverse_v1 import reconstruct as restore_local,Point,Row
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import row
from experiments.neural_hz_20260831.c99_circuit_consumer_v2 import CircuitSource

SCHEMA='c99_explicit_circuit_native_journal_v1'
MASK=UID_LIMIT-1


@dataclass(frozen=True)
class CircuitJournal:
    local: object
    state: dict
    circuit_tails: np.ndarray
    schema: str=SCHEMA

    def numeric_roots(self):
        return dict(local=self.local.numeric_roots(),complete_source=self.state,
                    circuit_tails=self.circuit_tails)

    def iter_circuit_words(self,*,pool):
        tails=self.circuit_tails;cursor=0
        for index,record in enumerate(self.state['auxiliary_records']):
            pool.charge('c99_complete_circuit_owner_stream',20)
            value=int(record[5]);degree=value//RADIX
            while cursor<len(tails) and int(tails[cursor])>>40==index:
                pool.charge('c99_circuit_tail_UID_change',12)
                raw=int(tails[cursor]);value+=(raw&MASK)-((raw>>20)&MASK);cursor+=1
            if value<0 or value//RADIX!=degree or value%RADIX>degree*MASK:
                raise ValueError('circuit retag changes incidence degree/domain')
            yield value
        if cursor!=len(tails):raise ValueError('circuit tail outside complete source')


def compile_journal(source,plans,*,pool,enabled=False):
    if not enabled:return None
    if type(source) is not CircuitSource:raise ValueError('complete explicit circuit source required')
    local=compile_local(source.eq_roots,source.eq_scales,plans,old_n_cont=source.old_n_cont,
        old_n_eq=source.old_n_eq,source_n_cont=source.hz.n_cont,source_schema=LOCAL,pool=pool,enabled=True)
    pool.charge('c99_complete_circuit_tail_scan',256+8*sum(len(p.tail) for p in plans))
    base=source.state['old_source_n_cont'];n=source.hz.n_cont;events=[]
    for p in plans:
        # Original MAIN pivots and their lower parent prefix cannot contain new
        # auxiliary coordinates. Only surviving consumer tails change UID.
        if not p.column<source.logical_n_cont<=base:raise ValueError('circuit pivot selected as MAIN')
        for col in p.tail:
            if base<=col<n:events.append(((col-base)<<40)|(p.consumer_uid<<20)|p.producer_uid)
    t=len(events)
    pool.charge('c99_circuit_tail_array_and_sort',t+4*t*max(1,(t-1).bit_length()))
    tails=np.sort(np.asarray(events,np.uint64),kind='stable');tails.flags.writeable=False
    return CircuitJournal(local,source.state,tails)


def reconstruct(source,hz,journal,plans,continuous,*,pool,enabled=False):
    """All ordinary unit/local equations plus every new auxiliary definition.

This checks reconstruction, not feasibility of all original constraints or a
concrete neural-network witness. Phase-added coordinates remain in full_point;
source_point drops only phase slots, original_point drops only circuit slots.
"""
    if not enabled:return None
    if (type(source) is not CircuitSource or type(journal) is not CircuitJournal
        or journal.schema!=SCHEMA or journal.state is not source.state):
        raise ValueError('explicit source/native circuit inverse binding required')
    full,proof=restore_local(source,hz,journal.local,plans,continuous,pool=pool,enabled=True)
    point=Point(full,pool=pool);count=terms=0
    for record in source.state['auxiliary_records']:
        pool.charge('c99_complete_circuit_inverse_header',64)
        physical,pivot=int(record[0]),int(record[2])
        mapped=journal.local.eq_row(physical,pool=pool)
        if mapped is None or not 0<=mapped<hz.n_eq:raise ValueError('circuit inverse row removed')
        cc,cv=row(source.hz.Ac,physical)
        exact=Row.compile(cc,cv,len(full),pool=pool);exact.bound(point,pool=pool)
        if (exact.execute(point,pool=pool)!=F(float(source.hz.b[physical]))
            or not F(-1)<=full[pivot]<=F(1)):
            raise ValueError('complete original circuit equation or box not restored')
        # Native full-row identity is proved by the complete consumer audit;
        # independently verify its mapped surviving row at the recovered point.
        nc,nv=row(hz.Ac,mapped)
        actual=Row.compile(nc,nv,len(full),pool=pool);actual.bound(point,pool=pool)
        if actual.execute(point,pool=pool)!=F(float(hz.b[mapped])):
            raise ValueError('mapped native circuit equation not restored')
        count+=1;terms+=len(cc)+len(nc)
    base=source.state['old_source_n_cont'];n=source.hz.n_cont
    pool.charge('c99_complete_inverse_coordinate_projection',n+base)
    proof.update(circuit_equations=count,circuit_original_and_native_terms=terms,
        circuit_boxes_and_actual_EQ_mapping_checked=True,
        feasibility_or_concrete_witness_claim=False,formal_gain=0)
    return dict(full_point=full,source_point=full[:n],original_point=full[:base],proof=proof)
