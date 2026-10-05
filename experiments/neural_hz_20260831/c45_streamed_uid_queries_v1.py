"""Default-off bounded query payment for the SAME complete C35 UID proof.

Old EQ ranks are visited in increasing order exactly once: sweep deleted
definitions instead of binary-searching their splice tags for every row.
Retired UID replacements use an explicit signed32 direct table on the already
fixed UID_LIMIT domain. This does NOT change source maps, HZ, old witness
lineage, C35's assignments/coverage checks, or independently checked incidence.

This is a diagnostic query adapter, not a SplicedState or admission receipt.
No actual source path has integrated or paid for this prototype yet.
"""
from types import SimpleNamespace
import hashlib
import weakref
import numpy as np
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import UID_LIMIT
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import ReversibleLineage,decode
from experiments.neural_hz_20260831.c35_current_uid_census_v1 import current_tables


class _Queries:
    def __init__(self, lineage, old_rows, pool):
        if type(lineage) is not ReversibleLineage:
            raise ValueError('actual closed-schema reversible lineage required')
        if type(old_rows) is not int or not 0 <= old_rows <= UID_LIMIT:
            raise ValueError('old row domain outside unchanged UID frame')
        # The two existing lineage seal checks are explicitly paid here; no
        # claim that source hashing/validation disappears with fast queries.
        pool.charge('query_index_header_guards',256)
        roots=[getattr(lineage,name) for name in ('eq_roots','eq_scales','columns','retired','tails')]
        if any(type(a) is not np.ndarray or a.ndim!=1 for a in roots):
            raise ValueError('closed lineage numeric arrays required')
        self.seal_charge=8*sum(a.size for a in roots)+256
        pool.charge('query_index_initial_lineage_seal',self.seal_charge)
        lineage.validate()
        pool.charge('query_index_owned_allocation_and_init',4*UID_LIMIT+64*len(lineage.columns)+256)
        self.source=lineage;self.old_rows=old_rows;self.pool=pool
        self.eq_roots=lineage.eq_roots;self.eq_scales=lineage.eq_scales
        self.columns=lineage.columns
        self.deleted=np.empty(len(lineage.columns),np.int32)
        for i,col in enumerate(lineage.columns):
            tag=decode(lineage.eq_roots[lineage.old_n_eq+int(col)-lineage.old_n_cont])
            if tag[0]!='splice' or not 0 <= tag[1] < old_rows:
                raise ValueError('splice definition outside complete original row frame')
            self.deleted[i]=tag[1]
        self.replacements=np.full(UID_LIMIT,-1,np.int32)
        for raw in lineage.retired:
            value=int(raw);old=value>>20;new=value&(UID_LIMIT-1)
            if self.replacements[old]!=-1:raise ValueError('duplicate retired UID key')
            self.replacements[old]=new
        self.deleted.flags.writeable=False;self.replacements.flags.writeable=False
        self.index_sha256=self.index_hash()
        self.cursor=0;self.next_row=0;self.retired_queries=0;self.closed=False
        self.seal=lineage.seal

    def index_hash(self):
        self.pool.charge('query_index_complete_owned_image',2*(len(self.deleted)+UID_LIMIT)+128)
        h=hashlib.sha256()
        for a,size in ((self.deleted,len(self.source.columns)),(self.replacements,UID_LIMIT)):
            if (type(a) is not np.ndarray or a.shape!=(size,) or a.dtype!=np.int32
                    or not a.flags.c_contiguous or not a.flags.owndata or a.flags.writeable):
                raise ValueError('unregistered query index owner or geometry')
            h.update(memoryview(a).cast('B'))
        return h.hexdigest()

    def validate(self):
        if self.closed or self.source.seal!=self.seal:
            raise ValueError('expired or replaced query source')
        # Called once by unchanged current_tables. Final validate below checks
        # the complete source bytes, so this is not a new provenance shortcut.
        self.pool.charge('query_index_reader_entry',64)

    def eq_row(self,old_row,*,pool):
        if pool is not self.pool or self.closed or type(old_row) is not int or old_row!=self.next_row or old_row>=self.old_rows:
            raise ValueError('rank sweep requires one complete ordered same-pool pass')
        pool.charge('query_index_ordered_EQ_rank',12)
        self.next_row+=1
        if self.cursor<len(self.deleted) and int(self.deleted[self.cursor])==old_row:
            self.cursor+=1;return None
        return old_row-self.cursor

    def retired_to(self,uid,*,pool):
        if pool is not self.pool or self.closed or type(uid) is not int or not 0<=uid<UID_LIMIT:
            raise ValueError('UID query outside paid exact domain')
        pool.charge('query_index_direct_UID_lookup',8);self.retired_queries+=1
        value=int(self.replacements[uid])
        return None if value<0 else value

    def finish(self):
        if self.closed or self.next_row!=self.old_rows or self.cursor!=len(self.deleted):
            raise ValueError('incomplete original-rank sweep')
        self.pool.charge('query_index_final_lineage_seal',self.seal_charge)
        self.source.validate()
        if self.source.seal!=self.seal:raise ValueError('query source changed during complete proof')
        if self.index_hash()!=self.index_sha256:raise ValueError('query index changed during complete proof')
        result=dict(all_original_EQ_ranks_visited=self.next_row,retired_UID_queries=self.retired_queries,
            query_owned_bytes=self.deleted.nbytes+self.replacements.nbytes,
            query_owned_entries=self.deleted.size+self.replacements.size,
            source_maps_unchanged=True,old_witness_lineage_unchanged=True)
        refs=[weakref.ref(self.deleted),weakref.ref(self.replacements)]
        self.deleted=None;self.replacements=None;self.closed=True
        if any(ref() is not None for ref in refs):raise ValueError('temporary query table retained outside proof')
        result['owned_query_tables_physically_retired']=True
        return result


def indexed_current_tables(state,*,pool,enabled=False):
    if not enabled:return None
    # Same internal precondition as current_tables: enclosing transaction must
    # authenticate actual SplicedState first and last. This function cannot
    # replace that ceremony and does not issue a source/native receipt.
    pool.charge('query_adapter_exact_input_view',256)
    view=_Queries(state.lineage,state.hz.n_eq+len(state.lineage.columns),pool)
    reader=SimpleNamespace(original_fields=state.original_fields,hz=state.hz,
        lineage=view,old_uid_ceiling=state.old_uid_ceiling)
    tables,report=current_tables(reader,pool=pool)
    payment=view.finish()
    report=dict(report,query_payment=payment,complete_source_admission_proved=False,
        actual_native_payment_proved=False,formal_gain=0)
    return tables,report
