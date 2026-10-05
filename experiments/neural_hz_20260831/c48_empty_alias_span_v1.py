"""Default-off exact no-hit routing BEFORE unchanged alias product checks.

The immutable lookup is already required by C31. On canonical owned source
rows, a successor index proves some rows contain no local alias. No product,
numeric eligibility, UID, box or source proof is weakened or cached here.
All checks finish before the enclosing quotient may rewrite any source row.
"""
import hashlib
import weakref
import numpy as np


def _sha(value):return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


class MissIndex:
    def __init__(self,lookup,*,pool):
        if (type(lookup) is not np.ndarray or lookup.dtype!=np.int64 or lookup.ndim!=1
                or not lookup.flags.c_contiguous or not 0<len(lookup)<=64_000_000):
            raise ValueError('complete original local-alias lookup required')
        n=len(lookup)
        # Build + two complete lookup/index image pairs + final retirement.
        # All source-row membership checks remain independently necessary.
        pool.charge('c48_successor_full_construction_seals_and_retirement',10*n+512)
        if np.any(lookup<-1):raise ValueError('invalid original lookup absence sentinel')
        hit=np.flatnonzero(lookup>=0)
        pool.charge('c48_successor_known_hit_scatter',4*len(hit))
        following=np.full(n+1,n,np.int32);following[hit]=hit
        np.minimum.accumulate(following[::-1],out=following[::-1])
        following.flags.writeable=False
        self.lookup=lookup;self.following=following;self.pool=pool
        self.lookup_sha=_sha(lookup);self.index_sha=_sha(following)
        self.closed=False;self.rows=0;self.queried=0;self.missed=0;self.skipped_nnz=0

    def may_hit(self,columns,*,original_width):
        """Columns are the SAME canonical slice scanned by the old quotient.

        C31 excludes a local alias's own defining pivot BEFORE this call.
        A terminal nonalias column may also be omitted for a range proof.
        Small rows always run the unchanged old scan; they cannot be skipped.
        """
        if self.closed:raise ValueError('closed source alias read transaction')
        self.pool.charge('c48_uniform_row_width_dispatch',1)
        self.rows+=1
        if original_width<=16:return True
        self.pool.charge('c48_exact_alias_span_query',16);self.queried+=1
        if (type(columns) is not np.ndarray or columns.ndim!=1 or columns.dtype not in (np.int32,np.int64)
                or type(original_width) is not int or len(columns) not in (original_width,original_width-1)):
            raise ValueError('same original canonical continuous row slice required')
        first,last=int(columns[0]),int(columns[-1])
        n=len(self.lookup)
        if not 0<=first<=last<n:raise ValueError('row span outside complete alias lookup')
        # A nonalias defining pivot often lies far beyond the parent block.
        # Removing that known absent endpoint makes the interval proof useful
        # without assuming that every row is a MAIN definition.
        if self.lookup[last]<0:last=int(columns[-2])
        possible=int(self.following[first])<=last
        if not possible:self.missed+=1;self.skipped_nnz+=original_width
        return possible

    def finish(self):
        if self.closed:raise ValueError('source alias read transaction already finished')
        if _sha(self.lookup)!=self.lookup_sha or _sha(self.following)!=self.index_sha:
            raise ValueError('source lookup or constructed successor index changed')
        summary=dict(rows=self.rows,queried_rows=self.queried,proved_empty_rows=self.missed,
            full_row_coefficient_scans_removed=self.skipped_nnz,
            temporary_successor_bytes=self.following.nbytes,
            complete_source_incidence_or_new_quotient_proved=False,
            original_product_and_rewrite_rules_unchanged=True,formal_gain=0)
        ref=weakref.ref(self.following);self.following=None;self.lookup=None;self.closed=True
        if ref() is not None:raise ValueError('successor index has an unregistered retaining owner')
        summary['temporary_successor_physically_retired']=True
        return summary


def index(lookup,*,pool,enabled=False):
    if not enabled:return None
    return MissIndex(lookup,pool=pool)
