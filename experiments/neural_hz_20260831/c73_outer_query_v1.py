"""Same immutable journal/wire, with exact sparse outer-interval query guards."""
from experiments.neural_hz_20260831.c68_local_splice_v1 import LocalSpliceJournal,compile_journal as original_compile

MASK=(1<<20)-1


class GuardedLocalSpliceJournal(LocalSpliceJournal):
    """Endpoints are read from the already checked immutable sparse arrays."""
    def eq_row(self,old_row,*,pool):
        if type(old_row) is not int or old_row<0:
            raise ValueError('nonnegative physical EQ row required')
        n=len(self.columns)
        if not n:
            pool.charge('c73_empty_EQ_index',4)
            return old_row
        pool.charge('c73_EQ_outer_interval_guard',16)
        first=(int(self.tags[0])>>28)&MASK
        last=(int(self.tags[-1])>>28)&MASK
        if old_row<first:return old_row
        if old_row>last:return old_row-n
        return super().eq_row(old_row,pool=pool)

    def retired_to(self,uid,*,pool):
        if not len(self.retired):
            pool.charge('c73_empty_retired_index',4)
            return None
        pool.charge('c73_UID_outer_interval_guard',16)
        first=int(self.retired[0])>>20;last=int(self.retired[-1])>>20
        if uid<first or uid>last:return None
        return super().retired_to(uid,pool=pool)


def compile_journal(roots,scales,plans,*,old_n_cont,old_n_eq,source_n_cont,
                    source_schema,pool,enabled=False):
    """Retain the complete original compiler/preconditions and exact payload."""
    if not enabled:return None
    pool.charge('c73_guarded_query_header',128)
    original=original_compile(roots,scales,plans,old_n_cont=old_n_cont,old_n_eq=old_n_eq,
        source_n_cont=source_n_cont,source_schema=source_schema,pool=pool,enabled=True)
    return GuardedLocalSpliceJournal(**vars(original))
