"""Conditional exact witness composition after C40 half-gauge deletion.

No old post-HZ is retained or reconstructed. An explicit old-row oracle reads
only the new predicates, packed half definitions and gauge intervals. It
restores the same continuous row prefix consumed by the existing C32 lineage.

Trusted original-HZ hash, original incidence degrees and paired lineage hash
MUST come from an enclosing independent source proof. This diagnostic does not
issue that proof, a native receipt, a feasible point or an ADV verdict.
"""
from fractions import Fraction as F
import math
import numpy as np
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import ReversibleLineage,decode
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import unpack_descriptors
from experiments.neural_hz_20260831.c40_compact_half_gauge_v1 import unpack_run
from experiments.neural_hz_20260831.c40_half_gauge_inverse_v1 import inverse_hashes
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def _seal_hz(hz,pool):
    if type(hz) is not SparseHZono or hz.exact is not True:
        raise ValueError('actual exact new HZ required')
    pool.charge('half_witness_numeric_headers',512)
    entries=hz.c.size+hz.b.size+hz.ub.size
    for name in ('Gc','Gb','Ac','Ab','Auc','Aub'):
        m=getattr(hz,name)
        entries+=m.data.size+m.indices.size+m.indptr.size
    if entries>64_000_000:raise MemoryError('unchanged complete entry ceiling')
    pool.charge('half_witness_complete_new_HZ_seal',4*int(entries)+256)
    return source_digest(hz)


class _OldRows:
    """Per-call numeric reader; deliberately not a CSR/SparseHZono facade."""
    def __init__(self,new,words,runs,degrees,original_sha,*,pool):
        if type(original_sha) is not str or len(original_sha)!=64 or any(c not in '0123456789abcdef' for c in original_sha):
            raise ValueError('trusted original post-HZ hash required')
        self.pool=pool;self.new=new;self.new_sha=_seal_hz(new,pool)
        self.desc=unpack_descriptors(words,pool=pool)
        # Complete inverse image check, not a sampled old row or a success flag.
        image=inverse_hashes(new,new,self.desc,runs,degrees,pool=pool,enabled=True)
        if image['complete_original_post_sha256']!=original_sha:
            raise ValueError('inverse image differs from trusted original post-HZ')
        pool.charge('half_witness_compact_row_metadata',128*len(self.desc)+64*len(runs)+256)
        self.deleted={int(t['definition']):t for t in self.desc}
        self.definitions=self.desc['definition']
        self.children=set(map(int,self.desc['column']))
        self.parents={int(t['parent']):(int(t['column']),int(t['sign'])) for t in self.desc}
        self.ranges=[unpack_run(r) for r in runs]
        self.rows_read=0;self.terms_read=0;self.deleted_rows_read=0;self.gauged_rows_read=0

    def _position(self,row,inequality):
        if type(row) is not int or type(inequality) is not bool:
            raise ValueError('literal old row and predicate kind required')
        limit=self.new.n_ineq if inequality else self.new.n_eq+len(self.desc)
        if not 0<=row<limit:raise ValueError('old predicate row outside bound frame')
        self.pool.charge('half_witness_old_row_routing',32+16*max(1,len(self.desc).bit_length())+16*len(self.ranges))
        if not inequality and row in self.deleted:return None,False
        at=row if inequality else row-int(np.searchsorted(self.definitions,row))
        gauged=any(kind==inequality and first<=row<first+n for kind,first,n in self.ranges)
        return at,gauged

    def terms(self,row,*,inequality=False,continuous=True):
        if type(continuous) is not bool:raise ValueError('explicit numeric column domain required')
        at,gauged=self._position(row,inequality);self.rows_read+=1
        if at is None:
            self.deleted_rows_read+=1
            if not continuous:return
            self.pool.charge('half_witness_deleted_definition_image',256)
            t=self.deleted[row];p=F(math.ldexp(1.,int(t['power'])))
            self.terms_read+=2
            yield int(t['parent']),-int(t['sign'])*p/2
            yield int(t['column']),p
            return
        matrix=getattr(self.new,('Au' if inequality else 'A')+('c' if continuous else 'b'))
        a,b=map(int,matrix.indptr[at:at+2]);self.pool.charge('half_witness_exact_old_row_terms',64*(b-a)+32)
        self.gauged_rows_read+=int(gauged)
        mapped=0
        for col,value in zip(matrix.indices[a:b],matrix.data[a:b]):
            col=int(col);value=F(float(value))
            if continuous and col in self.children:
                raise ValueError('erased half factor remains in a queried new predicate')
            if gauged:
                if continuous and col in self.parents:
                    col,sign=self.parents[col];value*=sign;mapped+=1
                else:value/=2
            self.terms_read+=1
            yield col,value
        if continuous and gauged and not mapped:
            raise ValueError('gauged row lacks a mapped half occurrence')

    def rhs(self,row,*,inequality=False):
        at,gauged=self._position(row,inequality)
        self.pool.charge('half_witness_old_RHS_value',64)
        if at is None:return F(0)
        value=F(float((self.new.ub if inequality else self.new.b)[at]))
        return value/2 if gauged else value

    def finish(self):
        if _seal_hz(self.new,self.pool)!=self.new_sha:
            raise ValueError('new HZ changed during witness composition')
        return dict(old_rows_read=self.rows_read,old_numeric_terms_read=self.terms_read,
            deleted_definition_reads=self.deleted_rows_read,gauged_old_row_reads=self.gauged_rows_read,
            complete_old_HZ_materialized=False,old_post_HZ_retained_by_reader=False,
            complete_original_image_checked_against_supplied_anchor=True)


def extend(new,lineage,words,runs,degrees,original_sha,lineage_sha,continuous,*,input_n_cont,pool,enabled=False):
    if not enabled:return None
    pool.charge('half_witness_lineage_header',512)
    if type(lineage) is not ReversibleLineage or type(input_n_cont) is not int or not 0<=input_n_cont<=lineage.old_n_cont:
        raise ValueError('old exact lineage and original input frame required')
    fields=[getattr(lineage,n) for n in ('eq_roots','eq_scales','columns','retired','tails')]
    if any(type(v) is not np.ndarray or v.ndim!=1 for v in fields):raise ValueError('complete old lineage arrays required')
    seal_work=8*sum(v.size for v in fields)+256
    pool.charge('half_witness_initial_lineage_seal',seal_work);lineage.validate()
    if lineage.seal!=lineage_sha:raise ValueError('paired trusted lineage image differs')
    oracle=_OldRows(new,words,runs,degrees,original_sha,pool=pool)
    if (type(continuous) is not np.ndarray or continuous.dtype!=np.float64
            or continuous.shape!=(new.n_cont,)):
        raise ValueError('complete native continuous coordinate vector required')
    pool.charge('half_witness_complete_Fraction_frame',64*new.n_cont+16*len(lineage.eq_roots)+256)
    if not np.isfinite(continuous).all() or np.any(np.abs(continuous)>1):raise ValueError('point outside finite latent box')
    result=[F(float(v)) for v in continuous]
    aliases=[];removed=set(map(int,lineage.columns))
    logical=lineage.old_n_cont+len(lineage.eq_roots)-lineage.old_n_eq
    if not lineage.old_n_cont<=logical<=new.n_cont:raise ValueError('old MAIN frame is incomplete')
    for raw in np.flatnonzero(lineage.eq_roots<0):
        at=int(raw);col=lineage.old_n_cont+at-lineage.old_n_eq;parent=-int(lineage.eq_roots[at])-1
        ratio=float(lineage.eq_scales.view(np.float64)[at])
        if not lineage.old_n_cont<=col<logical or not 0<=parent<col or not 2.**-60<=abs(ratio)<=1:
            raise ValueError('old alias has invalid global coordinates or ratio')
        aliases.append((col,parent,ratio));removed.add(col)
    if oracle.children & removed or set(oracle.parents) & removed:
        raise ValueError('half reconstruction depends on an already removed old factor')
    if any(not lineage.old_n_cont<=c<logical for c in oracle.children):raise ValueError('half child outside old MAIN frame')
    pool.charge('half_witness_exact_half_extension',128*len(oracle.desc))
    for t in oracle.desc:
        col,parent,sign=map(int,(t['column'],t['parent'],t['sign']))
        result[col]=sign*result[parent]/2
    # C32 still maps its pre-splice consumer rank to the OLD POST rank. The
    # oracle then composes C40's second deletion and inverse row gauge.
    parent_terms=0
    for raw in lineage.columns:
        col=int(raw);at=lineage.old_n_eq+col-lineage.old_n_cont
        _,_,consumer,inequality,pivot,sign=decode(lineage.eq_roots[at])
        row=consumer if inequality else lineage.eq_row(consumer,pool=pool)
        if row is None:raise ValueError('old unit witness lost its surviving consumer')
        prefix=F(0)
        for parent,value in oracle.terms(row,inequality=inequality):
            if parent<col:
                pool.charge('half_witness_old_unit_fraction_parent',128)
                prefix+=value*result[parent];parent_terms+=1
        pool.charge('half_witness_old_unit_fraction_finish',128)
        result[col]=(F(float(lineage.eq_scales.view(np.float64)[at]))-sign*prefix)/F(pivot)
        if abs(result[col])>1:raise ValueError('old unit extension outside proved box')
    pool.charge('half_witness_old_alias_fraction_extension',64*len(aliases))
    for col,parent,ratio in aliases:result[col]=F(ratio)*result[parent]
    if any(result[i]!=F(float(continuous[i])) for i in range(input_n_cont)):
        raise ValueError('composition changed an original input latent')
    proof=oracle.finish()
    pool.charge('half_witness_final_lineage_seal',seal_work);lineage.validate()
    if lineage.seal!=lineage_sha:raise ValueError('paired old lineage changed')
    proof.update(all_continuous_slots_reconstructed=len(result),half_factors_reconstructed=len(oracle.desc),
        old_unit_factors_reconstructed=len(lineage.columns),old_alias_factors_reconstructed=len(aliases),
        old_unit_parent_terms=parent_terms,original_input_latents_unchanged=True,
        binary_factors_never_rewritten=True,source_and_native_admission_proved=False,
        point_feasibility_or_concrete_network_validation_proved=False,formal_gain=0)
    return result,proof
