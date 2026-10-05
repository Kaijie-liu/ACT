"""Lossless complete native phase recovery; no original network execution."""
from fractions import Fraction as F
import hashlib
import json
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import decode
from experiments.neural_hz_20260831.c30_first_write_v1 import AppendView, row
from experiments.neural_hz_20260831.c70_native_proof_v1 import PACKET, equal


def recover(c, new, lineage, *, provenance, pool, enabled=False):
    """Caller authenticates all source/old-splice proofs before this inverse."""
    if not enabled:return None
    pool.charge('c72_complete_old_lineage_validation',8*len(lineage.eq_roots)+128)
    lineage.validate()
    pre=c.hz
    if (new.frame_id!=pre.frame_id or not new.exact or new.n_cont<pre.n_cont or new.n_bin<pre.n_bin
            or lineage.old_n_cont!=c.old_n_cont or lineage.old_n_eq!=c.old_n_eq
            or len(lineage.eq_roots)!=len(c.eq_roots)):
        raise ValueError('complete source/global/lineage binding differs')
    count=len(lineage.columns);old_eq=new.n_eq+count;old_le=new.n_ineq
    if old_eq<pre.n_eq or old_le<pre.n_ineq:raise ValueError('missing complete phase suffix')
    changes={};defs=set()
    for raw in lineage.columns:
        pool.charge('c72_bound_inverse_pair_header',160)
        col=int(raw);at=c.old_n_eq+col-c.old_n_cont
        kind,d,r,inequality,pivot,sign=decode(lineage.eq_roots[at])
        offset=float(lineage.eq_scales.view(np.float64)[at])
        if (kind!='splice' or not 0<=d<pre.n_eq or int(c.eq_roots[at])!=d
                or (inequality,r) in changes or d in defs):
            raise ValueError('complete old inverse population not source bound')
        pc,pv=row(pre.Ac,d)
        if (not len(pc) or int(pc[-1])!=col or float(pv[-1])!=pivot
                or pre.Ab.indptr[d]!=pre.Ab.indptr[d+1] or float(pre.b[d])!=offset):
            raise ValueError('original binary-free producer differs')
        target=r if inequality else lineage.eq_row(r,pool=pool)
        if target is None:raise ValueError('dependent consumer removed')
        m=new.Auc if inequality else new.Ac
        nc,nv=row(m,target);cut=int(np.searchsorted(nc,col))
        pool.charge('c72_all_original_surviving_parent_prefixes',4*len(pc)+64)
        if (cut!=len(pc)-1 or not equal(nc[:cut],pc[:-1])
                or not equal(nv[:cut],sign*pv[:-1]) or np.any(nc[cut:]<=col)):
            raise ValueError('complete source parent prefix does not invert')
        changes[inequality,r]=(col,pivot,sign,offset,cut)
        defs.add(d)
    if any(not kind and r in defs for kind,r in changes):raise ValueError('old selected dependencies')
    matrices={};rhs_out={};recovered=changed=0
    for kind,old_count,new_count in ((False,pre.n_eq,old_eq),(True,pre.n_ineq,old_le)):
        rows_c=[];rows_b=[];values=[]
        cm,bm,br=(new.Auc,new.Aub,new.ub) if kind else (new.Ac,new.Ab,new.b)
        for r in range(old_count,new_count):
            pool.charge('c72_every_original_phase_row',64)
            target=r if kind else lineage.eq_row(r,pool=pool)
            if target is None:raise ValueError('new phase cannot be a deleted old producer')
            cc,cv=row(cm,target);bc,bv=row(bm,target)
            p=changes.get((kind,r));value=float(br[target])
            if p is not None:
                col,pivot,sign,offset,cut=p
                pool.charge('c72_restore_exact_unit_head_RHS',128)
                original=F(value)-sign*F(offset);value=float(original)
                if F(value)!=original:raise ValueError('original phase RHS not exactly representable')
                cc=np.r_[np.asarray([col],np.int32),cc[cut:]]
                cv=np.r_[np.asarray([-sign*pivot],np.float64),cv[cut:]]
                changed+=1
            pool.charge('c72_copy_full_recovered_phase_row',4*(len(cc)+len(bc))+4)
            rows_c.append((cc.copy(),cv.copy()));rows_b.append((bc.copy(),bv.copy()));values.append(value)
            recovered+=1
        for name,rows,width in (('Auc' if kind else 'Ac',rows_c,new.n_cont),
                               ('Aub' if kind else 'Ab',rows_b,new.n_bin)):
            nnz=sum(len(cols) for cols,_ in rows)
            pool.charge('c72_complete_small_phase_CSR_assembly',3*nnz+4*(len(rows)+1))
            data=np.empty(nnz,np.float64);indices=np.empty(nnz,np.int32);ptr=np.zeros(len(rows)+1,np.int32)
            pos=0
            for i,(cols,vals) in enumerate(rows):
                end=pos+len(cols);indices[pos:end]=cols;data[pos:end]=vals;ptr[i+1]=end;pos=end
            m=sp.csr_matrix((data,indices,ptr),shape=(len(rows),width),copy=False)
            if not m.has_canonical_format or not np.isfinite(m.data).all() or np.any(m.data==0):
                raise ValueError('recovered original phase is not exact canonical nonzero CSR')
            matrices[name]=m
        rhs_out['ub' if kind else 'b']=np.asarray(values,np.float64)
    pool.charge('c72_copy_complete_source_and_phase_outputs',
        4*(len(pre.c)+len(new.c)+2*(pre.Gc.nnz+pre.Gb.nnz+new.Gc.nnz+new.Gb.nnz)
           +2*(pre.n_out+new.n_out+2))+512)
    packet=dict(schema=PACKET,offline_only=True,fresh_native_execution=False,
        pre_c=pre.c.copy(),pre_Gc=pre.Gc.copy(),pre_Gb=pre.Gb.copy(),frame_id=pre.frame_id,
        source_n_cont=pre.n_cont,source_n_bin=pre.n_bin,old_n_cont=c.old_n_cont,old_n_eq=c.old_n_eq,
        logical_n_cont=c.logical_n_cont,first_uid=c.report['radix_uid_base']+16384,provenance=provenance,
        eq_c=matrices['Ac'],eq_b=matrices['Ab'],eq_rhs=rhs_out['b'],
        le_c=matrices['Auc'],le_b=matrices['Aub'],le_rhs=rhs_out['ub'],
        c=new.c.copy(),Gc=new.Gc.copy(),Gb=new.Gb.copy())
    return packet,dict(all_old_inverse_pairs=count,all_original_phase_rows=recovered,
        changed_phase_rows_inverted=changed,unchanged_phase_rows=recovered-changed,
        full_original_native_hash_still_required=True,new_native_execution=False)


def native_digest(pre, packet, *, pool):
    """Exact source_digest byte stream of PRE+APPEND, no aggregate matrix."""
    view=AppendView(pre,*(packet[k] for k in ('eq_c','eq_b','eq_rhs','le_c','le_b','le_rhs','c','Gc','Gb')))
    view.validate_shape(pool)
    h=hashlib.sha256(json.dumps([pre.frame_id,len(view.c),view.n_cont,view.n_bin]).encode())
    def array(parts):
        dtype=parts[0].dtype;size=sum(p.size for p in parts)
        pool.charge('c72_complete_original_native_array_hash',4*int(size)+32)
        if any(p.dtype!=dtype or p.ndim!=1 for p in parts):raise ValueError('original flat native payload type differs')
        h.update(str((size,)).encode());h.update(str(dtype).encode())
        for p in parts:
            if not np.isfinite(p).all():raise ValueError('nonfinite original native payload')
            h.update(p.tobytes())
    for name in ('c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub'):
        if name in ('c','Gc','Gb'):
            value=getattr(view,name)
            for a in ((value.data,value.indices,value.indptr) if sp.issparse(value) else (value,)):array((a,))
        elif name in ('b','ub'):array(view.blocks(name))
        else:
            a,b=view.blocks(name)
            array((a.data,b.data));array((a.indices,b.indices))
            pool.charge('c72_original_native_appended_indptr',3*len(b.indptr))
            tail=b.indptr[1:]+np.int32(a.nnz)
            array((a.indptr,tail))
    return h.hexdigest()
