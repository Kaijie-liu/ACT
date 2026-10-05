"""Functional half-gauge writer with compact inverse descriptors, default off.

Only an enclosing full source/incidence proof may provide half/row records.
This component does not issue a native state receipt or claim complete LIVE.
"""
import math
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono

LIMIT=1<<20;MASK=LIMIT-1
DESC=np.dtype([('column','i4'),('parent','i4'),('definition','i4'),
    ('power','i1'),('sign','i1'),('negative_zero','?')])


def pack_run(kind,start,length):
    if type(kind) is not bool or type(start) is not int or type(length) is not int or not 0<=start<start+length<=LIMIT:
        raise ValueError('invalid complete row gauge interval')
    return np.uint64((int(kind)<<40)|((length-1)<<20)|start)


def unpack_run(raw):
    value=int(raw)
    if not 0<=value<(1<<41):raise ValueError('reserved row gauge interval bits')
    kind=bool(value>>40);start=value&MASK;length=((value>>20)&MASK)+1
    if start+length>LIMIT:raise ValueError('row gauge interval exceeds frame')
    return kind,start,length


def schedules(runs,*,pool):
    pool.charge('half_gauge_interval_decode',32*len(runs))
    if type(runs) is not np.ndarray or runs.dtype!=np.uint64 or runs.ndim!=1:raise ValueError('compact uint64 intervals required')
    out=[[],[]];previous=None
    for raw in runs:
        kind,start,length=unpack_run(raw)
        if previous is not None and (kind<previous[0] or (kind==previous[0] and start<=previous[1])):
            raise ValueError('noncanonical or unmerged gauge intervals')
        previous=(kind,start+length)
        pool.charge('half_gauge_diagnostic_row_schedule',4*length)
        out[int(kind)].extend(range(start,start+length))
    return out


def describe(hz,half,rows,*,pool):
    k=len(half);pool.charge('half_compact_definition_and_descriptor',256*k)
    if not k or len(rows)>=LIMIT:raise ValueError('nonempty complete half-alias transaction required')
    desc=np.empty(k,DESC);columns=set();parents=set();definitions=set()
    for i,item in enumerate(half):
        col,parent,d,sign=map(int,(item['column'],item['parent'],item['definition'],item['sign']))
        if (not 0<=parent<col<hz.n_cont or not 0<=d<hz.n_eq or sign not in (-1,1)
                or col in columns or parent in parents or d in definitions
                or item['definitions_seen']!=1 or item['seen']!=item['degree']-1):
            raise ValueError('incomplete independent half descriptor')
        a,b=map(int,hz.Ac.indptr[d:d+2]);p=float(hz.Ac.data[b-1]) if b>a else 0.
        if (b-a!=2 or list(map(int,hz.Ac.indices[a:b]))!=[parent,col]
                or hz.Ab.indptr[d]!=hz.Ab.indptr[d+1] or hz.b[d]!=0.
                or not 2.**-20<=p<=2.**40 or math.frexp(p)[0]!=.5
                or float(hz.Ac.data[a])!=-sign*p/2):
            raise ValueError('half descriptor differs from actual defining row')
        columns.add(col);parents.add(parent);definitions.add(d)
        desc[i]=(col,parent,d,math.frexp(p)[1]-1,sign,bool(np.signbit(hz.b[d])))
    if columns & parents:raise ValueError('internal half dependency')
    pool.charge('half_compact_descriptor_sort',12*k*max(1,k.bit_length()))
    desc.sort(order='definition')
    if set(map(int,hz.Gc.indices)) & columns:raise ValueError('half alias still output-live')
    pool.charge('half_complete_row_binding_and_compression',64*len(rows))
    intervals=[];previous=None;occurrences=0;total_c=total_b=0
    for r in rows:
        kind=bool(r['inequality']);row=int(r['row']);key=(kind,row)
        m,bm=(hz.Auc,hz.Aub) if kind else (hz.Ac,hz.Ab)
        if (previous is not None and key<=previous) or not 0<=row<m.shape[0] or (not kind and row in definitions):
            raise ValueError('reused/unordered/deleted actual gauge row')
        previous=key;a,b=map(int,m.indptr[row:row+2]);cc=m.indices[a:b]
        pool.charge('half_global_parent_isolation_and_occurrences',4*len(cc))
        actual=sum(int(c) in columns for c in cc)
        if (r['decision']!=1 or actual!=int(r['half_occurrences']) or actual==0
                or int(r['source_continuous_nnz'])!=len(cc) or int(r['new_continuous_nnz'])!=len(cc)
                or int(r['binary_nnz'])!=int(bm.indptr[row+1]-bm.indptr[row])
                or any(int(c) in parents for c in cc)):
            raise ValueError('complete row or GLOBAL selected-parent isolation failed')
        occurrences+=actual;total_c+=len(cc);total_b+=int(r['binary_nnz'])
        if intervals and intervals[-1][0]==kind and intervals[-1][1]+intervals[-1][2]==row:
            intervals[-1][2]+=1
        else:intervals.append([kind,row,1])
    if occurrences!=int(half['seen'].sum()):raise ValueError('incomplete all-consumer row schedule')
    pool.charge('half_compact_interval_payload',16*len(intervals))
    runs=np.array([pack_run(*r) for r in intervals],np.uint64)
    return desc,runs,dict(global_selected_parents_absent_from_all_gauged_source_rows=True,
        all_half_consumer_occurrences=occurrences,gauged_rows=len(rows),gauge_intervals=len(runs),
        touched_continuous_nnz=total_c,touched_binary_nnz=total_b,compact_payload_bytes=desc.nbytes+runs.nbytes)


def _matrix(source,desc,gauge,*,continuous,inequality,pool,ledger,name):
    deleted=[] if inequality else list(map(int,desc['definition']))
    if not deleted and not gauge:
        pool.charge('half_unchanged_matrix_identity',32);ledger[name]=dict(reused_by_identity=True)
        return source
    deleted_set=set(deleted);gauge_set=set(gauge)
    if not continuous:pool.charge('half_binary_gauge_header_checks',8*len(gauge))
    binary_shared=not continuous and all(source.indptr[r]==source.indptr[r+1] for r in gauge)
    if binary_shared:gauge_set=set()
    edits=sorted(deleted_set|gauge_set)
    nrow=source.shape[0]-len(deleted);nnz=source.nnz-(2*len(deleted) if continuous else 0)
    pool.charge('half_writer_schedule',64*len(edits)+4*len(edits)*max(1,len(edits).bit_length()))
    pool.charge('writer_row_pointer_allocation_and_write',4*(nrow+1))
    pool.charge('writer_matrix_allocation_and_publication',128)
    ptr=np.empty(nrow+1,np.int32);ptr[0]=0
    # Binary-only row-pointer deletion can retain exactly the original data/
    # index owners when no binary coefficient is on any actual gauged row.
    if binary_shared:data,indices=source.data,source.indices
    else:
        pool.charge('native_coefficient_and_index_transfers',2*int(nnz))
        data=np.empty(nnz,np.float64);indices=np.empty(nnz,np.int32)
    aliases={int(t['column']):(int(t['parent']),int(t['sign'])) for t in desc}
    dest=outrow=cursor=0
    def copy_run(first,stop):
        nonlocal dest,outrow
        if first==stop:return
        a,b=map(int,(source.indptr[first],source.indptr[stop]));n=b-a
        np.add(source.indptr[first+1:stop+1],dest-a,out=ptr[outrow+1:outrow+stop-first+1])
        if not binary_shared:
            indices[dest:dest+n]=source.indices[a:b];data[dest:dest+n]=source.data[a:b]
        dest+=n;outrow+=stop-first
    for row in edits:
        copy_run(cursor,row);cursor=row+1
        if row in deleted_set:continue
        a,b=map(int,source.indptr[row:row+2]);n=b-a
        if continuous:
            pool.charge('half_gauged_linear_merge_and_scale',8*n)
            replacement=[];retained=[]
            for c,v in zip(source.indices[a:b],source.data[a:b]):
                col=int(c);value=float(v)
                if col in aliases:
                    parent,sign=aliases[col];replacement.append((parent,sign*value))
                else:retained.append((col,2*value))
            k=len(replacement);pool.charge('half_gauged_parent_sort',12*k*max(1,k.bit_length()))
            replacement.sort();i=j=0
            while i<len(retained) or j<len(replacement):
                if j==len(replacement) or (i<len(retained) and retained[i][0]<replacement[j][0]):
                    col,value=retained[i];i+=1
                else:col,value=replacement[j];j+=1
                indices[dest]=col;data[dest]=value;dest+=1
        elif not binary_shared:
            pool.charge('half_gauged_binary_scale',n)
            indices[dest:dest+n]=source.indices[a:b];np.ldexp(source.data[a:b],1,out=data[dest:dest+n]);dest+=n
        else:dest+=n
        outrow+=1;ptr[outrow]=dest
    copy_run(cursor,source.shape[0])
    if (dest,outrow)!=(nnz,nrow):raise ValueError('half writer did not fill every output exactly once')
    result=sp.csr_matrix((data,indices,ptr),shape=(nrow,source.shape[1]),copy=False)
    # The enclosing source proof supplies canonical zero-free rows. Global
    # parent isolation plus distinct sorted destinations proves these flags;
    # do not hide another complete coefficient scan in a property accessor.
    result.has_sorted_indices=True;result.has_canonical_format=True;result._act_hz_zero_free=True
    if ((nnz and (not np.shares_memory(result.data,data) or not np.shares_memory(result.indices,indices)))
            or not np.shares_memory(result.indptr,ptr)):raise ValueError('hidden CSR constructor copy')
    ledger[name]=dict(reused_by_identity=False,binary_data_indices_shared=binary_shared,
        original_nnz=int(source.nnz),new_nnz=int(nnz),new_row_count=nrow)
    return result


def _rhs(source,desc,gauge,*,inequality,pool):
    deleted=set() if inequality else set(map(int,desc['definition']))
    if not deleted and not gauge:
        pool.charge('half_unchanged_RHS_identity',16);return source
    size=len(source)-len(deleted);pool.charge('native_RHS_transfers',size)
    pool.charge('half_RHS_schedule_and_scale',32*(len(deleted)+len(gauge)))
    out=np.empty(size,np.float64);dest=cursor=0
    for row in sorted(deleted|set(gauge)):
        n=row-cursor;out[dest:dest+n]=source[cursor:row];dest+=n;cursor=row+1
        if row not in deleted:out[dest]=math.ldexp(float(source[row]),1);dest+=1
    out[dest:]=source[cursor:]
    return out


def materialize(hz,final,half,rows,*,pool,enabled=False):
    if not enabled:return None
    pool.charge('half_writer_complete_geometry',512)
    if type(hz) is not SparseHZono or type(final) is not SparseHZono or not hz.exact or hz.frame_id!=final.frame_id:
        raise ValueError('actual exact post/final HZ required')
    for name in ('Gc','Gb','Ac','Ab','Auc','Aub'):
        m=getattr(hz,name)
        if (not sp.isspmatrix_csr(m) or m.dtype!=np.float64 or m.indices.dtype!=np.int32
                or m.indptr.dtype!=np.int32 or not m.has_canonical_format):raise ValueError('canonical f64/i32 CSR required')
    if any(getattr(final,n) is not getattr(hz,n) for n in ('Ac','Ab','Auc','Aub')):
        raise ValueError('post/final predicates must be shared')
    if sum(int(getattr(final,n).nnz) for n in ('Gc','Gb','Ac','Ab','Auc','Aub'))>64_000_000:
        raise MemoryError('unchanged complete entry cap')
    pool.charge('half_complete_output_liveness',4*int(hz.Gc.nnz+final.Gc.nnz))
    if set(map(int,final.Gc.indices)) & set(map(int,half['column'])):raise ValueError('final output-live half alias')
    desc,runs,report=describe(hz,half,rows,pool=pool);eq,le=schedules(runs,pool=pool)
    ledger={};mats={}
    for name in ('Ac','Ab','Auc','Aub'):
        kind=name.startswith('Au');mats[name]=_matrix(getattr(hz,name),desc,le if kind else eq,
            continuous=name in ('Ac','Auc'),inequality=kind,pool=pool,ledger=ledger,name=name)
    b=_rhs(hz.b,desc,eq,inequality=False,pool=pool);ub=_rhs(hz.ub,desc,le,inequality=True,pool=pool)
    def output(value):
        return SparseHZono(value.c,value.Gc,value.Gb,mats['Ac'],mats['Ab'],b,mats['Auc'],mats['Aub'],ub,
            frame_id=value.frame_id,exact=True)
    post=output(hz);end=output(final)
    if sum(getattr(hz,n).nnz-getattr(post,n).nnz for n in mats)!=2*len(desc):raise ValueError('strict predicate nnz reduction failed')
    report.update(matrices=ledger,actual_predicate_nnz_delta=-2*len(desc),new_HZ_constructed=True,
        source_inverse_not_yet_checked=True,complete_live_runtime_proof=False,formal_gain=0)
    return post,end,desc,runs,report
