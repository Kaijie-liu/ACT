"""Independent complete original-HZ image hashing from compact new predicates.

Expected original factor degrees MUST come from a complete bound ownership
proof. Matching every original byte plus all restored selected occurrences
proves that no selected factor remains in an untouched new predicate row.
Only changed rows and original EQ row pointers are reconstructed transiently.
"""
import hashlib
import json
import math
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c40_compact_half_gauge_v1 import DESC,schedules,unpack_run


def _update(hashes,shape,dtype,parts):
    for h in hashes:h.update(str(shape).encode());h.update(str(np.dtype(dtype)).encode())
    count=0
    for part in parts:
        if part.dtype!=np.dtype(dtype) or not part.flags.c_contiguous:raise ValueError('wrong inverse image scalar layout')
        count+=part.size
        for h in hashes:h.update(memoryview(part).cast('B'))
    if count!=math.prod(shape):raise ValueError('incomplete inverse array image')


def _literal(hashes,value):
    arrays=(value.data,value.indices,value.indptr) if sp.issparse(value) else (value,)
    for a in arrays:_update(hashes,a.shape,a.dtype,[a])


def _restore_row(matrix,row,desc,*,continuous,pool,counts,lookup):
    a,b=map(int,matrix.indptr[row:row+2]);cc=matrix.indices[a:b];cv=matrix.data[a:b]
    pool.charge('half_inverse_row_decode_and_merge',8*len(cc)+64)
    if not continuous:return cc,np.ldexp(cv,-1)
    columns,parents=lookup
    mapped=[];retained=[]
    for col,value in zip(cc,cv):
        col=int(col);value=float(value)
        if col in columns:raise ValueError('selected child remains in actual new gauge row')
        if col in parents:
            i,z,sign=parents[col];counts[i]+=1;mapped.append((z,sign*value))
        else:retained.append((col,math.ldexp(value,-1)))
    if not mapped:raise ValueError('spurious gauge row without a half alias')
    k=len(mapped);pool.charge('half_inverse_child_sort',12*k*max(1,k.bit_length()))
    mapped.sort();outc=np.empty(len(cc),np.int32);outv=np.empty(len(cc),np.float64);i=j=at=0
    while i<len(retained) or j<len(mapped):
        if j==len(mapped) or (i<len(retained) and retained[i][0]<mapped[j][0]):
            col,value=retained[i];i+=1
        else:col,value=mapped[j];j+=1
        outc[at]=col;outv[at]=value;at+=1
    if np.any(outc[1:]<=outc[:-1]) or not np.isfinite(outv).all():raise ValueError('noncanonical/nonfinite inverse row')
    return outc,outv


def _matrix_image(matrix,desc,gauge,*,continuous,inequality,pool,counts,lookup,ranges):
    deleted={} if inequality else {int(t['definition']):t for t in desc}
    if not deleted and not gauge:return [matrix.data],[matrix.indices],[matrix.indptr],len(matrix.indptr)
    edits=sorted(set(deleted)|set(gauge));old_rows=matrix.shape[0]+len(deleted)
    if not continuous:
        pool.charge('half_inverse_binary_zero_interval_proof',32*(len(ranges)+len(deleted))+64)
        ds=np.asarray(sorted(deleted),np.int32);empty=True
        for kind,start,length in ranges:
            if kind!=inequality:continue
            a,b=map(int,np.searchsorted(ds,[start,start+length]))
            if a!=b:raise ValueError('gauge interval intersects a deleted definition')
            lo=start-a;hi=lo+length
            if not 0<=lo<=hi<matrix.indptr.size:raise ValueError('binary inverse interval outside row frame')
            if matrix.indptr[lo]!=matrix.indptr[hi]:empty=False;break
        if empty:
            pool.charge('half_inverse_binary_pointer_segments',64*len(deleted))
            parts=[];cursor=0
            for i,d in enumerate(sorted(deleted)):
                at=d-i
                if not 0<=at<len(matrix.indptr):raise ValueError('binary inverse deletion outside row frame')
                parts.append(matrix.indptr[cursor:at+1]);parts.append(np.array([matrix.indptr[at]],np.int32));cursor=at+1
            parts.append(matrix.indptr[cursor:])
            return [matrix.data],[matrix.indices],parts,old_rows+1
    pool.charge('half_inverse_sparse_schedule_and_pointers',64*len(edits)+4*(old_rows+1))
    ptr=np.empty(old_rows+1,np.int32);ptr[0]=0;data=[];indices=[]
    old_cursor=new_cursor=removed=dest=0
    def unchanged(stop):
        nonlocal old_cursor,new_cursor,dest
        n=stop-old_cursor
        if not n:return
        a,b=map(int,(matrix.indptr[new_cursor],matrix.indptr[new_cursor+n]))
        data.append(matrix.data[a:b]);indices.append(matrix.indices[a:b])
        np.add(matrix.indptr[new_cursor+1:new_cursor+n+1],dest-a,out=ptr[old_cursor+1:stop+1])
        old_cursor=stop;new_cursor+=n;dest+=b-a
    for old_row in edits:
        if not 0<=old_row<old_rows:raise ValueError('inverse row outside original frame')
        unchanged(old_row)
        if old_row in deleted:
            t=deleted[old_row]
            if continuous:
                p=math.ldexp(1.,int(t['power']));cc=np.array([t['parent'],t['column']],np.int32)
                cv=np.array([-int(t['sign'])*p/2,p],np.float64)
            else:cc=np.empty(0,np.int32);cv=np.empty(0,np.float64)
            removed+=1
        else:
            if new_cursor!=old_row-removed:raise ValueError('inverse deletion rank mismatch')
            cc,cv=_restore_row(matrix,new_cursor,desc,continuous=continuous,pool=pool,counts=counts,lookup=lookup)
            new_cursor+=1
        data.append(cv);indices.append(cc);dest+=len(cv);ptr[old_row+1]=dest;old_cursor=old_row+1
    unchanged(old_rows)
    if new_cursor!=matrix.shape[0]:raise ValueError('incomplete original matrix image')
    return data,indices,[ptr],old_rows+1


def _rhs_image(values,desc,gauge,*,inequality,pool):
    deleted={} if inequality else {int(t['definition']):t for t in desc}
    if not deleted and not gauge:return [values],len(values)
    edits=sorted(set(deleted)|set(gauge));old_rows=len(values)+len(deleted)
    pool.charge('half_inverse_RHS_sparse_image',32*len(edits))
    parts=[];old_cursor=new_cursor=0
    for row in edits:
        n=row-old_cursor
        if n:parts.append(values[new_cursor:new_cursor+n]);new_cursor+=n
        if row in deleted:value=-0. if deleted[row]['negative_zero'] else 0.
        else:value=math.ldexp(float(values[new_cursor]),-1);new_cursor+=1
        parts.append(np.array([value],np.float64));old_cursor=row+1
    parts.append(values[new_cursor:]);return parts,old_rows


def inverse_hashes(post,final,desc,runs,expected_degrees,*,pool,enabled=False):
    if not enabled:return None
    pool.charge('half_inverse_complete_descriptor_checks',256*len(desc)+512)
    if (type(desc) is not np.ndarray or desc.dtype!=DESC or desc.ndim!=1
            or len(expected_degrees)!=len(desc) or not len(desc)
            or np.any(desc['definition'][1:]<=desc['definition'][:-1])
            or np.any(desc['parent']<0) or np.any(desc['parent']>=desc['column'])
            or np.any(desc['column']>=post.n_cont) or np.any((desc['power']<-20)|(desc['power']>40))
            or np.any((desc['sign']!=1)&(desc['sign']!=-1))
            or len(set(map(int,desc['column'])))!=len(desc) or len(set(map(int,desc['parent'])))!=len(desc)
            or set(map(int,desc['column']))&set(map(int,desc['parent']))):
        raise ValueError('invalid complete compact inverse descriptor')
    if any(getattr(post,n) is not getattr(final,n) for n in ('Ac','Ab','Auc','Aub')):
        raise ValueError('inverse post/final predicates are not shared')
    hashes=[]
    for hz in (post,final):
        h=hashlib.sha256(json.dumps([hz.frame_id,hz.n_out,hz.n_cont,hz.n_bin]).encode())
        for name in ('c','Gc','Gb'):_literal([h],getattr(hz,name))
        hashes.append(h)
    eq,le=schedules(runs,pool=pool);counts=np.ones(len(desc),np.int64)
    ranges=[unpack_run(raw) for raw in runs]
    pool.charge('half_inverse_sparse_lookup_metadata',32*len(desc))
    lookup=({int(t['column']) for t in desc},{int(t['parent']):(i,int(t['column']),int(t['sign'])) for i,t in enumerate(desc)})
    for name in ('Ac','Ab','b','Auc','Aub','ub'):
        kind=name in ('Auc','Aub','ub');gauge=le if kind else eq
        if name in ('b','ub'):
            parts,n=_rhs_image(getattr(post,name),desc,gauge,inequality=kind,pool=pool)
            _update(hashes,(n,),np.float64,parts)
        else:
            value=getattr(post,name)
            data,indices,ptr_parts,nptr=_matrix_image(value,desc,gauge,continuous=name in ('Ac','Auc'),
                inequality=kind,pool=pool,counts=counts,lookup=lookup,ranges=ranges)
            n=sum(part.size for part in data)
            _update(hashes,(n,),np.float64,data);_update(hashes,(n,),np.int32,indices)
            _update(hashes,(nptr,),np.int32,ptr_parts)
    if not np.array_equal(counts,np.asarray(expected_degrees,np.int64)):
        raise ValueError('restored child occurrences differ from complete bound source ownership')
    return dict(complete_original_post_sha256=hashes[0].hexdigest(),complete_original_final_sha256=hashes[1].hexdigest(),
        all_original_child_occurrences_reconstructed=True,complete_old_HZ_materialized=False,
        full_original_image_byte_hashes=True,inverse_arithmetic_charged=True,formal_gain=0)
