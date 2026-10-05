"""Fresh owned circuit row stream; no source/reference HZ enters construction."""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c97_prepared_owned_rows_v1 import PreparedOwnedEncoder
from experiments.neural_hz_20260831.c94_raw_mask_plan_v1 import raw_plan
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c96_word_row_v1 import construct
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, UID_LIMIT, validate_words

def tile_maps(node,parent,y,x,*,pool):
    op=node['op'];_,c,h,w=op.input_shape;_,k,oh,ow=op.output_shape;py,px=op._padding
    pool.charge('c92_complete_original_tile_maps_and_survival',1024+128*(c+k))
    ids=np.full((c,4,4),-1,np.int64);exps=np.zeros(ids.shape,np.int32)
    outs=np.full((k,2,2),-1,np.int64);powers=np.zeros(outs.shape,np.int32)
    pi=parent['slots'].reshape(c,h,w);pe=parent['exponents'].reshape(c,h,w);pm=parent['needed'].reshape(c,h,w)
    oi=node['slots'].reshape(k,oh,ow);oe=node['exponents'].reshape(k,oh,ow);om=node['needed'].reshape(k,oh,ow)
    for i in range(4):
        for j in range(4):
            sy,sx=y-py+i,x-px+j
            if 0<=sy<h and 0<=sx<w:
                active=pm[:,sy,sx];ids[active,i,j]=pi[active,sy,sx];exps[active,i,j]=pe[active,sy,sx]
    for i in range(min(2,oh-y)):
        for j in range(min(2,ow-x)):
            active=om[:,y+i,x+j];outs[active,i,j]=oi[active,y+i,x+j];powers[active,i,j]=oe[active,y+i,x+j]
    return ids,exps,outs,powers


def additional_bound(nodes, records, plan, *, main_count):
    """Conservative extra work AFTER raw_plan, no observed numerical result."""
    selected=[records[i] for i in plan['selected_positions']]
    if not selected:raise ValueError('no strictly smaller whole circuit plan')
    maps=source=numeric=literal=owner=metadata=0
    transformed=set()
    for item in selected:
        op=nodes[item['node']]['op'];c,k=op.input_shape[1],op.output_shape[1]
        b=item['cost']['bill'];nnz=b['nnz_upper'];rows=b['rows'];aux=b['new_factors']
        maps+=1024+128*(c+k)
        source+=128+24*16*c+32*4*k
        if item['node'] not in transformed:
            numeric+=302*c*k+9*int(op._kernel.size)
            transformed.add(item['node'])
        numeric+=1024+192*(c+k)+64*rows+16*nnz
        literal+=64*rows+16*nnz
        owner+=4*(b['direct_nnz']+nnz)
        metadata+=64*aux+32*b['output_rows']+64
    parts=dict(tile_maps=maps,source_postconditions=source,numerics=numeric,
        literal_headers=literal,owner_incidence=owner,inverse_records=metadata,
        stream_binding=2048,owner_range=8*(main_count+sum(r['cost']['bill']['new_factors'] for r in selected)))
    return dict(parts=parts,after_plan_upper=sum(parts.values()),
        new_factors=sum(r['cost']['bill']['new_factors'] for r in selected),
        replaced_rows=sum(r['cost']['bill']['output_rows'] for r in selected),
        direct_nnz=sum(r['cost']['bill']['direct_nnz'] for r in selected),
        nnz_upper=sum(r['cost']['bill']['nnz_upper'] for r in selected))


@dataclass
class CircuitStream:
    encoder: object
    old_nc: int
    old_neq: int
    auxiliary_records: object
    output_routes: object
    block_records: object
    packets: list
    selected: list
    summary: dict


def install(encoder,nodes,eq_roots,eq_scales,*,old_nc,old_neq,observe=None,enabled=False):
    if not enabled:return None
    if (type(encoder) is not PreparedOwnedEncoder or not encoder.heads_discarded
        or encoder.pending_uid is not None or encoder.in_auxiliary):
        raise ValueError('fresh complete C97 owned row producer required')
    pool=encoder.pool;start=pool.used;main=encoder.nc
    source_nc=main+len(encoder.def_rows);source_neq=len(encoder.eq)
    pool.charge('c98_owned_stream_binding',2048)
    plan_start=pool.used
    records,plan,masks=raw_plan(nodes,existing_aux=len(encoder.def_rows),
        existing_work=encoder.extra_work,pool=pool,enabled=True)
    planner=pool.used-plan_start;bound=additional_bound(nodes,records,plan,main_count=main-old_nc)
    if pool.used+bound['after_plan_upper']>pool.capacity:
        raise MemoryError('complete circuit owner/map/numeric bound exceeds coupled cap')
    total=bound['new_factors'];first_uid=encoder.radix_uid_base+len(encoder.def_rows)
    if first_uid+total>UID_LIMIT or source_nc+total>=2**31 or source_neq+total>=2**31:
        raise MemoryError('whole new coordinate/UID/index reserve exceeds domain')
    owners=encoder.ledger.words
    new_owners=np.zeros(total,np.int64);aux=np.empty((total,8),np.int64)
    packets=[];routes=[];blocks=[];selected=[];prepared={};seen=set();offset=0;direct=written=0
    empty_c=np.empty(0,np.int32);empty_v=np.empty(0,np.float64)
    bound_bits=np.array([-1.,1.],np.float64).view(np.int64)
    def change(columns,uid,sign):
        pool.charge('c98_actual_owner_incidence',4*len(columns))
        active=(columns>=old_nc)&(columns<main)
        np.add.at(owners,columns[active]-old_nc,sign*(RADIX+uid))
        active=columns>=source_nc
        np.add.at(new_owners,columns[active]-source_nc,sign*(RADIX+uid))
    for position in plan['selected_positions']:
        item=records[position];index=item['node'];node=nodes[index];parent=nodes[node['parents'][0]]
        ids,exps,outs,powers=tile_maps(node,parent,item['y'],item['x'],pool=pool)
        parents,pivots=ids[ids>=0],outs[outs>=0]
        pool.charge('c94_selected_actual_source_postconditions',128+24*len(parents)+32*len(pivots))
        if (np.any(parents<old_nc) or np.any(parents>=main)
            or np.any(pivots<old_nc) or np.any(pivots>=main)):
            raise ValueError('selected coordinate outside original MAIN')
        parent_roots=eq_roots[old_neq+parents-old_nc]
        physical=eq_roots[old_neq+pivots-old_nc]
        if np.any(parent_roots<0) or np.any(physical<0):
            raise ValueError('selected coordinate removed by original quotient')
        originals=[encoder.eq[int(r)] for r in physical]
        actual_direct=sum(len(r[0]) for r in originals)
        if (actual_direct!=item['cost']['bill']['direct_nnz']
            or any(len(r[2]) or r[4]!=0 for r in originals)):
            raise ValueError('selected actual original source row differs from raw bound')
        direct+=actual_direct
        if index not in prepared:
            op=node['op'];pool.charge('c92_fresh_selected_kernel_precision',4*int(op._kernel.size))
            w=op._kernel.astype(np.float32)
            if not np.array_equal(w.astype(np.float64),op._kernel):
                raise ValueError('whole original kernel is not binary32')
            checked,ready=prepare_words(w,pool=pool,enabled=True)
            if ready is None or not checked['original_dense'] or not ready.dense:
                raise ValueError('whole selected transform/density precondition unproved')
            prepared[index]=ready
        emitted,raw=construct(prepared[index],ids,exps,outs,powers,
            source_nc+offset,pool=pool,enabled=True)
        count=emitted['new_factors'];nr=emitted['rows'];b=item['cost']['bill']
        if count!=b['new_factors'] or emitted['nnz']>b['nnz_upper'] or nr!=b['rows']:
            raise ValueError('fresh full packet exceeds conditional topology upper')
        packet={k:raw[k] for k in ('columns','native','rhs','pivots','gauges')}
        packet.update(indptr=raw['indptr'].astype(np.int32),
            ab_indptr=np.zeros(nr+1,np.int32),new_factors=count)
        pool.charge('c98_complete_changed_literal_headers',64*nr+16*len(packet['native']))
        if (np.any(packet['rhs']!=0) or not np.isfinite(packet['native']).all()
            or np.any((np.abs(packet['native'])<2.**-20)|(np.abs(packet['native'])>2.**40))
            or not np.array_equal(packet['pivots'][:count],source_nc+offset+np.arange(count))):
            raise ValueError('fresh full native packet domain differs')
        begin=len(routes)
        for i,pivot in enumerate(packet['pivots']):
            pivot=int(pivot);a,z=map(int,packet['indptr'][i:i+2])
            cols,vals=packet['columns'][a:z],packet['native'][a:z]
            gauge=int(packet['gauges'][i]);p=np.flatnonzero(cols==pivot)
            if (np.any(cols<0) or np.any(cols>=source_nc+total) or np.any(np.diff(cols)<=0)
                or len(p)!=1 or vals[int(p[0])]<=0):
                raise ValueError('fresh positive canonical full-frame pivot required')
            row=(cols,vals,empty_c,empty_v,0.)
            if i<count:
                if np.any(cols[cols!=pivot]>=pivot):raise ValueError('non-topological fresh auxiliary')
                physical=source_neq+offset+i;uid=first_uid+offset+i
                if physical!=len(encoder.eq):raise ValueError('new row stream order differs')
                encoder.eq.append(row);encoder.eq_uids.append(uid)
                change(cols,uid,1)
                aux[offset+i]=(physical,uid,pivot,np.array(vals[int(p[0])],np.float64).view(np.int64).item(),
                    gauge,0,int(bound_bits[0]),int(bound_bits[1]))
            else:
                rank=old_neq+pivot-old_nc;physical=int(eq_roots[rank])
                if not old_nc<=pivot<main or not 0<=physical<source_neq or physical in seen:
                    raise ValueError('original output removed or replaced twice')
                seen.add(physical);uid=int(encoder.eq_uids[physical])
                change(encoder.eq[physical][0],uid,-1);change(cols,uid,1)
                encoder.eq[physical]=row;eq_scales[rank]=gauge
                routes.append((physical,pivot,gauge))
        pool.charge('c98_complete_inverse_owner_records',64*count+32*(nr-count)+64)
        blocks.append((source_nc,source_nc+offset,count,begin,len(routes),source_neq+offset,first_uid+offset,0))
        packets.append(packet);selected.append(dict(node=index,y=item['y'],x=item['x'],position=position))
        offset+=count;written+=len(packet['native'])
        if observe:observe('c98_fresh_circuit_installed',dict(node=index,y=item['y'],x=item['x'],
            new_factors=count,whole_work=pool.whole_base+pool.used,branch_work=pool.branch_base+pool.used))
    pool.charge('c98_complete_owner_range',8*(len(owners)+len(new_owners)))
    validate_words(owners);validate_words(new_owners);aux[:,5]=new_owners
    if offset!=total or written>=direct or pool.used-start>planner+bound['after_plan_upper']:
        raise ValueError('complete fresh stream strict reduction/work bound failed')
    summary=dict(plan=plan,additional_bound=bound,planner_work=planner,new_work=pool.used-start,
        old_source_n_cont=source_nc,old_source_n_eq=source_neq,new_factors=total,
        direct_nnz=direct,new_nnz=written,one_final_CSR=True,old_source_CSR_constructed=False)
    return CircuitStream(encoder,source_nc,source_neq,aux,np.array(routes,np.int64),
        np.array(blocks,np.int64),packets,selected,summary)


def matrices(stream,rows):
    if (type(stream) is not CircuitStream or type(stream.encoder) is not PreparedOwnedEncoder
        or (rows is not stream.encoder.eq and rows is not stream.encoder.ineq)):
        raise ValueError('complete fresh owned circuit stream required')
    encoder=stream.encoder;nc=stream.old_nc+len(stream.auxiliary_records)
    cptr=np.r_[0,np.cumsum([r[0].size for r in rows],dtype=np.int64)]
    bptr=np.r_[0,np.cumsum([r[2].size for r in rows],dtype=np.int64)]
    if max(nc,encoder.nb,len(rows),int(cptr[-1]),int(bptr[-1]))>=2**31:
        raise MemoryError('bounded final native index domain exceeded')
    def concat(i,dtype):
        return np.concatenate([r[i] for r in rows],dtype=dtype,casting='unsafe') if rows else np.empty(0,dtype)
    ac=sp.csr_matrix((concat(1,np.float64),concat(0,np.int32),cptr),shape=(len(rows),nc))
    ab=sp.csr_matrix((concat(3,np.float64),concat(2,np.int32),bptr),shape=(len(rows),encoder.nb))
    if not ac.has_canonical_format or not ab.has_canonical_format:raise ValueError('noncanonical final circuit source')
    return ac,ab,np.array([r[4] for r in rows],np.float64)
