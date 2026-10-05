"""Derive a complete mask-first plan; exact construction only after selection."""
import faulthandler
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c92_topology_first_v1 import upper_bill,select
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c88_inline_tile_v1 import prepare,construct
from experiments.neural_hz_20260831.c89_quotient_budget_v1 import bill
from experiments.neural_hz_20260831.c63_birth_blocks_v1 import route_rows
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c92_topology_first_20260913_v1'
FILES={
 'C9':('results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle','616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'),
 'C69':('results/c69_prepared_finite_20260913_v1/actual/physical_hz.pickle','acb6e5560503d42aa7b62fa4785f179f857471a9873b6d11189da88deea2db79'),
 'C89':('results/c89_quotient_budget_20260913_v1/result.json','f83db6d09f255a1bdc9632bcb7e905d78eec9dae67bc3ae924217cd90aefabc1'),
 'C90':('results/c90_actual_circuit_proof_20260913_v1/result.json','40d3a498200cc5748733dfa081040a048edae8eae9c3442a8477cb0ed3f67e84'),
 'native':('results/c90_actual_circuit_proof_20260913_v1/all_rebased_selected_native_packets.npz','f1e760a6bab4e722ce8cc8f1c76c823cddafe512b1aa836a126d53af9661c351')}


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


def complete(pool,branch,held,emit):
    inputs={}
    for key,(name,sha) in FILES.items():
        path=EXP/name
        if _sha256(path)!=sha:raise ValueError('complete source/reference hash changed: '+key)
        if key in ('C9','C69'):
            with path.open('rb') as stream:inputs[key],_=load(stream,expected_sha256=sha,pool=pool,enabled=True)
        elif key=='native':
            with np.load(path,allow_pickle=False) as ar:inputs[key]={n:ar[n] for n in ar.files}
        else:inputs[key]=json.loads(path.read_text())
    held['complete_inputs']=inputs;saved=inputs['C9'];source=inputs['C69'];fields=source['fields']
    layout=numeric_layout(inputs,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete declared input entry cap')
    pool.charge('c92_complete_input_before_after_fingerprints',2*int(layout.resident_entries)+2048)
    before,_=fingerprint(inputs,layout,pool,already_paid=True)
    src_layout=numeric_layout(source,pool);pool.charge('c92_current_source_proof_authentication',int(src_layout.resident_entries)+1024)
    proof=check_restored(source)
    if (proof['source_checkpoint_sha256']!=FILES['C9'][1]
        or proof['proof']['complete_source_boundary_sha256']!='2174cc144bcfe499ab00e97f67eae2b72d8a3c3d6f994780f63e349d00f36770'
        or fields['hz'].n_cont!=saved['hz'].n_cont):raise ValueError('original/current-source graph and quotient identity differ')
    _,binding=route_rows(saved,np.empty(0,np.int64),pool=branch,enabled=True)
    nodes=saved['definition_graph'];h=fields['hz'];records=[];maps={}
    held['all_fresh_tile_maps']=maps
    planner_start=branch.used
    branch.charge('c92_complete_node_headers',64*len(nodes)+8*sum(n['width'] for n in nodes))
    try:
        for index,node in enumerate(nodes):
            if node['kind']!='op':continue
            op=node['op']
            if (type(op) is not ImplicitConv2DOp or op._stride!=(1,1) or op._dilation!=(1,1)
                or op._groups!=1 or op._kernel.shape[-2:]!=(3,3)
                or op.input_shape[0]!=1 or op.output_shape[0]!=1 or not node['needed'].any()):continue
            parent=nodes[node['parents'][0]];_,k,oh,ow=op.output_shape
            for y in range(0,oh,2):
                for x in range(0,ow,2):
                    values=tile_maps(node,parent,y,x,pool=branch);ids,exps,outs,powers=values
                    parents,pivots=ids[ids>=0],outs[outs>=0]
                    if (np.any(parents<fields['old_n_cont']) or np.any(pivots<fields['old_n_cont'])
                        or np.any(parents>=fields['logical_n_cont']) or np.any(pivots>=fields['logical_n_cont'])):
                        raise ValueError('fresh source slot outside complete MAIN frame')
                    pr=fields['eq_roots'][fields['old_n_eq']+parents-fields['old_n_cont']]
                    rr=fields['eq_roots'][fields['old_n_eq']+pivots-fields['old_n_cont']]
                    item=dict(node=index,y=y,x=x)
                    if np.any(pr<0) or np.any(rr<0):
                        item['cost']=dict(topology_qualified=False,reason='source scalar aliases retained; dense direct count not proved')
                    elif np.any(h.b[rr]!=0) or np.any(h.Ab.indptr[rr+1]!=h.Ab.indptr[rr]):
                        item['cost']=dict(topology_qualified=False,reason='offset or binary original output equation')
                    else:
                        direct=int((h.Ac.indptr[rr+1]-h.Ac.indptr[rr]).sum())
                        item['cost']=upper_bill(ids,outs,direct,pool=branch,enabled=True)
                    maps[len(records)]=dict(ids=ids,exps=exps,outs=outs,powers=powers)
                    records.append(item)
            emit(dict(event='complete_topology_operator',node=index,classified_tiles=len(records),whole_work=pool.used))
        plan=select(records,existing_aux=len(fields['def_rows']),existing_work=fields['report']['actual_radix_work'],
            existing_entries=131072,pool=branch,enabled=True)
        planner_work=branch.used-planner_start
        # References become comparison inputs ONLY AFTER the fresh rule returns.
        reference=inputs['C89']['data'];old_selected=[reference['all_records'][i] for i in reference['plan']['selected_positions']]
        selected=[records[i] for i in plan['selected_positions']]
        key=lambda r:(r['node'],r['y'],r['x'])
        emit(dict(event='fresh_plan_returned_before_reference_comparison',plan=plan,
            selected=[key(r) for r in selected],planner_work=planner_work))
        if list(map(key,selected))!=list(map(key,old_selected)):
            raise ValueError('fresh sufficient rule did not recover the complete C91-proved plan in order')
        transforms={};native={};held['fresh_transforms']=transforms;held['fresh_selected_native']=native
        prepared={};offset=0;selected_proofs=[];emission_start=branch.used
        for position in plan['selected_positions']:
            item=records[position];index=item['node'];op=nodes[index]['op']
            if index not in prepared:
                branch.charge('c92_fresh_selected_kernel_precision',4*int(op._kernel.size))
                w=op._kernel.astype(np.float32)
                if not np.array_equal(w.astype(np.float64),op._kernel):raise ValueError('complete selected original kernel not binary32')
                checked,values=transform(w,pool=branch,enabled=True)
                if values is None or not checked['all_coefficients_exact_binary64']:raise ValueError('fresh exact selected transform failed')
                transforms[index]=values;prepared[index]=prepare(values,pool=branch)
                if not prepared[index].dense:raise ValueError('selected full transform not dense; entire plan rejected')
            m=maps[position]
            report,raw=construct(prepared[index],m['ids'],m['exps'],m['outs'],m['powers'],
                h.n_cont+offset,pool=branch,enabled=True)
            packet={k:raw[k] for k in ('columns','native','rhs','pivots','gauges')}
            packet.update(indptr=raw['indptr'].astype(np.int32),ab_indptr=np.zeros(report['rows']+1,np.int32))
            bound=item['cost']['bill'];actual=bill(packet,new_factors=report['new_factors'],direct_nnz=bound['direct_nnz'])
            if (report['new_factors']!=bound['new_factors'] or report['nnz']>bound['nnz_upper']
                or actual['declared_new_bytes']>bound['declared_new_bytes_upper'] or actual['entry_delta']>bound['entry_delta_upper']):
                raise ValueError('actual numerical construction exceeds conditional topology upper bound')
            prefix=f"node{index}_tile{item['y']}_{item['x']}_"
            pool.charge('c92_independent_complete_prior_native_comparison',4*sum(int(v.size) for v in packet.values())+128)
            for name,value in packet.items():
                expected=inputs['native'][prefix+name]
                if value.dtype!=expected.dtype or value.shape!=expected.shape or value.tobytes()!=expected.tobytes():
                    raise ValueError('fresh literal differs from complete independently proved C90 packet: '+name)
                native[prefix+name]=value
            offset+=report['new_factors']
            selected_proofs.append(dict(node=index,y=item['y'],x=item['x'],actual_bill=actual,
                all_actual_literals_equal_independent_C90=True,fresh_full_transform_checked=True))
            emit(dict(event='fresh_selected_exact_native',node=index,y=item['y'],x=item['x'],new_factors=report['new_factors'],whole_work=pool.used))
        if set(native)!=set(inputs['native']):raise ValueError('complete selected native field population differs')
        fresh_emission_work=branch.used-emission_start
        combined=numeric_layout(dict(inputs=inputs,maps=maps,transforms=transforms,native=native),pool)
        if combined.resident_entries>64_000_000:raise MemoryError('complete held input and candidate entry cap')
        pool.charge('c92_exclusive_fresh_native_export',sum(int(v.size) for v in native.values())+4096)
        with (RUN/'fresh_selected_native.npz').open('xb') as stream:np.savez(stream,**native)
        # A necessary additive bill, not a complete generator or free proof claim.
        fresh_work=planner_work+fresh_emission_work;prior_work=fields['report']['total_work_upper']
        proposed=prior_work+fresh_work
        return dict(all_records=records,plan=plan,selected_proofs=selected_proofs,source_binding=binding,
            classified_tiles=len(records),numeric_tiles_constructed=len(selected),fresh_operators_transformed=len(transforms),
            planner_work=planner_work,fresh_transform_and_emission_work=fresh_emission_work,
            total_new_fresh_planner_emission_work=fresh_work,existing_generation_work=prior_work,
            additive_postfold_generation_lower_bound=proposed,additive_postfold_excess=max(0,proposed-256_000_000),
            unchanged_postfold_integration_rejected=proposed>256_000_000,
            remaining_owner_UID_map_native_glue_not_included=True,
            complete_input_entries=layout.resident_entries,complete_input_bytes=layout.resident_bytes,
            complete_input_and_candidate_entries=combined.resident_entries,complete_input_and_candidate_bytes=combined.resident_bytes,
            actual_source_plan_and_prior_native_literals_match=True,fresh_original_expression_generator_executed=False,
            full_LIVE_or_native_admission=False,solver_executed=False,formal_gain=0)
    finally:
        unchanged=fingerprint(inputs,layout,pool,already_paid=True)[0]==before
        emit(dict(event='complete_inputs_preserved',unchanged=unchanged))
        if not unchanged or any(_sha256(EXP/n)!=s for n,s in FILES.values()):raise ValueError('original source/reference changed')


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));started=time.monotonic()
    pool=WorkPool(256_000_000);branch=BranchPool(pool);held={};result=dict(completed=False,formal_gain=0)
    with (RUN/'events.jsonl').open('x') as log,(RUN/'fatal.log').open('x') as fatal:
        def emit(v):log.write(json.dumps(dict(v,worker_elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            data,stats=measured(lambda:complete(pool,branch,held,emit),observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data)
        except Exception as exc:
            result['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='topology_first_failed',**result['failure']))
        finally:
            faulthandler.disable();result.update(wall_s=time.monotonic()-started,whole_work=pool.used,branch_work=branch.used,
                work_parts=pool.parts,branch_work_parts=branch.parts)
            _atomic_exclusive_json(RUN/'result.json',result);print(json.dumps({k:result[k] for k in ('completed','wall_s','whole_work','branch_work','formal_gain')}),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
