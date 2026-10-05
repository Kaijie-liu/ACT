"""Frozen C99 complete source-bound/native/proof/restore qualification."""
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import faulthandler
import hashlib
import json
import os
import pickle
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout,fingerprint
from experiments.neural_hz_20260831.c65_physical_archive_v1 import metadata
from experiments.neural_hz_20260831.c91_physical_archive_v1 import check,fingerprint as source_fingerprint
from experiments.neural_hz_20260831.c70_native_proof_v1 import digest,entries
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c94_raw_mask_plan_v1 import raw_plan
from experiments.neural_hz_20260831.c98_circuit_stream_v1 import tile_maps
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c99_unique_row_bound_v1 import minimum_final_nnz
from experiments.neural_hz_20260831.c99_circuit_consumer_v1 import CircuitSource,inject_phase,bind_phase
from experiments.neural_hz_20260831.c99_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c99_circuit_journal_v1 import compile_journal,CircuitJournal,reconstruct
from experiments.neural_hz_20260831.c99_native_proof_v1 import verify
from experiments.neural_hz_20260831.c99_writer_bound_v1 import writer_bound
from experiments.neural_hz_20260831.c73_outer_query_v1 import GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build as build_overlay
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c32_boundary_budget_v1 import WriterPool,remaining_pool,finish_local
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c99_circuit_native_20260913_v1'
SOURCE_ID='a6e5ac15c643e73b02d6ab5523c242b0a894f1c2986836d47c061c5db1f310bf'
FILES={
 'source':('results/c98_stream_circuit_20260913_v2/actual/physical_hz.pickle','a96e0801046ed046c4768899d13a64cc961e138612638b51c038abb78f1aaa5a'),
 'source_result':('results/c98_stream_circuit_20260913_v2/actual/result.json','c08ecfb4f089e4ac4394668f98a9301f6abb0bfca60f6d5d455b993b836166af'),
 'source_bound':('results/c98_stream_circuit_20260913_v1/preflight/result.json','62367ea903f7244b0c7ed0aee0f49f5434f16c6cd8f05f60721b181b3e53ff31'),
 'C9':('results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle','616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'),
 'phase':('results/c73_outer_query_20260913_v1/phase/packet.pickle','a0da43c9fd4cf4472cf8bf5bc7297dace9a5fa3175e22b7f9191ff7f5f16ad39'),
 'phase_result':('results/c73_outer_query_20260913_v1/phase/result.json','e9816b60083e49d1409c82991976afdc5b779d872061c5fd9b16c23cbfce571b'),
 'phase_exit':('results/c73_outer_query_20260913_v1/exit.json','29af04e34bd410fa050b0c0547b3efebf83e32bf3a9d6eaa5b43cb446815a1c8')}


def read(key,pool):
    name,sha=FILES[key];path=EXP/name
    if _sha256(path)!=sha:raise ValueError('complete immutable input changed: '+key)
    if path.suffix=='.pickle':
        with path.open('rb') as stream:return load(stream,expected_sha256=sha,pool=pool,enabled=True)[0]
    return json.loads(path.read_text())


def source_input(pool):
    saved=read('source',pool);layout=numeric_layout(saved,pool)
    pool.charge('c99_complete_source_proof_and_preservation',2*int(layout.resident_entries)+2048)
    proof=check(saved)
    if (proof['identity']!=SOURCE_ID or not proof['proof']['fresh_original_expression_generator_executed']
        or not proof['proof']['every_fresh_native_literal_matches_original_theorem']):
        raise ValueError('complete qualified C98 source theorem required')
    return saved,CircuitSource(saved['state']),proof


def preflight(pool,result,emit):
    saved,c,proof=source_input(pool);original=read('C9',pool);bound_record=read('source_bound',pool)
    source_result=read('source_result',pool);inputs=dict(source=saved,original=original)
    layout=numeric_layout(inputs,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete bound input union exceeds entries')
    pool.charge('c99_complete_bound_input_preservation',2*int(layout.resident_entries)+2048)
    before=fingerprint(inputs,layout,pool,already_paid=True)[0]
    branch=BranchPool(pool)
    records,plan,masks=raw_plan(original['definition_graph'],existing_aux=len(c.def_rows),
        existing_work=c.report['actual_radix_work'],pool=branch,enabled=True)
    old=bound_record['data']['bound']
    if not bound_record['completed'] or not source_result['completed'] or plan!=old['plan']:
        raise ValueError('complete original raw structural plan differs')
    lower=0;rows=[];prepared=set();aux=c.state['auxiliary_records'];routes=c.state['output_routes']
    for position,block in zip(plan['selected_positions'],c.state['block_records'],strict=True):
        item=records[position];node=original['definition_graph'][item['node']]
        ma=masks[item['node']];matches=np.flatnonzero(np.all(ma['positions']==[item['y'],item['x']],axis=1))
        if len(matches)!=1:raise ValueError('original coordinate tile not unique')
        t=int(matches[0]);b=minimum_final_nnz(ma['input_masks'][t],ma['output_masks'][t],pool=branch,enabled=True)
        ids,_,outs,_=tile_maps(node,original['definition_graph'][node['parents'][0]],item['y'],item['x'],pool=branch)
        parents=ids[ids>=0];pivots=outs[outs>=0]
        branch.charge('c99_complete_unique_original_slot_postcondition',32+8*len(parents)*max(1,len(parents).bit_length()))
        if len(np.unique(parents))!=len(parents) or np.intersect1d(parents,pivots).size:
            raise ValueError('guaranteed row lower bound requires distinct original slots')
        if item['node'] not in prepared:
            w=node['op']._kernel;branch.charge('c92_fresh_selected_kernel_precision',4*int(w.size))
            wf=w.astype(np.float32)
            if not np.array_equal(wf.astype(np.float64),w):raise ValueError('original dense filter not exact binary32')
            checked,ready=prepare_words(wf,pool=branch,enabled=True)
            if not checked['original_dense'] or ready is None or not ready.dense:raise ValueError('complete dense transform not proved')
            prepared.add(item['node'])
        _,first,count,begin,end,physical,uid,_=map(int,block)
        index=first-c.state['old_source_n_cont'];selected=aux[index:index+count]
        selected_rows=np.r_[selected[:,0],routes[begin:end,0]]
        branch.charge('c99_complete_actual_row_length_check',16*len(selected_rows))
        sizes=c.hz.Ac.indptr[selected_rows+1]-c.hz.Ac.indptr[selected_rows]
        if (len(selected)!=count or int(sizes[:count].sum())!=b['auxiliary_final_nnz']
            or int(sizes.sum())<b['guaranteed_final_nnz']):
            raise ValueError('complete actual rows violate guaranteed coefficient count')
        lower+=b['guaranteed_final_nnz'];rows.append(dict(node=item['node'],y=item['y'],x=item['x'],
            guarantee=b,actual_coefficients=int(sizes.sum())))
    credit=8*lower;whole=old['whole_work_upper']-credit;br=old['branch_work_upper']-credit
    if (c.report['total_work_upper']>whole or c.report['largest_branch_work_upper']>br
        or source_fingerprint(c.state)!=SOURCE_ID or fingerprint(inputs,layout,pool,already_paid=True)[0]!=before):
        raise ValueError('new bound or complete original source preservation failed')
    report=dict(schema='c99_unchanged_source_tighter_row_bound_v1',source_identity=SOURCE_ID,
        original_source_bound_sha256=FILES['source_bound'][1],whole_work_upper=whole,branch_work_upper=br,
        guaranteed_final_coefficients=lower,unchanged_row_work_upper_reduction=credit,
        complete_selected_rows=rows,original_bound=old,
        complete_input_entries=layout.resident_entries,complete_inputs_unchanged=True,
        unchanged_source_generator_and_tariffs=True,native_glue_not_yet_admitted=True,
        source_work_caps_fit=whole<=256_000_000 and br<=200_000_000)
    emit('complete_structural_source_bound',dict(whole=whole,branch=br,credit=credit))
    result['bound_branch_work']=branch.used
    return dict(bound=report)


def component(pool,result,emit):
    binding=json.loads((RUN/'component/input_binding.json').read_text())
    if _sha256(RUN/'preflight/result.json')!=binding['preflight_result_sha256']:raise ValueError('complete new bound changed')
    previous=json.loads((RUN/'preflight/result.json').read_text())
    if not previous['completed']:raise ValueError('complete new source bound required')
    bound=previous['data']['bound'];saved,c,source_proof=source_input(pool)
    packet=read('phase',pool);phase_result=read('phase_result',pool);phase_exit=read('phase_exit',pool)
    packet_layout=numeric_layout(packet,pool)
    pool.charge('c99_complete_original_phase_authentication',2*int(packet_layout.resident_entries)+2048)
    original=digest(packet)
    if (not phase_result['completed'] or not phase_exit['all_declared_stages_passed']
        or original!=phase_result['data']['packet_identity'] or phase_result['data']['packet_sha256']!=FILES['phase'][1]
        or phase_result['data']['recovery_proof']['full_original_native_hash_still_required']):
        raise ValueError('complete original phase theorem required')
    coupled=CoupledPool(bound['whole_work_upper'],bound['branch_work_upper']);writer=None
    try:
        local=remaining_pool(coupled)
        injected=inject_phase(c,packet,pool=local,enabled=True)
        view=bind_phase(c,packet,injected,pool=local,enabled=True)
        finish_local(coupled,local,'complete_source_phase_coordinate_correspondence')
        local=remaining_pool(coupled);first=packet['first_uid']
        overlay,event_report=build_overlay(c.owners,[(view.eq_c,first),(view.le_c,first+len(view.eq_rhs))],
            old_n_cont=c.old_n_cont,old_uid_ceiling=first,pool=local,enabled=True)
        finish_local(coupled,local,'complete_actual_append_overlay')
        local=remaining_pool(coupled);plans,discovery=discover_append(c,view,overlay,pool=local,enabled=True)
        finish_local(coupled,local,'all_actual_circuit_consumer_discovery')
        result.update(discovery=discovery,event_report=event_report);emit('complete_actual_population',discovery)
        local=remaining_pool(coupled);journal=compile_journal(c,plans,pool=local,enabled=True)
        bill=writer_bound(view,plans,pool=local,enabled=True)
        finish_local(coupled,local,'complete_circuit_journal_and_writer_bound')
        complete=dict(source_whole=coupled.whole_base,source_branch=coupled.branch_base,
            used_pre_writer=coupled.used,writer=bill,whole=coupled.whole_base+coupled.used+bill['incremental_upper'],
            branch=coupled.branch_base+coupled.used+bill['incremental_upper'],runtime_value_views_not_included=True)
        result['complete_native_bound']=complete;emit('complete_native_bound_before_allocation',complete)
        if complete['whole']>256_000_000 or complete['branch']>200_000_000:
            raise MemoryError('complete source/phase/discovery/journal/native bound exceeds unchanged caps')
        writer=WriterPool(coupled,payload_cap=bill['native_payload_upper']);start=coupled.used
        new,writer_report=splice_append(view,plans,pool=writer,enabled=True)
        if coupled.used-start!=bill['incremental_upper'] or writer.native.used!=bill['native_payload_upper']:
            raise ValueError('actual complete native writer differs from upfront work inventory')
        result['native_constructed']=True;result['writer']=writer_report
        branch=BranchPool(pool);semantic=verify(c,view,overlay,plans,new,journal,pool=branch)
        result.update(semantic_proof=semantic,semantic_branch_work=branch.used)
        emit('complete_native_rows_UIDs_MAIN_and_circuit_owners',semantic)
        if source_fingerprint(c.state)!=SOURCE_ID or digest(packet)!=original:raise ValueError('complete original source/phase mutated')
        pool.charge('c99_complete_native_fingerprint',entries(new)+1024)
        hz_sha=source_digest(new)
        j=dict(local=vars(journal.local),circuit_tails=journal.circuit_tails,schema=journal.schema)
        pool.charge('c99_complete_journal_fingerprint',sum(a.size for a in j['local'].values() if type(a) is np.ndarray)+len(journal.circuit_tails)+256)
        proof=dict(schema='c99_complete_offline_native_proof_v1',source_identity=SOURCE_ID,
            source_archive_sha256=FILES['source'][1],original_packet_identity=original,
            injected_packet_identity=digest(injected),native_sha256=hz_sha,journal_identity=digest(j),
            plans=[asdict(p) for p in plans],semantic=semantic,full_LIVE_admission=False,formal_gain=0)
        raw=json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        artifact=dict(schema='c99_complete_offline_native_archive_v1',source=saved,original_packet=packet,
            injected_packet=injected,hz=new,journal=j,events=overlay.events,proof_bytes=raw,
            proof_sha256=hashlib.sha256(raw).hexdigest(),full_LIVE_admission=False,formal_gain=0)
        full=numeric_layout(artifact,pool);meta=metadata(artifact,pool=pool)
        if full.resident_entries>64_000_000:raise MemoryError('complete source/native/proof union exceeds entries')
        pool.charge('c99_complete_native_archive',int(full.resident_entries)+1024)
        with (RUN/'component/native.pickle').open('xb') as stream:
            pickle.dump(artifact,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
        return dict(archive_sha256=_sha256(RUN/'component/native.pickle'),archive_bytes=(RUN/'component/native.pickle').stat().st_size,
            proof_sha256=artifact['proof_sha256'],native_sha256=hz_sha,complete_component=asdict(full),metadata=meta,
            complete_original_inputs_unchanged=True,all_circuit_inverse_records_retained=True,
            fresh_runtime_or_full_LIVE_admission=False,formal_gain=0)
    finally:
        result.update(coupled_increment=coupled.used,coupled_parts=coupled.parts,
            coupled_whole=coupled.whole_base+coupled.used,coupled_branch=coupled.branch_base+coupled.used,
            paid_native_payload_work=writer.native.used if writer is not None else 0)


def restore(pool,result,emit):
    binding=json.loads((RUN/'restore/input_binding.json').read_text())
    if _sha256(RUN/'component/result.json')!=binding['component_result_sha256']:raise ValueError('native result changed')
    done=json.loads((RUN/'component/result.json').read_text())
    if not done['completed']:raise ValueError('complete native component proof required')
    with (RUN/'component/native.pickle').open('rb') as stream:
        saved,decoder=load(stream,expected_sha256=done['data']['archive_sha256'],pool=pool,enabled=True)
    layout=numeric_layout(saved,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete restored component entries')
    pool.charge('c99_complete_restored_native_authentication',int(layout.resident_entries)+1024)
    source=check(saved['source']);proof=json.loads(saved['proof_bytes'])
    if (source['identity']!=SOURCE_ID or saved['schema']!='c99_complete_offline_native_archive_v1'
        or saved['full_LIVE_admission'] or source_digest(saved['hz'])!=proof['native_sha256']
        or digest(saved['original_packet'])!=proof['original_packet_identity']
        or digest(saved['injected_packet'])!=proof['injected_packet_identity']
        or digest(saved['journal'])!=proof['journal_identity']
        or hashlib.sha256(saved['proof_bytes']).hexdigest()!=done['data']['proof_sha256']):
        raise ValueError('complete restored source/native/phase/journal proof mismatch')
    c=CircuitSource(saved['source']['state']);j=saved['journal']
    journal=CircuitJournal(GuardedLocalSpliceJournal(**j['local']),c.state,j['circuit_tails'],j['schema'])
    plans=[Plan(**{**p,'tail':tuple(p['tail'])}) for p in proof['plans']]
    recovered=reconstruct(c,saved['hz'],journal,plans,[F(0)]*saved['hz'].n_cont,pool=pool,enabled=True)
    emit('all_unit_local_circuit_inverse_equations',recovered['proof'])
    return dict(complete_restored_entries=layout.resident_entries,complete_restored_bytes=layout.resident_bytes,
        inverse=recovered['proof'],decoder=decoder,zero_point_not_a_feasible_witness=True,
        fresh_runtime_or_full_LIVE_admission=False,formal_gain=0)


def main():
    stage=sys.argv[1];fn={'preflight':preflight,'component':component,'restore':restore}[stage]
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);result=dict(completed=False,formal_gain=0,stage=stage);start=time.monotonic()
    with (RUN/stage/'events.jsonl').open('x') as log,(RUN/stage/'fatal.log').open('x') as fatal:
        def emit(name,value):
            event=dict(event=name,worker_wall_s=time.monotonic()-start,**value)
            log.write(json.dumps(event,allow_nan=False)+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            freeze=json.loads((RUN/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen candidate changed')
            data,_=measured(lambda:fn(pool,result,emit),observe=lambda s:result.update(measurement=s))
            result.update(data=data,completed=True)
        except Exception as exc:result.update(failure=dict(type=type(exc).__name__,reason=str(exc)))
        finally:
            faulthandler.disable();result.update(wall_s=time.monotonic()-start,diagnostic_work=pool.used,diagnostic_parts=pool.parts)
            _atomic_exclusive_json(RUN/stage/'result.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
