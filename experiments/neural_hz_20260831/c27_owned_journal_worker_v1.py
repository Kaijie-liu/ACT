"""Fresh original-expression ownership transaction, NOT a new native HZ."""

import gc
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace
import weakref

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import scipy.sparse as sp
import torch
from act.back_end.hybridz_tf import tf_cnn as cnn
from experiments.neural_hz_20260831.c27_reversible_lineage_v1 import generate,compile_owned
from experiments.neural_hz_20260831.c27_source_image_v1 import verify as verify_original
from experiments.neural_hz_20260831.c27_reference_transfer_v1 import verify as verify_semantics
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.c24_closed_state_v1 import export
from experiments.neural_hz_20260831.c24_checked_overlay_v1 import build as build_overlay
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import check_append,incidence_oracle,verify_all_and_discover,BranchPool
from experiments.neural_hz_20260831.c26_transplant_audit_v1 import plans_from_checked_incidence
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest,live_value_rows
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c27_owned_journal_20260911_v1'
SNAPSHOT=EXP/'results/c5_first_terminal_20260905_v1/layer75.pickle'
LIVE=EXP/'results/c25_live_relu_20260911_v1'
REFERENCE=EXP/'results/c26_transplant_census_20260911_v1'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def original_expression(snapshot):
    original,net=snapshot['expr_cache'][75],snapshot['net']
    dense=net.by_id[77]
    if (dense.kind!='DENSE' or net.by_id[76].kind not in {'FLATTEN','RESHAPE'}
            or net.preds[77]!=[76] or net.preds[76]!=[75]):
        raise ValueError('registered original affine suffix changed')
    weight=sp.csr_matrix(dense.params['weight'].detach().cpu().double().numpy())
    bias=dense.params.get('bias')
    bias=None if bias is None else bias.detach().cpu().double().numpy().reshape(-1)
    expr=cnn._lazy_append_linear(original,weight,bias,64_000_000)
    events=[json.loads(line) for line in (SNAPSHOT.parent/'composition_events.jsonl').read_text().splitlines()]
    expected=[e for e in events if e['event']=='materialization_start' and e['layer']==78]
    sources,terms={},[]
    for term in expr.terms:
        source=term.source
        if id(source) not in sources:
            sources[id(source)]={'source_index':len(sources),'n_out':source.n_out,'n_cont':source.n_cont,
                'n_bin':source.n_bin,'live_value_rows':int(live_value_rows(source).sum())}
        terms.append({'source_index':sources[id(source)]['source_index'],
            'operators':[{'type':type(op).__name__,'shape':list(op.shape)} for op in term.operators]})
    if (len(expected)!=1 or terms!=expected[0]['terms'] or list(sources.values())!=expected[0]['sources']
            or expr.n_out!=expected[0]['n_out']):
        raise ValueError('fresh expression differs from original native materialization schema')
    same_frame=[hz for hz in snapshot['hz_cache'].values() if hz.frame_id==expr.frame_id]
    return expr,(max(hz.n_cont for hz in same_frame),max(hz.n_bin for hz in same_frame))


def discover(closed,post,coupled,diagnostic):
    check_append(closed.hz,post)
    ne,nl=post.n_eq-closed.hz.n_eq,post.n_ineq-closed.hz.n_ineq
    first=closed.report['radix_uid_base']+16384
    new_nnz=int(post.Ac.nnz-closed.hz.Ac.nnz+post.Auc.nnz-closed.hz.Auc.nnz)
    coupled.charge('actual_appended_CSR_slices',8*new_nnz+4*(ne+nl))
    overlay,report=build_overlay(closed,[(post.Ac[closed.hz.n_eq:],first),
        (post.Auc[closed.hz.n_ineq:],first+ne)],pool=coupled,enabled=True)
    diagnostic.charge('complete_closed_UID_metadata',32*(len(closed.owners)+post.n_eq+post.n_ineq))
    eq,le=closed_uid_tables(closed)
    eq=np.r_[eq,np.arange(first,first+ne,dtype=np.int64)]
    le=np.r_[le,np.arange(first+ne,first+ne+nl,dtype=np.int64)]
    words=incidence_oracle(post,eq,le,closed.old_n_cont,closed.logical_n_cont,pool=diagnostic)
    columns,proof=verify_all_and_discover(closed,post,overlay,words,eq,le,
        whole=diagnostic,branch=BranchPool(diagnostic))
    plans=plans_from_checked_incidence(closed,post,columns,words,eq,le,pool=diagnostic)
    # Dense diagnostic tables/words die here. They are never a retained runtime
    # owner table and their full discovery cost is NOT treated as paid online.
    return plans,overlay,{'event_report':report,'discovery':proof,'diagnostic_work':diagnostic.used}


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    record={'schema':'c27_fresh_owned_journal_qualification_v1','completed':False,'formal_gain':0,
        'new_native_relu_executed':False,'new_spliced_HZ_constructed':False,'solver_executed':False,
        'whole_live_path_proved':False,'complete_integration_work_proved':False,
        'source_sha256':freeze['source_sha256'],'provenance':freeze['provenance']}
    started=time.monotonic(); initial=None
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('frozen source/artifact drift before deserialization')
        with SNAPSHOT.open('rb') as handle:snapshot=pickle.load(handle)
        with (LIVE/'relu78.pickle').open('rb') as handle:saved=pickle.load(handle)
        if saved['schema']!='c25_live_relu_checkpoint_v1' or not saved['whole_live_path_proved']:
            raise ValueError('not the complete actual native phase archive')
        expr,widths=original_expression(snapshot)
        post=saved['post_relu_hz'];post_sha=source_digest(post)
        inputs={'snapshot':snapshot,'original_expression':expr,'complete_actual_archive':saved}
        initial=collect(SimpleNamespace(),inputs).fingerprint
        with (DIRECTORY/'events.jsonl').open('x') as stream:
            def emit(name,values):
                event={'event':name,**values,'worker_elapsed_s':time.monotonic()-started}
                stream.write(json.dumps(event)+'\n');stream.flush();print(json.dumps(event),flush=True)
            (draft,permit),construction=measured_build(lambda:generate(expr,np.ones(expr.n_out,bool),
                enabled=True,frame_widths=widths,observe=emit))
            record.update(fresh_generation_executed=True,generation=construction,generation_report=draft.report)
            emit('fresh_original_expression_generated',{'construction':construction,'report':draft.report})
            closed,binding=bind(draft,saved['closed_proof_bytes'],expected_proof_sha256=PROOF,enabled=True)
            coupled=CoupledPool(draft.report['total_work_upper'],draft.report['largest_branch_work_upper'])
            # New permit object, two weakrefs and issuance checks have a stated
            # incremental tariff; they are not buried in old generator work.
            coupled.charge('fresh_ownership_permit_issuance',32)
            diagnostic=WorkPool(256_000_000)
            plans,overlay,discovery=discover(closed,post,coupled,diagnostic)
            record.update(binding=binding,complete_source_discovery=discovery)
            emit('complete_source_derived_plan_discovery',discovery)
            if len(plans)!=268:raise ValueError('registered complete population changed')
            fields,raw=export(closed)
            addresses={k:{'object_id':id(fields[k]),'backing_pointer':int(fields[k].ctypes.data)}
                for k in ('eq_roots','eq_scales')}
            retired=[weakref.ref(n[k]) for n in draft.nodes for k in ('support','needed','slots','exponents')]
            draft_ref,closed_ref=weakref.ref(draft),weakref.ref(closed)
            del draft,closed
            gc.collect()
            if draft_ref() is not None or closed_ref() is not None or any(ref() is not None for ref in retired):
                raise ValueError('fresh construction graph or unpublished old Closed still alive')
            before_work=coupled.used
            # Two seal scans inside the compiler are authentication, NOT sparse
            # edit work. Price their full numeric reads in the diagnostic ledger.
            diagnostic.charge('compiler_full_lineage_seal_reads',
                2*(len(fields['eq_roots'])+len(fields['eq_scales'])+2*len(plans)+sum(
                    fields['old_n_cont']<=col<fields['logical_n_cont'] for p in plans for col in p.tail)))
            lineage,journal_build=measured_build(lambda:compile_owned(fields['eq_roots'],fields['eq_scales'],plans,
                old_n_cont=fields['old_n_cont'],old_n_eq=fields['old_n_eq'],pool=coupled,
                ownership=permit,enabled=True))
            if lineage is None:raise ValueError('missing complete reversible journal')
            after_addresses={k:{'object_id':id(fields[k]),'backing_pointer':int(fields[k].ctypes.data)}
                for k in ('eq_roots','eq_scales')}
            if after_addresses!=addresses:raise ValueError('new owned maps were replaced/copied')
            record.update(journal_construction=journal_build,owned_map_addresses=addresses,
                full_map_array_copies=0,retired_fresh_graph_arrays=len(retired),
                fresh_draft_and_old_Closed_physically_retired=True,journal_sparse_update_work=coupled.used-before_work)
            auth_start=time.monotonic()
            old_image=verify_original(fields,raw,expected_proof_sha256=PROOF)
            record['original_image_authentication_s']=time.monotonic()-auth_start
            record['original_image_proof']=old_image
            emit('fresh_owned_journal_and_original_image_passed',{
                'journal_work':coupled.used-before_work,'construction':journal_build,**old_image})
            # Independent reference is opened ONLY AFTER complete discovery,
            # fresh generation, exclusive mutation and old-image authentication.
            ref_result=json.loads((REFERENCE/'result.json').read_text())
            ref_exit=json.loads((REFERENCE/'exit.json').read_text())
            if (not ref_result['completed'] or ref_exit.get('worker_exit_code')!=0 or ref_exit.get('tests_exit_code')!=0
                    or ref_exit.get('source_drift') or ref_exit.get('provenance_drift') or ref_exit.get('timeout_s')):
                raise ValueError('independent reference did not complete')
            with (REFERENCE/'lineage.pickle').open('rb') as handle:reference=pickle.load(handle)
            if (reference['schema']!='c26_provisional_transplant_lineage_v1'
                    or reference['source_post_HZ_sha256']!=post_sha
                    or reference['closed_identity']!=old_image['complete_original_source_image_sha256']
                    or reference['source_archive_sha256']!=_sha256(LIVE/'relu78.pickle')
                    or ref_result['checkpoint_sha256']!=_sha256(REFERENCE/'lineage.pickle')
                    or reference['proof']!=ref_result['proof']):
                raise ValueError('independent complete source/semantic proof chain differs')
            inherited=reference['proof']
            if (not inherited['complete_transferred_incidence_equal']
                    or not inherited['independent_exact_source_and_box_proof_transferred']
                    or not inherited['all_binaries_and_global_frame_retained_in_reference']):
                raise ValueError('incomplete exact predicate/UID/box reference')
            semantics=verify_semantics(lineage,reference['draft'],
                expected_reference_fingerprint=reference['draft'].fingerprint(),pool=diagnostic)
            if (semantics['all_lineage_slots_compared']!=inherited['all_lineage_slots_checked']
                    or len(lineage.columns)!=inherited['all_tagged_unit_pairs_checked']):
                raise ValueError('partial reference proof transfer')
            record.update(new_semantic_reference_proof=semantics,inherited_exact_predicate_and_UID_proof=inherited,
                independent_diagnostic_work=diagnostic.used,independent_diagnostic_work_parts=dict(diagnostic.parts))
            # This is deliberately an INCOMPLETE integration subtotal: full
            # discovery/authentication and a real fused native writer remain.
            subtotal=coupled.whole_base+coupled.used
            record['cost_assessment']={'generation_overlay_permit_sparse_edit_subtotal':subtotal,
                'branch_subtotal':coupled.branch_base+coupled.used,'incremental_parts':dict(coupled.parts),
                'subtotal_headroom_only':256_000_000-subtotal,'NOT_complete_integration_cost':True,
                'independent_diagnostic_work_excluded_from_subtotal':diagnostic.used,
                'full_source_authentication_measured_separately':True,
                'native_fused_predicate_writer_implemented':False,'production_discovery_implemented':False,
                'full_map_array_copy_work_removed':488624,'new_sparse_entries':sum(len(getattr(lineage,k)) for k in ('columns','retired','tails')),
                'new_sparse_bytes':sum(getattr(lineage,k).nbytes for k in ('columns','retired','tails')),
                'formal_gain':0}
            if collect(SimpleNamespace(),inputs).fingerprint!=initial:raise ValueError('original sources mutated')
            record['all_original_snapshot_and_actual_archive_roots_unchanged']=True
            emit('complete_journal_semantic_proof',semantics)
            emit('incomplete_integration_cost_assessment',record['cost_assessment'])
            path=DIRECTORY/'journal.pickle'
            with path.open('xb') as handle:
                pickle.dump({'schema':'c27_provisional_reversible_journal_v1',
                    'original_fields_with_reversible_lineage':fields,'lineage':lineage,'original_proof_bytes':raw,
                    'original_image_proof':old_image,'semantic_reference_proof':semantics,
                    'reference_archive_sha256':_sha256(REFERENCE/'lineage.pickle'),
                    'actual_post_HZ_sha256':post_sha,'provenance':freeze['provenance'],
                    'source_sha256':freeze['source_sha256'],'formal_gain':0,
                    'live_admission_certificate':False,'contains_valid_new_closed_HZ':False},handle,protocol=5)
                handle.flush();os.fsync(handle.fileno())
            record.update(completed=True,checkpoint_sha256=_sha256(path),checkpoint_bytes=path.stat().st_size)
    except Exception as exc:
        record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        if initial is not None:
            record['all_original_snapshot_and_actual_archive_roots_unchanged']=collect(SimpleNamespace(),inputs).fingerprint==initial
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'result.json',record)
        print(json.dumps({'completed':record['completed'],'failure':record.get('failure'),'formal_gain':0}),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
