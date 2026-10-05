"""One fresh C51-qualified span generator and new complete source proof; no native solve.

Complete offline oracle loading is measured too. Offline input qualification,
fresh construction and independent proof diagnostics are separate explicit
boundaries, as C31, never a claim of combined full-request LIVE payment.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace
import weakref
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.neural_hz_20260831.c49_span_emission_v1 import lift
from experiments.neural_hz_20260831.c49_span_report_audit_v1 import audit
from experiments.neural_hz_20260831.c24_closed_state_v1 import close,export,restore
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.c27_owned_journal_worker_v1 import original_expression
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c51_source_custody_20260911_v1'
INPUTS={
    'snapshot':('results/c5_first_terminal_20260905_v1/layer75.pickle','d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'),
    'original':('results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle','616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'),
    'old_generator':('results/c31_prepared_generator_20260911_v1/closed_hz.pickle','535b579e853f1ac5092e962e1c307644e89739da4cd63f3732593896649128f5')}
OLD_PROOF='cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5'


def load_all(pool,emit):
    saved={};decoders={}
    for key,(name,sha) in INPUTS.items():
        with (EXP/name).open('rb') as f:saved[key],decoders[key]=load(f,expected_sha256=sha,pool=pool,enabled=True)
        emit('complete_offline_input_decoded',dict(input=key,decoder=decoders[key]))
    if (saved['original']['schema']!='c9_integrated_suffix_checkpoint_v1'
            or saved['old_generator']['schema']!='c31_new_checked_prepared_Closed_v1'
            or saved['old_generator']['closed_proof_sha256']!=OLD_PROOF):
        raise ValueError('wrong complete authenticated offline source inputs')
    pool.charge('c49_complete_offline_owner_qualification',1_048_576)
    roots=collect(SimpleNamespace(),saved);owner=roots.measure()
    return saved,dict(decoders=decoders,all_saved_input_fields_retained=True,
        numeric_roots=len(roots.numeric),numeric_bytes=owner.resident_bytes,numeric_entries=owner.resident_entries,
        python_shallow_bytes=roots.python_shallow_bytes,input_fingerprint=roots.fingerprint,
        offline_reference_union_not_full_verification_LIVE=True)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered new source generation')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.monotonic();saved=None;initial=None
    record=dict(completed=False,formal_gain=0,new_native_executed=False,solver_executed=False,
        default_changed=False,whole_verification_LIVE_or_runtime_payment_proved=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(name,values):
            log.write(json.dumps(dict(event=name,worker_elapsed_s=time.monotonic()-started,**values),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('all source/reference hashes must match')
            record.update(provenance=freeze['provenance'],source_sha256=freeze['source_sha256'])
            decoding=WorkPool(256_000_000)
            (saved,inputs),stats=measured(lambda:load_all(decoding,emit),
                observe=lambda stats:emit('complete_offline_input_load_measurement',dict(measurement=stats)))
            initial=inputs['input_fingerprint']
            record.update(offline_inputs=inputs,offline_input_measurement=stats,
                offline_input_work=decoding.used,offline_input_work_parts=dict(decoding.parts))
            old_saved=saved['old_generator'];old=restore(old_saved['fields'],old_saved['proof_bytes'],expected_proof_sha256=OLD_PROOF)
            old_fingerprint=old.fingerprint()
            original=saved['original'];original_hz=original['hz']
            maps={k:original[k] for k in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')}
            if source_digest(original_hz)!='16337ddcef267f17eff8313db81e92a8049589613a9031db799453b759089ba2':
                raise ValueError('complete original source differs from frozen C48 population')
            expr,widths=original_expression(saved['snapshot'])
            draft,stats=measured(lambda:lift(expr,np.ones(expr.n_out,bool),enabled=True,frame_widths=widths,observe=emit),
                observe=lambda stats:emit('complete_fresh_generator_measurement',dict(measurement=stats)))
            record.update(fresh_original_expression_generation_executed=True,generation=stats,generation_report=draft.report)
            emit('new_source_generator_completed',dict(measurement=stats,report=draft.report))
            diagnostic=WorkPool(256_000_000);branch=BranchPool(diagnostic);t=time.monotonic()
            report_proof=audit(draft,old,original_hz,maps,pool=branch,enabled=True)
            if not report_proof['strict_component_payment']:raise ValueError('complete new generator has no positive routing payment')
            record.update(complete_report_proof=report_proof,report_proof_wall_s=time.monotonic()-t,
                report_diagnostic_work=diagnostic.used,report_branch_work=branch.used,report_work_parts=dict(diagnostic.parts))
            emit('new_report_and_all_original_bits_proved',dict(proof=report_proof,diagnostic_work=diagnostic.used))
            try:bind(draft,old_saved['proof_bytes'],expected_proof_sha256=OLD_PROOF,enabled=True)
            except ValueError:record['old_source_proof_substitution_rejected']=True
            else:raise ValueError('old proof accepted the newly priced generator')
            graph_refs=[weakref.ref(n[k]) for n in draft.nodes for k in ('support','needed','slots','exponents')]
            draft_ref=weakref.ref(draft);t=time.monotonic()
            closed,proof=close(draft,original_hz,maps,enabled=True)
            record['new_full_source_proof_wall_s']=time.monotonic()-t
            if closed.hz is not draft.hz or closed.owners is not draft.owners:raise ValueError('checker substituted an archived generated state')
            fields,raw=export(closed);sha=hashlib.sha256(raw).hexdigest()
            if sha==OLD_PROOF or closed.fingerprint()==old_fingerprint:raise ValueError('old proof identity reused')
            del draft;gc.collect()
            if draft_ref() is not None or any(r() is not None for r in graph_refs):raise ValueError('new unpublished graph storage remains alive')
            if len(graph_refs)!=144:raise ValueError('complete source graph retirement population changed')
            if old.fingerprint()!=old_fingerprint or collect(SimpleNamespace(),saved).fingerprint!=initial:
                raise ValueError('complete original source inputs changed')
            closed.validate()
            with (DIRECTORY/'closed_proof.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            payload=dict(schema='c51_new_checked_span_Closed_v1',fields=fields,proof_bytes=raw,
                closed_proof_sha256=sha,identity=proof,complete_report_proof=report_proof,
                provenance=freeze['provenance'],source_sha256=freeze['source_sha256'],
                old_checked_proof_sha256=OLD_PROOF,origin_snapshot_sha256=INPUTS['snapshot'][1],
                all_new_graph_fields_physically_retired=True,whole_live_path_proved=False,formal_gain=0)
            path=DIRECTORY/'closed_hz.pickle'
            with path.open('xb') as f:pickle.dump(payload,f,protocol=5);f.flush();os.fsync(f.fileno())
            record.update(completed=True,new_complete_source_proof=proof,new_closed_identity=closed.fingerprint(),
                new_closed_proof_sha256=sha,new_HZ_sha256=source_digest(closed.hz),
                new_graph_arrays_retired=len(graph_refs),all_new_graph_fields_physically_retired=True,
                all_original_saved_input_fields_unchanged=True,checkpoint_sha256=_sha256(path),checkpoint_bytes=path.stat().st_size,
                actual_new_generator_work=draft_report_work(closed),new_source_proof_issued=True,
                formal_gain=0)
            emit('new_complete_source_owner_UID_proof_saved',dict(proof_sha256=sha,proof=proof,
                new_graph_arrays_retired=len(graph_refs),checkpoint_sha256=record['checkpoint_sha256']))
        except Exception as exc:
            record['failure']=dict(type=type(exc).__name__,reason=str(exc));emit('new_generator_rejected',record['failure'])
        finally:
            record.update(wall_s=time.monotonic()-started,max_rss_kib_including_offline_inputs=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY/'result.json',record)
            print(json.dumps(dict(completed=record['completed'],failure=record.get('failure'),formal_gain=0)),flush=True)
    if not record['completed']:raise SystemExit(1)


def draft_report_work(closed):
    return dict(whole=closed.report['total_work_upper'],branch=closed.report['largest_branch_work_upper'],
        is_actual_fresh_construction=True,full_native_or_runtime_payment_proved=False)


if __name__=='__main__':main()
