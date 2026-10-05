"""One complete actual-source consumer-directed discovery qualification."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c28_consumer_discovery_v1 import discover
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import check_append,incidence_oracle,verify_all_and_discover,BranchPool
from experiments.neural_hz_20260831.c26_transplant_audit_v1 import plans_from_checked_incidence
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import compile_lineage
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c28_consumer_discovery_20260911_v1'
LIVE=EXP/'results/c25_live_relu_20260911_v1'
REFERENCE=EXP/'results/c26_transplant_census_20260911_v1'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def plan_shallow_bytes(plans):
    seen=set()
    def visit(v):
        if id(v) in seen:return 0
        seen.add(id(v));amount=sys.getsizeof(v)
        if isinstance(v,dict):return amount+sum(visit(k)+visit(x) for k,x in v.items())
        if isinstance(v,(tuple,list)):return amount+sum(map(visit,v))
        if hasattr(v,'__dict__'):return amount+visit(vars(v))
        if type(v) not in (int,float,bool,str,type(None)):raise ValueError('unregistered plan metadata')
        return amount
    return visit(plans)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    record={'schema':'c28_complete_consumer_directed_discovery_v1','completed':False,'formal_gain':0,
        'new_generator_executed':False,'new_native_relu_executed':False,'new_spliced_HZ_constructed':False,
        'solver_executed':False,'whole_live_path_proved':False,'source_sha256':freeze['source_sha256'],
        'provenance':freeze['provenance']}
    started=time.monotonic()
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('frozen source/artifact drift before deserialization')
        auth_start=time.monotonic()
        with (LIVE/'relu78.pickle').open('rb') as handle:saved=pickle.load(handle)
        if saved['schema']!='c25_live_relu_checkpoint_v1' or not saved['whole_live_path_proved']:
            raise ValueError('not the complete independently bound native archive')
        closed=restore(saved['closed_fields'],saved['closed_proof_bytes'],expected_proof_sha256=PROOF)
        post=saved['post_relu_hz'];owned=saved['phase_ownership']
        overlay=Overlay(owned['base'],owned['events'],owned['old_uid_ceiling'])
        post_sha=source_digest(post);closed_sha=closed.fingerprint()
        event_sha=hashlib.sha256(overlay.events.tobytes()).hexdigest()
        if (owned['post_hz'] is not post or owned['base'] is not closed.owners
                or owned['post_sha256']!=post_sha or owned['event_sha256']!=event_sha):
            raise ValueError('actual source/phase/ownership binding differs')
        closed.validate();overlay.validate();check_append(closed.hz,post)
        record['source_authentication_s']=time.monotonic()-auth_start
        with (DIRECTORY/'events.jsonl').open('x') as stream:
            def emit(name,values):
                event={'event':name,**values,'worker_elapsed_s':time.monotonic()-started}
                stream.write(json.dumps(event)+'\n');stream.flush();print(json.dumps(event),flush=True)
            direct=WorkPool(256_000_000)
            (plans,stats),construction=measured_build(lambda:discover(closed,post,overlay,pool=direct,enabled=True))
            record.update(discovery_work=direct.used,discovery_work_parts=dict(direct.parts),
                discovery_statistics=stats,construction=construction,plan_python_shallow_bytes=plan_shallow_bytes(plans))
            emit('complete_consumer_directed_discovery',{'work':direct.used,'parts':dict(direct.parts),
                'statistics':stats,'construction':construction})
            # Only AFTER the new complete algorithm returns, run the original
            # independent factor-directed full-incidence oracle. No precomputed
            # column set or full UID table was passed into new discovery.
            diagnostic=WorkPool(256_000_000)
            diagnostic.charge('complete_closed_UID_metadata',32*(len(closed.owners)+post.n_eq+post.n_ineq))
            eq,le=closed_uid_tables(closed);first=overlay.old_uid_ceiling
            ne,nl=post.n_eq-closed.hz.n_eq,post.n_ineq-closed.hz.n_ineq
            eq=np.r_[eq,np.arange(first,first+ne,dtype=np.int64)]
            le=np.r_[le,np.arange(first+ne,first+ne+nl,dtype=np.int64)]
            actual=incidence_oracle(post,eq,le,closed.old_n_cont,closed.logical_n_cont,pool=diagnostic)
            columns,proof=verify_all_and_discover(closed,post,overlay,actual,eq,le,
                whole=diagnostic,branch=BranchPool(diagnostic))
            expected=plans_from_checked_incidence(closed,post,columns,actual,eq,le,pool=diagnostic)
            if [asdict(v) for v in plans]!=[asdict(v) for v in expected]:
                raise ValueError('complete consumer-directed plans differ from independent factor-directed oracle')
            if (len(plans)!=268 or stats['selected_new_consumers']!=199 or stats['selected_old_consumers']!=69):
                raise ValueError('registered complete population changed')
            independent_discovery=diagnostic.used
            without_full_oracle=independent_discovery-diagnostic.parts['independent_complete_incidence']-diagnostic.parts['complete_closed_UID_metadata']
            if direct.used>=without_full_oracle:
                raise ValueError('new discovery did not strictly reduce selection-only work')
            # Transfer the independently completed C26 full source/math/box/UID
            # proof by matching its entire deterministic lineage fingerprint.
            reference_result=json.loads((REFERENCE/'result.json').read_text())
            reference_exit=json.loads((REFERENCE/'exit.json').read_text())
            if (not reference_result['completed'] or reference_exit.get('worker_exit_code')!=0
                    or reference_exit.get('tests_exit_code')!=0 or reference_exit.get('source_drift')
                    or reference_exit.get('provenance_drift') or reference_exit.get('timeout_s')):
                raise ValueError('C26 independent exact reference did not complete')
            with (REFERENCE/'lineage.pickle').open('rb') as handle:reference=pickle.load(handle)
            if (reference['schema']!='c26_provisional_transplant_lineage_v1'
                    or reference['closed_identity']!=closed_sha or reference['source_post_HZ_sha256']!=post_sha
                    or reference['source_archive_sha256']!=_sha256(LIVE/'relu78.pickle')
                    or reference['proof']!=reference_result['proof']):
                raise ValueError('C26 complete source/semantic proof chain mismatch')
            lineage=compile_lineage(closed.eq_roots,closed.eq_scales,plans,old_n_cont=closed.old_n_cont,
                old_n_eq=closed.old_n_eq,pool=diagnostic,enabled=True)
            reference['draft'].validate()
            if lineage is None or lineage.fingerprint()!=reference['draft'].fingerprint():
                raise ValueError('complete new plans do not reproduce the independently proved lineage')
            if (closed.fingerprint()!=closed_sha or source_digest(post)!=post_sha
                    or hashlib.sha256(overlay.events.tobytes()).hexdigest()!=event_sha):
                raise ValueError('authenticated original source was mutated')
            assessment={'old_independent_discovery_plus_plan_work':independent_discovery,
                'old_selection_only_work_excluding_full_incidence_and_dense_UID_metadata':without_full_oracle,
                'actual_consumer_discovery_work':direct.used,
                'strict_selection_only_work_reduction':True,
                'C27_incomplete_integration_subtotal':255_910_983,
                'subtotal_plus_actual_new_discovery':255_910_983+direct.used,
                'subtotal_plus_discovery_within_256M':255_910_983+direct.used<=256_000_000,
                'remaining_native_fused_writer_implemented':False,'full_live_admission_proved':False,
                'discovery_is_not_an_incidence_membership_certificate':True,
                'source_authentication_and_independent_proof_excluded_from_subtotal':True,'formal_gain':0}
            record.update(independent_factor_directed_proof=proof,independent_diagnostic_work=diagnostic.used,
                independent_diagnostic_work_parts=dict(diagnostic.parts),inherited_exact_C26_proof=reference['proof'],
                complete_deterministic_lineage_sha256=lineage.fingerprint(),cost_assessment=assessment,
                all_original_source_fields_unchanged=True)
            emit('independent_complete_plan_and_semantic_equivalence',{'plans':len(plans),
                'old_MAIN_incidence_checked':len(actual),'deterministic_lineage_sha256':lineage.fingerprint()})
            emit('actual_discovery_cost_assessment',assessment)
            _atomic_exclusive_json(DIRECTORY/'plans.json',{'schema':'c28_provisional_consumer_plans_v1',
                'plans':[asdict(v) for v in plans],'source_closed_sha256':closed_sha,'actual_post_HZ_sha256':post_sha,
                'source_archive_sha256':_sha256(LIVE/'relu78.pickle'),
                'independent_C26_lineage_sha256':_sha256(REFERENCE/'lineage.pickle'),
                'source_sha256':freeze['source_sha256'],'provenance':freeze['provenance'],
                'live_admission_certificate':False,'formal_gain':0})
            record.update(completed=True,plan_artifact_sha256=_sha256(DIRECTORY/'plans.json'),
                plan_artifact_bytes=(DIRECTORY/'plans.json').stat().st_size)
    except Exception as exc:
        record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'result.json',record)
        print(json.dumps({'completed':record['completed'],'failure':record.get('failure'),'formal_gain':0}),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
