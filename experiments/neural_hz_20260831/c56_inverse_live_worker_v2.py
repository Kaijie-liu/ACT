"""One measured exact-lift prototype, with all positive/negative storage records."""
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
from experiments.neural_hz_20260831.c56_inverse_live_v2 import build,state_hash
from experiments.neural_hz_20260831.c56_inverse_live_audit_v2 import audit,check_points,comparison
from experiments.neural_hz_20260831.c54_packed_scalar_hz_v2 import build as exact_build
from experiments.neural_hz_20260831.c54_scalar_fixtures_v1 import fixture
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import reference
from experiments.neural_hz_20260831.c52_signed_first_write_audit_v1 import audit as reference_audit
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c56_inverse_live_20260912_v2'
GATES=('strict_predicate_nnz_decrease','strict_numeric_bytes_decrease','strict_numeric_entries_decrease','strict_combined_reported_accounting_decrease')


def complete(pool,emit):
    archive={};reports=[]
    for kind in ('chain','shared_add','conv_relu'):
        program=fixture(kind);original=reference(program,pool=pool);old_proof=reference_audit(program,original,pool=pool)
        exact=exact_build(program,enabled=True,pool=pool)
        retired=[weakref.ref(a) for a in exact['csr'].values() if hasattr(a,'dtype')]
        retired.extend(weakref.ref(exact[k]) for k in ('rhs','eq_uids','inverse'))
        retired.extend(weakref.ref(a) for a in exact['scalars'].values())
        state=build(exact,enabled=True,pool=pool);proof=audit(program,exact,state,pool=pool);del exact
        if len(retired)!=10 or any(ref() is not None for ref in retired):raise ValueError('old unshared exact arrays remain live')
        state['report']['unshared_exact_numeric_arrays_retired']=len(retired);state['seal']=state_hash(state)
        points=check_points(program,original,state);physical=comparison(program,original,state)
        report=dict(kind=kind,proof=proof,points=points,physical=physical,construction=state['report'],
            original_n_cont=program['nc'],lowered_n_cont=state['n_cont'],binary_factors=state['n_bin'],representation_accepted=all(physical[k] for k in GATES))
        archive[kind]=dict(program=program,reference=original,lowered=state,reference_proof=old_proof,proof=proof,report=report)
        reports.append(report);emit(dict(event='complete_cohort',**report))
    roots=collect(SimpleNamespace(),dict(complete_archive=archive));owner=roots.measure()
    if owner.resident_entries>64_000_000:raise MemoryError('full retained entry cap')
    pool.charge('c56_complete_retained_owners_and_points',4*owner.resident_entries+65536)
    path=DIRECTORY/'complete_sources_and_lifted_states.pickle'
    with path.open('xb') as stream:pickle.dump(archive,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    artifact_sha=_sha256(path)
    with path.open('rb') as stream:restored,decoder=load(stream,expected_sha256=artifact_sha,pool=pool,enabled=True)
    replay=[]
    for kind,item in restored.items():
        old=reference_audit(item['program'],item['reference'],pool=pool)
        if any(value!=item['reference_proof'][key] for key,value in old.items() if key!='proof_work'):raise ValueError('restored reference proof differs')
        exact=exact_build(item['program'],enabled=True,pool=pool)
        proof=audit(item['program'],exact,item['lowered'],pool=pool);del exact
        if proof!=item['proof']:raise ValueError('restored actual-row proof differs')
        points=check_points(item['program'],item['reference'],item['lowered'])
        if points!=item['report']['points']:raise ValueError('restored original point proof differs')
        physical=comparison(item['program'],item['reference'],item['lowered'])
        replay.append(dict(kind=kind,proof=proof,points=points,physical=physical,representation_accepted=all(physical[k] for k in GATES)))
    all_roots=collect(SimpleNamespace(),dict(fresh=archive,restored=restored));both=all_roots.measure()
    if both.resident_entries>64_000_000:raise MemoryError('full fresh and restored entry cap')
    pool.charge('c56_full_restored_owners_and_points',4*both.resident_entries+65536)
    return dict(cases=reports,authenticated_replay=replay,decoder=decoder,
        all_fixed_cohorts_pass=all(r['representation_accepted'] for r in reports+replay),
        complete_retained_numeric_bytes=owner.resident_bytes,complete_retained_numeric_entries=owner.resident_entries,
        complete_retained_shallow_bytes=roots.python_shallow_bytes,fresh_and_restored_numeric_bytes=both.resident_bytes,
        fresh_and_restored_numeric_entries=both.resident_entries,fresh_and_restored_shallow_bytes=all_roots.python_shallow_bytes,
        artifact_sha256=artifact_sha,artifact_bytes=path.stat().st_size,formal_gain=0)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));started=time.monotonic();pool=WorkPool(256_000_000)
    report=dict(diagnostic_completed=False,representation_accepted=False,formal_gain=0,native_solver_or_actual_target_executed=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('source freeze drift')
            data,stats=measured(lambda:complete(pool,emit),observe=lambda stats:emit(dict(event='full_prototype_measurement',measurement=stats)))
            report.update(diagnostic_completed=True,data=data,measurement=stats,representation_accepted=data['all_fixed_cohorts_pass'])
        except Exception as exc:report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='diagnostic_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_work=pool.used,work_parts=dict(pool.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report);print(json.dumps({k:report[k] for k in ('diagnostic_completed','representation_accepted','wall_s','formal_gain')}),flush=True)
    raise SystemExit(0 if report['diagnostic_completed'] and report['representation_accepted'] else 2 if report['diagnostic_completed'] else 1)


if __name__=='__main__':main()
