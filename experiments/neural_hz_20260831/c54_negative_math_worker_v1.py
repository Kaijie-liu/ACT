"""Complete diagnostic of frozen rejected scalar-HZ layout; no promotion."""
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c54_scalar_hz_v1 import build
from experiments.neural_hz_20260831.c54_scalar_hz_audit_v1 import audit,comparison
from experiments.neural_hz_20260831.c54_scalar_fixtures_v1 import fixture,check_points
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import reference
from experiments.neural_hz_20260831.c52_signed_first_write_audit_v1 import audit as reference_audit
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c54_exact_scalar_negative_20260912_v1'


def complete(pool,emit):
    archive={};reports=[]
    for kind in ('chain','shared_add','conv_relu'):
        program=fixture(kind);original=reference(program,pool=pool);state=build(program,enabled=True,pool=pool)
        old_proof=reference_audit(program,original,pool=pool);proof=audit(program,state,pool=pool)
        points=check_points(kind,program,original,state);physical=comparison(program,original,state)
        passed=all(physical[k] for k in ('strict_predicate_nnz_decrease','strict_numeric_bytes_decrease','strict_numeric_entries_decrease','strict_combined_reported_accounting_decrease'))
        report=dict(kind=kind,proof=proof,points=points,physical=physical,representation_accepted=passed,
            original_n_cont=program['nc'],compact_n_cont=state['n_cont'],binary_factors=program['nb'],construction=state['report'])
        archive[kind]=dict(program=program,reference=original,exact=state,reference_proof=old_proof,proof=proof,report=report)
        reports.append(report);emit(dict(event='fixed_cohort_complete_diagnostic',**report))
    roots=collect(SimpleNamespace(),dict(complete_archive=archive));owner=roots.measure()
    if owner.resident_entries>64_000_000:raise MemoryError('unchanged complete archive entry cap')
    pool.charge('c54_complete_ownership_points_and_archive_diagnostics',4*owner.resident_entries+65536)
    path=DIRECTORY/'complete_sources_and_exact_states.pickle'
    with path.open('xb') as stream:pickle.dump(archive,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    return dict(cases=reports,all_fixed_cohorts_pass=all(r['representation_accepted'] for r in reports),
        complete_retained_numeric_bytes=owner.resident_bytes,complete_retained_numeric_entries=owner.resident_entries,
        complete_numeric_roots=len(roots.numeric),complete_python_shallow_bytes=roots.python_shallow_bytes,
        artifact_sha256=_sha256(path),artifact_bytes=path.stat().st_size,formal_gain=0)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered diagnostic output')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));started=time.monotonic();pool=WorkPool(256_000_000)
    report=dict(diagnostic_completed=False,representation_accepted=False,formal_gain=0,
        focused_suite_passed=False,native_solver_or_actual_target_executed=False)
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
            data,stats=measured(lambda:complete(pool,emit),observe=lambda stats:emit(dict(event='entire_fixed_math_measurement',measurement=stats)))
            report.update(diagnostic_completed=True,data=data,measurement=stats,representation_accepted=data['all_fixed_cohorts_pass'])
        except Exception as exc:report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='diagnostic_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,work_parts=dict(pool.parts))
            _atomic_exclusive_json(DIRECTORY/'result.json',report);print(json.dumps({k:report[k] for k in ('diagnostic_completed','representation_accepted','wall_s','formal_gain')}),flush=True)
    raise SystemExit(0 if report['diagnostic_completed'] and report['representation_accepted'] else 2 if report['diagnostic_completed'] else 1)


if __name__=='__main__':main()
