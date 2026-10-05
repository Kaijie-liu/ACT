"""Fixed issued-source bridge qualification; no benchmark loading or solver."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import re
import sys
import time
from types import SimpleNamespace
import weakref
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c47_source_fixture_v1 import execute
from experiments.neural_hz_20260831.c47_bound_half_source_v1 import build
from experiments.neural_hz_20260831.c47_source_payment_floor_v1 import append_screen
from experiments.neural_hz_20260831.test_c47_bound_half_source_v1 import check_points
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c47_bound_source_20260911_v1'


def bridge(old,tf,layer,pool):
    pool.charge('c47_toy_complete_owner_comparison_allowance',262144)
    before=collect(SimpleNamespace(),old.numeric_roots());bm=before.measure()
    state,report=build(old,pool=pool,enabled=True)
    points=check_points(old,state,tf,layer,pool)
    after=collect(SimpleNamespace(),state.numeric_roots(pool=pool));am=after.measure()
    metrics=dict(scope='entire_bound_source_object_only_not_full_native_LIVE',before=asdict(bm),after=asdict(am),
        numeric_byte_delta=am.resident_bytes-bm.resident_bytes,numeric_entry_delta=am.resident_entries-bm.resident_entries,
        strict_both_numeric_metrics=am.resident_bytes<bm.resident_bytes and am.resident_entries<bm.resident_entries,
        python_shallow_before=before.python_shallow_bytes,python_shallow_after=after.python_shallow_bytes,
        retained_reference_HZs_in_bridge_transient_measurement=True,full_LIVE_proved=False)
    return state,dict(binding=report,points=points,owner_metrics=metrics,
        old_EQ=old.hz.n_eq,new_EQ=state.hz.n_eq,continuous_factors=state.hz.n_cont,binary_factors=state.hz.n_bin,
        old_unit_factors=len(state.lineage.columns),old_alias_factors=int((state.lineage.eq_roots<0).sum()))


def actual_lower_bound():
    # Full file SHA is independently frozen/verified. Read its fixed-format
    # top-level literal from a bounded prefix, not its57MB retained-root dump.
    with (EXP/'results/c34_changed_terminal_20260911_v1/terminal_gate.json').open('rb') as f:
        prefix=f.read(131072)
    found=re.findall(rb'^  "generation_plus_incremental_work": ([0-9]+),$',prefix,re.M)
    if len(found)!=1:raise ValueError('frozen C34 top-level work header unavailable in bounded prefix')
    used=int(found[0])
    diagnostic=json.loads((EXP/'results/c40_compact_half_gauge_20260911_v1/result.json').read_text())
    census=json.loads((EXP/'results/c39_half_alias_row_gauge_20260911_v1/result.json').read_text())
    n=diagnostic['source_binding']['complete_original_source_image']['full_original_lineage_slots_restored_in_hash']
    k=census['census']['proposed_half_aliases']
    return dict(map_entries=n,half_factors=k,
        C34_fresh_append=append_screen(used,n,k),
        C40_diagnostic_append=append_screen(diagnostic['whole_diagnostic_work'],n,k),
        actual_matrices_loaded=False,solver_executed=False)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered bound source diagnostic')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);pool.charge('c47_fixed_evidence_and_bounded_archived_header_checks',4_194_304)
    started=time.monotonic();report=dict(completed=False,formal_gain=0,configurations=[],
        actual_benchmark_source_loaded=False,actual_benchmark_native_admitted=False,solver_executed=False)
    events=0
    with (DIRECTORY/'events.jsonl').open('x') as log:
        def emit(value):
            nonlocal events
            events+=1
            if events>64:raise ValueError('unpaid extra evidence events')
            log.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');log.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('complete source freeze drift')
            for sign in (-1,1):
                for layer in (2,78,1001):
                    begin=pool.used;source_started=time.monotonic()
                    # Fixed small source/reference-test factory, not an actual
                    # benchmark-generation credit. It uses unchanged native
                    # component tracers and cannot nest inside C41's tracer.
                    pool.charge('c47_fixed_small_source_factory_allowance',1_048_576)
                    with pytest.MonkeyPatch.context() as monkeypatch:
                        tf,runtime,hz,want=execute(monkeypatch,sign=sign,layer=layer)
                        old=runtime['lifted']
                        source_measurements=dict(generation=runtime['construction'],native_phase=runtime['consumer_construction'])
                        source_wall=time.monotonic()-source_started
                        if any(not x['measured_transient_gate'] for x in source_measurements.values()):
                            raise ValueError('original native component transient gate failed')
                        (state,result),stats=measured(lambda:bridge(old,tf,layer,pool),
                            observe=lambda stats:emit(dict(event='complete_bridge_measurement',sign=sign,layer=layer,measurement=stats)))
                        old_ref=weakref.ref(old);hz_ref=weakref.ref(hz);receipt_ref=weakref.ref(old.receipt)
                        del old,tf,runtime,hz,want
                        pool.charge('c47_toy_retirement_and_post_retirement_validation_allowance',16384)
                        gc.collect()
                        if old_ref() is not None or hz_ref() is not None or receipt_ref() is not None:
                            raise ValueError('new binding retained original state, post HZ or receipt')
                        state.validate(pool=pool)
                        result.update(sign=sign,layer_label=layer,bridge_measurement=stats,
                            original_native_component_measurements=source_measurements,source_factory_wall_s=source_wall,
                            source_factory_outside_bridge_transient_window=True,
                            complete_source_factory_to_native_publication_window_measured=False,
                            original_state_HZ_receipt_physically_retired=True,charged_diagnostic_work=pool.used-begin)
                        report['configurations'].append(result)
                        del state
            report['actual_unchanged_append_screen']=actual_lower_bound()
            if not all(report['actual_unchanged_append_screen'][n]['append_impossible_from_lower_bound']
                       for n in ('C34_fresh_append','C40_diagnostic_append')):
                raise ValueError('frozen actual metadata no longer reproduces the append blocker')
            report.update(completed=True,issued_source_bridge_component_passed=True,
                configurations_completed=len(report['configurations']),
                complete_exact_feasible_points=sum(len(r['points']) for r in report['configurations']),
                actual_native_or_original_network_input_recovery_proved=False,
                full_LIVE_or_fresh_runtime_payment_proved=False,full_benchmark_replay_performed=False)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='bound_source_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,work_parts=dict(pool.parts),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
