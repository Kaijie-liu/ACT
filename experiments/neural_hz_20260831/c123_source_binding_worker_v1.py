"""Fresh unchanged ordinary sources, full Fraction-vs-word binding comparison."""
from dataclasses import asdict
import faulthandler
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, metadata, fingerprint
from experiments.neural_hz_20260831.c120_complete_source_fixture_v1 import (
    expression,lift,_binding,_whole_maps,_bind_actual_rows as fraction_bind)
from experiments.neural_hz_20260831.c123_support_word_binding_v1 import bind_actual_rows
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c123_source_binding_20260927_v1'


class BudgetView:
    """Monotone category ceiling; all payment reaches the SAME whole pool."""
    def __init__(self,parent,limit):
        self.parent,self.limit,self.spent = parent,limit,0

    @property
    def used(self):
        return self.parent.used

    @property
    def parts(self):
        return self.parent.parts

    def charge(self,name,amount):
        if type(amount) is not int or amount<0 or self.spent+amount>self.limit:
            raise MemoryError('complete preregistered category ceiling exceeded')
        self.parent.charge(name,amount)
        self.spent += amount


def pack(rows,gauges):
    lengths = np.array([len(r['coefficients']) for r in rows],np.int64)
    return dict(indptr=np.r_[0,np.cumsum(lengths)].astype(np.int64),
        columns=np.array([c for r in rows for c,v in r['coefficients']],np.int64),
        native=np.array([v for r in rows for c,v in r['coefficients']],np.float64),
        rhs=np.array([r['rhs'] for r in rows],np.float64),
        pivots=np.array([r['pivot'] for r in rows],np.int64),gauges=np.array(gauges,np.int64))


def state_identity(state,pool):
    layout = numeric_layout(state,pool)
    digest,shallow = fingerprint(state,layout,pool)
    return dict(fingerprint=digest,numeric_bytes=layout.resident_bytes,
                numeric_entries=layout.resident_entries,python_shallow_bytes=shallow)


def build(pool,held,authentication,emit):
    limits = dict(source=64_000_000,setup=4_000_000,old=32_000_000,
                  new=24_000_000,evidence=16_000_000,ledger=24_000_000)
    budgets = {k:BudgetView(pool,v) for k,v in limits.items()}
    reports = {}
    held.update(reports=reports,authentication=authentication)
    for mode in ('dense','masked'):
        budgets['source'].charge('c123_unchanged_complete_C97_source_reserve',32_000_000)
        expr = expression(c=16,k=32,h=6)
        if mode=='masked':
            source = expr.terms[0].source
            removed = (np.indices((16,6,6)).sum(axis=0).reshape(-1)%4)==0
            source.Gc.data[removed] = 0
            source.Gc.eliminate_zeros()
            source.Gb.data[removed] = 0
            source.Gb.eliminate_zeros()
            source.c[removed] = 0
        expression_before = _binding(expr,budgets['setup'])
        keep = np.ones(expr.n_out,bool)
        direct = lift(expr,keep,enabled=True,max_work=32_000_000,max_branch_work=32_000_000)
        fields = direct['fields']
        if fields['hz'].n_bin!=1 or fields['hz'].n_ineq!=1:
            raise ValueError('unchanged ordinary nonconvex source semantics lost')
        weights,maps = _whole_maps(direct['construction']['nodes'],budgets['setup'])
        case = dict(expression=expr,direct=direct,keep=keep,weights=weights,maps=maps)
        held[mode] = case
        before = state_identity(case,budgets['ledger'])
        emit(dict(event='fresh_original_source_built',mode=mode,work=pool.used))
        start,clock = pool.used,time.monotonic()
        old = fraction_bind(fields,weights,maps,budgets['old'])
        old_work,old_wall = pool.used-start,time.monotonic()-clock
        start,clock = pool.used,time.monotonic()
        new = bind_actual_rows(fields,weights,maps,pool=budgets['new'],enabled=True)
        new_work,new_wall = pool.used-start,time.monotonic()-clock
        after = state_identity(case,budgets['ledger'])
        if before!=after or _binding(expr,budgets['setup'])!=expression_before:
            raise ValueError('binding changed original source/maps/owners/expression')
        old_report,old_rows,old_gauges,old_by_pivot = old
        new_report,new_rows,new_gauges,new_by_pivot = new
        entries = sum(len(r['coefficients']) for r in old_rows)
        new_entries = sum(len(r['coefficients']) for r in new_rows)
        budgets['evidence'].charge('c123_complete_old_new_literal_comparison_and_encoding',
            1024+24*(entries+new_entries)+128*(len(old_rows)+len(new_rows)))
        if (old_rows!=new_rows or old_gauges!=new_gauges or old_by_pivot!=new_by_pivot
            or any(new_report.get(k)!=v for k,v in old_report.items())
            or len(new_rows)!=512 or new_work>=old_work):
            raise ValueError('every original row/gauge/survival proof or real work reduction differs')
        # Retain BOTH complete literal proofs, not only their agreement flag.
        case.update(old_binding=old,new_binding=new,identity_before=before,identity_after=after)
        count = 2*(2*entries+4*len(new_rows)+1)
        budgets['evidence'].charge('c123_complete_old_new_numeric_proof_arrays',1024+16*count)
        packets = dict(old=pack(old_rows,old_gauges),new=pack(new_rows,new_gauges))
        case['proof_arrays'] = packets
        flat = {prefix+'_'+name:a for prefix,packet in packets.items() for name,a in packet.items()}
        case['archive_views'] = flat
        if sum(int(a.size) for a in flat.values())!=count:
            raise ValueError('complete packed proof evidence count differs')
        path = RUN/(mode+'_complete_binding_arrays.npz')
        with path.open('xb') as stream:
            np.savez(stream,**flat)
            stream.flush()
            os.fsync(stream.fileno())
        authentication['evidence_bytes_hashed'] += path.stat().st_size
        authentication['evidence_hash_calls'] += 1
        report = dict(mode=mode,all_original_rows=512,all_original_nnz=entries,
            fraction_work=old_work,word_work=new_work,work_saved=old_work-new_work,
            fraction_wall_s=old_wall,word_wall_s=new_wall,
            complete_literals_gauges_pivot_lookup_and_survival_equal=True,
            original_source_unchanged=True,source_binding=new_report,
            original_numeric_identity=before,original_nonconvex_binary_factors=fields['hz'].n_bin,
            original_inequality_count=fields['hz'].n_ineq,
            original_source_reserved_work=32_000_000,
            original_generation_work_upper=int(fields['report']['total_work_upper']),
            arrays_sha256=_sha256(path),source_or_LIVE_admitted=False,formal_gain=0)
        reports[mode] = report
        budgets['evidence'].charge('c123_complete_case_JSON_reporting_reservation',65536)
        _atomic_exclusive_json(RUN/(mode+'_complete_binding.json'),report)
        emit(dict(event='complete_Fraction_word_binding_equal',mode=mode,
                  fraction_work=old_work,word_work=new_work,work=pool.used))
    layout = numeric_layout(held,budgets['ledger'])
    meta = metadata(held,budgets['ledger'])
    if layout.resident_entries>64_000_000:
        raise MemoryError('complete original sources and BOTH binding proofs exceed64M')
    budgets['ledger'].charge('c123_complete_ledger_and_result_JSON_reservation',262144)
    ledger = dict(numeric=asdict(layout),known_metadata=meta)
    _atomic_exclusive_json(RUN/'complete_held_ledger.json',ledger)
    if pool.used>164_000_000:
        raise MemoryError('complete ordinary binding qualification exceeds registered bound')
    return dict(reports=reports,complete_source_count=2,ledger=ledger,
        category_work={k:v.spent for k,v in budgets.items()},category_limits=limits,
        all_original_source_rows_compared=True,mixed_F4_constructor_executed=False,
        original_network_loaded=False,archived_HZ_loaded=False,source_or_LIVE_admitted=False,
        solver_calls=0,formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze = json.loads((RUN/'preregistered.json').read_text())
    pool,held = WorkPool(256_000_000),dict(freeze=freeze)
    auth = dict(source_file_bytes_hashed=0,source_file_hash_calls=0,
                evidence_bytes_hashed=0,evidence_hash_calls=0)
    record = dict(completed=False,solver_calls=0,formal_gain=0)
    started = time.monotonic()
    fatal = (RUN/'fatal.log').open('x')
    phases = (RUN/'phases.jsonl').open('x')
    faulthandler.enable(file=fatal,all_threads=True)

    def emit(event):
        value = dict(elapsed_s=time.monotonic()-started,detail=event)
        phases.write(json.dumps(value)+'\n')
        phases.flush()
        os.fsync(phases.fileno())
        print(json.dumps(value),flush=True)

    def drift():
        changed = False
        for n,h in freeze['source_sha256'].items():
            path = EXP/n
            changed |= _sha256(path)!=h
            auth['source_file_bytes_hashed'] += path.stat().st_size
            auth['source_file_hash_calls'] += 1
        return changed

    def input_drift():
        return any(_sha256(Path(n))!=h for n,h in freeze['input_sha256'].items())

    try:
        if drift() or input_drift() or _provenance(ROOT)!=freeze['provenance']:
            raise ValueError('frozen source/production drift')
        data,stats = measured(lambda:build(pool,held,auth,emit),
                              observe=lambda m:record.update(measurement=m))
        record.update(completed=True,data=data,measurement=stats)
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable()
        fatal.close()
        phases.close()
        changed = drift()
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-started,
            source_drift=changed,input_drift=input_drift(),
            provenance_drift=_provenance(ROOT)!=freeze['provenance'],
            authentication_traffic=auth,numeric_hash_traffic_in_token_pool=False,
            all_CPU_work_in_generation_cap=False)
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='data'}),flush=True)
    if (not record['completed'] or record['source_drift'] or record['input_drift']
        or record['provenance_drift']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
