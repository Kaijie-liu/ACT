"""Fixed complete algebra/resource diagnostic; no actual network or solver."""
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c46_ordered_fixture_v1 import build
from experiments.neural_hz_20260831.c46_half_lineage_extension_v1 import extend,_seal_hz
from experiments.neural_hz_20260831.test_c46_half_lineage_extension_v1 import residuals
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c46_witness_composition_20260911_v1'


def configuration(count,sign,unit_sign,alias_sign,pool):
    offset=F(1,8)
    pre,old,new,lin,words,runs,degrees,writer=build(count,sign,alias_sign,pool=pool,unit_sign=unit_sign,offset=float(offset))
    originals=[_seal_hz(hz,pool) for hz in (pre,old,new)];paired=lin.seal
    operations=4*new.n_cont+4*len(lin.eq_roots)
    for hz in (pre,old,new):
        operations+=hz.n_eq+hz.n_ineq+sum(getattr(hz,n).nnz for n in ('Ac','Ab','Auc','Aub'))
    points=[]
    for coordinate in (-.125,0.,.125):
        pool.charge('c46_independent_full_point_oracle',128*int(operations)+1024)
        x=F(coordinate);point=np.zeros(new.n_cont,np.float64);point[:count]=coordinate
        for i in range(count):point[count+4*i+2]=-2*unit_sign*sign*coordinate
        binary=[F(-1 if i%2 else 1) for i in range(count)]
        original_point=[F(float(v)) for v in point]
        if any(v!=0 for v in residuals(new,original_point,binary)) or any(v>0 for v in residuals(new,original_point,binary,True)):
            raise ValueError('independent new-point predicate check failed')
        got,proof=extend(new,lin,words,runs,degrees,originals[1],paired,point,input_n_cont=count,pool=pool,enabled=True)
        half_restored=list(original_point)
        for i in range(count):half_restored[count+4*i+1]=unit_sign*sign*x/2
        reference=lin.reconstruct_fraction(old,half_restored,pool=pool)
        if got!=reference or got[:count]!=[x]*count or any(abs(v)>1 for v in got):
            raise ValueError('complete old-reader or input/box comparison failed')
        for hz in (pre,old):
            if any(v!=0 for v in residuals(hz,got,binary)) or any(v>0 for v in residuals(hz,got,binary,True)):
                raise ValueError('independent pre/post source predicate check failed')
        for i in range(count):
            u,z,_,a=(count+4*i+j for j in range(4))
            if got[u]!=offset+sign*x/2 or got[z]!=unit_sign*sign*x/2 or got[a]!=alias_sign*got[u]/2:
                raise ValueError('independent symbolic unit/half/alias identity failed')
        if proof['deleted_definition_reads']!=count:raise ValueError('old unit reader did not cross every second deletion')
        digest=hashlib.sha256()
        for value in got:digest.update(f'{value.numerator}/{value.denominator};'.encode())
        points.append(dict(coordinate=coordinate,full_exact_vector_sha256=digest.hexdigest(),
            exact_pre_post_new_predicates_passed=True,complete_original_reader_equal=True,**proof))
    if [_seal_hz(hz,pool) for hz in (pre,old,new)]!=originals or lin.seal!=paired:
        raise ValueError('synthetic source changed')
    return dict(blocks=count,sign=sign,unit_sign=unit_sign,alias_sign=alias_sign,pivot=1.,offset=float(offset),
        pre_post_new_sha256=originals,paired_lineage_sha256=paired,points=points,
        input_continuous_factors=count,all_continuous_factors=new.n_cont,binary_factors=new.n_bin,
        old_post_EQ_rows=old.n_eq,new_EQ_rows=new.n_eq,
        half_predicate_nnz_delta=writer['actual_predicate_nnz_delta'],
        synthetic_new_HZ_constructed=True,actual_benchmark_HZ_constructed=False,
        trusted_source_association_still_requires_independent_runtime_binding=True)


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered witness diagnostic')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);pool.charge('c46_prepaid_evidence_events',16384);started=time.monotonic()
    report=dict(completed=False,formal_gain=0,actual_source_loaded=False,actual_benchmark_HZ_constructed=False,
        solver_executed=False,native_admitted=False,configurations=[])
    events=0
    with (DIRECTORY/'events.jsonl').open('x') as f:
        def emit(value):
            nonlocal events
            events+=1
            if events>128:raise ValueError('unpaid additional witness evidence event')
            f.write(json.dumps(dict(worker_elapsed_s=time.monotonic()-started,**value),sort_keys=True)+'\n');f.flush()
        try:
            freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('complete source freeze drift')
            for count in (1,16,128):
                for sign,unit_sign,alias_sign in product((-1,1),repeat=3):
                    begin=pool.used
                    result,stats=measured(lambda:configuration(count,sign,unit_sign,alias_sign,pool),
                        observe=lambda m:emit(dict(event='complete_measured_composition',blocks=count,
                            sign=sign,unit_sign=unit_sign,alias_sign=alias_sign,measurement=m)))
                    result.update(measurement=stats,charged_work=pool.used-begin)
                    report['configurations'].append(result)
            report.update(completed=True,conditional_witness_component_passed=True,
                configurations_completed=len(report['configurations']),
                complete_exact_point_checks=sum(len(c['points']) for c in report['configurations']),
                full_HZ_LIVE_gate_proved=False,whole_source_or_native_payment_proved=False,
                native_coordinate_input_recovery_wired=False,concrete_network_validation_performed=False)
        except Exception as exc:
            report['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='witness_composition_rejected',**report['failure']))
        finally:
            report.update(wall_s=time.monotonic()-started,whole_diagnostic_work=pool.used,work_parts=dict(pool.parts),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY/'result.json',report)
            print(json.dumps({k:report[k] for k in ('completed','wall_s','whole_diagnostic_work','formal_gain')}),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
