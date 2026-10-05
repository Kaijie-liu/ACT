"""Bounded full-population original ViT source query; never a verifier.

All project imports in main follow manifest authentication.  Pure bridge
helpers are also exercised in the single frozen mathematical test process.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d129_source_container_compat_20261002_v1'
CAP, MODEL_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, RESERVE, ENTRY_CAP = 1024**3, 65536, 64_000_000


class Bootstrap:
    def __init__(self):
        self.used = 0

    def charge(self, amount):
        if type(amount) is not int or amount < 0 or amount > MODEL_CAP-self.used:
            raise ValueError('pre-import work cap')
        self.used += amount


def read_bytes(path, budget, digest, maximum=8_000_000):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 <= path.stat().st_size <= maximum:
        raise ValueError('ordinary bounded authenticated file required')
    budget.charge(4096+path.stat().st_size)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('file identity mismatch: '+str(path))
    return raw


def metadata(raw, budget):
    def integer(text):
        budget.charge(3+len(text))
        if len(text) > 160:
            raise ValueError('integer width')
        value = int(text)
        if abs(value).bit_length() > 512:
            raise ValueError('integer width')
        return value
    def floating(text):
        budget.charge(3+len(text))
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite metadata')
        return value
    def pairs(items):
        budget.charge(5+5*len(items))
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate metadata key')
            result[key] = value
        return result
    def invalid(text):
        raise ValueError('nonstandard JSON constant')
    return json.loads(raw, parse_int=integer, parse_float=floating,
                      object_pairs_hook=pairs, parse_constant=invalid)


def authenticate(budget):
    path = RUN/'preregistered.json'
    digest = os.environ.get('NEURAL_HZ_SOURCE_MANIFEST_SHA256')
    if (os.environ.get('NEURAL_HZ_SOURCE_MANIFEST') != str(path)
            or type(digest) is not str or len(digest) != 64):
        raise ValueError('supervisor manifest identity required')
    raw = read_bytes(path, budget, digest)
    frozen = metadata(raw, budget)
    caps = dict(whole_work_cap=CAP, branch_work_cap=MODEL_CAP,
        evidence_prepaid_work=EVIDENCE_CAP, retained_entry_cap=ENTRY_CAP,
        rational_bit_cap=512, worker_wall_cap_s=240, summary_reserve_bytes=RESERVE,
        host_memory_cap_bytes=MEMORY_CAP, address_space_bytes=16*1024**3)
    if any(type(frozen.get(k)) is not int or frozen[k] != v for k,v in caps.items()):
        raise ValueError('source limits changed')
    if frozen.get('schema') != 'd129_source_container_compat_v1':
        raise ValueError('source schema')
    identities = frozen['source_sha256']
    imports = (
        HERE/'source_worker.py', HERE/'source_binding.py', HERE/'fast_query.py',
        HERE.parent/'d127_native_attention_component_20261002/attention.py',
        HERE.parent/'d127_native_attention_component_20261002/exp_interval.py',
        HERE.parent/'d119_curvature_component_20261002/curvature_transfer.py',
        HERE.parent/'d112_shared_endpoint_forward_20261002/endpoint_forward.py',
        HERE.parent/'d015_source_shielding_20260928/shield_kernel_v1.py',
        HERE.parent/'d015_source_shielding_20260928/source_packet_v1.py',
        HERE.parent/'d015_batch_binding_20260928_v2/census_worker_v2.py',
        HERE.parent/'d015_batch_binding_20260928_v2/source_binding_v2.py',
        HERE.parent/'d025_interval_capacity_20260930/evidence.py',
    )
    checked = {}
    for module in imports:
        parent = module.parent
        while parent != ROOT:
            budget.charge(8)
            init = parent/'__init__.py'
            if init.exists() and str(init) not in checked:
                read_bytes(init, budget, identities[str(init)])
                checked[str(init)] = identities[str(init)]
            parent = parent.parent
        read_bytes(module, budget, identities[str(module)])
        checked[str(module)] = identities[str(module)]
    spec = metadata(read_bytes(HERE/'inputs.json', budget,
                               identities[str(HERE/'inputs.json')]), budget)
    if spec != frozen.get('vit_source_population'):
        raise ValueError('source population changed')
    csv_raw = read_bytes(spec['instances_path'], budget, spec['instances_sha256'])
    first = {}
    for row, line in enumerate(csv_raw.decode('ascii').splitlines(), 1):
        budget.charge(8+len(line))
        columns = line.split(',')
        if len(columns) != 3:
            raise ValueError('official instance row shape')
        first.setdefault(columns[0], (row, columns[1]))
    if len(first) != 2 or spec['expected_rows'] != [96,96] or spec['expected_roots'] != 1152:
        raise ValueError('complete official model population differs')
    for source in spec['selected']:
        if first['onnx/'+source['model']+'.onnx'] != (source['row'],'vnnlib/'+source['spec']+'.vnnlib'):
            raise ValueError('not first official property')
        for name in ('model','spec'):
            if frozen['input_sha256'].get(source[name+'_path']) != source[name+'_sha256']:
                raise ValueError('unbound original input')
        if identities.get(source['graph_path']) != source['graph_sha256']:
            raise ValueError('unbound graph inventory')
    return frozen, digest, spec, (raw, frozen, checked, csv_raw)


def bridge_polygon(box, score, value, score_error, value_error, budget):
    """Exact center image plus certified coefficient-error rectangle.

    The two query-only columns do not replace native sources or phase bits.
    No model authentication or coefficient-bound proof is inferred here.
    """
    from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import attention as at
    ei = at.ei
    if (type(box) is not tuple or not 0 < len(box) <= 3072
            or type(score) is not at.Form or type(value) is not at.Form
            or type(score_error) is not F or type(value_error) is not F
            or score_error < 0 or value_error < 0):
        raise ValueError('complete source box and nonnegative exact errors required')
    for x in (score_error,value_error):
        ei.rational(x)
    # The complete original immutable box was bound upstream. This exact
    # temporary image need visit only columns actually used by either form.
    # Both forms use one original-ID map; no native column is changed/deleted.
    terms=score.terms+value.terms
    budget.charge(32+20*len(terms))
    for form in (score,value):
        if any(form.terms[j-1][0]>=form.terms[j][0] for j in range(1,len(form.terms))):
            raise ValueError('canonical strictly ordered original source terms required')
    original=[]
    a=b=0
    while a<len(score.terms) or b<len(value.terms):
        if b==len(value.terms) or (a<len(score.terms) and score.terms[a][0]<value.terms[b][0]):
            original.append(score.terms[a][0]); a+=1
        elif a==len(score.terms) or value.terms[b][0]<score.terms[a][0]:
            original.append(value.terms[b][0]); b+=1
        else:
            original.append(score.terms[a][0]); a+=1; b+=1
    original_columns=tuple(original)
    if any(type(i) is not int or not 0<=i<len(box) for i in original_columns):
        raise ValueError('unbound original source column')
    mapping={i:j for j,i in enumerate(original_columns)}
    n=len(original_columns)
    bounds=tuple(box[i] for i in original_columns)
    budget.charge(16+8*n)
    geometry = at.System(bounds+((F(-1),F(1)),)*2, (), (), (), 0)
    sf = at.Form(score.bias, tuple((mapping[i],a) for i,a in score.terms)+((n,score_error),))
    vf = at.Form(value.bias, tuple((mapping[i],a) for i,a in value.terms)+((n+1,value_error),))
    columns = tuple(sorted({i for i,_ in sf.terms+vf.terms}))
    budget.charge(len(columns)*max(1,len(columns).bit_length()))
    polygon = at._polygon(geometry, sf, vf, columns, budget)
    return polygon, 16*n+32*len(columns)+16*len(polygon)+256


def negative_form(form, budget):
    from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import attention as at
    budget.charge(1+len(form.terms))
    return at.Form(-form.bias,tuple((i,-a) for i,a in form.terms))


def composed_bounds(constant, positive, negative, budget):
    """Use only support UPPER endpoints, never F/H root lo as a witness."""
    from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import exp_interval as ei
    if (type(constant) is not tuple or len(constant) != 2
            or constant[0] > constant[1] or len(positive) != len(negative)
            or not positive):
        raise ValueError('bound composition shape')
    lo, hi = map(ei.rational,constant)
    rlo, rhi = lo, hi
    for upper, neg in zip(positive,negative):
        hi = ei.add(hi,upper['hi'],budget)
        lo = ei.sub(lo,neg['hi'],budget)
        rhi = ei.add(rhi,upper['rectangle_hi'],budget)
        rlo = ei.sub(rlo,neg['rectangle_hi'],budget)
    if lo > hi or rlo > rhi:
        raise ValueError('inconsistent certified interval')
    return dict(interval=(lo,hi), rectangle_interval=(rlo,rhi),
        relu_interval=(max(F(0),lo),max(F(0),hi)),
        rectangle_relu_interval=(max(F(0),rlo),max(F(0),rhi)),
        lower_improved=lo>rlo, upper_improved=hi<rhi,
        relu_lower_improved=max(F(0),lo)>max(F(0),rlo),
        relu_upper_improved=max(F(0),hi)<max(F(0),rhi),
        network_certified=False, validated_adv=False)


def _time_limit(signum, frame):
    raise TimeoutError('source internal 235 second stop for terminal evidence')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    started = time.monotonic()
    initial = next((int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:')), None)
    bootstrap = Bootstrap()
    budget = meter = model_start = query_budget = None
    qpaid = branch_used = entries = 0
    report = dict(source_census_completed=False,source_census_qualified=False,
        memory_gate_passed=False,models=[],rows=0,roots=0,formal_gain=0,
        new_benchmark_solves=0,diagnostic_solver_calls=0,model_forward_calls=0,
        native_HZ_admitted=False,actual_model_binding_qualified=False,
        actual_phase_column_binding_verified=False,gpu_computation_completed=False,
        complete_physical_qualification=False,binding_mathematical_only=True)
    tracemalloc.start()
    signal.signal(signal.SIGALRM,_time_limit)
    signal.setitimer(signal.ITIMER_REAL,235)
    try:
        if (initial is None or len(os.sched_getaffinity(0)) != 1
                or resource.getrlimit(resource.RLIMIT_AS) != (16*1024**3,)*2
                or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not __debug__
                or not sys.dont_write_bytecode
                or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
                or os.environ.get('TMPDIR') != str(RUN/'tmp')):
            raise ValueError('CPU1 AS16GiB isolated worker contract')
        frozen,digest,spec,authroots = authenticate(bootstrap)
        report.update(manifest_sha256=digest,selected_sources=spec['selected'])
        sys.path.insert(0,str(ROOT))
        from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
        from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import exp_interval as ei
        from experiments.neural_hz_20260831.definition_first_20260928.d129_source_container_compat_20261002 import source_binding as sb, fast_query as fq
        budget = k.WorkBudget(enabled=True)
        budget.charge(bootstrap.used+EVIDENCE_CAP+RESERVE)
        meter = evidence.Meter(EVIDENCE_CAP)
        query_budget = ei.Budget()
        for model_index,source in enumerate(spec['selected']):
            model_start = budget.used
            budget.limit = min(CAP,model_start+MODEL_CAP)
            report['current_model'] = source['model']
            report['current_stage'] = 'binding'
            raw = read_bytes(source['model_path'],budget,source['model_sha256'])
            prop = read_bytes(source['spec_path'],budget,source['spec_sha256'])
            graph = read_bytes(source['graph_path'],budget,source['graph_sha256'])
            binding = sb.bind(raw,prop,graph,model_sha256=source['model_sha256'],
                spec_sha256=source['spec_sha256'],enabled=True,budget=budget)
            query_box=binding['input_box']
            budget.charge(8192)
            print(json.dumps(dict(event='binding_complete',model=source['model'],work=budget.used)),flush=True)
            references=[]
            counts=dict(rows=0,roots=0,lower_improved=0,upper_improved=0,
                        relu_lower_improved=0,relu_upper_improved=0)
            for j in range(96):
                report.update(current_stage='direction',current_direction=j)
                direction = sb.direction(binding,j,budget=budget)
                query_budget.max_work = query_budget.work+budget.limit-budget.used
                if not 0 < query_budget.max_work <= ei.MAX_WORK:
                    raise ValueError('shared query budget bridge')
                positive,negative=[],[]
                temporary=0
                for head in direction['heads']:
                    polygons=[]
                    head_storage=0
                    for token in head['tokens']:
                        polygon,storage=bridge_polygon(query_box,token['score'],token['value'],
                            token['score_error'],token['value_error'],query_budget)
                        polygons.append(polygon)
                        head_storage+=storage
                    temporary=max(temporary,head_storage)
                    for sign,destination in ((1,positive),(-1,negative)):
                        tokens=[]
                        for polygon in polygons:
                            query_budget.charge(1+2*len(polygon))
                            signed=polygon if sign==1 else tuple((s,-v) for s,v in polygon)
                            tokens.append(fq.prepare_polygon(signed,query_budget))
                        answer=fq.bound(tuple(tokens),query_budget,steps=12)
                        temporary=max(temporary,head_storage+answer['entry_upper'])
                        destination.append(answer)
                        counts['roots']+=1
                        report['roots']+=1
                composed=composed_bounds(direction['constant'],positive,negative,query_budget)
                budget.charge(query_budget.work-qpaid)
                qpaid=query_budget.work
                row=dict(direction=j,positive=positive,negative=negative,**composed)
                filename='row_'+str(model_index)+'_'+str(j)+'.json'
                written=evidence.write_evidence(RUN/filename,row,meter,{})
                references.append(dict(path=filename,**written))
                counts['rows']+=1
                report['rows']+=1
                for key in ('lower_improved','upper_improved','relu_lower_improved','relu_upper_improved'):
                    counts[key]+=int(composed[key])
                entries=max(entries,int(binding['entry_upper'])+temporary+16384)
                if entries>ENTRY_CAP:
                    raise ValueError('retained plus temporary entry cap')
                budget.charge(8192)
                print(json.dumps(dict(event='direction_complete',model=source['model'],direction=j,
                    rows=report['rows'],roots=report['roots'],whole_work_used=budget.used)),flush=True)
            roots=dict(binding=binding,raw=raw,property=prop,graph=graph,
                authentication=authroots,rows=references,report=report)
            ledger=evidence.bounded_ledger(roots,meter)
            entries=max(entries,ledger['retained_entries']+temporary+16384)
            if entries>ENTRY_CAP:
                raise ValueError('physical-root entry cap')
            payload=dict(source=source,binding=binding,rows=references,summary=counts,
                         model_native_qualified=False)
            name='complete_'+str(model_index)+'.json'
            written=evidence.write_evidence(RUN/name,payload,meter,{})
            report['models'].append(dict(model=source['model'],evidence_file=name,
                evidence_sha256=written['sha256'],evidence_bytes=written['bytes'],
                summary=counts,held_ledger=ledger,retained_entry_upper=entries))
            sb.release(binding)
            # No previous model bank/geometry survives into the next model's
            # construction outside the report/receipt roots being measured.
            del raw,prop,graph,binding,query_box,direction,positive,negative,tokens
            del polygons,polygon,signed,token,head,row,roots,payload,ledger,references
            branch_used=max(branch_used,budget.used-model_start)
            model_start=None
        budget.limit=CAP
        report['terminal_ledger']=evidence.bounded_ledger((authroots,report,vars(budget),vars(meter)),meter)
        entries=max(entries,report['terminal_ledger']['retained_entries'])
        report['source_census_completed']=(len(report['models'])==2 and report['rows']==192 and report['roots']==1152)
    except Exception as exc:
        report['failure']=dict(type=type(exc).__name__,reason=str(exc)[:4096])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if budget is not None and query_budget is not None and query_budget.work>qpaid:
            try:
                budget.charge(query_budget.work-qpaid)
            except Exception as exc:
                report.setdefault('failure',dict(type=type(exc).__name__,reason=str(exc)))
        if budget is not None and model_start is not None:
            branch_used=max(branch_used,budget.used-model_start)
        _,peak=tracemalloc.get_traced_memory()
        meta=tracemalloc.get_tracemalloc_memory()
        growth=max(0,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-initial) if initial is not None else None
        wall=time.monotonic()-started
        report.update(wall_s=wall,initial_rss_bytes=initial,rss_highwater_growth_bytes=growth,
            traced_peak_bytes=peak,tracer_metadata_bytes=meta,final_summary_reserve_bytes=RESERVE,
            summary_reserve_bytes=RESERVE,whole_work_used=budget.used if budget else bootstrap.used,
            preimport_work_used=bootstrap.used,branch_work_used=branch_used,
            evidence_work_used=meter.used if meter else 0,retained_entries=entries,
            actual_cpu_affinity=list(os.sched_getaffinity(0)),address_space_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
            scope='all 192 first-CLS preactivations on two first official properties')
        report['memory_gate_passed']=(report['source_census_completed'] and 'failure' not in report
            and growth is not None and growth+RESERVE<=MEMORY_CAP and peak+meta+RESERVE<=MEMORY_CAP
            and entries<=ENTRY_CAP and wall<=240 and report['whole_work_used']<=CAP
            and branch_used<=MODEL_CAP and report['evidence_work_used']<=EVIDENCE_CAP)
        report['source_census_qualified']=report['source_census_completed'] and report['memory_gate_passed']
        data=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        if len(data)>RESERVE:
            raise ValueError('terminal summary reserve')
        with (RUN/'diagnostic.json').open('xb') as stream:
            stream.write(data)
        print(data.decode(),end='',flush=True)
        tracemalloc.stop()
    return 0 if report['source_census_qualified'] else 1


if __name__=='__main__':
    raise SystemExit(main())
