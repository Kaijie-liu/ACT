"""Frozen stage-one census/public attack/600-call queue, existing supervisor.

One single writer, independent per-arm loading and computation. Console chatter
is discarded by the registered logging policy; structured evidence and errors
are retained. No training, tuning, GPU, or borrowing another arm's answers.
"""
import time
ENTRY=time.monotonic()
import argparse
import gzip
import io
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save as atomic_save,sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
import genbab_cpu_ready as backend
from act.pipeline.moe.multilayer_study import coverage

CAP=128*2**20
ARTIFACT_ROOT=None
ARTIFACT_LIMIT=None
PRIOR_BYTES=0
EXECUTION_BINDING=None
OWNED_OUTPUT=False
ARMS=('act','genbab_naive','genbab_pruned')
CODE=('scripts/multilayer_stage1.py','scripts/multilayer_stage1_audit.py','act/pipeline/moe/multilayer_study.py',
      'tests/test_multilayer_stage1.py',
      'tests/test_multilayer_stage1_audit.py',
      'act/pipeline/moe/multilayer_training.py','act/back_end/moe/multilayer.py',
      'act/back_end/moe/multilayer_external.py','act/back_end/moe/multilayer_audit.py',
      'scripts/b2_observed_worker.py','scripts/genbab_cpu_ready.py',
      'scoped_proof/owned_session.py','scoped_proof/io.py','scoped_source/development_work.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def size(root):
    root=Path(root)
    if root.is_file():return root.stat().st_size
    console=root.with_suffix('.controller.log')
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())+(console.stat().st_size if console.is_file() else 0)


def room(extra):
    if ARTIFACT_ROOT is not None and size(ARTIFACT_ROOT)+extra>ARTIFACT_LIMIT:
        raise OSError('remaining stage artifact cap')


def save(path,value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    room(len(raw))
    return atomic_save(path,value)


def write_gzip(path,value):
    import zlib
    compressor=zlib.compressobj(6,zlib.DEFLATED,31)
    with path.open('xb') as stream:
        for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
            raw=compressor.compress(piece.encode())
            if raw:room(len(raw));stream.write(raw);stream.flush()
        raw=compressor.flush();room(len(raw));stream.write(raw)


def read_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def fingerprint(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode()+t.numpy().tobytes()).hexdigest()


def freeze(path):
    """No model/data forward: source and completed training records only."""
    from b1_candidate_history import audit
    history=audit()
    train=load(ROOT/'docs/multilayer_training_freeze_20261022.json')
    complete_path=Path('/data1/Kane/MOE/results/multilayer_training_20261022_r1/complete.json')
    final=load(complete_path)
    require(final['status']=='COMPLETE' and len(final['models'])==12,'all trained models')
    models=[]
    for config,row in zip(train['models'],final['models']):
        require(config['id']==row['id'] and sha(row['checkpoint'])==row['checkpoint_sha256'],'checkpoint identity')
        models.append(dict(**config,checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256']))
    control_manifest=ROOT/'docs/multilayer_stage1_freeze_20261022_r1.json'
    save(path,dict(schema='multilayer-stage1-20261022-r1',policy=POLICY,
        source_files={p:sha(ROOT/p) for p in CODE},models=models,genbab_config=backend.CONFIG,
        training_complete=dict(path=str(complete_path),sha256=sha(complete_path)),
        data_root=train['dataset_root'],test_batch_sha256=history['dataset']['raw_test_batch_sha256'],
        excluded_ranges=history['excluded_inclusive_ranges'],history_union_sha256=history['excluded_sorted_json_sha256'],
        history_sources=history['sources'],full_test_telemetry_disclosed=True,
        selection=dict(count=30,verification_count=20,order='ascending outside full named historical union',
            criterion='clean correct under all12 frozen float64 models; no route or verification selection',
            preprocessing='ToTensor float32 THEN cast float64; no normalization'),
        epsilon=2/255,census_multipliers=[.5,1.,2.],
        budgets=dict(request=300,stage=52*3600,slots=600,selection=300,
            pgd_model=100,pgd_domain=3,census_model=300,census_domain=3,artifacts=CAP),
        public_pgd=dict(steps=20,restarts=2,step_size=2/255/4,seed=20261022,
            per_domain_seconds=3,meaning='separate auxiliary cost, never shared with verifier arms'),
        census=dict(method='independent interval prefix traversal',points=5,seed=20261022,
            guarantee='numerical outer candidate count and observed point histories, not exact legal count'),
        endpoint=dict(primary='L3E8 incomplete traversal among non-replay-UNSAFE union',
            wilson_95_stop_upper_below=.1,wilson_95_enter_lower_above=.1,undefined_or_gray='REPORT',
            repeat_inputs_and_fixed_models_disclosed=True),
        execution=dict(threads=1,as_bytes=8*2**30,rss='telemetry only',gpu=False,
            stdout='discard non-evidence solver console; preserve structured returned statistics, progress and errors',
            full_order='rank ascending; config/seed catalog; arms rotated by (rank+model_position)%3',
            history_caps={'L1E8':28,'L2E8':784,'L3E8':21952,'L2E16':14400}),
        controls=dict(module='test_multilayer_stage1',batch_seconds=300,trained_models=False,
            result='/data1/Kane/MOE/results/multilayer_stage1_control_20261022_r1/complete.json',
            manifest_sha256=sha(control_manifest) if control_manifest.exists() else None),
        no_result_dependent_changes=True))


def setup(deadline):
    backend.limits(deadline)
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    return torch


def subject(item):
    from act.pipeline.moe.multilayer_training import load_checkpoint
    require(sha(item['checkpoint'])==item['checkpoint_sha256'],'same checkpoint')
    model,payload=load_checkpoint(item['checkpoint'])
    require(payload['architecture']==item['architecture'] and payload['seed']==item['seed'],'model architecture/seed')
    return model.double().eval()


def tensor_row(selection_root,sample,torch):
    path=selection_root/sample['file'];require(sha(path)==sample['sha256'],'same physical input')
    values=torch.load(path,map_location='cpu',weights_only=True)
    require({k:fingerprint(v) for k,v in values.items()}==sample['tensors'],'all tensor identities')
    return values


def select_inputs(folder,spec,deadline,torch):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    require(sha(Path(spec['data_root'])/'cifar-10-batches-py/test_batch')==spec['test_batch_sha256'],'test batch')
    # Training recipe used float32 ToTensor; preserve those exact values.
    torch.set_default_dtype(torch.float32)
    dataset=CIFAR10(spec['data_root'],train=False,download=False,transform=ToTensor())
    models=[subject(item) for item in spec['models']]
    excluded={i for a,b in spec['excluded_ranges'] for i in range(a,b+1)}
    rows=[];scanned=[]
    with torch.no_grad():
        for index in range(10000):
            if index in excluded:continue
            if time.monotonic()>=deadline:raise TimeoutError('selection deadline')
            raw,label=dataset[index];x=raw.unsqueeze(0).double()
            predictions=[int(model(x).argmax(1).item()) for model in models]
            scanned.append(dict(index=index,label=label,predictions=predictions))
            if not all(p==label for p in predictions):continue
            props=torch.zeros((9,10),dtype=torch.float64)
            for k,c in enumerate(j for j in range(10) if j!=label):props[k,label]=1.;props[k,c]=-1.
            tensors=dict(center=x,lower=(x-spec['epsilon']).clamp_min(0),upper=(x+spec['epsilon']).clamp_max(1),
                rows=props,thresholds=torch.zeros(9,dtype=torch.float64))
            file=f'input_{index}.pt';buffer=io.BytesIO();torch.save(tensors,buffer)
            raw=buffer.getvalue();room(len(raw))
            with (folder/file).open('xb') as stream:stream.write(raw)
            rows.append(dict(rank=len(rows),index=index,label=label,file=file,sha256=sha(folder/file),
                tensors={k:fingerprint(v) for k,v in tensors.items()}))
            if len(rows)==spec['selection']['count']:break
    require(len(rows)==30,'insufficient common clean-correct cohort; no replacement')
    return dict(status='COMPLETE',samples=rows,scanned=scanned,full_test_telemetry_disclosed=True)


def external_audit(result,experts,properties):
    """Disjoint mixed-radix coverage plus numerical-source links, not bound proof."""
    import itertools
    # Independently reconstruct the serialized OR-of-one-property bad-set rows.
    raw_spec=[[[[float(j==i) for j in range(properties)]],[0.]] for i in range(properties)]
    catalogs=[list(itertools.combinations(range(e),2)) for e in experts]
    counts=[len(c) for c in catalogs];expected=math.prod(counts)
    def interval(history):
        require(0<len(history)<=len(catalogs),'history depth')
        first=0
        for depth,pair in enumerate(history):
            pair=tuple(pair);require(pair in catalogs[depth],'canonical history')
            first+=catalogs[depth].index(pair)*math.prod(counts[depth+1:])
        return first,first+math.prod(counts[len(history):])
    all_regions=[];closed=[]
    bounds={tuple(map(tuple,r['prefix'])):r for r in result['prefix_bounds']}
    require(len(bounds)==len(result['prefix_bounds']),'duplicate prefix bounds')
    for row in result['exclusions']:
        a,b=interval(row['prefix']);require(b-a==row['covered'],'subtree count')
        reason=row['reason'];prefix=tuple(map(tuple,row['prefix']))
        require(reason['selected'] in prefix[-1] and reason['outside'] not in prefix[-1]
            and reason['upper'] < -reason['tolerance'],'strict exclusion direction')
        require([reason['selected'],reason['outside'],reason['upper']] in bounds[prefix[:-1]]['upper'],
            'bound-source link')
        all_regions.append((a,b));closed.append((a,b))
    for row in result['histories']:
        require(len(row['history'])==len(experts),'complete leaf')
        a,b=interval(row['history']);all_regions.append((a,b))
        require(json.loads(json.dumps(row['outcome']['actual_raw_spec']))==raw_spec,'all output properties')
        if row['outcome']['status']=='verified':closed.append((a,b))
    regions=sorted(all_regions)
    require(all(regions[i][1]<=regions[i+1][0] for i in range(len(regions)-1)),'overlap')
    require(sum(b-a for a,b in closed)==result['pruned_histories']+result['verified_histories'],'closed count')
    require(result['returned_histories']==len(result['histories']),'returned count')
    if result['status']=='POSITIVE':require(sum(b-a for a,b in closed)==expected,'positive coverage')
    return dict(status='PASS',kind='structural coverage and source links; not independent numeric reproof')


class Progress:
    """Replaceable live telemetry, not repeated copies of the proof object."""
    def __init__(self,folder,arm,expected):
        self.folder=folder;self.arm=arm;self.expected=expected;self.last=0.
        self.calls=0;self.phase=None;self.phase_time=time.monotonic();self.times={}
        self.query_identity=None;self.last_query_start=None;self.max_history=None
    def __call__(self,result,force=False):
        now=time.monotonic();self.calls+=1
        cov=coverage(result,self.arm,self.expected)
        active=cov['active']
        identity=json.dumps(active,sort_keys=True,separators=(',',':'))
        changed=identity!=self.query_identity
        if changed:
            self.query_identity=identity
            if active is not None:
                self.last_query_start=dict(**active,published_clock=now)
        if self.arm!='act' and result.get('histories'):
            seconds=result['histories'][-1].get('outcome',{}).get('seconds')
            if isinstance(seconds,(int,float)) and math.isfinite(seconds) and seconds>=0:
                self.max_history=max(self.max_history or 0.,seconds)
        phase=(cov['active'] or {}).get('phase','between_queries')
        if phase!=self.phase:
            if self.phase is not None:self.times[self.phase]=self.times.get(self.phase,0.)+now-self.phase_time
            self.phase=phase;self.phase_time=now
        # Starts/returns are mandatory; only repeated same-query heartbeats throttle.
        if not force and not changed and now-self.last<.25:return
        value=dict(clock=now,coverage=cov,publications=self.calls,phase_seconds=self.times,
            active_phase_since=self.phase_time,last_query_start=self.last_query_start,
            max_completed_history_solver_seconds=self.max_history,
            telemetry_schema='query-boundary-v2')
        temporary=self.folder/'progress.tmp'
        room(len(json.dumps(value,allow_nan=False).encode()))
        with temporary.open('w') as f:json.dump(value,f,allow_nan=False,separators=(',',':'));f.flush()
        os.replace(temporary,self.folder/'progress.json');self.last=now


def verify(folder,request,model,tensors,deadline,torch):
    from act.back_end.moe.class_separated_top1 import validate_replay
    arm=request['arm'];experts=[len(l.experts) for l in model.layers]
    expected=math.prod(math.comb(e,2) for e in experts)
    progress=Progress(folder,arm,expected)
    progress({},force=True)
    if arm=='act':
        from act.util.device_manager import initialize_device
        from act.util.options import PerformanceOptions
        from act.back_end.moe.multilayer import verify_multilayer_box
        from act.back_end.moe.multilayer_audit import audit_multilayer_result
        initialize_device('cpu','float64');PerformanceOptions.set_debug_output_file(os.devnull)
        result=verify_multilayer_box(model,**tensors,total_seconds=deadline-time.monotonic(),
            max_histories=expected,weighted_matmul=True,progress=progress,
            source_hashes={request['model']['checkpoint']:request['model']['checkpoint_sha256']})
        audit=audit_multilayer_result(result,expected_request_id=result.get('request_id'),replay=lambda _:
            result.get('witness') is not None and validate_replay(model,torch.tensor(result['witness'],dtype=torch.float64),
                tensors['lower'],tensors['upper'],tensors['rows'],tensors['thresholds']))
        # Failures before catalog creation are execution records, never positives.
        if 'request_id' in result:require(audit['status']=='PASS','ACT coverage audit '+str(audit))
        require(result['identity']['tensors']==request['sample']['tensors'],'ACT materialized identity')
    else:
        from act.back_end.moe.multilayer_external import verify_histories
        from b2_observed_worker import verify_graph
        result=verify_histories(model,**tensors,pruning=arm=='genbab_pruned',deadline=deadline,
            verify_graph=verify_graph,publish=progress)
        audit=external_audit(result,experts,len(tensors['rows']))
    replayed=False
    if result['status']=='UNSAFE_REPLAYED':
        replayed=bool(validate_replay(model,torch.tensor(result['witness'],dtype=torch.float64),
            tensors['lower'],tensors['upper'],tensors['rows'],tensors['thresholds']))
        require(replayed,'complete model replay')
    progress(result,force=True)
    return dict(status=result['status'],coverage=coverage(result,arm,expected),
        replayed=replayed,guarantee=result['evidence_grade'],audit=audit,result=result)


def worker(folder):
    global ARTIFACT_ROOT,ARTIFACT_LIMIT
    request=load(folder/'request.json');deadline=request['worker_deadline']
    ARTIFACT_ROOT=folder;ARTIFACT_LIMIT=request['artifact_limit']
    # Non-proof console is deliberately discarded; exceptions get their own file.
    fd=os.open(os.devnull,os.O_WRONLY);os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
    try:
        if 'execution_binding' in request:
            b=load(request['execution_binding']['path'],request['execution_binding']['sha256'])
            require(all(sha(ROOT/p)==h for p,h in b['source_files'].items()),'worker execution source')
        torch=setup(deadline)
        if request['kind']=='verification' and request['arm']!='act':
            environment=backend.probe()
            save(folder/'environment.json',dict(status='PASS',python=environment['python'],
                torch=environment['torch'],sources_sha256=hashlib.sha256(json.dumps(
                    environment['sources'],sort_keys=True).encode()).hexdigest(),modules=environment['modules']))
        spec=load(request['manifest'],request['manifest_sha256'])
        if request['kind']=='controls':
            import unittest
            sys.path.insert(0,str(ROOT/'tests'))
            output=io.StringIO()
            suite=unittest.defaultTestLoader.loadTestsFromName('test_multilayer_stage1')
            result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
            value=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                details=output.getvalue(),real_models=0)
        elif request['kind']=='selection':value=select_inputs(folder,spec,deadline,torch)
        else:
            model=subject(request['model'])
            selection_root=Path(request['selection_root'])
            if request['kind']=='verification':
                tensors=tensor_row(selection_root,request['sample'],torch)
                with torch.no_grad():require(int(model(tensors['center']).argmax(1))==request['sample']['label'],'clean object')
                value=verify(folder,request,model,tensors,deadline,torch)
            else:
                from act.pipeline.moe.multilayer_study import public_pgd,interval_census
                rows=[]
                for sample in request['samples']:
                    tensors=tensor_row(selection_root,sample,torch)
                    multipliers=spec['census_multipliers'] if request['kind']=='census' else [1.]
                    for multiplier in multipliers:
                        before=time.monotonic()
                        local_deadline=min(deadline-1,before+3)
                        if before>=deadline-1:
                            rows.append(dict(index=sample['index'],multiplier=multiplier,status='NOT_EXECUTED'));continue
                        if request['kind']=='pgd':
                            result=public_pgd(model,tensors,local_deadline,
                                spec['public_pgd']['seed']+10000*request['model']['seed']+sample['index'])
                        else:
                            center=tensors['center'];eps=spec['epsilon']*multiplier
                            result=interval_census(model,center,(center-eps).clamp_min(0),(center+eps).clamp_max(1),
                                local_deadline,spec['census']['seed']+sample['index'])
                        received=time.monotonic()
                        row=dict(index=sample['index'],multiplier=multiplier,result=result,seconds=received-before,
                            request_sha256=sha(folder/'request.json'),clock=received,deadline=local_deadline,
                            accepted=received<local_deadline)
                        rows.append(row);save(folder/f'domain_{len(rows):03d}.json',row)
                value=dict(status='COMPLETE',domains=rows)
        require(time.monotonic()<deadline,'on-time worker reception')
        write_gzip(folder/'outcome.json.gz',dict(request_sha256=sha(folder/'request.json'),
            value=value,clock=time.monotonic(),load_and_execution_seconds=time.monotonic()-ENTRY))
        require(time.monotonic()<deadline,'on-time serialization')
        save(folder/'published.json',dict(clock=time.monotonic(),sha256=sha(folder/'outcome.json.gz')))
    except BaseException as exc:
        # Reserved controller space remains available for the minimal error.
        ARTIFACT_LIMIT+=65536
        save(folder/'error.json',dict(kind=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),clock=time.monotonic()))
        return 1
    return 0


def invoke(folder,request,seconds,manifest,stage_deadline):
    from b1_observed_batch import session_ok
    began=time.monotonic();require(began+seconds<=stage_deadline,'insufficient remaining time; do not shorten full request')
    end=began+seconds
    folder.mkdir();(folder/'tmp').mkdir();(folder/'cache').mkdir()
    remaining=CAP-PRIOR_BYTES-size(folder.parent)
    require(remaining>256*1024,'remaining evidence/terminal capacity')
    query=dict(**request,manifest=str(manifest),manifest_sha256=sha(manifest),worker_deadline=end-6,
        artifact_limit=remaining-192*1024)
    if EXECUTION_BINDING is not None:
        query['execution_binding']=dict(path=str(EXECUTION_BINDING),sha256=sha(EXECUTION_BINDING))
    save(folder/'request.json',query)
    external=request['kind']=='verification' and request['arm']!='act'
    python=str(backend.ENVIRONMENT/'bin/python') if external else backend.ACT
    env=dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'))
    receipt=execute([python,'-I','-B',str(ROOT/'scripts/multilayer_stage1.py'),'worker',str(folder)],
        run_deadline=end-4,cleanup_deadline=end-2,env=env,rss_limit=None,development_policy=POLICY)
    save(folder/'session.json',receipt)
    value=None;status='ERROR';accepted=False
    if session_ok(receipt) and (folder/'published.json').is_file():
        publication=load(folder/'published.json')
        require(publication['sha256']==sha(folder/'outcome.json.gz'),'published outcome identity')
        outcome=read_gzip(folder/'outcome.json.gz')
        require(outcome['request_sha256']==sha(folder/'request.json'),'request/result binding')
        accepted=publication['clock']<end-6 and time.monotonic()<end
        if accepted:value=outcome['value'];status=value['status']
        else:status='TIMEOUT'
    elif (receipt.get('returncode')==124 or 'TIMEOUT' in receipt['status'] or time.monotonic()>=end-6):
        status='TIMEOUT'
    if not accepted and (folder/'progress.json').is_file():
        partial=load(folder/'progress.json')
    else:partial=None
    row=dict(status=status,accepted=accepted,through_reception_seconds=time.monotonic()-began,hard_budget=seconds,
        evidence=(None if value is None else {k:v for k,v in value.items() if k not in ('result','domains','scanned','samples')}),
        partial=partial,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        session_status=receipt['status'],returncode=receipt['returncode'])
    save(folder/'terminal.json',row)
    # The ledger adds this measured return cost, including terminal publication.
    # terminal.json deliberately reports only its pre-publication timestamp.
    row['seconds']=time.monotonic()-began
    row['terminal_sha256']=sha(folder/'terminal.json')
    require(row['seconds']<=seconds,'complete request budget exceeded')
    return row,value


def resume(folder,manifest,continuation):
    """New execution root, immutable old prefix; no query/time/artifact reset."""
    global ARTIFACT_ROOT,ARTIFACT_LIMIT,PRIOR_BYTES,EXECUTION_BINDING,OWNED_OUTPUT
    from multilayer_stage1_audit import audit,check_continuation,result_chain
    require(ROOT!=Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(),'immutable execution copy')
    require(not folder.exists(),'new continuation result root')
    spec=load(manifest);binding=load(continuation)
    parent=check_continuation(binding,manifest)
    require(all(sha(ROOT/p)==h for p,h in binding['source_files'].items()),'new execution source')
    require(binding['backend_sha256']==hashlib.sha256(json.dumps(backend.source_identity(),sort_keys=True).encode()).hexdigest(),
            'continuation backend source identity')
    require(spec['genbab_config']==backend.CONFIG,'unchanged strong numerical configuration')
    prefix=audit(parent,manifest)
    begin=binding['next_verify']
    require(prefix['full_calls']==begin and prefix['unexecuted'][0]==f'verify_{begin:03d}','exact stopped prefix')
    control=load(binding['control']['path'],binding['control']['sha256'])
    require(control['status']=='PASS','repair controls')
    PRIOR_BYTES=binding['prior_bytes'];ARTIFACT_LIMIT=CAP-PRIOR_BYTES;ARTIFACT_ROOT=folder
    EXECUTION_BINDING=continuation
    deadline=ENTRY+spec['budgets']['stage']-binding['carried_seconds']
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30));os.nice(10);folder.mkdir()
    OWNED_OUTPUT=True
    save(folder/'continuation.json',binding)
    save(folder/'start.json',dict(commit=(ROOT/'.frozen_commit').read_text().strip(),manifest_sha256=sha(manifest),
        entry=ENTRY,deadline=deadline,pid=os.getpid(),started_at=time.time(),
        execution_binding=dict(path=str(continuation),sha256=sha(continuation)),
        carried_seconds=binding['carried_seconds'],prior_bytes=PRIOR_BYTES))
    origin=result_chain(parent,manifest)[0]
    queries=load(origin/'catalog.json')['queries'];ordinal=25+begin
    for number in range(begin,len(queries)):
        require(time.monotonic()+300<=deadline,'stage execution budget: suffix unexecuted')
        terminal,value=invoke(folder/f'verify_{number:03d}',queries[number],300,manifest,deadline)
        q=queries[number]
        row=dict(id=f'verify_{number:03d}',kind='verification',**terminal,model=q['model']['id'],
            arm=q['arm'],index=q['sample']['index'],rank=q['sample']['rank'])
        ordinal+=1;save(folder/f'ledger_{ordinal:04d}.json',row)
        print(json.dumps({k:row[k] for k in ('id','status','seconds')}),flush=True)
        require(size(folder)+PRIOR_BYTES<=CAP,'cumulative artifacts')
        if row['status'] in ('ERROR','UNSUPPORTED'):raise RuntimeError('tooling failure retained: '+row['id'])
    save(folder/'complete.json',dict(status='COMPLETE',full_calls=600,auxiliary_calls=25,terminals=625,
        seconds=binding['carried_seconds']+time.monotonic()-ENTRY,bytes=PRIOR_BYTES+size(folder),
        current_full_calls=600-begin,prior_full_calls=begin))
    summary=audit(folder,manifest);require(time.monotonic()<deadline,'final audit deadline')
    save(folder/'audit.json',summary)
    save(folder/'final_accounting.json',dict(seconds_including_result_audit=binding['carried_seconds']+time.monotonic()-ENTRY,
        bytes_before_this_receipt=PRIOR_BYTES+size(folder),audit_sha256=sha(folder/'audit.json'),deadline=deadline,clock=time.monotonic()))


def freeze_resume(path,manifest,parent,control):
    """Bind a stopped execution chain; do not reset calls, time or bytes."""
    from multilayer_stage1_audit import audit,result_chain
    prefix=audit(parent,manifest)
    begin=prefix['full_calls'];roots=result_chain(parent,manifest)
    require(prefix['unexecuted'][0]==f'verify_{begin:03d}','exact prefix before freeze')
    receipt=load(control);require(receipt['status']=='PASS','compatibility control')
    codes=(*CODE,'act/pipeline/verification/torch2act.py','act/pipeline/verification/act2torch.py',
           'scripts/multilayer_lowering_control.py','tests/test_multilayer_resume.py')
    console=parent.with_suffix('.controller.log')
    stop=load(parent/'stopped.json')
    save(path,dict(schema='multilayer-stage1-continuation-r2',parent_root=str(parent),
        parent_files={str(p.relative_to(parent)):sha(p) for p in parent.rglob('*') if p.is_file()},
        parent_console=dict(path=str(console),sha256=sha(console),bytes=console.stat().st_size),
        manifest_sha256=sha(manifest),catalog_sha256=sha(roots[0]/'catalog.json'),next_verify=begin,
        source_files={p:sha(ROOT/p) for p in codes},
        backend_sha256=hashlib.sha256(json.dumps(backend.source_identity(),sort_keys=True).encode()).hexdigest(),
        control=dict(path=str(control),sha256=sha(control),seconds=receipt['seconds']),
        carried_seconds=math.ceil(stop.get('cumulative_seconds',stop['seconds'])+receipt['seconds']),
        prior_bytes=sum(size(owner) for owner in roots),
        reason=f'user-authorized tooling repair; retain all {begin} calls; unchanged models/inputs/methods/endpoints',
        errors_remain_in_denominator=True,repair_downtime_not_execution=True))


def batch(folder,manifest):
    global ARTIFACT_ROOT,ARTIFACT_LIMIT,OWNED_OUTPUT
    require(ROOT!=Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(),'immutable execution copy')
    require(not folder.exists(),'new execution directory')
    spec=load(manifest);require(all(sha(ROOT/p)==h for p,h in spec['source_files'].items()),'frozen source')
    require(spec['genbab_config']==backend.CONFIG,'fixed strong baseline config')
    control=load(spec['controls']['result'])
    control_folder=Path(spec['controls']['result']).parent/'fixed'
    control_query=load(control_folder/'request.json')
    require(control['status']=='PASS' and control_query['manifest_sha256']==spec['controls']['manifest_sha256'],
        'necessary controls bound to original freeze')
    control_spec=load(control_query['manifest'],control_query['manifest_sha256'])
    for name in ('tests/test_multilayer_stage1.py','act/pipeline/moe/multilayer_study.py',
                 'act/back_end/moe/multilayer_external.py'):
        require(control_spec['source_files'][name]==spec['source_files'][name],'unchanged controlled mechanisms')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    os.nice(10);folder.mkdir();deadline=ENTRY+spec['budgets']['stage']
    OWNED_OUTPUT=True
    ARTIFACT_ROOT=folder;ARTIFACT_LIMIT=CAP
    save(folder/'start.json',dict(commit=(ROOT/'.frozen_commit').read_text().strip(),manifest_sha256=sha(manifest),
        control_sha256=sha(spec['controls']['result']),entry=ENTRY,deadline=deadline,pid=os.getpid(),started_at=time.time()))
    ledger=[]
    def call(name,query,budget):
        require(time.monotonic()+budget<=deadline,'stage deadline: remaining suffix unexecuted, no shortened call')
        require(size(folder)<CAP,'stage artifact cap')
        terminal,value=invoke(folder/name,query,budget,manifest,deadline)
        row=dict(id=name,kind=query['kind'],**terminal)
        if 'model' in query:row['model']=query['model']['id']
        if 'arm' in query:row.update(arm=query['arm'],index=query['sample']['index'],rank=query['sample']['rank'])
        ledger.append(row);save(folder/f'ledger_{len(ledger):04d}.json',row)
        print(json.dumps({k:row[k] for k in ('id','kind','status','seconds')}),flush=True)
        require(size(folder)<=CAP,'stage artifact cap after call')
        if row['status'] in ('ERROR','UNSUPPORTED'):raise RuntimeError('tooling failure retained: '+name)
        return value
    selected=call('selection',dict(kind='selection'),spec['budgets']['selection'])
    require(selected is not None and len(selected['samples'])==30,'selection incomplete')
    samples=selected['samples'];selection_root=str(folder/'selection')
    queries=[]
    for rank,sample in enumerate(samples[:20]):
        for position,model in enumerate(spec['models']):
            arms=('genbab_pruned',) if model['architecture']['layers']==1 else ARMS
            shift=(rank+position)%len(arms);arms=arms[shift:]+arms[:shift]
            for arm in arms:queries.append(dict(kind='verification',model=model,sample=sample,
                arm=arm,selection_root=selection_root))
    require(len(queries)==600,'complete frozen catalog')
    save(folder/'catalog.json',dict(manifest_sha256=sha(manifest),queries=queries,
        selection_outcome_sha256=sha(folder/'selection/outcome.json.gz')))
    for model in spec['models']:
        call('pgd_'+model['id'],dict(kind='pgd',model=model,samples=samples[:20],selection_root=selection_root),100)
    for model in spec['models']:
        call('census_'+model['id'],dict(kind='census',model=model,samples=samples,selection_root=selection_root),300)
    for number,query in enumerate(queries):
        call(f'verify_{number:03d}',query,300)
    save(folder/'complete.json',dict(status='COMPLETE',full_calls=600,auxiliary_calls=25,
        seconds=time.monotonic()-ENTRY,bytes=size(folder),terminals=len(ledger)))
    from multilayer_stage1_audit import audit
    summary=audit(folder,manifest)
    require(time.monotonic()<deadline,'stage result audit deadline')
    save(folder/'audit.json',summary)
    save(folder/'final_accounting.json',dict(seconds_including_result_audit=time.monotonic()-ENTRY,
        bytes_before_this_receipt=size(folder),audit_sha256=sha(folder/'audit.json'),
        deadline=deadline,clock=time.monotonic()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','worker','batch','controls','resume','freeze-resume'))
    parser.add_argument('path',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--continuation',type=Path)
    parser.add_argument('--parent',type=Path);parser.add_argument('--control',type=Path)
    args=parser.parse_args()
    if args.mode=='freeze':freeze(args.path)
    elif args.mode=='freeze-resume':freeze_resume(args.path,args.manifest,args.parent,args.control)
    elif args.mode=='worker':sys.exit(worker(args.path))
    elif args.mode=='controls':
        args.path.mkdir()
        row,value=invoke(args.path/'fixed',dict(kind='controls'),295,args.manifest,ENTRY+300)
        save(args.path/'complete.json',dict(status='PASS' if value and value['status']=='PASS' else 'FAIL',
            terminal=row,control=value,seconds=time.monotonic()-ENTRY,bytes=size(args.path)))
    else:
        try:
            if args.mode=='resume':resume(args.path,args.manifest,args.continuation)
            else:batch(args.path,args.manifest)
        except BaseException as exc:
            if OWNED_OUTPUT:
                carried=load(args.continuation)['carried_seconds'] if args.mode=='resume' else 0
                save(args.path/'stopped.json',dict(kind=type(exc).__name__,error=str(exc),
                    traceback=traceback.format_exc(),seconds=time.monotonic()-ENTRY,
                    cumulative_seconds=carried+time.monotonic()-ENTRY))
            raise
