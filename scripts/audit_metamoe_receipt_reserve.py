"""Saved-only terminal, obligation, native-cap and cost audit; no optimization.

Optional separate original-model replay has no positive-proof authority.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import metamoe_receipt_reserve as control
import audit_metamoe_checked_paired as base
from audit_metamoe_current_assignment import read, require
from audit_metamoe_csr_paired_r4 import check_receipt, check_candidate
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def native_caps(folder, fraction):
    """Independently check the execution rule, not solver soundness."""
    rows=[]
    for p in sorted((folder/'protected').glob('evaluation_*/query_*/request.json')):
        req=read(p);q=p.parent
        expected=fraction if req['scope'].get('phase') in ('expanded','contracted') else 1.
        record={'query':str(q.relative_to(folder)), 'phase':req['scope'],
            'native_seconds':None,'applied_fraction':expected,
            'budget_present':(q/'native_budget.json').exists()}
        budget=None
        for name in ('native_started.json','native_budget.json','native_result.json'):
            if not (q/name).exists():continue
            v=read(q/name)
            require(all(v[k]==req[k] for k in ('token','model_sha256','query_sha256')),'native cap identity')
            require(v['applied_fraction']==expected,'native cap scope')
            if name=='native_budget.json':
                require(v['scope']==req['scope'] and v['deadline_monotonic']==req['deadline_monotonic'], 'changed query deadline')
                remaining=v['remaining_before_budget_publication'];cap=v['proposed_native_seconds']
                require(math.isfinite(remaining) and math.isfinite(cap) and 0<cap<=remaining*expected,'proposed cap')
                budget=v
            else:
                require(v['output_budget_fraction']==fraction,'worker variant drift')
                opts=v['options'] if name=='native_started.json' else v['effective_options']
                require(set(opts)=={'presolve','time_limit','mip_rel_gap'} and opts['presolve'] is True and
                    opts['mip_rel_gap']==0 and math.isfinite(opts['time_limit']) and opts['time_limit']>0,'native options drift')
                if name=='native_result.json':
                    require(budget is not None and opts['time_limit']<=budget['proposed_native_seconds'],'missing/exceeded native cap')
                    require(math.isfinite(v['native_seconds']) and v['native_seconds']>=0,'native cost')
                    record['native_seconds']=v['native_seconds']
        rows.append(record)
    return {'queries':rows, 'worker_starts':len(list((folder/'protected').glob('evaluation_*/child_*.stdout'))),
            'native_cost_missing':sum(r['native_seconds'] is None for r in rows)}


def replay(path):
    """Original full model only; no HZ, solver, or variant result reuse."""
    began=time.monotonic();cfg=read(path);control.validate(cfg)
    root=Path(cfg['output_root']);summary=read(root/'summary.json')
    require(summary['config_sha256']==sha256(path),'replay config')
    targets=[r for r in summary['rows'] if r['status']=='UNSAFE_REPLAYED']
    rows=[]
    if targets:
        import numpy as np
        import torch
        from metamoe_paired_model import load_full
        torch.set_num_threads(2)
        model=load_full(cfg['repo'],cfg['checkpoint'],cfg['files'][cfg['checkpoint']])
        for row in targets:
            request=next(r for r in cfg['requests'] if r['id']==row['id'])
            p=root/row['variant']/f"{row['id']}_act"/'result.json'
            require(sha256(p)==row['result_sha256'],'replay result identity')
            point=torch.tensor(read(p)['witness'],dtype=torch.float64)
            with np.load(request['tensor_file'],allow_pickle=False) as data:
                lo,hi=torch.from_numpy(data['lower']),torch.from_numpy(data['upper'])
            require(point.numel()==lo.numel(),'witness shape')
            point=point.reshape_as(lo)
            require(bool(torch.isfinite(point).all()) and not bool((point<lo).any() or (point>hi).any()),'witness domain')
            with torch.no_grad():out,scores=model(point)
            require(bool(torch.isfinite(out).all() and torch.isfinite(scores).all()),'undefined model')
            label=request['label']
            margin=min(float(out[0,label]-out[0,j]) for j in range(model.total_classes) if j!=label)
            require(margin<cfg['margin'],'not a full-model violation')
            rows.append({'id':row['id'],'variant':row['variant'],'result_sha256':sha256(p),
                'label':label,'prediction':int(out.argmax(1)),'minimum_margin':margin})
    return {'audit':'INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS','config_sha256':sha256(path),
        'summary_sha256':sha256(root/'summary.json'),'rows':rows,'separate_audit_seconds':time.monotonic()-began}


def check_replay(cfg, summary_hash, h, rows, record):
    require(record['audit']=='INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS' and record['config_sha256']==h and
        record['summary_sha256']==summary_hash and math.isfinite(record['separate_audit_seconds']) and
        record['separate_audit_seconds']>=0,'replay binding/cost')
    expected={(r['id'],r['variant'],r['result_sha256']) for r in rows if r['status']=='UNSAFE_REPLAYED'}
    got=[(r['id'],r['variant'],r['result_sha256']) for r in record['rows']]
    require(set(got)==expected and len(got)==len(expected),'missing/duplicate replay')
    labels={r['id']:r['label'] for r in cfg['requests']}
    for r in record['rows']:
        require(r['label']==labels[r['id']] and type(r['prediction']) is int and 0<=r['prediction']<20 and
            math.isfinite(r['minimum_margin']) and r['minimum_margin']<cfg['margin'],'replay witness')


def audit(path, replay_path):
    began=time.monotonic();cfg=read(path);control.validate(cfg)
    root=Path(cfg['output_root']);h=sha256(path)
    summary,launch=read(root/'summary.json'),read(root/'launch.json')
    require(summary['config_sha256']==launch['config_sha256']==h and launch['protocol']==cfg['protocol'] and
        launch['roster']==cfg['roster'] and not launch['automatic_followup'] and
        [[r['id'],r['variant']] for r in summary['rows']]==cfg['roster'],'batch identity/denominator')
    rows=[];blocked=False;accounted=0.
    for row in summary['rows']:
        require(row['arm']=='act','unexpected author arm')
        if row['status']=='NOT_STARTED_AFTER_ERROR':
            require(blocked and set(row)=={'id','variant','arm','status'},'unexplained omission')
            rows.append(row);continue
        require(not blocked,'execution after error')
        rid,variant=row['id'],row['variant'];v=control.view(cfg,variant)
        req=next(r for r in cfg['requests'] if r['id']==rid)
        folder=Path(v['output_root'])/f'{rid}_act';receipt=read(folder/'receipt.json')
        killed=receipt['status']!='COMPLETED'
        require(row==read(folder/'terminal.json') and row['receipt_sha256']==sha256(folder/'receipt.json'),'terminal binding')
        check_receipt(receipt,row,cfg,control.command(cfg,path,rid,variant))
        require(all(row[k]==value for k,value in control.collect_terminal(folder,receipt).items()),'outer precedence')
        for stream in ('stdout','stderr'):
            require(receipt[stream+'_sha256']==sha256(folder/f'{stream}.txt'),'stream changed')
        require(row['artifacts']=={n:sha256(folder/n) for n in (*control.ARTIFACTS,'runtime.json','trace.jsonl')
                if (folder/n).is_file()},'artifact binding')
        require(row['evidence_artifacts']==control.paired.evidence_inventory(folder),'evidence binding')
        post=row['postflight_inventory_seconds']
        require(math.isfinite(post) and post>=0,'postflight cost')
        accounted+=receipt['total_with_postflight_seconds']+post
        result=read(folder/'result.json') if row['result_sha256'] and not row['result_parse_error'] else None
        if result:
            require(result['config_sha256']==h and result['request_id']==rid and result['arm']=='act' and
                result['label']==req['label'] and result['tensor_file_sha256']==cfg['files'][req['tensor_file']] and
                0<=result['worker_seconds']<=row['seconds'],'candidate identity/cost')
            if not killed:check_candidate(result,'act')
        if (folder/'runtime.json').exists():
            require(read(folder/'runtime.json')['output_budget_fraction']==control.VARIANTS[variant],'runtime fraction')
        with patch.object(base.control,'identity',control.identity):
            details=base.check_act(folder,v,path,req,result,row,killed)
        caps=native_caps(folder,control.VARIANTS[variant])
        rows.append({**row,'details':details,'native_caps':caps,
            'grade':result.get('evidence_grade','NONE') if result and not killed else 'NONE'})
        blocked=row['status'] in ('ERROR','SOURCE_CHANGED')
    cost=read(root/'batch_cost.json')
    require(cost['config_sha256']==h and cost['charged_request_seconds']==sum(r.get('seconds',0.) for r in rows) and
        math.isfinite(cost['batch_wall_through_summary_seconds']) and cost['batch_wall_through_summary_seconds']>=accounted,'batch accounting')
    check_replay(cfg,sha256(root/'summary.json'),h,rows,read(replay_path))
    sets={v:{'positive':{r['id'] for r in rows if r['variant']==v and r['status']=='POSITIVE'},
             'solved':{r['id'] for r in rows if r['variant']==v and r['status'] in ('POSITIVE','UNSAFE_REPLAYED')}} for v in control.VARIANTS}
    delta={k:{'gained':sorted(sets['receipt_reserve'][k]-sets['full_native'][k]),
              'lost':sorted(sets['full_native'][k]-sets['receipt_reserve'][k])} for k in ('positive','solved')}
    return {'audit':'PASS','issues':0,'config_sha256':h,'summary_sha256':sha256(root/'summary.json'),
        'rows':rows,'counts':{v:dict(Counter(r['status'] for r in rows if r['variant']==v)) for v in control.VARIANTS},
        'paired':delta,'cost':cost,'separate_audit_seconds':time.monotonic()-began,
        'trust':'HZ numerical policy and structural audit; not independent source-complete positive proof'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-only',action='store_true');p.add_argument('--replay',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.replay_only:
        if a.replay:p.error('replay modes are separate')
        result=replay(a.config)
    else:
        if not a.replay:p.error('separate replay record required (including empty ledger)')
        result=audit(a.config,a.replay)
    write(a.output,result)
