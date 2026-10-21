"""Read-only stage-one endpoint/coverage/cost audit; no new model or solver calls."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,sha
from act.pipeline.moe.multilayer_study import coverage,endpoint


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def percentile(values,p):
    values=sorted(values)
    if not values:return None
    k=(len(values)-1)*p;a=math.floor(k);b=math.ceil(k)
    return values[a]+(values[b]-values[a])*(k-a)


def ratio_bootstrap(left,right,unsafe,exclude):
    # Resample the common input blocks, keeping fixed model seeds together.
    indices=sorted({r['index'] for r in left+right})
    def ratio(blocks):
        a=[r['seconds'] for i in blocks for r in left if r['index']==i
            and (not exclude or (r['model'],i) not in unsafe)]
        b=[r['seconds'] for i in blocks for r in right if r['index']==i
            and (not exclude or (r['model'],i) not in unsafe)]
        return statistics.median(a)/statistics.median(b) if a and b else None
    rng=random.Random(20261022);values=[]
    for _ in range(10000):
        v=ratio(rng.choices(indices,k=len(indices)))
        if v is not None:values.append(v)
    return dict(ratio=ratio(indices),bootstrap95=[percentile(values,.025),percentile(values,.975)],
        resampling='10000 common-input blocks; three fixed seeds retained',valid_resamples=len(values))


def audit(root,manifest):
    spec=load(manifest);start=load(root/'start.json')
    assert start['manifest_sha256']==sha(manifest)
    catalog=load(root/'catalog.json');assert catalog['manifest_sha256']==sha(manifest)
    queries=catalog['queries'];assert len(queries)==600
    assert len({(q['model']['id'],q['sample']['index'],q['arm']) for q in queries})==600
    selected=read(root/'selection/outcome.json.gz')['value']['samples']
    assert catalog['selection_outcome_sha256']==sha(root/'selection/outcome.json.gz')
    assert len(selected)==30 and [s['index'] for s in selected]==sorted(s['index'] for s in selected)
    excluded={i for a,b in spec['excluded_ranges'] for i in range(a,b+1)}
    assert hashlib.sha256(json.dumps(sorted(excluded),separators=(',',':')).encode()).hexdigest()==spec['history_union_sha256']
    assert not excluded.intersection(s['index'] for s in selected)
    expected_queries=[]
    for rank,sample in enumerate(selected[:20]):
        for position,model in enumerate(spec['models']):
            arms=['genbab_pruned'] if model['architecture']['layers']==1 else ['act','genbab_naive','genbab_pruned']
            shift=(rank+position)%len(arms);arms=arms[shift:]+arms[:shift]
            for arm in arms:expected_queries.append(dict(kind='verification',model=model,sample=sample,
                arm=arm,selection_root=str(root/'selection')))
    assert queries==expected_queries
    ledgers=[load(p) for p in sorted(root.glob('ledger_*.json'))]
    assert len({r['id'] for r in ledgers})==len(ledgers)
    byid={r['id']:r for r in ledgers};rows=[];pending=[]
    witness_sources={};aux_seconds=0.
    for ledger in ledgers:
        target=root/ledger['id']
        terminal=load(target/'terminal.json')
        assert all(ledger[k]==v for k,v in terminal.items())
        assert sha(target/'terminal.json')==ledger['terminal_sha256']
        assert ledger['seconds']>=terminal['through_reception_seconds']
        assert ledger['seconds']<=terminal['hard_budget']
        assert all(sha(target/p)==h for p,h in terminal['files'].items())
        request=load(target/'request.json')
        assert request['manifest_sha256']==sha(manifest)
        assert ledger['kind']==request['kind']
        if 'model' in ledger:assert ledger['model']==request['model']['id']
        if ledger['accepted']:
            result=read(target/'outcome.json.gz')
            assert result['request_sha256']==sha(target/'request.json')
            assert ledger['status']==result['value']['status']
            publication=load(target/'published.json')
            assert publication['sha256']==sha(target/'outcome.json.gz')
            assert publication['clock']<request['worker_deadline']
        if ledger['kind'] in ('pgd','census'):
            domain_keys=set()
            for p in sorted(target.glob('domain_*.json')):
                r=load(p);key=(r['index'],r['multiplier'])
                assert key not in domain_keys;domain_keys.add(key)
                assert r['request_sha256']==sha(target/'request.json')
                assert r['index'] in {s['index'] for s in request['samples']}
                assert r['accepted']==(r['clock']<r['deadline']) and r['deadline']<request['worker_deadline']
                value=r['result']
                if ledger['kind']=='pgd' and r['accepted'] and value.get('status')=='UNSAFE_REPLAYED':
                    assert value.get('witness') is not None and value['guarantee']=='FULL_MODEL_REPLAY'
                    witness_sources.setdefault((ledger['model'],r['index']),set()).add('public_pgd')
        if ledger['kind']!='verification':aux_seconds+=ledger['seconds']
    for number,q in enumerate(queries):
        key=f'verify_{number:03d}';ledger=byid.get(key)
        if ledger is None:pending.append(key);continue
        request=load(root/key/'request.json')
        assert all(request[k]==v for k,v in q.items())
        assert (ledger['model'],ledger['index'],ledger['arm'],ledger['rank'])==(
            q['model']['id'],q['sample']['index'],q['arm'],q['sample']['rank'])
        expected=math.comb(q['model']['architecture']['experts'],2)**q['model']['architecture']['layers']
        result=read(root/key/'outcome.json.gz')['value'] if ledger['accepted'] else None
        if result is not None:
            cov=coverage(result['result'],q['arm'],expected)
            assert cov==result['coverage']
            if q['arm']=='act':
                from act.back_end.moe.multilayer_audit import audit_multilayer_result
                checked=audit_multilayer_result(result['result'],expected_request_id=result['result'].get('request_id'),
                    replay=lambda _:result['replayed'])
                assert checked['status']=='PASS',checked
            else:
                from multilayer_stage1 import external_audit
                external_audit(result['result'],[q['model']['architecture']['experts']]*q['model']['architecture']['layers'],9)
            if result['status']=='UNSAFE_REPLAYED':
                assert result['replayed'] and result['result'].get('witness') is not None
                witness_sources.setdefault((q['model']['id'],q['sample']['index']),set()).add(q['arm'])
        else:
            cov=(ledger.get('partial') or {}).get('coverage',coverage({},q['arm'],expected))
        rows.append(dict(model=q['model']['id'],index=q['sample']['index'],rank=q['sample']['rank'],arm=q['arm'],
            status=ledger['status'],seconds=ledger['seconds'],coverage=cov,
            evidence_grade=(result or {}).get('guarantee','NONE')))
    unsafe=set(witness_sources);endpoints={}
    for config in ('L3E8','L2E16'):
        cohort=[r for r in rows if r['model'].startswith(config+'_') and r['arm']=='genbab_pruned']
        # Do not calculate a premature stopping decision on a completed prefix.
        sources_complete=(not pending and all('pgd_'+m['id'] in byid for m in spec['models']))
        endpoints[config]=endpoint(cohort,unsafe) if len(cohort)==60 and sources_complete else dict(
            status='PENDING_WITNESS_SOURCE_QUEUE',completed=len(cohort),denominator=60)
    summaries={}
    for config in ('L1E8','L2E8','L3E8','L2E16'):
        for arm in ('act','genbab_naive','genbab_pruned'):
            group=[r for r in rows if r['model'].startswith(config+'_') and r['arm']==arm]
            if not group:continue
            summaries[config+'/'+arm]=dict(n=len(group),statuses=dict(Counter(r['status'] for r in group)),
                median_seconds=statistics.median(r['seconds'] for r in group),
                mean_seconds=statistics.mean(r['seconds'] for r in group),
                traversed=sum(r['coverage']['traversal_complete'] for r in group),
                proof_closed=sum(r['coverage']['proof_complete'] for r in group),
                stopped_phases=dict(Counter((r['coverage']['active'] or {}).get('phase','none') for r in group)))
    pairs={}
    for config in ('L2E8','L3E8','L2E16'):
        groups={a:{(r['model'],r['index']):r for r in rows if r['model'].startswith(config+'_') and r['arm']==a}
            for a in ('act','genbab_pruned')}
        common=groups['act'].keys()&groups['genbab_pruned'].keys()
        accepted={a:{k for k in common if groups[a][k]['status']=='POSITIVE'} for a in groups}
        solved={a:{k for k in common if groups[a][k]['status'] in ('POSITIVE','UNSAFE_REPLAYED')} for a in groups}
        pairs[config]=dict(pairs=len(common),positive_shared=sorted(accepted['act']&accepted['genbab_pruned']),
            act_only_positive=sorted(accepted['act']-accepted['genbab_pruned']),
            genbab_only_positive=sorted(accepted['genbab_pruned']-accepted['act']),
            act_only_solved=sorted(solved['act']-solved['genbab_pruned']),
            genbab_only_solved=sorted(solved['genbab_pruned']-solved['act']),
            comparison='HZ numerical acceptance versus GenBaB numerical filter; neither G3 source proof')
    left=[r for r in rows if r['model'].startswith('L3E8_') and r['arm']=='genbab_pruned']
    right=[r for r in rows if r['model'].startswith('L1E8_') and r['arm']=='genbab_pruned']
    costs={str(exclude):ratio_bootstrap(left,right,unsafe,exclude) for exclude in (False,True)} if len(left)==len(right)==60 else None
    census={}
    for model in spec['models']:
        key='census_'+model['id'];ledger=byid.get(key)
        if not ledger:continue
        values=[load(p) for p in sorted((root/key).glob('domain_*.json'))]
        for multiplier in spec['census_multipliers']:
            group=[r['result'] for r in values if r['multiplier']==multiplier]
            census[key+'/'+str(multiplier)]=dict(executed=len(group),registered=30,
                lower_median=percentile([r['observed_lower'] for r in group],.5),
                lower_p90=percentile([r['observed_lower'] for r in group],.9),
                upper_median=percentile([r['candidate_upper'] for r in group],.5),
                upper_p90=percentile([r['candidate_upper'] for r in group],.9),exact_legal_count=None,
                late_diagnostic_receipts=sum(not r['accepted'] for r in values if r['multiplier']==multiplier),
                interval_traversals_complete=sum(r['traversal_complete'] for r in group))
    complete=load(root/'complete.json') if (root/'complete.json').exists() else None
    total_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert total_bytes<=spec['budgets']['artifacts']
    stage_seconds=None
    if complete:
        assert complete['status']=='COMPLETE' and complete['full_calls']==600 and complete['terminals']==625
        assert len(ledgers)==625 and not pending and complete['auxiliary_calls']==25
        stage_seconds=complete['seconds']
        if (root/'final_accounting.json').exists():
            final=load(root/'final_accounting.json');stage_seconds=final['seconds_including_result_audit']
            assert final['audit_sha256']==sha(root/'audit.json')
        assert sum(r['seconds'] for r in rows)+aux_seconds<=stage_seconds<=spec['budgets']['stage']
    return dict(status='PASS_COMPLETE' if not pending and complete else 'PASS_PREFIX',full_calls=len(rows),
        unexecuted=pending,expected_calls=600,endpoints=endpoints,summaries=summaries,paired=pairs,
        unsafe_union=[dict(model=k[0],index=k[1],sources=sorted(v)) for k,v in sorted(witness_sources.items())],
        unsafe_source_counts=dict(Counter(source for v in witness_sources.values() for source in v)),
        unsafe_source_overlap=dict(Counter('+'.join(sorted(v)) for v in witness_sources.values())),
        costs_L3_over_L1=costs,census=census,full_seconds=sum(r['seconds'] for r in rows),
        auxiliary_seconds=aux_seconds,stage_seconds=stage_seconds,bytes=total_bytes,
        scope='record/coverage audit; online full-model replay receipts, not independent bound/source proof',
        rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('manifest',type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.root,args.manifest),sort_keys=True,separators=(',',':')))
