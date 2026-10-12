"""Stdlib audit of fixed dense query, including honest failure terminals."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from scoped_source.hz_request_store import compact, require
from scoped_source.verify_scaled_endpoints import verify
from scripts.run_scaled_dense_capacity import (FILES, DESIGN, PROTOCOL, SOURCE, LIMIT,
    RSS_LIMIT, AS_LIMIT, PHASES, inventory, received)
from scripts.audit_scaled_endpoint_controls import NATIVE_PHASES


def prefix_audit(root, call, *, complete):
    paths=sorted(Path(root).glob('progress-*.json'))
    stages=list(PHASES); stages.insert(5,'raw_candidate_before_exact_acceptance')
    require(len(paths)<=len(stages),'bounded progress inventory')
    prior_time=0.; prior_work=None; unreadable=[]
    for i,p in enumerate(paths):
        require(p.name==f'progress-{i+1:02d}.json','sequential progress files')
        try: value=json.loads(p.read_bytes())
        except (ValueError,UnicodeDecodeError):
            require(not complete and i==len(paths)-1,'malformed progress only final interrupted prefix')
            unreadable.append(p.name); continue
        require(value['source']==SOURCE and value['stage']==stages[i],'progress source/stage binding')
        seconds=value['elapsed_seconds']
        require(math.isfinite(seconds) and prior_time<=seconds<=call['parent_observed_seconds'],'progress elapsed cost')
        prior_time=seconds; counts=value['work']
        require(type(counts) is dict and set(counts)=={'row_visits','coefficient_visits','arithmetic'}
                and all(type(v) is int and v>=0 for v in counts.values()),'progress work fields')
        if prior_work is not None: require(all(counts[k]>=prior_work[k] for k in counts),'progress work monotone')
        prior_work=counts
        phases=value['phases']
        require(set(phases)<=set(PHASES) and all(math.isfinite(v) and v>=0 for v in phases.values())
                and sum(phases.values())<=seconds,'progress phase costs')
    if complete: require(len(paths)==len(stages) and not unreadable,'all progress snapshots')
    return dict(files=len(paths),unreadable_interrupted_tail=unreadable,last_readable_work=prior_work)


def audit(root):
    start = time.monotonic(); root = Path(root); load=lambda name:json.loads((root/name).read_bytes())
    require(not root.is_symlink(), 'root symlink')
    e=load('execution.json'); t=load('terminal.json')
    require(e['schema']=='SCALED_DENSE_CAPACITY_V3' and e['protocol']==PROTOCOL
            and e['source']==SOURCE, 'fixed execution/source')
    require(e['work_cap']==8000000 and e['width']==8 and e['archive_cap']==t['archive_cap']==LIMIT
            and e['rss_cap']==RSS_LIMIT and e['as_cap']==AS_LIMIT, 'unchanged limits')
    require(e['gpu']=='NOT_POLLED_OR_RUN' and e['real_models']==t['real_models']==t['gpu_calls']==0,
            'no changed admission')
    require(e['available_memory_bytes']>=AS_LIMIT and e['disk_free_bytes']>=2**30,'resource admission')
    require(set(e['code'])==set(FILES) and t['code']==e['code'],'complete code inventory')
    require(e['code'][DESIGN]==PROTOCOL, 'protocol code identity')
    for dirname in ('code', *(['relocated/code'] if (root/'relocated/code').is_dir() else [])):
        files=inventory(root/dirname)
        require(set(files)==set(FILES), 'archived/relocated code files')
        for name,h in e['code'].items(): require(files[name]['sha256']==h,'bound code bytes')
    files=inventory(root); files.pop('terminal.json')
    require(files==t['files'], 'all original archive members and bytes')
    total=sum(f['bytes'] for f in files.values())
    require(total==t['payload_before_terminal_bytes'],'archive total including terminal')
    full_size=total+(root/'terminal.json').stat().st_size
    require(full_size<=LIMIT or t['status']=='ARCHIVE_LIMIT_REFUSED','oversize cannot be accepted')
    require(0<=len(t['calls'])<=2 and [c['name'] for c in t['calls']]==['query','relocated'][:len(t['calls'])],
            'one query, optional offline check, no retry')
    controls=load('supervisor_controls.json')
    require(controls['hard_cap_seconds']==300 and controls['parent_observed_seconds']>0,'control cost')
    if t['calls']:
        require(controls['status']=='COMPLETE' and controls['returncode']==0
                and controls['parent_observed_seconds']<300,'controls before query')
    for c in t['calls']:
        require(c['hard_cap_seconds']==300 and c['rss_cap_bytes']==RSS_LIMIT
                and math.isfinite(c['parent_observed_seconds']) and c['parent_observed_seconds']>0
                and type(c['sampled_peak_rss_bytes']) is int and c['sampled_peak_rss_bytes']>=0,'call costs')
        if c['status']=='COMPLETE':
            require(c['returncode']==0 and c['parent_observed_seconds']<300
                    and c['sampled_peak_rss_bytes']<=RSS_LIMIT, 'no late/memory success')
        else: require(c['status'] in ('TIMEOUT','ERROR','MEMORY_LIMIT'), 'failure class')
        require((root/(c['name']+'.stdout')).is_file() and (root/(c['name']+'.stderr')).is_file(),'logs retained')
    require(controls['parent_observed_seconds']+sum(c['parent_observed_seconds'] for c in t['calls'])<=t['outer_before_terminal_seconds'],
            'all supervised cost included')
    result=None; r=None; checked=None; c=t['calls'][0] if t['calls'] else dict(status='ERROR')
    rp=root/'query/receipt.json'
    unreadable_receipt=False
    if rp.is_file():
        try: r=load('query/receipt.json')
        except (ValueError,UnicodeDecodeError):
            require(c['status']!='COMPLETE','malformed receipt cannot complete')
            unreadable_receipt=True
    complete=received(c,root/'query')=='COMPLETE_QUERY'
    prefix=prefix_audit(root/'query',c,complete=complete) if t['calls'] else None
    if complete:
        require(r['source']==SOURCE and r['real_models']==r['gpu_calls']==0,'query scope')
        require(set(r['phases'])==set(PHASES), 'full phase costs')
        n=r['native']
        require({k for k in n if k.endswith('_seconds')}==NATIVE_PHASES,'full native costs')
        require(n['device']=='cpu' and n['dtype']=='float64' and n['expanded_constraint_entries']==0
                and n['algorithm']=='CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3'
                and n['endpoints']==2 and n['candidates']==4, 'fixed native complete algorithm')
        values=[n[k] for k in NATIVE_PHASES]
        require(all(math.isfinite(v) and v>=0 for v in values) and
                sum(values)<=r['phases']['complete_candidate_and_check'],'nested costs')
        checked=verify(root/'query/bundle',r['binding'],deadline=start+300)
        require(checked['source']==SOURCE and checked['forest']==r['forest'] and
                checked['result']==r['result'] and checked['stats']==r['stats'],'independent exact replay')
        result=checked['result']
        require(result['required_endpoints']==result['checked_endpoints']==2 and result['required']==1
                and result['missing_endpoints']==0 and len(result['results'])==1
                and result['results'][0]['pair']==[0,1], 'all pair/property/endpoints')
        proof=load('query/bundle/proof.json'); tiles=proof['tiles']
        require(len(tiles)==1 and len(tiles[0])==2,'one complete bounded tile')
        require(inventory(root/'query/partial')=={'00000.json':dict(bytes=len(compact(tiles[0])),
                sha256=hashlib.sha256(compact(tiles[0])).hexdigest())},'both raw candidates before check')
    if t['status']=='COMPLETE_QUERY_AND_PORTABLE_CHECK':
        require(complete and len(t['calls'])==2 and t['calls'][1]['status']=='COMPLETE','complete accepting path')
        moved=load('relocated.stdout')
        require(all(moved[k]==checked[k] for k in ('binding','source','forest','stats','result','work')),
                'portable independent replay')
        require(inventory(root/'query/bundle')==inventory(root/'relocated/bundle'), 'relocated exact byte inventory')
        status='PASS_COMPLETE_DENSE_CAPACITY_AUDIT'
    else:
        require(t['status'] in ('INCOMPLETE_QUERY','COMPLETE_QUERY_OFFLINE_CHECK_INCOMPLETE','ERROR',
                               'ARCHIVE_LIMIT_REFUSED'),'nonaccepting terminal')
        status='PASS_NONACCEPTING_DENSE_TERMINAL_AUDIT'
    require(not {'torch','numpy','scipy','highspy'}&set(sys.modules),'stdlib audit only')
    return dict(status=status,execution_status=t['status'],result=result,receipt=r,
        offline_checked_work=None if checked is None else checked['work'],
        calls=t['calls'],controls=controls,prefix=prefix,unreadable_receipt=unreadable_receipt,
        archive_bytes=full_size,archive_cap_satisfied=full_size<=LIMIT,
        outer_before_terminal_seconds=t['outer_before_terminal_seconds'],
        seconds=time.monotonic()-start,real_models=0,gpu_calls=0)


if __name__=='__main__': print(json.dumps(audit(sys.argv[1]),indent=2))
