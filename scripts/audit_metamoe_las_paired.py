"""Saved terminal audit plus independent structural lAs-repair checks.

Not a reproof of CROWN bounds or source conversion. No backend/model calls.
"""
import argparse
import json
import math
from pathlib import Path
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
import audit_metamoe_checked_paired as base
import metamoe_las_paired as control


def check_events(events, digest):
    schema = None
    restores = adds = 0
    last = -math.inf
    for i, e in enumerate(events):
        if e['seq'] != i or e['request_config_sha256'] != digest or e['monotonic'] < last:
            raise ValueError('event identity/sequence')
        last = e['monotonic']
        if e['event'] == 'INSTALLED':
            if i != 0 or e['repair'] != 'disconnected_zero_branching_metadata_v1' or e['bounds_unchanged'] is not True:
                raise ValueError('repair identity')
        elif e['event'] == 'SCHEMA':
            schema = e
            s=e['scope'];parts=s['parts'];g=s['graph']
            if s['final'] not in g or g[s['final']] != [p['node'] for p in parts]:
                raise ValueError('final concat identity')
            edge=0
            for p in parts:
                if p['start']!=edge or p['end']<=edge or p['node'] not in g:
                    raise ValueError('concat partition')
                edge=p['end']
            if edge!=s['width'] or any(n not in g for ps in g.values() for n in ps):
                raise ValueError('graph/width identity')
            if not set(e['initial_zero_keys'])<=set(e['schema'])<=set(g):
                raise ValueError('schema nodes')
        elif e['event'] == 'RESTORE_DISCONNECTED_ZERO':
            if schema is None:raise ValueError('restore without schema')
            s=schema['scope'];active=set()
            for batch in schema['C']:
                for row in batch:
                    if len(row)!=s['width'] or not all(math.isfinite(v) for v in row):
                        raise ValueError('property coefficients')
                    for p in s['parts']:
                        if any(v!=0 for v in row[p['start']:p['end']]):active.add(p['node'])
            todo=list(active)
            while todo:
                node=todo.pop()
                for parent in s['graph'][node]:
                    if parent not in active:active.add(parent);todo.append(parent)
            missing=set(e['missing']);native=set(e['native_keys']);keys=set(schema['schema'])
            if (not missing or missing & native or missing | native != keys or missing & active or
                    not missing<=set(schema['initial_zero_keys']) or set(e['active_ancestors'])!=active or
                    type(e['batch']) is not int or e['batch']<=0 or
                    e['shapes']!={k:[e['batch'],*v] for k,v in schema['schema'].items()}):
                raise ValueError('unjustified zero restoration')
            restores+=1
        elif e['event']=='DOMAIN_ADD_COMPLETE':
            if schema is None or set(e['keys'])!=set(schema['schema']) or e['remaining']<0:
                raise ValueError('domain insertion schema')
            adds+=1
        else:raise ValueError('unregistered event')
    if not events or events[0]['event']!='INSTALLED':raise ValueError('missing installation')
    return {'restorations':restores,'domain_add_completions':adds,'event_count':len(events),
            'all_restored_nodes_structurally_irrelevant':True}


def audit(path,replay=None):
    start=time.monotonic()
    cfg=json.loads(path.read_text());h=sha256(path);root=Path(cfg['output_root'])
    with control.protocol_context():a=base.audit(path,replay)
    checked={}
    for row in a['rows']:
        if row['arm']!='author' or row['status']=='NOT_STARTED_AFTER_ERROR':continue
        folder=root/f"{row['id']}_author"
        f=folder/'las_repair_events.jsonl'
        if not f.exists():
            if row['status'] not in ('TIMEOUT','ERROR','RESOURCE_LIMIT','UNSAFE_REPLAYED'):
                raise ValueError('missing backend repair log')
            continue
        launch=json.loads((folder/'backend_launch.json').read_text())
        expected=[cfg['python']['author'],str(control.ROOT/'scripts/metamoe_las_backend.py'),
            '--backend',cfg['backend_repo'],'--events',str(f),'--request-config',str(path.resolve()),
            '--config',str(folder/'backend.yaml')]
        if (launch['redirected']!=expected or launch['original']!=[cfg['python']['author'],
            str(Path(cfg['backend_repo'])/'complete_verifier/abcrown.py'),'--config',str(folder/'backend.yaml')]
            or launch['config_sha256']!=h or launch['repair']!=cfg['backend_las_repair']):
            raise ValueError('wrong launch binding')
        lines=f.read_text().splitlines(keepends=True);events=[];partial=False
        for i,line in enumerate(lines):
            try:events.append(json.loads(line))
            except ValueError:
                if i!=len(lines)-1 or row['receipt']['status']=='COMPLETED':raise
                partial=True
        v=check_events(events,h) if events else {'event_count':0,'restorations':0,'domain_add_completions':0}
        checked[row['id']]={**v,'partial_tail':partial,'sha256':sha256(f)}
    status={r['id']:r['status'] for r in a['rows']}
    d=checked.get('mnist_7',{})
    gate=(cfg['protocol']=='metamoe_las_control_r1' and replay is not None and
        status=={'mnist_1':'BACKEND_POSITIVE','mnist_7':status.get('mnist_7')} and
        status.get('mnist_7') in ('BACKEND_POSITIVE','UNKNOWN','TIMEOUT') and
        d.get('restorations',0)>0 and d.get('domain_add_completions',0)>0 and not d.get('partial_tail',True))
    return {**a,'repair_events':checked,'repair_control_gate':gate,
        'trust':'Structural scope/schema checks only; original backend bounds and source lowering remain trusted.',
        'audit_seconds':time.monotonic()-start,'automatic_followup':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--replay',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=audit(a.config,a.replay);write(a.output,v)
    print(v['audit'],'control_gate',v['repair_control_gate'],v['comparisons'])
