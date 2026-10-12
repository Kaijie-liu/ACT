"""Offline stdlib recheck of archived CPU differentials, not a second math kernel."""
import argparse
from pathlib import Path
import time

from scoped_proof.io import ROOT, load, save, sha
from scoped_source.hz_request_store import require, identity, Store
from scoped_source.check_store_endpoints import EndpointWork, check_directory, plan, evaluate_tile
from scripts.run_store_device_controls import CASES, EXPECTED, REFERENCE_SHA


def audit(root):
    root=Path(root); start=time.monotonic(); work=EndpointWork(start+300)
    freeze=load(root/'freeze.json'); calls=load(root/'calls.json')
    require(freeze['cases']==list(CASES) and freeze['reference']==REFERENCE_SHA
            and freeze['physical_cuda'] is False and freeze['budget_seconds']==300,
            'frozen source/device roster')
    require(sha(root/'reference.py')==REFERENCE_SHA,'archived old implementation')
    for name,h in freeze['sources'].items():
        require(sha(root/'implementation'/name)==h and sha(ROOT/name)==h,'source identity '+name)
    require([r['case'] for r in calls]==['unit',*CASES],'complete terminal denominator')
    for row in calls:
        name=row['case']; e=row['execution']
        require(row==load(root/f'{name}.terminal.json') and e['status']=='COMPLETED'
                and e['returncode']==0 and e['cleanup_status']=='LEADER_REAPED_NO_LIVE_GROUP'
                and not e['remaining_group']['live'] and 0<row['launch_to_receipt_seconds']<300
                and e['seconds']<=row['launch_to_receipt_seconds'] and row['budget_seconds']==300,
                'on-time complete owned call')
        path=root/'units.json' if name=='unit' else root/name/'result.json'
        require(sha(path)==row['receipt'],'parent receipt anchor')
        observation=load(root/f'{name}.observed.json')
        require(observation['terminal_sha256']==sha(root/f'{name}.terminal.json')
                and row['launch_to_receipt_seconds']<=observation['publication_observed_seconds']<300
                and observation['budget_seconds']==300,'terminal publication within total budget')
    units=load(root/'units.json')
    require(units['tests']==units['expected']==28 and len(units['outcomes'])==28
            and len({r['name'] for r in units['outcomes']})==28
            and all(r['status']=='PASS' for r in units['outcomes'])
            and units['physical_cuda'] is False and units['all_pass'],'nonempty complete test outcomes')
    results=[]
    for name in CASES:
        path=root/name; report=load(path/'result.json')
        contract=load(path/'contract.json'); anchor=identity(contract)
        require(report['source']==freeze['sources_declared'][name]==contract['request']
                and report['contract']==anchor and report['physical_cuda'] is False,'case source/property anchor')
        checked=[]
        for arm in ('old','new'):
            rec=report['arms'][arm]; stats=rec['candidate']
            value=check_directory(path/arm,expected_contract=anchor,expected_proof=rec['proof'],work=work)
            require(value==rec['result'],'independent complete obligation recheck')
            require((value['required_endpoints'],value['required'])==EXPECTED[name]
                    and value['checked_endpoints']==value['required_endpoints']
                    and stats['device']=='cpu' and stats['dtype']=='float64'
                    and stats['algorithm']=='CPU_BLOCK_PROJECTED_SUBGRADIENT_128_V1'
                    and stats['iterations_per_tile']==128,'frozen full algorithm')
            if arm=='new':
                dev=stats['device_report']
                require(dev['context_sha256'] is None and dev['hardware'] is None and dev['memory'] is None,
                        'no simulated metadata promoted to hardware observation')
                require(stats['api_seconds']<=rec['seconds'] and all(v>=0 for v in dev['completed_device_costs'].values()),
                        'inclusive device costs')
            checked.append(value)
        require(checked[0]==checked[1] and report['arms']['old']['proof']==report['arms']['new']['proof'],
                'full candidate/result differential')
        source=load(path/'source.json'); store=Store(path/'store',expected_manifest=contract['store'],
                                                   expected_request=contract['request'],work=work)
        inventory=plan(store,contract,expected_contract=anchor,declaration=source,work=work)
        residuals=[]
        for i,tasks in enumerate(inventory.tiles(8)):
            both=[]
            for arm in ('old','new'):
                records=load(path/arm/'proof'/f'{i:05d}.json')
                bounds,res=evaluate_tile(store,tasks,[r['candidate'] for r in records],work)
                both.append(dict(bounds=list(map(str,bounds)),residuals=[
                    [[j,str(v)] for j,v in sorted(row.items())] for row in res]))
            require(both[0]==both[1],'shared exact residual differential')
            residuals.append(identity(both[0]))
        require(residuals==report['residual_hashes'],'residual record binding')
        results.append(dict(case=name,status=checked[0]['status'],positive=checked[0]['positive'],
                            required=checked[0]['required'],endpoints=checked[0]['required_endpoints']))
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    require(size<=64*2**20,'frozen archive capacity'); work.tick()
    result=dict(status='PASS',issues=0,results=results,unit_tests=28,physical_cuda=False,
                bytes_before_audit=size,seconds=time.monotonic()-start,counts=work.counts,
                freeze_sha256=sha(root/'freeze.json'),calls_sha256=sha(root/'calls.json'))
    save(root/'audit.json',result); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args()
    print(audit(a.root))
