"""Recheck all saved normal endpoint calls without importing numerical code."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from fractions import Fraction as F
from scoped_source import hz_request_store as store_module
from scoped_source.hz_request_store import compact, digest, require, identity, Store
from scoped_source.check_store_endpoints import EndpointWork, check_directory, load, ALGORITHM

CASES=('source8_tile8','source8_tile1','negative3_tile8','mixed_tile8','given_large_tile8')


def recipe(path, case):
    if case=='given_large_tile8': return
    doc=load(path/'source.json',4*2**20)
    if case=='mixed_tile8':
        from scoped_source.hz_store_source_controls import fixture
        require(doc==fixture('mixed'), 'unchanged frozen mixed source')
        return
    n=3 if case=='negative3_tile8' else 8
    require(doc['lower']==['-1']*n and doc['upper']==['1']*n and len(doc['experts'])==n,
            'frozen source coordinate recipe')
    expected=[['0',[] if n==3 else [[i,'1']]] for i in range(n)]
    require(doc['router']==[dict(kind='affine',rows=expected)], 'frozen router recipe')
    for i,layers in enumerate(doc['experts']):
        require(len(layers)==3 and layers[:2]==[dict(kind='affine',rows=[['0',[[i,'1']]]]),dict(kind='relu')],
                'frozen private expert ReLU')
        output=[['2',[[0,'1']]]]+[['0',[[0,str(F(k,32))]]] for k in range(1,3 if n==3 else 10)]
        if n==3: output[-1]=['3',[[0,'1']]]
        require(layers[2]==dict(kind='affine',rows=output), 'frozen output/margin recipe')


def audit(root):
    root=Path(root); source=load(root/'source_contract.json',2**20); terms=load(root/'terminals.json',2**20)
    require(source['cases']==list(CASES) and [r['case'] for r in terms['calls']]==list(CASES), 'five frozen calls')
    for name,h in source['files'].items():
        require(digest((root/'code'/name).read_bytes())==h, 'executed code snapshot identity')
        # Bind the independent executing checker, not just its archived label.
        if name in ('scoped_source/check_store_endpoints.py','scoped_source/hz_request_store.py',
                    'scoped_source/check_hz_store_source.py'):
            require(digest((Path(__file__).resolve().parent.parent/name).read_bytes())==h, 'executing checker version')
    results=[]; reports=[]
    for row in terms['calls']:
        require(row['status']=='COMPLETE' and row['returncode']==0 and 0<row['launch_to_receipt_seconds']<=300,
                'normal call not complete within parent budget')
        path=root/row['case']; report=load(path/'result.json',4*2**20,row['result_sha256'])
        contract=load(path/'contract.json',8192,report['contract'])
        expected_classes=3 if row['case']=='negative3_tile8' else 2 if row['case']=='mixed_tile8' else 10
        expected_mode='given_hz' if row['case']=='given_large_tile8' else 'checked_source'
        require(contract['label']==0 and contract['margin']=='0' and contract['classes']==expected_classes
                and contract['mode']==expected_mode and report['case']==row['case'], 'frozen property/source protocol')
        manifest=load(path/'proof/manifest.json',2**20,report['proof'])
        require(manifest['tile_width']==(1 if row['case']=='source8_tile1' else 8), 'frozen tile width')
        # stdout is line-delimited canonical receipt, with one final newline.
        receipt=(root/f"{row['case']}.stdout").read_bytes()
        expected=compact(dict(case=row['case'],result=row['result_sha256']))+b'\n'
        prefix=(b'[ACT] Auto-detecting project root: .\n'
                b'[WARN] Gurobi license not found: modules/gurobi/gurobi.lic\n'
                b'[INFO] Please place gurobi.lic in: modules/gurobi\n')
        # Only the observed fixed import prefix, not arbitrary discarded lines.
        require(receipt in (expected,prefix+expected), 'parent receipt binding')
        recipe(path,row['case'])
        work=EndpointWork(time.monotonic()+300); before=time.monotonic()
        if row['case']=='given_large_tile8':
            from scoped_source.check_hz_store_control import check_recipe
            supplied=Store(path/'store',expected_manifest=contract['store'],expected_request=contract['request'],work=work)
            require(supplied.request==identity(dict(control='REQUEST_OWNED_HZ_STORE_V1',layout='large')), 'frozen large source identity')
            check_recipe(supplied,'large',store_module,work)
        result=check_directory(path,expected_contract=report['contract'],expected_proof=report['proof'],work=work)
        require(result==report['result'], 'independent output aggregation')
        require(report['candidate']['algorithm']==ALGORITHM and report['candidate']['device']=='cpu'
                and report['candidate']['dtype']=='float64' and report['candidate']['iterations_per_tile']==128,
                'frozen proposal configuration')
        require(report['candidate']['proposal_columns']==result['required_endpoints'], 'complete proposed endpoint roster')
        require(sum(report['counts'].values())<=source['max_visits'], 'whole-call work accounting')
        reports.append(report); results.append(dict(case=row['case'],status=result['status'],required=result['required'],
            positive=result['positive'],checked_endpoints=result['checked_endpoints'],
            offline_check_seconds=time.monotonic()-before,counts=work.counts))
    a,b=reports[:2]
    require(a['store']==b['store'] and a['contract']==b['contract'] and a['slots']==b['slots']
            and a['query_ids']==b['query_ids'], 'tile-independent original mathematical identity')
    require(all(r['result']['required']==252 and r['result']['positive']==252
                and r['result']['checked_endpoints']==504 for r in (a,b)), 'positive source full inventory')
    require(reports[2]['result']['status']=='UNKNOWN_NONPOSITIVE' and reports[2]['result']['required']==6
            and reports[2]['result']['checked_endpoints']==6, 'unsafe tie control remains unknown')
    require(reports[4]['result']['required']==252 and reports[4]['result']['checked_endpoints']==504
            and reports[4]['store_stats']['factors']==25648, 'combined given-HZ factor/query inventory')
    return dict(status='PASS',issues=0,cases=results,auditor_sha256=digest(Path(__file__).read_bytes()),
                additional_recipe_checker_sha256=digest((Path(__file__).resolve().parent.parent/'scoped_source/check_hz_store_control.py').read_bytes()),
                source_tile_bounds_equal=a['result']==b['result'],
                guarantee='finite synthetic component; not trained/native-float/production/GPU proof')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--output',required=True); a=p.parse_args()
    result=audit(a.root)
    with Path(a.output).open('xb') as stream: stream.write(compact(result))
    print(compact(result).decode())
