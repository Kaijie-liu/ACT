"""Read-only re-audit of fixed factored H2 calls; no proposal/model execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import PYTHON,load,sha
from scoped_source.factored_supervised import audit,POSITIVE,specification
from source_enclosure.format import identity


def roster():
    """Independent fixed call inventory, not inferred from surviving files."""
    result={}
    for intake,case in [('declared',v) for v in ('weighted_sign','tied_partial_reuse','unsafe_tied','unresolved_sign')]+[('model','weighted_sign')]:
        for mode in ('endpoints','mccormick'):
            status='UNKNOWN_NONPOSITIVE' if case=='unsafe_tied' or (case,mode)==('weighted_sign','mccormick') else POSITIVE
            result[intake+'-'+case+'-'+mode]=(specification(case,mode,intake),status,30)
    for fault in ('missing_certificate','missing_both','missing_endpoint','omit_property','omit_pair','wrong_mode',
                  'proposal_exception','partial_output','exception_after_bundle','wrong_invocation'):
        status='UNKNOWN_MISSING_EVIDENCE' if fault in ('missing_certificate','missing_both','proposal_exception') else 'ERROR'
        result['fault-'+fault]=(specification(control=fault),status,30)
    for fault in ('produce_delay','chunk_delay','construct_delay','proposal_delay','serialization_delay',
                  'check_delay','receive_delay','late_publish'):
        result['deadline-'+fault]=(specification(control=fault),'TIMEOUT',6)
    for fault in ('mutate_model','mutate_input','capture_exception','capture_delay'):
        result['model-'+fault]=(specification(intake='model',control=fault),'TIMEOUT' if fault=='capture_delay' else 'ERROR',6 if fault=='capture_delay' else 30)
    result['rebound-checker']=(specification(control='rebind_checker_context'),'ERROR',30)
    result['replaced-stdout']=(specification(mode='mccormick',control='rewrite_check_stdout'),'ERROR',30)
    for fault in ('descendant','memory'):
        result['resource-'+fault]=(specification(control=fault),'RESOURCE_LIMIT' if fault=='memory' else 'ERROR',6)
    result['final-publication']=(specification(),'TIMEOUT',5)
    result['prelaunch']=(specification(),'TIMEOUT',1e-9)
    return result


def declared_calls(root):
    expected=roster()
    for name,(spec,status,budget) in expected.items():
        path=root/name
        for file in ('caller_observation.json','spec.json','terminal.json','invocation.json','receipt.json','finish.json'):
            if not (path/file).is_file(): raise ValueError('missing registered call artifact: '+name+'/'+file)
        call=load(path/'caller_observation.json'); actual=load(path/'spec.json'); inv=load(path/'invocation.json')
        if (call['root']!=str(path) or call['status']!=status or actual!=spec or inv['budget']!=budget):
            raise ValueError('registered call identity/outcome: '+name)
    for path in root.iterdir():
        observation=path/'caller_observation.json'
        if path.is_dir() and observation.exists() and path.name not in expected:
            # Mutation copies retain the original root. They are not extra calls.
            if load(observation)['root']==str(path): raise ValueError('unregistered complete call: '+path.name)
    return [root/name for name in sorted(expected)]


def derive(root):
    paths=declared_calls(root)
    bindings=load(root/'implementation.json')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest or sha(ROOT/name)!=digest:
            raise ValueError('implementation snapshot drift: '+name)
    calls=[]; mathematical=[]; ordinary={}
    for path in paths:
        observed=path/'caller_observation.json'
        call=load(observed)
        spec=load(path/'spec.json')
        try: checked=audit(path,call)
        except ValueError as error:
            if (spec['control']!='rebind_checker_context' or call['status']!='ERROR' or
                    str(error)!='invocation identity'): raise
            checked={'status':'EXPECTED_REJECTION','reason':str(error),'positive_execution_accepted':False}
        else:
            if spec['control']=='rebind_checker_context': raise ValueError('rebound invocation was not rejected')
        terminal=load(path/'terminal.json')
        calls.append({'directory':path.name,'spec':spec,'observed':call,'audit':checked,
            'terminal_sha256':sha(path/'terminal.json'),'stage_seconds':terminal['stage_seconds'],
            'parent_and_publication_seconds':call['seconds']-terminal['stage_seconds'],
            'produce_events_sha256':sha(path/'produce_events.jsonl') if (path/'produce_events.jsonl').exists() else None})
        if (path/'built.json').exists() and spec['control'] not in ('rebind_checker_context','omit_pair',
                'omit_property','missing_endpoint','wrong_mode'):
            built=load(path/'built.json')
            if built['invocation']!=call['invocation']: continue
            out=subprocess.run([PYTHON,'-B','-I','-S',str(path/'bundle/verify.py'),
                '--manifest-sha',built['sha256'],'--source-sha',spec['source_manifest_sha256'],
                '--proof-sha',built['proof_manifest_sha256'],'--mode',spec['mode']],
                cwd=root,text=True,capture_output=True,timeout=30)
            if out.returncode: raise ValueError('fresh mathematical check failed '+path.name+': '+out.stderr)
            result=json.loads(out.stdout)
            mathematical.append({'directory':path.name,'result':result['result'],'bundle_bytes':result['bundle_bytes']})
            if not spec['control'] and path.name not in ('final-publication','prelaunch'):
                original=load(path/'check.stdout')
                if original['result']!=result['result']: raise ValueError('fresh/captured math disagreement')
                ordinary[(spec['case'],spec['intake'],spec['mode'])]=(path,result['result'])
    comparisons=[]
    for case,intake in sorted({(c,i) for c,i,m in ordinary}):
        left,lr=ordinary[(case,intake,'endpoints')]; right,rr=ordinary[(case,intake,'mccormick')]
        lm,rm=[load(p/'bundle/proof/manifest.json') for p in (left,right)]
        if lm['source_manifest_sha256']!=rm['source_manifest_sha256'] or lm['reuse_requested']!=rm['reuse_requested']:
            raise ValueError('source/reuse arm mismatch')
        lc=[load(left/'bundle/proof'/ref['file'])['context'] for ref in lm['pairs']]
        rc=[load(right/'bundle/proof'/ref['file'])['context'] for ref in rm['pairs']]
        if lc!=rc or lr['scopes']!=rr['scopes']: raise ValueError('base/guard/gate/facts arm mismatch')
        comparisons.append({'case':case,'intake':intake,'contexts_sha256':identity(lc),
            'required':lr['required'],'endpoint_positive':lr['positive'],'mc_positive':rr['positive'],
            'source_manifest_sha256':lm['source_manifest_sha256']})
    if len(comparisons)!=5: raise ValueError('complete frozen normal-control roster')
    return {'schema':'HF_SUPERVISION_ARCHIVE_V1','root':str(root),'status':'PASS',
        'implementation_sha256':sha(root/'implementation.json'),'calls':calls,'call_count':len(calls),
        'statuses':dict(Counter(c['observed']['status'] for c in calls)),
        'mathematical_rechecks':mathematical,'paired_source_comparisons':comparisons,
        'real_requests':0,'deployed_float_SAFE':False,'performance_comparison':False,
        'scope':'fixed synthetic declarations and the same captured synthetic model only'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path); p.add_argument('--check',type=Path); a=p.parse_args()
    result=derive(a.root)
    if a.check and load(a.check)!=result: raise ValueError('archive drift')
    if a.output:
        with a.output.open('x') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('calls','mathematical_rechecks')},sort_keys=True))
