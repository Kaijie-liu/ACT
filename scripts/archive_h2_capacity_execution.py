"""Finite capacity-control inventory and independent stored execution re-audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import PYTHON,load,sha,save
from scripts.h2_capacity_supervised import specification,audit,POSITIVE,producer_sources
from source_enclosure.format import identity

TEST_NAMES=(
    'test_archive_missing_calls_rejected','test_archive_complete_inventory',
    'test_checker_and_stdout_rebinding','test_cost_and_hash_chain_tampering',
    'test_cost_semantics_after_rebinding','test_deadlines_reached_and_no_late_acceptance',
    'test_faults_partial_and_missing_evidence','test_final_publication_and_prelaunch',
    'test_fixed_admission_and_mandatory_hashes','test_normal_both_arms_and_charged_capture',
    'test_observation_only_ast_and_aggregation','test_old_new_builder_differential_all_tiny_sources',
    'test_owned_cleanup_and_memory','test_receiver_aggregation_and_capture_binding',
    'test_relocation_and_inventory','test_stage_semantics_after_rebinding','test_startup_exception_accounted')


def implementation_names():
    return set(producer_sources(specification()))|{'scripts/test_h2_capacity_execution.py','scripts/archive_h2_capacity_execution.py'}


def check_control_bindings(root):
    outcome=load(root/'test_outcome.json')
    if (outcome['success'] is not True or outcome['failures'] or outcome['errors'] or outcome['skipped'] or
        outcome['tests']!=len(TEST_NAMES) or outcome.get('test_names')!=sorted(TEST_NAMES)):
        raise ValueError('fixed controls have not passed')
    bindings=load(root/'implementation.json')
    if set(bindings)!=implementation_names(): raise ValueError('implementation inventory')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest or sha(ROOT/name)!=digest: raise ValueError('implementation drift: '+name)
    return outcome


def roster():
    calls={}
    for mode in ('endpoints','mccormick'):
        calls['normal-'+mode]=(specification(mode=mode),POSITIVE if mode=='endpoints' else 'UNKNOWN_NONPOSITIVE',30,2*2**30)
    for fault in ('missing_certificate','missing_both','missing_endpoint','omit_property','omit_pair','wrong_mode',
                  'proposal_exception','partial_output','exception_after_bundle','wrong_invocation',
                  'mutate_model','mutate_input','capture_exception','missing_stdout'):
        status='UNKNOWN_MISSING_EVIDENCE' if fault in ('missing_certificate','missing_both','proposal_exception') else 'ERROR'
        calls['fault-'+fault]=(specification(control=fault),status,30,2*2**30)
    for fault in ('produce_delay','capture_delay','chunk_delay','construct_delay','pair_delay','endpoint_delay',
                  'native_delay','proposal_delay','serialization_delay','check_delay','receive_delay','late_publish'):
        calls['deadline-'+fault]=(specification(control=fault),'TIMEOUT',6,2*2**30)
    calls['rebound-checker']=(specification(control='rebind_checker_context'),'ERROR',30,2*2**30)
    calls['replaced-stdout']=(specification(mode='mccormick',control='rewrite_check_stdout'),'ERROR',30,2*2**30)
    calls['resource-descendant']=(specification(control='descendant'),'ERROR',6,2*2**30)
    calls['resource-memory']=(specification(control='memory'),'RESOURCE_LIMIT',6,64*2**20)
    calls['final-publication']=(specification(),'TIMEOUT',10,2*2**30)
    calls['prelaunch']=(specification(),'TIMEOUT',1e-9,2*2**30)
    calls['startup-error']=(specification(),'ERROR',6,2*2**30)
    return calls


def declared_calls(root):
    expected=roster()
    for name,(spec,status,budget,rss) in expected.items():
        path=root/name
        for file in ('caller_observation.json','spec.json','terminal.json','invocation.json','receipt.json','finish.json'):
            if not (path/file).is_file(): raise ValueError('missing registered call artifact: '+name+'/'+file)
        call=load(path/'caller_observation.json'); inv=load(path/'invocation.json')
        if (call['root']!=str(path) or call['status']!=status or load(path/'spec.json')!=spec or
            inv['budget']!=budget or inv['rss_limit']!=rss): raise ValueError('registered call identity/outcome: '+name)
    for path in root.iterdir():
        if path.is_dir() and (path/'caller_observation.json').exists() and path.name not in expected:
            if load(path/'caller_observation.json')['root']==str(path): raise ValueError('unregistered call')
    return [root/name for name in sorted(expected)]


def derive(root):
    paths=declared_calls(root); outcome=check_control_bindings(root)
    calls=[]; maths=[]
    for path in paths:
        call=load(path/'caller_observation.json'); spec=load(path/'spec.json'); terminal=load(path/'terminal.json')
        try: checked=audit(path,call)
        except ValueError as exc:
            if spec['control']!='rebind_checker_context' or str(exc)!='invocation identity': raise
            checked={'status':'EXPECTED_REJECTION','reason':str(exc),'positive_execution_accepted':False}
        else:
            if spec['control']=='rebind_checker_context': raise ValueError('rebound invocation accepted')
        calls.append({'directory':path.name,'observed':call,'audit':checked,
            'terminal_sha256':sha(path/'terminal.json'),'stage_seconds':terminal['stage_seconds'],
            'parent_publication_seconds':call['seconds']-terminal['stage_seconds'],
            'trace_sha256':sha(path/'produce_events.jsonl') if (path/'produce_events.jsonl').exists() else None})
        if checked.get('pipeline_complete'):
            b=load(path/'built.json')
            out=subprocess.run([PYTHON,'-B','-I','-S',str(path/'bundle/verify.py'),
                '--manifest-sha',b['sha256'],'--source-sha',spec['source_manifest_sha256'],
                '--proof-sha',b['proof_manifest_sha256'],'--mode',spec['mode']],
                cwd=root,text=True,capture_output=True,timeout=30)
            if out.returncode: raise ValueError('independent portable recheck: '+path.name+out.stderr)
            result=json.loads(out.stdout)['result']
            if result!=load(path/'accepted.json')['result']: raise ValueError('math result drift')
            maths.append({'directory':path.name,'result':result})
    contexts=[]
    for mode in ('endpoints','mccormick'):
        path=root/('normal-'+mode)/'bundle/proof'; m=load(path/'manifest.json')
        contexts.append([load(path/r['file'])['context'] for r in m['pairs']])
    if contexts[0]!=contexts[1]: raise ValueError('base/guard/gate differs between arms')
    return {'schema':'H2_CAPACITY_EXECUTION_CONTROLS_V1','status':'PASS','root':str(root),
        'implementation_sha256':sha(root/'implementation.json'),'test_outcome_sha256':sha(root/'test_outcome.json'),
        'tests':outcome['tests'],'calls':calls,'call_count':len(calls),
        'statuses':dict(Counter(c['observed']['status'] for c in calls)),
        'mathematical_rechecks':maths,'paired_context_sha256':identity(contexts[0]),
        'real_requests':0,'full_size_calls':0,'performance_claim':False,'native_float_SAFE':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path); p.add_argument('--check',type=Path); args=p.parse_args()
    result=derive(args.root)
    if args.check and load(args.check)!=result: raise ValueError('archive drift')
    if args.output: save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('calls','mathematical_rechecks')},sort_keys=True))
