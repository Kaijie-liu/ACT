"""Readonly H2 object-intake archive: accounting and mathematical recheck, no solve."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,save,sha
from scoped_source.endpoint_supervised import audit,POSITIVE
from scoped_source.endpoint_source_check import check
from scoped_source.endpoint_intake import CAPTURE_SHA256

CASES={'endpoints':POSITIVE,'mccormick':'UNKNOWN_NONPOSITIVE',
    'mutate_model':'ERROR','mutate_input':'ERROR','capture_exception':'ERROR','capture_delay':'TIMEOUT',
    'missing_certificate':'UNKNOWN_MISSING_EVIDENCE','omit_property':'ERROR','rewrite_check_stdout':'ERROR'}


def derive(root):
    bindings=load(root/'implementation.json')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest or sha(ROOT/name)!=digest:
            raise ValueError('use archived implementation for this report')
    records=[];common=[]
    for name,status in CASES.items():
        path=root/name; call=load(path/'caller_observation.json')
        if call['status']!=status:raise ValueError('control terminal: '+name)
        audited=audit(path,call);terminal=load(path/'terminal.json')
        record={'case':name,'status':status,'seconds':call['seconds'],'audit':audited,
            'caller_sha256':sha(path/'caller_observation.json'),'terminal_sha256':sha(path/'terminal.json'),
            'stage_seconds':{s['phase']:s['seconds'] for s in terminal['stages']},
            'overhead_and_publication_seconds':call['seconds']-terminal['stage_seconds']}
        if (path/'accepted.json').exists():
            accepted=load(path/'accepted.json'); source=load(path/'bundle/source.json');proof=load(path/'bundle/proof.json')
            mode=load(path/'spec.json')['mode']
            checked=check(source,proof,expected_source_sha256=CAPTURE_SHA256,expected_mode=mode,deadline=time.monotonic()+30)
            if checked!=accepted['result']:raise ValueError('mathematical recheck differs')
            record.update(required=checked['required'],positive=checked['positive'],missing=checked['missing'],
                lower_bounds=[r['lower_bound'] for r in checked['duties']],lp_bounds_checked=checked['lp_bounds_checked'],
                bundle_bytes=load(path/'built.json')['bundle_bytes'],source_sha256=CAPTURE_SHA256,
                capture_receipt_sha256=sha(path/'model_intake.json'),generation=load(path/'generation.json'))
            if name in ('endpoints','mccormick'):common.append((source,proof['request'],proof['reuse_requested']))
        record['produce_events']=[__import__('json').loads(s) for s in (path/'produce_events.jsonl').read_text().splitlines()]
        records.append(record)
    if len(common)!=2 or common[0]!=common[1]:raise ValueError('different source/P/gate/facts')
    return {'schema':'H2_OBJECT_INTAKE_ARCHIVE_V1','root':str(root),'source_bindings':bindings,'cases':records,
        'counts':dict(Counter(r['status'] for r in records)),'source_sha256':CAPTURE_SHA256,
        'real_model_experiments':0,'new_solves_during_archive':0,'performance_claim':False,
        'scope':'Previously fixed synthetic weights instantiated as supported live model; declared real graph, not native float proof.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--output',type=Path);g.add_argument('--check',type=Path)
    a=p.parse_args();r=derive(a.root)
    if a.output:save(a.output,r)
    elif load(a.check)!=r:raise ValueError('archive differs')
    print({'status':'PASS','cases':len(r['cases']),'counts':r['counts'],'new_solves':0})
