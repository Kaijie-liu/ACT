"""Preserve the R1 type-boundary failure before any LP proposal; no retry."""
import json
from scripts.conv_pre_f0_contract import ROOT,DEFAULT,FREEZE,read,save,sha,verify_freeze
from scripts.review_conv_request_sign_lp import check_journal

OUTPUT=ROOT/'act/pipeline/moe/results/conv_pre_f0_failure_20260915_r1.json'


def review():
    f=verify_freeze();r=read(DEFAULT/'runtime.json');d=DEFAULT/f['protocol']['job_id'];stage=r['stages']['capture']
    if (r['state']!='ERROR' or set(r['stages'])!={'capture'} or stage['return_code']!=1
            or stage['state']!='ERROR' or stage!=read(d/'capture.terminal.json') or r['result'] is not None):
        raise ValueError('failure terminal differs')
    if (d/'proposal.json').exists() or (d/'query_log.json').exists() or list(d.glob('*.certificate.json')):
        raise ValueError('unexpected proposal evidence')
    text=(d/'capture.log').read_text()
    if ('rational(weight)' not in text or 'finite int/float or rational string required' not in text):
        raise ValueError('unexpected failure')
    m=read(d/'manifest.json')
    if m['generation_complete'] or set(m['supports'])!={'gate_lower','gate_upper'}:
        raise ValueError('unexpected partial generation')
    p=r['publication']
    if p['remote_head']!=r['execution_head'] or p['local_head']!=r['execution_head'] or p['confirmed_unix']>=r['started_unix']:
        raise ValueError('publication order differs')
    journal=check_journal(d/'budget_journal.jsonl',sha(d/'job.json'),[[1,2]],0)
    return {'schema':'CONV_PRE_F0_R1_FAILURE_REVIEW','audit_status':'PASS','issues':[],
        'execution_status':'ERROR_RETAINED','execution_head':r['execution_head'],'freeze_sha256':sha(FREEZE),
        'failure':'Torch property scalar rejected by rational() during first difference export',
        'new_LP_proposals':0,'new_output_bounds':0,'complete_request_evidence':False,
        'stage':stage,'journal':journal,'old_results_unchanged':True,
        'repair_scope':'new version: canonicalize verified +1/0/-1 classification rows to Python numbers; no math, range, budget or sample change',
        'artifact_inventory':[{'path':str(p.relative_to(DEFAULT)),'sha256':sha(p),'bytes':p.stat().st_size}
                              for p in sorted(DEFAULT.rglob('*')) if p.is_file()]}


if __name__=='__main__':
    r=review()
    if OUTPUT.exists():
        if read(OUTPUT)!=r:raise ValueError('failure archive changed')
    else:save(OUTPUT,r)
    print(json.dumps({k:r[k] for k in ('audit_status','execution_status','new_LP_proposals')},indent=2))
