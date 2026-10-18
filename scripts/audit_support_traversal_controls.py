"""Read-only stdlib record audit, not reproof of source/LP or a speed trial."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys


def check(ok,reason):
    if not ok: raise ValueError(reason)


def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()


def audit(root):
    t=json.loads((root/'terminal.json').read_bytes())
    freeze=json.loads((root/'freeze.json').read_bytes())
    check(t['status']=='PASS' and t['returncode']==0 and not t['timed_out'],'complete terminal')
    check(t['wall_seconds']<300 and not (root/'late.json').exists(),'on-time terminal')
    for name,record in t['files'].items():
        path=root/name
        check(not Path(name).is_absolute() and '..' not in Path(name).parts,'relative artifact')
        raw=path.read_bytes()
        check(len(raw)==record['bytes'] and sha(raw)==record['sha256'],'artifact identity '+name)
    check(sum(r['bytes'] for r in t['files'].values())==t['artifact_bytes_before_terminal'],'archive count')
    r=json.loads((root/'worker.json').read_bytes())
    check(r['status']=='PASS' and r['tests']==len(freeze['tests'])==10
          and not r['failures'] and not r['errors'] and not r['unexpected_modules'],'finite test results')
    paired=[x for x in r['records'] if x['kind']=='paired']
    check(len(paired)==freeze['paired_executions']==16,'complete paired denominator')
    differences=[]
    for name in freeze['paired_sources']:
        for profile in freeze['profiles']:
            pair=[x for x in paired if x['source']==name and x['profile']==profile]
            check(len(pair)==2 and [x['enabled'] for x in pair]==[False,True],'unique paired arms')
            a,b=pair
            check(a['receipts']==b['receipts'] and a['result']==b['result'],'same complete evidence')
            for x in pair:
                check(sha(canonical(x['receipts']))==x['receipts_sha256'],'receipt digest')
                check(x['seconds']<300 and sum(x['work'].values())<=8000000,'original limits')
                owner=x['owner']
                check(owner['live_bytes']==0 and owner['allocated_bytes']==owner['released_bytes']
                      and owner['peak_bytes']<=134217728,'released owned payload')
                result=x['result']; rows=result['results']
                check(len(x['receipts'])==result['required_endpoints']==result['checked_endpoints']
                      and result['missing_endpoints']==0,'all endpoints')
                seen=set(); grouped={}
                for receipt in x['receipts']:
                    check(receipt['slot'] not in seen,'unique slot'); seen.add(receipt['slot'])
                    check(F(receipt['lower'])==max(F(receipt['proposed']),F(receipt['zero'])),'role aggregation')
                    grouped.setdefault((tuple(receipt['pair']),receipt['other']),[]).append(F(receipt['lower']))
                check(len(rows)==result['required']==len(grouped),'all duties')
                for row in rows:
                    bound=min(grouped[(tuple(row['pair']),row['other'])])
                    check(F(row['lower'])==bound and row['positive']==(bound>F(1,10000000)),
                          'endpoint/property aggregation')
                check(result['positive']==sum(row['positive'] for row in rows),'positive denominator')
            s=b['traversal']; delta=sum(a['work'].values())-sum(b['work'].values())
            check(delta==s['planned_iteration_saving']-s['setup_work']-s['selector_work'],'all incremental Work')
            check(a['work']['arithmetic']==b['work']['arithmetic'],'unchanged arithmetic count')
            differences.append(dict(source=name,profile=profile,work_saving=delta,
                scoped_calls=s['scoped_calls'],global_calls=s['global_calls'],
                seconds_difference=b['seconds']-a['seconds']))
    return dict(status='PASS',execution_head=t['execution_head'],tests=10,paired_executions=16,
        differences=differences,terminal_sha256=sha((root/'terminal.json').read_bytes()),
        worker_sha256=sha((root/'worker.json').read_bytes()),
        scope='record identity, complete pairing, aggregation and costs; no independent source/LP reproof')


if __name__=='__main__': print(json.dumps(audit(Path(sys.argv[1])),sort_keys=True,indent=2))
