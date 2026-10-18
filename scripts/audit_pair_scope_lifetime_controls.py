"""Independent saved-record arithmetic/cost audit; no source or LP reproof."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys


def require(ok,why):
    if not ok: raise ValueError(why)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def audit(root):
    terminal=json.loads((root/'terminal.json').read_bytes())
    freeze=json.loads((root/'freeze.json').read_bytes())
    require(terminal['status']=='PASS' and terminal['returncode']==0
            and not terminal['timed_out'] and not terminal['resource_stop']
            and terminal['report_error'] is None,'complete execution')
    require(terminal['wall_seconds']<300 and not (root/'late.json').exists(),'on-time publication')
    require(terminal['peak_group_rss_bytes']<=2*2**30,'sampled group RSS gate')
    for name,ref in terminal['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'relative artifact')
        raw=(root/name).read_bytes()
        require(len(raw)==ref['bytes'] and digest(raw)==ref['sha256'],'artifact identity '+name)
    require(sum(x['bytes'] for x in terminal['files'].values())==terminal['artifact_bytes_before_terminal'],
            'complete archive size')
    report=json.loads((root/'worker.json').read_bytes())
    require(report['status']=='PASS' and report['tests']==len(freeze['tests'])==12
            and report['failures']==report['errors']==0 and not report['unexpected_modules'],'all controls')
    paired=[r for r in report['records'] if r['kind']=='paired']
    require(len(paired)==freeze['paired_executions']==12,'four-arm denominator')
    summaries=[]
    for name in freeze['paired_sources']:
        for profile in freeze['profiles_by_source'][name]:
            arms=sorted([r for r in paired if r['source']==name and r['profile']==profile],key=lambda r:r['arm'])
            require([r['arm'] for r in arms]==list(range(4)),'unique complete four arms')
            common=[]; totals=[]
            for r in arms:
                require(r['receipts']==arms[0]['receipts'] and r['result']==arms[0]['result']
                        and r['candidate_sha256']==arms[0]['candidate_sha256'],'same candidates and exact results')
                require(digest(canonical(r['receipts']))==r['receipts_sha256'],'receipt identity')
                require(r['work']['arithmetic']==arms[0]['work']['arithmetic'],'unchanged arithmetic count')
                total=sum(r['work'].values()); totals.append(total)
                require(total<=8000000 and r['seconds']<300,'request limits')
                owner=r['owner']
                require(owner['live_bytes']==0 and owner['allocated_bytes']==owner['released_bytes']
                        and owner['peak_bytes']<=134217728,'owned payload and release')
                result=r['result']; rows=result['results']; grouped={}; seen=set()
                duties,endpoints,tiles=freeze['full_rosters'][name]
                require(result['required']==duties and result['required_endpoints']==endpoints
                        and result['checked_endpoints']==endpoints and result['missing_endpoints']==0
                        and len(r['receipts'])==endpoints,'all necessary endpoints')
                for receipt in r['receipts']:
                    require(receipt['slot'] not in seen,'unique slot'); seen.add(receipt['slot'])
                    require(F(receipt['lower'])==max(F(receipt['proposed']),F(receipt['zero'])),'role maximum')
                    grouped.setdefault((tuple(receipt['pair']),receipt['other']),[]).append(F(receipt['lower']))
                require(len(grouped)==len(rows)==duties,'complete duties')
                for row in rows:
                    lower=min(grouped[(tuple(row['pair']),row['other'])])
                    require(F(row['lower'])==lower and row['positive']==(lower>F(1,10000000)),'strict aggregation')
                require(result['positive']==sum(row['positive'] for row in rows),'positive count')
                s=r['traversal']; l=r['lifetime']
                component=l['scope_work']+l['global_seal_work']+l['lease_work']+l['plan_work']
                component+=s['setup_work']+s['selector_work']-s['planned_iteration_saving']
                common.append(total-component)
                pairs=len({tuple(row['pair']) for row in rows})
                require(l['scopes_created']==l['scopes_retired']==(pairs if r['arm']>=2 else tiles),
                        'creation/retirement lifetime')
                require(l['scopes_reused']==(tiles-pairs if r['arm']>=2 else 0),'same-pair reuse count')
                if r['arm']>=2:
                    # These frozen sources have even [8,2] / [8] tiles.
                    pulls=endpoints+endpoints//2
                    require(l['lease_work']==4*pairs+15*tiles+5*pulls and l['plan_work']==3*tiles,
                            'all complete lease/plan charges')
                    require(l['global_seal_work']==(4*pairs if r['arm']==2 else 0),'global minimal sealing')
            require(len(set(common))==1,'all Work differences explained, no free lifecycle overhead')
            summaries.append(dict(source=name,profile=profile,work_by_arm=totals,
                original_traversal_saving=totals[0]-totals[1],
                global_lifetime_saving=totals[0]-totals[2],
                matched_lifetime_traversal_saving=totals[2]-totals[3],
                total_vs_original_global=totals[0]-totals[3],
                seconds_by_arm=[r['seconds'] for r in arms],
                peak_owned_bytes_by_arm=[r['owner']['peak_bytes'] for r in arms]))
    oracles=[r for r in report['records'] if r['kind']=='oracle']
    replays=[r for r in report['records'] if r['kind']=='replay']
    require(len(oracles)==freeze['oracle_records']==3 and len(replays)==freeze['replay_executions']==2,'oracle/replay denominator')
    require(sum(r['candidate_checks'] for r in oracles)==freeze['oracle_role_checks']==656,'expanded exact role comparisons')
    for replay in replays:
        source=next(r for r in paired if r['source']==replay['source'] and r['profile']==replay['profile'] and r['arm']==3)
        require(replay['receipts_sha256']==source['receipts_sha256'] and replay['result']==source['result'],
                'fresh relocated receive agrees')
        require(sum(replay['work'].values())<=8000000 and replay['owner']['live_bytes']==0,'replay accounting')
    return dict(status='PASS',execution_head=terminal['execution_head'],tests=12,paired_executions=12,
        oracle_role_checks=656,replays=2,comparisons=summaries,
        terminal_sha256=digest((root/'terminal.json').read_bytes()),
        worker_sha256=digest((root/'worker.json').read_bytes()),
        scope='saved identity, full obligations, same-candidate aggregation and counted costs; no independent source/LP reproof')


if __name__=='__main__': print(json.dumps(audit(Path(sys.argv[1])),sort_keys=True,indent=2))
