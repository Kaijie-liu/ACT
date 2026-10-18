"""Read saved R1 records only; never upgrades the failed frozen control batch.

No source construction, candidate generation, model, optimizer or verifier call.
This checks record identities and arithmetic, not independent source soundness.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time


def require(ok, why):
    if not ok: raise ValueError(why)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def audit(root):
    started=time.monotonic()
    terminal=json.loads((root/'terminal.json').read_bytes())
    freeze=json.loads((root/'freeze.json').read_bytes())
    worker=json.loads((root/'worker.json').read_bytes())
    require(terminal['execution_head']=='7477c2d2be47fa1083b5dd3c4d77576d0ed3f0b7', 'frozen R1 execution')
    require(terminal['status']=='FAIL' and terminal['returncode']==1, 'preserve failure')
    require(not terminal['timed_out'] and not terminal['resource_stop'] and terminal['report_error'] is None,
            'failure was not resource/deadline/partial report')
    require(terminal['wall_seconds']<300 and not (root/'late.json').exists(), 'on-time failed terminal')
    require(worker['status']=='FAIL' and worker['tests']==12 and worker['failures']==1 and worker['errors']==0,
            'full method denominator')
    require(not worker['unexpected_modules'] and worker['real_models']==worker['gpu']==0, 'stdlib only')
    for name, ref in terminal['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative file')
        raw=(root/name).read_bytes()
        require(len(raw)==ref['bytes'] and sha(raw)==ref['sha256'], 'artifact binding '+name)
    require(sum(x['bytes'] for x in terminal['files'].values())==terminal['artifact_bytes_before_terminal'],
            'artifact cost')
    require(terminal['peak_group_rss_bytes']<=2*2**30, 'RSS gate')
    methods=worker['log'].split('\n\n',1)[0].splitlines()
    require(len(methods)==12 and sum(s.endswith(' ... ok') for s in methods)==11, '11 passed, 1 failed')
    require('test_07_initial_canonical_faults' in worker['log'] and
            'AssertionError: 1 not greater than 1' in worker['log'], 'observed fixture failure')
    paired=[r for r in worker['records'] if r['kind']=='paired']
    require(len(paired)==freeze['paired_executions']==12, 'complete paired records')
    comparisons=[]
    for name in freeze['paired_sources']:
        for profile in freeze['profiles_by_source'][name]:
            arms=sorted((r for r in paired if r['source']==name and r['profile']==profile),key=lambda r:r['arm'])
            require([r['arm'] for r in arms]==list(range(4)), 'four unique arms')
            common=[]; totals=[]
            for a in arms:
                require(a['receipts']==arms[0]['receipts'] and a['result']==arms[0]['result'] and
                        a['candidate_sha256']==arms[0]['candidate_sha256'], 'exact differential')
                require(sha(canonical(a['receipts']))==a['receipts_sha256'], 'receipt identity')
                require(a['work']['arithmetic']==arms[0]['work']['arithmetic'], 'same arithmetic count')
                result=a['result']; duties,endpoints,tiles=freeze['full_rosters'][name]
                require(result['required']==duties and result['required_endpoints']==endpoints and
                        result['checked_endpoints']==endpoints and result['missing_endpoints']==0 and
                        len(a['receipts'])==endpoints, 'complete endpoint denominator')
                grouped={}; slots=set()
                for r in a['receipts']:
                    require(r['slot'] not in slots, 'unique slot'); slots.add(r['slot'])
                    require(F(r['lower'])==max(F(r['proposed']),F(r['zero'])), 'role maximum')
                    grouped.setdefault((tuple(r['pair']),r['other']),[]).append(F(r['lower']))
                rows=result['results']; keys={(tuple(r['pair']),r['other']) for r in rows}
                require(len(keys)==len(rows)==duties and keys==set(grouped), 'all unique duties')
                for r in rows:
                    lower=min(grouped[(tuple(r['pair']),r['other'])])
                    require(F(r['lower'])==lower and r['positive']==(lower>F(1,10000000)), 'positive gate')
                require(result['positive']==sum(r['positive'] for r in rows), 'positive count')
                l=a['lifetime']; s=a['traversal']; total=sum(a['work'].values()); totals.append(total)
                component=sum(l[k] for k in ('scope_work','global_seal_work','lease_work','plan_work'))
                component+=s['setup_work']+s['selector_work']-s['planned_iteration_saving']
                common.append(total-component)
                pairs=len({tuple(r['pair']) for r in rows})
                require(l['scopes_created']==l['scopes_retired']==(pairs if a['arm']>=2 else tiles), 'retired scopes')
                require(l['scopes_reused']==(tiles-pairs if a['arm']>=2 else 0), 'reuse count')
                if a['arm']>=2:
                    require(l['lease_work']==4*pairs+15*tiles+5*(endpoints+endpoints//2) and
                            l['plan_work']==3*tiles and l['global_seal_work']==(4*pairs if a['arm']==2 else 0),
                            'retention charges')
                o=a['owner']
                require(total<=8000000 and a['seconds']<300 and o['peak_bytes']<=134217728 and
                        o['live_bytes']==0 and o['allocated_bytes']==o['released_bytes'], 'complete accounting')
            require(len(set(common))==1, 'all Work differences explained')
            comparisons.append(dict(source=name,profile=profile,work_by_arm=totals,
                original_traversal_saving=totals[0]-totals[1],global_lifetime_saving=totals[0]-totals[2],
                matched_lifetime_traversal_saving=totals[2]-totals[3],total_vs_original_global=totals[0]-totals[3],
                seconds_by_arm=[a['seconds'] for a in arms],peak_owned_bytes_by_arm=[a['owner']['peak_bytes'] for a in arms],
                lifetime_by_arm=[a['lifetime'] for a in arms],traversal_by_arm=[a['traversal'] for a in arms],
                common_work=common[0],candidate_sha256=arms[0]['candidate_sha256'],receipts_sha256=arms[0]['receipts_sha256'],
                result_summary={k:v for k,v in arms[0]['result'].items() if k!='results'},new_positive_duties=0))
    oracles=[r for r in worker['records'] if r['kind']=='oracle']
    replays=[r for r in worker['records'] if r['kind']=='replay']
    require(len(oracles)==3 and sum(r['candidate_checks'] for r in oracles)==656 and len(replays)==2,
            'oracle/replay denominators')
    for r in replays:
        original=next(a for a in paired if a['source']==r['source'] and a['profile']==r['profile'] and a['arm']==3)
        require(r['receipts_sha256']==original['receipts_sha256'] and r['result']==original['result'] and
                r['owner']['live_bytes']==0 and r['owner']['allocated_bytes']==r['owner']['released_bytes'] and
                sum(r['work'].values())<=8000000, 'relocated receive')
    return dict(schema='PAIR_LIFETIME_R1_FAILED_BATCH_RECORD_REVIEW_V1',status='FAILED_BATCH_RECORDED',
        execution_head=terminal['execution_head'],raw_directory=str(root),terminal=terminal,
        terminal_sha256=sha((root/'terminal.json').read_bytes()),worker_sha256=sha((root/'worker.json').read_bytes()),
        frozen_protocol_sha256=sha((root/'freeze.json').read_bytes()),
        tests=dict(run=12,passed=11,failed=1,errors=0,log=worker['log']),
        paired_executions=12,oracle_records=oracles,oracle_role_checks=656,replay_executions=2,
        comparisons=comparisons,declined_before_freeze=freeze['declined_before_freeze'],
        review_seconds=time.monotonic()-started,
        scope='Offline saved-record identity/arithmetic review, not passing control acceptance or fresh source/LP reproof',
        decision='Keep opt-in off; no automatic rerun. Factor fault coverage remains incomplete; G1-G6 remain open.')


if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1])),sort_keys=True,indent=2,allow_nan=False))
