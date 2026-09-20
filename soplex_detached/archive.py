"""Saved-artifact archival, lexical size diagnosis only; no candidate/LP solving."""
from pathlib import Path
import re
import time

from lp_sandwich.check import strict_json
from soplex_fidelity.io import save,sha
from soplex_execution.runtime import ROOT
from soplex_detached.run import frozen,OUTPUT,FREEZE,PROTOCOL


def lexical_sizes(path):
    # Never construct the over-limit integers/Fractions. Compare decimal strings
    # to a bounded 4096-bit maximum, independently of the receiver's length gate.
    maximum=str((1<<4096)-1);large=0;seen=0;maxdigits=0;maxtoken=0;first=None
    for line in path.read_text('ascii').splitlines():
        if not re.match(r'^x[0-9]+\s',line):continue
        fields=line.split()
        if len(fields)!=2:raise ValueError('lexical point row')
        name,value=fields;parts=value.lstrip('-').split('/')
        if len(parts)>2 or any(not p.isdecimal() for p in parts):raise ValueError('unexpected point token')
        oversized=any(len(p.lstrip('0') or '0')>len(maximum) or
                      (len(p.lstrip('0') or '0')==len(maximum) and (p.lstrip('0') or '0')>maximum) for p in parts)
        seen+=1;maxdigits=max(maxdigits,*map(len,parts));maxtoken=max(maxtoken,len(value))
        if oversized:
            large+=1
            if first is None:first=dict(coordinate=name,token_characters=len(value),decimal_digits=list(map(len,parts)))
    return dict(listed_coordinates=seen,coordinates_exceeding_serialized_4096_bit_cap=large,
        maximum_integer_decimal_digits=maxdigits,maximum_rational_token_characters=maxtoken,
        first_oversized=first,point_bytes=path.stat().st_size,
        scope='Lexical size of unaccepted serialized output; no exact feasibility, canonical fraction reduction or minimum required representation claim.')


def archive():
    started=time.monotonic();f,_=frozen();root=OUTPUT
    completed=strict_json((root/'completion.json').read_bytes());r=strict_json((root/'final_review.json').read_bytes())
    if completed['status']!='AUDITED' or completed['review_sha256']!=sha(root/'final_review.json') or r['status']!='PASS' or r['issues']:
        raise ValueError('final audit gate')
    if r['denominator']!=4 or r['optimization_calls']!=4:raise ValueError('finite roster')
    rows=[];artifact_count=0
    for job,row in zip(f['jobs'],r['rows']):
        if row['job_id']!=job['job_id']:raise ValueError('roster order')
        jr=root/job['job_id']
        for name,digest in row['artifacts'].items():
            if sha(jr/name)!=digest:raise ValueError('saved artifact drift')
            artifact_count+=1
        worker=strict_json((jr/'worker_result.json').read_bytes())
        if row['status']!='LIMIT' or row['upper_bound'] is not None or worker['error']!='rational token length':raise ValueError('unexpected endpoint')
        if row['phase_costs']['check']['state']!='NOT_ENTERED' or (jr/'bundle.json').exists():raise ValueError('unexpected acceptance path')
        lexical=lexical_sizes(jr/'point.txt')
        if not lexical['coordinates_exceeding_serialized_4096_bit_cap']:raise ValueError('limit not corroborated')
        stdout=(jr/'solver.stdout').read_text();native=strict_json((jr/'solver.process.json').read_bytes())
        if native['returncode']!=0 or native['termination'] is not None:raise ValueError('native process did not complete')
        rows.append(dict(job_id=job['job_id'],status='LIMIT',reason=worker['error'],checked_upper=None,
            whole_request_seconds=row['whole_request_seconds'],solve_phase_seconds=row['phase_costs']['solve']['seconds'],
            solver_peak_rss_kib=int(row['solver_gnu_time'].split()[1]),
            native_reported_optimal_UNTRUSTED=bool(re.search(r'SoPlex status\s*:\s*problem is solved \[optimal\]',stdout)),
            **lexical))
    result=dict(schema='SOPLEX_FINITE_V2_ARCHIVAL_SUMMARY',status='ARCHIVED_FINITE_STUDY_CLOSED',
        rows=rows,denominator=4,real_native_queries=4,checked_feasible_U=0,new_network_SAFE=0,new_network_UNSAFE=0,
        input_readback_all_fields_equal=4,artifact_hashes_rechecked=artifact_count,
        execution_head=r['execution']['git_head'],protocol_sha256=sha(PROTOCOL),execution_freeze_sha256=sha(FREEZE),
        raw_root=str(root),review_sha256=sha(root/'final_review.json'),
        total_request_seconds=sum(v['whole_request_seconds'] for v in rows),batch_wall_seconds=r['batch_wall_seconds'],
        guardian_lifecycle_seconds=strict_json((root/'guardian_terminal.json').read_bytes())['seconds'],
        postterminal_audit_seconds=completed['postterminal_audit_seconds'],
        archive_seconds=time.monotonic()-started,archive_native_queries=0,archive_feasibility_checks=0,
        interpretation='The frozen output admission contract prevents exact feasibility checking. Native optimal reports are untrusted; no LP obstruction or network safety/unsafety established. No larger bit cap, retry or custom arithmetic authorized.')
    save(ROOT/'docs/soplex_finite_real_v2_review.json',r)
    save(ROOT/'docs/soplex_finite_real_v2_summary.json',result)
    print({k:v for k,v in result.items() if k!='rows'})
    for row in rows:print(row)


if __name__=='__main__':archive()
