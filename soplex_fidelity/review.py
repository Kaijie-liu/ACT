"""Independent saved readback comparison, without importing exporter/reader logic."""
from fractions import Fraction as F
import argparse
import hashlib
import json
from pathlib import Path
import time

from lp_sandwich.check import identity,strict_json

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(lp,path):
    n=len(lp['c']);col_seen=set();row_seen=set();entries=0
    with Path(path).open() as f:
        if next(f).strip()!='SOPLEX_RATIONAL_READBACK_V1':raise ValueError('header')
        if next(f).split()!=['COLUMNS',str(n)]:raise ValueError('column count')
        for _ in range(n):
            name,*values=next(f).split()
            if not name.startswith('x'):raise ValueError('name')
            j=int(name[1:])
            if name!=f'x{j}' or j in col_seen or not 0<=j<n:raise ValueError('mapping')
            if list(map(F,values))!=[F(lp[k][j]) for k in ('c','lower','upper')]:raise ValueError('column mismatch')
            col_seen.add(j)
        count=len(lp['b'])+len(lp['h'])
        if next(f).split()!=['ROWS',str(count)]:raise ValueError('row count')
        for _ in range(count):
            name,left,right,num,*terms=next(f).split();i=int(name[1:])
            if name[0] not in ('a','e') or name!=f'{name[0]}{i}' or name in row_seen:raise ValueError('row name')
            m,r=(lp['A'],lp['b']) if name[0]=='a' else (lp['E'],lp['h'])
            if not 0<=i<len(r) or F(right)!=F(r[i]):raise ValueError('rhs')
            if (left!='-inf' if name[0]=='a' else F(left)!=F(r[i])):raise ValueError('lhs')
            if len(terms)!=2*int(num):raise ValueError('nnz count')
            actual={}
            for k in range(0,len(terms),2):
                col,val=terms[k:k+2];j=int(col[1:])
                if col!=f'x{j}' or j not in col_seen or j in actual:raise ValueError('entry mapping')
                actual[j]=F(val)
            start,end=m['indptr'][i:i+2]
            expected={m['indices'][k]:F(m['data'][k]) for k in range(start,end) if F(m['data'][k])}
            if {j:v for j,v in actual.items() if v}!=expected:raise ValueError('matrix mismatch')
            entries+=len(expected);row_seen.add(name)
        if next(f).strip()!='END' or f.read():raise ValueError('terminal')
    return dict(variables=n,rows=count,nonzero_entries=entries,offset_retained_in_original=str(F(lp['offset'])))


def review(receipt,output):
    if output.exists():raise FileExistsError(output)
    start=time.monotonic();r=strict_json(receipt.read_bytes());raw=Path(r['raw_root'])
    if not raw.is_absolute():raw=ROOT/raw
    if r['status']!='PASS' or r['optimization_calls']!=0:raise ValueError('receipt scope')
    for name,digest in r['artifacts'].items():
        if sha(raw/name)!=digest:raise ValueError('artifact mutation')
    for name,digest in r['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source mutation')
    if sha(r['parent']['path'])!=r['parent']['sha256']:raise ValueError('parent mutation')
    jobs=strict_json(Path(r['parent']['path']).read_bytes())['jobs'];rows=[]
    if [j['job_id'] for j in jobs]!=[v['job_id'] for v in r['rows']]:raise ValueError('roster')
    for job,saved in zip(jobs,r['rows']):
        if sha(job['export']['path'])!=job['export']['sha256']:raise ValueError('original export mutation')
        export=strict_json(Path(job['export']['path']).read_bytes());lp=export['lp'];s=job['statement']
        if identity(lp)!=s['lp_sha256'] or identity(s)!=job['statement_sha256']:
            raise ValueError('original identity')
        if (identity(export['source'])!=s['source_sha256'] or export['source_sha256']!=s['source_sha256'] or
            list(map(F,export['q']))!=list(map(F,s['property']['q'])) or F(export['offset'])!=F(s['property']['constant'])):
            raise ValueError('source/property')
        v=compare(lp,raw/job['job_id']/'readback.txt')
        if (v['variables']!=saved['comparison']['columns'] or
            v['nonzero_entries']!=saved['comparison']['nonzero_entries'] or
            saved['original_lp_sha256']!=s['lp_sha256'] or saved['status']!='PASS'):
            raise ValueError('saved counts/result')
        if (sum(saved['cost_seconds'].values())>saved['observed_seconds'] or
            saved['native']['returncode']!=0 or saved['native']['optimization_calls']!=0):
            raise ValueError('cost/native record')
        rows.append(dict(job_id=job['job_id'],**v))
    result=dict(status='PASS',issues=[],receipt_sha256=sha(receipt),reviewer_sha256=sha(Path(__file__)),
        artifacts=len(r['artifacts']),rows=rows,optimization_calls=0,native_imports=0,
        seconds=time.monotonic()-start,scope='Exact original LP vs saved rational readback, not proof of optimizer or network.')
    with output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(review(a.receipt,a.output),indent=2))
