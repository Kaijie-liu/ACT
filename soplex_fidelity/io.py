"""Sparse fraction-token export and complete original-coordinate readback.

No native feasibility, optimization or floating-point projection is used here.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

from lp_sandwich.check import (identity, inspect_bounds, matrix_rows, rational,
                               strict_json, validate_statement)

MAX_VARIABLES=16384
MAX_ROWS=16384
MAX_NNZ=1000000
MAX_BYTES=128*1024*1024
MAX_BITS=4096


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,v):
    with Path(path).open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)


def bounded(v):
    q=rational(v)
    if max(abs(q.numerator).bit_length(),q.denominator.bit_length())>MAX_BITS:
        raise ValueError('external coefficient/output bit limit')
    return q


def load_job(job,tick=lambda:None):
    tick();ref=job['export'];path=Path(ref['path'])
    if path.stat().st_size>MAX_BYTES:raise ValueError('export byte limit')
    raw=path.read_bytes();tick()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('export drift')
    v=strict_json(raw);s=job['statement'];p=v['lp']
    if (ref['sha256']!=s['export_sha256'] or identity(v['source'])!=s['source_sha256']
        or v['source_sha256']!=s['source_sha256'] or
        list(map(rational,v['q']))!=list(map(rational,s['property']['q'])) or
        rational(v['offset'])!=rational(s['property']['constant'])):
        raise ValueError('source/property binding')
    validate_statement(s,p,job['statement_sha256'],tick)
    validate(p,tick);return p


def validate(lp,tick=lambda:None):
    n=len(lp['c']);m=len(lp['b'])+len(lp['h'])
    if not 0<n<=MAX_VARIABLES or m>MAX_ROWS:raise ValueError('external shape limit')
    if sum(len(lp[key]['data']) for key in ('A','E'))>MAX_NNZ:raise ValueError('nnz limit')
    inspect_bounds(lp,None,None,'',tick)
    for key in ('c','lower','upper','b','h'):
        for i,v in enumerate(lp[key]):
            if i%256==0:tick()
            bounded(v)
    bounded(lp['offset'])
    for key in ('A','E'):
        for row in matrix_rows(lp[key],n,tick):
            for _,v in row:bounded(str(v))


def write_lp(lp,path,tick=lambda:None):
    validate(lp,tick);written=0
    with Path(path).open('x') as f:
        def put(line):
            nonlocal written
            tick();written+=len(line.encode())+1
            if written>MAX_BYTES:raise ValueError('LP text byte limit')
            f.write(line+'\n')
        def expression(prefix,items,suffix=''):
            # Small physical lines avoid native reader line/token-size surprises.
            line=prefix;terms=0
            for j,value in items:
                value=bounded(str(value))
                term=(' + ' if value>=0 else ' - ')+str(abs(value))+f' x{j}'
                if len(line)+len(term)>4096:put(line);line=' '
                line+=term;terms+=1
            if not terms:line+=' + 0 x0'
            if len(line)+len(suffix)>4096:put(line);line=' '
            put(line+suffix)
        put('Minimize');expression('obj:',enumerate(map(rational,lp['c'])))
        put('Subject To')
        for mat,rhs,prefix,op in [('A','b','a','<='),('E','h','e','=')]:
            for i,row in enumerate(matrix_rows(lp[mat],len(lp['c']),tick)):
                expression(f'{prefix}{i}:',row,' '+op+' '+str(bounded(lp[rhs][i])))
        put('Bounds')
        for j,(lo,hi) in enumerate(zip(lp['lower'],lp['upper'])):
            put(f'{bounded(lo)} <= x{j} <= {bounded(hi)}')
        put('End')
    tick();return written


def verify_readback(lp,path,tick=lambda:None):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES:raise ValueError('readback byte limit')
    n=len(lp['c']);seen_cols=set();seen_rows=set();stored=0;nonzero=0;bits=0;small=0
    def q(v):
        nonlocal bits
        out=bounded(v);bits=max(bits,abs(out.numerator).bit_length(),out.denominator.bit_length())
        return out
    with path.open() as f:
        def line():
            tick();s=f.readline(MAX_BYTES+1)
            if not s or not s.endswith('\n'):raise ValueError('partial readback')
            return s.rstrip('\n')
        def count(label,expected):
            if line().split()!=[label,str(expected)]:raise ValueError('native dimensions')
        if line()!='SOPLEX_RATIONAL_READBACK_V1':raise ValueError('readback schema')
        count('COLUMNS',n)
        for _ in range(n):
            name,c,lo,hi=line().split()
            if not name.startswith('x') or not name[1:].isdigit():raise ValueError('column name')
            j=int(name[1:])
            if not 0<=j<n or name!=f'x{j}' or j in seen_cols:raise ValueError('column mapping')
            if tuple(map(q,(c,lo,hi)))!=tuple(rational(lp[k][j]) for k in ('c','lower','upper')):
                raise ValueError('column coefficients differ')
            seen_cols.add(j)
        count('ROWS',len(lp['b'])+len(lp['h']))
        for _ in range(len(lp['b'])+len(lp['h'])):
            name,lo,hi,size,*entries=line().split()
            if name in seen_rows or not name or name[0] not in 'ae' or not name[1:].isdigit():
                raise ValueError('row mapping')
            i=int(name[1:]);mat,rhs=('A','b') if name[0]=='a' else ('E','h')
            if name!=f'{name[0]}{i}' or not 0<=i<len(lp[rhs]):raise ValueError('row index')
            if int(size)<0 or len(entries)!=2*int(size):raise ValueError('native row nnz')
            right=rational(lp[rhs][i])
            if q(hi)!=right or (lo!='-inf' if mat=='A' else q(lo)!=right):
                raise ValueError('native row sides differ')
            m=lp[mat];start,end=m['indptr'][i:i+2]
            expected={m['indices'][k]:rational(m['data'][k]) for k in range(start,end)
                      if rational(m['data'][k])}
            actual={};stored+=int(size)
            for k in range(0,len(entries),2):
                if k%512==0:tick()
                col,value=entries[k:k+2]
                if not col.startswith('x') or not col[1:].isdigit():raise ValueError('row column')
                j=int(col[1:]);v=q(value)
                if j not in seen_cols or col!=f'x{j}' or j in actual:raise ValueError('duplicate/invalid coordinate')
                actual[j]=v
                if v:nonzero+=1;small+=int(abs(v)<=F(1,10**9))
            if {j:v for j,v in actual.items() if v}!=expected:raise ValueError('native matrix differs')
            seen_rows.add(name)
        if line()!='END' or f.read(1):raise ValueError('incomplete/extra readback')
    return dict(columns=n,A_rows=len(lp['b']),E_rows=len(lp['h']),native_stored_entries=stored,
                nonzero_entries=nonzero,small_nonzero_entries_le_1e_9=small,max_rational_bits=bits,
                all_fields_equal=True,objective_translation=str(rational(lp['offset'])))
