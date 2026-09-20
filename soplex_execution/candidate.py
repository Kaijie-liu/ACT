"""Admit the pinned CLI's complete exact primal output, never native status."""
from fractions import Fraction as F
from pathlib import Path
import re

from lp_sandwich.check import identity, rational


class LimitError(Exception):
    pass


def token(value, bits=4096):
    # Bound lexical size BEFORE int/Fraction allocation; no decimals or exponent guessing.
    if len(value)>2*(int(bits*.302)+3)+2:
        raise LimitError('rational token length')
    if not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?', value):
        raise ValueError('not an exact integer/fraction token')
    a, _, b=value.partition('/')
    if max(abs(int(a)).bit_length(), int(b or '1').bit_length())>bits:
        raise LimitError('serialized rational bit cap')
    return F(value)


def parse(path, n, tick=lambda:None, byte_cap=64*1024**2):
    path=Path(path);tick()
    if not path.exists():
        return None, {'state':'NO_POINT_FILE','max_bits':None}
    if path.stat().st_size>byte_cap:
        raise LimitError('point byte cap')
    with path.open('r', encoding='ascii') as f:
        text=f.read(byte_cap+1)
    tick()
    if len(text)>byte_cap:raise LimitError('point byte cap')
    if text=='No primal (rational) solution available.\n':
        return None, {'state':'NATIVE_NO_POINT','max_bits':None}
    lines=text.splitlines()
    if not lines or lines[0]!='' or len(lines)<3 or lines[1]!='Primal solution (name, value):':
        raise ValueError('missing exact primal header (rays are not admitted)')
    if not text.endswith('\n'):
        raise ValueError('partial primal output')
    footer=re.fullmatch(r'All other variables are zero\. Solution has (0|[1-9][0-9]*) nonzero entries\.',lines[-1])
    if footer is None or len(footer[1])>6:raise ValueError('missing/invalid exact primal footer')
    count=int(footer[1])
    if count!=len(lines)-3 or count>n:raise ValueError('nonzero count mismatch')
    x=[F(0)]*n;seen=set();max_bits=1
    for line in lines[2:-1]:
        tick();fields=line.split()
        if len(fields)!=2:raise ValueError('point row')
        name,value=fields
        if len(name)>12 or not re.fullmatch(r'x(?:0|[1-9][0-9]*)',name):raise ValueError('coordinate name')
        j=int(name[1:])
        if not 0<=j<n or j in seen:raise ValueError('unknown/duplicate coordinate')
        q=token(value)
        if not q:raise ValueError('zero entry in declared nonzero list')
        max_bits=max(max_bits,abs(q.numerator).bit_length(),q.denominator.bit_length())
        seen.add(j);x[j]=q
    tick()
    return x, {'state':'COMPLETE_EXACT_PRIMAL','max_bits':max_bits,'nonzero_entries':count}


def bundle(lp, statement, point, tick=lambda:None):
    tick();p=None
    if point is not None:
        value=rational(lp['offset'])
        for a,b in zip(lp['c'],point):
            tick();value+=rational(a)*b
        token(str(value))
        p={'lp_sha256':identity(lp),'statement_sha256':identity(statement),
           'x':[str(x) for x in point],'claimed_objective':str(value)}
    tick()
    return dict(schema='LP_SANDWICH_V1',lp=lp,statement=statement,primal=p,dual=None)
