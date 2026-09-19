"""Analyze the saved intended submission; never import a real LP into HiGHS."""
import math
from fractions import Fraction
from single_check_portable.execution import ROOT, read
from portable_proof.runtime import digest
from lp_sandwich.check import identity, rational, matrix_rows, validate_statement

ARCHIVE = ROOT/'docs/sparse_supervised_real_v1_execution_results.json'
SAVED = ROOT/'data/moe/results/sparse_supervised_real_20260919_v1/input220_p0/native/input.json'


def summary(values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite submitted scalar')
    nz = [abs(v) for v in values if v]
    return {'count':len(values),'zeros':sum(v==0 for v in values),
            'min':min(values) if values else None,'max':max(values) if values else None,
            'min_abs_nonzero':min(nz) if nz else None,'max_abs':max(nz) if nz else None}


def inspect(v, options):
    lp, s, expected = v['lp'], v['statement'], v['submitted']
    validate_statement(s,lp,identity(s),lambda:None)
    n = len(lp['c'])
    if len(lp['lower'])!=n or len(lp['upper'])!=n:
        raise ValueError('bound dimensions')
    if any(rational(a)>rational(b) for a,b in zip(lp['lower'],lp['upper'])):
        raise ValueError('inverted variable bounds')
    rows, small, conversions = [], [], 0
    for kind,key in (('E','h'),('A','b')):
        if lp[kind]['shape'][0]!=len(lp[key]):raise ValueError('RHS shape')
        for i,entries in enumerate(matrix_rows(lp[kind],n,lambda:None)):
            rebuilt=[]
            for j,value in entries:
                x=float(value)
                if not math.isfinite(x):raise ValueError('conversion overflow')
                conversions += Fraction(x)!=value
                if value:
                    rebuilt.append([j,x])
                    if 0<abs(x)<=options['small_matrix_value']:
                        small.append({'kind':kind,'row':i,'column':j,'submitted_value':x,
                                      'exact_source_value':str(value)})
            rhs=float(rational(lp[key][i]))
            rows.append({'lower':rhs if kind=='E' else '-inf','upper':rhs,'entries':rebuilt})
    rebuilt={k:[float(rational(x)) for x in lp[k]] for k in ('c','lower','upper')}
    reconstructed={'cost':rebuilt.pop('c'),'offset':float(rational(lp['offset'])),**rebuilt,'rows':rows,'sense':'minimize'}
    if reconstructed!=expected:raise ValueError('saved submitted model differs from original conversion')
    vals=[x for row in rows for j,x in row['entries']]
    rhs=[row['upper'] for row in rows]
    bounds=expected['lower']+expected['upper']
    return {'statement_sha256':identity(s),'lp_sha256':identity(lp),'submitted_sha256':identity(expected),
            'variables':n,'E_rows':len(lp['h']),'A_rows':len(lp['b']),
            'canonical_CSR_valid':True,'duplicate_indices':0,'inverted_bounds':0,
            'saved_conversion_reproduced':True,'matrix_rational_to_float_changes':conversions,
            'matrix':summary(vals),'cost':summary(expected['cost']),'variable_bounds':summary(bounds),
            'row_RHS':summary(rhs),'offset':expected['offset'],
            'small_matrix_entries':small,'small_matrix_count':len(small),
            'matrix_at_or_above_large_threshold':sum(abs(x)>=options['large_matrix_value'] for x in vals),
            'cost_at_or_above_infinite_threshold':sum(abs(x)>=options['infinite_cost'] for x in expected['cost']),
            'finite_bounds_at_or_above_infinite_threshold':sum(abs(x)>=options['infinite_bound'] for x in bounds+rhs),
            'real_native_imports':0,'optimization_calls':0,
            'scope':'saved intended submission only; no actual real-model readback is available'}


def run(options):
    a=read(ARCHIVE)
    for path,h in a['artifact_sha256'].items():
        if digest((ROOT/path).read_bytes())!=h:raise ValueError('sealed artifact changed')
    if str(SAVED.relative_to(ROOT)) not in a['artifact_sha256']:
        raise ValueError('unbound submitted input')
    return {**inspect(read(SAVED),options),'input_path':str(SAVED),
            'input_sha256':digest(SAVED.read_bytes()),'archive_sha256':digest(ARCHIVE.read_bytes())}
