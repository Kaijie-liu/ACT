"""Standalone given-HZ control audit. No producer, solver or network access.

The companion checker is explicitly loaded so `python -B -I -S` works after
relocation without a repository import path. External callers bind both files.
"""
import argparse
from fractions import Fraction as F
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys
import time


def load_checker():
    path = Path(__file__).resolve().with_name('hz_request_store.py')
    spec = importlib.util.spec_from_file_location('hz_request_store', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def shape(layout):
    if layout == 'small': return dict(common=4, private=2, experts=3, outputs=2,
        common_outputs=3, common_eq=1, common_ine=2, private_eq=1, private_ine=2, guards=2)
    if layout == 'large': return dict(common=3984, private=2708, experts=8, outputs=10,
        common_outputs=3072, common_eq=128, common_ine=256, private_eq=384, private_ine=768, guards=12)
    raise ValueError('frozen layout')


def oracle(layout, pair, nonzero):
    """Independent index formula; never reads Row/View or producer output."""
    s = shape(layout); a,b = pair; n,k = s['common'],s['private']
    r = {j:F(0) for j in range(n)}
    r.update((n+a*k+j,F(1)) for j in range(k))
    r.update((n+b*k+j,F(-1)) for j in range(k))
    constant = F(s['outputs']*(a-b),16)+F(1,8)
    if nonzero:
        for start,length,eq,ine in ((0,n,s['common_eq'],s['common_ine']),
                (n+a*k,k,s['private_eq'],s['private_ine']),
                (n+b*k,k,s['private_eq'],s['private_ine'])):
            for j in range(eq): r[start+j%length] -= F(1,64)
            for j in range(ine): r[start+j%length] += F(1,32)
            constant -= F(ine,32)
        for j in range(s['guards']): r[j%n] += F((-1)**j,32)
    return constant-sum(map(abs,r.values())), tuple(sorted(r.items()))


def records(layout, manifest, request):
    s = shape(layout); result = []
    # JSON encoding here is the specified format, not checker arithmetic.
    import hashlib
    enc = lambda v: json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    for pair in combinations(range(s['experts']),2):
        q = dict(store=manifest,request=request,pair=list(pair),left=['1']*s['outputs'],
                 right=['-1']*s['outputs'],offset='1/8')
        h = hashlib.sha256(enc(q)).hexdigest()
        for nonzero in (False,True):
            bound,_ = oracle(layout,pair,nonzero)
            c = dict(query=h,lower=str(bound),
                equality=['1/64' if nonzero else '0']*(s['common_eq']+2*s['private_eq']),
                inequality=['-1/32' if nonzero else '0']*(s['common_ine']+2*s['private_ine']+s['guards']))
            result.append(dict(query=q,candidate=c,expected_query=h))
    return result


def check_recipe(store, layout, mod, work):
    """Check every coefficient against frozen formulae, without a flat LP."""
    s=shape(layout); n,k=s['common'],s['private']
    nc,pc=(3,1) if layout=='small' else (3856,2324)
    expected=[]
    for scope,length,continuous in [('common',n,nc)]+[(f'expert/{i}',k,pc) for i in range(s['experts'])]:
        for kind,count in [('continuous',continuous),('binary',length-continuous)]:
            for j in range(count): expected.append((f'{scope}/{kind}/{j}',kind,scope,'fixed-control'))
    mod.require(store.factors==tuple(expected) and store.experts==s['experts'],'frozen factor registry')
    def outputs(view,indices,count,constant):
        mod.require(view.output.count==count,'frozen output inventory')
        for i,row in enumerate(view.output):
            work.visit('row_visits'); work.visit('coefficient_visits',len(row.coefficients))
            mod.require(row.name==f'{view.name}/out/{i}' and row.rhs==constant
                        and row.coefficients==tuple((j,F(1)) for j in indices[i::count]),'frozen output coefficients')
    def constraints(view,kind,start,length,count):
        seq = getattr(view,kind) if view is store.common else getattr(view,kind).suffix(getattr(store.common,kind))
        seen=0
        for j,row in enumerate(seq):
            work.visit('row_visits'); work.visit('coefficient_visits',len(row.coefficients)); seen+=1
            mod.require(row.name==f'{view.name}/{kind}/{j}' and row.rhs==(0 if kind=='eq' else 1)
                        and row.coefficients==((start+j%length,F(1)),),'frozen constraint coefficients')
        mod.require(seen==count,'frozen constraint inventory')
    outputs(store.common,list(range(n)),s['common_outputs'],F(0))
    for kind in ('eq','ine'): constraints(store.common,kind,0,n,s[f'common_{kind}'])
    for i,view in enumerate(store.templates):
        outputs(view,list(range(n))+list(range(n+i*k,n+(i+1)*k)),s['outputs'],F(i+1,16))
        for kind in ('eq','ine'): constraints(view,kind,n+i*k,k,s[f'private_{kind}'])
    for pair,entry in store.entries.items():
        seen=0
        for j,row in enumerate(entry.ine.suffix(store.common.ine)):
            work.visit('row_visits'); work.visit('coefficient_visits',len(row.coefficients)); seen+=1
            mod.require(row.name==f'pair/{pair[0]}/{pair[1]}/guard/{j}' and row.rhs==0
                        and row.coefficients==((j%n,F((-1)**j)),),'frozen guard coefficients')
        mod.require(seen==s['guards'],'frozen guard inventory')


def inspect(store_path, manifest, request, layout, *, work=None, saved_records=None):
    mod = load_checker(); started = time.monotonic()
    if work is None: work = mod.Work(started+300)
    store = mod.Store(store_path,expected_manifest=manifest,expected_request=request,work=work)
    mod.require(request==mod.identity(dict(control='REQUEST_OWNED_HZ_STORE_V1',layout=layout)), 'frozen request identity')
    check_recipe(store,layout,mod,work)
    s = shape(layout); pairs = list(combinations(range(s['experts']),2))
    expected = records(layout,manifest,request)
    if saved_records is not None: mod.require(saved_records == expected, 'complete fixed control records')
    results = []; sharing = []
    mod.require(store.stats['factors'] == s['common']+s['experts']*s['private']
                and store.stats['referenced_factors'] == store.stats['factors'], 'actual factor use')
    for pair in pairs:
        entry,a,b,active = store.pair(pair)
        mod.require(len(active) == s['common']+2*s['private'], 'full active pair')
        nc,pc=(3,1) if layout=='small' else (3856,2324)
        continuous=sum(store.factors[j][1]=='continuous' for j in active)
        mod.require(continuous==nc+2*pc, 'full pair factor kinds')
        mod.require(a.factors & b.factors == store.common.factors, 'private factor disjointness')
        mod.require(entry.eq.parent is store.common.eq and entry.ine.parent is store.common.ine
                    and a.eq.parent is store.common.eq and b.eq.parent is store.common.eq,
                    'actual inherited row sharing')
        for kind in ('eq','ine'):
            full = list(store.constraints(pair,kind))
            common = list(getattr(store.common,kind))
            mod.require(all(full[i] is row for i,row in enumerate(common)), 'row object sharing')
            mod.require(len({row.name for row in full}) == len(full), 'pair row duplicate')
        count = sum(1 for k in ('eq','ine') for _ in store.constraints(pair,k))
        mod.require(count == s['common_eq']+s['common_ine']+2*(s['private_eq']+s['private_ine'])+s['guards'], 'full pair row count')
        sharing.append(dict(pair=list(pair),active=len(active),continuous=continuous,
                            binary=len(active)-continuous,constraints=count))
    for i,item in enumerate(expected):
        value,residual = mod.check_candidate(store,item['query'],item['candidate'],
                                             expected_query=item['expected_query'],work=work)
        bound,reference = oracle(layout,item['query']['pair'],bool(i%2))
        mod.require((value,residual) == (bound,reference), 'independent complete analytic residual')
        results.append(dict(pair=item['query']['pair'],nonzero=bool(i%2),lower=str(value),
            query=item['expected_query'],residual_sha256=mod.identity([[j,str(v)] for j,v in residual])))
    work.tick()
    return dict(status='CHECKED_GIVEN_HZ_CONTROL_NOT_NETWORK_PROOF',layout=layout,
        manifest=manifest,request=request,stats=dict(store.stats),pairs=sharing,results=results,
        work_counts=dict(work.counts),seconds=time.monotonic()-started,
        forbidden_modules=sorted(set(sys.modules)&{'torch','scipy','numpy','gurobipy','highspy'}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--store',required=True)
    parser.add_argument('--manifest',required=True); parser.add_argument('--request',required=True)
    parser.add_argument('--layout',choices=('small','large'),required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.store,args.manifest,args.request,args.layout),sort_keys=True))


if __name__ == '__main__': main()
