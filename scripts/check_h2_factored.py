"""Solver-free audit of fixed factored-versus-expanded representation controls."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scoped_source.factored_check import check
from scoped_source.factored_io import load, referenced
from scoped_source.factored_source import Source
from scoped_source.factored_ir import assemble, block
from scoped_source.sparse_ir import index
from scoped_source.graph import clock
from scoped_source.endpoint_source_check import check as legacy_check, reconstruct, reconstruct_mc
from source_enclosure.format import identity, compact

SAME=('status','required','positive','missing','lp_bounds_checked','source_blocks_checked','duties','scopes','origins')


def equivalent(doc, old, path, *, source_sha, proof_sha, legacy_source_sha, mode):
    """Small fixed-control oracle only; not a streaming production operation."""
    tick=clock(time.monotonic()+300); m=load(path,'manifest.json')
    if identity(m)!=proof_sha or m['source_manifest_sha256']!=source_sha or m['mode']!=mode:
        raise ValueError('differential manifest identity')
    view=Source(path/'source',source_sha,tick)
    request,nodes,outputs=index(doc,legacy_source_sha,tick)
    streamed=list(view.nodes())
    if streamed!=list(nodes.items()) or view.request!=request or view.outputs!=outputs:
        raise ValueError('differential source/request/ordered nodes')
    bank,previous,scopes=reconstruct(doc,legacy_source_sha,tick)
    refs={v['name']:v for v in m['blocks']}
    if list(refs)!=list(bank) or {name:block(path,ref) for name,ref in refs.items()}!=bank:
        raise ValueError('differential source blocks')
    if m['reuse_requested']!=old['reuse_requested']: raise ValueError('differential reuse configuration')
    bases={}; records={}
    for ref in m['pairs']:
        record=referenced(path,ref); context=record['context']; pair=tuple(context['pair'])
        bases[pair]=assemble(path,refs,context['blocks'],context['guards'],tick)
        records[pair]=record
    compared=0; objectives=0
    for prior in previous['duties']:
        tick(); pair=tuple(prior['pair']); j=prior['competitor']; record=records[pair]; context=record['context']
        names=context['blocks']; forms=[]
        for expert in pair:
            ends=view.outputs[f'expert{expert}']; y=ends[request['label']]; other=ends[j]
            forms.append({'c':[str(int(v==y)-int(v==other)) for v in names],
                          'offset':str(-F(request['margin']))})
        current={'pair':list(pair),'competitor':j,'variables':names,'base':bases[pair],
                 'a':forms[0],'b':forms[1],'gate':context['gate']}
        if current!=prior: raise ValueError('differential pair/property/base/gate')
        proof=next(v for v in record['duties'] if v['competitor']==j)
        if mode=='endpoints':
            for end in proof['endpoints']:
                weight=F(end['weight'])
                lhs={**current['base'],'c':[str(F(b)+weight*(F(a)-F(b))) for a,b in zip(forms[0]['c'],forms[1]['c'])],
                     'offset':str(F(forms[1]['offset'])+weight*(F(forms[0]['offset'])-F(forms[1]['offset'])))}
                rhs={**prior['base'],'c':[str(weight*F(a)+(1-weight)*F(b)) for a,b in zip(prior['a']['c'],prior['b']['c'])],
                     'offset':str(weight*F(prior['a']['offset'])+(1-weight)*F(prior['b']['offset']))}
                if lhs!=rhs or identity(lhs)!=end['lp_sha256']: raise ValueError('differential endpoint LP')
                objectives+=1
        else:
            lhs=reconstruct_mc(current); rhs=reconstruct_mc(prior)
            if lhs!=rhs or identity(lhs)!=proof['lp_sha256']: raise ValueError('differential MC LP')
            objectives+=1
        compared+=1
    tick()
    return {'ordered_nodes':len(nodes),'pair_bases':len(bases),'properties':compared,
            'target_lps':objectives,'same_reuse_configuration':True}


def derive(root):
    bindings=load(root,'implementation.json')
    for name,digest in bindings.items():
        for path in (ROOT/name,root/'implementation'/name):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('use archived implementation')
    results=load(root,'results.json'); cfg=load(root,'protocol.json')
    if identity(cfg)!='d3120813b9613c9e4430728a77416bb3e8bb7351ca445e0f2cc87ed34997822a':
        raise ValueError('frozen representation protocol')
    expected=[(case,mode) for case in cfg['cases'] for mode in cfg['arms']]
    if [(r['case'],r['mode']) for r in results['records']]!=expected: raise ValueError('control inventory')
    output=[]; common={}
    for record in results['records']:
        name,mode=record['case'],record['mode']; path=root/(name+'-'+mode)
        checked=check(path,expected_source_manifest=record['source_manifest_sha256'],
                      expected_proof_manifest=record['proof_manifest_sha256'],expected_mode=mode,deadline=time.monotonic()+300)
        if checked!=record['result']: raise ValueError('recorded factored result differs')
        doc=load(root,name+'-source.json'); old=load(root,name+'-'+mode+'-legacy.json')
        if identity(old)!=record['legacy_package_sha256']: raise ValueError('legacy package identity')
        before=legacy_check(doc,old,expected_source_sha256=record['legacy_source_sha256'],expected_mode=mode,deadline=time.monotonic()+300)
        if {k:checked[k] for k in SAME}!={k:before[k] for k in SAME}: raise ValueError('complete bound/coverage differential')
        equality=equivalent(doc,old,path,source_sha=record['source_manifest_sha256'],
            proof_sha=record['proof_manifest_sha256'],legacy_source_sha=record['legacy_source_sha256'],mode=mode)
        m=load(path,'manifest.json')
        if identity(m)!=record['proof_manifest_sha256']: raise ValueError('manifest changed during audit')
        if len(compact(old))!=record['legacy_package_bytes']: raise ValueError('legacy byte count')
        contexts=[referenced(path,v)['context'] for v in m['pairs']]
        facts=(record['source_manifest_sha256'],contexts,m['reuse_requested'])
        if name in common and common[name]!=facts: raise ValueError('arm source/base/gate/reuse difference')
        common[name]=facts
        output.append({'case':name,'mode':mode,'source_manifest_sha256':record['source_manifest_sha256'],
            'proof_manifest_sha256':record['proof_manifest_sha256'],'status':checked['status'],
            'required':checked['required'],'positive':checked['positive'],'checked_bounds':checked['lp_bounds_checked'],
            'source_blocks':checked['source_blocks_checked'],'pair_bases':checked['pair_bases_reconstructed'],
            'package_bytes':checked['package_bytes'],'legacy_package_bytes':record['legacy_package_bytes'],
            'same_complete_results':True,'exact_representation_comparison':equality,
            'producer_reported_proposal_stats':m['stats']})
    return {'schema':'H2_FACTORED_ARCHIVE_V1','root':str(root),'status':'PASS','source_bindings':bindings,
            'controls':output,'real_requests':0,'new_solves_during_check':0,
            'hard_budget_supervision':False,'portable_check_complete':False,'performance_claim':False,
            'scope':'Fixed synthetic source/LP representation equivalence, not real capacity or native execution proof.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--output',type=Path); g.add_argument('--check',type=Path)
    args=p.parse_args(); result=derive(args.root)
    if args.output:
        with args.output.open('x') as f: json.dump(result,f,sort_keys=True,indent=2); f.write('\n')
    elif json.loads(args.check.read_text())!=result: raise ValueError('archive differs')
    print(json.dumps({'status':result['status'],'arms':len(result['controls']),'new_solves':0}))
