"""Untrusted factored H2 producer; fixed mathematics, one pair basis at a time.

Capacity observer version of the frozen builder. Removing observer statements
recovers the original build AST; every proposer still receives an isolated copy.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import time
from scoped_source.graph import clock
from scoped_source.sparse_ir import row
from scoped_source.factored_source import Source
from scoped_source.factored_io import write, HEADER_LIMIT
from scoped_source.factored_ir import assemble
from scoped_source.endpoint_source_build import mc_lp, common_certificate, THRESHOLD
from source_enclosure.format import identity


def build(root, *, expected_source_manifest, mode, reuse_keys=(), deadline, proposer=None, observer=None):
    _observe=observer or (lambda *args,**kwargs: None)
    _observe('source_begin')
    tick=clock(deadline); root=Path(root)
    if mode not in ('endpoints','mccormick'): raise ValueError('factored arm')
    if (root/'manifest.json').exists(): raise FileExistsError('never overwrite a proof')
    source=Source(root/'source',expected_source_manifest,tick); r=source.request
    pairs=list(combinations(range(r['experts']),2))
    keys=[(p,j) for p in pairs for j in range(r['classes']) if j!=r['label']]
    reuse=set(reuse_keys)
    if len(reuse)!=len(reuse_keys) or not reuse<=set(keys): raise ValueError('reuse inventory')
    refs={}; bounds={}; le_counts={}
    for name,node in source.nodes():
        tick(); rows=[]
        if node['kind']=='input': lo,hi=node['bounds']
        elif node['kind']=='affine':
            lo=hi=node['bias']
            for parent,v in node['terms'].items():
                a,b=bounds[parent]; lo+=min(v*a,v*b); hi+=max(v*a,v*b)
            rows=[row('eq',{name:F(1),**{p:-v for p,v in node['terms'].items()}},node['bias'])]
        else:
            parent=node['parent']; a,b=bounds[parent]; lo,hi=max(F(0),a),max(F(0),b)
            if b<=0: rows=[row('eq',{name:F(1)},F(0))]
            elif a>=0: rows=[row('eq',{name:F(1),parent:F(-1)},F(0))]
            else:
                slope=b/(b-a)
                rows=[row('le',{name:F(-1)},F(0)),row('le',{parent:F(1),name:F(-1)},F(0)),
                      row('le',{name:F(1),parent:-slope},-slope*a)]
        bounds[name]=(lo,hi); le_counts[name]=sum(v['sense']=='le' for v in rows)
        refs[name]={'name':name,**write(root,f'b/{len(refs):06d}.json',{'bounds':list(map(str,(lo,hi))),'rows':rows})}
        _observe('source_block_published',node=name,blocks=len(refs))
    _observe('source_blocks_complete',blocks=len(refs))
    outputs=source.outputs; pair_refs=[]; errors=[]; stats={'native_proposals':0,'common_proposals':0}
    def propose(lp,conflict,origin,key):
        tick()
        if origin!='PROPOSED':
            stats['common_proposals']+=1
            return common_certificate(lp,conflict if origin=='EMPTY_GUARD_DUAL' else None)
        if proposer is None: return None
        stats['native_proposals']+=1
        try: result=deepcopy(proposer(deepcopy(lp),time_limit=max(0.,deadline-time.monotonic())))
        except (ValueError,RuntimeError) as error:
            errors.append({'duty':list(key),'error':str(error)}); result=None
        tick(); return result
    for ordinal,pair in enumerate(pairs):
        tick(); a,b=pair
        _observe('pair_begin',pair=list(pair))
        names=[v for v in refs if v.split('/')[0] in ('input','router',f'expert{a}',f'expert{b}')]
        ra,rb=(outputs['router'][i] for i in pair)
        low,high=bounds[ra][0]-bounds[rb][1],bounds[ra][1]-bounds[rb][0]
        gate=['1/2','1/2'] if low==high==0 else ['0','1/2'] if high<=0 else ['1/2','1'] if low>=0 else ['0','1']
        guards=[]; conflict=None; inequalities=sum(le_counts[v] for v in names)
        for selected in pair:
            for other in range(r['experts']):
                if other in pair: continue
                s,o=outputs['router'][selected],outputs['router'][other]
                gap=bounds[o][0]-bounds[s][1]
                if conflict is None and gap>0:
                    conflict={'selected':selected,'outside':other,'inequality_index':inequalities,'gap':str(gap)}
                guards.append(row('le',{o:F(1),s:F(-1)},F(0))); inequalities+=1
        base=assemble(root,refs,names,guards,tick)
        context={'pair':list(pair),'blocks':names,'guards':guards,'gate':gate,
                 'router_margin':list(map(str,(low,high))),'guard_conflict':conflict,'base_sha256':identity(base)}
        context_hash=identity(context); records=[]
        _observe('pair_assembled',pair=list(pair),variables=len(names),base_sha256=context['base_sha256'],inequalities=base['A']['shape'][0],equalities=base['E']['shape'][0],nonzeros=len(base['A']['data'])+len(base['E']['data']))
        for j in range(r['classes']):
            if j==r['label']: continue
            tick(); forms=[]; facts=[]
            for expert in pair:
                y,k=outputs[f'expert{expert}'][r['label']],outputs[f'expert{expert}'][j]
                forms.append({'c':[str(int(v==y)-int(v==k)) for v in names],'offset':str(-F(r['margin']))})
                facts.append(bounds[y][0]-bounds[k][1]-F(r['margin']))
            origin='EMPTY_GUARD_DUAL' if conflict is not None else \
                   'SOURCE_BOX_REUSE' if (pair,j) in reuse and min(facts)>THRESHOLD else 'PROPOSED'
            duty={'pair':list(pair),'competitor':j,'variables':names,'base':base,'a':forms[0],'b':forms[1],'gate':gate}
            record={'competitor':j,'origin':origin,'pair_context_sha256':context_hash}
            _observe('property_begin',pair=list(pair),competitor=j,origin=origin,mode=mode)
            if mode=='endpoints':
                ends=[]
                for weight in sorted(set(map(F,gate))):
                    lp={**base,'c':[str(F(b)+weight*(F(a)-F(b))) for a,b in zip(forms[0]['c'],forms[1]['c'])],
                        'offset':str(F(forms[1]['offset'])+weight*(F(forms[0]['offset'])-F(forms[1]['offset'])))}
                    _observe('endpoint_begin',pair=list(pair),competitor=j,weight=str(weight),origin=origin)
                    cert=propose(lp,conflict,origin,(pair,j)); tick()
                    ends.append({'weight':str(weight),'lp_sha256':identity(lp),'certificate':cert})
                record['endpoints']=ends
            else:
                _observe('mccormick_begin',pair=list(pair),competitor=j,origin=origin)
                lp=mc_lp(duty)
                if origin=='SOURCE_BOX_REUSE':
                    stats['common_proposals']+=1; cert={'kind':'SOURCE_BOX_FACT','lower_bound':str(min(facts))}
                else: cert=propose(lp,conflict,origin,(pair,j))
                record.update(lp_sha256=identity(lp),certificate=cert)
            _observe('property_record_complete',pair=list(pair),competitor=j)
            records.append(record); del lp,duty,forms
        pair_refs.append({'pair':list(pair),**write(root,f'p/{ordinal:06d}.json',{'context':context,'duties':records})})
        _observe('pair_published',pair=list(pair),properties=len(records))
        del base,records
    manifest={'schema':'H2_FACTORED_PROOF_V1','source_manifest_sha256':expected_source_manifest,
              'mode':mode,'reuse_requested':[[list(p),j] for p,j in sorted(reuse)],
              'blocks':list(refs.values()),'pairs':pair_refs,'stats':stats,'proposal_errors':errors}
    _observe('proof_manifest_begin',pairs=len(pair_refs))
    tick(); write(root,'manifest.json',manifest,HEADER_LIMIT); tick(); return identity(manifest)
