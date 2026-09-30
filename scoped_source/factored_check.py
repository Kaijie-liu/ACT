"""Independent factored source and all-duty checker; stdlib, no producer/solver.

Rebuild source rules once and each pair once, stream properties. Hashes bind
stored members; all affine/ReLU/guard/gate/property semantics are derived here.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from scoped_source.graph import clock
from scoped_source.factored_source import Source
from scoped_source.factored_io import load, referenced, inventory, HEADER_LIMIT
from scoped_source.factored_ir import block, assemble
from scoped_source.sparse_ir import row
from scoped_source.sparse_check import check_bound
from scoped_source.endpoint_source_check import reconstruct_mc
from scoped_source.endpoint_check import exact, THRESHOLD
from source_enclosure.format import identity


def check(root, *, expected_source_manifest, expected_proof_manifest, expected_mode, deadline):
    tick=clock(deadline); root=Path(root)
    m=load(root,'manifest.json',HEADER_LIMIT)
    if (set(m)!={'schema','source_manifest_sha256','mode','reuse_requested','blocks','pairs','stats','proposal_errors'} or
            m['schema']!='H2_FACTORED_PROOF_V1' or identity(m)!=expected_proof_manifest or
            m['source_manifest_sha256']!=expected_source_manifest or m['mode']!=expected_mode or
            expected_mode not in ('endpoints','mccormick')): raise ValueError('factored manifest/arm identity')
    source=Source(root/'source',expected_source_manifest,tick); r=source.request
    pairs=list(combinations(range(r['experts']),2))
    keys=[(p,j) for p in pairs for j in range(r['classes']) if j!=r['label']]
    reuse=[]
    for item in m['reuse_requested']:
        if (type(item) is not list or len(item)!=2 or type(item[0]) is not list or len(item[0])!=2 or
                any(type(v) is not int for v in item[0]) or type(item[1]) is not int): raise ValueError('reuse syntax')
        reuse.append((tuple(item[0]),item[1]))
    if reuse!=sorted(set(reuse)) or not set(reuse)<=set(keys): raise ValueError('reuse inventory')
    refs={}; bounds={}; le_counts={}; actual_nodes=0
    for i,(name,node) in enumerate(source.nodes()):
        tick(); actual_nodes+=1
        if i>=len(m['blocks']): raise ValueError('missing source block')
        ref=m['blocks'][i]
        if ref.get('name')!=name or ref.get('file')!=f'b/{i:06d}.json': raise ValueError('block order/source binding')
        rows=[]
        if node['kind']=='input': lower,upper=node['bounds']
        elif node['kind']=='affine':
            lower=upper=node['bias']; coefficients={name:F(1)}
            for parent,v in node['terms'].items():
                a,b=bounds[parent]
                lower+=v*(a if v>=0 else b); upper+=v*(b if v>=0 else a); coefficients[parent]=-v
            rows=[row('eq',coefficients,node['bias'])]
        elif node['kind']=='relu':
            parent=node['parent']; a,b=bounds[parent]; lower,upper=max(F(0),a),max(F(0),b)
            if b<=0: rows=[row('eq',{name:F(1)},F(0))]
            elif a>=0: rows=[row('eq',{parent:F(-1),name:F(1)},F(0))]
            else: rows=[row('le',{name:F(-1)},F(0)),row('le',{name:F(-1),parent:F(1)},F(0)),
                        row('le',{name:F(1),parent:b/(a-b)},a*b/(a-b))]
        else: raise ValueError('source rule')
        if block(root,ref)!={'bounds':list(map(str,(lower,upper))),'rows':rows}:
            raise ValueError('source range/row mismatch')
        refs[name]=ref; bounds[name]=(lower,upper); le_counts[name]=sum(v['sense']=='le' for v in rows)
    if len(m['blocks'])!=actual_nodes or len(m['pairs'])!=len(pairs): raise ValueError('complete source/pair inventory')
    # Exact filesystem inventory, including all source chunks, without parsing them again.
    expected={'manifest.json'}|{v['file'] for v in refs.values()}|{f'p/{i:06d}.json' for i in range(len(pairs))}
    expected.update('source/'+name for name in source.files)
    package_bytes=inventory(root,expected)
    outputs=source.outputs; results=[]; origins=[]; scopes=[]; checked=0; missing=0
    for ordinal,pair in enumerate(pairs):
        tick(); ref=m['pairs'][ordinal]
        if (set(ref)!={'pair','file','bytes','sha256'} or ref['pair']!=list(pair) or
                any(type(v) is not int for v in ref['pair']) or ref['file']!=f'p/{ordinal:06d}.json'):
            raise ValueError('pair reference/coverage')
        received=referenced(root,ref)
        if set(received)!={'context','duties'}: raise ValueError('pair record fields')
        families={'input','router'}|{f'expert{v}' for v in pair}
        names=[name for name in refs if name.split('/')[0] in families]
        left,right=(outputs['router'][v] for v in pair)
        lo=bounds[left][0]-bounds[right][1]; hi=bounds[left][1]-bounds[right][0]
        if lo==hi==0: gate=['1/2','1/2']
        elif hi<=0: gate=['0','1/2']
        elif lo>=0: gate=['1/2','1']
        else: gate=['0','1']
        guards=[]; conflict=None; count=sum(le_counts[v] for v in names)
        for selected in pair:
            for outside in range(r['experts']):
                if outside in pair: continue
                s,o=outputs['router'][selected],outputs['router'][outside]
                guards.append(row('le',{s:F(-1),o:F(1)},F(0)))
                if conflict is None and bounds[o][0]>bounds[s][1]:
                    conflict={'selected':selected,'outside':outside,'inequality_index':count,
                              'gap':str(bounds[o][0]-bounds[s][1])}
                count+=1
        base=assemble(root,refs,names,guards,tick)
        context={'pair':list(pair),'blocks':names,'guards':guards,'gate':gate,
                 'router_margin':list(map(str,(lo,hi))),'guard_conflict':conflict,'base_sha256':identity(base)}
        if received['context']!=context: raise ValueError('pair base/guard/gate/map reconstruction')
        context_hash=identity(context); competitors=[j for j in range(r['classes']) if j!=r['label']]
        if len(received['duties'])!=len(competitors): raise ValueError('complete property inventory')
        for j,record in zip(competitors,received['duties']):
            tick(); forms=[]; facts=[]
            for expert in pair:
                ends=outputs[f'expert{expert}']; coefficients={ends[r['label']]:F(1),ends[j]:F(-1)}
                forms.append({'c':[str(coefficients.get(v,F(0))) for v in names],'offset':str(-F(r['margin']))})
                facts.append(bounds[ends[r['label']]][0]-bounds[ends[j]][1]-F(r['margin']))
            origin='EMPTY_GUARD_DUAL' if conflict is not None else \
                   'SOURCE_BOX_REUSE' if (pair,j) in reuse and min(facts)>THRESHOLD else 'PROPOSED'
            fields={'competitor','origin','pair_context_sha256'}|({'endpoints'} if expected_mode=='endpoints' else {'lp_sha256','certificate'})
            if (set(record)!=fields or type(record['competitor']) is not int or record['competitor']!=j or
                    record['origin']!=origin or record['pair_context_sha256']!=context_hash): raise ValueError('property/fact binding')
            duty={'pair':list(pair),'competitor':j,'variables':names,'base':base,'a':forms[0],'b':forms[1],'gate':gate}
            if expected_mode=='endpoints':
                weights=sorted(set(map(F,gate))); values=[]
                if len(record['endpoints'])!=len(weights): raise ValueError('complete endpoint inventory')
                for t,end in zip(weights,record['endpoints']):
                    if set(end)!={'weight','lp_sha256','certificate'} or exact(end['weight'])!=t:
                        raise ValueError('endpoint weight identity')
                    lp={**base,'c':[str(t*F(a)+(1-t)*F(b)) for a,b in zip(forms[0]['c'],forms[1]['c'])],
                        'offset':str(t*F(forms[0]['offset'])+(1-t)*F(forms[1]['offset']))}
                    if identity(lp)!=end['lp_sha256']: raise ValueError('endpoint LP identity')
                    if end['certificate'] is None: values.append(None); missing+=1
                    else: values.append(F(check_bound(lp,end['certificate'])['checked_lower_bound'])); checked+=1
                    tick()
                lower=None if None in values else min(values)
                extra={'endpoint_bounds':[None if v is None else str(v) for v in values]}
            else:
                lp=reconstruct_mc(duty)
                if identity(lp)!=record['lp_sha256']: raise ValueError('MC LP identity')
                cert=record['certificate']; extra={}
                if origin=='SOURCE_BOX_REUSE':
                    lower=min(facts)
                    if cert!={'kind':'SOURCE_BOX_FACT','lower_bound':str(lower)}: raise ValueError('source mixture fact')
                elif cert is None: lower=None; missing+=1
                else: lower=F(check_bound(lp,cert)['checked_lower_bound']); checked+=1
            tick(); results.append({'pair':list(pair),'competitor':j,**extra,
                                  'lower_bound':None if lower is None else str(lower),'positive':lower is not None and lower>THRESHOLD})
            scopes.append({'pair':list(pair),'competitor':j,'router_margin':list(map(str,(lo,hi))),
                           'facts':list(map(str,facts)),'fact_domain':'GLOBAL_INPUT_BOX','guard_conflict':conflict})
            origins.append(origin); del lp,duty,forms
        del base,received
    if inventory(root,expected)!=package_bytes: raise ValueError('package inventory changed during check')
    tick(); positive=sum(v['positive'] for v in results)
    return {'status':'CHECKED_DECLARED_SOURCE_POSITIVE' if positive==len(keys) else
                    'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE',
            'required':len(keys),'positive':positive,'missing':missing,'lp_bounds_checked':checked,
            'source_blocks_checked':actual_nodes,'duties':results,'origins':origins,'scopes':scopes,
            'source_manifest_sha256':expected_source_manifest,'proof_manifest_sha256':expected_proof_manifest,
            'mode':expected_mode,'pair_bases_reconstructed':len(pairs),'package_bytes':package_bytes,
            'routing_coverage':'ALL_UNORDERED_TOP2_PAIRS_NO_DUTIES_DROPPED',
            'hard_budget_supervision':False,'deployed_float_SAFE':False,
            'remaining_trust':['declared_graph_correspondence_to_native_program','source_parser_and_exact_checker_implementation']}
