"""Diagnostic-only independent expansion. Never on the block execution path."""
from copy import deepcopy
from fractions import Fraction as F

from act.back_end.moe.check_hz_endpoints import _source
from act.back_end.moe.check_batched_support import validated_records, check_batch as check_flat_batch
from act.back_end.moe.check_block_support import check_batch
from scoped_source.rowwise_bound import clock, identity, rational, rows, check_bound


def csr(matrix, width):
    out={'shape':[len(matrix),width],'indptr':[0],'indices':[],'data':[]}
    for row in matrix:
        for j,v in sorted(row.items()):
            if v:out['indices'].append(j);out['data'].append(str(v))
        out['indptr'].append(len(out['data']))
    return out


def normalized(base, tick):
    n=len(base['lower'])
    return {'matrix_format':'csr_v1',
            **{k:[str(rational(v)) for v in base[k]] for k in ('b','h','lower','upper')},
            **{k:csr([{j:v for j,v in row if v} for row in rows(base[k],(len(base[rhs]),n),tick)],n)
               for k,rhs in (('A','b'),('E','h'))}}


def expand(batch, tick):
    # Independently derive local-to-global indices and spans from source shapes.
    parsed={k:_source(v,tick) for k,v in batch['sources'].items()}
    common,sc,sb=parsed['common'];a,ac,ab=parsed['a'];b,bc,bb=parsed['b']
    nc,nb=ac+bc-sc,ab+bb-sb;n=nc+nb
    cm={'entry':list(range(sc)),'a':list(range(ac)),'b':list(range(sc))+list(range(ac,nc))}
    bm={'entry':list(range(sb)),'a':list(range(ab)),'b':list(range(sb))+list(range(ab,nb))}
    out={'matrix_format':'csr_v1','lower':['-1']*n,'upper':['1']*n};spans=[]
    for key,ck,bk,rhs,result_rhs in (('A','Auc','Aub','ub','b'),('E','Ac','Ab','b','h')):
        matrix=[];right=[];sizes=[]
        for name in ('entry','a','b'):
            h=parsed[name][0];start=0 if name=='entry' else len(common[rhs])
            sizes.append(len(h[rhs])-start)
            for cr,br in zip(h[ck][start:],h[bk][start:]):
                matrix.append({cm[name][j]:v for j,v in cr.items()} | {nc+bm[name][j]:v for j,v in br.items()})
            right.extend(map(str,h[rhs][start:]))
        out[key]=csr(matrix,n);out[result_rhs]=right;spans.append(sizes)
    objectives=[]
    for q in batch['queries']:
        vector=[F(0)]*n;constant=rational(q['offset']);count=len(a['c']);weights=list(map(rational,q['q']))
        for name,weights_part in (('a',weights[:count]),('b',weights[count:])):
            h=parsed[name][0]
            for weight,center,cr,br in zip(weights_part,h['c'],h['Gc'],h['Gb']):
                constant+=weight*center
                for j,v in cr.items():vector[cm[name][j]]+=weight*v
                for j,v in br.items():vector[nc+bm[name][j]]+=weight*v
        sign=1 if q['side']=='min' else -1
        objectives.append((list(map(str,[sign*v for v in vector])),str(sign*constant)))
    return out,objectives,spans


def compare_pair(old, block, old_candidate, block_candidate, deadline):
    tick=clock(deadline);ob=old['batch'];bb=block['batch']
    base,objects,sizes=expand(bb,tick)
    records=validated_records(ob,expected_batch_sha256=identity(ob),deadline=deadline)
    check_flat_batch(ob,old_candidate,expected_batch_sha256=identity(ob),deadline=deadline)
    old_by_id={item['id']:item for item in old_candidate['entries']}
    if (base!=normalized(ob['base'],tick) or ob['queries']!=bb['queries']
            or old['gate']['bounds']!=block['gate']['bounds'] or old['pair']!=block['pair']
            or ob['n_relaxed_binaries']!=bb['layout']['n_bin']):
        raise ValueError('full block/flat domain/objective differential')
    if objects!=[(q['c'],q['constant']) for q in ob['queries']]:raise ValueError('independent property projection')
    exchanges=[]
    for direction in ('flat_to_block','block_to_flat'):
        bc=deepcopy(block_candidate)
        if direction=='flat_to_block':
            for item,(_,lp) in zip(bc['entries'],records):
                olditem=old_by_id[item['id']]
                cert=olditem['certificate'];yi=ti=0
                if cert['lp_sha256']!=identity(lp):raise ValueError('original flat candidate objective identity')
                for j,name in enumerate(('entry','a','b')):
                    ny,nt=sizes[0][j],sizes[1][j]
                    item['duals'][j]={'source':name,'y':cert['inequality_dual'][yi:yi+ny],
                                      't':cert['equality_dual'][ti:ti+nt]};yi+=ny;ti+=nt
                item['claimed_lower_bound']=cert['claimed_lower_bound']
        checked=check_batch(bb,bc,expected_batch_sha256=identity(bb),deadline=deadline)
        for (q,lp),item,got in zip(records,bc['entries'],checked['results']):
            cert={'lp_sha256':identity(lp),'claimed_lower_bound':item['claimed_lower_bound'],
                  'inequality_dual':[v for d in item['duals'] for v in d['y']],
                  'equality_dual':[v for d in item['duals'] for v in d['t']]}
            flat=check_bound(lp,cert,deadline=deadline)
            if (flat['residual']!=got['residual'] or flat['checked_lower_bound']!=got['checked_lower_bound']):
                raise ValueError('same-candidate exact residual/lower differential')
            exchanges.append({'pair':old['pair'],'query':q['id'],'direction':direction,
                              'dual_sha256':identity(item['duals']),
                              'residual_sha256':identity(flat['residual']),'bound':flat['checked_lower_bound']})
    return {'pair':old['pair'],'base_sha256':identity(base),'queries_sha256':identity(ob['queries']),
            'global_factors':len(base['lower']),'rows':len(base['b'])+len(base['h']),
            'block_constraint_nonzeros':len(base['A']['data'])+len(base['E']['data']),
            'snapshot_constraint_entries':sum(len(src[k]['data']) for name,src in bb['sources'].items()
                                              if name!='common' for k in ('Ac','Ab','Auc','Aub')),
            'crosschecks':exchanges}


def compare_sources(old,block,deadline):
    for key in ('input','router','common','templates'):
        if old[key]!=block[key]:raise ValueError('fresh source/template differential')
    x,y=old['endpoint_request'],block['endpoint_request']
    if any(x[k]!=y[k] for k in ('experts','classes','properties')) or len(x['pairs'])!=len(y['pairs']):
        raise ValueError('source roster differential')
    return [compare_pair(a,b,ap['candidates'],bp['candidates'],deadline)
            for a,b,ap,bp in zip(x['pairs'],y['pairs'],old['proof']['pairs'],block['proof']['pairs'])]
