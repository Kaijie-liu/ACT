# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded mathematical prototype: shared exact input forms, full native costs."""
from collections import Counter
from fractions import Fraction as F
import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import T
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import native_row, project_outputs, extend_actual
from experiments.neural_hz_20260831.c88_inline_tile_v1 import actual_rows
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c96_word_row_v1 import construct, _row

MODES = ('dense', 'scaled', 'centered', 'masked', 'shared')
GRIDS = ((1,1), (1,2), (2,2), (2,3), (3,3))


def key_form(terms):
    """Actual column/coefficient equality, with only a common sign removed."""
    values = {}
    for col, value in terms:
        values[col] = values.get(col, F(0)) + F(value)
    values = tuple((c,v) for c,v in sorted(values.items()) if v)
    sign = -1 if values and values[0][1] < 0 else 1
    return tuple((c,v*sign) for c,v in values), sign


def word_terms(key):
    """Exact finite dyadic words for the unchanged C96 native-row constructor."""
    terms = []
    for col, value in key:
        denominator = value.denominator
        if denominator & (denominator-1):
            raise ValueError('partial coefficients must be dyadic')
        terms.append((col, value.numerator, -(denominator.bit_length()-1)))
    return terms


def fixture(channels, outputs, mode, *, grid=(2,2), pool, enabled=False):
    """Construct original C96 circuits and the complete proposed exact rewrite."""
    if not enabled:
        return None
    if mode not in MODES or channels not in (1,3) or outputs not in (2,4):
        raise ValueError('unregistered ordinary proof geometry')
    gy,gx = grid
    if grid not in GRIDS:
        raise ValueError('unregistered tile grid')
    h,w = 2*gy+2, 2*gx+2
    oh,ow = h-2,w-2
    base_inputs = channels*h*w
    ids = np.arange(base_inputs, dtype=np.int64).reshape(channels,h,w)
    powers = np.zeros(ids.shape, np.int32)
    if mode == 'scaled': powers = (np.arange(ids.size).reshape(ids.shape)%3).astype(np.int32)
    if mode == 'shared' and channels > 1: ids[1:] = ids[0]
    if mode == 'masked': ids[np.indices(ids.shape).sum(axis=0)%4 == 0] = -1
    centers = np.zeros(ids.shape, np.float64)
    if mode == 'centered': centers = ((np.arange(ids.size).reshape(ids.shape)%5)-2)/8.
    centers[ids < 0] = 0
    weight = np.empty((outputs,channels,3,3), np.float32)
    for k,c in np.ndindex(outputs,channels):
        weight[k,c] = (-1 if (k+c)%2 else 1)*(1+k+3*c)*np.arange(1,10).reshape(3,3)/32.
    bias = np.arange(outputs, dtype=np.float64)/16 if mode == 'centered' else np.zeros(outputs)
    out_ids = np.arange(base_inputs, base_inputs+outputs*oh*ow).reshape(outputs,oh,ow)
    alias_id = base_inputs+out_ids.size
    base = alias_id+1
    pool.charge('c109_complete_fixture_plan_and_proof_reserve', 500000+5000*(channels+outputs)*gy*gx)
    _, transformed = prepare_words(weight, pool=pool, enabled=True)
    if transformed is None or not transformed.dense:
        raise ValueError('complete ordinary dense transformed filter required')
    aux, output_rows, descriptions, expected = [], [], {}, []
    for y in range(0,oh,2):
        for x in range(0,ow,2):
            tile_ids, tile_powers = ids[:,y:y+4,x:x+4], powers[:,y:y+4,x:x+4]
            oi = out_ids[:,y:y+2,x:x+2]
            rep, packet = construct(transformed,tile_ids,tile_powers,oi,
                np.full(oi.shape,20,np.int32),base+len(aux),pool=pool,enabled=True)
            local, emitted = actual_rows(rep,packet)
            v_index = 0
            for c,a,b in np.ndindex(channels,4,4):
                pairs, raw_terms = [], []
                for j in range(4):
                    if not T[b,j]: continue
                    terms = [(int(tile_ids[c,i,j]), F(int(T[a,i]))*F(2)**int(tile_powers[c,i,j]))
                             for i in range(4) if T[a,i] and tile_ids[c,i,j] >= 0]
                    key, sign = key_form(terms)
                    pairs.append((key,sign*int(T[b,j])))
                    raw_terms.extend(terms)
                # Every full output is present and the transformed kernel is dense;
                # hence any multi-term V form has at least two actual users.
                if len(raw_terms) > 1:
                    row = local[v_index]
                    descriptions[row['slot']] = pairs
                    v_index += 1
            if v_index != rep['kept_v']:
                raise ValueError('actual C96 input-form coverage differs')
            for k,i,j in np.ndindex(outputs,2,2):
                col = int(oi[k,i,j]); poly = {col:F(2)**20}; constant=F(float(bias[k]))
                for c,dy,dx in np.ndindex(channels,3,3):
                    source = int(ids[c,y+i+dy,x+j+dx])
                    if source < 0: continue
                    coefficient = F(float(weight[k,c,dy,dx]))
                    constant += coefficient*F(float(centers[c,y+i+dy,x+j+dx]))
                    poly[source] = poly.get(source,F(0))-coefficient*F(2)**int(powers[c,y+i+dy,x+j+dx])
                if constant: poly[-1] = -constant
                expected.append({c:v for c,v in poly.items() if v})
                row = emitted[(k*2+i)*2+j]
                row['rhs'] = float(constant*F(2)**row['gauge'])
                if F(row['rhs']) != constant*F(2)**row['gauge']:
                    raise ValueError('ordinary source center must remain exact')
            aux.extend(local); output_rows.extend(emitted)
    counts = Counter(key for pairs in descriptions.values() for key,_ in pairs if len(key)==2)
    chosen = tuple(sorted(key for key,count in counts.items() if count>=4))
    if len(chosen)>16384:
        raise MemoryError('whole partial auxiliary budget exceeded')
    partials, routes = [], {}
    for key in chosen:
        slot = base+len(partials)
        row = _row([(c,-n,e) for c,n,e in word_terms(key)],slot,pool=pool)
        partials.append(dict(coefficients=tuple(zip(map(int,row['columns']),map(float,row['native']))),
            rhs=0., slot=slot, gauge=row['gauge']))
        routes[key] = (slot,F(2)**row['pivot_power'])
    shift = len(partials)
    remap = lambda col: col+shift if col>=base else col
    new_aux = list(partials)
    for old in aux:
        slot=old['slot']; gauge=F(2)**old['gauge']
        if slot in descriptions:
            terms={}
            for key,sign in descriptions[slot]:
                parts=[(routes[key][0],routes[key][1])] if key in routes else key
                for col,value in parts:
                    terms[col]=terms.get(col,F(0))-sign*value*gauge
            terms[remap(slot)]=F(dict(old['coefficients'])[slot])
            rewritten=native_row(sorted(terms.items()),F(old['rhs']),slot=remap(slot))
            # Its actual stored gauge includes the original literal gauge.
            rewritten['gauge'] += old['gauge']
        else:
            rewritten=dict(old,slot=remap(slot),coefficients=tuple((remap(c),v) for c,v in old['coefficients']))
        new_aux.append(rewritten)
    new_out=[dict(row,slot=remap(row['slot']) if row.get('slot') is not None else None,
        coefficients=tuple((remap(c),v) for c,v in row['coefficients'])) for row in output_rows]
    # The complete unchanged nonconvex source context is retained by identity.
    source=dict(n_cont=base,binary_ids=(0,),centers=centers,powers=powers,
        parent_ids=ids,weights=weight,bias=bias,output_ids=out_ids,
        old_inverse=np.array([[alias_id,0,1,2]],np.int64),
        source_rows=((True,((alias_id,1.),(0,-.5)),(),0.),
                     (True,((0,1.),),((0,-.5),),0.),
                     (False,((1,1.),),((0,.25),),1.5)))
    old=dict(source=source,aux=aux,outputs=output_rows,base=base,n_cont=base+len(aux))
    new=dict(source=source,aux=new_aux,outputs=new_out,base=base,n_cont=base+len(new_aux))
    report=dict(mode=mode,channels=channels,outputs=outputs,grid=list(grid),
        partial_groups=len(counts),maximum_occurrences=max(counts.values(),default=0),
        occurrence_histogram={str(n):sum(v==n for v in counts.values()) for n in sorted(set(counts.values()))},
        selected_partial_groups=shift,selected_occurrences=sum(counts[key] for key in chosen),
        unchanged_source_identity=old['source'] is new['source'],work=pool.used,
        real_source_bound=False,formal_gain=0)
    return old,new,expected,report


def packed(problem):
    """All actual native sparse arrays, normalized boxes and inverse/owner data."""
    nc=problem['n_cont']; source=problem['source']; nb=len(source['binary_ids'])
    rows=[]; upper=[]; lower=[]
    for equality,cc,bc,rhs in source['source_rows']:
        rows.append(tuple(sorted(tuple(cc)+tuple((nc+c,v) for c,v in bc))))
        lower.append(rhs if equality else -np.inf);upper.append(rhs)
    definitions=[*problem['aux'],*problem['outputs']]
    for row in definitions:
        rows.append(row['coefficients']);lower.append(row['rhs']);upper.append(row['rhs'])
    ptr=np.r_[0,np.cumsum([len(row) for row in rows])].astype(np.int32)
    columns=np.array([c for row in rows for c,v in row],np.int32)
    values=np.array([v for row in rows for c,v in row],np.float64)
    if np.any(values==0):raise ValueError('uncoalesced zero entry')
    matrix=sp.csr_matrix((values,columns,ptr),shape=(len(rows),nc+nb))
    if not matrix.has_canonical_format:raise ValueError('actual canonical native matrix required')
    pivots=np.array([row['slot'] for row in problem['aux']],np.int32)
    inverse=np.array([[i+len(source['source_rows']),row['slot']] for i,row in enumerate(problem['aux'])],np.int64).reshape(-1,2)
    result=dict(data=matrix.data,indices=matrix.indices,indptr=matrix.indptr,
        row_lb=np.array(lower),row_ub=np.array(upper),var_lb=-np.ones(nc+nb),
        var_ub=np.ones(nc+nb),integrality=np.r_[np.zeros(nc,np.uint8),np.ones(nb,np.uint8)],
        owner_count=np.bincount(columns,minlength=nc+nb).astype(np.int64),
        source_ids=np.arange(nc+nb,dtype=np.int64),inverse=inverse,
        inverse_pivots=pivots,old_inverse=source['old_inverse'])
    # All fixed numeric source inputs remain real owners of the complete fixture.
    for key in ('centers','powers','parent_ids','weights','bias','output_ids'):
        result['source_'+key]=source[key]
    return result


def prove_and_measure(old,new,expected,report,*,pool):
    """Independent native-polynomial projection and complete measured cost bill."""
    for problem in (old,new):
        if project_outputs(problem['aux'],problem['outputs'],problem['base']) != expected:
            raise ValueError('actual complete native output polynomial differs')
        for binary in (-1,1):
            point=[F((i%3)-1,4) for i in range(problem['base'])]
            point[0]=F(binary,2);point[int(problem['source']['old_inverse'][0,0])]=point[0]/2
            for row,poly in zip(problem['outputs'],expected,strict=True):
                output=next(c for c,v in poly.items() if c in problem['source']['output_ids'])
                point[output]=-(poly.get(-1,F(0))+sum((v*point[c] for c,v in poly.items() if c not in (-1,output)),F(0)))/poly[output]
            extended=extend_actual(problem['aux'],point)
            for row in problem['outputs']:
                if sum((F(v)*extended[c] for c,v in row['coefficients']),F(0)) != F(row['rhs']):
                    raise ValueError('actual inverse/output equation differs')
            for equality,cc,bc,rhs in problem['source']['source_rows']:
                value=sum((F(v)*extended[c] for c,v in cc),F(0))+sum(F(v)*binary for c,v in bc)
                if (equality and value != F(rhs)) or (not equality and value > F(rhs)):
                    raise ValueError('original binary/EQ/INEQ source meaning lost')
    before,after=packed(old),packed(new)
    def size(arrays):
        unique={id(a):a for a in arrays.values()}
        return dict(bytes=sum(a.nbytes for a in unique.values()),entries=sum(a.size for a in unique.values()),
            nnz=arrays['data'].size,rows=arrays['row_lb'].size,variables=arrays['var_lb'].size,
            arrays={k:dict(bytes=a.nbytes,entries=a.size,dtype=str(a.dtype)) for k,a in arrays.items()})
    b,a=size(before),size(after)
    pool.charge('c109_complete_native_array_ledger',4*(b['entries']+a['entries']))
    report.update(before=b,after=a,nnz_delta=a['nnz']-b['nnz'],byte_delta=a['bytes']-b['bytes'],
        entry_delta=a['entries']-b['entries'],new_factors=new['n_cont']-old['n_cont'],
        exact_all_output_polynomials=True,actual_inverse_both_binary_phases=True,
        all_new_boxes_redundant=True,all_source_EQ_INEQ_retained=True,
        strict_complete_numeric_win=a['nnz']<b['nnz'] and a['bytes']<b['bytes'] and a['entries']<b['entries'],
        source_runtime_LIVE_admitted=False,work=pool.used)
    return report,before,after
