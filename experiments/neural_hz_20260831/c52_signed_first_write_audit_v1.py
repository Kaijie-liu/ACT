"""Independent Fraction induction over every source definition and predicate."""
from fractions import Fraction as F
import numpy as np
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import validate_program,source_hash,state_hash,storage
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def _fraction_row(row,roots):
    continuous={}
    for col,value in zip(row['cc'],row['cv']):
        root,sign=roots[int(col)];continuous[root]=continuous.get(root,F(0))+sign*F(float(value))
    return {k:v for k,v in continuous.items() if v},dict(zip(map(int,row['bc']),map(lambda v:F(float(v)),row['bv']))),F(row['rhs'])


def _actual(matrix,row,ids=None):
    a,b=map(int,matrix.indptr[row:row+2]);columns=matrix.indices[a:b]
    if ids is not None:columns=ids[columns]
    return {int(c):F(float(v)) for c,v in zip(columns,matrix.data[a:b])}


def audit(program,state,*,pool=None):
    pool=WorkPool(256_000_000) if pool is None else pool
    protected,nnz=validate_program(program,pool)
    if source_hash(program)!=state['source_sha256'] or state_hash(state)!=state['seal']:raise ValueError('source or generated state changed')
    nc=program['nc'];roots=[(i,1) for i in range(nc)];removed=[];retained={'eq':[],'ineq':[]};derived=0
    pool.charge('c52_independent_fraction_source_induction',128*(nnz+nc+len(program['commands']))+512)
    for row in program['commands']:
        coefficients,binary,rhs=_fraction_row(row,roots);col=row['column'];erase=False
        if state['report']['signed_first_write'] and row['kind']=='def' and not protected[col] and len(coefficients)==2 and not binary and not rhs:
            pivot=coefficients[col];parent=next(c for c in coefficients if c!=col);ratio=-coefficients[parent]/pivot
            if ratio in (-1,1):
                if not parent<col or roots[parent]!=(parent,1):raise ValueError('induction root is not fixed/topological')
                roots[col]=(parent,int(ratio));removed.append((row['uid'],col));erase=True
                derived+=int(len(row['cc'])>2)
        if not erase:retained['ineq' if row['kind']=='ineq' else 'eq'].append((row['uid'],coefficients,binary,rhs))
    kept=[i for i,(root,sign) in enumerate(roots) if (root,sign)==(i,1)]
    position={col:i for i,col in enumerate(kept)}
    inverse=[sign*(position[root]+1) for root,sign in roots] if removed else []
    if (not np.array_equal(state['global_ids'],kept) or not np.array_equal(state['inverse'],inverse)
            or not np.array_equal(state['removed'],np.asarray(removed,np.int64).reshape(-1,2))):raise ValueError('complete source-induced global frame or removed UID relation differs')
    hz=state['hz']
    if hz.n_cont!=len(kept) or hz.n_bin!=program['nb'] or hz.frame_id!=program['frame_id'] or not hz.exact:
        raise ValueError('source continuous/binary/shared frame changed')
    for kind,cmat,bmat,rhs,uids in (('eq',hz.Ac,hz.Ab,hz.b,state['eq_uids']),('ineq',hz.Auc,hz.Aub,hz.ub,state['ineq_uids'])):
        expected=retained[kind]
        if len(rhs)!=len(expected) or not np.array_equal(uids,[r[0] for r in expected]):raise ValueError('retained predicate UID population/order differs')
        if not cmat.has_canonical_format or not bmat.has_canonical_format:raise ValueError('published predicate is not canonical')
        for i,(_,continuous,binary,value) in enumerate(expected):
            if _actual(cmat,i,state['global_ids'])!=continuous or _actual(bmat,i)!=binary or F(float(rhs[i]))!=value:
                raise ValueError('a complete original predicate image differs')
    for i,row in enumerate(program['outputs']):
        continuous,binary,_=_fraction_row(row,roots)
        if _actual(hz.Gc,i,state['global_ids'])!=continuous or _actual(hz.Gb,i)!=binary:
            raise ValueError('complete original output map differs')
    if not np.array_equal(hz.c,program['bias']):raise ValueError('original output bias changed')
    if sorted(list(state['eq_uids'])+list(state['ineq_uids'])+list(state['removed'][:,0]))!=sorted(r['uid'] for r in program['commands']):
        raise ValueError('an original predicate UID or eliminated definition was lost')
    original_nnz=sum(len(r['cv'])+len(r['bv']) for r in program['commands'])
    new_nnz=hz.Ac.nnz+hz.Ab.nnz+hz.Auc.nnz+hz.Aub.nnz
    if (state['report']['eliminated']!=len(removed) or state['report']['original_predicate_nnz']!=original_nnz
            or state['report']['new_predicate_nnz']!=new_nnz or state['report']['emitted_predicate_coefficients']!=new_nnz):
        raise ValueError('first-write population/nnz counters differ from complete source proof')
    return dict(schema='c52_complete_fraction_source_induction_v1',source_sha256=state['source_sha256'],state_sha256=state['seal'],
        complete_source_rows=len(program['commands']),complete_output_rows=len(program['outputs']),
        all_binary_factors_and_predicate_UIDs_retained=True,all_original_continuous_coordinates_reconstructible=True,
        two_way_box_preserving_relation_proved=True,all_predicates_and_outputs_exact=True,
        eliminated_definitions=len(removed),new_unit_relations_exposed_by_shared_coalescing=derived,
        proof_work=pool.used,original_network_binding_or_native_admission_proved=False,formal_gain=0)


def comparison(program,before,after):
    bm,am=storage(program,before),storage(program,after)
    bn=sum(getattr(before['hz'],n).nnz for n in ('Ac','Ab','Auc','Aub'))
    an=sum(getattr(after['hz'],n).nnz for n in ('Ac','Ab','Auc','Aub'))
    return dict(before=bm,after=am,predicate_nnz_before=bn,predicate_nnz_after=an,
        strict_predicate_nnz_decrease=an<bn,strict_numeric_bytes_decrease=am['numeric_bytes']<bm['numeric_bytes'],
        strict_numeric_entries_decrease=am['numeric_entries']<bm['numeric_entries'],
        strict_combined_reported_accounting_decrease=am['numeric_plus_reported_shallow_bytes']<bm['numeric_plus_reported_shallow_bytes'],
        full_original_request_LIVE_or_runtime_proved=False,formal_gain=0)
