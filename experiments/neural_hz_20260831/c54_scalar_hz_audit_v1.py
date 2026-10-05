"""Independent Fraction source induction over exact scalar HZ and full inverse."""
from fractions import Fraction as F
import numpy as np
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import validate_program,source_hash,storage as reference_storage
from experiments.neural_hz_20260831.c54_scalar_hz_v1 import state_hash,storage
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import unpack
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def scalars(state):
    return [F(m)*F(2)**e for m,e in unpack(state['scalars'])]


def _row(matrix,index,values,global_ids=None):
    a,b=map(int,matrix['indptr'][index:index+2]);result={}
    for col,coefficient in zip(matrix['indices'][a:b],matrix['coefficients'][a:b]):
        result[int(col) if global_ids is None else int(global_ids[int(col)])]=values[int(coefficient)]
    return result


def layout(state):
    if state_hash(state)!=state['seal']:raise ValueError('exact HZ seal changed')
    values=scalars(state);used=set();count=len(values)
    def ids(a):
        if type(a) is not np.ndarray or a.dtype!=np.dtype(np.int32) or a.ndim!=1 or np.any(a<0) or np.any(a>=count):raise ValueError('complete scalar index vector required')
        used.update(map(int,a))
    if state['schema']!='c54_exact_scalar_nonconvex_HZ_v1':raise ValueError('unregistered exact HZ schema')
    for ckey,bkey,rhs,width in [('Ac','Ab','b',len(state['eq_uids'])),('Auc','Aub','ub',len(state['ineq_uids'])),('Gc','Gb','c',state['n_out'])]:
        for key,ncol in [(ckey,state['n_cont']),(bkey,state['n_bin'])]:
            matrix=state[key]
            if type(matrix) is not dict or set(matrix)!={'shape','indptr','indices','coefficients'} or matrix['shape']!=(width,ncol):raise ValueError('complete exact CSR shape required')
            for name in ('indptr','indices','coefficients'):
                a=matrix[name]
                if type(a) is not np.ndarray or a.dtype!=np.dtype(np.int32) or a.ndim!=1:raise ValueError('canonical exact CSR indices required')
            ptr=matrix['indptr'];cols=matrix['indices']
            if (len(ptr)!=width+1 or ptr[0]!=0 or ptr[-1]!=len(cols) or len(matrix['coefficients'])!=len(cols)
                    or np.any(np.diff(ptr)<0) or np.any(cols<0) or np.any(cols>=ncol)):raise ValueError('exact CSR spans differ')
            ids(matrix['coefficients'])
            for i in range(width):
                a,b=map(int,ptr[i:i+2])
                if np.any(np.diff(cols[a:b])<=0) or any(not values[int(c)] for c in matrix['coefficients'][a:b]):raise ValueError('exact row is not canonical')
        ids(state[rhs])
        if len(state[rhs])!=width:raise ValueError('complete exact RHS/bias required')
    ids(state['inverse_values'])
    if used!=set(range(count)):raise ValueError('unreachable scalar pool data retained')
    return values


def audit(program,state,*,pool=None):
    pool=WorkPool(256_000_000) if pool is None else pool
    protected,nnz=validate_program(program,pool);values=layout(state)
    if state['source_sha256']!=source_hash(program):raise ValueError('wrong original source program')
    if state['frame_id']!=program['frame_id'] or state['n_bin']!=program['nb'] or state['n_out']!=len(program['outputs']) or state['original_nc']!=program['nc']:
        raise ValueError('original nonconvex frame differs')
    pool.charge('c54_independent_complete_Fraction_source_induction',256*(nnz+program['nc']+len(program['commands']))+1024)
    roots=[(i,F(1)) for i in range(program['nc'])];removed=[];kept={'eq':[],'ineq':[]};derived=0
    def translated(row):
        result={}
        for c,v in zip(row['cc'],row['cv']):
            root,weight=roots[int(c)];result[root]=result.get(root,F(0))+F(float(v))*weight
        return {c:v for c,v in result.items() if v},{int(c):F(float(v)) for c,v in zip(row['bc'],row['bv'])},F(row['rhs'])
    for row in program['commands']:
        continuous,binary,rhs=translated(row);column=row['column'];erase=False
        if row['kind']=='def' and not protected[column] and len(continuous)==2 and not binary and not rhs:
            pivot=continuous[column];parent=next(c for c in continuous if c!=column);ratio=-continuous[parent]/pivot
            power_pivot=(pivot>0 and pivot.numerator & (pivot.numerator-1)==0 and pivot.denominator & (pivot.denominator-1)==0)
            if power_pivot and 0<abs(ratio)<=1:
                if not parent<column or roots[parent]!=(parent,F(1)):raise ValueError('source root not permanently retained')
                roots[column]=(parent,ratio);removed.append((row['uid'],column));erase=True;derived+=int(len(row['cc'])>2)
        if not erase:kept['ineq' if row['kind']=='ineq' else 'eq'].append((row['uid'],continuous,binary,rhs))
    global_ids=[i for i,r in enumerate(roots) if r==(i,F(1))];positions={c:i for i,c in enumerate(global_ids)}
    expected_roots=[positions[c] for c,w in roots] if removed else []
    expected_values=[w for c,w in roots] if removed else []
    if (state['n_cont']!=len(global_ids) or not np.array_equal(state['global_ids'],global_ids)
            or not np.array_equal(state['inverse_roots'],expected_roots)
            or [values[int(i)] for i in state['inverse_values']]!=expected_values
            or not np.array_equal(state['removed'],np.asarray(removed,np.int64).reshape(-1,2))):raise ValueError('complete source-induced inverse or global frame differs')
    for kind,cm,bm,rhs,uids in [('eq','Ac','Ab','b','eq_uids'),('ineq','Auc','Aub','ub','ineq_uids')]:
        expected=kept[kind]
        if not np.array_equal(state[uids],[r[0] for r in expected]):raise ValueError('original predicate UID population differs')
        for i,(_,continuous,binary,r) in enumerate(expected):
            if _row(state[cm],i,values,global_ids)!=continuous or _row(state[bm],i,values)!=binary or values[int(state[rhs][i])]!=r:
                raise ValueError('a complete original predicate image differs')
    for i,row in enumerate(program['outputs']):
        continuous,binary,_=translated(row)
        if (_row(state['Gc'],i,values,global_ids)!=continuous or _row(state['Gb'],i,values)!=binary
                or values[int(state['c'][i])]!=F(float(program['bias'][i]))):raise ValueError('original network output image differs')
    original=sum(len(r['cv'])+len(r['bv']) for r in program['commands'])
    actual=sum(len(state[k]['indices']) for k in ('Ac','Ab','Auc','Aub'))
    if (state['report']['eliminated']!=len(removed) or state['report']['coalescing_exposed_relations']!=derived
            or state['report']['original_predicate_nnz']!=original or state['report']['new_predicate_nnz']!=actual):raise ValueError('claimed elimination/nnz count differs')
    if sorted([*state['eq_uids'],*state['ineq_uids'],*state['removed'][:,0]])!=sorted(r['uid'] for r in program['commands']):raise ValueError('original UID missing or duplicated')
    nonf64=sum(F(float(v))!=v for v in values)
    matrix_ids=set(int(i) for k in ('Ac','Ab','Auc','Aub','Gc','Gb') for i in state[k]['coefficients'])
    return dict(schema='c54_complete_Fraction_source_audit_v1',all_source_predicates_outputs_and_inverse_exact=True,
        all_binary_factors_and_original_UIDs_preserved=True,two_way_box_preserving_relation_proved=True,
        source_sha256=state['source_sha256'],state_sha256=state['seal'],eliminated=len(removed),coalescing_exposed_relations=derived,
        exact_scalar_count=len(values),non_binary64_scalar_count=nonf64,non_binary64_matrix_scalar_count=sum(F(float(values[i]))!=values[i] for i in matrix_ids),
        maximum_mantissa_bits=max((abs(m).bit_length() for m,e in unpack(state['scalars'])),default=0),
        diagnostic_proof_work=pool.used,native_lowering_or_original_network_binding_proved=False,formal_gain=0)


def reconstruct(state,compact):
    values=layout(state);compact=[F(v) for v in compact]
    if len(compact)!=state['n_cont'] or any(abs(v)>1 for v in compact):raise ValueError('complete compact latent box required')
    if not len(state['inverse_roots']):return compact
    return [compact[int(i)]*values[int(v)] for i,v in zip(state['inverse_roots'],state['inverse_values'])]


def feasible(state,continuous,binary):
    values=layout(state)
    if len(continuous)!=state['n_cont'] or len(binary)!=state['n_bin'] or any(abs(v)>1 for v in continuous) or any(v not in (-1,1) for v in binary):return False
    for cm,bm,rhs,equality in [('Ac','Ab','b',True),('Auc','Aub','ub',False)]:
        for row,i in enumerate(state[rhs]):
            total=sum(v*continuous[c] for c,v in _row(state[cm],row,values).items())+sum(v*binary[c] for c,v in _row(state[bm],row,values).items())
            if (total!=values[int(i)]) if equality else (total>values[int(i)]):return False
    return True


def value(state,continuous,binary):
    values=layout(state)
    return [values[int(state['c'][row])]+sum(v*continuous[c] for c,v in _row(state['Gc'],row,values).items())+sum(v*binary[c] for c,v in _row(state['Gb'],row,values).items()) for row in range(state['n_out'])]


def comparison(program,reference,state):
    before=reference_storage(program,reference);after=storage(program,state)
    return dict(before=before,after=after,
        strict_predicate_nnz_decrease=state['report']['new_predicate_nnz']<state['report']['original_predicate_nnz'],
        strict_numeric_bytes_decrease=after['numeric_bytes']<before['numeric_bytes'],
        strict_numeric_entries_decrease=after['numeric_entries']<before['numeric_entries'],
        strict_combined_reported_accounting_decrease=after['numeric_plus_reported_shallow_bytes']<before['numeric_plus_reported_shallow_bytes'],
        comparator='same_source_unchanged_C52_binary64_reference_not_an_enlarged_exact_format_reference',
        native_or_C31_whole_request_reduction_proved=False,formal_gain=0)
