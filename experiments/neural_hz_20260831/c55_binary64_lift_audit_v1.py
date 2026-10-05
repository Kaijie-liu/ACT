"""Independent Fraction elimination of ACTUAL C55 rows, including redundant boxes."""
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
from experiments.neural_hz_20260831.c55_binary64_lift_v1 import layout,state_hash,SHARED,exact_binding
from experiments.neural_hz_20260831.c54_packed_scalar_hz_v2 import audit as exact_audit
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import unpack
from experiments.neural_hz_20260831.c52_neural_source_fixtures_v1 import source_points,feasible as old_feasible,value as old_value
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import storage as reference_storage
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


def rational(value):
    m,e=value;return F(m<<e) if e>=0 else F(m,1<<-e)


def rows(state):
    m=state['csr'];ptr=m['indptr']
    for i in range(len(ptr)-1):
        a,b=map(int,ptr[i:i+2])
        yield {int(c):F(float(v)) for c,v in zip(m['indices'][a:b],m['data'][a:b])}


def derived(state):
    table=layout(state);base=state['base_nc'];nc=state['n_cont'];old_ne=state['base_ne'];native=list(rows(state))
    expressions=[{i:F(1)} for i in range(base)]
    for i in range(nc-base):
        col=base+i;row=dict(native[old_ne+i])
        if row.pop(col,None)!=1 or state['rhs'][old_ne+i]!=0 or any(j>=col for j in row):raise ValueError('auxiliary defining row/order differs')
        expression={}
        for parent,value in row.items():
            for root,ratio in expressions[parent].items():expression[root]=expression.get(root,F(0))-value*ratio
        expression={k:v for k,v in expression.items() if v}
        if sum(map(abs,expression.values()),F(0))>1:raise ValueError('new continuous box is not redundant')
        expressions.append(expression)
    for col,idx,start,count in state['lifts']:
        if expressions[int(start+count-1)]!={int(col):abs(rational(table[int(idx)]))/F(2)**(abs(table[int(idx)][0]).bit_length()+table[int(idx)][1])}:
            raise ValueError('actual emitted digit recurrence differs from exact scalar')
    return table,native,expressions


def audit(program,exact,state,*,pool=None):
    source_proof=exact_audit(program,exact,pool=pool)
    table,native,expressions=derived(state);base=state['base_nc'];nc=state['n_cont'];ne=state['n_eq'];old_ne=state['base_ne'];extra=nc-base
    if state['exact_semantic_sha256']!=exact_binding(exact) or base!=exact['n_cont'] or old_ne!=exact['n_eq']:raise ValueError('exact source/frame boundary differs')
    if any(state[k]!=exact[k] for k in ('n_bin','n_out','n_ineq','frame_id','original_nc','source_sha256')):raise ValueError('source/discrete dimensions changed')
    for k in ('global_ids','inverse','removed','ineq_uids'):
        if not np.array_equal(state[k],exact[k]):raise ValueError('original inverse or UID payload differs')
    if unpack(state['scalars'])!=unpack(exact['scalars']):raise ValueError('original exact inverse scalar pool differs')
    if not np.array_equal(state['eq_uids'][:old_ne],exact['eq_uids']):raise ValueError('original equality UIDs differ')
    original_ids=set(map(int,exact['eq_uids']))|set(map(int,exact['ineq_uids']))|set(map(int,exact['removed'][:,0]))
    new_ids=list(map(int,state['eq_uids'][old_ne:]))
    if len(set(new_ids))!=extra or original_ids.intersection(new_ids):raise ValueError('new equality UID collision')
    source=exact['csr'];ptr=source['indptr'];checked=0
    for i in range(source['shape'][0]):
        j=i if i<old_ne else i+extra;actual={}
        for col,value in native[j].items():
            mapping=expressions[col] if col<nc else {base+col-nc:F(1)}
            for root,ratio in mapping.items():actual[root]=actual.get(root,F(0))+value*ratio
        actual={k:v for k,v in actual.items() if v};a,b=map(int,ptr[i:i+2])
        wanted={int(col):rational(table[int(idx)]) for col,idx in zip(source['indices'][a:b],source['coefficients'][a:b])}
        if actual!=wanted or F(float(state['rhs'][j]))!=rational(table[int(exact['rhs'][i])]):raise ValueError('original exact predicate/output/RHS changed')
        checked+=1
    if pool is not None:pool.charge('c55_independent_actual_row_Fraction_proof',256*(sum(map(len,native))+checked+extra)+4096)
    return dict(all_original_rows_checked=checked,all_auxiliary_equalities_checked=extra,
        every_new_box_proved_redundant=True,all_original_inverse_UIDs_binaries_preserved=True,
        exact_source_proof=source_proof['schema'],native_solver_admission_proved=False,formal_gain=0)


def check_points(program,reference,state):
    table,native,expressions=derived(state);nc=state['n_cont'];base=state['base_nc'];ne=state['n_eq'];nu=state['n_ineq'];count=0
    for inputs,original,binary in source_points(program):
        compact=[original[int(i)] for i in state['global_ids']]
        lifted=[sum((ratio*compact[root] for root,ratio in expr.items()),F(0)) for expr in expressions]
        if any(abs(x)>1 for x in lifted):raise ValueError('extended witness violates a continuous box')
        inverse=state['inverse'];recovered=[]
        for word in inverse:
            word=int(word);recovered.append(compact[word&((1<<32)-1)]*rational(table[word>>32]))
        if len(inverse) and recovered!=original:raise ValueError('complete original witness inverse differs')
        if not old_feasible(reference['hz'],original,binary):raise ValueError('original point is not feasible')
        values=[sum((v*(lifted[c] if c<nc else binary[c-nc]) for c,v in row.items()),F(0)) for row in native]
        if any(values[i]!=F(float(state['rhs'][i])) for i in range(ne)):raise ValueError('native-coefficient equality violated')
        if any(values[i]>F(float(state['rhs'][i])) for i in range(ne,ne+nu)):raise ValueError('native-coefficient inequality violated')
        output=[values[i]+F(float(state['rhs'][i])) for i in range(ne+nu,len(values))]
        if output!=old_value(reference['hz'],original,binary):raise ValueError('exact original network output differs')
        count+=1
    return dict(feasible_original_vectors=count,complete_lift_inverse_and_outputs_exact=True,benchmark_counterexamples=0)


def storage(program,state):
    layout(state);roots=collect(SimpleNamespace(),dict(complete_common_source=program,complete_state=state));m=roots.measure()
    return dict(numeric_bytes=m.resident_bytes,numeric_entries=m.resident_entries,python_shallow_bytes=roots.python_shallow_bytes,
        numeric_roots=len(roots.numeric),numeric_plus_reported_shallow_bytes=m.resident_bytes+roots.python_shallow_bytes)


def comparison(program,reference,state):
    before=reference_storage(program,reference);after=storage(program,state)
    nnz=int(state['csr']['indptr'][state['n_eq']+state['n_ineq']]);old_nnz=sum(len(r['cc'])+len(r['bc']) for r in program['commands'])
    return dict(before=before,after=after,predicate_nnz_before=old_nnz,predicate_nnz_after=nnz,
        strict_predicate_nnz_decrease=nnz<old_nnz,
        strict_numeric_bytes_decrease=after['numeric_bytes']<before['numeric_bytes'],
        strict_numeric_entries_decrease=after['numeric_entries']<before['numeric_entries'],
        strict_combined_reported_accounting_decrease=after['numeric_plus_reported_shallow_bytes']<before['numeric_plus_reported_shallow_bytes'],
        comparator='same_source_unchanged_C52_binary64_reference',real_native_or_C31_whole_request_gain_proved=False)
