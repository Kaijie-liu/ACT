# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagnostic-only exact point encoding; v1 failed gate remains closed."""
from fractions import Fraction as F
import numpy as np
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift as original_lift
from experiments.neural_hz_20260831.c113_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c98_source_audit_v1 import audit
from experiments.neural_hz_20260831.c90_actual_circuit_proof_v1 import prove
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import row,extend,recover
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct,SCHEMA
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding


def complete_source(mode,*,pool):
    """Two complete source constructions; caller reserves both 32M budgets."""
    if mode not in ('dense','masked'):raise ValueError('ordinary registered geometry required')
    expr=expression(c=16,k=32,h=6)
    if mode=='masked':
        source=expr.terms[0].source
        removed=(np.indices((16,6,6)).sum(axis=0).reshape(-1)%4)==0
        source.Gc.data[removed]=0;source.Gc.eliminate_zeros()
        source.Gb.data[removed]=0;source.Gb.eliminate_zeros()
        source.c[removed]=0
    keep=np.ones(expr.n_out,bool);identity=expression_binding(expr)
    old=original_lift(expr,keep,enabled=True,max_work=32_000_000,max_branch_work=32_000_000)['fields']
    draft=lift(expr,keep,enabled=True,max_work=32_000_000,max_branch_work=32_000_000)
    state=draft['state'];fields=state['fields'];generation=fields['report']['circuit_generation']
    proof=audit(old,state,draft['construction']['circuits'],pool=pool,enabled=True)
    block_proofs=[]
    for packet in draft['construction']['circuits']:
        originals=[];gauges=[]
        for pivot in packet['pivots'][packet['new_factors']:]:
            rank=old['old_n_eq']+int(pivot)-old['old_n_cont'];physical=int(old['eq_roots'][rank])
            columns,values=row(old['hz'].Ac,physical)
            originals.append(dict(coefficients=list(zip(columns,values)),rhs=old['hz'].b[physical],pivot=int(pivot)))
            gauges.append(int(old['eq_scales'][rank]))
        block_proofs.append(prove(packet,originals,gauges,old_n_cont=old['hz'].n_cont,
            first_aux=int(packet['pivots'][0]),new_factors=packet['new_factors'],pool=pool,enabled=True))
    expected=actual_words(fields['hz'],fields['old_n_cont'],fields['logical_n_cont'],
        draft['construction']['eq_uids'],draft['construction']['ineq_uids'])
    if not np.array_equal(fields['owners'],expected):raise ValueError('complete MAIN ownership differs')
    point=[F((i%5)-2,8) for i in range(old['hz'].n_cont)]
    expected_point=reconstruct(point,old['eq_roots'],old['eq_scales'],
        old_n_cont=old['old_n_cont'],old_n_eq=old['old_n_eq'],n_cont=old['hz'].n_cont,schema=SCHEMA)
    expanded=extend(state,point,pool=pool)
    if recover(state,expanded,pool=pool)!=expected_point:raise ValueError('actual complete original inverse differs')
    pool.charge('c113_complete_exact_point_encoding',16*(len(point)+len(expanded)+len(expected_point)))
    encoded={name:[(value.numerator,value.denominator) for value in values]
        for name,values in (('point',point),('expanded',expanded),('expected_point',expected_point))}
    held=dict(original=expr,old=old,new=draft,expected_owners=expected,
        exact_points=encoded,proof=proof,block_proofs=block_proofs)
    layout=numeric_layout(held,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete comparison exceeds64M')
    if (expression_binding(expr)!=identity or fields['hz'].n_bin!=1
        or generation['once_prepared_tiles']!=generation['consumed_tiles']
        or generation['once_prepared_tiles']<2
        or not generation['all_factor_counts_known_before_owner_allocation']):
        raise ValueError('source identity, binary or complete multi-tile plan differs')
    return dict(mode=mode,complete_original_source_proof=proof,block_proofs=block_proofs,
        all_MAIN_owners_independently_equal=True,actual_complete_inverse_equal=True,
        original_expression_unchanged=True,global_generation=generation,
        old_nnz=int(old['hz'].Ac.nnz),new_nnz=int(fields['hz'].Ac.nnz),
        old_generation_work=old['report']['total_work_upper'],
        new_generation_work=fields['report']['total_work_upper'],
        new_generation_branch_work=fields['report']['largest_branch_work_upper'],
        complete_numeric_bytes=layout.resident_bytes,complete_numeric_entries=layout.resident_entries,
        formal_gain=0,source_runtime_LIVE_admitted=False),held
