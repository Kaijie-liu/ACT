# SPDX-License-Identifier: AGPL-3.0-or-later
"""All unchanged failed-geometry tiles, from complete fresh original source."""
import numpy as np
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift as original_lift
from experiments.neural_hz_20260831.c94_raw_mask_plan_v1 import raw_plan,selected_source_check
from experiments.neural_hz_20260831.c107_circuit_stream_v1 import tile_maps
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c96_word_row_v1 import construct as original_construct
from experiments.neural_hz_20260831.c113_prepared_tile_v1 import construct
from experiments.neural_hz_20260831.c90_actual_circuit_proof_v1 import prove
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import row
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding


def diagnose(*,pool):
    expr=expression(c=16,k=32,h=6);src=expr.terms[0].source
    removed=(np.indices((16,6,6)).sum(axis=0).reshape(-1)%4)==0
    src.Gc.data[removed]=0;src.Gc.eliminate_zeros()
    src.Gb.data[removed]=0;src.Gb.eliminate_zeros();src.c[removed]=0
    identity=expression_binding(expr);keep=np.ones(expr.n_out,bool)
    old=original_lift(expr,keep,enabled=True,max_work=32_000_000,max_branch_work=32_000_000)
    fields=old['fields'];nodes=old['construction']['nodes'];hz=fields['hz']
    records,plan,masks=raw_plan(nodes,existing_aux=len(fields['def_rows']),
        existing_work=fields['report']['actual_radix_work'],pool=pool,enabled=True)
    if plan['selected_positions']:raise ValueError('unchanged failed source plan not reproduced')
    packets=[];maps=[];reports=[];kernels={}
    for index,item in enumerate(records):
        node=nodes[item['node']];parent=nodes[node['parents'][0]]
        ids,powers,outs,opowers=tile_maps(node,parent,item['y'],item['x'],pool=pool)
        maps.append(dict(ids=ids,powers=powers,outs=outs,output_powers=opowers))
        survival=selected_source_check(fields,maps[-1],item['cost']['bill']['direct_nnz'],pool=pool,enabled=True)
        if item['node'] not in kernels:
            kernel=node['op']._kernel
            pool.charge('c113_diagnostic_original_kernel_precision',4*int(kernel.size))
            w=kernel.astype(np.float32)
            if not np.array_equal(w.astype(np.float64),kernel):raise ValueError('original full kernel not binary32')
            checked,kernels[item['node']]=prepare_words(w,pool=pool,enabled=True)
            if not checked['original_dense'] or not kernels[item['node']].dense:raise ValueError('complete density differs')
        originals=[];gauges=[]
        for pivot in outs.reshape(-1):
            if pivot<0:continue
            rank=fields['old_n_eq']+int(pivot)-fields['old_n_cont'];physical=int(fields['eq_roots'][rank])
            cols,vals=row(hz.Ac,physical)
            originals.append(dict(coefficients=list(zip(cols,vals)),rhs=hz.b[physical],pivot=int(pivot)))
            gauges.append(int(fields['eq_scales'][rank]))
        case=dict(index=index,node=item['node'],y=item['y'],x=item['x'],raw_cost=item['cost'],
            source_survival=survival,outputs=len(originals),implementations={})
        for name,fn in (('C96',original_construct),('C113',construct)):
            start=pool.used;report,packet=fn(kernels[item['node']],ids,powers,outs,opowers,hz.n_cont,pool=pool,enabled=True)
            constructor=pool.used-start
            packet['ab_indptr']=np.zeros(len(packet['rhs'])+1,np.int32)
            proof=prove(packet,originals,gauges,old_n_cont=hz.n_cont,first_aux=hz.n_cont,
                new_factors=report['new_factors'],pool=pool,enabled=True)
            packets.append(packet)
            n=report['nnz'];a=report['new_factors'];o=len(originals);d=item['cost']['bill']['direct_nnz']
            case['implementations'][name]=dict(report=report,constructor_work=constructor,proof=proof,
                actual_packet_byte_formula=12*n+88*a+32*o+72,
                original_byte_formula=12*d+16*o+8,
                actual_packet_entry_delta_formula=2*(n-d)+13*a+3*o+8,
                whole_source_and_owner_not_constructed=True)
        reports.append(case)
    held=dict(original=expr,source=old,keep=keep,raw_records=records,raw_plan=plan,
        all_masks=masks,all_original_maps=maps,
        all_transforms={key:dict(transformed=value.transformed,dense=value.dense) for key,value in kernels.items()},
        all_packets=packets,all_reports=reports)
    layout=numeric_layout(held,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('all original source and packet roots exceed64M')
    if expression_binding(expr)!=identity:raise ValueError('original source changed during diagnosis')
    return dict(complete=True,tiles=len(reports),raw_selected=plan['selected_positions'],reports=reports,
        complete_numeric_bytes=layout.resident_bytes,complete_numeric_entries=layout.resident_entries,
        original_generation_work=fields['report']['total_work_upper'],original_expression_unchanged=True,
        source_runtime_LIVE_admitted=False,actual_source_eliminations=0,formal_gain=0),held
