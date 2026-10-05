# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete C96/C111/C112 comparison, including all exact source semantics."""
from fractions import Fraction as F
import numpy as np
from experiments.neural_hz_20260831.c109_shared_partial_v1 import fixture,prove_and_measure
from experiments.neural_hz_20260831.c88_inline_tile_v1 import actual_rows
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c96_word_row_v1 import construct as before_construct
from experiments.neural_hz_20260831.c111_source_first_row_v1 import construct as middle_construct
from experiments.neural_hz_20260831.c112_source_first_row_v1 import construct as after_construct


def complete(channels,outputs,mode,grid,*,pool):
    original,unused,expected,_=fixture(channels,outputs,mode,grid=grid,pool=pool,enabled=True)
    del unused
    source=original['source'];base=original['base']
    _,transformed=prepare_words(source['weights'],pool=pool,enabled=True)
    by_output={next(c for c in poly if c in source['output_ids']):poly for poly in expected}
    problems=[];costs=[];blocks=[];price_parts=[]
    for construct in (before_construct,middle_construct,after_construct):
        aux=[];out=[];reports=[];start=pool.used;parts=dict(pool.parts)
        _,oh,ow=source['output_ids'].shape
        for y in range(0,oh,2):
            for x in range(0,ow,2):
                ids=source['parent_ids'][:,y:y+4,x:x+4]
                powers=source['powers'][:,y:y+4,x:x+4]
                outputs_here=source['output_ids'][:,y:y+2,x:x+2]
                report,packet=construct(transformed,ids,powers,outputs_here,
                    np.full(outputs_here.shape,20,np.int32),base+len(aux),pool=pool,enabled=True)
                a,o=actual_rows(report,packet)
                for row in o:
                    constant=-by_output[row['slot']].get(-1,F(0))
                    exact=constant*F(2)**row['gauge'];row['rhs']=float(exact)
                    if F(row['rhs'])!=exact:raise ValueError('original complete centered RHS not exact')
                aux.extend(a);out.extend(o);reports.append(report)
        problems.append(dict(source=source,aux=aux,outputs=out,base=base,n_cont=base+len(aux)))
        costs.append(pool.used-start);blocks.append(reports)
        price_parts.append({k:v-parts.get(k,0) for k,v in pool.parts.items() if v!=parts.get(k,0)})
    old,middle,new=problems
    if middle['aux']!=new['aux'] or middle['outputs']!=new['outputs']:
        raise ValueError('factored keys changed the complete C111 native quotient')
    if old['aux']!=original['aux'] or old['outputs']!=original['outputs']:
        raise ValueError('fresh original C96 fixture does not reproduce frozen comparator')
    omitted=len(old['aux'])-len(new['aux'])
    if omitted!=sum(r['reused_v'] for r in blocks[2]):
        raise ValueError('full physical factor omission differs from producer routes')
    report=dict(mode=mode,channels=channels,outputs=outputs,grid=list(grid),
        source_first=True,omitted_factors=omitted,
        constructor_before_work=costs[0],constructor_intermediate_work=costs[1],constructor_after_work=costs[2],
        constructor_work_delta=costs[2]-costs[0],constructor_vs_C111_delta=costs[2]-costs[1],
        all_constructor_price_parts=price_parts,all_tile_reports=blocks,complete_C111_quotient_unchanged=True,
        original_source_coordinates_all_retained=True,old_auxiliary_names_are_comparator_only=True,
        original_source_inverse_retained_by_identity=old['source']['old_inverse'] is new['source']['old_inverse'],
        original_native_comparator_reproduced=True,formal_gain=0)
    norm_old=price_parts[1].get('c111_complete_existing_form_keys',0)
    norm_new=sum(price_parts[2].get(k,0) for k in ('c112_complete_channel_frame_binding',
        'c112_all_form_route_binding','c112_exact_mask_templates','c111_complete_existing_form_keys'))
    report.update(C111_normalization_work=norm_old,C112_normalization_work=norm_new,
        normalization_reduction_at_least_20_percent=5*norm_new<=4*norm_old,
        whole_constructor_work_reduced=costs[2]<costs[1])
    if not norm_old or 5*norm_new>4*norm_old or costs[2]>=costs[1]:
        raise ValueError('registered common-case complete work improvement failed')
    report,before,after=prove_and_measure(old,new,expected,report,pool=pool)
    if omitted and not report['strict_complete_numeric_win']:
        raise ValueError('complete source-first numeric payload does not strictly shrink')
    if not omitted and (old['aux']!=new['aux'] or old['outputs']!=new['outputs']):
        raise ValueError('no-hit source-first constructor changed the literal comparator')
    return report,before,after,old,new,expected
