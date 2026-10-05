"""Unchanged complete C65 bound plus new fixed packed-frontier bindings."""
from experiments.neural_hz_20260831.c65_source_budget_v1 import bound as previous_bound


def bound(saved,legacy,reference,*,pool,enabled=False):
    if not enabled:return None
    result=previous_bound(saved,legacy,reference,pool=pool,enabled=True)
    result['work_parts']['c66_packed_frontier_bindings']=1024
    for key in ('coupled_extra_upper','whole_work_upper','branch_work_upper'):result[key]+=1024
    for key in ('whole_headroom','branch_headroom'):result[key]-=1024
    result['schema']='c66_complete_packed_frontier_construction_bound_v1'
    result['all_inherited_operation_prices_unchanged']=True
    result['work_caps_fit']=result['whole_headroom']>=0 and result['branch_headroom']>=0
    return result
