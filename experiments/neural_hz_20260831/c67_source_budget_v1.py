"""The complete unchanged C66 source bound plus direct-assembly control."""
from experiments.neural_hz_20260831.c66_source_budget_v1 import bound as previous_bound


def bound(saved,legacy,reference,*,pool,enabled=False):
    if not enabled:return None
    result=previous_bound(saved,legacy,reference,pool=pool,enabled=True)
    result['work_parts']['c67_direct_final_index_assembly']=1024
    for key in ('coupled_extra_upper','whole_work_upper','branch_work_upper'):result[key]+=1024
    for key in ('whole_headroom','branch_headroom'):result[key]-=1024
    result['schema']='c67_complete_direct_CSR_construction_bound_v1'
    result['work_caps_fit']=result['whole_headroom']>=0 and result['branch_headroom']>=0
    return result
