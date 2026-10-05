"""Complete paid functional transaction and explicit checkpoint-only closure.

No old receipt is reused on new predicates. The saved closure is a new schema,
not a native-admitted SplicedState. Original C34 caller-only roots are missing
from its archive; neither closure measurements nor this object prove full LIVE.
"""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
from experiments.neural_hz_20260831.c39_half_alias_census_v1 import census
from experiments.neural_hz_20260831.c40_compact_half_gauge_v1 import materialize,DESC
from experiments.neural_hz_20260831.c40_half_gauge_inverse_v1 import inverse_hashes
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


def pack_descriptors(desc,*,pool):
    pool.charge('half_descriptor_lossless_u64_encoding',64*len(desc))
    if type(desc) is not np.ndarray or desc.dtype!=DESC or desc.ndim!=1:raise ValueError('complete half descriptor required')
    out=np.empty(2*len(desc),np.uint64)
    for i,t in enumerate(desc):
        col,parent,d,power,sign=map(int,(t['column'],t['parent'],t['definition'],t['power'],t['sign']))
        if not (0<=parent<col<(1<<31) and 0<=d<(1<<20) and -20<=power<=40 and sign in (-1,1)):
            raise ValueError('half descriptor outside lossless word domain')
        out[2*i]=col|(parent<<32)
        out[2*i+1]=d|((power+20)<<32)|((sign==-1)<<38)|(int(t['negative_zero'])<<39)
    return out


def unpack_descriptors(words,*,pool):
    if type(words) is not np.ndarray or words.dtype!=np.uint64 or words.ndim!=1 or len(words)%2:
        raise ValueError('complete paired uint64 descriptors required')
    pool.charge('half_descriptor_lossless_u64_decoding',64*(len(words)//2))
    out=np.empty(len(words)//2,DESC)
    for i in range(len(out)):
        a,b=int(words[2*i]),int(words[2*i+1]);col=a&((1<<32)-1);parent=a>>32;d=b&((1<<32)-1)
        exponent=((b>>32)&63)-20;sign=-1 if b&(1<<38) else 1
        if b>>40 or not (0<=parent<col<(1<<31) and d<(1<<20) and -20<=exponent<=40):
            raise ValueError('reserved or malformed half descriptor word')
        out[i]=(col,parent,d,exponent,sign,bool(b&(1<<39)))
    return out


def functional(state,final,*,pool,branch=None,observe=None,enabled=False):
    if not enabled:return None
    if type(state) is not SplicedState:raise ValueError('complete actual source SplicedState required')
    proof,half,rows=census(state,final,pool=pool,branch=branch,observe=observe,enabled=True)
    if (not proof['all_joint_row_arithmetic_proved'] or not len(half)
            or proof['observed_parent_overlaps'] or proof['observed_cancellations']):
        raise ValueError('complete independent nonoverlap half-gauge cohort required')
    post,end,desc,runs,writer=materialize(state.hz,final,half,rows,pool=pool,enabled=True)
    pool.charge('half_source_degree_binding_and_final_authentication',256+32*len(half))
    degree_by_col={int(t['column']):int(t['degree']) for t in half}
    degrees=np.array([degree_by_col[int(t['column'])] for t in desc],np.int32)
    inverse=inverse_hashes(post,end,desc,runs,degrees,pool=pool,enabled=True)
    if (inverse['complete_original_post_sha256']!=proof['complete_post_HZ_sha256']
            or inverse['complete_original_final_sha256']!=proof['complete_final_HZ_sha256']):
        raise ValueError('complete inverse differs from independently bound original source/final')
    after=state.validate()
    if (after['complete_new_HZ_sha256']!=proof['complete_post_HZ_sha256']
            or source_digest(final)!=proof['complete_final_HZ_sha256']):
        raise ValueError('functional writer changed original source bytes')
    packed=pack_descriptors(desc,pool=pool);decoded=unpack_descriptors(packed,pool=pool)
    if not np.array_equal(decoded.view(np.uint8),desc.view(np.uint8)):raise ValueError('descriptor codec lost original fields')
    new_post_sha=source_digest(post);new_final_sha=source_digest(end)
    writer['source_inverse_not_yet_checked']=False
    report=dict(complete_functional_HZ_and_inverse_passed=True,original_source_and_final_unchanged=True,
        complete_original_post_sha256=proof['complete_post_HZ_sha256'],complete_original_final_sha256=proof['complete_final_HZ_sha256'],
        new_post_sha256=new_post_sha,new_final_sha256=new_final_sha,
        arithmetic_qualification=proof,actual_writer=writer,complete_inverse=inverse,
        packed_half_descriptor_bytes=packed.nbytes,gauge_interval_bytes=runs.nbytes,
        eliminated_factor_absence_proved_by_complete_original_incidence_and_inverse_counts=True,
        native_admission_certificate=False,whole_C34_LIVE_gate_proved=False,
        complete_old_witness_lineage_adapter_proved=False,runtime_payment_proved=False,
        solver_executed=False,formal_gain=0,diagnostic_work=pool.used,work_parts=dict(pool.parts))
    if observe:observe(dict(event='actual_functional_HZ_and_complete_inverse_passed',
        new_final_sha256=new_final_sha,charged_work=pool.used,formal_gain=0))
    return post,end,packed,runs,report


def checkpoint_payload(saved,old_post,new_post,new_final,packed,runs,*,pool):
    """Preserve EVERY saved key/value except explicitly replaced current HZs.

The new schema refuses to masquerade as C34's native checkpoint. Its old
lineage is retained as original-source metadata, NOT applied to new row ranks.
    """
    pool.charge('half_checkpoint_root_rebinding',512+64*(len(saved)+len(saved['hz_cache'])))
    if saved.get('schema')!='c34_reconstructable_final_native_checkpoint_v1':raise ValueError('complete original C34 checkpoint required')
    old_final=saved['final_hz'];fields=dict(saved['spliced_state_fields']);runtime=dict(saved['runtime_numeric_roots'])
    if fields.pop('hz') is not old_post or runtime.pop('hz') is not old_post:
        raise ValueError('unregistered source/current HZ ownership in checkpoint')
    cache={}
    for key,value in saved['hz_cache'].items():
        if value is old_post:cache[key]=new_post
        elif value is old_final:cache[key]=new_final
        else:
            if getattr(value,'Ac',None) is old_post.Ac:raise ValueError('unregistered additional current-predicate cache owner')
            cache[key]=value
    result=dict(saved)
    del result['spliced_state_fields'];del result['runtime_numeric_roots']
    result.update(schema='c40_functional_half_gauge_checkpoint_v1',original_splice_fields_without_current_HZ=fields,
        retained_runtime_source_metadata_without_current_HZ=runtime,new_post_hz=new_post,
        final_hz=new_final,hz_cache=cache,half_alias_descriptor_u64=packed,half_row_gauge_intervals=runs,
        original_C34_whole_live_boundary_restorable=False,native_admission_certificate=False,formal_gain=0)
    return result


def _measurement_roots(payload):
    # C5's closed schema does not accept ReversibleLineage as an opaque object.
    # Expose ALL of its fields on BOTH sides without copying any numeric map.
    out=dict(payload)
    name='spliced_state_fields' if 'spliced_state_fields' in payload else 'original_splice_fields_without_current_HZ'
    fields=dict(payload[name]);lineage=fields['lineage'];lineage.validate()
    fields['lineage']=dict(vars(lineage));out[name]=fields
    return out


def checkpoint_closure(saved,candidate,*,pool):
    pool.charge('half_checkpoint_closure_schema_metadata',1024+64*(len(saved)+len(candidate)))
    before=collect(SimpleNamespace(),dict(complete_saved_checkpoint=_measurement_roots(saved)))
    after=collect(SimpleNamespace(),dict(complete_saved_checkpoint=_measurement_roots(candidate)))
    bm,am=before.measure(),after.measure()
    return dict(scope='complete_serialized_C34_checkpoint_roots_only',
        before=asdict(bm),after=asdict(am),numeric_resident_byte_delta=am.resident_bytes-bm.resident_bytes,
        resident_entry_delta=am.resident_entries-bm.resident_entries,
        python_shallow_before=before.python_shallow_bytes,python_shallow_after=after.python_shallow_bytes,
        numeric_root_count_before=len(before.numeric),numeric_root_count_after=len(after.numeric),
        strict_checkpoint_numeric_decrease=am.resident_bytes<bm.resident_bytes and am.resident_entries<bm.resident_entries,
        original_caller_roots_missing_from_archive=True,whole_C34_LIVE_gate_proved=False,
        runtime_payment_proved=False,formal_gain=0)
