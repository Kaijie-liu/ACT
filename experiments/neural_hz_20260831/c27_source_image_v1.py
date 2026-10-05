"""Authenticate the complete C24 original source image of reversible maps.

This checks the OLD proof image, not the correctness of a NEW splice. A new
unit-plan/row/ownership proof is independently required. No old maps are
materialized; no old Closed receipt is attached to the changed object.
"""

from dataclasses import fields as dataclass_fields
import hashlib
import json
from experiments.neural_hz_20260831.c24_closed_state_v1 import Closed,MAPS
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays,operator_digest
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c27_reversible_lineage_v1 import inverse_digest


def fingerprint(fields):
    expected={f.name for f in dataclass_fields(Closed)}-{'origin_binding','receipt','seal'}
    if type(fields) is not dict or set(fields)!=expected:
        raise ValueError('complete registered original closed-source fields required')
    h=hashlib.sha256(b'c24_checked_closed_nonconvex_HZ_v1')
    def token(v):h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode()+b'\0')
    expr=fields['expression']; sources={}; operators={}
    token((source_digest(fields['hz']),fields['old_n_cont'],fields['old_n_bin'],fields['old_n_eq'],
        fields['logical_n_cont'],expr.frame_id,expr.n_out,digest_arrays(expr.bias,fields['keep']),fields['report']))
    for term in expr.terms:
        source=term.source; sid=sources.setdefault(id(source),len(sources))
        ops=[(operators.setdefault(id(op),len(operators)),operator_digest(op)) for op in term.operators]
        token((sid,source_digest(source),ops))
    token(inverse_digest(fields['eq_roots'],fields['eq_scales'],
        tuple(fields[k] for k in (*MAPS[2:],'owners','uid_slabs'))))
    return h.hexdigest()


def verify(fields,raw_proof,*,expected_proof_sha256):
    if type(raw_proof) is not bytes or hashlib.sha256(raw_proof).hexdigest()!=expected_proof_sha256:
        raise ValueError('independently anchored full original proof is missing or changed')
    proof=json.loads(raw_proof)
    original=proof.get('identity',{}).get('original_affine_proof',{})
    if (proof.get('schema')!='c24_independent_closed_source_proof_v1' or proof.get('completed') is not True
            or proof.get('identity',{}).get('status')!='EXACT_ORIGINAL_DAG_AND_QUOTIENT'
            or original.get('all_redundant_main_and_radix_boxes_proved') is not True
            or proof.get('complete_graph_free_row_maps_equal') is not True):
        raise ValueError('incomplete full original source/box/UID proof')
    actual=fingerprint(fields)
    if actual!=proof['closed_identity']:
        raise ValueError('reversible lineage does not reproduce the complete original source image')
    return {'complete_original_source_image_sha256':actual,'complete_original_proof_sha256':expected_proof_sha256,
        'full_original_lineage_slots_restored_in_hash':len(fields['eq_roots']),
        'full_old_map_arrays_allocated':False,'new_splice_math_proved_by_this_check':False,'formal_gain':0}
