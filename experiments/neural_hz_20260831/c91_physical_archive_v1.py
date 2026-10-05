"""Explicit portable circuit-source proof; never an old Closed/source receipt."""
import hashlib
import json
from experiments.neural_hz_20260831.c65_physical_archive_v1 import fingerprint as field_fingerprint
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import SCHEMA


def fingerprint(state):
    if set(state)!={'schema','fields','old_source_n_cont','old_source_n_eq','auxiliary_records','output_routes',
        'block_records','original_source_proof','original_circuit_proof','native_or_LIVE_admission','formal_gain'}:
        raise ValueError('complete explicit circuit state schema differs')
    if state['schema']!=SCHEMA or state['native_or_LIVE_admission'] or state['formal_gain']!=0:
        raise ValueError('explicit non-admitted circuit source required')
    h=hashlib.sha256(SCHEMA.encode());h.update(field_fingerprint(state['fields']).encode())
    h.update(digest_arrays(state['auxiliary_records'],state['output_routes'],state['block_records']).encode())
    h.update(json.dumps([state['old_source_n_cont'],state['old_source_n_eq']]).encode())
    h.update(state['original_source_proof']);h.update(state['original_circuit_proof'])
    return h.hexdigest()


def bind(state,proof):
    identity=fingerprint(state)
    raw=json.dumps(dict(schema='c91_complete_physical_circuit_proof_v1',identity=identity,proof=proof,
        native_or_LIVE_admission=False,formal_gain=0),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return identity,raw


def check(payload):
    if payload['schema']!='c91_complete_physical_circuit_archive_v1':raise ValueError('not a circuit source archive')
    raw=payload['proof_bytes'];record=json.loads(raw)
    if (record['schema']!='c91_complete_physical_circuit_proof_v1' or record['native_or_LIVE_admission']
        or hashlib.sha256(raw).hexdigest()!=payload['proof_sha256'] or fingerprint(payload['state'])!=record['identity']):
        raise ValueError('complete circuit source proof/fingerprint differs')
    return record


def physical_view(payload,record):
    return dict(state=payload['state'],checked_proof_record=(record['identity'],payload['proof_bytes']))
