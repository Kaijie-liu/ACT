"""Portable proof-bound physical state and complete known Python metadata.

This is not a native-admitted Closed or a solver permit. A persisted state is
trusted only through an independent expected archive hash and its full proof.
"""
import hashlib
import json
import sys
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c62_physical_measure_v1 import OPAQUE
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays,operator_digest

MAPS=('keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','owners','uid_slabs','radix_gauges')
FIELDS=frozenset(('expression','hz','old_n_cont','old_n_bin','old_n_eq','logical_n_cont','report',*MAPS))


def fingerprint(fields):
    if set(fields)!=FIELDS:raise ValueError('unregistered physical state field')
    h=hashlib.sha256(b'c65_complete_physical_original_boundary_v1')
    def token(v):h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode()+b'\0')
    expression=fields['expression'];sources={};operators={}
    token((source_digest(fields['hz']),fields['old_n_cont'],fields['old_n_bin'],fields['old_n_eq'],fields['logical_n_cont'],
        expression.frame_id,expression.n_out,digest_arrays(expression.bias),fields['report']))
    for term in expression.terms:
        if id(term.source) not in sources:sources[id(term.source)]=(len(sources),source_digest(term.source))
        ops=[]
        for op in term.operators:
            if id(op) not in operators:operators[id(op)]=(len(operators),operator_digest(op))
            ops.append(operators[id(op)])
        sid,sha=sources[id(term.source)];token((sid,sha,ops))
    token(digest_arrays(*(fields[k] for k in MAPS)))
    return h.hexdigest()


def metadata(value,*,pool):
    """Traverse all known expression/operator fields; no opaque-ID cancellation."""
    seen=set();total=0
    def visit(v,buffer=False):
        nonlocal total
        if id(v) in seen:return
        seen.add(id(v));pool.charge('c65_complete_known_python_metadata',8)
        size=sys.getsizeof(v)
        if type(v) is np.ndarray:
            total+=size-(v.nbytes if v.flags.owndata else 0)
            if v.base is not None:visit(v.base,True)
        elif type(v) is memoryview:total+=size;visit(v.obj,True)
        elif type(v) in (bytes,bytearray):total+=size-(len(v) if buffer else 0)
        elif type(v) is dict:
            total+=size
            for k,x in v.items():visit(k);visit(x)
        elif type(v) in (tuple,list):
            total+=size
            for x in v:visit(x)
        elif type(v) in (SparseHZono,*OPAQUE) or sp.isspmatrix_csr(v):
            total+=size;visit(vars(v))
        elif v is None or type(v) in (str,int,float,bool) or isinstance(v,np.generic):total+=size
        else:raise ValueError('unregistered complete Python metadata: '+type(v).__name__)
    visit(value)
    return dict(nonoverlapping_known_metadata_bytes=int(total),all_expression_and_operator_metadata_traversed=True,
        opaque_identity_cancellation_used=False,python_allocator_occupancy_not_measured=True)


def bind_proof(fields,proof,*,source_sha256):
    identity=fingerprint(fields)
    record=dict(schema='c65_source_bound_physical_proof_record_v1',physical_identity=identity,
        source_checkpoint_sha256=source_sha256,proof=proof,native_or_LIVE_admission=False,formal_gain=0)
    raw=json.dumps(record,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    return identity,raw


def check_restored(payload):
    if payload['schema']!='c65_graph_free_physical_archive_v1' or payload['native_or_LIVE_admission']:
        raise ValueError('explicit non-admitted physical archive required')
    fields=payload['fields'];raw=payload['proof_bytes'];record=json.loads(raw)
    if (hashlib.sha256(raw).hexdigest()!=payload['proof_sha256'] or record['physical_identity']!=fingerprint(fields)
            or record['native_or_LIVE_admission'] or record['proof']['new_native_or_LIVE_admission']):
        raise ValueError('restored complete physical state does not bind its source proof')
    return record
