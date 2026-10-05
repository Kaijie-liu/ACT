"""Complete authenticated mask census; no HZ restoration or verifier run."""
from dataclasses import asdict
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, metadata, fingerprint
from experiments.neural_hz_20260831.c121_f4_mask_worker_v1 import StagePool
from experiments.neural_hz_20260831.c122_channel_route_v1 import route, STATS_FIELDS
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c122_channel_route_20260927_v1'
PREV = EXP/'results/c121_f4_mask_20260922_v1'


def payload_digest(arrays, authentication):
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(repr((name, array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
        authentication['numeric_payload_bytes_hashed'] += int(array.nbytes)
    authentication['numeric_payload_hash_calls'] += 1
    return digest.hexdigest()


def build(pool, held, authentication):
    # Fixed reservation is nonrefundable and precedes archive decode/inspection.
    pool.charge('c122_complete_mask_archive_decode_and_validation_reservation', 4_000_000)
    jp, ap = PREV/'complete_census.json', PREV/'complete_census_arrays.npz'
    if jp.stat().st_size > 128*1024 or ap.stat().st_size > 128*1024:
        raise MemoryError('authenticated ordinary mask archive exceeds frozen decode envelope')
    raw = jp.read_bytes()
    source = json.loads(raw)
    old = source['report']
    if (old['nodes_scanned'] != 36 or old['eligible_operators'] != 8
        or old['active_operators'] != 4 or old['total_tiles'] != 52
        or len(old['records']) != 52 or old['selected_positions']
        or not old['all_structurally_eligible_nodes_inspected']):
        raise ValueError('complete authenticated structural inventory differs')
    operators = [op for op in old['operators'] if op['active']]
    if len(operators) != 4 or sum(op['tiles'] for op in operators) != 52:
        raise ValueError('complete active operator/tile inventory differs')
    expected = {}
    for op in operators:
        node, count = op['node'], op['tiles']
        c, k = op['input_shape'][0], op['output_shape'][0]
        if not op['original_kernel_all_nonzero'] or not op['kernel_scanned']:
            raise ValueError('the archived dense-original precondition was not proved')
        for name, dtype, shape in (
            ('input_masks','uint64',(count,c)), ('output_masks','uint16',(count,k)),
            ('positions','int32',(count,2)), ('bills','int64',(count,len(old['bill_fields']))),
            ('direct_known','bool',(count,)), ('topology_candidates','bool',(count,))):
            expected[f'evidence_{node}_{name}'] = (dtype, shape)
    if len(expected) != 24 or sum(np.prod(shape) for _,shape in expected.values()) > 20_000:
        raise ValueError('complete mask archive shape envelope differs')
    held.update(original_census_raw=raw, original_census=source, expected_schema=expected,
                arrays={}, records=[], evidence={}, authentication=authentication)
    # Validate uncompressed NPY headers before numpy can allocate their shapes.
    with zipfile.ZipFile(ap) as zipped:
        members = zipped.infolist()
        if (len(members) != len(expected)
            or {m.filename for m in members} != {n+'.npy' for n in expected}
            or sum(m.file_size for m in members) > 128*1024):
            raise ValueError('complete uncompressed archive member envelope differs')
        for member in members:
            if member.compress_type != zipfile.ZIP_STORED:
                raise ValueError('only original uncompressed mask archives authorized')
            dtype, shape = expected[member.filename[:-4]]
            with zipped.open(member) as stream:
                if np.lib.format.read_magic(stream) != (1,0):
                    raise ValueError('original NPY header version differs')
                got_shape, fortran, got_dtype = np.lib.format.read_array_header_1_0(stream)
                if (got_shape != shape or got_dtype != np.dtype(dtype) or fortran
                    or member.file_size-stream.tell() != int(np.prod(shape))*got_dtype.itemsize):
                    raise ValueError('complete original NPY allocation header differs')
    with np.load(ap, allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != set(expected):
            raise ValueError('archive members omit or add a numeric root')
        for name, (dtype, shape) in expected.items():
            value = archive[name]
            if value.dtype != np.dtype(dtype) or value.shape != shape:
                raise ValueError('authenticated packed array dtype/shape differs')
            held['arrays'][name] = value
    before = payload_digest(held['arrays'], authentication)
    route_pool = StagePool(pool, 16_000_000)
    direct_column = old['bill_fields'].index('direct_nnz')
    seen = set()
    for source_row in old['records']:
        node, tile = source_row['node'], source_row['tile']
        prefix = f'evidence_{node}_'
        key = f'node_{node}_tile_{tile}'
        if key in seen:
            raise ValueError('duplicate tile hides an uninspected original tile')
        seen.add(key)
        arrays = held['arrays']
        if arrays[prefix+'positions'][tile].tolist() != [source_row['y'],source_row['x']]:
            raise ValueError('original tile coordinate binding differs')
        direct = source_row['cost']['bill']['direct_nnz']
        if (not arrays[prefix+'direct_known'][tile]
            or int(arrays[prefix+'bills'][tile,direct_column]) != direct):
            raise ValueError('original direct count binding differs')
        report, evidence = route(arrays[prefix+'input_masks'][tile],
            arrays[prefix+'output_masks'][tile], direct, pool=route_pool, enabled=True)
        held['records'].append(dict(node=node,tile=tile,y=source_row['y'],x=source_row['x'],report=report))
        held['evidence'][key] = evidence
    if len(seen) != 52:
        raise ValueError('incomplete structural channel census')
    if payload_digest(held['arrays'], authentication) != before:
        raise ValueError('channel routing altered original archive payloads')
    emit_pool = StagePool(pool, 2_000_000)
    flat = {key+'_'+name:array for key, evidence in held['evidence'].items()
            for name,array in evidence.items()}
    held['evidence_archive_views'] = flat
    emit_pool.charge('c122_complete_numeric_evidence_encoding',
                     1024+16*sum(int(a.size) for a in flat.values()))
    with (RUN/'complete_channel_arrays.npz').open('xb') as stream:
        np.savez(stream, **flat)
        stream.flush()
        os.fsync(stream.fileno())
    report = dict(all_original_52_tiles_inspected=True, tiles=len(held['records']),
        conditional_candidates=sum(int(r['report']['conditional_candidate']) for r in held['records']),
        empty_routes=sum(int(r['report']['selected_channels']==0) for r in held['records']),
        records=held['records'], stats_fields=list(STATS_FIELDS),
        source_mask_payload_digest=before,
        source_census_sha256=_sha256(jp), source_array_archive_sha256=_sha256(ap),
        new_evidence_sha256=_sha256(RUN/'complete_channel_arrays.npz'),
        original_HZ_loaded=False, original_runtime_root_ledger_is_historical_only=True,
        current_ledger_scope='all_current_archived_masks_and_new_channel_census_state',
        source_coordinate_survival_unproved=True, actual_global_reserves_unbound=True,
        whole_HZ_physical_reduction_unproved=True, kernel_transform_executed=False,
        source_or_LIVE_admitted=False, solver_calls=0, formal_gain=0)
    held['complete_report'] = report
    emit_pool.charge('c122_complete_JSON_report_encoding_reservation', 262144)
    _atomic_exclusive_json(RUN/'complete_channel_census.json', report)
    ledger_pool = StagePool(pool, 12_000_000)
    layout = numeric_layout(held, ledger_pool)
    meta = metadata(held, ledger_pool)
    digest, shallow = fingerprint(held, layout, ledger_pool)
    if layout.resident_entries > 64_000_000 or meta['opaque_inherited_ids']:
        raise MemoryError('complete current mask census storage gate failed')
    ledger = dict(numeric=asdict(layout), known_metadata=meta,
                  fingerprint=digest, python_shallow_bytes=shallow)
    ledger_pool.charge('c122_post_ledger_JSON_reporting_reserve', 262144)
    _atomic_exclusive_json(RUN/'complete_held_ledger.json', ledger)
    if pool.used > 197_276_188:
        raise MemoryError('carried and complete new diagnostic bound exceeded')
    return dict(report=report, ledger=ledger, current_arrays_only=True,
                historical_C121_runtime_roots_not_restored=True)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3,16*1024**3))
    freeze = json.loads((RUN/'preregistered.json').read_text())
    pool = WorkPool(256_000_000)
    pool.charge('carried_C121_complete_diagnostic_and_reporting', 163_276_188)
    auth = dict(source_file_bytes_hashed=0,source_file_hash_calls=0,
                numeric_payload_bytes_hashed=0,numeric_payload_hash_calls=0)
    held, record = dict(freeze=freeze), dict(completed=False,solver_calls=0,formal_gain=0)
    started = time.monotonic()
    fatal = (RUN/'fatal.log').open('x')
    faulthandler.enable(file=fatal, all_threads=True)

    def drift():
        changed = False
        for name, expected in freeze['source_sha256'].items():
            p = EXP/name
            changed |= _sha256(p) != expected
            auth['source_file_bytes_hashed'] += p.stat().st_size
            auth['source_file_hash_calls'] += 1
        return changed

    def input_drift():
        return any(_sha256(Path(n)) != h for n,h in freeze['input_sha256'].items())

    try:
        if drift() or input_drift() or _provenance(ROOT) != freeze['provenance']:
            raise ValueError('complete frozen source/production drift')
        data, stats = measured(lambda: build(pool,held,auth),
                              observe=lambda m: record.update(measurement=m))
        record.update(completed=True,data=data,measurement=stats)
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable()
        fatal.close()
        changed = drift()
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-started,
            source_drift=changed,input_drift=input_drift(),
            provenance_drift=_provenance(ROOT)!=freeze['provenance'],
            authentication_traffic=auth,numeric_hash_traffic_in_token_pool=False,
            all_CPU_work_in_generation_cap=False)
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='data'}),flush=True)
    if (not record['completed'] or record['source_drift'] or record['input_drift']
        or record['provenance_drift']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
