"""Fresh full circuit-source restore and same complete C69 physical comparator."""
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c91_physical_archive_v1 import check,physical_view
from experiments.neural_hz_20260831.c65_physical_archive_v1 import metadata
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c91_physical_circuit_20260913_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));started=time.monotonic();pool=WorkPool(256_000_000)
    if len(sys.argv)!=2 or _sha256(RUN/'result.json')!=sys.argv[1]:raise ValueError('frozen actual source result required')
    actual=json.loads((RUN/'result.json').read_text());held={};result=dict(completed=False,formal_gain=0,full_LIVE_admission=False)
    try:
        if not actual['completed']:raise ValueError('completed full source physical qualification required')
        def build():
            with (RUN/'complete_physical_circuit.pickle').open('rb') as stream:
                payload,decoder=load(stream,expected_sha256=actual['data']['archive_sha256'],pool=pool,enabled=True)
            held['complete_restored_archive']=payload;layout=numeric_layout(payload,pool)
            if layout.resident_entries>64_000_000:raise MemoryError('complete restored entry cap')
            pool.charge('c91_complete_restored_source_fingerprint',int(layout.resident_entries)+1024)
            proof=check(payload);view=physical_view(payload,proof)
            if proof['identity']!=actual['data']['physical_identity']:raise ValueError('complete source proof identity changed on restore')
            resident=numeric_layout(view,pool);meta=metadata(view,pool=pool);before=actual['data']['physical']
            comparison=dict(restored_numeric_bytes=resident.resident_bytes,restored_entries=resident.resident_entries,
                restored_known_metadata_bytes=meta['nonoverlapping_known_metadata_bytes'],
                writer_numeric_byte_delta=resident.resident_bytes-before['after_numeric_bytes'],
                writer_entry_delta=resident.resident_entries-before['after_entries'],
                same_C69_numeric_byte_delta=resident.resident_bytes-before['before_numeric_bytes'],
                same_C69_entry_delta=resident.resident_entries-before['before_entries'])
            comparison['same_C69_known_byte_delta']=comparison['same_C69_numeric_byte_delta']+comparison['restored_known_metadata_bytes']-before['before_known_metadata_bytes']
            if any(comparison[k]>=0 for k in ('same_C69_numeric_byte_delta','same_C69_entry_delta','same_C69_known_byte_delta')):
                raise ValueError('fresh complete source no longer beats same C69 physical comparator: '+json.dumps(comparison))
            return dict(physical_identity=proof['identity'],physical=comparison,decoder=decoder,
                complete_restored_source_proof_bound=True,full_LIVE_admission=False,fresh_generation_work_proved=False)
        data,stats=measured(build,observe=lambda s:result.update(measurement=s));result.update(completed=True,data=data)
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,whole_work=pool.used,work_parts=pool.parts)
        _atomic_exclusive_json(RUN/'restore_result.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
