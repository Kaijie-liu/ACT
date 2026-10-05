"""Fresh read-only restoration of an externally hash-bound C65 physical proof."""
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored,metadata
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c66_packed_frontier_20260913_v1'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    binding=json.loads((RUN/'restore_input_binding.json').read_text())
    if _sha256(RUN/'actual/result.json')!=binding['actual_result_sha256']:raise ValueError('independent completed physical result changed')
    actual=json.loads((RUN/'actual/result.json').read_text())
    if not actual['completed']:raise ValueError('completed independent source/storage proof required')
    expected=actual['data']['archive_sha256'];pool=WorkPool(256_000_000);started=time.monotonic()
    result=dict(completed=False,formal_gain=0,native_or_LIVE_admission=False)
    try:
        def build():
            with (RUN/'actual/physical_hz.pickle').open('rb') as stream:
                saved,decoder=load(stream,expected_sha256=expected,pool=pool,enabled=True)
            layout=numeric_layout(saved,pool)
            if layout.resident_entries>64_000_000:raise MemoryError('complete restored entry cap')
            pool.charge('c65_complete_restored_state_fingerprint',int(layout.resident_entries)+1024)
            proof=check_restored(saved);meta=metadata(saved,pool=pool)
            if proof['physical_identity']!=actual['data']['physical_identity']:raise ValueError('restored source proof identity differs')
            return dict(physical_identity=proof['physical_identity'],entries=layout.resident_entries,
                numeric_bytes=layout.resident_bytes,metadata=meta,decoder=decoder,
                archive_sha256=expected,complete_restored_source_proof_bound=True)
        data,stats=measured(build,observe=lambda s:result.update(measurement=s));result.update(completed=True,data=data)
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,diagnostic_work=pool.used,work_parts=dict(pool.parts))
        _atomic_exclusive_json(RUN/'restore_result.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
