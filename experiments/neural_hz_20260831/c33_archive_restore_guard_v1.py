"""Fresh-process full actual native splice/source/reconstruction restoration."""
import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c32_splice_binding_v1 import admit
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c33_live_splice_20260911_v1'


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered restore guard target')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();result=dict(completed=False,formal_gain=0,solver_executed=False)
    try:
        freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('guard source drift')
        qualification=json.loads((DIRECTORY/'qualification.json').read_text())
        if not qualification['passed'] or _sha256(DIRECTORY/'relu78.pickle')!=qualification['checkpoint_sha256']:
            raise ValueError('completed actual native checkpoint hash missing/changed')
        with (DIRECTORY/'relu78.pickle').open('rb') as f:saved=pickle.load(f)
        fields=saved['spliced_state_fields']
        if fields['source_proof_sha256']!=_sha256(DIRECTORY/'closed_proof.json') or fields['transfer_proof_sha256']!=_sha256(DIRECTORY/'transfer_proof.json'):
            raise ValueError('new source/splice proof substitution')
        restored,proof=admit(enabled=True,**fields)
        if restored.hz is not saved['post_relu_hz'] or restored.hz is not saved['hz_cache'][78]:
            raise ValueError('archive lost actual cache/source identity')
        if source_digest(restored.hz)!=qualification['actual_hz_sha256']:raise ValueError('actual new post-HZ changed')
        if restored.lineage.eq_roots is not restored.original_fields['eq_roots']:
            raise ValueError('archive copied shared reconstruction source map')
        restored.numeric_roots()
        old=restored.construction_report['native_payload_work'];restored.construction_report['native_payload_work']=0
        try:restored.validate()
        except ValueError:pass
        else:raise ValueError('altered construction report accepted')
        restored.construction_report['native_payload_work']=old;restored.validate()
        result.update(completed=True,whole_actual_HZ_and_source_image_authenticated=True,
            whole_semantic_lineage_and_events_authenticated=True,all_map_and_actual_cache_identities_preserved=True,
            changed_report_rejected=True,proof=proof,actual_HZ_sha256=source_digest(restored.hz),
            checkpoint_sha256=qualification['checkpoint_sha256'])
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'restore_guard.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
