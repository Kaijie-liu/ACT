"""Read-only fresh-process reconstruction of the completed C28 Plan artifact."""

from dataclasses import fields
import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan,compile_lineage
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
PRIOR=EXP/'results/c28_consumer_discovery_20260911_v1'
DIRECTORY=EXP/'results/c28_archive_restore_guard_20260911_v1'
LIVE=EXP/'results/c25_live_relu_20260911_v1/relu78.pickle'
REFERENCE=EXP/'results/c26_transplant_census_20260911_v1/lineage.pickle'
INPUTS={'plans.json':'1a0f0d72c2a2aebf8e4814116512e187a597e5160a27934168ea6543ed3b8b6c',
    'result.json':'c0d17f83f0d9d354a7452e0a99e5689afe8130337bca098c0eede7a8f42f9598',
    'exit.json':'77dffbda0b1303358dea110f745f2c5ed36a5800353ca459bc9d02a012a0a827'}
LIVE_SHA='685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba'
REF_SHA='55aa00df127ca394dc3e53d4b2923c8d6fc7bce543062b5cbf5057414e79b789'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    DIRECTORY.mkdir()
    freeze={'input_sha256':INPUTS,'source_archive_sha256':LIVE_SHA,'exact_reference_sha256':REF_SHA,
        'guard_source_sha256':_sha256(Path(__file__)),'provenance':_provenance(ROOT),
        'fresh_discovery_or_HZ_generation_authorized':False,'formal_gain':0}
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',freeze)
    started=time.monotonic();record={'completed':False,'formal_gain':0}
    try:
        if (any(_sha256(PRIOR/n)!=sha for n,sha in INPUTS.items())
                or _sha256(LIVE)!=LIVE_SHA or _sha256(REFERENCE)!=REF_SHA):
            raise ValueError('completed archive drift')
        prior=json.loads((PRIOR/'result.json').read_text());ending=json.loads((PRIOR/'exit.json').read_text())
        if (not prior['completed'] or ending.get('worker_exit_code')!=0 or ending.get('tests_exit_code')!=0
                or ending.get('timeout_s') or ending['source_drift'] or ending['provenance_drift']
                or prior['provenance']!=freeze['provenance']):raise ValueError('not a completed matching-source qualification')
        if any(_sha256(EXP/n)!=sha for n,sha in prior['source_sha256'].items()):raise ValueError('complete source manifest changed')
        saved=json.loads((PRIOR/'plans.json').read_text())
        if (saved['schema']!='c28_provisional_consumer_plans_v1' or saved['live_admission_certificate'] is not False
                or saved['source_archive_sha256']!=LIVE_SHA or saved['independent_C26_lineage_sha256']!=REF_SHA):
            raise ValueError('wrong provisional schema/source/reference')
        expected_fields={f.name for f in fields(Plan)}
        if any(set(v)!=expected_fields for v in saved['plans']):raise ValueError('Plan fields lost or added in persistence')
        plans=[Plan(**{**v,'tail':tuple(v['tail'])}) for v in saved['plans']]
        with LIVE.open('rb') as handle:source=pickle.load(handle)
        with REFERENCE.open('rb') as handle:reference=pickle.load(handle)
        closed=restore(source['closed_fields'],source['closed_proof_bytes'],expected_proof_sha256=PROOF)
        if (closed.fingerprint()!=saved['source_closed_sha256'] or source_digest(source['post_relu_hz'])!=saved['actual_post_HZ_sha256']
                or reference['closed_identity']!=saved['source_closed_sha256']
                or reference['source_post_HZ_sha256']!=saved['actual_post_HZ_sha256']):
            raise ValueError('complete original source identity changed')
        pool=WorkPool(256_000_000)
        lineage=compile_lineage(closed.eq_roots,closed.eq_scales,plans,old_n_cont=closed.old_n_cont,
            old_n_eq=closed.old_n_eq,pool=pool,enabled=True)
        reference['draft'].validate()
        if (lineage is None or lineage.fingerprint()!=reference['draft'].fingerprint()
                or lineage.fingerprint()!=prior['complete_deterministic_lineage_sha256']):
            raise ValueError('restored plans changed complete independently proved semantics')
        record.update(completed=True,all_restored_plans=len(plans),all_lineage_slots=len(lineage.eq_roots),
            full_source_and_semantic_identity_equal=True,complete_source_manifest_checked=True,
            metadata_compile_work=pool.used,new_discovery_executed=False,new_HZ_generated=False,
            new_native_relu_executed=False,solver_executed=False,live_admission_certificate=False)
    except Exception as exc:record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256=INPUTS,guard_source_sha256=freeze['guard_source_sha256'],provenance=freeze['provenance'])
        _atomic_exclusive_json(DIRECTORY/'result.json',record);print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
