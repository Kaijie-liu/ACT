"""Fresh-process proof check of the saved provisional C27 journal only."""

import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c27_source_image_v1 import verify as verify_original
from experiments.neural_hz_20260831.c27_reference_transfer_v1 import verify as verify_semantics
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
PRIOR=EXP/'results/c27_owned_journal_20260911_v1'
REFERENCE=EXP/'results/c26_transplant_census_20260911_v1'
DIRECTORY=EXP/'results/c27_archive_restore_guard_20260911_v1'
INPUTS={'journal.pickle':'47f9984ce00727f82234b4b0683a78ef14a7397513172b0f94ff0fdcb83ba1b6',
    'result.json':'5179eaf4753b0d81ff817fd6786845f09161303e66e91cbf4261dace638125fa',
    'exit.json':'d64ed3602e28caa89b97ef0826f354ebbf6f7ad341f5585ed8732e6ded9e63ad'}
REF_SHA='55aa00df127ca394dc3e53d4b2923c8d6fc7bce543062b5cbf5057414e79b789'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    DIRECTORY.mkdir()
    freeze={'input_sha256':INPUTS,'independent_reference_sha256':REF_SHA,
        'guard_source_sha256':_sha256(Path(__file__)),'provenance':_provenance(ROOT),
        'new_generator_native_or_solver_authorized':False,'formal_gain':0}
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',freeze)
    started=time.monotonic();record={'completed':False,'formal_gain':0}
    try:
        if any(_sha256(PRIOR/n)!=sha for n,sha in INPUTS.items()) or _sha256(REFERENCE/'lineage.pickle')!=REF_SHA:
            raise ValueError('completed archive drift before deserialization')
        prior=json.loads((PRIOR/'result.json').read_text())
        ending=json.loads((PRIOR/'exit.json').read_text())
        if (not prior['completed'] or ending.get('worker_exit_code')!=0 or ending.get('tests_exit_code')!=0
                or ending.get('timeout_s') or ending['source_drift'] or ending['provenance_drift']
                or freeze['provenance']!=prior['provenance']):
            raise ValueError('not a completed matching-source ownership qualification')
        if any(_sha256(EXP/n)!=sha for n,sha in prior['source_sha256'].items()):
            raise ValueError('inherited complete source manifest changed')
        with (PRIOR/'journal.pickle').open('rb') as handle:saved=pickle.load(handle)
        with (REFERENCE/'lineage.pickle').open('rb') as handle:reference=pickle.load(handle)
        if (saved['schema']!='c27_provisional_reversible_journal_v1'
                or saved['live_admission_certificate'] is not False
                or saved['contains_valid_new_closed_HZ'] is not False
                or saved['reference_archive_sha256']!=REF_SHA
                or saved['actual_post_HZ_sha256']!=reference['source_post_HZ_sha256']):
            raise ValueError('wrong journal scope/reference or invented native admission')
        fields=saved['original_fields_with_reversible_lineage'];lineage=saved['lineage']
        if any(k in fields for k in ('receipt','seal','origin_binding')):
            raise ValueError('old runtime Closed receipt was retained')
        if lineage.eq_roots is not fields['eq_roots'] or lineage.eq_scales is not fields['eq_scales']:
            raise ValueError('persistence duplicated the edited maps')
        old=verify_original(fields,saved['original_proof_bytes'],expected_proof_sha256=PROOF)
        if old!=saved['original_image_proof'] or old['complete_original_source_image_sha256']!=reference['closed_identity']:
            raise ValueError('restored complete original image differs')
        pool=WorkPool(256_000_000)
        semantic=verify_semantics(lineage,reference['draft'],
            expected_reference_fingerprint=reference['draft'].fingerprint(),pool=pool)
        if semantic!=saved['semantic_reference_proof'] or semantic!=prior['new_semantic_reference_proof']:
            raise ValueError('restored complete new semantics differ')
        record.update(completed=True,complete_original_image=old,complete_new_semantics=semantic,
            shared_edited_maps_preserved=True,old_receipt_not_restored=True,
            fresh_ownership_permit_not_restored=True,live_admission_certificate=False,
            new_generator_executed=False,new_native_relu_executed=False,solver_executed=False,
            complete_inherited_source_manifest_checked=True,diagnostic_work=pool.used)
    except Exception as exc:
        record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256=INPUTS,source_sha256=freeze['guard_source_sha256'],provenance=freeze['provenance'])
        _atomic_exclusive_json(DIRECTORY/'result.json',record);print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
