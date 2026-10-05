"""Fresh-process restore of NEW C31 full source proof, no fresh generator."""

import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c31_prepared_report_audit_v1 import audit
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
PRIOR=EXP/'results/c31_prepared_generator_20260911_v1'
DIRECTORY=EXP/'results/c31_archive_restore_guard_20260911_v1'
OLD_PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'
INPUTS={'closed_hz.pickle':'535b579e853f1ac5092e962e1c307644e89739da4cd63f3732593896649128f5',
    'closed_proof.json':'cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5',
    'result.json':'c0d9a310c5d7a814f903ed48686d8db43bd7724a9c4e270d084a4b60a6c7a20a',
    'exit.json':'3f76bfbbb78a488be55a36d79ff42b4b4d6ea16ae8d12a3a14c2acff36ffa8f1'}


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    DIRECTORY.mkdir()
    freeze=dict(input_sha256=INPUTS,guard_source_sha256=_sha256(Path(__file__)),provenance=_provenance(ROOT),
        new_generator_native_solver_authorized=False,formal_gain=0)
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',freeze)
    started=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        if any(_sha256(PRIOR/n)!=sha for n,sha in INPUTS.items()):raise ValueError('completed C31 archive drift')
        prior=json.loads((PRIOR/'result.json').read_text());ending=json.loads((PRIOR/'exit.json').read_text())
        if (not prior['completed'] or ending.get('worker_exit_code')!=0 or ending.get('tests_exit_code')!=0
                or ending.get('timeout_s') or ending['source_drift'] or ending['provenance_drift']
                or prior['provenance']!=freeze['provenance']):
            raise ValueError('not a completed matching-source new proof qualification')
        if any(_sha256(EXP/n)!=sha for n,sha in prior['source_sha256'].items()):
            raise ValueError('all original source/dependency/reference hashes must match')
        with (PRIOR/'closed_hz.pickle').open('rb') as f:saved=pickle.load(f)
        with (EXP/'results/c25_live_relu_20260911_v1/relu78.pickle').open('rb') as f:old_saved=pickle.load(f)
        raw=(PRIOR/'closed_proof.json').read_bytes()
        if (saved['schema']!='c31_new_checked_prepared_Closed_v1' or saved['proof_bytes']!=raw
                or saved['closed_proof_sha256']!=INPUTS['closed_proof.json']
                or saved['identity']!=prior['new_complete_source_proof']
                or saved['source_sha256']!=prior['source_sha256'] or saved['provenance']!=freeze['provenance']):
            raise ValueError('new full proof archive binding changed')
        restored=restore(saved['fields'],raw,expected_proof_sha256=INPUTS['closed_proof.json'])
        old=restore(old_saved['closed_fields'],old_saved['closed_proof_bytes'],expected_proof_sha256=OLD_PROOF)
        pool=WorkPool(256_000_000)
        report_proof=audit(restored,old,pool=pool)
        if report_proof!=prior['complete_report_proof'] or report_proof!=saved['complete_report_proof']:
            raise ValueError('restored complete source/counter/report proof differs')
        try:restore(saved['fields'],old_saved['closed_proof_bytes'],expected_proof_sha256=OLD_PROOF)
        except ValueError:old_rejected=True
        else:raise ValueError('old full proof incorrectly accepts the new report')
        if (restored.fingerprint()!=prior['new_closed_identity'] or hasattr(restored,'nodes')
                or source_digest(restored.hz)!=prior['new_HZ_sha256']
                or restored.report['prepared_encoding']['head_metadata_retained_after_alias']
                or not saved['all_new_graph_fields_physically_retired']):
            raise ValueError('restored new identity/graph/head retirement changed')
        restored.validate();old.validate()
        record.update(completed=True,new_closed_identity=restored.fingerprint(),
            new_HZ_sha256=source_digest(restored.hz),new_proof_sha256=hashlib.sha256(raw).hexdigest(),
            all_HZ_maps_owners_UIDs_and_report_fields_rechecked=True,
            all_MAIN_owner_words=len(restored.owners),all_lineage_slots=len(restored.eq_roots),
            all_UID_slabs=len(restored.uid_slabs),all_radix_definitions=len(restored.def_rows),
            old_proof_substitution_rejected=old_rejected,no_definition_graph_or_head_lists_restored=True,
            diagnostic_work=pool.used,new_generator_executed=False,new_native_relu_executed=False,
            solver_executed=False,whole_live_path_proved=False)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256=INPUTS,guard_source_sha256=freeze['guard_source_sha256'],provenance=freeze['provenance'])
        _atomic_exclusive_json(DIRECTORY/'result.json',record);print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
