"""Whole archived pre/post row-codec qualification; no fresh NN/HZ path."""

import json
import math
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c29_prepared_row_v1 import make_encoder,decode_head
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c29_prepared_rows_20260911_v1'
ORIGINAL=EXP/'results/c9_live_relu_20260906_v1/relu78.pickle'
LIVE=EXP/'results/c25_live_relu_20260911_v1/relu78.pickle'
C27=EXP/'results/c27_owned_journal_20260911_v1/result.json'
C28=EXP/'results/c28_consumer_discovery_20260911_v1/plans.json'
PROOF='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'


def qualify_rows(sources,closed,emit):
    codec=WorkPool(256_000_000);diagnostic=WorkPool(256_000_000)
    heads={};summaries={};removed=physical_rows=0
    for label,hz in sources:
        stats={'physical_rows':0,'coefficient_entries':0,'dyadic_heads':0,'dyadic_MAIN_heads':0,
            'codec_work_before':codec.used,'all_coefficients_RHS_coordinates_equal':True}
        for kind,cm,bm,rhs in ((False,hz.Ac,hz.Ab,hz.b),(True,hz.Auc,hz.Aub,hz.ub)):
            # This is an EXTERNAL diagnostic array, not a hidden free runtime
            # candidate index. Each primitive retains only its current row.
            diagnostic.charge('complete_head_oracle_array',cm.shape[0])
            codes=np.zeros(cm.shape[0],np.uint8)
            for row in range(cm.shape[0]):
                a,b=int(cm.indptr[row]),int(cm.indptr[row+1])
                c,d=int(bm.indptr[row]),int(bm.indptr[row+1])
                n=b-a+d-c
                # NEW codec accounting: retain old12/RHS-row, credit only ONE
                # actually removed absolute-value pass per real coefficient.
                # Other removed concat/min/max operations get no credit.
                codec.charge('prepared_unchanged_encoding_except_abs',11*n+12)
                current=make_encoder(hz.n_cont,hz.n_bin,n+1,head_pool=codec,enabled=True)
                index,shift=current.encode(cm.indices[a:b],cm.data[a:b],bm.indices[c:d],bm.data[c:d],
                    float(rhs[row]),inequality=kind)
                diagnostic.charge('complete_row_payload_bits_and_head',8*n+48)
                if index!=0 or shift!=0 or current.def_rows or current.packed_rows or current.relays:
                    raise ValueError('already normalized archived row changed its physical representation')
                payload=(current.ineq if kind else current.eq)[0]
                code=(current.ineq_heads if kind else current.eq_heads)[0]
                if (not np.array_equal(payload[0],cm.indices[a:b]) or not np.array_equal(payload[2],bm.indices[c:d])
                        or payload[1].dtype!=np.dtype(np.float64) or payload[3].dtype!=np.dtype(np.float64)
                        or not np.array_equal(payload[1].view(np.uint64),cm.data[a:b].view(np.uint64))
                        or not np.array_equal(payload[3].view(np.uint64),bm.data[c:d].view(np.uint64))
                        or np.float64(payload[4]).tobytes()!=np.float64(rhs[row]).tobytes()):
                    raise ValueError('prepared row changed complete coordinate/coefficient/RHS bits')
                expected=None if a==b or math.frexp(abs(float(cm.data[a])))[0]!=.5 else float(cm.data[a])
                if decode_head(code)!=expected:raise ValueError('derived head differs from independent actual coefficient')
                codes[row]=code
                stats['dyadic_heads']+=int(code!=0)
                stats['dyadic_MAIN_heads']+=int(code!=0 and closed.old_n_cont<=int(cm.indices[a])<closed.logical_n_cont)
                if current.omitted_post_magnitude_elements!=n:raise ValueError('removed-pass count differs from actual emitted coefficients')
                removed+=n;physical_rows+=1
                stats['physical_rows']+=1;stats['coefficient_entries']+=n
            heads[label+('_INEQ' if kind else '_EQ')]=codes
            emit('complete_row_block_qualified',{'source':label,'inequality':kind,'rows':cm.shape[0],
                'cumulative_codec_work':codec.used,'cumulative_diagnostic_work':diagnostic.used})
        stats['codec_work']=codec.used-stats.pop('codec_work_before')
        summaries[label]=stats
    return {'heads':heads,'source_statistics':summaries,'codec_work':codec.used,'codec_work_parts':dict(codec.parts),
        'independent_diagnostic_work':diagnostic.used,'independent_diagnostic_parts':dict(diagnostic.parts),
        'actually_omitted_full_abs_elements':removed,'all_physical_rows_checked':physical_rows,
        'old_same_rows_codec_work_model':12*(removed+physical_rows),
        'actual_new_same_rows_work_model':codec.used,'new_full_HZ_or_generator_proved':False}


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    record={'schema':'c29_complete_prepared_archived_row_qualification_v1','completed':False,'formal_gain':0,
        'fresh_original_expression_generation_executed':False,'new_native_relu_executed':False,
        'new_spliced_HZ_constructed':False,'solver_executed':False,'whole_live_path_proved':False,
        'source_sha256':freeze['source_sha256'],'provenance':freeze['provenance']}
    started=time.monotonic()
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):
            raise ValueError('source/artifact drift before deserialization')
        with ORIGINAL.open('rb') as handle:original_saved=pickle.load(handle)
        with LIVE.open('rb') as handle:live_saved=pickle.load(handle)
        if original_saved['schema']!='c9_live_relu_checkpoint_v1' or live_saved['schema']!='c25_live_relu_checkpoint_v1':
            raise ValueError('not the complete archived pre/post sources')
        closed=restore(live_saved['closed_fields'],live_saved['closed_proof_bytes'],expected_proof_sha256=PROOF)
        pre,post=original_saved['preactivation_hz'],live_saved['post_relu_hz']
        sources=[('before_alias',pre),('after_alias_and_actual_phase',post)]
        before={label:source_digest(hz) for label,hz in sources};closed_sha=closed.fingerprint()
        with (DIRECTORY/'events.jsonl').open('x') as stream:
            def emit(name,values):
                event={'event':name,**values,'worker_elapsed_s':time.monotonic()-started}
                stream.write(json.dumps(event)+'\n');stream.flush();print(json.dumps(event),flush=True)
            qualified,construction=measured_build(lambda:qualify_rows(sources,closed,emit))
            heads=qualified.pop('heads');record.update(qualified,construction=construction)
            # The saved C28 selection is used ONLY AFTER complete classification
            # to check coverage; it was never a producer/selector input.
            expected=json.loads(C28.read_text())
            if expected['actual_post_HZ_sha256']!=before['after_alias_and_actual_phase'] or expected['source_closed_sha256']!=closed_sha:
                raise ValueError('independent consumer proof belongs to another source')
            for p in expected['plans']:
                code=int(heads['after_alias_and_actual_phase'+('_INEQ' if p['inequality'] else '_EQ')][p['consumer']])
                if decode_head(code)!=-p['sign']*p['pivot']:
                    raise ValueError('complete source-derived head classification lost a previously proved consumer')
            if len(expected['plans'])!=268:raise ValueError('complete independent consumer population changed')
            record['all_268_consumer_heads_independently_covered']=True
            # An explicit prospective GENERATOR model, NOT a run or gate pass.
            old=json.loads(C27.read_text())['generation_report'];alias=old['alias_quotient']
            old_rows=alias['old_predicate_entries']-alias['old_predicate_nnz']
            # Ignore any credit for extra radix links. At most every original
            # coefficient is a single avoided abs map; RHS/centers are not.
            original_coefficient_credit=alias['old_predicate_nnz']-2*old['radix_auxiliaries']
            if old['old_predicate_work_upper']%12:raise ValueError('old source encoding ledger is not integral')
            independently_counted=(old['old_predicate_work_upper']//12-closed.old_n_eq-len(closed.ineq_roots)
                +old['continuous_edges']+old['binary_edges']+old['auxiliaries'])
            if original_coefficient_credit!=independently_counted:
                raise ValueError('actual physical-minus-radix and original logical coefficient credits disagree')
            attempts=old_rows+old['packed_logical_rows']
            new_head_work=16*attempts
            if (old_rows!=qualified['source_statistics']['before_alias']['physical_rows']
                    or alias['old_predicate_nnz']!=qualified['source_statistics']['before_alias']['coefficient_entries']):
                raise ValueError('archived unaliased population differs from original generator ledger')
            assessment={'original_generator_coefficient_abs_credit_model':original_coefficient_credit,
                'independent_logical_coefficient_count':independently_counted,
                'conservative_credit_excludes_extra_radix_links':True,'all_prepare_attempts_model':attempts,
                'new_head_decision_work_model':new_head_work,'changed_heads_recapture_not_integrated':True,
                'all_existing_rewritten_rows':alias['rewritten_rows'],
                'optional_changed_head_reclassification_work_model':16*alias['rewritten_rows'],
                'C27_plus_C28_naive_work':259_288_127,
                'prospective_plus_prepared_emission':259_288_127-original_coefficient_credit+new_head_work,
                'prospective_plus_all_changed_head_reclassification':259_288_127-original_coefficient_credit+new_head_work+16*alias['rewritten_rows'],
                'fresh_generator_or_live_integration_executed':False,'full_integration_gate_passed':False,
                'native_fused_predicate_writer_implemented':False,'archive_head_tables_are_diagnostic_only':True,'formal_gain':0}
            record['prospective_accounting_NOT_a_gate']=assessment
            if {label:source_digest(hz) for label,hz in sources}!=before or closed.fingerprint()!=closed_sha:
                raise ValueError('original complete source fields mutated')
            record['all_original_source_fields_unchanged']=True
            if qualified['codec_work']>=qualified['old_same_rows_codec_work_model']:
                raise ValueError('actually implemented row codec did not reduce its same-row model')
            emit('complete_rows_and_consumer_head_coverage',{
                'rows':qualified['all_physical_rows_checked'],'coefficients':qualified['actually_omitted_full_abs_elements'],
                'codec_work':qualified['codec_work'],'diagnostic_work':qualified['independent_diagnostic_work'],
                'all_268_heads_covered':True,'construction':construction})
            emit('prospective_generator_accounting_NOT_a_gate',assessment)
            with (DIRECTORY/'head_codes.npz').open('xb') as handle:
                np.savez(handle,**heads);handle.flush();os.fsync(handle.fileno())
            record.update(completed=True,head_array_entries=sum(len(v) for v in heads.values()),
                head_array_bytes=sum(v.nbytes for v in heads.values()),head_file_sha256=_sha256(DIRECTORY/'head_codes.npz'),
                source_HZ_sha256=before,source_closed_sha256=closed_sha)
    except Exception as exc:record['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY/'result.json',record)
        print(json.dumps({'completed':record['completed'],'failure':record.get('failure'),'formal_gain':0}),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
