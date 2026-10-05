"""Full C99 + original C9 + original network inputs -> new text-only runtime proof."""
from dataclasses import asdict
import faulthandler
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c91_physical_archive_v1 import check
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c70_native_proof_v1 import digest
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c100_native_binding_v1 import admit_source,admit_native
from experiments.neural_hz_20260831.c99_circuit_journal_v2 import CircuitJournal
from experiments.neural_hz_20260831.c73_outer_query_v1 import GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c100_native_terminal_v1 import affine
from experiments.neural_hz_20260831.c100_live_runtime_v1 import extra_bound
from experiments.neural_hz_20260831.c100_runtime_final_binding_v1 import final_extra
from experiments.neural_hz_20260831.c100_terminal_preflight_v1 import dimensions
from experiments.neural_hz_20260831.c34_terminal_binding_v1 import suffix,array
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _model_float_dtype
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import ANCHOR,ANCHOR_SHA256
from experiments.neural_hz_20260831.bn_graph_faithfulness_certificate_prototype import audit_graph_faithfulness
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
from act.back_end.solver.solver_hz import sparse_hz_linear
from act.front_end.specs import InKind
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c100_fresh_circuit_terminal_20260913_v1'
OLD=EXP/'results/c99_circuit_native_20260913_v2'
ARCHIVE_SHA='76fa5ab974a38cbd5d25c20bac4967805177c04ad8f138ed0b16c1bd3ab6bc9e'
C9=EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
C9_SHA='616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
INPUT_SHA='f8f15a3928f239d57cedd38d10732183a2cc85de2f98c41e8f3ab6797c883fff'


def original_network():
    """Measurement routing only; family/iid never enters the representation rule."""
    from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch,get_onnx_input_shape
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.verifiable_model import InputLayer,InputSpecLayer,OutputSpecLayer,VerifiableModel
    from act.pipeline.verification.torch2act import TorchToACT
    family='tinyimagenet_2024';root=Path('/data1/Kane/data/vnncomp2025_benchmarks/benchmarks')/family
    rows=[line.split(',') for line in (root/'instances.csv').read_text().splitlines() if line.strip()]
    model_path=root/rows[143][0].replace('./','')
    spec_path=EXP/'vnnlib_v2_full_v1'/family/rows[143][1].replace('./','')
    if (_sha256(model_path)!='234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776'
        or _sha256(ANCHOR)!=ANCHOR_SHA256):raise ValueError('original model/graph anchor differs')
    shape=tuple(get_onnx_input_shape(model_path));model=convert_onnx_to_pytorch(model_path).eval()
    dtype=_model_float_dtype(model)
    labeled=LabeledInputTensor(tensor=torch.zeros(shape,dtype=dtype),label=torch.tensor([0]))
    queries=parse_vnnlib_queries(spec_path,labeled_tensor=labeled)
    if len(queries)!=1:raise ValueError('original single query required')
    inp,out=queries[0];inp.lb=inp.lb.to(dtype=dtype);inp.ub=inp.ub.to(dtype=dtype)
    net=TorchToACT(VerifiableModel(input_layer=InputLayer(labeled_input=labeled,shape=shape,dtype=dtype),
        input_spec=InputSpecLayer(inp),model=model,output_spec=OutputSpecLayer(out)),
        repair_batchnorm_producer_graph=True).run()
    cert=audit_graph_faithfulness(net.layers,net.preds,net.succs)
    if not cert.accepted or cert.graph_sha256!=json.loads(ANCHOR.read_text())['candidate_certificate']['graph_sha256']:
        raise ValueError('original complete corrected graph differs')
    return model,net,inp,out,shape,labeled,dict(model_sha256=_sha256(model_path),
        spec_sha256=_sha256(spec_path),graph_sha256=cert.graph_sha256)


def prepare(pool,held,result,emit):
    done=json.loads((OLD/'component/result.json').read_text())
    inverse=json.loads((OLD/'restore/result.json').read_text());end=json.loads((OLD/'exit.json').read_text())
    preflight=json.loads((OLD/'preflight/result.json').read_text())
    if not all((done['completed'],inverse['completed'],end['all_declared_stages_passed'],preflight['completed'])):
        raise ValueError('complete C99 qualification required')
    if not inverse['data']['inverse']['all_equations_exact']:
        raise ValueError('complete independent unit/local/circuit inverse required')
    with (OLD/'component/native.pickle').open('rb') as f:
        saved,decoder=load(f,expected_sha256=ARCHIVE_SHA,pool=pool,enabled=True)
    held['complete_C99_archive']=saved;layout=numeric_layout(saved,pool)
    pool.charge('c100_complete_independent_archive_authentication',int(layout.resident_entries)+1024)
    source=check(saved['source']);proof=json.loads(saved['proof_bytes']);j=saved['journal']
    if (saved['schema']!='c99_complete_offline_native_archive_v1' or saved['full_LIVE_admission']
        or hashlib.sha256(saved['proof_bytes']).hexdigest()!=done['data']['proof_sha256']
        or source_digest(saved['hz'])!=proof['native_sha256']
        or digest(saved['original_packet'])!=proof['original_packet_identity']
        or digest(saved['injected_packet'])!=proof['injected_packet_identity']
        or digest(j)!=proof['journal_identity']):raise ValueError('complete C99 source/native/phase/journal differs')
    state=saved['source']['state'];p=saved['injected_packet'];h=state['fields']['hz']
    transfer=dict(schema='c100_complete_circuit_native_transfer_v1',complete_C99_archive_authenticated=True,
        complete_independent_inverse_restored=True,archive_sha256=ARCHIVE_SHA,
        component_proof_sha256=saved['proof_sha256'],component_result_sha256=_sha256(OLD/'component/result.json'),
        restore_result_sha256=_sha256(OLD/'restore/result.json'),source_identity=source['identity'],
        source_proof_sha256=saved['source']['proof_sha256'],packet_identity=proof['injected_packet_identity'],
        new_HZ_sha256=proof['native_sha256'],journal_identity=proof['journal_identity'],
        event_sha256=hashlib.sha256(saved['events'].tobytes()).hexdigest(),
        packet_header={k:p[k] for k in ('schema','offline_only','fresh_native_execution','provenance',
            'coordinate_injection','original_packet_schema')},complete_component_proof=proof,
        inverse=inverse['data']['inverse'],original_source_proof_utf8=state['original_source_proof'].decode('utf8'),
        original_circuit_proof_utf8=state['original_circuit_proof'].decode('utf8'),full_LIVE_admission=False,formal_gain=0)
    raw=json.dumps(transfer,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    sha=hashlib.sha256(raw).hexdigest()
    s,_=admit_source(state,saved['source']['proof_bytes'],expected_sha256=transfer['source_proof_sha256'],enabled=True)
    native,_=admit_native(enabled=True,source=s,hz=saved['hz'],
        lineage=CircuitJournal(GuardedLocalSpliceJournal(**j['local']),state,j['circuit_tails'],j['schema']),
        events=saved['events'],actual_phase_image=p,transfer_proof_bytes=raw,expected_transfer_sha256=sha,
        construction_report=dict(complete_offline_C99=True,new_phase_binaries=len(p['eq_rhs']),formal_gain=0))
    held['source_native_binding']=native
    emit('complete_new_source_native_bound',dict(entries=layout.resident_entries))
    with C9.open('rb') as f:original,old_decoder=load(f,expected_sha256=C9_SHA,pool=pool,enabled=True)
    held['complete_original_C9_checkpoint']=original
    model,net,input_spec,out_spec,shape,labeled,origin=original_network()
    held.update(original_model=model,original_net=net,original_input_spec=input_spec,
        original_output_spec=out_spec,original_labeled_input=labeled)
    inputs=[layer for layer in net.layers if layer.kind=='INPUT_SPEC']
    if len(inputs)!=1 or inputs[0].params.get('kind') not in (InKind.BOX,InKind.LINF_BALL):
        raise ValueError('original single-box input graph required')
    inp=original['original_prefix_hz_cache'].get(inputs[0].id)
    if inp is None or source_digest(inp)!=INPUT_SHA:raise ValueError('complete original input HZ anchor differs')
    assertions=[layer for layer in net.layers if layer.kind=='ASSERT' and not net.succs.get(layer.id)]
    if len(assertions)!=1 or len(net.preds[assertions[0].id])!=1:raise ValueError('unique original terminal required')
    dense_id=net.preds[assertions[0].id][0]
    if len(net.preds[dense_id])!=1:raise ValueError('single original affine producer required')
    producer=net.preds[dense_id][0];dense,_,_=suffix(net,producer,pool=pool)
    weight=array(dense.params['weight']);bias=dense.params.get('bias')
    bias=None if bias is None else array(bias).reshape(-1)
    pool.charge('c100_original_final_affine_application',16*weight.size+16*native.hz.n_out)
    final=sparse_hz_linear(native.hz,sp.csr_matrix(weight),bias);held['final_hz']=final
    # Original model tensors, all original checkpoint objects and complete new
    # circuit state remain strongly held through every proof and measurement.
    roots=dict(complete_C99=saved,complete_C9=original,original_net=net,
        original_model_state=model.state_dict(),original_input_spec=input_spec,original_output_spec=out_spec,
        input_tensor=labeled.tensor,input_label=labeled.label,original_final=final,
        circuit_custody=native.numeric_roots(),original_model_provenance=origin)
    pool.charge('c100_complete_preparation_roots',64_000_000+2048)
    full=collect(SimpleNamespace(),roots);held['complete_numeric_roots']=full
    physical=full.measure();result['complete_input_union']=asdict(physical)
    if physical.resident_entries>64_000_000:raise MemoryError('complete new+original proof input union exceeds64M')
    emit('complete_original_model_input_and_native_root_union',asdict(physical))
    final_proof=affine(native,net,producer,final,inp,shape,out_spec,pool=pool,enabled=True)
    native_dims=dimensions(final,pool=pool)
    final_proof.update(complete_C99_archive_sha256=ARCHIVE_SHA,complete_original_C9_sha256=C9_SHA,
        original_model_provenance=origin,whole_terminal_LIVE_gate_proved=False,
        original_network_reexecuted=False,**native_dims)
    b=preflight['data']['bound'];extra=extra_bound(h.n_out,h.Gc.nnz+h.Gb.nnz,len(p['eq_rhs']),
        p['eq_c'].nnz+p['le_c'].nnz,len(p['eq_rhs'])+len(p['le_rhs']))
    if state['fields']['report']['total_work_upper']>b['whole_work_upper'] or state['fields']['report']['largest_branch_work_upper']>b['branch_work_upper']:
        raise ValueError('complete original source bound differs')
    terminal_extra=final_extra(len(net.layers))
    bound=dict(source_whole=b['whole_work_upper'],source_branch=b['branch_work_upper'],
        complete_C99_component_increment=done['coupled_increment'],runtime_extra_parts=extra,
        original_phase_injection_also_conservatively_paid=True,actual_layer_count=len(net.layers),
        terminal_extra=terminal_extra,whole=b['whole_work_upper']+done['coupled_increment']+sum(extra.values())+terminal_extra,
        branch=b['branch_work_upper']+done['coupled_increment']+sum(extra.values())+terminal_extra,
        separate_paid_native_payload=done['paid_native_payload_work'],
        hash_authentication_separate_and_reported=True,full_CPU_work_256M_claim=False,formal_gain=0)
    bound['fits']=bound['whole']<=256_000_000 and bound['branch']<=200_000_000
    result['bound']=bound
    if not bound['fits']:raise MemoryError('complete new runtime bound exceeds unchanged caps')
    if source_digest(inp)!=INPUT_SHA or check(saved['source'])['identity']!=source['identity']:
        raise ValueError('complete original inputs mutated during proof')
    for name,data in [('source_proof.json',s.proof_bytes),('transfer_proof.json',raw)]:
        with (RUN/name).open('xb') as f:f.write(data)
    _atomic_exclusive_json(RUN/'final_proof.json',final_proof)
    _atomic_exclusive_json(RUN/'native_bound.json',bound)
    _atomic_exclusive_json(RUN/'live_inputs.json',dict(source_proof_sha256=transfer['source_proof_sha256'],
        transfer_proof_sha256=sha,final_proof_sha256=_sha256(RUN/'final_proof.json'),
        native_bound_sha256=_sha256(RUN/'native_bound.json'),**native_dims))
    return dict(bound=bound,complete_input_union=asdict(physical),source_authentication=s.authentication,
        native_authentication=native.authentication,decoder=decoder,original_decoder=old_decoder,
        final_proof=final_proof,original_inputs_strongly_retained=True,text_only_fresh_runtime_inputs=True,
        whole_LIVE_or_score_admission=False,formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    pool=WorkPool(256_000_000);started=time.monotonic();held={};result=dict(completed=False,formal_gain=0)
    with (RUN/'prepare_events.jsonl').open('x') as log,(RUN/'prepare_fatal.log').open('x') as fatal:
        def emit(name,data):
            log.write(json.dumps(dict(event=name,worker_wall_s=time.monotonic()-started,**data))+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            freeze=json.loads((RUN/'preregistered.json').read_text())
            if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source changed')
            data,_=measured(lambda:prepare(pool,held,result,emit),observe=lambda m:result.update(measurement=m))
            result.update(completed=True,data=data)
        except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
        finally:
            faulthandler.disable();result.update(wall_s=time.monotonic()-started,
                diagnostic_work=pool.used,diagnostic_parts=pool.parts,complete_original_inputs_retained=bool(held))
            _atomic_exclusive_json(RUN/'prepare_result.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
