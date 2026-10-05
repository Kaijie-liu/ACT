"""Complete original Tiny model exact-filter screen, bounded and automatically saved."""
import faulthandler
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import onnx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import prove_basis
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c85_conv_geometry_cost_v1 import count
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c85_exact_conv_20260913_v1'
MODEL_SHA='234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776'
MANIFEST_SHA='a8a0dc7504af2c6b89d099fd5c74f27aa5ae98458c5c151cbb6d0da2ef5c1f59'


def complete(pool,held,emit):
    basis=prove_basis();pool.charge('c85_complete_polynomial_basis',576*256)
    manifest_path=EXP/'manifests/tinyimagenet_2024_universe_v1.json'
    if _sha256(manifest_path)!=MANIFEST_SHA:raise ValueError('frozen current universe changed')
    manifest=json.loads(manifest_path.read_text())
    models={v['model_relative_path']:v['model_sha256'] for v in manifest['instances']}
    if len(models)!=1 or list(models.values())!=[MODEL_SHA]:raise ValueError('registered complete model set differs')
    path=Path(manifest['source_benchmark_root'])/manifest['family']/next(iter(models))
    hash_started=time.monotonic()
    if _sha256(path)!=MODEL_SHA:raise ValueError('original model changed')
    hash_elapsed=time.monotonic()-hash_started
    model=onnx.load(path,load_external_data=False);held['complete_original_model']=model
    if any(v.external_data for v in model.graph.initializer):raise ValueError('unbound external model tensor')
    shape_failure=None
    try:shaped=onnx.shape_inference.infer_shapes(model,strict_mode=True,data_prop=False)
    except Exception as exc:
        shaped=model;shape_failure=dict(type=type(exc).__name__,reason=str(exc))
    held['complete_shape_model']=shaped
    shapes={v.name:[int(d.dim_value) for d in v.type.tensor_type.shape.dim]
        for v in [*shaped.graph.input,*shaped.graph.value_info,*shaped.graph.output]}
    initializers={v.name:v for v in model.graph.initializer};arrays={};records=[]
    held['complete_selected_weights_and_transforms']=arrays
    branch=BranchPool(pool);branch.charge('c85_complete_model_node_headers',64*len(model.graph.node))
    eligible=0;conv_count=0
    for index,node in enumerate(model.graph.node):
        if node.op_type!='Conv':continue
        conv_count+=1
        attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute}
        if len(node.input)<2 or node.input[1] not in initializers:raise ValueError('non-initializer Conv weights')
        tensor=initializers[node.input[1]];dims=list(tensor.dims)
        stride=list(attrs.get('strides',[1,1]));dilation=list(attrs.get('dilations',[1,1]));groups=int(attrs.get('group',1))
        record=dict(graph_node=index,weight_name=tensor.name,weight_shape=dims,stride=stride,dilation=dilation,groups=groups)
        selected=(dims[-2:]==[3,3] and stride==[1,1] and dilation==[1,1] and groups==1)
        record['structurally_selected']=selected
        if selected:
            eligible+=1;weights=onnx.numpy_helper.to_array(tensor)
            report,data=transform(weights,pool=branch,enabled=True)
            record['coefficient_proof']=report
            arrays[f'node{index}_original_weights']=weights
            if data is not None:
                for name,value in data.items():arrays[f'node{index}_{name}']=value
                record['geometry_only_cost']=count(weights,data['native'],shapes.get(node.input[0],[]),
                    shapes.get(node.output[0],[]),list(attrs.get('pads',[0,0,0,0])),pool=branch)
            else:record['geometry_only_cost']=dict(geometry_known=False,complete_HZ_cost_proved=False)
            emit(dict(event='complete_original_filter_block',record=record,whole_work=pool.used,branch_work=branch.used))
        records.append(record)
    if conv_count!=19 or eligible!=14:raise ValueError('complete original-model geometry differs from C84')
    selected=[r for r in records if r['structurally_selected']]
    passed=all(r['coefficient_proof']['all_coefficients_exact_binary64'] and
               r['coefficient_proof'].get('coefficient_only_native_window_pass',False) for r in selected)
    if sum(v.size for v in arrays.values())>64_000_000:raise MemoryError('complete retained numeric-entry cap')
    pool.charge('c85_full_numeric_archive_export',sum(int(v.size) for v in arrays.values()))
    with (RUN/'complete_original_and_transformed_weights.npz').open('xb') as out:np.savez(out,**arrays)
    hash_started=time.monotonic()
    if _sha256(path)!=MODEL_SHA:raise ValueError('original model drift after read-only diagnostic')
    hash_elapsed+=time.monotonic()-hash_started
    return dict(basis=basis,model_sha256=MODEL_SHA,universe_manifest_sha256=MANIFEST_SHA,
        model_nodes=len(model.graph.node),all_Conv_nodes=conv_count,selected_Conv_nodes=eligible,
        all_layer_records=records,shape_inference_failure=shape_failure,
        kernel_count=sum(r['coefficient_proof']['kernels'] for r in selected),
        transformed_coefficient_count=sum(r['coefficient_proof']['transformed_coefficients'] for r in selected),
        all_selected_weights_pass_necessary_gate=passed,
        complete_original_models_and_numeric_arrays_strongly_retained=True,
        numeric_array_entries=sum(int(v.size) for v in arrays.values()),
        numeric_array_payload_bytes=sum(int(v.nbytes) for v in arrays.values()),
        numeric_payload_sum_is_not_unique_owner_or_HZ_ledger=True,
        branch_work=branch.used,branch_work_parts=branch.parts,full_HZ_physical_gate_proved=False,
        original_file_hash_seconds_separate=hash_elapsed,original_model_unchanged=True,
        actual_HZ_constructed=False,network_executed=False,solver_executed=False,formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    started=time.monotonic();pool=WorkPool(256_000_000);held={}
    result=dict(completed=False,necessary_gate_passed=False,formal_gain=0)
    with (RUN/'events.jsonl').open('x') as log,(RUN/'fatal.log').open('x') as fatal:
        def emit(v):log.write(json.dumps(dict(v,worker_elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');log.flush()
        try:
            faulthandler.enable(file=fatal,all_threads=True)
            data,_=measured(lambda:complete(pool,held,emit),observe=lambda s:result.update(measurement=s))
            result.update(completed=True,data=data,necessary_gate_passed=data['all_selected_weights_pass_necessary_gate'])
        except Exception as exc:
            result['failure']=dict(type=type(exc).__name__,reason=str(exc));emit(dict(event='diagnostic_failed',**result['failure']))
        finally:
            faulthandler.disable()
            result.update(wall_s=time.monotonic()-started,whole_work=pool.used,work_parts=pool.parts)
            _atomic_exclusive_json(RUN/'result.json',result)
            print(json.dumps({k:result[k] for k in ('completed','necessary_gate_passed','wall_s','whole_work','formal_gain')}),flush=True)
    raise SystemExit(0 if result['necessary_gate_passed'] else 2 if result['completed'] else 1)


if __name__=='__main__':main()
