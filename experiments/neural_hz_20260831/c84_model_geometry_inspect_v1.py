"""Read-only original-model Conv geometry; no execution, weights changed or HZ gain."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import onnx

EXP=Path(__file__).resolve().parent
MANIFESTS={
    'tinyimagenet_2024_universe_v1.json':'a8a0dc7504af2c6b89d099fd5c74f27aa5ae98458c5c151cbb6d0da2ef5c1f59',
    'cifar100_2024_universe_v1.json':'fa30dafe17cdcafeb08b56da66189795e1623b5556ced7a8247903fef507d948'}


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):value.update(chunk)
    return value.hexdigest()


def main():
    records=[]
    for name,expected in MANIFESTS.items():
        path=EXP/'manifests'/name
        if sha(path)!=expected:raise ValueError('frozen source-universe manifest drift')
        manifest=json.loads(path.read_text())
        root=Path(manifest['source_benchmark_root'])
        models={row['model_relative_path']:row['model_sha256'] for row in manifest['instances']}
        for rel,digest in sorted(models.items()):
            model_path=root/rel
            if sha(model_path)!=digest:raise ValueError('original model content drift')
            model=onnx.load(model_path,load_external_data=False)
            initializers={value.name:value for value in model.graph.initializer}
            groups=Counter();conv_count=0
            for node in model.graph.node:
                if node.op_type!='Conv':continue
                conv_count+=1
                if len(node.input)<2 or node.input[1] not in initializers:
                    raise ValueError('non-initializer Conv weights need a different diagnostic')
                weight=initializers[node.input[1]]
                if weight.external_data:raise ValueError('external tensors not covered by this file hash')
                attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute}
                key=(tuple(weight.dims),tuple(attrs.get('strides',[1,1])),
                     tuple(attrs.get('dilations',[1,1])),int(attrs.get('group',1)))
                groups[key]+=1
            records.append(dict(family=manifest['family'],model_relative_path=rel,model_sha256=digest,
                total_Conv_nodes=conv_count,groups=[dict(weight_shape=list(k[0]),stride=list(k[1]),
                    dilation=list(k[2]),groups=k[3],count=v,
                    F2x2_3x3_geometry_candidate=(list(k[0][-2:])==[3,3] and k[1]==(1,1)
                                               and k[2]==(1,1) and k[3]==1)) for k,v in sorted(groups.items())]))
    print(json.dumps(dict(schema='c84_original_model_geometry_only_v1',models=records,
        model_execution=False,weights_inspected_for_exact_transforms=False,
        HZ_reduction_proved=False,formal_gain=0),indent=2))


if __name__=='__main__':main()
