"""Fixed full-size synthetic object; preparation only, no verifier calls.

Do not widen the old tiny-model factory. Preserve the registered factory's
float32 initialization followed by binary64 promotion and freeze its bytes.
"""
from scoped_proof.io import ROOT,load,sha
from source_enclosure.format import identity

CONFIG='configs/h2_capacity_recipe_20261001.json'
PROTOCOL_SHA='0666ef7d0cf982501a92bdb9ded128f47dcfd322e33e4d2353b507966f6868b9'
FILES=(CONFIG,'scoped_source/capacity_intake.py','scoped_source/capacity_prepare.py',
       'scoped_source/capacity_sourcecheck.py','scoped_source/capacity_prep_audit.py',
       'scripts/archive_h2_capacity_preparation.py',
       'scoped_source/capture.py','scoped_source/sparse_intake.py','scoped_source/graph.py',
       'scoped_source/factored_source.py','scoped_source/factored_io.py',
       'router_source/checker.py','source_enclosure/format.py',
       'scoped_proof/io.py','scoped_proof/supervisor.py',
       'act/back_end/moe/factory.py','act/back_end/moe/model.py','act/back_end/moe/schema.py',
       'scoped_source/endpoint_intake.py','scoped_source/endpoint_source_controls.py',
       'configs/h2_model_intake_controls_20260930.json','configs/h2_source_controls_20260930.json')


def protocol():
    value=load(ROOT/CONFIG)
    if identity(value)!=PROTOCOL_SHA: raise ValueError('frozen capacity recipe')
    return value


def sources():
    protocol()
    # Bind the repository-side helper bundle, including transitive/lazy imports.
    # External libraries remain separately versioned dependencies, not source proofs.
    names=set(FILES)
    for directory in ('scoped_source','scoped_proof','source_enclosure','router_source',
                      'upstream_source','full_source','act'):
        names.update(str(path.relative_to(ROOT)) for path in (ROOT/directory).rglob('*.py'))
    return {name:sha(ROOT/name) for name in sorted(names)}


def fixture(case):
    cfg=protocol()
    if case=='tiny_control':
        from scoped_source.endpoint_intake import model_fixture
        return model_fixture()
    if case!='full_size': raise ValueError('fixed capacity case only')
    import torch
    from act.back_end.moe.factory import OutputMoEFactoryConfig,build_output_moe
    from act.back_end.moe.schema import GateKind
    if torch.get_default_dtype()!=torch.float32 or str(torch.get_default_device())!='cpu' or torch.cuda.is_available():
        raise ValueError('fixed CPU float32 initialization environment')
    config=OutputMoEFactoryConfig(input_shape=cfg['input_shape'],num_classes=cfg['num_classes'],
        num_experts=cfg['num_experts'],top_k=cfg['top_k'],gate=GateKind(cfg['gate']),
        router_hidden=cfg['router_hidden'],expert_hidden=cfg['expert_hidden'],seed=cfg['seed'])
    model=build_output_moe(config).cpu().double().eval()
    state=model.state_dict()
    if len(state)!=cfg['tensor_count'] or sum(t.numel() for t in state.values())!=cfg['parameter_count']:
        raise ValueError('registered full-size parameter inventory')
    center=torch.full((1,*cfg['input_shape']),.5,dtype=torch.float64,device='cpu')
    return model,center,{k:cfg[k] for k in ('label','radius','margin','clip')}


def expected_request(case,request):
    """Small metadata admission; it does not establish graph semantics."""
    cfg=protocol()
    if case=='tiny_control':
        from scoped_source.endpoint_intake import expected_source
        if request!=expected_source()['request']: raise ValueError('tiny request identity')
        return
    if case!='full_size': raise ValueError('capacity case')
    from fractions import Fraction as F
    expected={k:cfg[k] for k in ('label','top_k','radius','margin','clip')}
    expected.update(experts=cfg['num_experts'],classes=cfg['num_classes'],training=False,
                    gate='SELECTED_SOFTMAX',tie_policy='ANY_LEGAL_TOPK')
    if any(request[k]!=v for k,v in expected.items()): raise ValueError('full-size domain/property')
    if (request['model_state']['parameter_count']!=cfg['parameter_count'] or
        request['model_state']['tensor_count']!=cfg['tensor_count'] or
        request['center']['shape']!=[1,*cfg['input_shape']] or request['center']['dtype']!='torch.float64'):
        raise ValueError('full-size shape/state')
    import hashlib
    import struct
    from source_enclosure.format import compact
    h=hashlib.sha256(); h.update(b'torch.float64'); h.update(compact([1,*cfg['input_shape']]))
    h.update(struct.pack('<d',.5)*3072)
    if request['center']['sha256']!=h.hexdigest(): raise ValueError('fixed full-size input identity')
    if F(request['radius'])<=0: raise ValueError('nonzero full input domain')
