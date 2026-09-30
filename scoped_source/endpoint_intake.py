"""H2 fixed model-object adapter, reusing the existing capture contract.

No checkpoint, arbitrary factory, dataset, forward, or training. Heavy imports
are deferred to the timed worker. This is not native floating execution proof.
"""
from scoped_proof.io import ROOT, load
from source_enclosure.format import identity

SCHEMA = 'H2_CAPTURE_CONTROL_V1'
PROTOCOL_SHA256 = 'c496b003c786eadbf3c2dac4f03ddabe00e57cf9336e26ccf2dbb86bde0735ca'
CAPTURE_SHA256 = 'd4d33ce1e9c56cb264e27b5325f51559423f9797deb75315a624641e85b38a2e'
PRODUCER_FILES = ('configs/h2_model_intake_controls_20260930.json',
    'scoped_source/endpoint_intake.py','scoped_source/capture.py','scoped_source/sparse_intake.py',
    'act/back_end/moe/model.py','act/back_end/moe/schema.py')
FAULTS = ('','mutate_model','mutate_input','capture_exception','capture_delay',
          'missing_certificate','omit_property','rewrite_check_stdout')


def protocol():
    value = load(ROOT/'configs/h2_model_intake_controls_20260930.json')
    source = load(ROOT/'configs/h2_source_controls_20260930.json')
    if identity(value) != PROTOCOL_SHA256 or identity(source) != value['source_protocol_sha256']:
        raise ValueError('frozen H2 intake/source protocol identity')
    if value['expected_source_sha256'] != CAPTURE_SHA256: raise ValueError('captured source anchor')
    return value, source['weighted_sign']


def validate_spec(spec):
    protocol()
    if (set(spec) != {'schema','case','mode','reuse','source_sha256','control','protocol_sha256'} or
            spec['schema'] != SCHEMA or spec['case'] != 'weighted_sign' or
            spec['mode'] not in ('endpoints','mccormick') or spec['reuse'] != [] or
            spec['source_sha256'] != CAPTURE_SHA256 or spec['protocol_sha256'] != PROTOCOL_SHA256 or
            spec['control'] not in FAULTS):
        raise ValueError('fixed H2 captured object only')
    if spec['control'] == 'rewrite_check_stdout' and spec['mode'] != 'mccormick':
        raise ValueError('MC checker-output corruption control')


def specification(mode='endpoints', control=''):
    result = {'schema':SCHEMA,'case':'weighted_sign','mode':mode,'reuse':[],
        'source_sha256':CAPTURE_SHA256,'control':control,'protocol_sha256':PROTOCOL_SHA256}
    validate_spec(result)
    return result


def expected_source():
    """Independent declaration for source identity, not a captured object/cache."""
    from scoped_source.endpoint_source_controls import weighted_source
    cfg,_ = protocol(); doc = weighted_source()
    if identity(doc) != cfg['handwritten_source_sha256']: raise ValueError('original fixed declaration')
    doc['trust'] = 'declared real graph; graph/program correspondence and native floating execution are separate'
    if identity(doc) != CAPTURE_SHA256: raise ValueError('independent captured declaration')
    return doc


def model_fixture():
    """Private CPU float64 layers with frozen coefficients, without inference."""
    _, cfg = protocol()
    import torch
    from act.back_end.moe.model import OutputLevelMoE
    from act.back_end.moe.schema import OutputLevelMoESpec, GateKind
    nn = torch.nn
    def linear(weights, bias):
        layer = nn.Linear(len(weights[0]),len(weights),dtype=torch.float64,device='cpu')
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weights,dtype=torch.float64,device='cpu'))
            layer.bias.copy_(torch.tensor(bias,dtype=torch.float64,device='cpu'))
        return layer
    # Initialization is overwritten fully and does not perturb caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        router = nn.Sequential(linear(cfg['router_weights'],cfg['router_bias']))
        experts = []
        for i, carry in enumerate(cfg['output_carry_coefficients']):
            experts.append(nn.Sequential(linear(cfg['hidden_weights'],cfg['hidden_bias']),nn.ReLU(),
                linear([cfg['output_relu_coefficients']+[carry],[0.,0.,0.]],
                       [cfg['output_bias_binary64'][i],0.])))
        model = OutputLevelMoE(router,experts,
            OutputLevelMoESpec(3,2,GateKind.SELECTED_SOFTMAX)).eval()
    center = torch.zeros((1,1),dtype=torch.float64,device='cpu')
    request = {k:cfg[k] for k in ('label','radius','margin','clip')}
    return model, center, request


def receipt(doc, invocation, producer_sources):
    return {'schema':'H2_MODEL_CAPTURE_RECEIPT_V1','invocation':invocation,
        'source_sha256':identity(doc),'request_sha256':identity(doc['request']),
        'model_state':doc['request']['model_state'],'center':doc['request']['center'],
        'capture_producer_sources':{name:producer_sources[name] for name in PRODUCER_FILES},
        'kind':'SUPPORTED_CPU_FLOAT64_EVAL_TOP2','native_float_proof':False}
