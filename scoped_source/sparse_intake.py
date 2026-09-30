"""H1 model-object intake controls; no checkpoint/dataset or real-run selector.

Reuse capture(), retaining the declared-real-source/native-FP distinction.
The supported live object is snapshotted, not subsequently consulted by proof.
"""
from scoped_source.graph import clock, validate
from source_enclosure.format import identity

SCHEMA = 'H1_CAPTURE_CONTROL_V1'
PRODUCER_FILES = ('scoped_source/capture.py', 'scoped_source/sparse_intake.py',
    'scoped_source/sparse_worker.py', 'scoped_source/sparse_build.py',
    'act/back_end/moe/model.py', 'act/back_end/moe/schema.py')


def validate_fixture(options):
    if set(options) != {'seed','experts','classes','width','hidden'}:
        raise ValueError('registered dense object control only')
    for k, lo, hi in [('seed',0,1000),('experts',2,5),('classes',2,5),('width',1,16)]:
        if type(options[k]) is not int or not lo <= options[k] <= hi:
            raise ValueError('bounded object control dimensions')
    hidden = options['hidden']
    if not isinstance(hidden,list) or not 1 <= len(hidden) <= 3 or any(
            type(w) is not int or not 1 <= w <= 16 for w in hidden):
        raise ValueError('bounded dense hidden dimensions')


def model_fixture(options):
    """Ordinary seeded dense initialization; no outcome-based weight selection."""
    validate_fixture(options)
    import torch
    from act.back_end.moe.model import OutputLevelMoE
    from act.back_end.moe.schema import OutputLevelMoESpec, GateKind
    nn = torch.nn
    def network(out):
        layers = [nn.Flatten(1)]; incoming = options['width']
        for width in options['hidden']:
            layers += [nn.Linear(incoming,width,dtype=torch.float64),nn.ReLU()]
            incoming = width
        layers += [nn.Linear(incoming,out,dtype=torch.float64)]
        return nn.Sequential(*layers)
    # Preserve caller RNG; seed affects every network independently in sequence.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(options['seed'])
        model = OutputLevelMoE(network(options['experts']),
            [network(options['classes']) for _ in range(options['experts'])],
            OutputLevelMoESpec(options['experts'],2,GateKind.SELECTED_SOFTMAX)).eval()
    center = torch.zeros((1,1,options['width']),dtype=torch.float64)
    request = {'label':0,'radius':'1/8','margin':'1/100','clip':['-1','1']}
    return model, center, request


def validate_object(model):
    """Reject known overrides of execution/serialization before taking a snapshot.

Not a proof against arbitrary monkey-patching of Python classes or PyTorch.
"""
    import torch
    from act.back_end.moe.model import OutputLevelMoE
    nn = torch.nn
    if type(model) is not OutputLevelMoE or type(model.experts) is not nn.ModuleList:
        raise ValueError('exact supported model/container type required')
    if model.spec.num_experts != len(model.experts):
        raise ValueError('mutated expert count')
    if (nn.modules.module._global_forward_hooks or nn.modules.module._global_forward_pre_hooks):
        raise ValueError('global forward hooks change captured execution')
    todo = [model]; seen = set()
    while todo:
        module = todo.pop()
        if id(module) in seen: continue
        seen.add(id(module))
        if type(module) not in (OutputLevelMoE,nn.ModuleList,nn.Sequential,nn.Linear,nn.ReLU,nn.Flatten):
            raise ValueError('unsupported source object type')
        if any(callable(v) for v in vars(module).values()):
            raise ValueError('instance callable overrides unsupported')
        if any(v for k,v in vars(module).items() if k.endswith('_hooks')):
            raise ValueError('hooked execution or serialization unsupported')
        for parameter in module._parameters.values():
            if parameter is not None and type(parameter) is not nn.Parameter:
                raise ValueError('tensor subclass unsupported')
        for tensor in list(module._parameters.values())+list(module._buffers.values()):
            if tensor is not None:
                if type(tensor) not in (nn.Parameter,torch.Tensor) or any(callable(v) for v in vars(tensor).values()):
                    raise ValueError('tensor serialization overrides unsupported')
        todo.extend(m for m in module._modules.values() if m is not None)


def capture_bound(model, center, request, *, expected_source_sha256, deadline):
    """Same declared object/domain/property anchor, before any LP construction."""
    from scoped_source.capture import capture
    import torch
    if type(center) is not torch.Tensor or any(callable(v) for v in vars(center).values()):
        raise ValueError('exact input tensor without serialization overrides required')
    tick = clock(deadline); validate_object(model); tick()
    doc = capture(model,center,deadline=deadline,**request)
    # This validates operator shapes, all inventories, domain and properties too.
    validate(doc,expected_source_sha256,tick)
    return doc


def prepare(options):
    """Control preparation only: freeze independently of the timed worker."""
    import time
    from scoped_source.capture import capture
    model, center, request = model_fixture(options)
    validate_object(model)
    doc = capture(model,center,deadline=time.monotonic()+300,**request)
    validate(doc,identity(doc),lambda:None)
    return {'schema':SCHEMA,'fixture':options,'mode':'dependency','reuse':[],
        'source_sha256':identity(doc),'control':''}, doc
