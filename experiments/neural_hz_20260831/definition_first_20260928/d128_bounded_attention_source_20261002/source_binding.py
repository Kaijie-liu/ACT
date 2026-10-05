"""Default-off, fixed original-source REAL coefficient binding, not a verifier.

The two authenticated models have constant first CLS queries and inference BN.
Every coefficient encloses a fixed real number; errors are query-only outer
rectangles, never fresh native inputs. No ONNX execution/shape inference occurs.
All mutable output dictionaries contain deeply immutable tuple/Fraction banks.
Only one prepared bank is retained privately; release() drops that reference.
"""
from fractions import Fraction as F
import hashlib
import json

from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import source_packet_v1 as base
from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import census_worker_v2 as helper
from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef

ZERO, ONE = F(0), F(1)
Z = (ZERO, ZERO)
MAX_ENTRIES = 64_000_000
SOURCES = {
    '246326387574617a274b6f73f7f52771df1195e03c99065ce4ad18c46b8e0984': (
        'pgd_2_3_16', 16,
        'c97aa0ec74aaf13cf3bbeba0860ee1b4e0680fe8a850b1d89d8aeb1735a30cac',
        'd5d0f9e7d273fb277e8c711fb44015a51a461e1267c28487adc517b8330c2d9f'),
    '9cc53b9edb35d40d70de1d816008a434c7e60b0404cf3cb38b9aefc189692883': (
        'ibp_3_3_8', 8,
        'e53ef11eacedf1d66b077143a1a4e84060d2f592de888a9a2f95d98583b946d4',
        'b4a4436858aa08f23fcc8791c3718367d383ed9c8896c56eec997103a6dac1d0'),
}
_ACTIVE = None


def _bits(value, budget):
    # The caller prepays two numerator/denominator bit checks, including every
    # multiply and partial sum. The arithmetic is exact, not float arithmetic.
    if type(value) is not F or max(value.numerator.bit_length(),
                                  value.denominator.bit_length()) > budget.max_bits:
        raise ValueError('binding rational bit cap/type')
    return value


def _iv(value, budget):
    budget.charge(5)
    if type(value) is not tuple or len(value) != 2:
        raise ValueError('interval tuple required')
    _bits(value[0], budget); _bits(value[1], budget)
    if value[0] > value[1]:
        raise ValueError('reversed coefficient interval')
    return value


def _padd(acc, weight, value, budget):
    """Certified inputs only: two endpoint products and two partial sums.

    16 = two reads + one sign + two mul + two add + eight bit checks + loop.
    This avoids repeated public-wrapper validation, not arithmetic accounting.
    """
    budget.charge(16)
    a, b = value if weight >= 0 else (value[1], value[0])
    p = _bits(weight * a, budget)
    q = _bits(weight * b, budget)
    return _bits(acc[0] + p, budget), _bits(acc[1] + q, budget)


def _iadd(acc, left, right, budget):
    """Known signs need two products; only two crossing intervals need four.

    18 = two reads, four sign tests, two mul, two add, eight bit checks.
    Both crossing pays another 12: two mul/four checks/six min/max tests.
    The same intervals and information are used, without a numeric rescue.
    """
    budget.charge(18)
    lp,ln,rp,rn=left[0]>=0,left[1]<=0,right[0]>=0,right[1]<=0
    if lp:
        low=(left[0],right[0]) if rp else (left[1],right[0])
        high=(left[0],right[1]) if rn else (left[1],right[1])
    elif ln:
        low=(left[1],right[1]) if rn else (left[0],right[1])
        high=(left[1],right[0]) if rp else (left[0],right[0])
    elif rp:
        low,high=(left[0],right[1]),(left[1],right[1])
    elif rn:
        low,high=(left[1],right[0]),(left[0],right[0])
    else:
        low=high=None
    if low is not None:
        p=_bits(low[0]*low[1],budget)
        q=_bits(high[0]*high[1],budget)
        return _bits(acc[0]+p,budget),_bits(acc[1]+q,budget)
    budget.charge(12)
    p = (_bits(left[0] * right[0], budget),
         _bits(left[0] * right[1], budget),
         _bits(left[1] * right[0], budget),
         _bits(left[1] * right[1], budget))
    return (_bits(acc[0] + min(p), budget),
            _bits(acc[1] + max(p), budget))


def _plus(a, b, budget):
    budget.charge(8)
    return _bits(a[0] + b[0], budget), _bits(a[1] + b[1], budget)


def _round(value, budget):
    return k.dyadic_enclose(value, budget, bits=64)


def _point(value, budget):
    budget.charge(3)
    _bits(value, budget)
    return value, value


def _patch_columns(patch, budget):
    if type(patch) is not int or patch not in (8, 16):
        raise ValueError('fixed disjoint patch layout required')
    budget.charge(32 + 8 * 3072)
    return ((),) + tuple(tuple((c * 32 + py * patch + y) * 32 + px * patch + x
        for c in range(3) for y in range(patch) for x in range(patch))
        for py in range(32 // patch) for px in range(32 // patch))


def _identity(raw_model, raw_spec, inventory, model_sha256, spec_sha256, budget):
    if (type(raw_model) is not bytes or type(raw_spec) is not bytes
            or type(inventory) is not bytes or model_sha256 not in SOURCES
            or len(raw_model) > 1_000_000 or len(raw_spec) > 1_000_000
            or len(inventory) > 1_000_000):
        raise ValueError('unregistered original source or byte cap')
    expected = SOURCES[model_sha256]
    budget.charge(4096 + len(raw_model) + 8 * len(raw_spec) + 4 * len(inventory))
    if (spec_sha256 != expected[2]
            or hashlib.sha256(raw_model).hexdigest() != model_sha256
            or hashlib.sha256(raw_spec).hexdigest() != expected[2]
            or hashlib.sha256(inventory).hexdigest() != expected[3]):
        raise ValueError('original model/spec/graph identity mismatch')
    return expected, json.loads(inventory)


def _layout(reader, inventory, patch, budget):
    """Pinned complete bytes plus explicit checked first-block layout proof.

    Shape expressions are the authenticated B=1, N=1+(32/K)^2,
    [B,N,48]->[B,N,3,16] programs, not inferred data-dependent shapes.
    """
    budget.charge(4096 + 32 * len(reader.nodes))
    if reader.opsets != {'': 9} or len(reader.model.graph.input) != 1:
        raise ValueError('fixed standard-opset9 single-input graph required')
    inp = reader.model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    if (inp.name != 'input' or inp.type.tensor_type.elem_type != 1 or len(dims) != 4
            or tuple(d.dim_value for d in dims[1:]) != (3, 32, 32)
            or not (dims[0].dim_param == 'batch_size' or dims[0].dim_value == 1)):
        raise ValueError('original NCHW input annotation mismatch')
    records = inventory.get('nodes')
    if type(records) is not list or len(records) != len(reader.nodes):
        raise ValueError('complete graph inventory required')
    for index, (node, record) in enumerate(zip(reader.nodes, records)):
        if (record.get('index') != index or record.get('op') != node.op_type
                or record.get('domain') != node.domain or record.get('name') != node.name
                or tuple(record.get('inputs', ())) != tuple(node.input)
                or tuple(record.get('outputs', ())) != tuple(node.output)):
            raise ValueError('saved graph/source port mismatch')
    transposes = {9:(0,2,1), 18:(0,2,1), 20:(0,2,1),
        46:(0,2,1,3), 49:(0,2,1,3), 50:(0,2,3,1),
        56:(0,2,1,3), 65:(0,2,1), 67:(0,2,1)}
    for index, perm in transposes.items():
        node = reader.nodes[index]
        attrs = reader.attrs(node, {'perm'})
        if node.op_type != 'Transpose' or reader.ints_attr(attrs, 'perm', ()) != perm:
            raise ValueError('head/token/channel transpose mismatch')
    ints = {1:(0,), 6:(-1,), 11:(1,), 12:(48,), 22:(0,),
        31:(-1,), 32:(3,), 33:(16,), 36:(-1,), 37:(3,),
        38:(16,), 41:(-1,), 42:(3,), 43:(16,), 58:(-1,), 59:(48,)}
    for index, values in ints.items():
        attrs = records[index]['attributes']
        if (reader.nodes[index].op_type != 'Constant' or len(attrs) != 1
                or tuple(attrs[0].get('tensor', {}).get('small_integer_values', ())) != values):
            raise ValueError('authenticated integer shape program mismatch')
    for index, axis in ((2,0),(23,0),(7,0),(13,0),(16,1),(34,0),(39,0),(44,0),(60,0),(54,3)):
        attrs = reader.attrs(reader.nodes[index], {'axis'})
        if reader.int_attr(attrs, 'axis', -99) != axis:
            raise ValueError('fixed query/CLS/softmax axis mismatch')
    for index in (10,30,35,40,57):
        attrs = reader.attrs(reader.nodes[index], {'axes'})
        if reader.ints_attr(attrs, 'axes', ()) != (0,):
            raise ValueError('fixed shape unsqueeze mismatch')
    expected_ops = {3:'Conv',8:'Reshape',14:'ConstantOfShape',15:'Add',17:'Add',
        19:'BatchNormalization',24:'MatMul',25:'Add',26:'MatMul',27:'Add',
        28:'MatMul',29:'Add',45:'Reshape',47:'Reshape',48:'Reshape',
        51:'MatMul',52:'Constant',53:'Mul',54:'Softmax',55:'MatMul',
        61:'Reshape',62:'MatMul',63:'Add',64:'Add',66:'BatchNormalization',
        68:'MatMul',69:'Add',70:'Relu'}
    if any(reader.nodes[i].op_type != op for i, op in expected_ops.items()):
        raise ValueError('unsupported fixed-query first block')
    # Full model + inventory authentication fixes all these exact source ports;
    # this independently rejects an accidentally altered parser orientation.
    for dst, sources in {51:(46,50),53:(51,52),55:(54,49),64:(63,17)}.items():
        if tuple(reader.nodes[dst].input) != tuple(reader.nodes[i].output[0] for i in sources):
            raise ValueError('query/key/value or residual orientation mismatch')
    if tuple(reader.nodes[69].input).count(reader.nodes[68].output[0]) != 1:
        raise ValueError('MLP preactivation port mismatch')
    return 1 + (32 // patch)**2


def _matrix(reader, index, shape, budget):
    node = reader.nodes[index]
    if node.op_type != 'MatMul' or len(node.input) != 2:
        raise ValueError('right-hand exact MatMul weight required')
    value = reader.needed(node.input[1], 1)
    budget.charge(8 + 3 * len(value['values']))
    if value['shape'] != shape:
        raise ValueError('MatMul [input, output] initializer shape mismatch')
    return value['values']


def _bias(reader, index, previous, width, budget):
    node = reader.nodes[index]
    if node.op_type != 'Add' or len(node.input) != 2 or node.input.count(previous) != 1:
        raise ValueError('exact affine bias port mismatch')
    value = reader.needed(node.input[1] if node.input[0] == previous else node.input[0], 1)
    budget.charge(8 + 3 * len(value['values']))
    if value['shape'] != (width,):
        raise ValueError('affine bias shape mismatch')
    return value['values']


def _payload(reader, patch, tokens, budget):
    # Exact authenticated needed FLOAT population, prepaid before decoding.
    expected_scalars = 48*(3*patch*patch) + 14594 + 48*tokens
    budget.charge(4096 + 8*expected_scalars)
    conv = reader.conv(3, 'input', (1,3,32,32), 1)
    if (conv['weight_shape'] != (48,3,patch,patch) or conv['strides'] != (patch,patch)
            or conv['pads'] != (0,0,0,0) or conv['dilations'] != (1,1)
            or conv['group'] != 1 or conv['output_shape'] != (1,48,32//patch,32//patch)):
        raise ValueError('non-disjoint patch convolution')
    attrs = reader.attrs(reader.nodes[14], {'value'})
    if set(attrs) != {'value'} or attrs['value'].type != reader.onnx.AttributeProto.TENSOR:
        raise ValueError('explicit CLS fill tensor required')
    fill = reader.tensor(attrs['value'].t)
    cls = reader.needed('0.cls_token', 1)
    pos = reader.needed('0.positions', 1)
    if (fill['shape'] != (1,) or fill['dtype'] != 1 or cls['shape'] != (48,)
            or pos['shape'] != (tokens,48)):
        raise ValueError('CLS/position shape mismatch')
    bn1 = reader.bn(19, reader.nodes[18].output[0], (1,48,tokens), 1)
    bn2 = reader.bn(66, reader.nodes[65].output[0], (1,48,tokens), 1)
    aff1 = tuple(helper.post_affine((bn1,), i, k, budget) for i in range(48))
    aff2 = tuple(helper.post_affine((bn2,), i, k, budget) for i in range(48))
    scale = reader.needed(reader.nodes[52].output[0], 1)
    if scale['shape'] != () or len(scale['values']) != 1:
        raise ValueError('scalar attention scaling required')
    result = dict(width=48, heads=3, head_width=16, outputs=96, d=3*patch*patch,
        tokens=tokens, patch_weights=conv['weights'], patch_bias=conv['bias'],
        cls=cls['values'], fill=fill['values'][0], positions=pos['values'],
        bn1=aff1, bn2=aff2, scale=scale['values'][0])
    for key, index, width in (('query',24,48),('key',26,48),('value',28,48),('out',62,48),('mlp',68,96)):
        result[key] = _matrix(reader, index, (48,width), budget)
        result[key+'_bias'] = _bias(reader, index+1, reader.nodes[index].output[0], width, budget)
    if reader.decoded_scalars != expected_scalars:
        raise ValueError('needed FLOAT scalar population differs')
    return result


def _fold(p, budget):
    """Small generic algebra kernel; real bind separately fixes 48/3/16/96.

    Every weight is validated once. Large dots use exact scalar weights where
    possible. Retained interval coefficients round outward to a dyadic64 grid.
    """
    w, heads, r, out, d, n = (p[key] for key in ('width','heads','head_width','outputs','d','tokens'))
    if (any(type(v) is not int or v <= 0 for v in (w,heads,r,out,d,n))
            or w != heads*r or w > 48 or heads > 3 or out > 96 or d > 768 or n > 17):
        raise ValueError('bounded factored coefficient dimensions required')
    lengths = dict(patch_weights=w*d, patch_bias=w, cls=w, positions=n*w,
        query=w*w,key=w*w,value=w*w,out=w*w,mlp=w*out,
        query_bias=w,key_bias=w,value_bias=w,out_bias=w,mlp_bias=out)
    for key, length in lengths.items():
        values = p[key]
        if type(values) is not tuple or len(values) != length:
            raise ValueError('exact coefficient payload shape mismatch')
        budget.charge(3*length)
        for value in values:
            _bits(value, budget)
    budget.charge(6)
    _bits(p['fill'], budget); _bits(p['scale'], budget)
    for key in ('bn1','bn2'):
        if type(p[key]) is not tuple or len(p[key]) != w:
            raise ValueError('BN affine shape mismatch')
        for item in p[key]:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError('BN affine tuple required')
            _iv(item[0], budget); _iv(item[1], budget)
    # Token-zero raw residual is constant, including ConstantOfShape fill.
    cls = []
    for a in range(w):
        budget.charge(8)
        cls.append(_bits(_bits(p['fill']+p['cls'][a],budget)+p['positions'][a],budget))
    cls = tuple(cls)
    hidden = []
    for token in range(n):
        row = []
        for a in range(w):
            if token == 0:
                value = cls[a]
            else:
                budget.charge(4)
                value = _bits(p['patch_bias'][a]+p['positions'][token*w+a],budget)
            row.append(_padd(p['bn1'][a][1], value, p['bn1'][a][0], budget))
        hidden.append(tuple(row))
    hidden = tuple(hidden)
    q = []
    for b in range(w):
        acc = _point(p['query_bias'][b],budget)
        for a in range(w):
            acc = _padd(acc,p['query'][a*w+b],hidden[0][a],budget)
        q.append(_round(acc,budget))
    score_hidden, score_offset = [], []
    for h in range(heads):
        row = []
        for a in range(w):
            acc = Z
            for b in range(r):
                acc = _padd(acc,p['key'][a*w+h*r+b],q[h*r+b],budget)
            row.append(_round(_padd(Z,p['scale'],acc,budget),budget))
        acc = Z
        for b in range(r):
            acc = _padd(acc,p['key_bias'][h*r+b],q[h*r+b],budget)
        score_hidden.append(tuple(row))
        score_offset.append(_round(_padd(Z,p['scale'],acc,budget),budget))
    # Fold BN into 51 channel readouts first, then dot exact Conv FLOAT weights.
    transfer = tuple(score_hidden) + tuple(tuple((p['value'][a*w+b],)*2 for a in range(w)) for b in range(w))
    coefficients, biases = [], []
    for row_index, row in enumerate(transfer):
        scaled = tuple(_round(_iadd(Z,row[a],p['bn1'][a][0],budget),budget) for a in range(w))
        coeff = []
        for pixel in range(d):
            acc = Z
            for a in range(w):
                acc = _padd(acc,p['patch_weights'][a*d+pixel],scaled[a],budget)
            coeff.append(_round(acc,budget))
        token_bias = []
        for token in range(n):
            acc = score_offset[row_index] if row_index < heads else _point(p['value_bias'][row_index-heads],budget)
            for a in range(w):
                acc = _iadd(acc,row[a],hidden[token][a],budget)
            token_bias.append(_round(acc,budget))
        coefficients.append(tuple(coeff)); biases.append(tuple(token_bias))
    # Signed output projection is pushed into head values BEFORE attention.
    directions, constants = [], []
    for j in range(out):
        post = tuple(_padd(Z,p['mlp'][a*out+j],p['bn2'][a][0],budget) for a in range(w))
        flat = []
        for b in range(w):
            acc = Z
            for a in range(w):
                acc = _padd(acc,p['out'][b*w+a],post[a],budget)
            flat.append(_round(acc,budget))
        directions.append(tuple(tuple(flat[h*r:(h+1)*r]) for h in range(heads)))
        acc = _point(p['mlp_bias'][j],budget)
        for a in range(w):
            budget.charge(4)
            resid = _bits(p['out_bias'][a]+cls[a],budget)
            bnval = _padd(p['bn2'][a][1],resid,p['bn2'][a][0],budget)
            acc = _padd(acc,p['mlp'][a*out+j],bnval,budget)
        constants.append(_round(acc,budget))
    return dict(score_coeff=tuple(coefficients[:heads]), value_coeff=tuple(coefficients[heads:]),
        score_bias=tuple(biases[:heads]), value_bias=tuple(biases[heads:]),
        directions=tuple(directions), constants=tuple(constants),
        width=w, heads=heads, head_width=r, outputs=out, d=d, tokens=n)


def _immutable(value, budget):
    budget.charge(1)
    if type(value) is tuple:
        return 1 + sum(_immutable(v,budget) for v in value)
    if type(value) is F:
        budget.charge(2); _bits(value,budget)
        return 3
    if type(value) in (str,int,bool) or value is None:
        return 1
    raise ValueError('prepared bank must be deeply immutable')


def _seal(bank, budget, *, start=None):
    global _ACTIVE
    # Cached score centers/errors are direction-independent. Plain tuples only;
    # no Form objects or independently realizable native output are retained.
    cache=[]
    for h in range(bank['heads']):
        centered=tuple(_center(v,budget) for v in bank['score_coeff'][h])
        tokens=[]
        for token,columns in enumerate(bank['patch_columns']):
            form,error=_centered_form(centered if token else (),
                bank['score_bias'][h][token],columns,bank['input_box'],budget)
            tokens.append((form.bias,form.terms,error))
        cache.append(tuple(tokens))
    bank['score_query_data']=tuple(cache)
    entries = 1 + sum(1+_immutable(v,budget) for v in bank.values())
    if entries > MAX_ENTRIES:
        raise ValueError('retained coefficient bank entry cap')
    bank['entries'] = entries + 4
    bank['entry_upper'] = entries + 4096 + 128*bank['tokens']*bank['d']
    if bank['entry_upper'] > MAX_ENTRIES:
        raise ValueError('binding/transient entry cap')
    budget.charge(8*len(bank))
    if start is not None:
        bank['work']=budget.used-start
    _ACTIVE = (bank, tuple(bank.items()))
    return bank


def release(binding):
    """Drop the sole private retained reference, without changing the bank."""
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE[0] is binding:
        _ACTIVE = None


def _prepared(binding, budget):
    budget.charge(64)
    if type(binding) is not dict or _ACTIVE is None or _ACTIVE[0] is not binding:
        raise ValueError('live authenticated prepared binding required')
    original = _ACTIVE[1]
    if len(binding) != len(original) or any(binding.get(key) is not value for key,value in original):
        raise ValueError('prepared coefficient/source identity changed')


def bind(raw_model, raw_spec, inventory, *, model_sha256, spec_sha256,
         enabled=False, budget=None):
    if enabled is False:
        return None
    if enabled is not True or not isinstance(budget,k.WorkBudget) or not budget.enabled:
        raise ValueError('strict opt-in and shared enabled budget required')
    start = budget.used
    expected, graph = _identity(raw_model,raw_spec,inventory,model_sha256,spec_sha256,budget)
    import onnx
    model = onnx.ModelProto()
    model.ParseFromString(raw_model)
    if model.functions or model.training_info or model.graph.sparse_initializer:
        raise ValueError('functions/training/sparse initializers unsupported')
    reader = base._Reader(onnx,model)
    n = _layout(reader,graph,expected[1],budget)
    box = helper.input_box(raw_spec,(1,3,32,32))
    payload = _payload(reader,expected[1],n,budget)
    bank = _fold(payload,budget)
    bank.update(input_box=tuple(box[i] for i in range(3072)),
        patch_columns=_patch_columns(expected[1],budget),
        model=expected[0], model_sha256=model_sha256, spec_sha256=spec_sha256,
        graph_sha256=expected[3], original_pixel_count=3072,
        frame=12801 if expected[1] == 16 else 12802,
        scope='first_cls_node69_real_arithmetic_coefficient_outer',
        native_qualified=False, gpu_qualified=False, exact_original_model=False,
        decoded_scalars=reader.decoded_scalars, work=budget.used-start)
    return _seal(bank,budget,start=start)


def _center(interval, budget):
    budget.charge(12)
    total = _bits(interval[0]+interval[1],budget)
    width = _bits(interval[1]-interval[0],budget)
    return _bits(total/2,budget), _bits(width/2,budget)


def _centered_form(centered, bias, columns, box, budget):
    center, error = _center(bias,budget)
    terms = []
    if len(centered) != len(columns):
        raise ValueError('source coefficient/column dimension mismatch')
    for (a,radius),index in zip(centered,columns):
        budget.charge(12)
        bound = max(abs(box[index][0]),abs(box[index][1]))
        term = _bits(radius*bound,budget)
        error = _bits(error+term,budget)
        if a:
            terms.append((index,a))
    # Original NCHW patch IDs are strictly sorted, without duplicate columns.
    # Form's dict insertion and sort of this ordered run are linear operations.
    budget.charge(32 + 16*len(terms))
    return ef.Form(center,tuple(terms)),error


def _form(coefficients, bias, columns, box, budget):
    return _centered_form(tuple(_center(v,budget) for v in coefficients),
                          bias,columns,box,budget)


def direction(binding, index, *, budget, sign=1):
    """One signed CLS successor direction; all forms use original pixel IDs.

    Worker may query both signs from this same returned positive direction by
    reflecting value forms and constants; error radii do not change.
    """
    if not isinstance(budget,k.WorkBudget) or not budget.enabled:
        raise ValueError('shared enabled budget required')
    start = budget.used
    _prepared(binding,budget)
    if (type(index) is not int or not 0 <= index < binding['outputs']
            or type(sign) is not int or sign not in (-1,1)):
        raise ValueError('direction index/sign outside complete population')
    n,d,r = binding['tokens'],binding['d'],binding['head_width']
    box,columns = binding['input_box'],binding['patch_columns']
    heads = []
    for h in range(binding['heads']):
        weights = binding['directions'][index][h]
        coeff = []
        for pixel in range(d):
            acc = Z
            for b in range(r):
                acc = _iadd(acc,weights[b],binding['value_coeff'][h*r+b][pixel],budget)
            coeff.append(_round(acc,budget))
        coeff = tuple(coeff)
        centered=tuple(_center(v,budget) for v in coeff)
        tokens = []
        for token in range(n):
            acc = Z
            for b in range(r):
                acc = _iadd(acc,weights[b],binding['value_bias'][h*r+b][token],budget)
            acc = _round(acc,budget)
            scol = binding['score_coeff'][h] if token else ()
            vcol = coeff if token else ()
            sbias,sterms,se=binding['score_query_data'][h][token]
            budget.charge(32+16*len(sterms))
            score=ef.Form(sbias,sterms)
            value,ve = _centered_form(centered if token else (),acc,columns[token],box,budget)
            if sign < 0:
                budget.charge(32+20*len(value.terms))
                value = -value
            tokens.append(dict(score=score,value=value,score_error=se,value_error=ve,
                columns=columns[token], score_coefficients=scol, value_coefficients=vcol))
        heads.append(dict(head=h,tokens=tuple(tokens)))
    constant = binding['constants'][index]
    if sign < 0:
        budget.charge(8)
        constant = (-constant[1],-constant[0])
    return dict(index=index,sign=sign,constant=constant,heads=tuple(heads),
        work=budget.used-start,original_pixel_count=binding['original_pixel_count'],
        model_sha256=binding['model_sha256'],spec_sha256=binding['spec_sha256'])
