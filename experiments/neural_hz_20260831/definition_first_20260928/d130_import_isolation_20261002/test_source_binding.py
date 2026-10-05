"""Six small synthetic controls; no real model/property files are read."""
from fractions import Fraction as F
import struct
import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d130_import_isolation_20261002 import source_binding as sb


def _budget(limit=2_000_000):
    return sb.k.WorkBudget(enabled=True,limit=limit)


def _payload(outputs=2):
    z,o=F(0),F(1)
    eye=(o,z,z,o)
    aff=(((o,o),(z,z)),)*2
    return dict(width=2,heads=1,head_width=2,outputs=outputs,d=2,tokens=3,
        patch_weights=eye,patch_bias=(F(1,4),F(-1,2)),
        cls=(F(1,2),F(-1,4)),fill=F(1,4),
        positions=(z,z,F(1,4),z,z,F(1,4)),bn1=aff,bn2=aff,scale=F(1,2),
        query=eye,key=eye,value=eye,out=eye,
        query_bias=(z,z),key_bias=(F(1,4),z),value_bias=(z,F(1,8)),
        out_bias=(F(1,8),F(-1,8)),
        mlp=tuple((o if j%2==a else z) for a in range(2) for j in range(outputs)),
        mlp_bias=tuple(F(j,16) for j in range(outputs)))


def _bank(outputs=2):
    b=_budget()
    bank=sb._fold(_payload(outputs),b)
    bank.update(input_box=((F(-2),F(1)),(F(-1),F(3)),(F(-1),F(1)),(F(0),F(2))),
        patch_columns=((),(0,1),(2,3)),original_pixel_count=4,
        model_sha256='synthetic',spec_sha256='synthetic',frame=12800)
    return sb._seal(bank,b),b


def test_binding_exact_float_payload():
    import onnx
    model=onnx.ModelProto()
    model.opset_import.add(domain='',version=9)
    reader=sb.base._Reader(onnx,model)
    tensor=onnx.TensorProto()
    tensor.data_type=onnx.TensorProto.FLOAT
    tensor.dims.extend((3,))
    tensor.raw_data=struct.pack('<fff',0.1,-0.25,1.5)
    values=reader.tensor(tensor)
    assert values['shape']==(3,) and values['dtype']==1
    assert values['values']==(F(13421773,134217728),F(-1,4),F(3,2))
    tensor.raw_data=struct.pack('<fff',float('inf'),0.0,0.0)
    with pytest.raises(ValueError):
        reader.tensor(tensor)


def test_binding_layout_and_identity():
    b=_budget()
    for patch in (8,16):
        cols=sb._patch_columns(patch,b)
        assert cols[0]==() and len(cols)==1+(32//patch)**2
        assert tuple(sorted(i for row in cols for i in row))==tuple(range(3072))
        assert cols[1][0]==0 and cols[1][patch]==32 and cols[1][patch*patch]==1024
        assert all(tuple(sorted(row))==row for row in cols)
    with pytest.raises(ValueError):
        sb._patch_columns(4,b)
    identity=next(iter(sb.SOURCES))
    with pytest.raises(ValueError):
        sb._identity(b'not-model',b'not-spec',b'{}',identity,sb.SOURCES[identity][2],b)
    assert sb.bind(None,None,None,model_sha256=None,spec_sha256=None) is None
    # Actual protobuf containers, not Python-list mocks: D128 failed here.
    import onnx
    model=onnx.ModelProto()
    model.opset_import.add(domain='',version=9)
    bias=model.graph.initializer.add()
    bias.name='bias'
    bias.data_type=onnx.TensorProto.FLOAT
    bias.dims.extend((2,))
    bias.float_data.extend((0.25,-0.5))
    for i,ports in enumerate((('x','bias'),('bias','x'),('x','x'),('other','bias'))):
        node=model.graph.node.add()
        node.op_type='Add'
        node.input.extend(ports)
        node.output.extend(('out_'+str(i),))
    parsed=onnx.ModelProto()
    parsed.ParseFromString(model.SerializeToString())
    reader=sb.base._Reader(onnx,parsed)
    assert sb._bias(reader,0,'x',2,b)==(F(1,4),F(-1,2))
    assert sb._bias(reader,1,'x',2,b)==(F(1,4),F(-1,2))
    for i in (2,3):
        with pytest.raises(ValueError):
            sb._bias(reader,i,'x',2,b)
    with pytest.raises(ValueError):
        sb._bias(reader,0,'x',3,b)


def test_binding_batchnorm_intervals():
    b=_budget()
    intervals=((F(-3),F(-1)),(F(-2),F(3)),(F(0),F(0)),(F(1),F(4)))
    for left in intervals:
        for right in intervals:
            products=tuple(x*y for x in left for y in right)
            assert sb._iadd((F(1),F(2)),left,right,b)==(1+min(products),2+max(products))
    raw=dict(kind='batchnorm',gamma=(F(3),),beta=(F(1,4),),
        mean=(F(1,2),),variance=(F(2),),epsilon=F(0))
    a,c=sb.helper.post_affine((raw,),0,sb.k,b)
    # a=3/sqrt(2), c=1/4-a/2, checked without a float sqrt.
    assert a[0]>0 and 2*a[0]*a[0]<=9<=2*a[1]*a[1]
    assert c[0]<=F(1,4)-a[1]/2<=F(1,4)-a[0]/2<=c[1]
    raw['variance']=(F(-1),)
    with pytest.raises(ValueError):
        sb.helper.post_affine((raw,),0,sb.k,b)
    # General nonsymmetric box pays abs-max rather than a symmetric shortcut.
    form,error=sb._form(((F(1),F(3)),),(F(-1),F(1)),(0,),((F(-2),F(5)),),b)
    assert form.bias==0 and form.terms==((0,F(2)),) and error==6


def test_binding_shared_patch_templates():
    bank,b=_bank()
    try:
        assert bank['score_coeff']==(((F(3,8),F(3,8)),(F(0),F(0))),)
        # CLS has q=(3/4,0), scale=1/2, key bias=(1/4,0).
        assert bank['score_bias'][0]==((F(3,8),F(3,8)),(F(9,32),F(9,32)),(F(3,16),F(3,16)))
        result=sb.direction(bank,0,budget=b)
        tokens=result['heads'][0]['tokens']
        assert tokens[0]['score'].terms==tokens[0]['value'].terms==()
        assert tokens[1]['value_coefficients'] is tokens[2]['value_coefficients']
        assert tokens[1]['score_coefficients'] is tokens[2]['score_coefficients']
        assert tokens[1]['score'].terms==((0,F(3,8)),)
        assert tokens[2]['score'].terms==((2,F(3,8)),)
        assert tokens[1]['value'].bias==F(1,2)
        assert all(t['score_error']==t['value_error']==0 for t in tokens)
    finally:
        sb.release(bank)


def test_binding_all_cls_directions():
    bank,b=_bank(96)
    try:
        assert len(bank['directions'])==len(bank['constants'])==96
        # [input, output] orientation selects alternating input coordinates.
        for j in range(96):
            result=sb.direction(bank,j,budget=b)
            expected=F(7,8) if j%2==0 else F(-1,8)
            assert result['constant']==(expected+F(j,16),)*2
            assert bank['directions'][j][0][j%2]==(F(1),F(1))
            assert bank['directions'][j][0][1-j%2]==(F(0),F(0))
        plus=sb.direction(bank,1,budget=b)
        minus=sb.direction(bank,1,budget=b,sign=-1)
        assert minus['constant']==(-plus['constant'][1],-plus['constant'][0])
        for p,m in zip(plus['heads'][0]['tokens'],minus['heads'][0]['tokens']):
            assert m['score']==p['score'] and m['value']==-p['value']
            assert m['value_error']==p['value_error']
        assert bank['entry_upper']>=bank['entries'] and b.used>0
    finally:
        sb.release(bank)


def test_binding_fail_closed():
    with pytest.raises(ValueError):
        sb.bind(None,None,None,model_sha256=None,spec_sha256=None,enabled=1,budget=_budget())
    with pytest.raises(sb.k.BudgetExceeded):
        sb._fold(_payload(),_budget(1))
    bad=_payload(); bad['query']=(F(1<<513),)*4
    with pytest.raises(ValueError):
        sb._fold(bad,_budget())
    bank,b=_bank()
    original=bank['input_box']
    try:
        bank['input_box']=((F(0),F(0)),)*4
        with pytest.raises(ValueError):
            sb.direction(bank,0,budget=b)
        bank['input_box']=original
        with pytest.raises(ValueError):
            sb.direction(bank,True,budget=b)
        with pytest.raises(ValueError):
            sb.direction(bank,96,budget=b)
        bank['unexpected']=1
        with pytest.raises(ValueError):
            sb.direction(bank,0,budget=b)
        del bank['unexpected']
        sb.release(bank)
        with pytest.raises(ValueError):
            sb.direction(bank,0,budget=b)
    finally:
        sb.release(bank)
