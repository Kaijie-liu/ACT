from fractions import Fraction as F
import pytest
from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import attention as at
from experiments.neural_hz_20260831.definition_first_20260928.d128_bounded_attention_source_20261002 import source_worker as sw


def test_bridge_error_rectangle():
    b=at.ei.Budget()
    box=((F(-1),F(1)),)
    score=at.Form(F(0),((0,F(1)),))
    value=at.Form(F(1),((0,F(-1)),))
    polygon,entries=sw.bridge_polygon(box,score,value,F(1,4),F(1,2),b)
    assert min(s for s,v in polygon)==F(-5,4)
    assert max(s for s,v in polygon)==F(5,4)
    assert min(v for s,v in polygon)==F(-1,2)
    assert max(v for s,v in polygon)==F(5,2)
    # Every joint point still satisfies this shared-source slab.
    assert min(s+v for s,v in polygon)==F(1,4)
    assert max(s+v for s,v in polygon)==F(7,4)
    assert entries>0 and b.work>0


def test_bridge_original_identity():
    box=((F(-1),F(1)),(F(0),F(2)))
    snapshot=box
    s=at.Form(F(0),((0,F(1)),))
    v=at.Form(F(0),((1,F(1)),))
    original=(s,v)
    sw.bridge_polygon(box,s,v,F(0),F(0),at.ei.Budget())
    assert box is snapshot
    assert s is original[0] and v is original[1]
    assert all(i<2 for form in original for i,c in form.terms)


def test_bridge_signed_composition():
    b=at.ei.Budget()
    f=at.Form(F(2),((0,F(3)),(1,F(-4))))
    neg=sw.negative_form(f,b)
    assert neg.bias==F(-2) and neg.terms==((0,F(-3)),(1,F(4)))
    result=sw.composed_bounds((F(1),F(2)),
        ({'hi':F(3),'rectangle_hi':F(4)},),
        ({'hi':F(1),'rectangle_hi':F(2)},),b)
    assert result['interval']==(F(0),F(5))
    assert result['rectangle_interval']==(F(-1),F(6))
    assert result['lower_improved'] and result['upper_improved']
    assert not result['network_certified'] and not result['validated_adv']


def test_bridge_no_root_lower_as_witness():
    # These lo fields are deliberately irrelevant: only negative-support hi
    # can give a universal lower bound, not a product-root lower endpoint.
    b=at.ei.Budget()
    result=sw.composed_bounds((F(0),F(0)),
        ({'lo':F(90),'hi':F(2),'rectangle_hi':F(3)},),
        ({'lo':F(80),'hi':F(1),'rectangle_hi':F(2)},),b)
    assert result['interval']==(F(-1),F(2))
    assert result['relu_interval']==(F(0),F(2))


def test_bridge_budget_and_errors():
    box=((F(-1),F(1)),)
    x=at.Form(F(0),((0,F(1)),))
    for error in (F(-1),0,F(1<<513)):
        with pytest.raises(ValueError):
            sw.bridge_polygon(box,x,x,error,F(0),at.ei.Budget())
    with pytest.raises(ValueError):
        sw.bridge_polygon({1:box[0]},x,x,F(0),F(0),at.ei.Budget())
    with pytest.raises(ValueError):
        sw.bridge_polygon(box,x,x,F(0),F(0),at.ei.Budget(1))
    with pytest.raises(ValueError):
        sw.composed_bounds((F(2),F(1)),(),(),at.ei.Budget())


def test_bridge_shared_budget():
    box=((F(-1),F(1)),)
    x=at.Form(F(0),((0,F(1)),))
    b=at.ei.Budget()
    first=sw.bridge_polygon(box,x,-x,F(0),F(0),b)[0]
    used=b.work
    second=sw.bridge_polygon(box,x,-x,F(0),F(0),b)[0]
    assert first==second and b.work==2*used
    b.max_work=b.work
    with pytest.raises(ValueError):
        sw.bridge_polygon(box,x,-x,F(0),F(0),b)
