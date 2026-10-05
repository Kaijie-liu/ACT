"""Ordinary shared fields, circuit UID retags and nonzero inverse with v2."""
from fractions import Fraction as F
import pytest
from experiments.neural_hz_20260831.test_c99_circuit_native_v1 import fixture,pool
from experiments.neural_hz_20260831.c99_circuit_consumer_v2 import CircuitSource,inject_phase,bind_phase,source_binding_work
from experiments.neural_hz_20260831.c99_append_discovery_v2 import discover_append
from experiments.neural_hz_20260831.c99_circuit_journal_v2 import compile_journal,reconstruct
from experiments.neural_hz_20260831.c99_native_proof_v2 import verify
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import extend


@pytest.mark.parametrize('value',[F(-1,4),F(1,8),F(3,8)])
def test_once_published_complete_fields_and_actual_native_circuit(value):
    previous,packet=fixture();p=pool()
    p.charge('c99_once_source_field_publication',source_binding_work(previous.state))
    c=CircuitSource(previous.state)
    assert all(vars(c)[k] is v for k,v in c.state['fields'].items())
    image=inject_phase(c,packet,pool=p,enabled=True)
    view=bind_phase(c,packet,image,pool=p,enabled=True);first=packet['first_uid']
    overlay,_=build(c.owners,[(view.eq_c,first),(view.le_c,first+len(view.eq_rhs))],
        old_n_cont=c.old_n_cont,old_uid_ceiling=first,pool=p,enabled=True)
    plans,_=discover_append(c,view,overlay,pool=p,enabled=True)
    new,_=splice_append(view,plans,pool=p,enabled=True)
    journal=compile_journal(c,plans,pool=p,enabled=True)
    assert verify(c,view,overlay,plans,new,journal,pool=p)['all_circuit_incidence_equal']
    original=[value,F(1,8)+value/4]
    point=extend(c.state,original,pool=p)+[F(0),F(0)];point[1]=F(7,8)
    result=reconstruct(c,new,journal,plans,point,pool=p,enabled=True)
    assert result['original_point']==original
    assert result['proof']['circuit_equations']==1
