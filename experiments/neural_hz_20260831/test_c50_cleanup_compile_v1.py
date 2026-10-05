import ast
import hashlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import pytest
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import guard,canaries,compiled
from experiments.neural_hz_20260831.c50_cleanup_compile_v1 import compile_cleanup,exact_canaries
from experiments.neural_hz_20260831.c50_cleanup_probe_v1 import probe
from experiments.neural_hz_20260831.c50_cleanup_runner_v1 import module_audit


def test_every_new_entry_is_default_off():
    assert compile_cleanup(object(),object()) is None
    assert exact_canaries() is None and probe() is None


def test_plain_mode_is_rejected_for_changed_passing_and_failing_evaluation_traces():
    with pytest.raises(ValueError,match='side effects/short circuits'):
        canaries(enabled=True)


def test_original_evaluation_messages_short_circuits_walrus_and_owner_release_all_match():
    result=exact_canaries(enabled=True)
    assert result['cases']==17 and result['all_outcomes_messages_and_side_effects_equal']
    assert result['corpus'][1]['identical_rows'][0]['trace']==['equal','release:left','release:right','after']
    assert result['corpus'][0]['compile']['possibly_unassigned_targets_still_cleared']>0
    assert all(c['compile']['cleanup_code_bytes']<c['compile']['original_rewritten_code_bytes'] for c in result['corpus'])


@pytest.mark.parametrize('kind',['branch','loop','try','with','nested'])
def test_cleanup_never_deletes_an_unassigned_short_circuit_or_outer_scope_temp(kind):
    bodies={
        'branch':"if flag:\n    assert x == 0 or x < 5\nelse:\n    assert x >= 0\nassert x == 0 or x <= 8",
        'loop':"for j in range(3):\n    if j % 2:\n        assert x == 0 or x <= 8\n    else:\n        assert x >= 0\nassert x >= 0",
        'try':"try:\n    assert x == 0 or x < 8\nexcept AssertionError:\n    assert x == 8\nfinally:\n    assert x >= 0",
        'with':"with cm():\n    assert x == 0 or x <= 8\nassert x >= 0",
        'nested':"def nested(z):\n    assert z == 0 or z <= 8\n    return z\nassert nested(x) >= 0",
    }
    body=bodies[kind];source=('from contextlib import nullcontext as cm\ndef run(x,flag):\n'+''.join('    '+l+'\n' for l in body.splitlines())+'    return x\n').encode()
    old={};new={};exec(compiled(source,'<c50-control>',rewritten=True),old)
    code,stats=compile_cleanup(source,'<c50-control>',enabled=True);exec(code,new)
    for x in (0,1,4,8):
        for flag in (False,True):
            outcomes=[]
            for ns in (old,new):
                try:value=ns['run'](x,flag);outcomes.append(('returned',value))
                except Exception as exc:outcomes.append(('raised',type(exc).__name__))
            assert outcomes[0]==outcomes[1]


def test_class_and_module_scratch_names_are_not_optimized_as_function_locals():
    source=b'assert 2 > 1\nclass C:\n    assert 3 > 2\n'
    code,stats=compile_cleanup(source,'<c50-nonlocal>',enabled=True)
    assert stats['changed_cleanup_assignments']==0
    assert stats['cleanup_code_bytes']==stats['original_rewritten_code_bytes']


def test_python_optimization_is_rejected_without_relying_on_an_assert(monkeypatch):
    monkeypatch.setattr(sys,'flags',SimpleNamespace(optimize=1))
    with pytest.raises(ValueError,match='optimization is forbidden'):guard()


def loaded_fixture(mode='cleanup'):
    path=Path(__file__).with_name('c50_cleanup_fixture_v1.py').resolve();source=path.read_bytes()
    module=types.ModuleType('c50_synthetic_loaded_test');module.__file__=str(path)
    if mode=='cleanup':code=compile_cleanup(source,str(path),enabled=True)[0]
    elif mode=='removed_assert':code=compile(source,str(path),'exec',dont_inherit=True,optimize=2)
    else:code=compiled(source,str(path),rewritten=mode=='old_rewrite')
    exec(code,vars(module))
    return module,path,hashlib.sha256(source).hexdigest()


def test_loaded_test_and_helper_code_are_both_independently_bound():
    module,path,sha=loaded_fixture()
    proof=module_audit(module,path,sha,[SimpleNamespace(obj=module.test_scalar)],None)
    assert proof['all_loaded_test_and_helper_code_checked'] and proof['functions']==2
    assert proof['compile']['source_assert_statements']==2


@pytest.mark.parametrize('bad',['hash','plain','old_rewrite','file','removed_assert'])
def test_source_hash_or_altered_loaded_assertions_cannot_pass_by_name(bad):
    module,path,sha=loaded_fixture(bad if bad in ('plain','old_rewrite','removed_assert') else 'cleanup')
    if bad=='hash':sha='0'*64
    elif bad=='file':module.__file__=str(path.with_name('not_the_frozen_test.py'))
    with pytest.raises(ValueError):module_audit(module,path,sha,[SimpleNamespace(obj=module.test_scalar)],None)
