import hashlib
from pathlib import Path
import sys
import types
import pytest
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import compiled
from experiments.neural_hz_20260831.c50_cleanup_compile_v1 import compile_cleanup
from experiments.neural_hz_20260831.c51_source_custody_v1 import custody,function,Receipt

ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent


def loaded(monkeypatch):
    collection=EXP/'c51_origin_collection_fixture_v1.py';definition=EXP/'c51_origin_definition_fixture_v1.py'
    sources={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (collection,definition)}
    reg=custody([collection],sources,ROOT,enabled=True);modules={}
    for path in (definition,collection):
        name='experiments.neural_hz_20260831.'+path.stem
        module=types.ModuleType(name);module.__file__=str(path)
        monkeypatch.setitem(sys.modules,name,module)
        exec(reg.prepare(path),vars(module));modules[path]=module
    return reg,collection,definition,modules[collection],modules[definition]


def test_custody_is_default_off():
    assert custody(object(),object(),object()) is None


def test_imported_test_and_both_complete_modules_are_bound_without_recompilation(monkeypatch):
    reg,c,d,cm,dm=loaded(monkeypatch)
    for path,module in ((c,cm),(d,dm)):
        report=reg.audit_module(module,path)
        assert report['all_loaded_test_and_helper_code_checked']
        assert report['collection_recompilations']==0
    path,module=reg.definition(cm,c,cm.test_origin,'test_origin')
    assert path==d and module is dm and reg.compile_count==2
    assert cm.test_origin(1)==1
    with pytest.raises(AssertionError):cm.test_origin(0)
    assert reg.specs[c][1]=='cleanup' and reg.specs[d][1]=='rewrite'


@pytest.mark.parametrize('kind',['plain','removed','cleanup_instead_of_rewrite','helper','filename','globals','wrapper','origin','alias','receipt','policy','images'])
def test_origin_or_assertion_substitutions_fail_closed(monkeypatch,kind):
    reg,c,d,cm,dm=loaded(monkeypatch)
    if kind in ('plain','removed','cleanup_instead_of_rewrite','helper'):
        source=d.read_bytes();namespace={'__name__':dm.__name__}
        if kind=='removed':code=compile(source,str(d),'exec',dont_inherit=True,optimize=2)
        elif kind=='cleanup_instead_of_rewrite':code=compile_cleanup(source,str(d),enabled=True)[0]
        else:code=compiled(source,str(d))
        exec(code,namespace);name='helper' if kind=='helper' else 'test_origin'
        getattr(dm,name).__code__=namespace[name].__code__
    elif kind=='filename':dm.__file__=str(c)
    elif kind=='globals':
        fn=types.FunctionType(dm.test_origin.__code__,dict(vars(dm)));dm.test_origin=fn;cm.test_origin=fn
    elif kind=='wrapper':
        def wrapped(value):return 1
        wrapped.__wrapped__=dm.test_origin;cm.test_origin=wrapped
    elif kind=='origin':dm.test_origin.__module__='unregistered.definition'
    elif kind=='alias':cm.test_origin=cm.local_helper
    elif kind=='receipt':reg._receipts.pop(d)
    elif kind=='policy':reg._receipts[d]=reg._receipts[d]._replace(policy='cleanup')
    elif kind=='images':reg._images[d]={}
    with pytest.raises(ValueError):
        reg.audit_module(dm,d);reg.definition(cm,c,cm.test_origin,'test_origin')


def test_source_change_after_import_is_rejected(monkeypatch):
    reg,c,d,cm,dm=loaded(monkeypatch);read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:read(p)+b'\n# changed' if p==d else read(p))
    with pytest.raises(ValueError,match='source changed'):reg.audit_module(dm,d)


def test_unregistered_dependency_is_rejected_and_duplicate_import_reuses_only_immutable_code(monkeypatch):
    reg,c,d,cm,dm=loaded(monkeypatch)
    assert reg.prepare(d) is reg._compiled[d] and reg.compile_count==2
    with pytest.raises(ValueError,match='not frozen'):
        custody([c],{c:reg.specs[c][0]},ROOT,enabled=True)


def test_receipt_is_immutable_and_not_a_saved_test_result(monkeypatch):
    reg,c,d,cm,dm=loaded(monkeypatch);receipt=reg._receipts[d]
    assert type(receipt) is Receipt and 'passed' not in receipt._fields
    with pytest.raises(AttributeError):receipt.policy='plain'
    with pytest.raises(TypeError):reg._images[d][('test_origin',1)]='fake'
    with pytest.raises(AssertionError):cm.test_origin(-1)


def test_code_replacement_after_audit_is_not_hidden_by_parametrized_item_reuse(monkeypatch):
    reg,c,d,cm,dm=loaded(monkeypatch)
    reg.definition(cm,c,cm.test_origin,'test_origin')
    assert reg.definition(cm,c,cm.test_origin,'test_origin')[0]==d
    source=d.read_bytes();ns={};exec(compile(source,str(d),'exec',dont_inherit=True,optimize=2),ns)
    dm.test_origin.__code__=ns['test_origin'].__code__
    with pytest.raises(ValueError,match='binding changed'):reg.definition(cm,c,cm.test_origin,'test_origin')


def test_original_v2_autouse_fixture_rebinds_and_restores_the_actual_original_namespace():
    from experiments.neural_hz_20260831 import test_c6_support_affine_plan_v1 as original
    from experiments.neural_hz_20260831 import test_c6_support_affine_plan_v2 as reuse
    before=(original.plan,original.SupportEngine)
    with pytest.MonkeyPatch.context() as change:
        function(reuse.use_v2_for_the_complete_original_suite)(change)
        assert original.plan is reuse.v2.plan and original.SupportEngine is reuse.v2.SupportEngine
    assert (original.plan,original.SupportEngine)==before
