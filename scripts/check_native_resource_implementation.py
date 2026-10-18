"""Read-only V2 implementation audit: AST/bytes only, NOT runtime tests.

No production import, source construction, solver, Torch, subprocess or limits.
The static record is regenerated on stdout and committed separately; --check
compares it. A PASS is not evidence of a working native or portable execution.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = 'b6ce3eccc2f4ad3921ad262df8e7a254810f6bae'
REPORT = 'docs/hz_native_resource_implementation_20261018.json'
REGISTRY = 'docs/hz_native_resource_execution_approvals.json'
DOC = 'docs/hz_native_resource_implementation_20261018.md'
FILES = (
    'scoped_source/native_resources.py',
    'scoped_source/native_resource_protocol.py',
    'scoped_source/native_resource_production.py',
    'scoped_source/native_resource_portable.py',
    'scoped_source/native_resource_supervised.py',
    'scoped_proof/owned_resource_v2.py',
)


def require(ok, text):
    if not ok: raise ValueError(text)


def raw(name): return (ROOT/name).read_bytes()
def sha(name): return hashlib.sha256(raw(name)).hexdigest()
def tree(name): return ast.parse(raw(name), filename=name)


def function(name, fn):
    return next(n for n in tree(name).body if isinstance(n, ast.FunctionDef) and n.name == fn)


def calls(node, selected):
    return [ast.dump(n, include_attributes=False) for n in ast.walk(node)
            if isinstance(n, ast.Call) and ast.unparse(n.func) in selected]


def analyze():
    proposal = json.loads(raw('docs/hz_native_resource_interface_proposal_20261018.json'))
    for name, expected in proposal['anchors'].items():
        require(sha(name) == expected, 'old frozen source changed: '+name)
    registry = json.loads(raw(REGISTRY))
    require(registry == {'schema':'NATIVE_RESOURCE_APPROVALS_V2','envelopes':[]},
            'implementation approval must not enable execution')
    ast_summary = {}
    for name in FILES:
        t = tree(name)
        compile(t,name,'exec')  # Compile only; never execute/import the object.
        require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ('read_bytes','read_text') for n in ast.walk(t)),
            'new runtime code must bound actual file reads: '+name)
        require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in ('str','repr') and n.args and isinstance(n.args[0], ast.Name)
            and n.args[0].id == 'exc' for n in ast.walk(t)), 'no arbitrary exception formatting')
        require(not any(isinstance(n, (ast.Assign,ast.AugAssign,ast.AnnAssign))
            and 'POLICY' in ast.unparse(n).split('=')[0] for n in ast.walk(t)), 'no policy mutation')
        ast_summary[name] = dict(functions=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
            for n in ast.walk(t)),lines=len(raw(name).splitlines()),compiled_not_executed=True)
    sup = 'scoped_source/native_resource_supervised.py'
    exe = function(sup,'execute')
    defaults = dict(zip((a.arg for a in exe.args.kwonlyargs),exe.args.kw_defaults,strict=True))
    require(isinstance(defaults['enabled'],ast.Constant) and defaults['enabled'].value is False
            and defaults['started'] is None, 'default off and explicit original launcher t0')
    command = function(sup,'command')
    bootstrap = next(n.value.value for n in ast.walk(command) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='bootstrap' for t in n.targets)
        and isinstance(n.value,ast.Constant) and type(n.value.value) is str)
    compile(ast.parse(bootstrap),'<bounded-bootstrap>','exec')
    require('stream.read(cap+1)' in bootstrap and 'read_bytes(' not in bootstrap
            and 'resource.setrlimit' in bootstrap, 'bounded bootstrap before imported actors')
    text = ast.unparse(exe)
    require(text.index('session.close()') < text.index('accepted =') < text.index('return out'),
            'final acceptance after explicit close')
    require('started + 299.5' in text and 'started + 300' in text and 'started + 299' in text,
            'same cumulative deadlines')
    finalizer = ast.unparse(function(sup,'finalize'))
    require('online_report_received=False' in finalizer and 'online_positive=False' in finalizer
            and 'online_report_received=True' not in finalizer, 'durable files cannot self-accept')
    require('min(cap, slot[\'encoded\'])' in ast.unparse(function(sup,'suffix')),
            'actual suffix read respects live backing')
    resource = 'scoped_source/native_resources.py'
    require('stream.read(cap + 1)' in ast.unparse(function(resource,'bounded_read')),
            'stable actual bounded reads')
    require('POLICY[\'work_bits\']' not in ast.unparse(function(resource,'tokens')),
            'opaque JSON integers are not exact-rational arithmetic')
    production = 'scoped_source/native_resource_production.py'
    acquire = ast.unparse(function(production,'acquire'))
    require(acquire.index("spec['kind'] == 'fixed_control'") < acquire.index('original_acquire(root'),
            'refuse external mutable input before first old parser')
    require(acquire.index('original_acquire(root') < acquire.index('layout_preflight(owned.root'),
            'fixed factory bytes then preflight before production lowering')
    profile = ast.unparse(function('scoped_source/native_resource_protocol.py','profile'))
    require("value['source']['kind'] == 'fixed_control'" in profile, 'closed initial source scope')
    retained = {'validate','Lifecycle','build','CheckedSharedForest','contract','Driver','Writer',
        'life.setup','life.submit','driver.finish','life.close','driver.abort'}
    require(calls(function(production,'produce'),retained) ==
            calls(function('scoped_source/native_production.py','produce'),retained),
            'reviewed mathematical call expressions unchanged')
    portable = ast.unparse(function('scoped_source/native_resource_portable.py','check_bound_package'))
    require('resources.verify_plan' in portable and 'layout_preflight' in portable
            and 'check_package(' in portable, 'resource identity then old complete recheck')
    owned = 'scoped_proof/owned_resource_v2.py'
    owned_execute = ast.unparse(function(owned,'execute'))
    require(owned_execute.index('os.killpg(') < owned_execute.index('bounded_reap('),
            'unreaped owned leader pins group before signal')
    require('sample = None' in owned_execute and 'failed_observation' in owned_execute,
            'running roster released and finite failure summary retained')
    gate = ast.unparse(function(owned,'complete'))
    for fragment in ("'COMPLETED'","'leader_reaped'","'LEADER_REAPED_NO_LIVE_GROUP'",
                     "'complete'","'live'","'cleanup_observed_at'"):
        require(fragment in gate, 'complete cleanup acceptance gate: '+fragment)
    paths = set(FILES)|{REGISTRY,DOC,'scripts/check_native_resource_implementation.py',
        'docs/hz_native_resource_interface_proposal_20261018.json'}
    return dict(schema='NATIVE_RESOURCE_IMPLEMENTATION_STATIC_V2',base_head=BASE,
        status='IMPLEMENTED_DEFAULT_OFF_RUNTIME_UNTESTED',
        anchors={name:sha(name) for name in sorted(paths)},
        unchanged_frozen_anchors=len(proposal['anchors']),ast=ast_summary,
        retained_mathematical_calls=sorted(retained),
        permissions=dict(implementation=True,static_audit=True,controls=False,
            native_execution=False,real_models=False,GPU=False,quota_increase=False),
        execution_registry_entries=0,concrete_resource_plan=None,
        work_limit=8000000,artifact_bytes=128*2**20,seconds=300,
        source_raw_cap=4*2**20,source_canonical_old_grammar_bound=32*2**20,
        canonical_bound_is_not_new_artifact_or_heap_quota=True,
        source_scope='FIXED_SYNTHETIC_CONTROL_ONLY_NO_DECLARED_ARTIFACT_INTAKE',
        source_assumption='PRIVATE_MANAGED_AND_RELOCATED_COPY_NO_CONCURRENT_EXTERNAL_WRITER',
        ownership_scope='EXTERNALLY_REVIEWED_NAMED_RESERVATIONS_NOT_ALLOCATOR_OR_ALIAS_PROOF',
        pending=['reviewed concrete slot/joint/failure bounds',
            'separate immutable control freeze and authority',
            'trusted launcher t0 and paired identical C/L audit',
            'runtime encoding/deadline/fault/relocation controls',
            'external-source parser-input binding',
            'standalone portable invocation/return archival under a separate freeze'],
        native_calls=0,control_attempts=0,model_runs=0,gpu_runs=0,
        full_own_heap_upper=None,full_shape_admitted=False,
        full_shape_first_snapshot_floor=13926024,full_shape_expanded_membership_lower=126632,
        scientific_upgrade=False,goals='G1-G6 OPEN')


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(result == json.loads(raw(REPORT)), 'static implementation record drift')
        print('PASS: V2 source/AST/binding audit only; runtime controls 0, model/GPU runs 0')
    else:
        require(not sys.argv[1:], 'only --check is supported')
        print(json.dumps(result,sort_keys=True,indent=2))
