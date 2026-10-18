"""Default-off native execution contract with externally frozen permissions.

An implementation approval is not a numerical resource limit or run approval.
The registry starts empty; a separately approved freeze must bind
source, implementation, native environment and all limits before any launch.
"""
import ast
import math
import os
from pathlib import Path
import resource
import time

from scoped_source.hz_request_store import POLICY, Work, compact, decode, digest, identity, require, sha
from scoped_source.native_method import validate

APPROVAL_FILE = Path(__file__).resolve().parent.parent/'docs/hz_native_execution_approvals.json'
PYTHON = '/data1/Kane/miniconda3/envs/act-py312/bin/python'
SITE = '/data1/Kane/miniconda3/envs/act-py312/lib/python3.12/site-packages'
REPORT_CAP = 2**20
LOG_CAP = 65536
META_RESERVE = 8*2**20  # Inside existing 128 MiB; not a new resource allowance.
META_MEMBERS = 16
COUNTERS = ('row_visits', 'coefficient_visits', 'arithmetic')
STAGES = ('production', 'replay', 'reception')
CONTROL_FAULTS = ('source_binding', 'setup_error', 'proposal_block', 'close_error',
    'partial_candidate', 'replay_tamper', 'reception_block', 'late_receipt',
    'limit_mismatch', 'cleanup_unconfirmed')
META_CAPS = {'invocation.json':REPORT_CAP, 'controller.json':REPORT_CAP,
             'terminal.json':REPORT_CAP, 'controller.log':REPORT_CAP}
for _stage in STAGES:
    META_CAPS.update({_stage+'.json':REPORT_CAP, _stage+'.log':LOG_CAP, _stage+'.progress':LOG_CAP})


def counts(value, *, complete=True):
    require(type(value) is dict and set(value) == set(COUNTERS)
            and all(type(value[k]) is int and value[k] >= 0 for k in COUNTERS), 'closed cumulative Work')
    if complete: require(sum(value.values()) <= POLICY['visits'], 'unchanged cumulative Work capacity')
    return dict(value)


def restore(value, deadline):
    initial = counts(value)
    work = Work(deadline); work.counts.update(initial)
    return work


def read_json(path, *, expected=None, cap=REPORT_CAP, deadline=None):
    if deadline is not None: require(time.monotonic() < deadline, 'native read deadline')
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_nlink == 1
            and path.stat().st_size <= cap, 'bounded unaliased native record')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream: raw = stream.read(cap+1)
    require(len(raw) <= cap and (expected is None or digest(raw) == sha(expected)), 'native record hash/cap')
    obj = decode(raw); require(compact(obj) == raw, 'native record canonical form')
    if deadline is not None: require(time.monotonic() < deadline, 'native read completed timely')
    return obj, digest(raw)


def write_json(path, obj, *, deadline, cap=REPORT_CAP):
    require(time.monotonic() < deadline, 'native publish deadline')
    raw = compact(obj); require(len(raw) <= cap, 'native metadata reserved capacity')
    # No-overwrite publication; a partial or late file is never a valid receipt.
    with Path(path).open('xb', buffering=0) as stream:
        offset = 0
        while offset < len(raw):
            require(time.monotonic() < deadline, 'native partial publish deadline')
            n = stream.write(memoryview(raw)[offset:offset+65536])
            require(n and n > 0, 'native metadata short write'); offset += n
        os.fsync(stream.fileno())
    require(time.monotonic() < deadline, 'native publish timely')
    return dict(bytes=len(raw), sha256=digest(raw))


def source_spec(value):
    require(type(value) is dict and value.get('kind') in ('declared_artifact','fixed_control'), 'closed native source kind')
    fields = {'kind','source','directory'} if value['kind'] == 'declared_artifact' else {'kind','source','case'}
    require(set(value) == fields, 'closed native source binding'); sha(value['source'])
    if value['kind'] == 'fixed_control':
        require(value['case'] in ('mixed','clipped','guarded'), 'finite synthetic control identity')
    else:
        p = Path(value['directory'])
        require(p.is_absolute() and p.resolve() == p and p.is_relative_to('/data1/Kane/MOE'), 'declared source location')
    return dict(value)


def validate_profile(value):
    # Must be the first gate: no native imports, source reads, limit changes or
    # actor launches are permitted by a caller's self-authored profile.
    approved, _ = approval_registry()
    require(identity(value) in approved, 'no approved native execution profile')
    base = {'schema','method','source','policy','seconds',
        'as_bytes','file_bytes','rss_bytes','implementation','environment'}
    require(type(value) is dict and set(value) in (base, base | {'control'})
        and value['schema'] == 'NATIVE_SHARED_EXECUTION_V1', 'closed execution profile')
    validate(value['method']); source_spec(value['source'])
    require(value.get('control') is None or value['control'] in CONTROL_FAULTS,
            'closed separately registered control fault')
    require(value.get('control') is None or value['source']['kind'] == 'fixed_control',
            'fault injection forbidden for declared real artifacts')
    require(value['policy'] == dict(POLICY) and type(value['seconds']) is int and value['seconds'] == 300,
            'unchanged policy and total deadline')
    for name in ('as_bytes','file_bytes','rss_bytes'):
        require(type(value[name]) is int and value[name] > 0, 'separately frozen positive OS limit')
    require(value['file_bytes'] <= POLICY['total_bytes'], 'file ceiling inside original request storage')
    require(type(value['implementation']) is dict and value['implementation'], 'frozen implementation roster')
    for name, h in value['implementation'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts, 'implementation path')
        sha(h)
    env = value['environment']
    require(type(env) is dict and set(env) == {'python','site','torch_version','files'}
            and env['python'] == PYTHON and env['site'] == SITE and type(env['torch_version']) is str
            and type(env['files']) is list and env['files'], 'frozen existing native environment')
    require(len(env['files']) <= 128 and len({r['file'] for r in env['files']}) == len(env['files']), 'unique environment files')
    names = set()
    for r in env['files']:
        require(type(r) is dict and set(r) == {'file','bytes','sha256'} and type(r['bytes']) is int and r['bytes'] > 0,
                'environment artifact binding')
        p = Path(r['file']); require(p.is_absolute() and p.is_relative_to(Path(PYTHON).parent.parent), 'existing environment only')
        sha(r['sha256']); names.add(p.name)
    require({'python','__init__.py','libtorch_cpu.so','libc10.so'} <= names
            and any(n.startswith('_C.') and n.endswith('.so') for n in names), 'minimum native environment identities')
    return value


def approval_registry():
    # An external trusted, committed permission record avoids a self-referential
    # profile-hash -> implementation-hash -> allowed-profile-hash cycle.
    # Only separately approved concrete control freezes may populate it.
    require(not APPROVAL_FILE.is_symlink() and APPROVAL_FILE.stat().st_size <= 65536,
            'bounded trusted approval registry')
    raw = APPROVAL_FILE.read_bytes()
    require(len(raw) <= 65536, 'bounded trusted approval registry read')
    obj = decode(raw)
    require(type(obj) is dict and set(obj) == {'schema','profiles'}
            and obj['schema'] == 'NATIVE_EXECUTION_APPROVAL_REGISTRY_V1'
            and type(obj['profiles']) is list and len(obj['profiles']) <= 128
            and obj['profiles'] == sorted(set(obj['profiles'])), 'closed approved profile registry')
    for h in obj['profiles']: sha(h)
    return frozenset(obj['profiles']), digest(raw)


def closure(root):
    """Closed local import roster without importing production/native code."""
    root = Path(root)
    pending = ['scoped_source/native_supervised.py','scoped_source/native_production.py',
        'scoped_source/native_package.py','scoped_source/native_portable.py',
        'act/back_end/moe/source_row_endpoints.py','act/back_end/moe/scaled_forest_endpoints.py']
    found = set()
    while pending:
        name = pending.pop()
        if name in found: continue
        require(len(found) < 256, 'bounded local dependency roster')
        path = root/name; require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 128*2**10,
                                  'bounded local implementation')
        found.add(name)
        tree = ast.parse(path.read_bytes(),filename=name)
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                if module and module.split('.')[0] in ('scoped_source','scoped_proof','source_enclosure','upstream_source','router_source'):
                    target = module.replace('.','/')+'.py'
                    if (root/target).is_file(): pending.append(target)
    return tuple(sorted(found))


def verify_code(root, expected, deadline):
    require(set(expected) == set(closure(root)), 'complete local implementation closure')
    for name, h in expected.items():
        require(time.monotonic() < deadline, 'implementation verification deadline')
        path = Path(root)/name
        require(path.stat().st_nlink == 1 and digest(path.read_bytes()) == sha(h), 'frozen executed implementation')
    require(time.monotonic() < deadline, 'implementation verification timely')


def hard_limits(profile, role):
    require(role in ('controller', *STAGES), 'closed limit role')
    target = profile['as_bytes']; hard_file = profile['file_bytes']
    soft_file = min(REPORT_CAP, hard_file) if role == 'controller' else hard_file
    require(resource.getrlimit(resource.RLIMIT_AS) == (target,target)
            and resource.getrlimit(resource.RLIMIT_FSIZE) == (soft_file,hard_file), 'hard limits inherited and verified')
    return dict(address_space_bytes=target,file_soft_bytes=soft_file,file_hard_bytes=hard_file,
                per_process=True,aggregate_rss_is_sampled=True,bootstrap_trusted=True)


def validate_receipt(value, *, stage, invocation, previous, initial, deadline, started):
    require(type(value) is dict and value.get('schema') == 'NATIVE_ACTOR_RECEIPT_V1'
            and value.get('stage') == stage and value.get('invocation') == invocation
            and value.get('previous') == previous and value.get('status') == 'COMPLETE'
            and value.get('initial_work') == counts(initial), 'native complete phase chain')
    final = counts(value['work'])
    require(all(final[k] >= initial[k] for k in COUNTERS)
            and value['delta_work'] == {k:final[k]-initial[k] for k in COUNTERS}, 'one cumulative Work chain')
    require(value['deadline'] == deadline and type(value['begin']) in (int,float)
            and type(value['end']) in (int,float) and math.isfinite(value['begin']) and math.isfinite(value['end'])
            and started <= value['begin'] <= value['end'] < deadline, 'one timely actor interval')
    return final
