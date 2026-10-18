"""Independent V2 authority chain. Old V1 approvals cannot enable this path.

No concrete resource plan or execution envelope is supplied by this module.
The committed registry is deliberately empty until a separately approved freeze.
"""
import ast
import hashlib
import math
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY, decode, require, sha
from scoped_source.native_method import validate
from scoped_source.native_protocol import PYTHON, SITE, REPORT_CAP, COUNTERS, STAGES, counts
from scoped_source.native_resources import Resources, bounded_read, lexical, measure, parse, tick

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT/'docs/hz_native_resource_execution_approvals.json'
SCHEMA = 'NATIVE_RESOURCE_ENVELOPE_V2'
SLOTS = ('session','metadata','source','observer','observer_next','phase','terminal','caller')


def registry(deadline):
    # This small committed root of authority and the interpreter bootstrap stay
    # trusted; they are bounded reads, never caller-authored Python objects.
    raw, h = bounded_read(REGISTRY, 65536, deadline)
    lexical(raw, deadline, 64)
    obj = decode(raw)
    require(type(obj) is dict and set(obj) == {'schema','envelopes'}
            and obj['schema'] == 'NATIVE_RESOURCE_APPROVALS_V2'
            and type(obj['envelopes']) is list and len(obj['envelopes']) <= 128, 'V2 authority schema')
    result = {}
    for row in obj['envelopes']:
        tick(deadline)
        require(type(row) is dict and set(row) == {'sha256','resources','scope'}
                and row['scope'] == 'SEPARATELY_FROZEN_NATIVE_CPU_CONTROL_V2', 'V2 explicit execution scope')
        key = sha(row['sha256']); require(key not in result, 'unique envelope authority')
        result[key] = row['resources']
    return result, h


def canonical_hash(obj, deadline, cap=REPORT_CAP, depth=64):
    return measure(obj, deadline, cap, depth)['sha256']


def common_contract(value):
    require(type(value) is dict and set(value) == {'schema','policy','seconds','algorithm',
        'counter_contract','mathematical_contract','source_contract','ownership_formula'}
        and value['schema'] == 'NATIVE_COMMON_CONTRACT_V2', 'closed common contract')
    require(value['policy'] == dict(POLICY) and type(value['seconds']) is int
            and value['seconds'] == 300, 'no policy or time change')
    expected = dict(device='cpu',dtype='float64',threads=1,updates=128,evaluations=129,
        profile=0,roles=['proposed','zero'],candidate_width=8,check_width=2,
        production='pair_scoped',replay='pair_scoped')
    require(value['algorithm'] == expected, 'fixed common two-arm algorithm')
    for field in ('counter_contract','mathematical_contract','source_contract','ownership_formula'):
        sha(value[field])


def profile(value):
    # Pure validation, NOT a call to V1's execution-authorizing validate_profile.
    require(type(value) is dict and set(value) == {'schema','method','source','policy','seconds',
        'as_bytes','file_bytes','rss_bytes','implementation','environment'}
        and value['schema'] == 'NATIVE_SHARED_EXECUTION_V2', 'V2 profile schema')
    validate(value['method'])
    from scoped_source.native_protocol import source_spec
    source_spec(value['source'])
    # V2 is not a general external-source intake yet. A raw/layout preflight
    # followed by V1 reopening an externally mutable manifest is insufficient
    # for a fine-grained parser reservation. Do not hide that TOCTOU gap.
    require(value['source']['kind'] == 'fixed_control', 'V2 fixed synthetic declarations only')
    require(value['policy'] == dict(POLICY) and type(value['seconds']) is int
            and value['seconds'] == 300, 'unchanged complete numerical gates')
    for k in ('as_bytes','file_bytes','rss_bytes'):
        require(type(value[k]) is int and value[k] > 0, 'separately approved OS limits')
    require(value['file_bytes'] <= POLICY['total_bytes'], 'old file ceiling')
    code = value['implementation']
    require(type(code) is dict and 0 < len(code) <= 256, 'complete bounded code roster')
    for name, h in code.items():
        require(type(name) is str and name == str(Path(name)) and not Path(name).is_absolute()
                and '..' not in Path(name).parts, 'relative implementation member')
        sha(h)
    env = value['environment']
    require(type(env) is dict and set(env) == {'python','site','torch_version','files'}
            and env['python'] == PYTHON and env['site'] == SITE and type(env['torch_version']) is str
            and type(env['files']) is list and 0 < len(env['files']) <= 128, 'existing environment only')
    names = set(); paths = set()
    for v in env['files']:
        require(type(v) is dict and set(v) == {'file','bytes','sha256'}
                and type(v['file']) is str and type(v['bytes']) is int and v['bytes'] > 0, 'environment member')
        p = Path(v['file'])
        require(p.is_absolute() and p.is_relative_to(Path(PYTHON).parent.parent)
                and v['file'] not in paths, 'unique environment binding')
        paths.add(v['file']); names.add(p.name); sha(v['sha256'])
    require({'python','__init__.py','libtorch_cpu.so','libc10.so'} <= names
            and any(n.startswith('_C.') and n.endswith('.so') for n in names), 'native environment coverage')


class Session:
    """Decoded metadata is owned by this persistent reservation until close."""
    def __init__(self, raw, expected, role, started):
        require(type(raw) is bytes and 0 < len(raw) <= REPORT_CAP
                and type(started) in (int,float) and math.isfinite(started), 'bounded immutable intake')
        self.started = started; self.deadline = started+300
        tick(self.deadline); expected = sha(expected)
        approved, rh = registry(self.deadline)
        require(expected in approved, 'no V2 execution approved')
        require(hashlib.sha256(raw).hexdigest() == expected, 'approved envelope raw identity')
        self.resources = Resources(approved[expected],expected,role,self.deadline)
        require(set(self.resources.slots) == set(SLOTS), 'complete finite reservation roster')
        self._lease = self.resources.hold('session'); self._lease.__enter__()
        try:
            e, ref, self.lexical = parse(raw,self.resources,'session')
            require(type(e) is dict and set(e) == {'schema','common','layout','profile','resources','root'}
                    and e['schema'] == SCHEMA, 'closed frozen envelope')
            require(e['resources'] == approved[expected], 'externally bound ownership plan')
            common_contract(e['common']); profile(e['profile'])
            self.common = canonical_hash(e['common'],self.deadline)
            self.layout_hash = canonical_hash(e['layout'],self.deadline)
            require(type(e['layout']) is dict and e['layout'].get('schema') == 'NATIVE_SOURCE_LAYOUT_V2'
                    and e['layout'].get('common') == self.common, 'layout and common binding')
            require(e['resources']['formula'] == e['common']['ownership_formula'], 'same reviewed object formulas')
            require(type(e['root']) is str, 'frozen output root')
            p = Path(e['root'])
            require(p.is_absolute() and p.resolve() == p and p.is_relative_to('/data1/Kane/MOE')
                    and p != Path('/data1/Kane/MOE'), 'independent project-local root')
            self.raw, self.envelope, self.expected = raw, e, expected
            self.registry_hash, self.role = rh, role
            self.profile = e['profile']; self.root = p
        except BaseException:
            self._lease.__exit__(None,None,None); raise

    def close(self):
        # No caller may use decoded metadata after this explicit lifetime end.
        self.raw = self.envelope = self.profile = None
        self._lease.__exit__(None,None,None)
        self.resources.closed = True

    def refs(self):
        return dict(envelope=self.expected,registry=self.registry_hash,
                    common=self.common,layout=self.layout_hash)

    def verify_invocation(self, value, root):
        require(type(value) is dict and set(value) == {'schema','root','code_root','profile','bindings',
            'envelope','nonce','started','run_deadline','cleanup_deadline','deadline'}
            and value['schema'] == 'NATIVE_RESOURCE_INVOCATION_V2', 'V2 invocation schema')
        require(value['root'] == str(root) == str(self.root) and value['code_root'] == str(ROOT)
                and value['profile'] == self.profile and value['bindings'] == self.refs()
                and value['envelope'] == self.expected and value['started'] == self.started,
                'one bound invocation')
        require(value['run_deadline'] == self.started+299
                and value['cleanup_deadline'] == self.started+299.5
                and value['deadline'] == self.deadline, 'original cumulative deadlines')
        tick(value['run_deadline'])


def closure(root, deadline):
    pending = ['scoped_source/native_resource_supervised.py','scoped_source/native_resource_production.py',
        'scoped_source/native_resource_portable.py',
        'act/back_end/moe/source_row_endpoints.py','act/back_end/moe/scaled_forest_endpoints.py']
    found = set()
    while pending:
        tick(deadline); name = pending.pop()
        if name in found: continue
        require(len(found) < 256, 'old closure size gate')
        raw, _ = bounded_read(Path(root)/name,128*2**10,deadline)
        found.add(name)
        for node in ast.walk(ast.parse(raw,filename=name)):
            tick(deadline)
            modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                if module and module.split('.')[0] in ('scoped_source','scoped_proof','source_enclosure','upstream_source','router_source'):
                    target = module.replace('.','/')+'.py'
                    if (Path(root)/target).is_file(): pending.append(target)
    return tuple(sorted(found))


def verify_code(root, expected, deadline):
    require(set(expected) == set(closure(root,deadline)), 'complete V2 implementation closure')
    for name, h in expected.items():
        bounded_read(Path(root)/name,128*2**10,deadline,sha(h))
    tick(deadline)


def layout_preflight(directory, layout, source, resources, work):
    """Before the FIRST TensorRows construction, with all old checks still next."""
    require(type(layout) is dict and set(layout) == {'schema','common','raw','source','canonical_bytes',
        'request','request_bytes','lexical','tensor_count','chunk_count','max_rank','members','total_bytes'}, 'source layout schema')
    require(layout['source'] == sha(source), 'layout source identity')
    with resources.hold('source') as slot:
        raw, h = bounded_read(Path(directory)/'manifest.json',4*2**20,resources.deadline,layout['raw']['sha256'])
        require(layout['raw'] == dict(bytes=len(raw),sha256=h), 'raw manifest binding')
        # Noncanonical UTF-8/16/32 source bytes stay valid, unlike native metadata.
        m, canonical, stats = parse(raw,resources,'source',canonical=False)
        require(stats == layout['lexical'] and canonical == dict(bytes=layout['canonical_bytes'],sha256=source), 'complete source lexical/canonical binding')
        request = m['declaration']['request']
        rr = measure(request,resources.deadline,slot['encoded'],slot['depth'])
        require(rr == dict(bytes=layout['request_bytes'],sha256=layout['request']), 'unaltered request types and identity')
        tensors = m['tensors']; chunks = 0; rank = 0; total = len(raw); members = 1
        for i,t in enumerate(tensors):
            work.visit('row_visits'); shape = t['shape']
            require(type(shape) is list and all(type(n) is int and n > 0 for n in shape), 'strict tensor shape')
            rank = max(rank,len(shape)); end = 0
            for j,c in enumerate(t['chunks']):
                work.visit('row_visits')
                require(type(c) is dict and set(c) == {'file','bytes','sha256','offset'}
                        and c['file'] == f't/{i:05d}-{j:05d}.bin'
                        and type(c['offset']) is int and c['offset'] == end
                        and type(c['bytes']) is int and 0 < c['bytes'] <= 2**20 and c['bytes'] % 8 == 0,
                        'complete contiguous source chunks')
                sha(c['sha256']); end += c['bytes']; total += c['bytes']; members += 1; chunks += 1
            require(end == 8*math.prod(shape), 'all tensor bytes covered')
        require((len(tensors),chunks,rank,members,total) == (layout['tensor_count'],layout['chunk_count'],
            layout['max_rank'],layout['members'],layout['total_bytes']), 'actual not hard-coded source layout')
        require(total <= POLICY['total_bytes'] and members <= POLICY['members'], 'original source limits')
        # No TensorRows/lowering/value check was removed: the caller still runs
        # them, copied-source validation and final seal. This is metadata only.
        work.tick()
        return dict(raw=h,source=source,request=rr['sha256'],members=members,bytes=total,
                    scope='COST_PREFLIGHT_NOT_NETWORK_PROOF')
