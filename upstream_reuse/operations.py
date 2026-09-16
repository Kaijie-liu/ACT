"""Request-local dependency binding. No proof verdict or source validation cache."""
from pathlib import Path
from copy import deepcopy

from act.back_end.solver import lp_certificate as lp, sparse_lp_certificate as sparse
from act.back_end.solver import check_hz_lp_export as hz, check_rational_mccormick as weighted
from act.back_end.solver import rational_mccormick as construction
from exact_matrix_cache.checker import _bind
from exact_matrix_cache.cache import MatrixCache
from moe_evidence.generate import reference
from moe_evidence.schema import validate_request
from scripts.optional_evidence_dev_contract import read, save
from upstream_reuse import native, schedule
from upstream_reuse.storage import SourceCache
from upstream_reuse.timing import Timers


class Operations:
    def __init__(self, root, request, budget, *, source_enabled=False, matrix_enabled=False):
        if budget.total != 300 or type(source_enabled) is not bool or type(matrix_enabled) is not bool:
            raise ValueError('300-second budget and explicit boolean flags required')
        self.root = Path(root).resolve(); self.request = deepcopy(request)
        self.scope = validate_request(request); self.tick = lambda: budget.remaining(2)
        self.timers = Timers(clock=budget.clock, tick=self.tick)
        self.source = SourceCache(self.scope, enabled=source_enabled, tick=self.tick, measure=self.timers.call)
        self.matrix = MatrixCache(self.scope, enabled=matrix_enabled, tick=self.tick)
        self.refs = {}
        manifest = self.read(self.root/'manifest.json')
        for item in manifest['supports'].values(): self.pin(item['export'])
        for item in manifest['contexts'].values(): self.pin(item['joint_source'])
        identity = self.wrap('identity', lp.identity)
        evaluate = self.wrap('dual_evaluate', _bind(sparse.evaluate, rows=self.rows, identity=identity))
        sparse_check = _bind(sparse.check, evaluate=evaluate, identity=identity)
        def check(program, certificate):
            return sparse_check(program, certificate) if 'matrix_format' in program else lp.check(program, certificate)
        self.export_check = self.wrap('export_check', _bind(hz.check_export, _entries=self.entries, check=check, identity=identity))
        self.weighted_check = self.wrap('construction_check', _bind(weighted.check_construction, _entries=self.entries, check=check, identity=identity))
        self.build = self.wrap('construction', _bind(construction.build, rows=self.rows, identity=identity))
        self.evaluate = evaluate; self.sparse_check = sparse_check; self.identity = identity
        self.loop = _bind(schedule.propose_all, read=self.read, save=self.wrap('serialize_save', save),
                          reference=self.wrap('serialize_reference', reference), identity=identity,
                          propose=self.wrap('proposal', self.propose), check_export=self.export_check,
                          check_construction=self.weighted_check, build=self.build)

    def wrap(self, name, fn):
        return lambda *a, **kw: self.timers.call(name, fn, *a, **kw)

    def path(self, path):
        path = Path(path)
        if path.parent.resolve() != self.root or path.is_symlink():
            raise ValueError('source outside request root or symlink')
        return path

    def pin(self, ref):
        path = self.path(self.root/ref['file'])
        if path.name in self.refs and self.refs[path.name] != ref['sha256']:
            raise ValueError('conflicting source references')
        self.refs[path.name] = ref['sha256']

    def read(self, path):
        path = self.path(path)
        if path.name == 'manifest.json':
            obj = self.timers.call('manifest_read', read, path)
            if obj['request'] != self.request or obj['request_id'] != self.scope:
                raise ValueError('request binding mismatch')
            # Revalidate references on every mutable manifest read.
            for item in list(obj['supports'].values()) + list(obj['contexts'].values()):
                ref = item.get('export', item.get('joint_source'))
                if ref and ref['file'] in self.refs and self.refs[ref['file']] != ref['sha256']:
                    raise ValueError('source reference changed')
            return obj
        if path.name not in self.refs: raise ValueError('unbound source read')
        return self.source.load(path, self.refs[path.name], scope=self.scope)

    def rows(self, matrix, n):
        # Original builder concatenates lists; never expose cached immutable rows.
        parsed = self.timers.call('csr_rows', lambda: [list(row) for row in self.matrix.rows(matrix, n)])
        for row in parsed:
            self.tick(); yield row

    def entries(self, matrix):
        return self.timers.call('csr_entries', self.matrix.entries, matrix)

    def propose(self, program, *, time_limit=None):
        if 'matrix_format' not in program:
            return lp.propose(program, time_limit=time_limit)
        def dependencies():
            from scipy.optimize import linprog
            from scipy.sparse import csr_matrix
            return linprog, csr_matrix
        linprog, csr_matrix = self.timers.call('proposal_imports', dependencies)
        fn = _bind(native.propose, rows=self.rows, evaluate=self.evaluate, check=self.sparse_check,
                   identity=self.identity, linprog=self.wrap('native_linprog', linprog), csr_matrix=csr_matrix)
        return fn(program, time_limit=time_limit)

    def close(self):
        self.source.clear(); self.matrix.clear()

    def report(self):
        return {'source_cache': self.source.stats(), 'matrix_cache': self.matrix.stats(),
                'timings': self.timers.values, 'scope': self.scope,
                'check_elision': False, 'schedule_changed': False, 'acceptance_changed': False}
