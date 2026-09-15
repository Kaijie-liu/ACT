"""Opt-in execution contract over frozen ACT, not a new numerical policy.

Single worker / main thread only. Local allocations include construction;
native SciPy limits are rechecked after durable READY publication. Public ACT
defaults and all historical source inventories remain unchanged on disk.
"""
import functools
import importlib
import inspect
import math
import sys
import threading
import time

from act.pipeline.moe.request_budget import RequestBudget, BudgetExhausted
from scripts.f0_timing_trace import Recorder, TraceFailure, outcome


class Grant(float):
    def __new__(cls, value, deadline):
        result = super().__new__(cls, value)
        result.deadline = deadline
        return result


def property_scope(arguments):
    encodings = arguments.get('encodings')
    if encodings is None:
        encodings = [arguments['encoding']]
    if not encodings:
        raise ValueError('property query lacks encodings')
    first = encodings[0]
    row = list(first.property_row)
    constant = float(first.property_constant)
    if any(list(e.property_row) != row or float(e.property_constant) != constant for e in encodings):
        raise ValueError('mixed properties in union')
    return {'pairs':[list(e.pair) for e in encodings], 'row':row, 'constant':constant}


class Contract:
    active = False

    def __init__(self, *, started, total=300., reserve=5., journal, clock=time.monotonic):
        if not all(math.isfinite(v) for v in (started, total, reserve)) or not 0 < reserve < total:
            raise ValueError('invalid execution budget')
        if started > clock():
            raise ValueError('future request start')
        self.started, self.total, self.reserve = float(started), float(total), float(reserve)
        self.clock, self.journal = clock, journal
        self.outer_deadline = self.started + self.total
        self.work_deadline = self.outer_deadline - self.reserve
        self.deadlines = []
        self.patches = []
        self.pending_property = None

    def emit(self, kind, **fields):
        return self.journal.emit(kind, clock_elapsed=self.clock()-self.started, **fields)

    def make_budget_class(self):
        context = self
        class ReservedBudget(RequestBudget):
            def __init__(self, seconds, *, started=None, clock=None):
                if float(seconds) != context.total or started != context.started or clock is not None:
                    raise ValueError('request budget identity differs from execution contract')
                super().__init__(seconds, started=started, clock=context.clock)
                self.deadline = context.work_deadline

            def limit(self, stage, cap=None, obligations=1, until=None):
                value = super().limit(stage, cap, obligations, until)
                event = self.events[-1]
                deadline = self.started + event['elapsed_seconds'] + value
                context.emit('GRANT', stage=stage, allocation=value,
                             deadline=deadline-context.started, obligations=obligations)
                return Grant(value, deadline)

            def record(self):
                return {**super().record(), 'execution_contract_version':2,
                        'terminal_reserve_seconds':context.reserve,
                        'work_deadline_elapsed':context.total-context.reserve,
                        'outer_deadline_elapsed':context.total}
        return ReservedBudget

    def scope_wrapper(self, fn, *, property_query=False):
        signature = inspect.signature(fn)
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            allocation = arguments['time_limit']
            if not math.isfinite(allocation) or allocation <= 0:
                raise BudgetExhausted('invalid/exhausted local allocation')
            deadline = min(self.work_deadline,
                           getattr(allocation, 'deadline', self.clock()+float(allocation)),
                           *self.deadlines)
            scope = property_scope(arguments) if property_query else None
            token = self.emit('PROPERTY_BEGIN' if property_query else 'LOCAL_BEGIN',
                              function=fn.__name__, deadline=deadline-self.started,
                              allocation=float(allocation), scope=scope)
            self.deadlines.append(deadline)
            try:
                if self.clock() >= deadline:
                    raise BudgetExhausted('construction exhausted local allocation')
                result = fn(*args, **kwargs)
                if property_query:
                    self.emit('PROPERTY_RESULT', token=token, result=outcome(result),
                              candidate_present=getattr(result, 'candidate_input', None) is not None,
                              evidence_role='INTERMEDIATE_NOT_REQUEST_VERDICT')
                    self.pending_property = token
                else:
                    self.emit('LOCAL_END', token=token)
                return result
            except BaseException as exc:
                self.emit('PROPERTY_RAISE' if property_query else 'LOCAL_RAISE',
                          token=token, exception=type(exc).__name__)
                raise
            finally:
                self.deadlines.pop()
        return wrapped

    def native_wrapper(self, fn):
        signature = inspect.signature(fn)
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            options = dict(bound.arguments.get('options') or {})
            requested = options.get('time_limit')
            if requested is None or not math.isfinite(requested) or requested <= 0:
                raise ValueError('unbounded native call is outside V2 contract')
            deadline = min(self.work_deadline, *self.deadlines) if self.deadlines else self.work_deadline
            token = self.emit('NATIVE_READY', requested=float(requested),
                              deadline=deadline-self.started,
                              original_options=options)
            # Recheck AFTER fsync. READY binds the deadline, not a fabricated
            # exact launch time. Returned calls record their actual entry/limit.
            entered = self.clock()
            effective = min(float(requested), deadline-entered)
            if effective < .001:
                self.emit('NATIVE_SKIPPED', token=token, reason='NO_POSITIVE_NATIVE_BUDGET')
                raise BudgetExhausted('native entry budget exhausted')
            options['time_limit'] = effective
            bound.arguments['options'] = options
            try:
                result = fn(*bound.args, **bound.kwargs)
            except BaseException as exc:
                self.emit('NATIVE_RAISE', token=token, entered=entered-self.started,
                          effective=effective, exception=type(exc).__name__)
                raise
            self.emit('NATIVE_RETURN', token=token, entered=entered-self.started,
                      effective=effective, result=outcome(result))
            return result
        return wrapped

    def replay_wrapper(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # Only the direct F0 caller can consume its preceding property
            # token. Do not infer scope from stale pair/index loop locals.
            caller = sys._getframe(1)
            expected = ((caller.f_globals.get('__name__'), caller.f_code.co_name) in {
                ('act.pipeline.moe.staged_verifier','_run_f0_impl'),
                ('act.pipeline.moe.paired_monolithic','_run_monolithic')})
            del caller
            token = self.pending_property if expected else None
            result = fn(*args, **kwargs)
            if token is not None:
                self.emit('PROPERTY_REPLAY', token=token, result=result,
                          evidence_role='REPLAY_OBSERVATION_NOT_STANDALONE_CERTIFICATE')
                self.pending_property = None
            return result
        return wrapped

    def __enter__(self):
        if Contract.active or threading.current_thread() is not threading.main_thread():
            raise RuntimeError('one main-thread execution adapter per process required')
        Contract.active = True
        try:
            # Eager import covers static aliases; subsequent imports read these
            # patched module globals. Nothing is installed in SciPy itself.
            targets = {
                'act.pipeline.moe.experiment1':('_solve_output',),
                'act.back_end.solver.solver_hz':('hz_support_bounds','hz_minimize_output'),
                'act.back_end.moe.weighted_top2':('compute_weighted_top2_gate_range',
                    'compute_weighted_top2_difference_range','solve_weighted_top2_f0'),
                'act.back_end.moe.monolithic_f0':('solve_monolithic_weighted_top2_f0',)}
            for name in ('act.pipeline.moe.staged_verifier', 'act.pipeline.moe.paired_monolithic',
                         'act.pipeline.moe.route_complexity_schedule', 'act.back_end.hybridz_tf.tf_mlp'):
                importlib.import_module(name)
            replacements = {id(RequestBudget):(RequestBudget,self.make_budget_class())}
            for name, functions in targets.items():
                module = importlib.import_module(name)
                for key in functions:
                    original = getattr(module,key)
                    replacements[id(original)] = (original,self.scope_wrapper(original,
                        property_query=key in ('solve_weighted_top2_f0','solve_monolithic_weighted_top2_f0')))
            from scipy.optimize import milp
            from act.pipeline.moe.experiment1 import _forward_validate
            replacements[id(milp)] = (milp,self.native_wrapper(milp))
            replacements[id(_forward_validate)] = (_forward_validate,self.replay_wrapper(_forward_validate))
            for name, module in list(sys.modules.items()):
                if not name.startswith('act.') or module is None: continue
                for key, original in list(vars(module).items()):
                    replacement = replacements.get(id(original))
                    if replacement is not None and original is replacement[0]:
                        self.patches.append((module,key,original))
                        setattr(module,key,replacement[1])
            self.emit('CONTRACT_INSTALLED', version=2, total=self.total, reserve=self.reserve,
                      work_deadline=self.work_deadline-self.started,
                      aliases=[f'{m.__name__}.{k}' for m,k,_ in self.patches])
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self, *_):
        for module,key,original in reversed(self.patches): setattr(module,key,original)
        self.patches.clear()
        Contract.active = False


def verify_v2(model, center, epsilon, config, *, journal_path, started, identity,
              progress_callback=None, **kwargs):
    """Explicit API; caller still owns the300s process-group watchdog.

    WORK_COMPLETE is not proof acceptance. Journals cannot independently promote
    a request; the standard complete package and independent terminal auditor
    remain mandatory. No CROWN/highspy or unscheduled execution in this version.
    """
    schedule = config.get('route_complexity_schedule', {})
    if (schedule.get('total_seconds') != 300 or schedule.get('multi_pair_tier1_fraction') != .25
            or config.get('comparison_method') not in ('staged','monolithic_f0')
            or not config.get('scoped_proof_reuse')
            or any(config[phase][part].get('backend' if part=='solver' else 'solver_backend','scipy') != 'scipy'
                   for phase in ('tier1','f0') for part in ('solver','support'))):
        raise ValueError('unsupported V2 execution contract')
    journal = Recorder(journal_path, started, identity)
    try:
        with Contract(started=started, journal=journal) as contract:
            from act.pipeline.moe.staged_verifier import verify_staged_linf
            def progress(record):
                contract.emit('STATE', record=record)
                if progress_callback: progress_callback(record)
            report = verify_staged_linf(model, center, epsilon, config,
                budget_started_at=started, progress_callback=progress, **kwargs)
            report.evidence['execution_budget_contract'] = {
                'version':2, 'terminal_reserve_seconds':5., 'total_seconds':300.,
                'local_construction_charged':True, 'native_entry_rechecked':True,
                'journal_identity':identity, 'partial_journal_can_establish_SAFE':False}
            contract.emit('WORK_COMPLETE', status=report.status,
                          evidence_role='REQUIRES_COMPLETE_PACKAGE_AND_TERMINAL_AUDIT')
            return report
    finally:
        journal.close()
