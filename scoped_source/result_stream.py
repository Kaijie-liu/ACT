"""Default-off bounded result publication, not a replacement for mathematical checks."""
from fractions import Fraction
import re

from scoped_source.hz_request_store import POLICY, compact, decode, identity, require, sha
from scoped_source.forest_endpoints import THRESHOLD

PAGE_ROWS = 8
PAGE_BYTES = 65536
INDEX_BYTES = 2**20
MAX_ROWS = 252
MAX_PAGES = 32
MAX_RATIONAL = 2 * ((POLICY['work_bits'] * 30103 + 99999) // 100000) + 2
SUMMARY_KEYS = {'status', 'required', 'positive', 'required_endpoints', 'checked_endpoints',
                'missing_endpoints', 'native_float_proof', 'checkpoint_correspondence_checked'}
CONTEXT_KEYS = {'schema', 'source', 'forest', 'forest_index', 'contract', 'method', 'invocation', 'roster'}


def error_code(exc):
    # No user-controlled args/str/repr/traceback, including during cleanup.
    return 'DEADLINE' if isinstance(exc, TimeoutError) else 'REJECTED' if isinstance(exc, ValueError) else 'EXECUTION_ERROR'


def lower(value):
    if value is None:
        return None
    require(type(value) is str and len(value) <= MAX_RATIONAL and
            re.fullmatch(r'-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?', value), 'bounded result rational')
    q = Fraction(value)
    require(str(q) == value and max(q.numerator.bit_length(), q.denominator.bit_length()) <= POLICY['work_bits'],
            'canonical work-bit result rational')
    return q


def row(value):
    require(type(value) is dict and set(value) == {'pair', 'other', 'lower', 'positive'}, 'result row fields')
    pair = value['pair']
    require(type(pair) is list and len(pair) == 2 and all(type(i) is int and 0 <= i < 8 for i in pair)
            and pair[0] < pair[1] and type(value['other']) is int and 0 <= value['other'] < 10,
            'result pair/property types')
    q = lower(value['lower'])
    require(type(value['positive']) is bool and value['positive'] is (q is not None and q > THRESHOLD),
            'exact result sign')
    return dict(pair=list(pair), other=value['other'], lower=value['lower'], positive=value['positive'])


def summary(value):
    require(type(value) is dict and set(value) == SUMMARY_KEYS, 'result summary fields')
    for key, cap in (('required', MAX_ROWS), ('positive', MAX_ROWS), ('required_endpoints', 504),
                     ('checked_endpoints', 504), ('missing_endpoints', 504)):
        require(type(value[key]) is int and 0 <= value[key] <= cap, 'result summary count')
    require(0 < value['required'] and value['positive'] <= value['required'] and
            value['required'] <= value['required_endpoints'] <= 2 * value['required'] and
            value['checked_endpoints'] + value['missing_endpoints'] == value['required_endpoints'] and
            value['native_float_proof'] is False and value['checkpoint_correspondence_checked'] is False,
            'result summary completeness')
    expected = ('CHECKED_POSITIVE_CAPTURED_REAL_SOURCE' if value['positive'] == value['required'] else
                'UNKNOWN_MISSING_EVIDENCE' if value['missing_endpoints'] else 'UNKNOWN_NONPOSITIVE')
    require(value['status'] == expected and not (value['positive'] == value['required'] and value['missing_endpoints']),
            'result summary status')
    return dict(value)


def context(value):
    require(type(value) is dict and set(value) == CONTEXT_KEYS and value['schema'] == 'SHARED_RESULT_CONTEXT_V1',
            'result context fields')
    for key in CONTEXT_KEYS - {'schema'}:
        sha(value[key])
    return dict(value)


def plan_context(forest, plan, method, invocation, forest_index, work):
    work.tick()
    # The caller is Session.finish, with its private independently rebuilt plan.
    duties = [dict(pair=list(pair), other=other, slots=list(slots)) for pair, other, slots in plan.duties]
    require(0 < len(duties) <= MAX_ROWS, 'bounded private roster')
    roster = identity(dict(schema='SHARED_RESULT_ROSTER_V1', contract=plan.contract, duties=duties))
    result = context(dict(schema='SHARED_RESULT_CONTEXT_V1', source=forest.source, forest=forest.identity,
        forest_index=forest_index, contract=plan.contract, method=identity(method), invocation=invocation, roster=roster))
    work.tick()
    return result


def candidate_identity(refs):
    require(type(refs) is list and 0 < len(refs) <= 504, 'bounded candidate roster')
    for i, ref in enumerate(refs):
        require(type(ref) is dict and set(ref) == {'file', 'bytes', 'sha256'} and ref['file'] == f'tiles/{i:04d}.bin'
                and type(ref['bytes']) is int and 0 < ref['bytes'] <= 4*2**20, 'candidate result binding')
        sha(ref['sha256'])
    return identity(refs)


class Writer:
    """At most one mutable page; immutable archive owns all published bytes."""
    def __init__(self, put, method, invocation, forest_index, candidates, work, *, fault=''):
        from scoped_source.direct_method import validate, paged
        self.method = validate(method)
        require(paged(self.method), 'paged writer opt-in')
        require(fault in ('', 'after_result_page_error', 'after_result_page_block', 'index_publication_expired'),
                'fixed result fault')
        self.put, self.work, self.fault = put, work, fault
        self.invocation, self.forest_index = sha(invocation), sha(forest_index)
        self.candidates = candidate_identity(candidates)
        self.context = None
        self.page, self.pages = [], []
        self.count = self.positive = self.missing_rows = 0
        self.ref = None
        self.closed = False

    def bind(self, forest, plan, work):
        require(work is self.work and self.context is None and not self.closed, 'one result source/Work binding')
        self.context = plan_context(forest, plan, self.method, self.invocation, self.forest_index, work)

    def emit(self, value):
        self.work.tick()
        require(self.context is not None and not self.closed and self.count < MAX_ROWS, 'open bounded result writer')
        saved = row(value)  # precheck type/length before encoding; no mutable aliases
        self.page.append(saved)
        self.count += 1
        self.positive += saved['positive']
        self.missing_rows += saved['lower'] is None
        if len(self.page) == PAGE_ROWS:
            self.flush()

    def flush(self):
        self.work.tick()
        require(not self.closed and self.page and len(self.pages) < MAX_PAGES, 'bounded next result page')
        ordinal = len(self.pages)
        raw = compact(dict(schema='SHARED_RESULT_PAGE_V1', context=identity(self.context), ordinal=ordinal,
                           start=self.count-len(self.page), rows=self.page))
        require(len(raw) <= PAGE_BYTES, 'result page byte ceiling')
        ref = self.put(f'result/{ordinal:04d}.json', raw)
        self.pages.append(dict(ref, start=self.count-len(self.page), count=len(self.page)))
        self.page.clear()
        self.work.tick()
        if self.fault == 'after_result_page_error':
            raise RuntimeError('fixed page fault')
        if self.fault == 'after_result_page_block':
            import time
            while True:
                time.sleep(.01)  # deliberately uncooperative; hard supervisor owns it

    def finish(self, value):
        self.work.tick()
        require(self.context is not None and not self.closed, 'one result seal')
        value = summary(value)
        require(value['required'] == self.count and value['positive'] == self.positive and
                (self.missing_rows == 0) == (value['missing_endpoints'] == 0), 'writer complete aggregate')
        if self.page:
            self.flush()
        raw = compact(dict(schema='SHARED_RESULT_STREAM_V1', context=self.context,
                           candidates=self.candidates, pages=self.pages, summary=value))
        require(len(raw) <= INDEX_BYTES, 'result index byte ceiling')
        self.ref = self.put('result/index.json', raw)
        if self.fault == 'index_publication_expired':
            import time
            while time.monotonic() < self.work.deadline:
                time.sleep(min(.01, max(0., self.work.deadline-time.monotonic())))
        self.work.tick()
        self.closed = True
        return dict(value)
