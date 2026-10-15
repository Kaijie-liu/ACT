"""Bounded result reader: fresh endpoint comparison and separate parent aggregation."""
from itertools import combinations
import time

from scoped_source.hz_request_store import compact, decode, digest, identity, require, sha
from scoped_source.factored_io import read
from scoped_source.result_stream import (PAGE_ROWS, PAGE_BYTES, INDEX_BYTES, MAX_PAGES, context,
    row, summary, candidate_identity, plan_context)


class Reader:
    def __init__(self, root, ref, candidates, work):
        self.root, self.work = root, work
        require(type(ref) is dict and ref.get('file') == 'result/index.json', 'fixed result index path')
        self.ref = dict(ref)
        raw = self.member(ref, INDEX_BYTES)
        obj = decode(raw)
        require(type(obj) is dict and compact(obj) == raw and set(obj) == {'schema', 'context', 'candidates', 'pages', 'summary'}
                and obj['schema'] == 'SHARED_RESULT_STREAM_V1', 'result index schema')
        self.context = context(obj['context'])
        require(obj['candidates'] == candidate_identity(candidates), 'result candidate identity')
        self.summary = summary(obj['summary'])
        self.pages = obj['pages']
        require(type(self.pages) is list and 0 < len(self.pages) <= MAX_PAGES and
                len(self.pages) == (self.summary['required']+PAGE_ROWS-1)//PAGE_ROWS, 'complete result page roster')
        for i, ref in enumerate(self.pages):
            require(type(ref) is dict and set(ref) == {'file', 'bytes', 'sha256', 'start', 'count'}
                    and ref['file'] == f'result/{i:04d}.json' and type(ref['start']) is int and ref['start'] == i*PAGE_ROWS
                    and type(ref['count']) is int and ref['count'] == min(PAGE_ROWS, self.summary['required']-i*PAGE_ROWS),
                    'ordered complete result pages')
            self.check_ref({k:ref[k] for k in ('file', 'bytes', 'sha256')}, PAGE_BYTES)
        self.used = False
        self.pages_read = 0

    @staticmethod
    def check_ref(ref, cap):
        require(type(ref) is dict and set(ref) == {'file', 'bytes', 'sha256'} and type(ref['bytes']) is int
                and 0 < ref['bytes'] <= cap, 'bounded result member')
        sha(ref['sha256'])

    def member(self, ref, cap):
        self.check_ref(ref, cap)
        self.work.tick()
        raw = read(self.root, ref['file'], cap, ref['sha256'])
        require(len(raw) == ref['bytes'], 'result member length')
        self.work.tick()
        return raw

    def rows(self, *, expire_after_page=False):
        require(not self.used, 'one full result traversal')
        self.used = True
        for i, ref in enumerate(self.pages):
            raw = self.member({k:ref[k] for k in ('file', 'bytes', 'sha256')}, PAGE_BYTES)
            obj = decode(raw)
            require(type(obj) is dict and compact(obj) == raw and set(obj) == {'schema', 'context', 'ordinal', 'start', 'rows'}
                    and obj['schema'] == 'SHARED_RESULT_PAGE_V1' and obj['context'] == identity(self.context)
                    and type(obj['ordinal']) is int and obj['ordinal'] == i and type(obj['start']) is int
                    and obj['start'] == ref['start'] and type(obj['rows']) is list and len(obj['rows']) == ref['count'],
                    'result page fields/order')
            self.pages_read += 1
            for item in obj['rows']:
                self.work.tick()
                yield row(item)
            if expire_after_page:
                while time.monotonic() < self.work.deadline:
                    time.sleep(min(.01, max(0., self.work.deadline-time.monotonic())))
            self.work.tick()


class Comparator:
    """A comparison sink, never an alternative to complete source/endpoint replay."""
    def __init__(self, root, ref, method, invocation, forest_index, candidates, work):
        from scoped_source.direct_method import validate, paged
        self.method = validate(method)
        require(paged(self.method), 'paged comparison opt-in')
        self.reader = Reader(root, ref, candidates, work)
        self.work, self.invocation, self.forest_index = work, sha(invocation), sha(forest_index)
        self.iterator = None
        self.closed = False

    def bind(self, forest, plan, work):
        require(work is self.work and self.iterator is None and not self.closed, 'one replay result binding')
        expected = plan_context(forest, plan, self.method, self.invocation, self.forest_index, work)
        require(self.reader.context == expected, 'independent plan result context')
        self.iterator = self.reader.rows()

    def emit(self, value):
        require(self.iterator is not None and not self.closed, 'bound result comparison')
        saved = next(self.iterator, None)
        require(saved == value, 'independently recomputed result row')
        self.work.tick()

    def finish(self, value):
        require(self.iterator is not None and not self.closed and next(self.iterator, None) is None,
                'no unverified result suffix')
        require(self.reader.summary == value, 'independently recomputed result summary')
        self.work.tick()
        self.closed = True
        return dict(value)


def aggregate(root, ref, candidates, request, expected_context, work, *, expire_after_page=False):
    """Parent roster/sign check. Endpoint roster digest was bound by fresh replay."""
    reader = Reader(root, ref, candidates, work)
    require(reader.context == context(expected_context), 'parent result context')
    require(type(request['experts']) is int and 2 <= request['experts'] <= 8 and
            type(request['classes']) is int and 2 <= request['classes'] <= 10 and
            type(request['label']) is int and 0 <= request['label'] < request['classes'], 'source request roster')
    expected = ((list(pair), c) for pair in combinations(range(request['experts']), 2)
                for c in range(request['classes']) if c != request['label'])
    count = positive = missing = 0
    for value in reader.rows(expire_after_page=expire_after_page):
        require((value['pair'], value['other']) == next(expected, None), 'source-derived full property roster')
        count += 1
        positive += value['positive']
        missing += value['lower'] is None
    require(next(expected, None) is None and count == reader.summary['required'] and positive == reader.summary['positive']
            and missing <= reader.summary['missing_endpoints'] <= 2*missing, 'parent full result aggregate')
    work.tick()
    return dict(reader.summary)
