"""New method binding over unchanged bounded result serialization/checks."""
from scoped_source.hz_request_store import require, sha
from scoped_source.native_method import validate
from scoped_source.result_stream import Writer as BaseWriter, candidate_identity
from scoped_source.check_result_stream import Comparator as BaseComparator, Reader


class Writer(BaseWriter):
    def __init__(self, put, method, invocation, forest_index, candidates, work):
        self.method = validate(method)
        self.put, self.work, self.fault = put, work, ''
        self.invocation, self.forest_index = sha(invocation), sha(forest_index)
        self.candidates = candidate_identity(candidates)
        self.context = None
        self.page, self.pages = [], []
        self.count = self.positive = self.missing_rows = 0
        self.ref = None; self.closed = False


class Comparator(BaseComparator):
    def __init__(self, root, ref, method, invocation, forest_index, candidates, work):
        self.method = validate(method)
        self.reader = Reader(root, ref, candidates, work)
        self.work, self.invocation, self.forest_index = work, sha(invocation), sha(forest_index)
        self.iterator = None; self.closed = False
