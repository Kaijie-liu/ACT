"""Default-off direct captured-source construction; no native proposal calls."""
from scoped_source.scaled_references import Builder as ScaledBuilder
from scoped_source.tensor_rows import TensorRows
from scoped_source.hz_request_store import require
from scoped_source.direct_source_format import encode


class Builder(ScaledBuilder):
    def serialize(self,roles,cut):
        raw=encode(self,roles,cut)
        return raw,dict(native_seconds=self.native_seconds,native_coordinates=self.native_coordinates)


def build(root,source,*,work,enabled=False):
    require(enabled is True,'direct source is opt-in')
    reader=TensorRows(root,source,deadline=work.deadline)
    require(reader.request['experts']<=8,'unchanged expert limit')
    return Builder(reader,work).finish(None)
