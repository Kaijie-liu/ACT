"""Prepaid exact compact JSON bytes, published once without re-encoding."""
import hashlib
import json
import os
from pathlib import Path
import secrets


class JsonAllowance:
    """A nonrefundable, single-use serialization allowance paid at creation.

    Creation may precede the payload (notably terminal success/failure records).
    The byte limit includes1024 bytes of fixed reporting overhead.  Failure
    consumes the allowance; it never silently borrows unused category capacity.
    """
    def __init__(self, pool, limit, label):
        if type(limit) is not int or limit < 1024 or type(label) is not str:
            raise ValueError('fixed positive complete JSON reservation required')
        pool.charge(label, limit)
        self.limit, self.label, self.used = limit, label, False

    def write(self, path, payload):
        if self.used:
            raise ValueError('exclusive prepaid JSON allowance already consumed')
        self.used = True
        encoded = (json.dumps(payload, sort_keys=True, separators=(',', ':'),
                              allow_nan=False, ensure_ascii=True)+'\n').encode('utf-8')
        if len(encoded)+1024 > self.limit:
            raise MemoryError('actual exact compact JSON bytes exceed prepaid reservation')
        # This exact immutable object, not a second serialization, is written.
        publish_bytes(path, encoded)
        return dict(file=Path(path).name, encoded_bytes=len(encoded), overhead_bytes=1024,
                    prepaid_bytes=self.limit, sha256=hashlib.sha256(encoded).hexdigest(),
                    exact_checked_bytes_published=True, encoding='sorted_compact_ascii_JSON_newline')


def publish_bytes(output, encoded):
    """Exclusive durable publication of an already-qualified immutable payload."""
    output = Path(output)
    if type(encoded) is not bytes:
        raise ValueError('immutable already-encoded bytes required')
    if os.path.lexists(output):
        raise FileExistsError('refusing to overwrite existing evidence: '+str(output))
    temporary = output.with_name('.'+output.name+'.tmp.'+str(os.getpid())+'.'+secrets.token_hex(8))
    descriptor = None
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('zero-byte evidence write')
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, output)
        published = True
        parent = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            if not published:
                raise
