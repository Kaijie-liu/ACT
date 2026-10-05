"""Default-off authenticated protocol5 readonly numeric owner restoration.

Only trusted, independently hash-anchored checkpoints may be loaded. This is
not a safe unpickler for untrusted files and issues no HZ/source receipt.
All fields still need the unchanged complete source and closed-root checks.
"""
import hashlib
import json
import math
import pickle
import time
import numpy as np


class _Reader:
    def __init__(self,file):
        self.file=file;self.digest=hashlib.sha256();self.size=0
    def read(self,n=-1):
        data=self.file.read(n);self.digest.update(data);self.size+=len(data);return data
    def readline(self,n=-1):
        data=self.file.readline(n);self.digest.update(data);self.size+=len(data);return data
    def readinto(self,buffer):
        n=self.file.readinto(buffer)
        if n:self.digest.update(memoryview(buffer)[:n]);self.size+=n
        return n


class _OwnedUnpickler(pickle.Unpickler):
    def __init__(self,reader,pool):
        super().__init__(reader);self.pool=pool;self.owners={};self.headers=0
        self.copied_entries=0;self.copied_bytes=0;self.restored_arrays=0
        self.image=hashlib.sha256()

    def find_class(self,module,name):
        if module in ('numpy._core.numeric','numpy.core.numeric') and name=='_frombuffer':
            # Resolve the existing local constructor too: no invented reducer.
            original=super().find_class(module,name)
            if not callable(original):raise ValueError('unavailable NumPy protocol5 reducer')
            return self.frombuffer
        return super().find_class(module,name)

    def frombuffer(self,data,dtype,shape,order):
        self.pool.charge('owned_decode_array_header',32+4*len(shape) if type(shape) is tuple else 32)
        dtype=np.dtype(dtype)
        if (type(data) not in (bytes,bytearray) or dtype.hasobject or dtype.kind not in 'biuf'
                or not dtype.isnative or type(shape) is not tuple or len(shape)>32
                or any(type(n) is not int or n<0 for n in shape) or order not in ('C','F')):
            raise ValueError('unsupported exact serialized numeric owner/layout')
        n=math.prod(shape)
        if n>64_000_000 or n*dtype.itemsize!=len(data):
            raise ValueError('serialized numeric payload does not fill its exact frame')
        self.headers+=1
        if type(data) is bytearray:
            # This is the SAME already-registered writable pickle owner path.
            # No new full writable-array copy or external-owner acceptance.
            return np.frombuffer(data,dtype=dtype).reshape(shape,order=order)
        identity=id(data)
        if identity not in self.owners:
            self.pool.charge('owned_decode_readonly_copy_and_bits',64+9*n)
            flat=np.empty(n,dtype=dtype)
            flat[:]=np.frombuffer(data,dtype=dtype)
            if memoryview(flat).cast('B')!=memoryview(data):
                raise ValueError('readonly owner copy changed serialized numeric bits')
            flat.flags.writeable=False
            self.owners[identity]=(data,flat,dtype)
            self.copied_entries+=n;self.copied_bytes+=len(data)
            self.image.update(json.dumps([dtype.str,n,hashlib.sha256(data).hexdigest()]).encode())
        source,flat,owner_dtype=self.owners[identity]
        if source is not data or owner_dtype!=dtype:
            raise ValueError('incompatible serialized readonly backing dtype alias')
        out=flat.reshape(shape,order=order)
        if out.flags.writeable or flat.base is not None or (n and not np.shares_memory(out,flat)):
            raise ValueError('readonly owner restoration lost physical layout/alias')
        self.restored_arrays+=1
        return out


def load(file,*,expected_sha256,pool,enabled=False):
    if not enabled:return None
    if type(expected_sha256) is not str or len(expected_sha256)!=64 or file.tell()!=0:
        raise ValueError('independent complete checkpoint hash and initial stream required')
    started=time.monotonic();h=hashlib.sha256();size=0
    while block:=file.read(1024*1024):h.update(block);size+=len(block)
    if h.hexdigest()!=expected_sha256:raise ValueError('checkpoint differs before decoding')
    file.seek(0);reader=_Reader(file);decoder=_OwnedUnpickler(reader,pool)
    try:
        value=decoder.load()
        if reader.read(1) or reader.size!=size or reader.digest.hexdigest()!=expected_sha256:
            raise ValueError('checkpoint changed during decoding or has trailing bytes')
        count=len(decoder.owners)
        pool.charge('owned_decode_temporary_root_retirement',64+64*count)
        report=dict(schema='c41_authenticated_readonly_numeric_decode_v1',
            checkpoint_sha256=expected_sha256,checkpoint_bytes=size,
            numeric_reducer_calls=decoder.headers,readonly_array_reducer_calls=decoder.restored_arrays,
            readonly_backing_groups=count,copied_numeric_entries=decoder.copied_entries,
            copied_numeric_bytes=decoder.copied_bytes,readonly_source_image_sha256=decoder.image.hexdigest(),
            encoded_object_aliases_preserved_by_pickle_memo=True,
            encoded_shared_readonly_backings_preserved=True,all_restored_array_bits_and_readonly_flags_checked=True,
            original_pre_pickle_view_aliases_not_claimed=True,source_binding_still_required=True,
            whole_C34_LIVE_gate_proved=False,formal_gain=0)
    finally:
        # Standard pickle memo and the decoder cache otherwise retain raw
        # serialization buffers. Neither table is part of the returned state.
        decoder.owners.clear();decoder.memo.clear()
    report.update(decoder_temporary_roots_released=True,elapsed_s=time.monotonic()-started)
    return value,report
