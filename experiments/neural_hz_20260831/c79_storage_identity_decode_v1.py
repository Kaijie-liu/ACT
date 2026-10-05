"""One authenticated checkpoint's explicit original tensor-storage identities.

No equal-value deduplication; distinct Tensor objects/views remain distinct.
Not a safe unpickler for untrusted bytes. C41 numeric-owner checks stay intact.
"""
import hashlib
import io
import json
import pickletools
import time
import torch
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import _Reader,_OwnedUnpickler

INTEGER={'BININT','BININT1','BININT2','LONG1','LONG4'}


def header(raw,pool):
    if type(raw) is not bytes:raise ValueError('literal original storage bytes required')
    pool.charge('c79_complete_storage_header_hash_and_comparison',4096+2*((len(raw)+7)//8))
    stream=io.BytesIO(raw[:4096]);parts=[]
    for _ in range(5):
        ops=[(op.name,arg) for op,arg,_ in pickletools.genops(stream)]
        if not ops or ops[0]!=('PROTO',2):raise ValueError('registered legacy protocol2 storage header required')
        parts.append([(name,arg) for name,arg in ops if name not in ('PROTO','BINPUT','LONG_BINPUT')])
    if (parts[0]!=[('LONG1',torch.serialization.MAGIC_NUMBER),('STOP',None)]
            or parts[1]!=[('BININT2',torch.serialization.PROTOCOL_VERSION),('STOP',None)]):
        raise ValueError('original legacy storage magic/version differs')
    expected=[('EMPTY_DICT',None),('MARK',None),('BINUNICODE','protocol_version'),
        ('BININT2',torch.serialization.PROTOCOL_VERSION),('BINUNICODE','little_endian'),('NEWTRUE',None),
        ('BINUNICODE','type_sizes'),('EMPTY_DICT',None),('MARK',None),('BINUNICODE','short'),
        ('BININT1',2),('BINUNICODE','int'),('BININT1',4),('BINUNICODE','long'),('BININT1',4),
        ('SETITEMS',None),('SETITEMS',None),('STOP',None)]
    if parts[2]!=expected:raise ValueError('registered CPU legacy system header differs')
    p=parts[3]
    if (len(p)!=10 or [a for a,_ in p[:5]]!=['MARK','BINUNICODE','GLOBAL','BINUNICODE','BINUNICODE']
            or p[0][1] is not None or p[1][1]!='storage' or p[4][1]!='cpu'
            or p[5][0] not in INTEGER or type(p[5][1]) is not int or not 0<=p[5][1]<=64_000_000
            or p[6:]!=[('NONE',None),('TUPLE',None),('BINPERSID',None),('STOP',None)]):
        raise ValueError('one ordinary explicit CPU typed storage required')
    dtype_name=p[2][1];key=p[3][1];numel=p[5][1]
    if (type(key) is not str or not key.isascii() or not key.isdecimal() or not 0<len(key)<=64
            or dtype_name not in {'torch '+n for n in ('ByteStorage','CharStorage','ShortStorage',
                'IntStorage','LongStorage','HalfStorage','FloatStorage','DoubleStorage','BoolStorage','BFloat16Storage')}):
        raise ValueError('explicit original storage identity/dtype required')
    if parts[4]!=[('EMPTY_LIST',None),('BINUNICODE',key),('APPEND',None),('STOP',None)]:
        raise ValueError('complete storage key table differs')
    dtype=getattr(torch,dtype_name.split(' ',1)[1]).dtype
    nbytes=numel*torch._utils._element_size(dtype);at=stream.tell()
    if (len(raw)!=at+8+nbytes or int.from_bytes(raw[at:at+8],'little',signed=True)!=numel):
        raise ValueError('complete raw storage extent differs from its identity header')
    return dict(original_storage_key=key,storage_type=dtype_name,numel=numel,nbytes=nbytes,
        header_bytes=at,complete_raw_sha256=hashlib.sha256(raw).hexdigest()),dtype


class AliasUnpickler(_OwnedUnpickler):
    def __init__(self,reader,pool):
        super().__init__(reader,pool);self.storage_identities={};self.storage_calls=0;self.storage_reuses=0
    def find_class(self,module,name):
        original=super().find_class(module,name)
        if module=='torch.storage' and name=='_load_from_bytes':
            def storage(raw):
                info,dtype=header(raw,self.pool);key=info['original_storage_key'];self.storage_calls+=1
                if key in self.storage_identities:
                    old_info,old_raw,value=self.storage_identities[key]
                    if info!=old_info or raw!=old_raw:raise ValueError('one original storage identity has conflicting complete bytes')
                    self.storage_reuses+=1;return value
                value=original(raw)
                if (value.dtype!=dtype or value._size()!=info['numel']
                        or value._untyped_storage.nbytes()!=info['nbytes']
                        or str(value._untyped_storage.device)!='cpu'):
                    raise ValueError('original loader did not restore the complete declared storage')
                self.storage_identities[key]=(info,raw,value)
                return value
            return storage
        return original


def load(file,*,expected_sha256,pool,enabled=False):
    if not enabled:return None
    if type(expected_sha256) is not str or len(expected_sha256)!=64 or file.tell()!=0:
        raise ValueError('independent complete checkpoint hash and initial stream required')
    started=time.monotonic();h=hashlib.sha256();size=0
    while block:=file.read(1024*1024):h.update(block);size+=len(block)
    if h.hexdigest()!=expected_sha256:raise ValueError('checkpoint differs before decoding')
    file.seek(0);reader=_Reader(file);decoder=AliasUnpickler(reader,pool)
    try:
        value=decoder.load()
        if reader.read(1) or reader.size!=size or reader.digest.hexdigest()!=expected_sha256:
            raise ValueError('checkpoint changed during decoding or has trailing bytes')
        count=len(decoder.owners)
        pool.charge('owned_decode_temporary_root_retirement',64+64*count)
        pool.charge('c79_storage_identity_map_retirement',64+64*len(decoder.storage_identities))
        report=dict(schema='c79_authenticated_original_tensor_storage_decode_v1',
            checkpoint_sha256=expected_sha256,checkpoint_bytes=size,
            numeric_reducer_calls=decoder.headers,readonly_array_reducer_calls=decoder.restored_arrays,
            readonly_backing_groups=count,copied_numeric_entries=decoder.copied_entries,
            copied_numeric_bytes=decoder.copied_bytes,readonly_source_image_sha256=decoder.image.hexdigest(),
            encoded_object_aliases_preserved_by_pickle_memo=True,
            encoded_shared_readonly_backings_preserved=True,all_restored_array_bits_and_readonly_flags_checked=True,
            tensor_storage_calls=decoder.storage_calls,tensor_storage_reuses=decoder.storage_reuses,
            original_tensor_storage_identities=len(decoder.storage_identities),
            original_tensor_storage_provenance=[v[0] for _,v in sorted(decoder.storage_identities.items())],
            tensor_storage_map_scoped_to_one_authenticated_checkpoint=True,
            distinct_tensor_objects_and_geometry_preserved=True,distinct_equal_storage_keys_never_merged=True,
            source_binding_still_required=True,whole_LIVE_gate_proved=False,formal_gain=0)
    finally:
        decoder.owners.clear();decoder.memo.clear();decoder.storage_identities.clear()
    report.update(decoder_temporary_roots_released=True,elapsed_s=time.monotonic()-started)
    return value,report
