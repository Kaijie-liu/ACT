"""Content-bound immutable parse storage, with fresh containers AND Fractions.

The cache stores no source acceptance, route decision or lower bound. Complete
current JSON content is rebound on every call. Hash collisions fail closed.
Serialized-byte/cell/entry limits bound retention, not total process memory;
an inherited outer watchdog remains necessary before production integration.
"""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
import time

from source_enclosure.format import unpack as reference_unpack

POLICY = 'EXACT_SOURCE_PARSE_REUSE_R1'
LIMITS = {'entries': 64, 'payload_bytes': 64 * 2**20, 'cells': 2_000_000}


def digest(payload): return hashlib.sha256(payload).hexdigest()


def scope_key(scope):
    if (type(scope) is not dict or set(scope) != {'invocation', 'source_sha256'}
            or type(scope['invocation']) is not str or not 0 < len(scope['invocation']) <= 100
            or type(scope['source_sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}', scope['source_sha256']) is None):
        raise ValueError('explicit invocation and complete request/source identity required')
    return scope['invocation'], scope['source_sha256']


def snapshot(value, tick, depth=0):
    """JSON-only defensive copy before canonical serialization and parsing.

    No tuple->list/int-key->string coercion, custom encoder, or mutable alias
    authority. No promise of concurrent mutation safety across native threads.
    """
    tick()
    if depth > 64: raise ValueError('source snapshot nesting limit')
    if type(value) is dict:
        if any(type(k) is not str for k in value): raise ValueError('source keys must be strings')
        return {k: snapshot(v, tick, depth+1) for k, v in value.items()}
    if type(value) is list: return [snapshot(v, tick, depth+1) for v in value]
    if type(value) in (str, int, bool) or value is None: return value
    if type(value) is float and math.isfinite(value): return value
    raise ValueError('canonical finite JSON source required')


def freeze(value, tick):
    tick()
    if type(value) is Fraction:
        # Fraction's private slots can be mutated in Python. Never share a
        # Fraction object with a consumer or retain one in a cache entry.
        return ('rational', value.numerator, value.denominator), 1
    if type(value) in (tuple, list):
        items=[]; cells=1
        for v in value:
            item,n=freeze(v,tick);items.append(item);cells+=n
        return ('tuple' if type(value) is tuple else 'list', tuple(items)), cells
    if type(value) is dict:
        items=[];cells=1
        for k,v in value.items():
            a,na=freeze(k,tick);b,nb=freeze(v,tick);items.append((a,b));cells+=na+nb
        return ('dict',tuple(items)), cells
    if type(value) in (str,int): return ('atom',value), 1
    raise ValueError('unexpected reference parser result')


def thaw(value, tick):
    tick();kind=value[0]
    if kind=='rational': return Fraction(value[1],value[2])
    if kind=='atom': return value[1]
    if kind=='dict': return {thaw(k,tick):thaw(v,tick) for k,v in value[1]}
    if kind=='list': return [thaw(v,tick) for v in value[1]]
    if kind=='tuple': return tuple(thaw(v,tick) for v in value[1])
    raise ValueError('invalid immutable parse')


@dataclass(frozen=True, slots=True)
class Entry:
    scope: tuple
    payload: bytes
    value: tuple
    cells: int


class SourceParser:
    def __init__(self, scope, *, enabled, tick, limits=None):
        self.__scope=scope_key(scope)
        if type(enabled) is not bool or not callable(tick): raise ValueError('explicit mode/deadline required')
        self.__enabled=enabled;self.__tick=tick
        self.__limits=dict(LIMITS if limits is None else limits)
        if (set(self.__limits)!=set(LIMITS) or
                any(type(v) is not int or not 1<=v<=LIMITS[k] for k,v in self.__limits.items())):
            raise ValueError('bounded parse retention')
        self.__items=OrderedDict();self.__bytes=self.__cells=0
        self.__closed=False;self.__busy=False
        self.__counts=dict(lookups=0,hits=0,parses=0,evictions=0,oversized=0,failures=0,
                           peak_entries=0,peak_payload_bytes=0,peak_cells=0)
        self.__seconds={k:0. for k in ('snapshot','identity_lookup','reference_parse','freeze','copy','retention','total')}

    def _timed(self,name,fn):
        started=time.monotonic()
        try:return fn()
        finally:self.__seconds[name]+=time.monotonic()-started

    def unpack(self,state,*,scope):
        began=time.monotonic();succeeded=False
        if self.__closed or scope_key(scope)!=self.__scope:raise ValueError('closed/wrong invocation scope')
        if self.__busy:raise ValueError('one parser owner; no concurrent/reentrant cache access')
        self.__busy=True
        try:
            self.__tick();self.__counts['lookups']+=1
            if not self.__enabled:
                self.__counts['parses']+=1
                result=self._timed('reference_parse',lambda:reference_unpack(state))
                self.__tick();succeeded=True;return result
            def capture():
                obj=snapshot(state,self.__tick)
                payload=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                self.__tick();return obj,payload
            current,payload=self._timed('snapshot',capture)
            def lookup():
                key=(self.__scope,POLICY,digest(payload))
                item=self.__items.get(key)
                if item is not None:
                    if item.scope!=self.__scope or item.payload!=payload:
                        raise ValueError('source identity collision/transplanted entry')
                    self.__items.move_to_end(key)
                self.__tick();return key,item
            key,item=self._timed('identity_lookup',lookup)
            if item is not None:self.__counts['hits']+=1
            else:
                self.__counts['parses']+=1
                parsed=self._timed('reference_parse',lambda:reference_unpack(current))
                self.__tick()
                value,cells=self._timed('freeze',lambda:freeze(parsed,self.__tick))
                item=Entry(self.__scope,payload,value,cells)
                self._timed('retention',lambda:self._retain(key,item))
            result=self._timed('copy',lambda:thaw(item.value,self.__tick))
            self.__tick();succeeded=True;return result
        finally:
            if not succeeded:self.__counts['failures']+=1
            self.__seconds['total']+=time.monotonic()-began
            self.__busy=False

    def _retain(self,key,item):
        self.__tick()
        if len(item.payload)>self.__limits['payload_bytes'] or item.cells>self.__limits['cells']:
            self.__counts['oversized']+=1;return
        while self.__items and (len(self.__items)>=self.__limits['entries'] or
                self.__bytes+len(item.payload)>self.__limits['payload_bytes'] or
                self.__cells+item.cells>self.__limits['cells']):
            _,old=self.__items.popitem(last=False);self.__bytes-=len(old.payload);self.__cells-=old.cells
            self.__counts['evictions']+=1;self.__tick()
        self.__items[key]=item;self.__bytes+=len(item.payload);self.__cells+=item.cells
        for k,v in [('peak_entries',len(self.__items)),('peak_payload_bytes',self.__bytes),('peak_cells',self.__cells)]:
            self.__counts[k]=max(self.__counts[k],v)

    def stats(self):
        parts=sum(v for k,v in self.__seconds.items() if k!='total')
        return {**self.__counts,'enabled':self.__enabled,'policy':POLICY,
                'scope':list(self.__scope),'limits':dict(self.__limits),
                'live_entries':len(self.__items),'live_payload_bytes':self.__bytes,'live_cells':self.__cells,
                'seconds':dict(self.__seconds),'other_seconds':self.__seconds['total']-parts,
                'closed':self.__closed,'results_or_bounds_cached':False}

    def close(self):
        self.__items.clear();self.__bytes=self.__cells=0;self.__closed=True
