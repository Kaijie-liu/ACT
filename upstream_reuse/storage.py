"""Bounded request-local JSON decode reuse; callers never receive cache aliases."""
from collections import OrderedDict
from types import MappingProxyType
from portable_proof.runtime import digest,strict_json

POLICY='UPSTREAM_DECODE_COPY_V1'
LIMITS={'entries':8,'payload_bytes':64*1024*1024,'nodes':2_000_000}


class SourceCache:
    def __init__(self,scope,*,enabled=False,tick=lambda:None,measure=None,limits=None):
        if not isinstance(scope,str) or not scope or type(enabled) is not bool:raise ValueError('scope/cache flag')
        self.scope=scope;self.enabled=enabled;self.tick=tick
        self.measure=measure or (lambda name,fn,*a,**kw:fn(*a,**kw))
        self.limits=dict(LIMITS if limits is None else limits)
        if set(self.limits)!=set(LIMITS) or any(type(v)is not int or v<1 for v in self.limits.values()):raise ValueError('limits')
        self._items=OrderedDict();self._bytes=0;self._nodes=0
        self.counts=dict(lookups=0,hits=0,decodes=0,evictions=0,oversized=0,peak_payload_bytes=0,peak_nodes=0)

    def _freeze(self,value,counter):
        counter[0]+=1
        if counter[0]%1024==0:self.tick()
        if counter[0]>self.limits['nodes']:raise OverflowError('decode cache node limit')
        if isinstance(value,dict):return MappingProxyType({k:self._freeze(v,counter) for k,v in value.items()})
        if isinstance(value,list):return tuple(self._freeze(v,counter) for v in value)
        return value

    def _copy(self,value,counter):
        counter[0]+=1
        if counter[0]%1024==0:self.tick()
        if isinstance(value,MappingProxyType):return {k:self._copy(v,counter) for k,v in value.items()}
        if isinstance(value,tuple):return [self._copy(v,counter) for v in value]
        return value

    def load(self,path,expected_sha,*,scope):
        self.tick()
        if scope!=self.scope:raise ValueError('cross-request source cache')
        raw=self.measure('source_read_bytes',path.read_bytes)
        if self.measure('source_hash',digest,raw)!=expected_sha:raise ValueError('source bytes do not match current reference')
        key=(scope,POLICY,expected_sha);self.counts['lookups']+=1
        if self.enabled and key in self._items:
            stored_scope,payload,value,nodes=self._items[key]
            if stored_scope!=scope or payload!=raw:raise ValueError('cache collision/pollution')
            self._items.move_to_end(key);self.counts['hits']+=1
            result=self.measure('source_copy',self._copy,value,[0]);self.tick();return result
        value=self.measure('source_decode',strict_json,raw);self.counts['decodes']+=1
        if self.enabled:
            if len(raw)>self.limits['payload_bytes']:self.counts['oversized']+=1
            else:
                count=[0]
                try:frozen=self.measure('source_freeze',self._freeze,value,count)
                except OverflowError:self.counts['oversized']+=1
                else:
                    while self._items and (len(self._items)>=self.limits['entries'] or
                            self._bytes+len(raw)>self.limits['payload_bytes'] or self._nodes+count[0]>self.limits['nodes']):
                        _,(_,old,_,n)=self._items.popitem(last=False);self._bytes-=len(old);self._nodes-=n;self.counts['evictions']+=1
                    self._items[key]=(scope,raw,frozen,count[0]);self._bytes+=len(raw);self._nodes+=count[0]
                    self.counts['peak_payload_bytes']=max(self.counts['peak_payload_bytes'],self._bytes)
                    self.counts['peak_nodes']=max(self.counts['peak_nodes'],self._nodes)
        self.tick();return value # miss returns decoded original; stored representation is separate and immutable

    def clear(self):self._items.clear();self._bytes=0;self._nodes=0

    def stats(self):return {**self.counts,'policy':POLICY,'enabled':self.enabled,'limits':self.limits,
                           'live_entries':len(self._items),'live_payload_bytes':self._bytes,'live_nodes':self._nodes}
