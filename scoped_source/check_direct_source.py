"""Independent direct intake plus existing V3 enclosure mathematics, stdlib.

Does not import/call the source builder; no old forest JSON reconstruction.
"""
from scoped_source.check_scaled_references import Checker as Equations, CheckedScaledForest
from scoped_source.direct_source_format import intake
from scoped_source.hz_request_store import POLICY,identity,require,sha
from scoped_source.tensor_rows import TensorRows
from scoped_source.forest_endpoints import frozen


class Checker(Equations):
    def __init__(self,raw,root,source,expected,work):
        self.work=work; d,nodes,views,scales,counts=intake(raw,expected,work)
        self.initialize_direct(d,nodes,views,scales,counts,root,source,work,len(raw))

    def initialize_direct(self,d,nodes,views,scales,counts,root,source,work,forest_bytes,reader=None):
        self.work=work
        require(d['source']==sha(source) and type(d['frame']) is int and d['frame']==1
                and d['cut'] is None and d['policy']==dict(POLICY),'direct source/frame/full/policy')
        if reader is not None:
            from scoped_source.shared_source import Snapshot
            require(type(reader) is Snapshot and reader.work is work, 'closed shared source replay')
        self.reader=TensorRows(root,source,deadline=work.deadline) if reader is None else reader
        require(self.reader.request['experts']<=8 and d['request']==identity(self.reader.request),'direct request binding')
        require(type(d['factors']) is list and 0<len(d['factors'])<=POLICY['factors']
                and all(type(f) is list for f in d['factors']),'direct factors')
        self.factors=tuple(tuple(f) for f in d['factors'])
        require(all(len(f)==4 and type(f[0]) is str and len(f[0])<=200 and f[1] in ('continuous','binary')
                    and type(f[2]) is int and -1<=f[2]<self.reader.request['experts'] and type(f[3]) is int for f in self.factors),'factor fields')
        require(len({f[0] for f in self.factors})==len(self.factors) and len(scales)==len(self.factors),'factor names/scales')
        self.scales=scales; self.views=views
        for i,v in enumerate(views):
            work.visit('row_visits')
            require(type(v['name']) is str and type(v['scope']) is int and
                    (v['parent'] is None or type(v['parent']) is int and 0<=v['parent']<i),'view ancestry')
            require(type(v['factors']) is list and len(v['factors'])==2 and all(type(j) is int for j in v['factors']),'view factor interval')
        require(len({v['name'] for v in views})==len(views),'unique views')
        self.nodes=nodes; self.local_norms=[]; self.norms=[]; depths=[]; active=set(); nnz=refs=0
        for i,(owner,local,parents) in enumerate(nodes):
            work.visit('row_visits'); require(0<=owner<len(views),'node owner')
            norm=self.local_norm_start()
            for entries,size in ((local,len(self.factors)),(parents,i)):
                last=-1
                for j,q in entries:
                    work.visit('coefficient_visits'); require(last<j<size and q!=0,'coefficient order/range/zero'); last=j
            for j,q in local: norm=self.local_norm_add(norm,j,q); active.add(j)
            nnz+=len(local); refs+=len(parents)
            require(nnz<=POLICY['nnz'] and refs<=65536 and len(active)<=POLICY['active'],'local/reference/active capacity')
            depth=1+max((depths[j] for j,_ in parents),default=0); require(depth<=POLICY['depth'],'node depth'); depths.append(depth)
            self.local_norms.append(self.local_norm_finish(norm)); self.norms.append(None)
        self.fid=self.vid=self.nid=0; self.receipt=d['source_receipt']; self.roles=d['roles']; self.cut=None
        self.stats=dict(nodes=len(nodes),local_nnz=nnz,references=refs,factors=len(self.factors),views=len(views),forest_bytes=forest_bytes,**counts)
        self.visible=[]; self.depths=[]


class CheckedDirectForest(CheckedScaledForest):
    __slots__=('_root',)

    def __init__(self,raw,root,source,*,expected,work,enabled=False):
        require(enabled is True,'direct checker is opt-in')
        checker=Checker(raw,root,source,expected,work); result=checker.run()
        self.identity=sha(expected); self.source=sha(source); self.request=result['request']
        self.declaration=frozen(checker.reader.request); self.factors=checker.factors
        self.nodes=checker.nodes; self.views=frozen(checker.views); self.roles=frozen(checker.roles)
        self.norms=tuple(checker.norms); self.scales=checker.scales; self.receipt=frozen(result)
        self.complete=True; self.session=work; self._root=str(root)
        self.factor_identity=identity(dict(factors=self.factors,scales=[str(q) for q in self.scales]))
        self._sealed=True; work.tick()

    def seal(self):
        # Source files are not assumed immutable merely because graph values are.
        from scoped_source.wire_forest import SourceRows
        SourceRows(self._root,self.source,self.session).seal()
