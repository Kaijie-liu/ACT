"""Independent shared-source equations, never imports/calls shared_build."""
from fractions import Fraction as F

from scoped_source.check_direct_source import Checker as Base, CheckedDirectForest
from scoped_source.check_source_dag import nearest, upper
from scoped_source.hz_request_store import identity, require, sha
from scoped_source.forest_endpoints import frozen
from scoped_source.shared_source import Snapshot, SourceLocal
from scoped_source.shared_format import intake


class Checker(Base):
    def __init__(self, packet, root, source, expected, work):
        snapshot=Snapshot(root,source,work)
        d,nodes,views,scales,counts=intake(packet,expected,work,snapshot)
        self.initialize_direct(d,nodes,views,scales,counts,root,source,work,packet.size,reader=snapshot)

    def node(self,owner,ni,*,refs=()):
        # First-affine nodes use the dedicated branch below. No other source
        # operator may smuggle a numerically equal (e.g. zero) source alias.
        require(type(ni) is int and 0<=ni<len(self.nodes)
                and type(self.nodes[ni][1]) is not SourceLocal, 'source alias only at first affine')
        return super().node(owner,ni,refs=refs)

    def affine(self,parent,layer,stream):
        rows=self.views[parent]['output']
        if rows!=self.views[0]['output']: return super().affine(parent,layer,stream)
        name=f'{layer.graph}/{layer.index}'; scope=-1 if layer.graph=='router' else int(layer.graph[6:])
        i,v=self.view(name,scope,parent,'Linear'); basis=self.fid
        require(len(v['output'])==layer.outputs and not v['eq'] and not v['ine'], 'shared affine roster')
        pending=[]
        for k,(stored,n) in enumerate(v['output']):
            row=next(stream); key=(layer.graph,layer.index,k)
            require((row.graph,row.layer,row.row)==key and type(n) is int and n==self.nid and n<len(self.nodes),
                    'shared affine row/node identity')
            self.nid+=1; owner,local,refs=self.nodes[n]
            require(owner==i and not refs and type(local) is SourceLocal and local.row is self.reader.row(key)
                    and local.row.width==len(rows), 'exact source row alias/owner/factor map')
            self.norms[n]=self.local_norms[n]; center=row.bias
            for j,weight in enumerate(row.coefficients):
                self.work.visit('coefficient_visits'); pc,pn=rows[j]
                require(not self.nodes[pn][2] and self.nodes[pn][1]==((j,F(1)),)
                        and local.row.coefficient(j)==weight, 'shared unit input/source coefficient including zeros')
                if pc and weight: center=self.work.exact(center+weight*pc)
            c=nearest(center); require(stored==c, 'independent shared affine center')
            require(all(j>=basis for j,_ in local.tail), 'shared private compensation scope')
            error=self.work.exact(abs(center-c)); delta=F(0); expected={}
            if error:
                f=self.factor(f'{name}/error/{k}','continuous',scope,i); q=nearest(error); delta=abs(error-q)
                if q: expected[f]=q
            pending.append((local.tail,expected,delta))
        for k,(tail,expected,delta) in enumerate(pending):
            if delta: expected[self.factor(f'{name}/lift/output/{k}','continuous',scope,i)]=upper(delta)
            require(tail==tuple(sorted(expected.items())), 'independent shared compensation/extra term')
        self.done(i); return i


class CheckedSharedForest(CheckedDirectForest):
    __slots__=()
    def __init__(self, packet, root, source, *, expected, work, enabled=False):
        require(enabled is True, 'shared checker is opt-in')
        checker=Checker(packet,root,source,expected,work); result=checker.run()
        self.identity=sha(expected); self.source=sha(source); self.request=result['request']
        self.declaration=frozen(checker.reader.request); self.factors=checker.factors
        self.nodes=checker.nodes; self.views=frozen(checker.views); self.roles=frozen(checker.roles)
        self.norms=tuple(checker.norms); self.scales=checker.scales; self.receipt=frozen(result)
        self.complete=True; self.session=work; self._root=str(root)
        self.factor_identity=identity(dict(factors=self.factors,scales=[str(q) for q in self.scales]))
        self._sealed=True; work.tick()
