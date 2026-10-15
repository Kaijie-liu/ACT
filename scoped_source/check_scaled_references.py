"""Independent V3 source check. Standard library; no producer or native code.

Stored coefficients denote exact binary64 rationals. Radius grouping does not
identify different factors. Later-layer triangle radii are outer bounds.
"""
from fractions import Fraction as F

from scoped_source.check_affine_references import Checker as Base
from scoped_source.check_source_dag import binary,nearest,upper
from scoped_source.forest_endpoints import CheckedForest as V1, frozen
from scoped_source.hz_request_store import identity,rational,require,sha


class Checker(Base):
    schema='SOURCE_SCALED_REFERENCE_FOREST_V3'
    extra_fields=frozenset({'scales'})

    def init_extra(self,doc):
        require(type(doc['scales']) is list and len(doc['scales'])==len(self.factors),'complete scale inventory')
        scales=[]
        for value in doc['scales']:
            self.work.visit('coefficient_visits'); q=binary(rational(value))
            require(q>=0,'nonnegative factor scale'); scales.append(q)
        self.scales=tuple(scales)

    def local_norm_start(self): return {}

    def local_norm_add(self,state,j,q):
        d=self.scales[j]; state[d]=self.work.exact(state.get(d,F(0))+abs(q)); return state

    def local_norm_finish(self,state):
        result=F(0)
        for d,value in state.items(): result=self.work.exact(result+self.work.exact(d*value))
        return result

    def factor(self,name,kind,scope,owner):
        j=super().factor(name,kind,scope,owner)
        if owner!=0: require(self.scales[j]==1,'unit auxiliary factor scale')
        return j

    def initial(self):
        i,v=self.view('input',-1,None,'input')
        require(not v['eq'] and not v['ine'] and len(v['output'])==len(self.reader.inputs),'input roster')
        for k,((lo,hi),(c,n)) in enumerate(zip(self.reader.inputs,v['output'],strict=True)):
            f=self.factor(f'input/c/{k}','continuous',-1,i); d=self.node(i,n)
            expected=nearest(self.work.exact((lo+hi)/2)); radius=upper(max(expected-lo,hi-expected))
            require(c==expected and d=={f:F(1)} and self.scales[f]==radius
                    and c-radius<=lo<=hi<=c+radius,'scaled input containment/identity')
        self.done(i); return i

    def affine(self,parent,layer,stream):
        rows=self.views[parent]['output']
        if rows!=self.views[0]['output']: return super().affine(parent,layer,stream)
        name=f'{layer.graph}/{layer.index}'; scope=-1 if layer.graph=='router' else int(layer.graph[6:])
        i,v=self.view(name,scope,parent,'Linear'); basis=self.fid
        require(len(v['output'])==layer.outputs and not v['eq'] and not v['ine'],'affine roster')
        pending=[]
        for k,(stored,n) in enumerate(v['output']):
            row=next(stream)
            require((row.graph,row.layer,row.row)==(layer.graph,layer.index,k),'affine source order')
            local=self.node(i,n); center=row.bias
            for j,weight in enumerate(row.coefficients):
                self.work.visit('coefficient_visits'); pc,pn=rows[j]; entries=self.nodes[pn][1]
                require(not self.nodes[pn][2] and len(entries)==1 and entries[0][1]==1,'checked unit input')
                if pc and weight: center=self.work.exact(center+weight*pc)
                f=entries[0][0]
                require(local.pop(f,F(0))==weight,'source weight/extra input term')
            c=nearest(center); require(stored==c,'independent affine center')
            require(len(local)<=2 and all(j>=basis for j in local),'affine private/compensation tail')
            error=self.work.exact(abs(center-c)); delta=F(0); exact={}
            if error:
                f=self.factor(f'{name}/error/{k}','continuous',scope,i); q=nearest(error)
                delta=abs(error-q)
                if q: exact[f]=q
            # Rebuild only the <=2-term tail: a popped dict keeps dense capacity.
            pending.append((dict(local),exact,delta))
        for k,(local,exact,delta) in enumerate(pending):
            if delta: exact[self.factor(f'{name}/lift/output/{k}','continuous',scope,i)]=upper(delta)
            require(local==exact,'source weight/center compensation/extra term')
        self.done(i); return i


def check(raw,root,source,*,expected,work,cut=None):
    return Checker(raw,root,source,expected,work,cut).run()


class CheckedScaledForest(V1):
    __slots__=('scales','norms','factor_identity','session')

    def __init__(self,raw,root,source,*,expected,work,cut=None):
        checker=Checker(raw,root,source,expected,work,cut); result=checker.run()
        self.identity=sha(expected); self.source=sha(source); self.request=result['request']
        self.declaration=frozen(checker.reader.request); self.factors=checker.factors
        self.nodes=tuple(checker.nodes); self.views=frozen(checker.views); self.roles=frozen(checker.roles)
        self.norms=tuple(checker.norms); self.scales=checker.scales; self.receipt=frozen(result)
        self.complete=cut is None; self.session=work
        self.factor_identity=identity(dict(factors=self.factors,scales=[str(q) for q in self.scales]))
        self._sealed=True; work.tick()


def checked(forest,work):
    from scoped_source.wire_forest import CheckedWireForest
    admitted=type(forest) in (CheckedScaledForest,CheckedWireForest)
    if not admitted and type(forest).__module__=='scoped_source.check_direct_source':
        from scoped_source.check_direct_source import CheckedDirectForest
        admitted=type(forest) is CheckedDirectForest
    if not admitted and type(forest).__module__=='scoped_source.check_shared_source':
        from scoped_source.check_shared_source import CheckedSharedForest
        admitted=type(forest) is CheckedSharedForest
    require(admitted and getattr(forest,'_sealed',False) is True
            and forest.complete and forest.session is work,'complete checked V3 / cumulative Work binding')
    work.tick()
