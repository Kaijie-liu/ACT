"""Independent stdlib V2 source checker. Never imports the producer/native.

The triangle range bound is deliberately not an expanded exact L1 norm.
Checks captured real-source enclosure; no output or deployed float conclusion.
"""
from fractions import Fraction as F

from scoped_source.check_source_dag import Checker as Base, nearest, upper
from scoped_source.hz_request_store import require, sha
from scoped_source.forest_endpoints import frozen


class Checker(Base):
    schema='SOURCE_AFFINE_REFERENCE_FOREST_V2'

    def affine(self,parent,layer,stream):
        rows=self.views[parent]['output']
        if rows==self.views[0]['output']: return super().affine(parent,layer,stream)
        name=f'{layer.graph}/{layer.index}'; scope=-1 if layer.graph=='router' else int(layer.graph[6:])
        i,v=self.view(name,scope,parent,'Linear')
        require(len(v['output'])==layer.outputs and not v['eq'] and not v['ine'],'affine output/constraint roster')
        pending=[]
        for k,(stored_c,n) in enumerate(v['output']):
            row=next(stream)
            require((row.graph,row.layer,row.row)==(layer.graph,layer.index,k),'affine source order')
            center=row.bias; combined={}
            for j,weight in enumerate(row.coefficients):
                self.work.visit('coefficient_visits'); c,p=rows[j]
                if weight:
                    center=self.work.exact(center+weight*c)
                    combined[p]=self.work.exact(combined.get(p,F(0))+weight)
            refs=tuple((p,combined[p]) for p in sorted(combined) if combined[p])
            d=self.node(i,n,refs=refs); exact_c=nearest(center)
            require(stored_c==exact_c,'affine independent center')
            error=self.work.exact(abs(center-exact_c)); expected={}; delta=F(0)
            if error:
                f=self.factor(f'{name}/error/{k}','continuous',scope,i); q=nearest(error)
                delta=abs(error-q)
                if q: expected[f]=q
            pending.append((d,expected,delta))
        for k,(d,expected,delta) in enumerate(pending):
            if delta: expected[self.factor(f'{name}/lift/output/{k}','continuous',scope,i)]=upper(delta)
            require(d==expected,'affine center compensation/extra local term')
        self.done(i); return i

    def guard(self,i,actual,other,selected):
        c,n=other; sc,sn=selected; rhs,root=actual
        refs=() if n==sn else tuple(sorted(((n,F(1)),(sn,F(-1)))))
        require(rhs==upper(self.work.exact(sc-c)) and self.node(i,root,refs=refs)=={},'guard reference/sign/rhs')


def check(raw,root,source,*,expected,work,cut=None):
    result=Checker(raw,root,source,expected,work,cut).run()
    return dict(result,representation='EXACT_AFFINE_REFERENCES_TRIANGLE_RADIUS_V2')


class CheckedAffineForest:
    __slots__=('identity','nodes','factors','views','roles','norms','receipt','complete','_sealed')

    def __setattr__(self,k,v):
        if getattr(self,'_sealed',False): raise AttributeError('immutable checked affine forest')
        object.__setattr__(self,k,v)

    def __delattr__(self,k): raise AttributeError('immutable checked affine forest')

    def __init__(self,raw,root,source,*,expected,work,cut=None):
        checker=Checker(raw,root,source,expected,work,cut); result=checker.run()
        self.identity=sha(expected); self.nodes=tuple(checker.nodes); self.factors=checker.factors
        self.views=frozen(checker.views); self.roles=frozen(checker.roles); self.norms=tuple(checker.norms)
        self.receipt=frozen(result); self.complete=cut is None; self._sealed=True; work.tick()
