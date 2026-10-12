"""Untrusted fixed controls. This module is never copied to portable checking."""
from fractions import Fraction as F
from itertools import combinations

from scoped_source.hz_store_write import Writer


def build(root, *, large, work, page_rows=64):
    from scoped_source.hz_request_store import identity
    if large:
        nc,nb,pc,pb,e,out,co,ce,ci,pe,pi,g = 3856,128,2324,384,8,10,3072,128,256,384,768,12
    else:
        nc,nb,pc,pb,e,out,co,ce,ci,pe,pi,g = 3,1,1,1,3,2,3,1,2,1,2,2
    n,k = nc+nb,pc+pb
    factors = []
    for scope,c,b in [('common',nc,nb)]+[(f'expert/{i}',pc,pb) for i in range(e)]:
        for kind,count in [('continuous',c),('binary',b)]:
            for j in range(count): factors.append([f'{scope}/{kind}/{j}',kind,scope,'fixed-control'])
    request = identity(dict(control='REQUEST_OWNED_HZ_STORE_V1',layout='large' if large else 'small'))
    w = Writer(root,request,e,factors,work=work,page_rows=page_rows)
    def outputs(name,active,count,rhs):
        rows = [[f'{name}/out/{i}',str(rhs),[]] for i in range(count)]
        for pos,j in enumerate(active): rows[pos%count][2].append([j,'1'])
        return w.rows('output',rows)
    def constraints(name,kind,start,length,count):
        return w.rows(kind,([f'{name}/{kind}/{j}','0' if kind == 'eq' else '1',[[start+j%length,'1']]] for j in range(count)))
    common_out = outputs('common',range(n),co,F(0))
    w.view('common',scope='common',add_factors=range(n),output=common_out,
        eq=constraints('common','eq',0,n,ce),ine=constraints('common','ine',0,n,ci))
    for i in range(e):
        start = n+i*k; scope = f'expert/{i}'
        w.view(scope,scope=scope,parent='common',add_factors=range(start,start+k),
            output=outputs(scope,list(range(n))+list(range(start,start+k)),out,F(i+1,16)),
            eq=constraints(scope,'eq',start,k,pe),ine=constraints(scope,'ine',start,k,pi))
    entries = []
    for a,b in combinations(range(e),2):
        name = f'pair/{a}/{b}'
        refs = w.rows('ine',([f'{name}/guard/{j}','0',[[j%n,str((-1)**j)]]] for j in range(g)))
        w.view(name,scope='common',parent='common',guard=[a,b],output=common_out,ine=refs)
        entries.append(dict(pair=[a,b],view=name))
    manifest = w.finish(dict(common='common',templates=[f'expert/{i}' for i in range(e)],entries=entries))
    return dict(manifest=manifest,request=request)
