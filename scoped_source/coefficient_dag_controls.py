"""Fixed given-row recipes and independent expanded-row oracle, no optimizer."""
from fractions import Fraction as F
from scoped_source.coefficient_dag import SCHEMA
from scoped_source.hz_request_store import identity

REQUEST = identity({'control': 'coefficient-dag-frozen-20261013'})


def small(pair=(0, 1)):
    a, b = pair; frame = 17
    domains = [dict(name='common', frame=frame, experts=[], guard=None, parents=[])]
    for e in (a, b):
        domains.append(dict(name=f'expert/{e}', frame=frame, experts=[e], guard=None, parents=[0]))
    domains.append(dict(name=f'pair/{a}/{b}', frame=frame, experts=[a,b], guard=[a,b], parents=[0,1,2]))
    factors = [['shared/x','continuous',-1],['shared/y','continuous',-1]] + [
        [f'private/{e}', 'binary' if e == 2 else 'continuous', e] for e in range(3)]
    nodes = [
        ['affine',0,[[0,'1'],[1,'2']],[]],
        ['active-alias',0,[],[[0,'1']]],
        ['negative-with-private',1,[[a+2,'1/2']],[[0,'-1']]],
        ['other-expert',2,[[b+2,'-1/2']],[[0,'1']]],
        ['diamond',3,[],[[1,'1'],[2,'1']]],
        ['guard',3,[[0,'1']],[]]]
    roots = [['out/0',0,'output','3/4',0],['out/1',0,'output','7/4',1],
        ['out/2',2,'output','1/8',3],['out/3',3,'output','0',4],
        ['relu/eq',1,'eq','-1/4',2],['guard/0',3,'ine','0',5],['guard/1',3,'ine','1',5]]
    # Construct original rows independently, not by expanding the DAG.
    rows = [('output','3/4',{0:F(1),1:F(2)}), ('output','7/4',{0:F(1),1:F(2)}),
        ('output','1/8',{0:F(1),1:F(2),b+2:F(-1,2)}),
        ('output','0',{a+2:F(1,2)}), ('eq','-1/4',{0:F(-1),1:F(-2),a+2:F(1,2)}),
        ('ine','0',{0:F(1)}),('ine','1',{0:F(1)})]
    source = identity([[k,c,[[j,str(v)] for j,v in sorted(row.items())]] for k,c,row in rows])
    return dict(schema=SCHEMA,request=REQUEST,source=source,frame=frame,factors=factors,
                domains=domains,target=3,nodes=nodes,roots=roots), rows


def expanded(rows, weights, duals, offset='1/8'):
    """Independent direct row residual; does not read DAG nodes or helpers."""
    constant = F(offset); residual = {}; counters = dict(output=0,eq=0,ine=0)
    for kind, rhs, entries in rows:
        values = weights if kind == 'output' else duals[kind]
        w = F(values[counters[kind]]); counters[kind] += 1
        constant += w*F(rhs)
        for j,v in entries.items(): residual[j] = residual.get(j,F(0)) + (w if kind=='output' else -w)*v
    return constant-sum(map(abs,residual.values())),constant,residual


def dense():
    n, width = 256, 3072; frame = 23; nodes = []; roots = []
    for r in range(n):
        entries = [[c,str(F((-1 if (r+c+2)%2 else 1)*((17*r+7*c+6)%31+1),32))] for c in range(width)]
        nodes.append([f'original/{r}',0,entries,[]])
        roots.append([f'output/{r}',0,'output',str(F(r-2,16)),r])
    for r in range(n):
        nodes.append([f'negative/{r}',0,[],[[r,'-1']]])
        roots.append([f'eq/{r}',0,'eq',str(F(3*r-1,32)),n+r])
    source = identity(dict(recipe='fixed-dense-duplicate-v1',width=width,rows=n,salt=2,
                           rhs='output:(r-2)/16;eq:(3r-1)/32'))
    return dict(schema=SCHEMA,request=REQUEST,source=source,frame=frame,
        factors=[[f'x/{j}','continuous',-1] for j in range(width)],
        domains=[dict(name='given',frame=frame,experts=[],guard=None,parents=[])],
        target=0,nodes=nodes,roots=roots)


def dense_reference(cancel, tick):
    """Original two sets of rows, independently integer-scaled to denominator 512.

    Each repeated coefficient is consumed, not reconstructed via the DAG. This
    oracle's own operation counts are not the support checker's work counter.
    It shares the absolute deadline and is separately costed.
    """
    residual = [0]*3072; constant = F(0); count = 0
    for kind in ('output','eq'):
        for r in range(256):
            tick(); weight = 2 if kind == 'output' else (-2 if cancel else r%3+1)
            rhs = F(r-2,16) if kind=='output' else F(3*r-1,32)
            constant += F(weight,16)*rhs
            for c in range(3072):
                if c%256 == 0: tick()
                # Eq coefficient is minus original and residual subtracts dual.
                numerator = (1 if (r+c)%2==0 else -1)*(1+(17*r+7*c+6)%31)
                residual[c] += weight*numerator; count += 1
    exact = {j:F(v,512) for j,v in enumerate(residual)}
    return constant-sum(map(abs,exact.values())), constant, exact, count
