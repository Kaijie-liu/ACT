"""Fixed final-affine equality potentials, one row per score. Untrusted."""
from fractions import Fraction as F
from scoped_source.graph import clock
from source_enclosure.format import sparse, unpack
from shared_route_residual.format import SCHEMA, binding


def propose(doc, prefix, *, invocation, deadline):
    tick=clock(deadline)
    state=prefix['router']['steps'][-1]['state'];h,ci,bi=unpack(state)
    final=doc['networks'][0]['layers'][-1];e=doc['request']['experts']
    if len(h['c'])!=e:raise ValueError('score dimension')
    z=[{} for _ in range(e)]
    if final['kind']=='Linear':
        tag='router/layer/'+str(final['index'])+'/value/'
        ids={name:j for j,name in enumerate(ci)}
        cols={i:ids[tag+str(i)] for i in range(e) if tag+str(i) in ids}
        definitions={col:[] for col in cols.values()}
        for row,values in enumerate(h['Ac']):
            tick()
            for col in values:
                if col in definitions:definitions[col].append(row)
        for i,col in cols.items():
            tick()
            found=definitions[col]
            if len(found)!=1:raise ValueError('final affine factor definition')
            row=found[0];value=h['Gc'][i].get(col,F(0))/h['Ac'][row][col]
            if value:z[i][row]=value
    result={'schema':SCHEMA,'binding':binding(doc,prefix,invocation),
        'score_equalities':sparse(z,len(h['b']))}
    tick();return result
