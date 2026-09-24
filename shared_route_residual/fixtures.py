"""Fixed synthetic routers only; never a trained model or a sealed input."""
from checked_route_frontier.fixtures import encode, rebind
from source_construction_lab.fixtures import document


def fixture(kind):
    if kind not in ('prunable','tied','random'):
        raise ValueError('fixed synthetic fixture')
    doc=document(experts=8,classes=10,width=32,depth=3,seed=724,radius='1/8')
    if kind!='random':
        layer=doc['networks'][0]['layers'][-1]
        layer['weight']=encode([1/32]*256,[8,32])
        layer['bias']=encode([float(8-i)*3 if kind=='prunable' else 0. for i in range(8)],[8])
        rebind(doc)
    return doc
