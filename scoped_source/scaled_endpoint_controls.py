"""Finite V3 fixtures and test-only expanded matrix oracle, no large source."""
from fractions import Fraction as F
from pathlib import Path

from scoped_source.hz_request_store import require,rational

CASES=('mixed','deep','clipped','zero','half','guarded')


def fixture(root,case):
    require(case in CASES,'frozen scaled control source')
    if case in ('deep','clipped','zero'):
        from scoped_source.affine_reference_controls import deep
        return deep(root,**({} if case=='deep' else dict(center=(.875,-.75),radius='1/4' if case=='clipped' else '0')))
    from scoped_source.forest_endpoint_controls import fixture as f
    return f(root,case)


def expanded(forest,work):
    from scoped_source.affine_reference_controls import expanded as e
    rows=e(forest.nodes,work)
    for row in rows:
        for j in row: work.visit('coefficient_visits'); row[j]=work.exact(row[j]*forest.scales[j])
    return rows


def scalar_bound(forest,task,candidate,rows,work):
    """Independent expanded exact formula. Does not call production pullback."""
    low=rational(task.offset); objective={}
    for expert,weights in zip(task.pair,(task.left,task.right),strict=True):
        outputs=forest.views[forest.roles['experts'][expert]]['output']
        for (center,n),weight in zip(outputs,weights,strict=True):
            work.visit('row_visits'); weight=rational(weight)
            low=work.exact(low+work.exact(weight*center))
            for j,q in rows[n].items():
                work.visit('coefficient_visits'); objective[j]=work.exact(objective.get(j,F(0))+work.exact(weight*q))
    for kind,key in (('eq','equality'),('ine','inequality')):
        for row,weight in zip(forest.rows(task.pair,kind),candidate[key],strict=True):
            work.visit('row_visits'); rhs,n=forest.row(row); weight=rational(weight)
            low=work.exact(low+work.exact(weight*rhs))
            for j,q in rows[n].items():
                work.visit('coefficient_visits'); objective[j]=work.exact(objective.get(j,F(0))-work.exact(weight*q))
    for q in objective.values(): low=work.exact(low-abs(q))
    return low,objective


def differential(forest,contract,proof,work):
    """Separately charged control, not a second production acceptance pass."""
    from scoped_source.scaled_endpoints import Session
    from scoped_source.hz_request_store import identity
    rows=expanded(forest,work); s=Session(forest,contract,expected=identity(contract),work=work,width=proof['width'])
    count=0; receipts=[]
    for i,records in enumerate(proof['tiles']):
        tasks,_=s.prepare(i); checked=s.accept(records); receipts.extend(checked)
        for task,record,receipt in zip(tasks,records,checked,strict=True):
            if receipt is None: continue
            for role in ('proposed','zero'):
                expected,_=scalar_bound(forest,task,record[role],rows,work)
                require(expected==F(receipt.record[role]),'same-V3 exact candidate differential'); count+=1
    return dict(candidates=count,expanded_nnz=sum(map(len,rows)),result=s.finish(receipts))
