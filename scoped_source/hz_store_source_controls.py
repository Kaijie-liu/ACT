"""Frozen source recipes and solver-free saved-object controls; no producer."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

from scoped_source.hz_request_store import Row, Rows, Work, identity, require
from scoped_source import check_hz_store_source as check


def affine(indices, weights=None):
    return dict(kind='affine',rows=[['0',[[j,str(weights[i] if weights else 1)]]]
                                   for i,j in enumerate(indices)])


def fixture(name):
    if name=='wide':
        n=3072; w=F(1)+F(1,2**52)
        lower=[str(-w)]*n; upper=[str(w)]*n
        router=[affine(range(n),[w]*n),affine([0,1,n-1])]
        experts=[[affine([0,1])] for _ in range(3)]
    elif name=='mixed':
        n=70; center=[F(0)]*n; radius=[F(1)]*n
        for i in (0,65): center[i]=F(1,2**54)
        for i,(c,r) in enumerate(((2,1),(-2,1),(0,0),(1,1),(-1,1)),1):
            center[i],radius[i]=F(c),F(r)
        lower=[str(c-r) for c,r in zip(center,radius)]
        upper=[str(c+r) for c,r in zip(center,radius)]
        hidden=[affine(range(n)),{'kind':'relu'},affine(range(n)),{'kind':'relu'}]
        router=deepcopy(hidden)+[affine([0,1,65])]
        experts=[deepcopy(hidden)+[affine([0,1])] for _ in range(3)]
    elif name=='input':
        lower=['1/3','0','1']; upper=['2/3','1','1']
        router=[dict(kind='affine',rows=[['1',[]] for _ in range(3)])]
        experts=[[affine([0,1])] for _ in range(3)]
    else: raise ValueError('unregistered source recipe')
    return dict(schema=check.SCHEMA,lower=lower,upper=upper,router=router,experts=experts)


def signature(store):
    # Complete coefficient differential, excluding only page identities.
    return identity(dict(factors=store.factors,views=[dict(name=v.name,scope=v.scope,guard=v.guard,
        parent=v.parent.name if v.parent else None,frame=v.frame,factors=sorted(v.factors),
        rows={kind:[[r.name,str(r.rhs),[[j,str(q)] for j,q in r.coefficients]]
                    for r in getattr(v,kind)] for kind in ('output','eq','ine')})
        for v in store.views.values()]))


def analytic(store, name, work):
    require(store.request==identity(fixture(name)), 'frozen declaration identity')
    if name=='wide':
        values=tuple(store.views['router/0'].output)
        require(len(values)==3072, 'wide full output inventory')
        for i,row in enumerate(values):
            work.tick()
            require(row.rhs==0 and row.coefficients==((i,F(1)+F(1,2**51)),(3072+i,F(1,2**104))),
                    'wide exact all-coordinate analytic residual')
        return dict(checked_affine_rows=3072,each_residual=str(F(1,2**104)))
    if name=='mixed':
        indices={v[0]:j for j,v in enumerate(store.factors)}
        for prefix in ('router','expert0','expert1','expert2'):
            view=store.views[f'{prefix}/1']; outputs=tuple(view.output)
            eq=tuple(view.eq.suffix(view.parent.eq))
            for i,eqi in ((0,0),(65,60)):
                # Unstable row 65 follows 60 preceding unstable rows (1..5 stable).
                require(dict(outputs[i].coefficients)[indices[f'{prefix}/1/lift/output/{i}']]==F(1,2**54),
                        'mixed nonzero output enclosure oracle')
                require(dict(eq[eqi].coefficients)[indices[f'{prefix}/1/lift/eq/{eqi}']]==F(1,2**53),
                        'mixed nonzero equality enclosure oracle')
            second=store.views[f'{prefix}/3']
            require(view.ine.count-view.parent.ine.count==130 and
                    second.ine.count-second.parent.ine.count==4,
                    'mixed inherited and second-layer inequalities')
        return dict(first_unstable=65,first_active=3,first_inactive=2,
                    second_unstable=2,checked_nonzero_locations=8)
    require(len(store.entries)==3 and all(not r.coefficients and r.rhs==0
        for v in store.entries.values() for r in v.ine.suffix(store.common.ine)),
        'all tie-legal guards retained')
    return dict(tie_pairs=3,input_coordinates=3)


def changed(view, kind, fn):
    rows=getattr(view,kind)
    local=list(rows) if kind=='output' else list(rows.suffix(getattr(view.parent,kind)))
    fn(local)
    parent=None if kind=='output' else getattr(view.parent,kind)
    new=Rows(parent,(tuple(local),),len(local)+(parent.count if parent else 0))
    return replace(view,**{kind:new})


def remove_coefficient(row, column):
    return replace(row,coefficients=tuple((j,q) for j,q in row.coefficients if j!=column))


NEGATIVES=(
    'input_tail_inward','input_wrong_coordinate','input_tail_missing','affine_missing_error','affine_target_only',
    'relu_output_lift_missing','relu_equality_lift_missing','relu_missing_tail',
    'relu_block_order','inherited_prefix_changed','operator_factor_owner',
    'common_not_input','guard_rhs_inward','wrong_source_identity','missing_source_layer',
    'extra_source_parameter','missing_store_layer','capture_byte_anchor','source_deadline_tail',
)

EXPECTED={
    'input_tail_inward':'input enclosure inward','input_wrong_coordinate':'input coordinate map',
    'input_tail_missing':'input coordinate inventory',
    'affine_missing_error':'source factor allocation/kind/owner','affine_target_only':'source factor allocation/kind/owner',
    'relu_output_lift_missing':'missing/inward source enclosure','relu_equality_lift_missing':'missing/inward source enclosure',
    'relu_missing_tail':'source new row inventory','relu_block_order':'inward source inequality',
    'inherited_prefix_changed':'source inherited constraints','operator_factor_owner':'source factor allocation/kind/owner',
    'common_not_input':'common input restoration','guard_rhs_inward':'inward source inequality',
    'wrong_source_identity':'source declaration identity','missing_source_layer':'source declaration identity',
    'extra_source_parameter':'source declaration identity','missing_store_layer':'missing source layer',
    'capture_byte_anchor':'source declaration identity','source_deadline_tail':'injected source tail deadline',
}


def negative(store, doc, name, work):
    """Reconstruct a specific immutable-row mutation; never edit parent archives."""
    if name.startswith('input_'):
        view=store.views['input']
        if name=='input_tail_inward':
            target=changed(view,'output',lambda r:r.__setitem__(-1,replace(r[-1],coefficients=())))
        elif name=='input_wrong_coordinate':
            target=changed(view,'output',lambda r:r.__setitem__(0,replace(r[0],coefficients=((1,F(1)),))))
        else: target=changed(view,'output',lambda r:r.pop())
        fake=SimpleNamespace(views={'input':target},factors=store.factors)
        return check.check_input(fake,doc['lower'],doc['upper'],work)
    if name.startswith('affine_'):
        src=store.views['input']; target=store.views['router/0']
        if name=='affine_missing_error':
            target=changed(target,'output',lambda r:r.__setitem__(0,remove_coefficient(r[0],3072)))
        else:
            target=changed(target,'output',lambda r:r.__setitem__(-1,replace(r[-1],coefficients=
                tuple(sorted((*r[-1].coefficients,(0,F(1))))))))
        return check.check_affine(store,src,target,doc['router'][0],work)
    src=store.views['router/0']; target=store.views.get('router/1')
    if name.startswith('relu_'):
        if name=='relu_missing_tail': target=changed(target,'output',lambda r:r.pop())
        elif name=='relu_block_order':
            def swap(rows):
                a,b=rows[0],rows[65]
                rows[0]=replace(b,name=a.name); rows[65]=replace(a,name=b.name)
            target=changed(target,'ine',swap)
        else:
            kind='output' if name=='relu_output_lift_missing' else 'eq'
            label=f'router/1/lift/{kind}/0'
            j=next(i for i,f in enumerate(store.factors) if f[0]==label)
            target=changed(target,kind,lambda r:r.__setitem__(0,remove_coefficient(r[0],j)))
        return check.check_relu(store,src,target,work)
    if name=='inherited_prefix_changed':
        src=store.views['router/2']; target=store.views['router/3']
        target=replace(target,ine=Rows(None,(),0))
        return check.check_relu(store,src,target,work)
    if name=='operator_factor_owner':
        factors=list(store.factors); j=next(i for i,f in enumerate(factors) if f[0]=='router/1/sign/0')
        f=list(factors[j]); f[3]='expert0/1'; factors[j]=tuple(f)
        return check.check_relu(SimpleNamespace(factors=tuple(factors)),src,target,work)
    if name=='common_not_input':
        router=store.views['router/4']; common=replace(store.common,output=router.output)
        return check.check_common(SimpleNamespace(common=common,factors=store.factors),store.views['input'],router,work)
    if name=='guard_rhs_inward':
        pair=(0,1); guard=changed(store.entries[pair],'ine',lambda r:r.__setitem__(0,replace(r[0],rhs=r[0].rhs-1)))
        fake=SimpleNamespace(common=store.common,entries={pair:guard},factors=store.factors,experts=store.experts)
        return check.check_guard(fake,store.views['router/4'],pair,work)
    if name=='wrong_source_identity': return check.check_source(store,doc,expected_source='f'*64,work=work)
    if name=='missing_store_layer':
        fake=SimpleNamespace(**{k:getattr(store,k) for k in ('request','identity','experts','factors','common','templates','entries')})
        fake.views=dict(store.views); del fake.views['router/1']
        return check.check_source(fake,doc,expected_source=store.request,work=work)
    if name=='capture_byte_anchor':
        a=deepcopy(doc); b=deepcopy(doc); b['lower'][0]='-999'; expected=identity(b)
        original=check.compact
        def switched(value):
            raw=original(value); a['lower'][0]='-999'; return raw
        check.compact=switched
        try: return check.capture(a,expected,work)
        finally: check.compact=original
    if name in ('missing_source_layer','extra_source_parameter'):
        altered=deepcopy(doc)
        if name=='missing_source_layer': altered['router'].pop()
        else: altered['router'][1]['unused']='1'
        # External identity protects the intended declaration, not self-hashing.
        return check.check_source(store,altered,expected_source=store.request,work=work)
    if name=='source_deadline_tail':
        class Tail(Work):
            def visit(self, kind, amount=1):
                super().visit(kind,amount)
                if self.counts['row_visits']>=30: raise TimeoutError('injected source tail deadline')
        return check.check_source(store,doc,expected_source=store.request,work=Tail(work.deadline))
    raise ValueError('unregistered source mutation')
