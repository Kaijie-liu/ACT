"""Full validation once per owned immutable object; guard every use.

No global cache, no persisted trusted flag, no numerical reuse. External source
or plan data always goes through full validation. Handle identity, generation,
source and scope remain checked on every call. Arbitrary replacement of Python
code/private memory is outside this in-process API contract; the independent
original-LP checker is still the certificate gate.
"""
from .engine import POLICY
from .immutable import ExactRational,Row,Source,OwnedPlan,Handle,Admission
from . import field
from lp_sandwich.check import identity


def freeze(system,scope,b,stats):
    b.where('source','freeze')
    if type(scope) is not str or not scope:raise ValueError('explicit source scope')
    m=len(system)
    if m>POLICY['equations']:b.fail('equations',m,POLICY['equations'])
    result=[];nnz=0;units=0
    for row,rhs in system:
        if type(row) is not dict:raise ValueError('admission requires ordinary source dictionaries')
        entries=[]
        for j,v in row.items():
            b.visit();nnz+=1
            if type(j) is not int or not 0<=j<m:raise ValueError('source column')
            if nnz>POLICY['input_nnz']:b.fail('input_nnz',nnz,POLICY['input_nnz'])
            value=b.value(v)
            entries.append((j,ExactRational(value.numerator,value.denominator)))
            units+=1
        value=b.value(rhs);units+=1
        result.append((Row(tuple(entries)),ExactRational(value.numerator,value.denominator)))
        b.owned_source_units=units
        # Caller input may remain live while the owned copy is used.
        live=nnz+units
        stats['peak_live_nnz']=max(stats['peak_live_nnz'],live)
        if live>POLICY['live_nnz']:b.fail('live_nnz',live,POLICY['live_nnz'])
    frozen=tuple(result)
    binding=field.bind(frozen,scope,b)
    stats['source_full_checks']+=1;stats['owned_source_units']=units
    return Source(frozen,scope,binding,units)


class Session:
    def __init__(self,system,b,stats,*,scope,amortize_source=True,amortize_plan=True):
        if type(amortize_source) is not bool or type(amortize_plan) is not bool:
            raise ValueError('fixed boolean policy flags')
        self._b,self._stats=b,stats
        for key,value in (('plan_builds',0),('plan_replays',0),('plan_invalidations',[]),
                          ('peak_plan_units',0),('source_full_checks',0),('plan_full_checks',0),
                          ('request_guards',0),('plan_guard_hits',0),('owned_source_units',0)):
            stats.setdefault(key,value)
        self._source=freeze(system,scope,b,stats)
        self._validated_source=self._source
        self._policy=(amortize_source,amortize_plan)
        self._owner=object();self._generation=0;self._admission=None
        self._rounds=0;self._closed=False

    @property
    def system(self):return self._source.system

    @property
    def source_id(self):return self._source.binding

    @property
    def handle(self):return None if self._admission is None else self._admission.handle

    def export_plan(self):
        # Mutable dataclass wrapper is new; every payload member is a tuple,
        # int or str. Mutating this wrapper cannot alter the owned plan.
        if self._admission is None:return None
        p=self._admission.plan
        return field.Plan(p.binding,p.initial,p.steps,p.checksum)

    def _guard(self,scope,source_id,handle=None):
        b=self._b;b.where('validation','request_guard');b.visit()
        if self._closed or self._source is not self._validated_source:
            raise ValueError('closed or replaced request source')
        if (type(scope) is not str or type(source_id) is not str or scope!=self._source.scope
                or source_id!=self._source.binding):
            raise ValueError('request source/scope mismatch')
        if handle is not None:
            a=self._admission
            if (a is None or type(handle) is not Handle or handle is not a.handle
                    or handle.owner is not self._owner or handle.generation!=self._generation
                    or a.source is not self._source):raise ValueError('foreign/stale/forged handle')
        self._stats['request_guards']+=1

    def _check_owned_plan(self):
        b=self._b;a=self._admission
        if a is None:return
        b.where('validation','plan_guard');b.visit()
        if (type(a) is not Admission or a.source is not self._source or type(a.plan) is not OwnedPlan
                or a.handle.owner is not self._owner or a.handle.generation!=self._generation):
            raise ValueError('plan receipt binding')
        self._stats['plan_guard_hits']+=1
        b.plan_units=a.plan.units
        if not self._policy[1]:
            raw=self.export_plan()
            field.validate(raw,self.source_id,len(self.system),b,self._stats)
            self._stats['plan_full_checks']+=1

    def admit_plan(self,raw,*,scope,source_id):
        """Imported data is NEVER admitted using a serialized validated flag."""
        self._guard(scope,source_id)
        b=self._b;b.where('plan','admit')
        if type(raw) is not field.Plan:raise ValueError('untrusted plan schema')
        # Exact builtin types prevent mutable/proxy descendants from entering
        # the owned object. validate checks all indices, permutations and hash.
        if type(raw.binding) is not str or type(raw.checksum) is not str:
            raise ValueError('plan identity types')
        old_units=0 if self._admission is None else self._admission.plan.units
        previous=self._admission;generation=self._generation
        try:
            field.validate(raw,self.source_id,len(self.system),b,self._stats)
            self._stats['plan_full_checks']+=1
            total=old_units+raw.units
            if total>POLICY['plan_entries']:b.fail('plan_entries',total,POLICY['plan_entries'])
            self._stats['peak_plan_units']=max(self._stats['peak_plan_units'],total)
            source_nnz=self._source.units-len(self.system)
            live=source_nnz+b.owned_source_units+total
            self._stats['peak_live_nnz']=max(self._stats['peak_live_nnz'],live)
            if live>POLICY['live_nnz']:b.fail('live_nnz',live,POLICY['live_nnz'])
            owned=OwnedPlan(raw.binding,raw.initial,raw.steps,raw.checksum,raw.units)
            self._generation+=1;handle=Handle(self._owner,self._generation)
            self._admission=Admission(self._source,owned,handle)
            b.tick()
        except BaseException:
            # A late deadline/exception during receipt publication must not
            # leave an admission that the caller was told had failed.
            self._admission=previous;self._generation=generation
            raise
        finally:
            # On failure the old receipt is retained. Partial construction is
            # not an admitted plan; no reset of time/operations is permitted.
            b.plan_units=0 if self._admission is None else self._admission.plan.units
        return self._admission.handle

    def solve(self,p,*,scope,source_id,handle=None):
        self._guard(scope,source_id,handle)
        b=self._b;st=self._stats
        # Discarded partial dynamic plans do not survive a failed prime, but
        # an admitted old plan remains resident through any replacement.
        b.plan_units=0 if self._admission is None else self._admission.plan.units
        if self._rounds and not self._policy[0]:
            if field.bind(self.system,scope,b)!=self.source_id:raise ValueError('owned source mismatch')
            st['source_full_checks']+=1
        self._rounds+=1
        self._check_owned_plan()
        if self._admission is not None:
            try:
                x=field.replay(self.system,p,b,st,self._admission.plan)
                if not field.field_residual(self.system,x,p,b,st):
                    raise field.PlanMismatch('field_residual_mismatch')
                st['plan_replays']+=1
                return x
            except field.PlanMismatch as exc:
                st['plan_invalidations'].append({'prime':p,'reason':str(exc),'operations':b.operations})
        x,initial,steps=field.dynamic(self.system,p,b,st,capture=True)
        if not field.field_residual(self.system,x,p,b,st):raise ValueError('dynamic field residual')
        b.where('plan','publish')
        raw=field.Plan(self.source_id,initial,steps,identity(field.payload(self.source_id,initial,steps)))
        self.admit_plan(raw,scope=scope,source_id=source_id)
        st['plan_builds']+=1
        return x

    def close(self):
        self._closed=True;self._admission=None
        self._b.plan_units=0
