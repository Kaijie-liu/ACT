"""Stage-one measurement helpers, not a new verification algorithm.

Interval census is numerical outer approximation. Sampled histories are only
observed witnesses. Neither quantity is an exact count of legal histories.
"""
from __future__ import annotations
import itertools
import math
import statistics
import time


def coverage(value, arm, expected):
    if arm == 'act':
        records=value.get('records',[])
        attempted=len(records)
        returned=sum(r.get('phase')=='complete' for r in records)
        pruned=sum(r.get('phase')=='complete' and r['status']=='EXCLUDED' for r in records)
        verified=sum(r.get('phase')=='complete' and r['status']=='ACCEPTED' for r in records)
        unvisited=expected-attempted
        unproved=returned-pruned-verified
        active=None if not records or records[-1].get('phase')=='complete' else {
            k:records[-1].get(k) for k in ('phase','history','active_property')}
    else:
        attempted=value.get('attempted_histories',0)
        returned=value.get('returned_histories',len(value.get('histories',[])))
        pruned=value.get('pruned_histories',0)
        verified=value.get('verified_histories',0)
        unvisited=expected-pruned-attempted
        unproved=returned-verified
        active=value.get('active')
    if (any(type(v) is not int or v<0 for v in (expected,attempted,returned,pruned,verified))
            or min(unvisited,unproved,attempted-returned,expected-pruned-verified)<0
            or not 0<=value.get('prefix_queries',0)<=value.get('attempted_prefix_queries',0)):
        raise ValueError('inconsistent history counters')
    return dict(expected=expected,attempted=attempted,returned=returned,pruned=pruned,verified=verified,
        never_started=unvisited,in_flight=attempted-returned,returned_unproved=unproved,
        proof_unclosed=expected-pruned-verified,traversal_complete=unvisited==0 and returned==attempted,
        proof_complete=pruned+verified==expected,active=active,
        prefix_attempts=value.get('attempted_prefix_queries',0),prefix_returns=value.get('prefix_queries',0))


def wilson(k,n):
    if not 0<=k<=n:raise ValueError('binomial count')
    if n==0:return dict(n=0,k=0,interval=None,decision='GRAY_UNDEFINED')
    z=1.959963984540054;p=k/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    lo,hi=center-half,center+half
    return dict(n=n,k=k,rate=p,interval=[lo,hi],decision=(
        'STOP_NO_BOTTLENECK' if hi<.1 else 'ENTER_STAGE2' if lo>.1 else 'GRAY_REPORT'))


def endpoint(records,unsafe):
    """Exclude the cross-source replay union, never rewrite any arm terminal."""
    eligible=[r for r in records if (r['model'],r['index']) not in unsafe]
    incomplete=lambda r:not r['coverage']['traversal_complete']
    return dict(raw=wilson(sum(map(incomplete,records)),len(records)),
        adjusted=wilson(sum(map(incomplete,eligible)),len(eligible)),
        excluded_true_unsafe=len(records)-len(eligible))


def affine_interval(layer,lo,hi):
    import torch
    from torch import nn
    if isinstance(layer,nn.Sequential):
        for item in layer:lo,hi=affine_interval(item,lo,hi)
        return lo,hi
    if isinstance(layer,nn.Linear):
        positive=layer.weight.clamp_min(0);negative=layer.weight.clamp_max(0)
        return (torch.nn.functional.linear(lo,positive,layer.bias)+torch.nn.functional.linear(hi,negative),
                torch.nn.functional.linear(hi,positive,layer.bias)+torch.nn.functional.linear(lo,negative))
    if isinstance(layer,nn.ReLU):return lo.clamp_min(0),hi.clamp_min(0)
    if isinstance(layer,(nn.Identity,nn.Flatten)):return lo.flatten(1),hi.flatten(1)
    raise TypeError('unsupported census layer '+type(layer).__name__)


def pair_interval(a,b,ea,eb,rlo,rhi):
    import torch
    l=torch.sigmoid(rlo[:,a:a+1]-rhi[:,b:b+1])
    h=torch.sigmoid(rhi[:,a:a+1]-rlo[:,b:b+1])
    lo=torch.minimum(l*ea[0]+(1-l)*eb[0],h*ea[0]+(1-h)*eb[0])
    hi=torch.maximum(l*ea[1]+(1-l)*eb[1],h*ea[1]+(1-h)*eb[1])
    return lo,hi


def interval_census(model,center,lower,upper,deadline,seed):
    import torch
    from act.back_end.moe.multilayer_external import excluded_pair
    started=time.monotonic();counts=[math.comb(len(l.experts),2) for l in model.layers]
    expected=math.prod(counts);pruned=leaves=prefixes=0;reasons=[]
    # Fixed five full-model points per domain; no search and no solver.
    rng=torch.Generator().manual_seed(seed)
    points=[center,lower,upper]+[lower+(upper-lower)*torch.rand(center.shape,
        generator=rng,dtype=center.dtype) for _ in range(2)]
    observed=set()
    def tick():
        if time.monotonic()>=deadline:raise TimeoutError('census domain deadline')
    def walk(depth,lo,hi,prefix):
        nonlocal pruned,leaves,prefixes
        tick()
        if depth==len(counts):leaves+=1;return
        layer=model.layers[depth];e=len(layer.experts)
        rlo,rhi=affine_interval(layer.router,lo,hi);prefixes+=1
        bounds={(i,j):float(rhi[0,i]-rlo[0,j]) for i in range(e) for j in range(e) if i!=j}
        live=[]
        for pair in itertools.combinations(range(e),2):
            reason=excluded_pair(pair,e,bounds)
            if reason:
                if any(h[:depth+1]==(*prefix,pair) for h in observed):
                    raise ValueError('census excluded an observed full-model history')
                pruned+=math.prod(counts[depth+1:])
                if len(reasons)<16:reasons.append(dict(prefix=[list(p) for p in (*prefix,pair)],reason=reason))
            else:live.append(pair)
        if depth==len(counts)-1:leaves+=len(live);return
        boxes={}
        for i in sorted(set(v for p in live for v in p)):
            tick();boxes[i]=affine_interval(layer.experts[i],lo,hi)
        for a,b in live:
            nextlo,nexthi=pair_interval(a,b,boxes[a],boxes[b],rlo,rhi)
            walk(depth+1,nextlo,nexthi,(*prefix,(a,b)))
    complete=False
    with torch.no_grad():
        try:
            for point in points:
                tick();_,decisions=model.forward_with_routing(point)
                observed.add(tuple(tuple(sorted(map(int,d.indices[0].tolist()))) for d in decisions))
            walk(0,lower.flatten(1),upper.flatten(1),())
            complete=True
        except TimeoutError:pass
    if len(observed)>expected-pruned:raise ValueError('census bound conflicts with observed histories')
    return dict(structural=expected,candidate_upper=expected-pruned,observed_lower=len(observed),
        observed_histories=[[list(p) for p in h] for h in sorted(observed)],
        prefix_queries=prefixes,retained_leaves=leaves,pruned_leaves=pruned,
        traversal_complete=complete,seconds=time.monotonic()-started,
        first_exclusions=reasons,exact_legal_count=None,
        guarantee='NUMERICAL_INTERVAL_OUTER_COUNT_AND_SAMPLED_NATIVE_HISTORIES_NOT_EXACT_PROOF')


def public_pgd(model,tensors,deadline,seed,steps=20,restarts=2):
    """Uniform fixed attack; no ACT/GenBaB answers and no per-class reselection."""
    import torch
    from act.back_end.moe.class_separated_top1 import validate_replay
    x,lo,hi,rows,thresholds=[tensors[k] for k in ('center','lower','upper','rows','thresholds')]
    started=time.monotonic();rng=torch.Generator().manual_seed(seed);best=None;best_margin=math.inf
    iterations=0;complete=True
    for restart in range(restarts):
        point=lo+(hi-lo)*torch.rand(x.shape,dtype=x.dtype,generator=rng)
        for step in range(steps+1):
            if time.monotonic()>=deadline:complete=False;break
            point=point.detach().requires_grad_(step<steps)
            margins=model(point)@rows.T-thresholds
            value=margins.min()
            if float(value.detach())<best_margin:best_margin=float(value.detach());best=point.detach().clone()
            if best_margin<0 and time.monotonic()<deadline:
                valid=bool(validate_replay(model,best,lo,hi,rows,thresholds))
                if valid and time.monotonic()<deadline:
                    return dict(status='UNSAFE_REPLAYED',completed_fixed_search=False,
                        stop_reason='first_replayed_counterexample',iterations=iterations,
                        minimum_observed_margin=best_margin,witness=best.tolist(),
                        seconds=time.monotonic()-started,guarantee='FULL_MODEL_REPLAY')
            if step<steps:
                gradient=torch.autograd.grad(value,point,only_inputs=True)[0]
                point=torch.maximum(lo,torch.minimum(hi,point.detach()-(2/255/4)*gradient.sign()))
                iterations+=1
        if not complete:break
    valid=False
    if best is not None and time.monotonic()<deadline:
        valid=bool(validate_replay(model,best,lo,hi,rows,thresholds))
        if time.monotonic()>=deadline:valid=False;complete=False
    return dict(status='UNSAFE_REPLAYED' if valid else 'NO_REPLAYED_COUNTEREXAMPLE',
        completed_fixed_search=complete,iterations=iterations,minimum_observed_margin=best_margin if best is not None else None,
        witness=best.tolist() if valid else None,seconds=time.monotonic()-started,
        guarantee='FULL_MODEL_REPLAY' if valid else 'ATTACK_FAILURE_IS_NOT_SAFETY')
