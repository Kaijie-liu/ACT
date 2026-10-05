"""Fixed ordinary toy neural programs: chain, shared Add and Conv/ReLU."""
from fractions import Fraction as F
from itertools import product
import numpy as np
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import reconstruct


def row(continuous=(),binary=(),rhs=0.):
    continuous=sorted(continuous);binary=sorted(binary)
    return dict(cc=np.asarray([c for c,v in continuous],np.int64),cv=np.asarray([v for c,v in continuous],np.float64),
        bc=np.asarray([c for c,v in binary],np.int64),bv=np.asarray([v for c,v in binary],np.float64),rhs=float(rhs))


def fixture(kind,width=128):
    if kind not in ('chain','shared_add','conv_relu','nonunit','zero_hit'):raise ValueError('unknown fixed toy structure')
    old=4 if kind=='conv_relu' else 2;nb=old//2;commands=[];next_col=old
    def predicate(kind,continuous=(),binary=(),rhs=0.,column=-1):
        commands.append(dict(kind=kind,uid=100+len(commands),column=column,**row(continuous,binary,rhs)))
    # r=2*ReLU(x)-1, binary b=-1/+1: an exact nonconvex graph on[-1,1].
    for phase in range(nb):
        x,r=2*phase,2*phase+1
        predicate('ineq',[(r,1.)],[(phase,-1.)])
        predicate('ineq',[(x,-2.),(r,1.)],[(phase,1.)])
        predicate('ineq',[(x,2.),(r,-1.)],rhs=1.)
    def definition(terms,binary=(),rhs=0.):
        nonlocal next_col
        col=next_col;next_col+=1
        predicate('def',[*( (p,-a) for p,a in terms),(col,1.)],[(p,-a) for p,a in binary],rhs,column=col)
        return col
    copies=[]
    if kind!='zero_hit':
        for i in range(width):
            parent=(copies[-1] if copies and kind=='chain' else (1+2*(i%nb)))
            sign=-1. if kind=='chain' and i%3==1 else 1.
            copies.append(definition([(parent,sign)]))
    if kind=='shared_add':
        # Raw two-parent rows become signed-unit definitions only after shared
        # identity coalescing. A post-hoc raw one-parent census misses these.
        merged=[definition([(copies[i],.5),(copies[i+1],.5)]) for i in range(0,width,2)]
        last=definition([(merged[-1],1.)])
    elif kind=='conv_relu':
        # Three-tap averaging over a shared r0,r1,r0 signal, a small Conv row.
        last=definition([(copies[0],.25),(copies[1],.5),(copies[2],.25)])
    elif kind=='nonunit':
        half=definition([(copies[-1],.5)]);last=definition([(half,1.)])
    elif kind=='chain':
        # Final output uses the original r. The copy chain is independently
        # constrained and also consumed by a later ordinary inequality.
        predicate('ineq',[(copies[-1],.5)],rhs=1.)
        last=definition([(1,1.)])
    else:last=definition([(1,.5)])
    outputs=[row([(last,.5)])]
    return dict(old_nc=old,nc=next_col,nb=nb,frame_id=52,commands=commands,outputs=outputs,bias=np.array([.5]))


def feasible(hz,x,z):
    if any(abs(v)>1 for v in x) or any(v not in (-1,1) for v in z):return False
    for a,b,rhs,equality in ((hz.Ac,hz.Ab,hz.b,True),(hz.Auc,hz.Aub,hz.ub,False)):
        for i,value in enumerate(rhs):
            total=F(0)
            for matrix,point in ((a,x),(b,z)):
                j,k=map(int,matrix.indptr[i:i+2]);total+=sum((F(float(v))*point[int(c)] for c,v in zip(matrix.indices[j:k],matrix.data[j:k])),F(0))
            if (total!=F(float(value))) if equality else (total>F(float(value))):return False
    return True


def value(hz,x,z):
    result=[]
    for i,bias in enumerate(hz.c):
        total=F(float(bias))
        for matrix,point in ((hz.Gc,x),(hz.Gb,z)):
            a,b=map(int,matrix.indptr[i:i+2]);total+=sum((F(float(v))*point[int(c)] for c,v in zip(matrix.indices[a:b],matrix.data[a:b])),F(0))
        result.append(total)
    return result


def source_points(program):
    for inputs in product((F(-1),F(-1,2),F(0),F(1,2),F(1)),repeat=program['nb']):
        phase_choices=[(-1,1) if not x else ((-1,) if x<0 else (1,)) for x in inputs]
        for phases in product(*phase_choices):
            original=[]
            for x in inputs:original.extend((x,2*max(x,F(0))-1))
            for command in program['commands']:
                if command['kind']!='def':continue
                total=sum((F(float(v))*original[int(c)] for c,v in zip(command['cc'][:-1],command['cv'][:-1])),F(0))
                total+=sum((F(float(v))*phases[int(c)] for c,v in zip(command['bc'],command['bv'])),F(0))
                original.append((F(command['rhs'])-total)/F(float(command['cv'][-1])))
            yield inputs,original,list(phases)


def check_points(kind,program,original,candidate):
    count=0
    for inputs,x,z in source_points(program):
        compact=[x[int(i)] for i in candidate['global_ids']]
        if reconstruct(candidate,compact)!=x or not feasible(original['hz'],x,z) or not feasible(candidate['hz'],compact,z):
            raise ValueError('ordinary neural feasible point or full inverse differs')
        before=value(original['hz'],x,z);after=value(candidate['hz'],compact,z)
        if before!=after:raise ValueError('original toy-network value differs')
        expected=(sum(max(t,F(0)) for t in inputs)/2 if kind=='conv_relu'
            else (max(inputs[0],F(0))/2+F(1,4) if kind in ('nonunit','zero_hit') else max(inputs[0],F(0))))
        if after!=[expected]:raise ValueError('concrete toy ReLU/Conv function differs')
        count+=1
    return dict(feasible_full_inverse_points=count,original_toy_network_outputs_exact=True,benchmark_counterexamples=0)
