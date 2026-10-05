"""Fixed ordinary general-scalar chains, shared Add and Conv/ReLU programs."""
from fractions import Fraction as F
from functools import reduce
import operator
import numpy as np
from experiments.neural_hz_20260831.c52_neural_source_fixtures_v1 import row,source_points,feasible as old_feasible,value as old_value
from experiments.neural_hz_20260831.c54_scalar_hz_audit_v1 import reconstruct,feasible,value

WEIGHTS=(.9,-.8,.7,-.6,.9,.8,-.7,-.6)


def fixture(kind,width=128):
    if kind not in ('chain','shared_add','conv_relu','zero_hit') or width!=128:raise ValueError('fixed ordinary cohort required')
    old=4 if kind=='conv_relu' else 2;nb=old//2;commands=[];next_col=old
    def predicate(kind,continuous=(),binary=(),rhs=0.,column=-1):
        commands.append(dict(kind=kind,uid=100+len(commands),column=column,**row(continuous,binary,rhs)))
    for phase in range(nb):
        x,r=2*phase,2*phase+1
        predicate('ineq',[(r,1.)],[(phase,-1.)]);predicate('ineq',[(x,-2.),(r,1.)],[(phase,1.)]);predicate('ineq',[(x,2.),(r,-1.)],rhs=1.)
    def define(terms):
        nonlocal next_col
        col=next_col;next_col+=1;predicate('def',[(p,-a) for p,a in terms]+[(col,1.)],column=col)
        return col
    if kind=='zero_hit':last=1;out=.5
    elif kind=='shared_add':
        merged=[]
        for i in range(64):
            a=define([(1,.9)]);b=define([(1,.7)]);merged.append(define([(a,.3),(b,.4)]))
        last=define([(col,1/128) for col in merged]);out=1.
    else:
        ends=[]
        for i in range(16):
            parent=1+2*(i%nb)
            for weight in WEIGHTS:parent=define([(parent,weight)])
            ends.append(parent)
        if kind=='chain':last=define([(col,1/32) for col in ends]);out=1.
        else:last=define([(ends[0],.25),(ends[1],.5),(ends[2],.25)]);out=.5
    return dict(old_nc=old,nc=next_col,nb=nb,frame_id=54,commands=commands,outputs=[row([(last,out)])],bias=np.array([.5]))


def check_points(kind,program,reference,state):
    product=reduce(operator.mul,(F(v) for v in WEIGHTS),F(1));count=0
    for inputs,original,binary in source_points(program):
        compact=[original[int(i)] for i in state['global_ids']]
        if reconstruct(state,compact)!=original or not old_feasible(reference['hz'],original,binary) or not feasible(state,compact,binary):raise ValueError('full inverse or feasible toy point differs')
        before=old_value(reference['hz'],original,binary);after=value(state,compact,binary)
        if kind=='conv_relu':expected=F(1,2)+product*(sum(max(x,F(0)) for x in inputs)-1)/2
        else:
            factor=F(1) if kind=='zero_hit' else (F(.3)*F(.9)+F(.4)*F(.7) if kind=='shared_add' else product)
            expected=F(1,2)+factor*(2*max(inputs[0],F(0))-1)/2
        if before!=after or after!=[expected]:raise ValueError('independent ordinary toy network function differs')
        count+=1
    return dict(feasible_full_inverse_points=count,exact_independent_toy_function_outputs=True,benchmark_counterexamples=0)
