"""Bounded analytic passModel/readback only. run() is explicitly forbidden."""
from pathlib import Path
import math
from unittest.mock import patch
from single_check_portable.execution import save_new
from portable_proof.runtime import digest
from sparse_basis.native import OPTIONS, VERSION

THRESHOLDS=('small_matrix_value','large_matrix_value','infinite_cost','infinite_bound')


def runtime():
    import highspy
    import highspy._core as core
    h=highspy.Highs()
    if h.version()!=VERSION:raise ValueError('runtime version drift')
    opts={}
    for k in THRESHOLDS:
        status,value=h.getOptionValue(k)
        if status!=highspy.HighsStatus.kOk:raise ValueError('cannot read option')
        opts[k]=value
    return {'version':h.version(),'binary_sha256':digest(Path(core.__file__).read_bytes()),'defaults':opts}


def model(a, equality=False):
    return {'cost':[-1.,0.],'offset':0.,'lower':[0.,0.],'upper':[1.,1.],
            'rows':[{'lower':0. if equality else '-inf','upper':0.,'entries':[[0,1.],[1,a]]}],
            'sense':'minimize'}


def readback(h, hs):
    lp=h.getLp(); mat=lp.a_matrix_
    rows=[[] for _ in range(lp.num_row_)]
    colwise=mat.format_==hs.MatrixFormat.kColwise
    if not colwise and mat.format_!=hs.MatrixFormat.kRowwise:raise ValueError('unexpected layout')
    for j in range(lp.num_col_ if colwise else lp.num_row_):
        for k in range(mat.start_[j],mat.start_[j+1]):
            r,c=(int(mat.index_[k]),j) if colwise else (j,int(mat.index_[k]))
            rows[r].append([c,float(mat.value_[k])])
    return {'cost':list(map(float,lp.col_cost_)),'offset':float(lp.offset_),
            'lower':list(map(float,lp.col_lower_)),'upper':list(map(float,lp.col_upper_)),
            'rows':[{'lower':'-inf' if lp.row_lower_[i]==-math.inf else float(lp.row_lower_[i]),
                     'upper':float(lp.row_upper_[i]),'entries':sorted(row)} for i,row in enumerate(rows)],
            'sense':'minimize' if lp.sense_==hs.ObjSense.kMinimize else 'OTHER'}


def compare(a,b):
    if len(a['rows'])!=len(b['rows']):raise ValueError('row count changed')
    changed=[]
    for i,(x,y) in enumerate(zip(a['rows'],b['rows'])):
        xx,yy=dict(x['entries']),dict(y['entries'])
        for col in sorted(set(xx)|set(yy)):
            if xx.get(col)!=yy.get(col):
                changed.append({'row':i,'column':col,'before':xx.get(col),'after':yy.get(col)})
    return {'matrix_changes':changed,
            'other_fields_unchanged':all(a[k]==b[k] for k in ('cost','offset','lower','upper','sense')),
            'row_bounds_unchanged':all((x['lower'],x['upper'])==(y['lower'],y['upper']) for x,y in zip(a['rows'],b['rows']))}


def submit(case, intended, root):
    import highspy as hs
    # This tool cannot be used to import any of the real 7k–9k-variable models.
    n,m=len(intended['cost']),len(intended['rows'])
    if not 1<=n<=4 or not 1<=m<=4 or sum(len(r['entries']) for r in intended['rows'])>16:
        raise ValueError('analytic-only dimensions')
    if len(intended['lower'])!=n or len(intended['upper'])!=n:raise ValueError('box size')
    root.mkdir(exist_ok=False)
    with patch.object(hs.Highs,'run',side_effect=AssertionError('optimization prohibited')) as forbidden:
        h=hs.Highs()
        for k,v in OPTIONS.items():
            if h.setOptionValue(k,v)!=hs.HighsStatus.kOk:raise ValueError('option rejection')
        # Logging is the only override. No numeric threshold/options are changed.
        for k,v in {'output_flag':True,'log_to_console':False,'log_file':str(root/'native.log')}.items():
            if h.setOptionValue(k,v)!=hs.HighsStatus.kOk:raise ValueError('logging option rejection')
        options={k:h.getOptionValue(k)[1] for k in (*OPTIONS,*THRESHOLDS)}
        lp=hs.HighsLp();lp.num_col_,lp.num_row_=n,m
        lp.col_cost_,lp.col_lower_,lp.col_upper_=intended['cost'],intended['lower'],intended['upper']
        lp.offset_=intended['offset'];lp.sense_=hs.ObjSense.kMinimize
        lp.row_lower_=[-hs.kHighsInf if r['lower']=='-inf' else r['lower'] for r in intended['rows']]
        lp.row_upper_=[r['upper'] for r in intended['rows']]
        ptr,idx,data=[0],[],[]
        for row in intended['rows']:
            for j,x in row['entries']:idx.append(j);data.append(x)
            ptr.append(len(idx))
        lp.a_matrix_.format_=hs.MatrixFormat.kRowwise
        lp.a_matrix_.start_,lp.a_matrix_.index_,lp.a_matrix_.value_=ptr,idx,data
        status=h.passModel(lp)
        actual=readback(h,hs)
        h.setOptionValue('log_file','')
        forbidden.assert_not_called()
    result={'case':case,'status':str(status),'intended':intended,'readback':actual,
            'comparison':compare(intended,actual),'options':options,
            'log':(root/'native.log').read_text(),'optimization_calls':0,'analytic_import_calls':1,
            'real_native_imports':0,'native_warning_is_accepted_as_certificate':False}
    save_new(root/'result.json',result)
    return result
