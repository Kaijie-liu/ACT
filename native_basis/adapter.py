"""Capture one small native solve, then separately map its untrusted basis.

No original LP change, presolve, scaling, free-row inference, or retry.
Exact reconstruction and the full rational checker remain separate steps.
"""
from fractions import Fraction as F
import math
from pathlib import Path
import time
from lp_sandwich.check import identity,validate_statement,inspect_bounds,matrix_rows,deadline_tick
from exact_basis.propose import manifest,col,LIMITS
from single_check_portable.execution import save_new

VERSION='1.14.0'
OPTIONS={'presolve':'off','solver':'simplex','simplex_scale_strategy':0,'threads':1,
         'parallel':'off','random_seed':0,'output_flag':False}
CAPTURES=[]

def snapshot_input(lp):
    n=len(lp['c']);rows=[]
    for kind,rhs in (('E','h'),('A','b')):
        for i,row in enumerate(matrix_rows(lp[kind],n,lambda:None)):
            right=float(F(lp[rhs][i]));rows.append({'lower':right if kind=='E' else '-inf',
                'upper':right,'entries':[[j,float(v)] for j,v in row if v]})
    return {'cost':[float(F(v)) for v in lp['c']],'offset':float(F(lp['offset'])),
        'lower':[float(F(v)) for v in lp['lower']],'upper':[float(F(v)) for v in lp['upper']],
        'rows':rows,'sense':'minimize'}

def capture(lp,statement,destination,*,deadline):
    start=time.monotonic()
    if not math.isfinite(deadline) or deadline>start+300:raise ValueError('bounded original deadline')
    tick=deadline_tick(deadline);tick()
    n=len(lp['c']);m=len(lp['b'])+len(lp['h'])
    if n>LIMITS['variables'] or m>LIMITS['equations'] or\
       len(lp['A']['data'])+len(lp['E']['data'])>LIMITS['input_nnz']:raise ValueError('analytic size cap')
    validate_statement(statement,lp,identity(statement),tick);inspect_bounds(lp,None,None,identity(statement),tick)
    root=Path(destination).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    expected=snapshot_input(lp);save_new(root/'input.json',{'lp':lp,'statement':statement,'submitted':expected})
    import highspy
    import highspy._core as core
    from hashlib import sha256
    h=highspy.Highs()
    if h.version()!=VERSION:raise ValueError('untested native version')
    def ok(status):
        if status!=highspy.HighsStatus.kOk:raise ValueError('native API status: '+str(status))
    for k,v in OPTIONS.items():ok(h.setOptionValue(k,v))
    model=highspy.HighsLp();model.num_col_=n;model.num_row_=m
    model.col_cost_=expected['cost'];model.col_lower_=expected['lower'];model.col_upper_=expected['upper']
    model.offset_=expected['offset'];model.sense_=highspy.ObjSense.kMinimize
    model.row_lower_=[-highspy.kHighsInf if r['lower']=='-inf' else r['lower'] for r in expected['rows']]
    model.row_upper_=[r['upper'] for r in expected['rows']]
    ptr=[0];indices=[];data=[]
    for r in expected['rows']:
        for j,v in r['entries']:indices.append(j);data.append(v)
        ptr.append(len(data))
    model.a_matrix_.format_=highspy.MatrixFormat.kRowwise
    model.a_matrix_.start_=ptr;model.a_matrix_.index_=indices;model.a_matrix_.value_=data
    ok(h.passModel(model))
    def snapshot():
        loaded=h.getLp();rows=[]
        for i in range(m):
            status,ix,vals=h.getRowEntries(i);ok(status)
            rows.append({'lower':'-inf' if math.isinf(loaded.row_lower_[i]) and loaded.row_lower_[i]<0 else float(loaded.row_lower_[i]),
                'upper':float(loaded.row_upper_[i]),'entries':sorted([[int(j),float(v)] for j,v in zip(ix,vals)])})
        return {'cost':list(map(float,loaded.col_cost_)),'offset':float(loaded.offset_),
            'lower':list(map(float,loaded.col_lower_)),'upper':list(map(float,loaded.col_upper_)),
            'rows':rows,'sense':'minimize' if loaded.sense_==highspy.ObjSense.kMinimize else 'unsupported'}
    before=snapshot();tick();grant=min(10.,deadline-time.monotonic())
    if grant<=0:raise TimeoutError('no native budget')
    ok(h.setOptionValue('time_limit',grant))
    actual={}
    for name in (*OPTIONS,'time_limit'):
        status,value=h.getOptionValue(name);ok(status);actual[name]=value
    save_new(root/'submission.json',{'expected':expected,'readback':before,'options':actual})
    if before!=expected or any(actual[k]!=v for k,v in OPTIONS.items()):raise ValueError('submission altered')
    entered=time.monotonic();run_status=h.run();native_seconds=time.monotonic()-entered
    basis=h.getBasis();solution=h.getSolution();info=h.getInfo();basic_status,basic_variables=h.getBasicVariables()
    record={'schema':'NATIVE_ORIGINAL_BASIS_CAPTURE_V1','lp_sha256':identity(lp),'statement_sha256':identity(statement),
        'input_sha256':identity({'lp':lp,'statement':statement}),
        'version':h.version(),'native_binary_sha256':sha256(Path(core.__file__).read_bytes()).hexdigest(),
        'options':actual,'submitted':expected,'readback_before':before,'readback_after':snapshot(),
        'run_status':str(run_status),'model_status':str(h.getModelStatus()),
        'native_objective':float(info.objective_function_value),'basis_valid':bool(basis.valid),
        'column_status':[x.name for x in basis.col_status],'row_status':[x.name for x in basis.row_status],
        'basic_variables_status':str(basic_status),'basic_variables':[int(x) for x in basic_variables],
        'value_valid':bool(solution.value_valid),'column_values':list(map(float,solution.col_value)),
        'row_values':list(map(float,solution.row_value)),'native_seconds':native_seconds,
        'seconds':time.monotonic()-start,'deadline_monotonic':deadline,'native_calls':1,
        'trusted':False,'network_SAFE':False,'network_UNSAFE':False}
    # Capture is immutable and precedes mapping, reconstruction and checking.
    save_new(root/'capture.json',record)
    CAPTURES.append({'version':record['version'],'column_status':record['column_status'],
        'row_status':record['row_status'],'model_status':record['model_status'],'native_calls':1})
    tick();return record

def map_capture(lp,statement,r,expected_capture):
    if identity(r)!=expected_capture or r['schema']!='NATIVE_ORIGINAL_BASIS_CAPTURE_V1' or\
       (r['lp_sha256'],r['statement_sha256'],r['input_sha256'])!=\
       (identity(lp),identity(statement),identity({'lp':lp,'statement':statement})):raise ValueError('capture binding')
    if r['version']!=VERSION or any(r['options'].get(k)!=v for k,v in OPTIONS.items()) or\
       set(r['options'])!=set(OPTIONS)|{'time_limit'} or not 0<r['options']['time_limit']<=10:
        raise ValueError('unsupported version/options')
    expected=snapshot_input(lp)
    if any(r[k]!=expected for k in ('submitted','readback_before','readback_after')):raise ValueError('original matrix mapping')
    def unsupported(reason):return {'status':'UNSUPPORTED_MAPPING','reason':reason,'hint':None,
        'capture_sha256':expected_capture,'network_SAFE':False,'network_UNSAFE':False}
    n=len(lp['c']);ne=len(lp['h']);na=len(lp['b'])
    if not r['basis_valid'] or not r['value_valid']:return unsupported('no valid native basis/point')
    if len(r['column_status'])!=n or len(r['row_status'])!=ne+na or len(r['column_values'])!=n:
        raise ValueError('native dimensions')
    if not all(math.isfinite(v) for v in r['column_values']):return unsupported('nonfinite native point')
    basic=[];anchors=[]
    for i,status in enumerate(r['column_status']):
        if status=='kBasic':basic.append(col('x',i))
        elif status in ('kLower','kUpper'):
            anchors.append({'column':col('x',i),'at':'lower' if status=='kLower' else 'upper'})
        else:return unsupported('unsupported structural status: '+status)
    for i,status in enumerate(r['row_status']):
        if i<ne:
            if status not in ('kLower','kUpper'):return unsupported('basic/free equality row lacks a slack coordinate')
        elif status=='kBasic':basic.append(col('slack',i-ne))
        elif status=='kUpper':anchors.append({'column':col('slack',i-ne),'at':'zero'})
        else:return unsupported('unsupported <= row status: '+status)
    if len(basic)!=ne+na:return unsupported('mapped basis count differs from augmented equations')
    c={'lp_sha256':identity(lp),'statement_sha256':identity(statement),'x':r['column_values']}
    hint=manifest(lp,statement,c,basic,anchors)
    return {'status':'MAPPED_HINT_ONLY','hint':hint,'candidate':c,'capture_sha256':expected_capture,
        'network_SAFE':False,'network_UNSAFE':False}
