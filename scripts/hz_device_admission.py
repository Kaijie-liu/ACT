"""Read-only, bounded GPU observations. Not an exclusive reservation."""
import math
import re
import subprocess
import time

UUID_PATTERN = r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'


def valid_uuid(value):
    return type(value) is str and re.fullmatch(UUID_PATTERN, value) is not None


def parse(gpu_text, process_text, gpu_uuid):
    if not valid_uuid(gpu_uuid): raise ValueError('invalid GPU UUID')
    rows=[line.strip().split(',') for line in gpu_text.splitlines() if line.strip()]
    matches=[[v.strip() for v in row] for row in rows if row[0].strip()==gpu_uuid]
    if len(matches)!=1 or len(matches[0])!=4: raise ValueError('missing/duplicate GPU')
    row=matches[0]
    total,used,util=(int(v) for v in row[1:])
    if not 0<=used<=total or total<=0 or not 0<=util<=100: raise ValueError('invalid GPU counters')
    pids=[]
    for line in process_text.splitlines():
        if not line.strip(): continue
        fields=[v.strip() for v in line.split(',')]
        if len(fields)!=2 or not valid_uuid(fields[0]) or not fields[1].isdigit():
            raise ValueError('unknown compute-process inventory')
        if fields[0]==gpu_uuid:
            if int(fields[1])<=0: raise ValueError('invalid PID')
            pids.append(int(fields[1]))
    return {'gpu_uuid':gpu_uuid,'total_mib':total,'used_mib':used,'utilization_percent':util,
            'compute_pids':sorted(pids),'ready':not pids and util<=5 and total-used>=4096}


def snapshot(gpu_uuid, deadline):
    start=time.monotonic()
    def query(fields):
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError('resource admission deadline')
        return subprocess.run(['nvidia-smi',fields,'--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=min(1.,remaining)).stdout
    gpu=query('--query-gpu=uuid,memory.total,memory.used,utilization.gpu')
    processes=query('--query-compute-apps=gpu_uuid,pid')
    result=parse(gpu,processes,gpu_uuid)
    result.update(start=start,end=time.monotonic(),gpu_text=gpu,process_text=processes)
    if result['end']>=deadline: raise TimeoutError('late admission observation')
    return result


def validate(record, invocation, spec, deadline):
    if (record.get('invocation')!=invocation or record.get('device')!=spec['device']
            or record.get('gpu_uuid')!=spec['gpu_uuid'] or record.get('deadline')!=deadline
            or record.get('exclusive_reservation') is not False):
        raise ValueError('admission binding')
    if spec['device']=='cpu':
        if record.get('status')!='CPU_NO_CUDA' or record.get('observations')!=[] or record.get('simulated') is not False:
            raise ValueError('CPU admission')
        return
    obs=record.get('observations')
    if type(obs) is not list or not 1<=len(obs)<=2: raise ValueError('admission inventory')
    last=None
    for item in obs:
        rebuilt=parse(item['gpu_text'],item['process_text'],spec['gpu_uuid'])
        if any(item.get(k)!=v for k,v in rebuilt.items()): raise ValueError('observation does not match raw counters')
        if not all(type(item[k]) in (int,float) and math.isfinite(item[k]) for k in ('start','end')):
            raise ValueError('observation clock')
        if not item['start']<=item['end']<deadline or last is not None and item['start']<last+.2:
            raise ValueError('stale/overlapping admission')
        last=item['end']
    ready=len(obs)==2 and all(o['ready'] for o in obs)
    if record.get('status')!=('READY' if ready else 'RESOURCE_UNAVAILABLE'):
        raise ValueError('false admission')
    simulated=spec['control']=='admission_busy_stub'
    if record.get('simulated') is not simulated or simulated and ready: raise ValueError('stub cannot admit CUDA')


def observe(spec, inv, deadline):
    record={'invocation':inv['invocation'],'device':spec['device'],'gpu_uuid':spec['gpu_uuid'],
            'deadline':deadline,'exclusive_reservation':False,'simulated':False,'observations':[]}
    if spec['device']=='cpu': record['status']='CPU_NO_CUDA'; return record
    for _ in range(2):
        if spec['control']=='admission_busy_stub':
            now=time.monotonic(); gpu=f"{spec['gpu_uuid']}, 10000, 10, 0"
            processes=f"{spec['gpu_uuid']}, 99999"
            item={**parse(gpu,processes,spec['gpu_uuid']),'start':now,'end':now,
                  'gpu_text':gpu,'process_text':processes}
            record['simulated']=True
        else: item=snapshot(spec['gpu_uuid'],deadline)
        record['observations'].append(item)
        if not item['ready']: break
        if len(record['observations'])==1: time.sleep(.2)
    record['status']='READY' if len(record['observations'])==2 and all(i['ready'] for i in record['observations']) else 'RESOURCE_UNAVAILABLE'
    validate(record,inv['invocation'],spec,deadline)
    return record
