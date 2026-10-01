"""Owned workers for the fixed, CPU-only support supervision controls."""
import argparse
from pathlib import Path
import subprocess
import time

from scoped_proof.io import PYTHON, Events, load, save, tick
from source_enclosure.format import identity
from scripts.hz_batch_support_supervised import bind, required_hash, validate_spec, receive, LIMIT


def fixture(case):
    """Creation is inside the charged producer; no archived candidates are read."""
    import numpy as np
    import scipy.sparse as sp
    from act.back_end.solver.solver_hz import SparseHZono
    def query(key='plus-min', q=(1,), offset='1/4', side='min'):
        return {'id': key, 'q': list(q), 'offset': offset, 'side': side}
    if case in ('guarded_two_sides_positive', 'guarded_nonpositive'):
        hz = SparseHZono(c=np.zeros(1), Gc=sp.csr_matrix([[1.]]), Gb=sp.csr_matrix((1,0)),
            Ac=sp.csr_matrix((0,1)), Ab=sp.csr_matrix((0,0)), b=np.array([]),
            Auc=sp.csr_matrix([[1.],[-1.]]), Aub=sp.csr_matrix((2,0)), ub=np.zeros(2), frame_id=314)
        queries = ([query(offset='-1/4')] if case=='guarded_nonpositive' else
                   [query(sign+'-'+side, (value,), side=side)
                    for sign,value in (('plus',1),('minus',-1)) for side in ('min','max')])
    elif case == 'equality_coupled':
        hz = SparseHZono(c=np.zeros(2), Gc=sp.eye(2, format='csr'), Gb=sp.csr_matrix((2,0)),
            Ac=sp.csr_matrix([[1.,-1.]]), Ab=sp.csr_matrix((1,0)), b=np.zeros(1))
        queries = [query(q=(1,-1))]
    elif case == 'private_binary_relaxation':
        hz = SparseHZono(c=np.zeros(2), Gc=sp.csr_matrix([[1.],[1.]]), Gb=sp.eye(2,format='csr'),
            Ac=sp.csr_matrix((0,1)), Ab=sp.csr_matrix((0,2)), b=np.array([]), frame_id=314)
        queries = [query(side,(1,-1),0,side) for side in ('min','max')]
    else:
        raise ValueError('fixed fixture')
    return hz, queries, {'request':'synthetic-fixed-controls','domain':'factor-box-minus-plus-one','guard':'two-sided-zero'}


def run(phase, root, deadline, invocation_sha, payload_sha=None, stdout_sha=None):
    spec = load(root/'spec.json')
    validate_spec(spec)
    inv = bind(root, spec, invocation_sha)
    expected_end = inv['work_deadline']-min(5.,inv['budget']/4) if phase=='produce' else inv['work_deadline']
    if deadline != expected_end: raise ValueError('worker deadline extension/change')
    tick(deadline)
    events = Events(root, phase, inv['start'])
    fault = spec['control']
    def delay(name):
        save(root/(name+'_unaccepted.json'), {'control':name,'invocation':inv['invocation'],'accepted':False})
        time.sleep(10)
    if fault == phase+'_delay': delay(phase)
    if phase == 'produce':
        if fault == 'descendant':
            child = subprocess.Popen([PYTHON,'-B','-c','import time; time.sleep(10)'])
            save(root/'descendant_control.json',{'pid':child.pid,'invocation':inv['invocation']})
            return
        hz, queries, context = events.call('imports_and_hz_creation', lambda: fixture(spec['case']))
        import torch
        torch.set_num_threads(1)
        from act.back_end.moe.batched_support import prepare_batch, propose_batch
        def prepare():
            if fault == 'prepare_delay': delay('prepare')
            return prepare_batch(hz,queries,context=context,deadline=deadline)
        batch = events.call('hz_export_and_validation', prepare)
        if identity(batch) != spec['batch_sha256']: raise ValueError('created HZ/request differs from frozen case')
        events.call('batch_publication', lambda: save(root/'batch.json', batch))
        if fault == 'candidate_exception':
            events.emit('FAULT',operation=fault)
            raise RuntimeError('controlled candidate exception')
        candidates = events.call('multiobjective_candidates', lambda: propose_batch(
            batch,expected_batch_sha256=spec['batch_sha256'],deadline=deadline))
        if fault == 'partial_candidate': candidates['entries'].pop()
        if fault == 'wrong_query': batch['queries'][0]['side'] = 'max'
        payload = {'invocation':inv['invocation'], 'request_sha256':identity(spec),
                   'batch_sha256':spec['batch_sha256'], 'batch':batch, 'candidates':candidates}
        if fault == 'wrong_invocation': payload['invocation'] = 'different-request'
        if fault == 'serialization_delay': delay('serialization')
        events.call('candidate_serialization', lambda: save(root/'payload.json',payload))
    elif phase == 'check':
        if fault == 'check_exception':
            events.emit('FAULT',operation=fault)
            raise RuntimeError('controlled check exception')
        if fault == 'missing_stdout':
            save(root/'missing_stdout_unaccepted.json',{'invocation':inv['invocation'],'control':fault,'accepted':False})
            return
        def check():
            payload = load(root/'payload.json', required_hash(payload_sha), limit=LIMIT)
            if (payload['invocation'] != inv['invocation'] or payload['request_sha256'] != identity(spec)
                    or payload['batch_sha256'] != spec['batch_sha256']):
                raise ValueError('candidate invocation/request binding')
            from act.back_end.moe.check_batched_support import check_batch
            result = check_batch(payload['batch'],payload['candidates'],expected_batch_sha256=spec['batch_sha256'],deadline=deadline)
            return {'schema':'HZ_BATCH_CHECK_OUTPUT_V1','invocation':inv['invocation'],
                    'request_sha256':identity(spec),'batch_sha256':spec['batch_sha256'],
                    'payload_sha256':payload_sha,'result':result}
        result = events.call('independent_exact_reception', check)
        events.call('checker_output_publication', lambda: save(root/'check.stdout',result))
    elif phase == 'receive':
        if fault == 'rewrite_stdout':
            # Controlled mutation of this invocation only; parent already anchored bytes.
            value = load(root/'check.stdout')
            value['result']['results'][0]['bound'] = '999'
            import json
            (root/'check.stdout').write_text(json.dumps(value))
        result = events.call('identity_and_full_roster_reception', lambda: receive(
            root,spec,invocation_sha,payload_sha,stdout_sha,deadline))
        if fault == 'late_publish':
            save(root/'late_candidate_unaccepted.json',result)
            delay('late_publish')
        events.call('accepted_publication', lambda: save(root/'accepted.json',result))
    else:
        raise ValueError('worker phase')
    tick(deadline)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase'); p.add_argument('root',type=Path)
    p.add_argument('--deadline',type=float,required=True); p.add_argument('--invocation-sha',required=True)
    p.add_argument('--payload-sha'); p.add_argument('--stdout-sha')
    a=p.parse_args()
    run(a.phase,a.root,a.deadline,a.invocation_sha,a.payload_sha,a.stdout_sha)
