"""Private persistent SciPy worker. No ACT/torch import or acceptance authority.

One parent owns stdin and all paths. Model/query arrays use NPZ without pickle;
every candidate binds their hashes and a unique query token. Partial files are
not terminal evidence. The parent enforces wall deadlines and validates points.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def publish(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix+'.part')
    with tmp.open('x') as f:
        json.dump(value, f, allow_nan=False, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def finite_or_none(value):
    import math
    return float(value) if value is not None and math.isfinite(float(value)) else None


def serve(model_file, model_hash):
    import numpy as np
    from scipy import sparse
    from scipy.optimize import milp, Bounds, LinearConstraint
    if digest(model_file) != model_hash:
        raise ValueError('base model binding')
    with np.load(model_file, allow_pickle=False) as z:
        model = {k: z[k].copy() for k in z.files}
    A = sparse.csr_matrix((model['data'], model['indices'], model['indptr']), shape=tuple(model['shape']))
    for line in sys.stdin:
        msg = json.loads(line)
        folder = Path(msg['folder'])
        req = json.loads((folder/'request.json').read_text())
        try:
            query_file = folder/'extra.npz'
            if req['token'] != msg['token'] or req['model_sha256'] != model_hash or digest(query_file) != req['query_sha256']:
                raise ValueError('query binding')
            with np.load(query_file, allow_pickle=False) as z:
                extra = {k: z[k].copy() for k in z.files}
            E = sparse.csr_matrix((extra['data'], extra['indices'], extra['indptr']), shape=tuple(extra['shape']))
            combined = sparse.vstack([A, E], format='csr') if E.shape[0] else A
            lo = np.concatenate([model['row_lb'], extra['lb']])
            hi = np.concatenate([model['row_ub'], extra['ub']])
            remaining = req['deadline_monotonic']-time.monotonic()
            if remaining <= .001:
                raise TimeoutError('native entry exhausted')
            # This is still a soft native limit; the parent supplies the hard stop.
            options = {'presolve': True, 'time_limit': remaining, 'mip_rel_gap': 0.0}
            publish(folder/'native_started.json', {'token': req['token'], 'model_sha256': model_hash,
                'query_sha256': req['query_sha256'], 'options': options,
                'started_monotonic': time.monotonic(), 'n_integral': int(np.count_nonzero(model['integrality']))})
            remaining = req['deadline_monotonic']-time.monotonic()
            if remaining <= .001:
                raise TimeoutError('publication exhausted')
            # Publication cost is conservatively subtracted, not extra native time.
            options['time_limit'] = min(options['time_limit'], remaining)
            native_start = time.monotonic()
            raw = milp(c=np.zeros(model['var_lb'].size), integrality=model['integrality'],
                bounds=Bounds(model['var_lb'], model['var_ub']),
                constraints=LinearConstraint(combined, lo, hi) if combined.shape[0] else None,
                options=options)
            ended = time.monotonic()
            candidate = None
            if raw.x is not None:
                file = folder/'candidate.npz'
                with file.open('xb') as f:
                    np.savez(f, x=raw.x)
                    f.flush()
                    os.fsync(f.fileno())
                candidate = digest(file)
            publish(folder/'native_result.json', {'token': req['token'], 'model_sha256': model_hash,
                'query_sha256': req['query_sha256'], 'status': int(raw.status), 'success': bool(raw.success),
                'message': str(raw.message), 'candidate_sha256': candidate,
                'fun': finite_or_none(getattr(raw, 'fun', None)),
                'mip_dual_bound': finite_or_none(getattr(raw, 'mip_dual_bound', None)),
                'mip_gap': finite_or_none(getattr(raw, 'mip_gap', None)),
                'mip_node_count': int(getattr(raw, 'mip_node_count', 0) or 0),
                'effective_options': options, 'native_seconds': ended-native_start,
                'finished_monotonic': time.monotonic()})
        except Exception as exc:
            publish(folder/'native_error.json', {'token': req.get('token'),
                'exception': type(exc).__name__, 'message': str(exc),
                'finished_monotonic': time.monotonic()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    serve(a.model, a.sha256)
