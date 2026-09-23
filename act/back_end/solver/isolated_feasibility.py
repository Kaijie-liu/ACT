"""Budgeted native feasibility proposals; original ACT acceptance in the parent.

Native subprocess is in the request's process group so its outer watchdog also
owns cleanup. Only this object's child PID is killed at a query deadline.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
from scipy import sparse
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.native_feasibility_worker import digest, publish


def csr_arrays(A):
    A = A.tocsr()
    return {'data': A.data, 'indices': A.indices, 'indptr': A.indptr, 'shape': np.array(A.shape)}


def save_npz(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())


class NativeSession:
    def __init__(self, model, directory, *, command_factory=None):
        self.model, self.root = model, Path(directory)
        self.proc, self.logs = None, []
        self.queries, self.launches = [], 0
        self.command_factory = command_factory  # Controls only; production uses the fixed private worker.
        self.model_file = self.root/'base_model.npz'
        self.model_hash = None

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self.close()
        worker = Path(__file__).with_name('native_feasibility_worker.py')
        command = [sys.executable, str(worker), '--model', str(self.model_file), '--sha256', self.model_hash]
        if self.command_factory is not None:
            command = self.command_factory(command)
        out = (self.root/f'child_{self.launches:03d}.stdout').open('x')
        err = (self.root/f'child_{self.launches:03d}.stderr').open('x')
        self.logs = [out, err]
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                     text=True, env=env)  # Inherits outer-owned group, no detached grandchild.
        self.launches += 1

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc = None
        for f in self.logs:
            f.close()
        self.logs = []

    def query(self, deadline, *, scope, extra_A=None, extra_lb=None, extra_ub=None, tolerance=1e-7):
        started = time.monotonic()
        if not math.isfinite(deadline):
            raise ValueError('finite query deadline required')
        folder = self.root/f'query_{len(self.queries):03d}'
        folder.mkdir(exist_ok=False)
        token = uuid.uuid4().hex
        record = {'token': token, 'scope': scope, 'started_monotonic': started,
            'deadline_monotonic': deadline, 'feasibility_status': 'unknown', 'terminal': 'NOT_STARTED',
            'native_started': False, 'incumbent_valid': None, 'native_result': None}
        result = sh._MILPResult('unknown', None)
        try:
            if started < deadline:
                A, lb, ub = sh._combined_constraints(self.model, extra_A, extra_lb, extra_ub)
            if started < deadline and self.model_hash is None:
                save_npz(self.model_file, **csr_arrays(self.model.A), row_lb=self.model.row_lb,
                    row_ub=self.model.row_ub, var_lb=self.model.var_lb, var_ub=self.model.var_ub,
                    integrality=self.model.integrality, value_center=self.model.value_center,
                    **{'value_'+k: v for k, v in csr_arrays(self.model.value_matrix).items()})
                self.model_hash = digest(self.model_file)
            if started >= deadline:
                record['terminal'] = 'LOCAL_DEADLINE'
            elif self.model.n_var == 0:
                x = np.zeros(0)
                valid = sh._valid_milp_point(self.model, x, A, lb, ub, max(tolerance, 1e-7))
                result = sh._MILPResult('feasible' if valid else 'infeasible', x if valid else None)
                record.update(terminal='CONSTANT', incumbent_valid=valid)
            else:
                E = extra_A if extra_A is not None else sparse.csr_matrix((0, self.model.n_var))
                save_npz(folder/'extra.npz', **csr_arrays(E),
                    lb=np.asarray(extra_lb) if extra_lb is not None else np.zeros(0),
                    ub=np.asarray(extra_ub) if extra_ub is not None else np.zeros(0))
                request = {'token': token, 'scope': scope, 'model_sha256': self.model_hash,
                    'query_sha256': digest(folder/'extra.npz'), 'deadline_monotonic': deadline}
                publish(folder/'request.json', request)
                if time.monotonic() >= deadline:
                    record['terminal'] = 'CONSTRUCTION_DEADLINE'
                else:
                    self.start()
                    self.proc.stdin.write(json.dumps({'folder': str(folder), 'token': token})+'\n')
                    self.proc.stdin.flush()
                    record['terminal'] = 'TIMEOUT'
                    while time.monotonic() < deadline:
                        if (folder/'native_result.json').exists():
                            record['terminal'] = 'COMPLETED'
                            break
                        if (folder/'native_error.json').exists() or self.proc.poll() is not None:
                            record['terminal'] = 'ERROR'
                            break
                        time.sleep(min(.005, max(0, deadline-time.monotonic())))
                    if record['terminal'] == 'COMPLETED':
                        raw = json.loads((folder/'native_result.json').read_text())
                        if (any(raw[k] != request[k] for k in ('token', 'model_sha256', 'query_sha256')) or
                                not math.isfinite(raw['finished_monotonic']) or raw['finished_monotonic'] >= deadline):
                            raise ValueError('native identity/lateness')
                        record['native_result'] = raw
                        x = None
                        if raw['candidate_sha256'] is not None:
                            if digest(folder/'candidate.npz') != raw['candidate_sha256']:
                                raise ValueError('incumbent identity')
                            with np.load(folder/'candidate.npz', allow_pickle=False) as z:
                                x = z['x'].copy()
                            record['incumbent_valid'] = sh._valid_milp_point(
                                self.model, x, A, lb, ub, max(tolerance, 1e-7))
                        # EXACT original feasibility acceptance: a valid point
                        # may be accepted even at status1, status2 excludes.
                        if record['incumbent_valid']:
                            result = sh._MILPResult('feasible', x, raw['mip_node_count'])
                        elif raw['status'] == 2:
                            result = sh._MILPResult('infeasible', None, raw['mip_node_count'])
                        if time.monotonic() >= deadline:
                            record['terminal'] = 'ACCEPTANCE_DEADLINE'
                            result = sh._MILPResult('unknown', None)
        except Exception as exc:
            record.update(terminal='ERROR', exception=type(exc).__name__, message=str(exc))
            result = sh._MILPResult('unknown', None)
        finally:
            cleanup = time.monotonic()
            if record['terminal'] not in ('COMPLETED', 'CONSTANT'):
                self.close()
            record['cleanup_seconds'] = time.monotonic()-cleanup
            record['native_started'] = (folder/'native_started.json').exists()
            record['feasibility_status'] = result.status
            record['elapsed_through_cleanup_seconds'] = time.monotonic()-started
            record['artifact_hashes'] = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
            publish(folder/'receipt.json', record)
            # Cleanup, validation, artifact hashing and receipt publication are
            # charged before the caller may use a conclusive result.
            if result.status != 'unknown' and time.monotonic() >= deadline:
                record['accepted_after_receipt'] = False
                result = sh._MILPResult('unknown', None)
            else:
                record['accepted_after_receipt'] = result.status != 'unknown'
            record['returned_status'] = result.status
            record['return_elapsed_seconds'] = time.monotonic()-started
            publish(folder/'return.json', record)
            if result.status != 'unknown' and time.monotonic() >= deadline:
                record.update(returned_status='unknown', accepted_after_receipt=False,
                              terminal='RETURN_PUBLICATION_DEADLINE')
                result = sh._MILPResult('unknown', None)
                publish(folder/'late_return_rejected.json', {'reason': 'return_publication_deadline'})
            self.queries.append(record)
        return result
