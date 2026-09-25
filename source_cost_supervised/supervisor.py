"""Hard supervision, including a separate bounded candidate receiver.

Parent only reads small receipts; large hashes/JSON are checked in the watched
receiver. Terminal publication cost is measured by the enclosing caller and
must be audited before the result is accepted. No retries or late rescue.
"""
import math
import os
from pathlib import Path
import time
import uuid

from scoped_proof.io import PYTHON, load, save, sha
from scoped_proof.supervisor import execute

ENV = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
           NUMEXPR_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')


def supervise(root, spec, *, budget=300., rss_limit=8 * 2**30, command_factory=None):
    from source_cost_supervised.audit import validate_spec
    started = time.monotonic()
    validate_spec(spec)
    if (type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 300
            or type(rss_limit) is not int or not 0 < rss_limit <= 8 * 2**30):
        raise ValueError('bounded resource policy')
    root = Path(root).resolve()
    if not root.is_relative_to('/data1/Kane/MOE'): raise ValueError('workspace root required')
    root.mkdir(parents=True, exist_ok=False)
    deadline = started + budget; work_end = deadline - min(2., budget / 10)
    spec_record = save(root / 'spec.json', spec)
    inv = {'invocation': uuid.uuid4().hex, 'started_monotonic': started,
           'deadline_monotonic': deadline, 'work_deadline_monotonic': work_end,
           'budget_seconds': budget, 'rss_limit': rss_limit, 'spec_sha256': spec_record['sha256']}
    save(root / 'invocation.json', inv)
    stages = []; status = 'ERROR'; error = None; accepted = None
    try:
        for phase, module in [('profile', 'source_cost_supervised.worker'),
                              ('receive', 'source_cost_supervised.audit')]:
            begin = time.monotonic()
            command = ([PYTHON, '-S', '-m', module, str(root)] if command_factory is None
                       else command_factory(phase, root, work_end))
            stage = execute(command, root / (phase + '.log'), work_end, ENV, rss_limit)
            stage.setdefault('deadline_monotonic', work_end); stage.setdefault('cleanup_included', True)
            stage.update(phase=phase, start_seconds=begin - started, end_seconds=time.monotonic() - started)
            stages.append(stage); save(root / (phase + '_stage.json'), stage)
            if stage['status'] != 'COMPLETED': status = stage['status']; break
        else:
            accepted = load(root / 'received.json', limit=65536)
            if (accepted['invocation'] != inv['invocation'] or accepted['spec_sha256'] != inv['spec_sha256']
                    or accepted['status'] != 'PROFILE_COMPLETE_NOT_OUTPUT_PROOF'
                    or accepted['complete_output_positive_proof'] is not False):
                raise ValueError('unbound/overclaimed reception')
            status = 'COMPLETED'
    except Exception as exc:
        status = 'ERROR'; error = repr(exc); accepted = None
    if time.monotonic() >= deadline: status = 'TIMEOUT'; accepted = None
    before = time.monotonic() - started
    cost = {'schema': 'SOURCE_COST_LEDGER_V1', **inv, 'stages': stages,
            'status_before_publication': status, 'error': error,
            'seconds_before_ledger': before, 'overhead_seconds': before - sum(s['seconds'] for s in stages),
            'received_sha256': sha(root / 'received.json') if accepted else None,
            'includes': 'imports/generation/construction/all checks/publication/receipt validation/owned cleanup',
            'excludes': 'own ledger and terminal writes (enclosing wall includes); later independent audit'}
    record = save(root / 'cost.json', cost)
    elapsed = time.monotonic() - started
    terminal = {'invocation': inv['invocation'], 'spec_sha256': inv['spec_sha256'],
                'cost_sha256': record['sha256'], 'seconds_before_terminal': elapsed,
                'status': 'TIMEOUT' if elapsed >= budget else status,
                'complete_output_positive_proof': False}
    save(root / 'terminal.json', terminal)
    digest = sha(root / 'terminal.json')
    returned = time.monotonic() - started
    return {'terminal_sha256': digest, 'seconds_including_terminal': returned,
            'status': 'TIMEOUT' if returned >= budget else terminal['status']}
