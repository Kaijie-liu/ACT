"""Check process-inventory observations, not general CUDA driver correctness."""
import time

from scripts.hz_device_admission import parse, valid_uuid
from scripts.hz_propagation_supervised import finite, required_hash


def validate(record, *, invocation, gpu_uuid, producer_sha, producer_pid,
             after, deadline, simulated):
    required_hash(producer_sha); finite(after); finite(deadline)
    if (type(producer_pid) is not int or producer_pid <= 0 or not valid_uuid(gpu_uuid)
            or record.get('schema') != 'OWNED_DEVICE_RELEASE_OBSERVATION_V1'
            or record.get('invocation') != invocation or record.get('gpu_uuid') != gpu_uuid
            or record.get('producer_sha256') != producer_sha or record.get('owned_pids') != [producer_pid]
            or record.get('after') != after or record.get('deadline') != deadline
            or record.get('simulated') is not simulated):
        raise ValueError('owned release binding')
    rows = record['observations']
    if type(rows) is not list or len(rows) != 2:
        raise ValueError('two complete observations required')
    previous = after; absent = True
    for i, row in enumerate(rows):
        for field in ('start', 'end'):
            finite(row[field])
        if not previous+(0.05 if i else 0) <= row['start'] <= row['end'] < deadline:
            raise ValueError('stale/late/overlapping release observation')
        rebuilt = parse(row['gpu_text'], row['process_text'], gpu_uuid)
        if any(row.get(k) != v for k,v in rebuilt.items()):
            raise ValueError('release inventory differs from raw query')
        if producer_pid in rebuilt['compute_pids']:
            absent = False  # includes PID-reuse ambiguity; never guess it away
        previous = row['end']
    return {'status': 'OWNED_PID_ABSENT_OBSERVED' if absent else 'OWNED_PID_STILL_LISTED_OR_AMBIGUOUS',
            'release_confirmed': absent, 'next_admission_ready': False if not rows[-1]['ready'] else None,
            'last_snapshot_meets_admission_thresholds': bool(rows[-1]['ready']),
            'next_admission_established': False,
            'simulated': simulated, 'physical_driver_release_proved': False}


def simulated_record(inv, plan, producer, control):
    """Explicit fake query data; all actual candidate computation stays on CPU."""
    gpu_uuid = inv['spec']['gpu_uuid']
    record = {'schema': 'OWNED_DEVICE_RELEASE_OBSERVATION_V1', 'invocation': inv['invocation'],
              'gpu_uuid': gpu_uuid, 'producer_sha256': plan['inputs']['produce'],
              'owned_pids': [producer['pid']], 'after': inv['start']+producer['end_seconds'],
              'deadline': plan['run_deadline'], 'simulated': True, 'observations': []}
    for i in range(2):
        if i: time.sleep(.05)
        now = time.monotonic()
        foreign = producer['pid']+1  # synthetic identifier; no process is queried/signalled
        pids = ([producer['pid']] if control=='release_owned_stays' else
                [foreign] if control=='release_foreign_tenant' else [])
        gpu = f'{gpu_uuid}, 97887, {32000 if pids else 34}, {95 if pids else 0}'
        processes = '\n'.join(f'{gpu_uuid}, {pid}' for pid in pids)
        record['observations'].append({**parse(gpu, processes, gpu_uuid), 'start': now,
            'end': time.monotonic(), 'gpu_text': gpu, 'process_text': processes})
    if control=='release_wrong_request': record['invocation'] = 'stale-other-request'
    return record
