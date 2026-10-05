"""Conservative read-only extraction; never diagnose ENOMEM's unique cause."""
import re

TRACE_CAP = 16 * 1024**2
LINE_CAP = 4096
EVENT_CAP = 1024
AS_CAP = 16 * 1024**3
PREFIX = re.compile(r'^(\d+)\s+(\d+\.\d+)\s+(.*)$')
CALL = re.compile(r'^(mmap|mremap|brk|ioctl)\((.*)\)\s+=\s+(.*)$')
EXIT = re.compile(r'^\+\+\+ exited with \d+ \+\+\+$')
SIGNAL = re.compile(r'^--- SIG[A-Z0-9]+(?: .*)? ---$')


def summarize(lines, *, worker_pid, tracer_exit, trace_bytes, worker_exit,
              log_error=False):
    """Require a complete worker exit; retain positive observations regardless.

    Resumed syscall syntax is not reassembled. Its presence therefore makes
    absence-based conclusions unavailable, rather than silently dropping it.
    mmap argument 2 and mremap argument 3 are requested mapping sizes on Linux.
    brk requests are retained as observations, not guessed failures.
    """
    complete = (type(worker_pid) is int and type(tracer_exit) is int
                and tracer_exit == worker_exit and tracer_exit in (0, 1)
                and 0 <= trace_bytes < TRACE_CAP and not log_error)
    events, exit_seen, lines_seen, bytes_seen = [], False, 0, 0
    for line in lines:
        lines_seen += 1
        bytes_seen += len(line.encode('utf-8'))
        if bytes_seen > TRACE_CAP or len(line) > LINE_CAP:
            complete = False
            break
        match = PREFIX.match(line.rstrip('\n'))
        if not match:
            complete = False
            continue
        pid, stamp, body = match.groups()
        if int(pid) == worker_pid and body.startswith(
                '+++ exited with ' + str(worker_exit) + ' +++'):
            exit_seen = True
        if '<unfinished ...>' in body or ' resumed>' in body:
            complete = False
        call = CALL.match(body)
        if not call:
            if not EXIT.fullmatch(body) and not SIGNAL.fullmatch(body):
                complete = False
            continue
        name, arguments, result = call.groups()
        if name == 'brk' or result.startswith('-1 '):
            if len(events) >= EVENT_CAP:
                complete = False
                continue
            event = dict(pid=int(pid), timestamp_s=stamp, syscall=name,
                         arguments=arguments, result=result,
                         failed=result.startswith('-1 '), requested_bytes=None,
                         request_exceeds_as_cap=False)
            if name in ('mmap', 'mremap'):
                parts = arguments.split(',')
                index = 1 if name == 'mmap' else 2
                if len(parts) > index and parts[index].strip().isdigit():
                    size = int(parts[index].strip())
                    event['requested_bytes'] = size
                    event['request_exceeds_as_cap'] = size > AS_CAP
            events.append(event)
    complete = complete and exit_seen and bytes_seen == trace_bytes
    oversized = [event for event in events
                 if event['request_exceeds_as_cap'] and event['failed']
                 and 'ENOMEM' in event['result']]
    return dict(trace_complete=complete, worker_exit_seen=exit_seen,
                lines_seen=lines_seen, bytes_seen=bytes_seen, events=events,
                failed_reservation_exceeds_total_as_cap=bool(oversized),
                unique_failure_cause_proved=False,
                absence_claim_authorized=False,
                interpretation=('observed failed reservation cannot fit AS16GiB'
                    if oversized else 'no unique cause established'))
