"""Pure synthetic parser checks: no CUDA import, subprocess or filesystem write."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'd020_trace_summary_tests', Path(__file__).with_name('trace_summary.py'))
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)


def report(text, **changes):
    values = dict(worker_pid=7, tracer_exit=1, worker_exit=1,
                  trace_bytes=len(text.encode()), log_error=False)
    values.update(changes)
    return PARSER.summarize(text.splitlines(keepends=True), **values)


def test_large_failed_reservation_is_positive_not_unique_evidence():
    result = report('7 123.000 mmap(NULL, 68719476736, PROT_NONE, '
                    'MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = -1 ENOMEM '
                    '(Cannot allocate memory) <0.000001>\n'
                    '7 123.001 +++ exited with 1 +++\n')
    assert result['trace_complete']
    assert result['failed_reservation_exceeds_total_as_cap']
    assert result['events'][0]['requested_bytes'] == 68719476736
    assert not result['unique_failure_cause_proved']


def test_missing_exit_and_unfinished_trace_are_incomplete():
    result = report('7 123.000 ioctl(3, 0x1000, 0x0 <unfinished ...>\n')
    assert not result['trace_complete']
    assert not result['absence_claim_authorized']
    malformed = report('7 123.000 mmap BROKEN\n'
                       '7 123.001 +++ exited with 1 +++\n')
    assert not malformed['trace_complete']


def test_small_enomem_and_brk_do_not_establish_address_space_cause():
    result = report('7 123.000 mmap(NULL, 4096, PROT_NONE, '
                    'MAP_PRIVATE, -1, 0) = -1 ENOMEM (Cannot allocate memory)\n'
                    '7 123.001 brk(0x123456) = 0x123456\n'
                    '7 123.002 +++ exited with 1 +++\n')
    assert result['trace_complete']
    assert not result['failed_reservation_exceeds_total_as_cap']
    assert not result['events'][1]['failed']


def test_log_cap_and_tracer_errors_never_admit_complete_trace():
    text = '7 123.001 +++ exited with 1 +++\n'
    assert not report(text, trace_bytes=PARSER.TRACE_CAP)['trace_complete']
    assert not report(text, log_error=True)['trace_complete']
    assert not report(text, tracer_exit=-9)['trace_complete']
