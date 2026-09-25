"""Saved-source checks only; unchanged hard budget and full segment accounting."""
import argparse
from pathlib import Path
import time

from parsed_source_reuse import execution as previous
from scoped_proof.io import PYTHON, load, save, sha, tick
from source_enclosure.format import identity
from source_cost_supervised.supervisor import supervise as watched
from types import FunctionType

POLICIES = {'none':'EXACT_SOURCE_PARSE_REUSE_R1', 'copy':'EXACT_SOURCE_PARSE_REUSE_R1',
            'readonly':'READONLY_EXACT_SOURCE_VIEW_R1'}


def validate_method(method, spec):
    if (method.get('representation') not in POLICIES
            or method.get('enabled') is not (method['representation'] != 'none')):
        raise ValueError('explicit representation and matching mode')
    previous.validate_method({k:v for k,v in method.items() if k != 'representation'}, spec)


def supervise(root, spec, method, *, budget=300., rss_limit=8*2**30, command_hook=None):
    validate_method(method, spec)
    def command(phase, folder, deadline):
        if phase == 'profile': save(folder/'method.json', method)
        args = [PYTHON, '-S', '-m', 'readonly_source.execution', phase, str(folder),
                '--method-sha256', identity(method)]
        return command_hook(phase, folder, deadline, args) if command_hook else args
    return watched(root, spec, budget=budget, rss_limit=rss_limit, command_factory=command)


def work(root, expected):
    began = time.monotonic()
    inv = load(root/'invocation.json', limit=65536)
    spec = load(root/'spec.json', inv['spec_sha256'], limit=65536)
    method = load(root/'method.json', expected, limit=65536); validate_method(method, spec)
    deadline = inv['work_deadline_monotonic']; tick(deadline)
    reading = time.monotonic(); objects = {}
    for name in ('source', 'construction'):
        record = method[name]; path = Path(record['path'])
        if path.stat().st_size != record['bytes']: raise ValueError('saved file size')
        objects[name] = load(path, record['sha256']); tick(deadline)
    read_seconds = time.monotonic()-reading; checking = time.monotonic()
    # Import cost is part of the checker segment and outer budget in ALL arms.
    if method['representation'] == 'readonly':
        from readonly_source.check import check
    else:
        from parsed_source_reuse.check import check
    def retain(value):
        save(root/'parse_stats.json', {'invocation':inv['invocation'], 'method_sha256':expected, 'stats':value})
    result = check(objects['source'], objects['construction'], invocation=objects['construction']['invocation'],
        expected_source_sha256=spec['source_sha256'], deadline=deadline, enabled=method['enabled'],
        stats_sink=retain, cache_invocation=inv['invocation'])
    check_seconds = time.monotonic()-checking
    if identity(result) != method['expected_result_sha256']: raise ValueError('original checker differential')
    save(root/'candidate.json', {'schema':'PARSED_SOURCE_CHECK_CANDIDATE_R1', 'invocation':inv['invocation'],
        'spec_sha256':inv['spec_sha256'], 'method_sha256':expected, 'result':result,
        'stats_record':{'sha256':sha(root/'parse_stats.json'), 'bytes':(root/'parse_stats.json').stat().st_size},
        'read_seconds':read_seconds, 'check_seconds':check_seconds,
        'seconds_before_candidate':time.monotonic()-began, 'complete_output_positive_proof':False})
    tick(deadline)


# All previous receipt/cost/count checks retained, without importing the parser.
_receive = FunctionType(previous.receive.__code__, dict(previous.receive.__globals__, validate_method=validate_method))
_receive.__kwdefaults__ = dict(previous.receive.__kwdefaults__)


def receive(root, expected, *, deadline=None):
    result = _receive(root, expected, deadline=deadline)
    method = load(root/'method.json', expected); stats = load(root/'parse_stats.json')['stats']
    if (stats['schema'] != 'SOURCE_PARSE_REUSE_CHECK_R1'
            or stats['parser']['policy'] != POLICIES[method['representation']]
            or stats['after_close']['policy'] != POLICIES[method['representation']]):
        raise ValueError('representation identity')
    tick(load(root/'invocation.json')['work_deadline_monotonic'] if deadline is None else deadline)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('profile','receive')); p.add_argument('root', type=Path)
    p.add_argument('--method-sha256', required=True); a = p.parse_args()
    if a.phase == 'profile': work(a.root, a.method_sha256)
    else:
        save(a.root/'received.json', receive(a.root, a.method_sha256))
        tick(load(a.root/'invocation.json')['work_deadline_monotonic'])
