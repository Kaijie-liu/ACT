"""New construction/source-check entry, inheriting one absolute deadline."""
import argparse
from pathlib import Path
import sys
from scoped_proof.io import load, save, sha, tick, Events
from scoped_proof.evidence import bind_source
from source_enclosure.format import identity
from scoped_parse_proof.contract import policy, validate_receipt


def work(phase, root, deadline):
    invocation = load(root/'invocation.json')
    spec = load(root/'spec.json', invocation['spec_file_sha256'])
    p = policy(spec); tick(deadline)
    events = Events(root, phase, invocation['started_monotonic'])
    if phase == 'construct':
        from source_construction_lab.build import construct
        doc = events.call('read_source', lambda: load(root/'source.json'))
        digest = bind_source(doc, spec['scope'])
        def emit(row): events.emit(row['event'], **{k:v for k,v in row.items() if k != 'event'})
        bundle, report = construct(doc, expected_source_sha256=digest,
            deadline=deadline, mode=p['mode'], emit=emit)
        record = events.call('serialize_construction', lambda: save(root/'construction.json', bundle))
        tick(deadline)
        receipt = {'invocation': invocation['invocation'], 'spec_sha256': invocation['spec_file_sha256'],
            'policy_sha256': identity(p), 'report': report, 'construction': record,
            'source': {'sha256': sha(root/'source.json'), 'bytes': (root/'source.json').stat().st_size}}
        events.call('serialize_construction_receipt', lambda: save(root/'construction_receipt.json', receipt))
    elif phase == 'source_check':
        accepted = events.call('check_construction_receipt', lambda: validate_receipt(root))
        events.call('serialize_construction_acceptance', lambda: save(root/'construction_acceptance.json', accepted))
        from scoped_proof.worker import work as original
        original(phase, root, deadline)
        if any(n.startswith(('source_construction_lab', 'scoped_source.build')) for n in sys.modules):
            raise ValueError('independent checker imported producer/cache')
    else:
        raise ValueError('only separately versioned construction/check entry')
    tick(deadline)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('construct','source_check'))
    p.add_argument('root',type=Path); p.add_argument('--deadline',type=float,required=True); a=p.parse_args()
    if not sys.flags.no_site: p.error('construction and independent check require python -S')
    if a.phase == 'source_check':
        def forbid(event, args):
            if event == 'import' and (args[0].split('.')[0] in ('torch','numpy','scipy','highspy','act','gurobipy','source_construction_lab') or args[0] == 'scoped_source.build'):
                raise ImportError('checker cannot import model, solver or construction adapter')
            if event.startswith(('subprocess.', 'socket.')) or event in ('os.system','os.fork','os.exec'):
                raise PermissionError('checker cannot execute external work')
        sys.addaudithook(forbid)
    work(a.phase,a.root.resolve(),a.deadline)
