"""Default-off V2 full direct path, fixed proposer only; use hard supervisor.

This cooperative inner function cannot stop a blocked stub on its own.
Production returns before fresh replay, releasing source/root/candidate owners.
"""
import gc
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY, compact, identity, require, sha
from scoped_source.tensor_rows import TensorRows
from scoped_source.direct_source_build import build
from scoped_source.check_direct_source import CheckedDirectForest
from scoped_source.direct_candidates import Driver
from scoped_source.direct_lifecycle import Lifecycle, FAULTS, inject
from scoped_source.direct_method import validate, is_shared, paged
from scoped_source.scaled_endpoints import contract
from scoped_source.wire_archive import Archive, receive_file
from scoped_source.direct_lifecycle_package import code_files, check_package, manifest_schema, bundle_schema

RESULT_FAULTS = ('after_result_page_error', 'after_result_page_block', 'index_publication_expired')
EXEC_FAULTS = FAULTS + ('replay_block', 'after_tile_error', 'before_transfer', 'after_transfer') + RESULT_FAULTS


def execute(root, source_root, source, config, invocation, code_root, *, work,
            enabled=False, fault='', archive_limit=POLICY['total_bytes'],
            member_limit=POLICY['members'], progress=None, owned_source=None):
    require(enabled is True, 'lifecycle execution is opt-in')
    method = validate(config)
    shared=is_shared(method)
    streaming=paged(method)
    require(fault not in RESULT_FAULTS or streaming, 'result faults require paged method')
    def error(exc):
        if streaming:
            from scoped_source.result_stream import error_code
            return error_code(exc)
        return type(exc).__name__ + ': ' + str(exc)[:500]
    if shared:
        from scoped_source.shared_archive import OwnedSource
        require(type(owned_source) is OwnedSource and owned_source.root==Path(source_root)
                and owned_source.source==source, 'new shared path requires owned source')
    else: require(owned_source is None, 'legacy copy ownership unchanged')
    sha(invocation)
    require(fault in EXEC_FAULTS, 'registered lifecycle execution fault')
    require(type(member_limit) is int and 0 < member_limit <= POLICY['members'], 'remaining member ceiling')
    start = time.monotonic()
    initial = dict(work.counts)
    phases, snapshots = [], {}
    archive = None

    class RequestArchive(Archive):
        def put(self, *args, **kw):
            require(len(self.attempted) < member_limit, 'whole request member reserve')
            return super().put(*args, **kw)

    def phase(name, fn, *, cleanup=False):
        before = time.monotonic()
        counts = dict(work.counts)
        complete = False
        try:
            if not cleanup:
                work.tick()
            if progress and not cleanup:
                progress(name, 'ENTER', work)
            value = fn()
            work.tick()
            complete = True
            return value
        finally:
            phases.append(dict(name=name, complete=complete, seconds=time.monotonic() - before,
                               work={k:work.counts[k] - v for k,v in counts.items()}))
            if progress and not cleanup:
                progress(name, 'EXIT' if complete else 'ERROR', work)

    def production():
        # Local source objects die before the fresh online checker is started.
        f = raw = records = d = sink = None
        life = Lifecycle(method, fault if fault in FAULTS else '')
        primary = None
        try:
            def copy_source():
                reader = TensorRows(source_root, source, deadline=work.deadline)
                for name in sorted(reader._files):
                    work.tick()
                    archive.copy(Path(source_root) / name, 'package/source/' + name)
            if shared:
                from scoped_source.shared_archive import transfer
                from scoped_source.shared_build import build as shared_build
                from scoped_source.check_shared_source import CheckedSharedForest
                snapshots['source_transfer']=phase('source_transfer',lambda:transfer(owned_source,archive,
                    fault=fault if fault in ('before_transfer','after_transfer') else '',member_limit=member_limit))
                require(len(archive.attempted)<=member_limit, 'transfer remaining member budget')
                raw,construction=phase('source_construct',lambda:shared_build(archive.root/'package/source',source,work=work,enabled=True))
                fr=phase('source_publish',lambda:raw.publish(put))
                f=phase('source_check',lambda:CheckedSharedForest(raw,archive.root/'package/source',source,
                    expected=fr['sha256'],work=work,enabled=True))
            else:
                phase('source_copy', copy_source)
                raw, construction = phase('source_construct', lambda:build(archive.root / 'package/source', source, work=work, enabled=True))
                fr = phase('source_publish', lambda:put('forest.bin', raw))
                f = phase('source_check', lambda:CheckedDirectForest(raw, archive.root / 'package/source', source,
                                                                   expected=fr['sha256'], work=work, enabled=True))
            snapshots['source_stats']=dict(f.receipt['stats'])
            c = phase('property_plan', lambda:contract(f))
            cr = phase('contract_publish', lambda:put('contract.json', compact(c)))
            d = phase('driver_start', lambda:Driver(f, c, expected=identity(c), work=work,
                                                   pair_scoped=method['production']['kind'] == 'pair_scoped'))
            phase('setup', lambda:life.setup(f, work))
            refs = []
            def publish(i, raw):
                ref = put(f'tiles/{i:04d}.bin', raw)
                refs.append(ref)
                if fault == 'after_tile_error':
                    raise RuntimeError('fixed after tile publication')
                return ref['sha256'], ref['bytes']
            for i in range(d.session.tile_count):
                token, records = phase(f'prepare_propose/{i}', lambda:life.propose(d))
                phase(f'candidate_accept/{i}', lambda:d.submit(token, records, publish, profile=method['producer']['profile']))
                records = None
            phase('close', life.close, cleanup=True)
            def finish():
                nonlocal sink
                if streaming:
                    from scoped_source.result_stream import Writer
                    sink = Writer(put,method,invocation,fr['sha256'],refs,work,
                                  fault=fault if fault in RESULT_FAULTS else '')
                return d.finish(sink=sink)
            result = phase('production_finish', finish)
            snapshots['consumer'] = method['production']
            output = dict(result=result, construction=construction, forest=fr, contract=cr, tiles=refs)
            if streaming:
                output['result_ref'] = dict(sink.ref)
                snapshots['result_stream'] = dict(pages=len(sink.pages),rows=sink.count,
                    page_bytes=[r['bytes'] for r in sink.pages],index_bytes=sink.ref['bytes'])
            return output
        except BaseException as exc:
            primary = error(exc)
            snapshots['primary_error'] = primary
            raise
        finally:
            # Even setup failure must close once. Exceptions cannot retain an accepting result.
            try:
                if not life.closed:
                    phase('close_on_exit', life.close, cleanup=True)
            finally:
                before = time.monotonic()
                clean = False
                try:
                    snapshots['lifecycle'] = dict(life.proposer.stats)
                    if d is not None:
                        try:
                            if not d.closed:
                                d.abort()
                        finally:
                            snapshots.update(owner=dict(d.owner.stats), driver_cost=dict(d.costs))
                    records = raw = f = d = life = sink = None
                    gc.collect()
                    clean = True
                finally:
                    phases.append(dict(name='production_cleanup', complete=clean, seconds=time.monotonic() - before,
                                       work={k:0 for k in work.counts}))

    def put(name, raw):
        return dict(archive.put('package/' + name, raw), file=name)

    outcome = None
    try:
        archive = phase('archive_start', lambda:RequestArchive(root, work, limit=archive_limit))
        produced = production()
        work.tick()
        def package():
            code = []
            for name in code_files(method):
                work.tick()
                path = Path(code_root) / name
                require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 64 * 2**10,
                        'bounded checker implementation')
                ref = archive.copy(path, 'package/code/' + name)
                code.append(dict(ref, file='code/' + name))
            m = dict(schema=manifest_schema(method), source=dict(directory='source', sha256=source),
                     forest=produced['forest'], contract=produced['contract'], tiles=produced['tiles'],
                     method=method, invocation=invocation, code=code)
            if streaming: m['result'] = produced['result_ref']
            put('manifest.json', compact(m))
            members = [dict(r, file=n.removeprefix('package/')) for n,r in sorted(archive.entries.items())
                       if n.startswith('package/')]
            b = compact(dict(schema=bundle_schema(method), manifest='manifest.json', members=members))
            require(len(b) <= 2**20, 'bounded inventory')
            return put('bundle.json', b)
        br = phase('package_publish', package)
        inject(fault, 'replay', work)
        replay = phase('portable_recheck', lambda:check_package(archive.root / 'package', expected=br['sha256'],
                         expected_method=method, invocation=invocation, work=work, enabled=True))
        require(replay['result'] == produced['result'], 'complete production/replay equality')
        if streaming:
            require(replay['result_ref'] == produced['result_ref'], 'same replay result reference')
            rr = phase('result_publish', lambda:dict(produced['result_ref']))
        else:
            rr = phase('result_publish', lambda:archive.put('result.json', compact(produced['result'])))
        phase('archive_check', archive.audit)
        tr = phase('terminal_publish', lambda:archive.put('terminal.json', compact(dict(
            status='PENDING_RECEPTION', method=method, invocation=invocation, source=source,
            result=rr, package=br, work=dict(work.counts))), terminal=True))
        phase('inner_terminal_receive', lambda:receive_file(archive, tr))
        outcome = dict(status='COMPLETE', result=produced['result'], result_ref=rr, terminal_ref=tr,
                       package=br, replay=replay, construction=produced['construction'])
    except BaseException as exc:
        outcome = dict(status='TIMEOUT' if isinstance(exc, TimeoutError) else 'ERROR',
                       error=error(exc))
    if outcome['status'] == 'COMPLETE' and time.monotonic() >= work.deadline:
        outcome = dict(status='TIMEOUT', error='lifecycle/replay cleanup deadline')
    if streaming: outcome['schema']='PAGED_LIFECYCLE_REPORT_V1'
    return dict(**outcome, method=method, invocation=invocation, source=source, phases=phases,
                snapshots=snapshots, work=dict(work.counts), initial_work=initial,
                deadline=work.deadline, seconds=time.monotonic() - start,
                archive=None if archive is None else archive.ledger())
