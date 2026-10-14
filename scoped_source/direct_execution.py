"""Optional full captured-source connection with a supplied candidate callback.

This stage uses trusted fixed stubs, never native/GPU proposal code. One
cooperative Work/deadline covers production AND fresh portable replay. An
arbitrarily blocking callback needs an outer process supervisor, not provided
by this synchronous interface. A durable proposed terminal is not acceptance.
"""
from pathlib import Path
import time

from scoped_source.hz_request_store import compact,digest,identity,require
from scoped_source.tensor_rows import TensorRows
from scoped_source.check_direct_source import CheckedDirectForest
from scoped_source.direct_candidates import Driver
from scoped_source.scaled_endpoints import contract
from scoped_source.wire_archive import Archive,receive_file
from scoped_source.direct_package import MANIFEST,BUNDLE,CODE_FILES,check_package


def execute(root,source_root,source,propose,code_root,*,work,enabled=False,profile=0):
    require(enabled is True,'direct execution is opt-in')
    require(callable(propose) and type(profile) is int and profile in (0,1),'candidate callback/profile')
    start=time.monotonic(); initial_work=dict(work.counts); initial_deadline=work.deadline
    phases=[]; archive=None; driver=None; outcome=None; terminal_received=False
    def phase(name,fn):
        before=time.monotonic(); counts=dict(work.counts); complete=False
        try:
            work.tick(); result=fn(); work.tick(); complete=True; return result
        finally:
            phases.append(dict(name=name,seconds=time.monotonic()-before,complete=complete,
                               work={k:work.counts[k]-v for k,v in counts.items()}))
    try:
        archive=phase('archive_start',lambda:Archive(root,work))
        def put(name,raw):
            ref=archive.put('package/'+name,raw); return dict(ref,file=name)
        def copy_source():
            reader=TensorRows(source_root,source,deadline=work.deadline)
            for name in sorted(reader._files):
                work.tick(); archive.copy(Path(source_root)/name,'package/source/'+name)
            work.tick()
        phase('source_copy',copy_source)
        from scoped_source.direct_source_build import build
        raw,construction=phase('source_construct',lambda:build(archive.root/'package/source',source,work=work,enabled=True))
        fr=phase('source_publish',lambda:put('forest.bin',raw))
        f=phase('source_check',lambda:CheckedDirectForest(raw,archive.root/'package/source',source,expected=fr['sha256'],work=work,enabled=True))
        c=phase('property_plan',lambda:contract(f)); cr=phase('contract_publish',lambda:put('contract.json',compact(c)))
        driver=phase('driver_start',lambda:Driver(f,c,expected=identity(c),work=work,pair_scoped=True))
        refs=[]
        def publish(i,b):
            ref=put(f'tiles/{i:04d}.bin',b); refs.append(ref); return ref['sha256'],ref['bytes']
        for i in range(driver.session.tile_count):
            tasks,_,token=phase(f'prepare/{i}',driver.prepare)
            records=phase(f'fixed_candidate/{i}',lambda:propose(f,tasks))
            phase(f'candidate_accept/{i}',lambda:driver.submit(token,records,publish,profile=profile))
        result=phase('production_finish',driver.finish)
        def package():
            code=[]
            for name in CODE_FILES:
                work.tick(); path=Path(code_root)/name
                require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=64*2**10,'bounded checker code')
                ref=archive.copy(path,'package/code/'+name); code.append(dict(ref,file='code/'+name))
            m=dict(schema=MANIFEST,source=dict(directory='source',sha256=source),forest=fr,contract=cr,
                   candidate_width=8,check_width=2,tiles=refs,code=code)
            put('manifest.json',compact(m))
            members=[dict(r,file=n.removeprefix('package/')) for n,r in sorted(archive.entries.items()) if n.startswith('package/')]
            raw_bundle=compact(dict(schema=BUNDLE,manifest='manifest.json',members=members))
            require(len(raw_bundle)<=2**20,'bounded bundle inventory')
            return put('bundle.json',raw_bundle)
        br=phase('package_publish',package)
        replay=phase('portable_recheck',lambda:check_package(archive.root/'package',expected=br['sha256'],work=work,enabled=True))
        require(replay['result']==result,'production/replay complete result equality')
        rr=phase('result_publish',lambda:archive.put('result.json',compact(result)))
        phase('archive_check',archive.audit)
        terminal=dict(status='PENDING_RECEPTION',result=rr,package=br,work=dict(work.counts),
                      mathematical_status=result['status'],required=result['required'],positive=result['positive'])
        tr=phase('terminal_publish',lambda:archive.put('terminal.json',compact(terminal),terminal=True))
        phase('terminal_receive',lambda:receive_file(archive,tr))
        terminal_received=True
        outcome=dict(status='COMPLETE',result=result,package=br,replay=replay,construction=construction)
    except BaseException as exc:
        outcome=dict(status='TIMEOUT' if isinstance(exc,TimeoutError) else 'ERROR',error=type(exc).__name__+': '+str(exc)[:500])
    finally:
        cleanup=time.monotonic()
        clean=True
        try:
            if driver is not None and not driver.closed: driver.abort()
        except BaseException as exc:
            clean=False; primary=None if outcome is None else outcome.get('error')
            outcome=dict(status='ERROR',error='cleanup: '+type(exc).__name__+': '+str(exc)[:500],primary_error=primary)
        phases.append(dict(name='cleanup',seconds=time.monotonic()-cleanup,complete=clean,work={k:0 for k in work.counts}))
    if outcome['status']=='COMPLETE' and time.monotonic()>=work.deadline:
        outcome=dict(status='TIMEOUT',error='cleanup/terminal over cumulative deadline')
    return dict(**outcome,phases=phases,work=dict(work.counts),initial_work=initial_work,
                initial_deadline=initial_deadline,terminal_received=terminal_received and outcome['status']=='COMPLETE',
                elapsed_seconds=time.monotonic()-start,
                cleanup_after_deadline=max(0.,time.monotonic()-work.deadline),
                archive=None if archive is None else archive.ledger(),
                owner=None if driver is None else dict(driver.owner.stats),
                driver_cost=None if driver is None else dict(driver.costs))
