"""Archive four frozen finite interface fixtures, never an optimizer/model run.

Run from the ACT root. The destination must not exist. The given-source fixture
preparation and old encoder are timed, but are NOT a bounded live producer.
"""
import argparse
from pathlib import Path
import tempfile
import time

from scoped_source.hz_request_store import Work,POLICY,compact,digest,identity,require
from scoped_source.scaled_endpoint_controls import fixture
from scoped_source.scaled_references import build
from scoped_source.wire_forest import encode,CheckedWireForest
from scoped_source.scaled_endpoints import Session,contract
from scoped_source.wire_proof import encode_tile,decode_tile,ProofHash,MANIFEST
from scoped_source.wire_package import bundle,check_package
from scoped_source.wire_archive import Archive,receive_file
from scripts.test_wire_evidence import candidates,CODE_FILES


def run(destination):
    destination=Path(destination).resolve()
    require(destination.is_relative_to('/data1/Kane/MOE'),'owned control destination')
    destination.mkdir(exist_ok=False); records=[]; repo=Path(__file__).resolve().parents[1]
    for case in ('mixed','deep','half','guarded'):
        start=time.monotonic(); w=Work(start+300); phases={}; a=Archive(destination/case,w)
        with tempfile.TemporaryDirectory(prefix='wire-input-',dir='/data1/Kane/MOE') as temporary:
            source_root,source=fixture(Path(temporary)/'source',case); w.tick()
            phases['fixture_seconds']=time.monotonic()-start; t=time.monotonic()
            raw,_=build(source_root,source,work=w); phases['old_construction_seconds']=time.monotonic()-t
            t=time.monotonic(); wire=encode(raw,source_root,source,expected=digest(raw),work=w)
            phases['encoding_seconds']=time.monotonic()-t; t=time.monotonic()
            source_bytes=0
            for p in sorted(source_root.rglob('*')):
                if p.is_file():
                    source_bytes+=p.stat().st_size; a.copy(p,'source/'+str(p.relative_to(source_root)))
            fref=a.put('forest.bin',wire); phases['input_publication_seconds']=time.monotonic()-t
            t=time.monotonic(); f=CheckedWireForest(wire,a.root/'source',source,expected=fref['sha256'],work=w)
            phases['source_check_seconds']=time.monotonic()-t
            c=contract(f); cref=a.put('contract.json',compact(c))
            s=Session(f,c,expected=identity(c),work=w,retain_raw=False); ph=ProofHash(identity(c),8,w)
            tiles=[]; receipts=[]; phases['candidate_seconds']=phases['tile_publication_seconds']=phases['endpoint_check_seconds']=0.
            for i in range(s.tile_count):
                t=time.monotonic(); tasks,_=s.prepare(i); proposal=candidates(f,tasks)
                tile=encode_tile(proposal,f,tasks,w); phases['candidate_seconds']+=time.monotonic()-t
                t=time.monotonic(); tiles.append(a.put(f'tiles/{i:04d}.bin',tile)); phases['tile_publication_seconds']+=time.monotonic()-t
                t=time.monotonic(); logical=decode_tile(tile,f,tasks,w); ph.tile(logical); receipts.extend(s.accept(logical))
                phases['endpoint_check_seconds']+=time.monotonic()-t; del logical,tile,proposal
            t=time.monotonic(); outcome=s.finish(receipts); proof_hash,proof_bytes=ph.finish()
            phases['aggregation_and_source_reseal_seconds']=time.monotonic()-t
            manifest=dict(schema=MANIFEST,source=dict(directory='source',sha256=source),forest=fref,
                          contract=cref,width=8,tiles=tiles,logical_proof=proof_hash,logical_bytes=proof_bytes)
            t=time.monotonic(); a.put('manifest.json',compact(manifest))
            for name in CODE_FILES: a.copy(repo/name,'code/'+name)
            br=bundle(a); snapshot_bytes=a.bytes
            a.relocate('relocated',tuple(a.entries))
            phases['code_bundle_two_copy_seconds']=time.monotonic()-t
            t=time.monotonic(); replay=check_package(a.root/'relocated',expected=br['sha256'],work=w)
            require(replay['result']==outcome,'same all-obligations result after relocation')
            phases['relocated_independent_recheck_seconds']=time.monotonic()-t
            # Even temporary given-source fixture storage is disclosed, not
            # hidden in the two-copy package count or a live-capture claim.
            require(source_bytes+a.bytes+a.reserve<=POLICY['total_bytes'],'finite control full disk envelope')
            peak_with_fixture=source_bytes+a.bytes
            t=time.monotonic()
        phases['fixture_cleanup_seconds']=time.monotonic()-t; w.tick()
        record=dict(case=case,source=source,logical_forest=digest(raw),wire_forest=digest(wire),
            logical_forest_bytes=len(raw),wire_forest_bytes=len(wire),logical_proof_bytes=proof_bytes,
            wire_tile_bytes=sum(x['bytes'] for x in tiles),snapshot_bytes=snapshot_bytes,
            peak_control_disk_bytes_with_fixture=peak_with_fixture,package_sha256=br['sha256'],
            portable_snapshot=str(a.root/'relocated'),outcome=outcome,
            phases=phases,work=dict(w.counts),source_access=dict(f.source_access.stats),
            real_model_experiment=False,optimizer_used=False,gpu_used=False)
        # Timely durable reception and archive reconciliation, not status text,
        # determine completion. Terminal does not contain its own duration/hash.
        terminal=a.put('terminal.json',compact(record),terminal=True); received=receive_file(a,terminal)
        require(received==record,'durable terminal differential'); w.tick()
        record.update(terminal=terminal,archive=a.ledger(),total_seconds=time.monotonic()-start)
        records.append(record)
    return dict(schema='WIRE_INTERFACE_FINITE_CONTROLS_V1',cases=records,
                scope='finite codec/lifecycle controls, not capacity admission or new real certificates')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
    report=run(args.output)
    for row in report['cases']:
        events=row['archive'].pop('events')
        row['archive']['publication_seconds']=sum(e['seconds'] for e in events)
        row['archive']['all_publications_timely']=all(e['timely_published'] for e in events)
    print(compact(report).decode())
