"""Relocated stdlib checks and mutated evidence, without new optimization."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO))
from scoped_source.hz_request_store import compact, digest, identity, require

KIT=('scoped_source/hz_request_store.py','scoped_source/check_hz_store_source.py',
     'scoped_source/check_store_endpoints.py','scripts/check_store_endpoints.py')


def write(path,value):
    # Only new, explicitly disposable mutation copies, never archived originals.
    Path(path).write_bytes(compact(value))


def run(archive):
    archive=Path(archive); target=archive/'relocation'; target.mkdir(exist_ok=False)
    kit=target/'kit'
    for name in KIT:
        p=kit/name; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(REPO/name,p)
    tests=[]; parent=archive/'source8_tile8'; original=json.loads((parent/'result.json').read_bytes())
    for case in ('positive','missing_manifest_tail','partial_last_candidate','wrong_bound','foreign_property','tampered_source'):
        dest=target/case; shutil.copytree(parent,dest)
        contract=original['contract']; proof=original['proof']; expected=0
        if case in ('missing_manifest_tail','partial_last_candidate','wrong_bound'):
            manifest=json.loads((dest/'proof/manifest.json').read_bytes())
            if case=='missing_manifest_tail': manifest['tiles'].pop(); expected=1
            else:
                ref=manifest['tiles'][-1]; path=dest/'proof'/ref['file']; records=json.loads(path.read_bytes())
                if case=='partial_last_candidate': records[-1]['candidate']=None
                else: records[-1]['candidate']['lower']='999'; expected=1
                write(path,records); ref['sha256']=digest(path.read_bytes()); ref['bytes']=path.stat().st_size
            write(dest/'proof/manifest.json',manifest); proof=identity(manifest)
        elif case=='foreign_property':
            doc=json.loads((dest/'contract.json').read_bytes()); doc['label']=1
            write(dest/'contract.json',doc); contract=identity(doc); expected=1
        elif case=='tampered_source':
            doc=json.loads((dest/'source.json').read_bytes()); doc['upper'][0]='2'
            write(dest/'source.json',doc); expected=1
        start=time.monotonic()
        done=subprocess.run([sys.executable,'-B','-I','-S',str(kit/'scripts/check_store_endpoints.py'),str(dest),
                             '--contract',contract,'--proof',proof],capture_output=True,timeout=300,cwd=target)
        elapsed=time.monotonic()-start
        (target/f'{case}.stdout').write_bytes(done.stdout); (target/f'{case}.stderr').write_bytes(done.stderr)
        require((done.returncode==0)==(expected==0), f'portable control {case}')
        result=json.loads(done.stdout)['result'] if done.returncode==0 else None
        if case=='positive': require(result==original['result'], 'relocated complete result differential')
        if case=='partial_last_candidate':
            require(result['status']=='UNKNOWN_MISSING_EVIDENCE' and result['missing_endpoints']==1
                    and result['checked_endpoints']==503 and result['required']==252,
                    'partial endpoint cannot close request')
        tests.append(dict(case=case,expected_success=not expected,returncode=done.returncode,
            launch_to_receipt_seconds=elapsed,status=None if result is None else result['status'],
            stdout_sha256=digest(done.stdout),stderr_sha256=digest(done.stderr),
            contract=contract,proof=proof))
    report=dict(status='PASS',controls=tests,kit={name:digest((kit/name).read_bytes()) for name in KIT},
        controller=digest(Path(__file__).read_bytes()),new_solves=0,
        scope='relocated saved synthetic source/proof; not production budget admission')
    with (archive/'portable.json').open('xb') as stream: stream.write(compact(report))
    print(compact(report).decode())


if __name__=='__main__': run(sys.argv[1])
