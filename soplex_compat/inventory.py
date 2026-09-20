"""Read-only installation inventory and original-checker regression receipt."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from soplex_compat.controls import PREFIX, ROOT, save, sha


def main():
    target=ROOT/'docs/soplex_compat_installation_v1.json'
    if target.exists():raise FileExistsError(target)
    def command(args):
        p=subprocess.run(args,capture_output=True,text=True,timeout=30)
        return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    commit=command(['git','-C',str(PREFIX/'src'),'rev-parse','HEAD'])['stdout'].strip()
    if commit!='13e2ab2467e0016d02116802ac4dc7a89560dbc1':raise ValueError('upstream drift')
    if command(['git','-C',str(PREFIX/'src'),'status','--porcelain'])['stdout']:
        raise ValueError('modified upstream')
    names=['bin/soplex','bin/exact_io_probe','build/lib/libsoplex.a','build/CMakeCache.txt',
           'build/soplex/config.h','configure.log','configure_attempt002.log','build.log',
           'build_attempt002.log','probe_build.log','deps/include/gmp.h','deps/include/gmpxx.h',
           'deps/include/mpfr.h','deps/lib/libgmp.a','deps/lib/libgmpxx.a','deps/lib/libmpfr.a',
           'src/LICENSE','src/settings/exact.set','src/src/soplexmain.cpp','src/src/soplex.hpp']
    h=hashlib.sha256();count=0;size=0
    for p in sorted((PREFIX/'deps/include/boost').rglob('*')):
        if p.is_file():
            h.update((str(p.relative_to(PREFIX))+'\0'+sha(p)+'\n').encode())
            count+=1;size+=p.stat().st_size
    started=time.monotonic()
    regression=command([sys.executable,'-m','unittest','lp_sandwich.tests'])
    regression['seconds']=time.monotonic()-started
    if regression['returncode'] or 'Ran 21 tests' not in regression['stderr']:
        raise ValueError('original checker regression failed: '+str(regression))
    result=dict(schema='SOPLEX_ISOLATED_INSTALLATION_V1',prefix=str(PREFIX),version='8.0.3',
        source_url='https://github.com/scipopt/soplex',upstream_commit=commit,
        user_authorized_install=True,act_environment_modified=False,
        dependencies='Existing GMP 6.3.0 / MPFR 4.2.1 static libraries and Boost 1.90.0 headers copied into isolated prefix; no package-manager changes.',
        files={name:dict(sha256=sha(PREFIX/name),bytes=(PREFIX/name).stat().st_size) for name in names},
        boost_headers=dict(tree_sha256=h.hexdigest(),files=count,bytes=size),
        compiler=command(['/usr/bin/g++','--version']),cmake=command(['/usr/bin/cmake','--version']),
        dynamic_links={name:command(['ldd',str(PREFIX/'bin'/name)]) for name in ('soplex','exact_io_probe')},
        banner=command([str(PREFIX/'bin/soplex')]),regression=regression,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'soplex_compat').glob('*') if p.is_file()},
        guidance_sha256=sha(Path('/data1/Kane/MOE/Advice/dd.md')),
        historical_checker_sha256=sha(ROOT/'lp_sandwich/check.py'),
        failed_build_retained='build.log: static MPFR requires GMP repeated after MPFR; fixed in link flags only, upstream unchanged.',
        real_queries=0,production_integration=False)
    save(target,result)
    print(json.dumps(dict(status='RECORDED',path=str(target),regression_tests=21,boost_files=count)))


if __name__=='__main__':main()
