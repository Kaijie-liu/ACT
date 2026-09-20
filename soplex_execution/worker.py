"""One supplied-LP attempt. External solver is an untrusted point proposer."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import signal
import time

from lp_sandwich.check import strict_json
from soplex_execution.candidate import LimitError,parse,bundle
from soplex_execution.runtime import Clock,PYTHON
from soplex_fidelity.io import load_job,write_lp,verify_readback,save,sha


def execute(spec):
    root=Path(spec['root']);job=spec['job'];cfg=spec['runtime']
    clock=Clock(root,spec['start'],spec['proposal'],spec['work'])
    def alarm(signum,frame):raise TimeoutError('phase absolute timer')
    signal.signal(signal.SIGALRM,alarm)
    result=dict(status='ERROR',upper_bound=None,network_SAFE=False,network_UNSAFE=False)
    try:
        with clock.phase('load',True):
            lp=load_job(job,clock.tick)
            for ref in cfg.values():
                if sha(ref['path'])!=ref['sha256']:raise ValueError('runtime identity drift')
        with clock.phase('export',True):write_lp(lp,root/'input.lp',clock.tick)
        with clock.phase('import',True):
            r=clock.native('reader',[cfg['reader']['path'],root/'input.lp',root/'readback.txt'])
            if r['returncode']!=0:raise RuntimeError('native read-only import failed')
        with clock.phase('readback',True):save(root/'readback_receipt.json',verify_readback(lp,root/'readback.txt',clock.tick))
        with clock.phase('solve',True):
            remaining=clock.proposal-time.monotonic();clock.tick()
            r=clock.native('solver',[cfg['soplex']['path'],'--loadset='+cfg['settings']['path'],
                 '-t'+str(remaining),'-X='+str(root/'point.txt'),root/'input.lp'])
            if r['returncode']!=0:raise RuntimeError('native solver process failed')
        with clock.phase('capture',True):
            point,receipt=parse(root/'point.txt',len(lp['c']),clock.tick)
            save(root/'candidate_receipt.json',receipt)
        with clock.phase('package'):
            b=bundle(lp,job['statement'],point,clock.tick)
            save(root/'bundle.json',b)
            shutil.copyfile(cfg['checker']['path'],root/'check.py')
            if sha(root/'check.py')!=cfg['checker']['sha256']:raise ValueError('copied checker drift')
            bundle_sha=sha(root/'bundle.json')
        with clock.phase('check'):
            r=clock.native('checker',[PYTHON,'-I','-S',root/'check.py',root/'bundle.json',
                '--bundle-sha256',bundle_sha,'--statement-sha256',job['statement_sha256'],
                '--timeout-seconds',str(clock.work-time.monotonic())])
            if r['returncode']==3:raise TimeoutError('isolated checker deadline')
            if r['returncode']!=0:raise ValueError('isolated original-LP checker rejected bundle')
        with clock.phase('review'):
            checked=strict_json((root/'checker.stdout').read_bytes())
            if (checked['status']!='CHECKED_LP_DIAGNOSTIC' or
                checked['statement_sha256']!=job['statement_sha256'] or
                checked['lp_sha256']!=job['statement']['lp_sha256'] or
                checked['network_SAFE'] or checked['network_UNSAFE'] or
                not checked['isolated'] or not checked['site_disabled'] or checked['solver_or_model_imported']):
                raise ValueError('checker receipt binding')
            result.update(status='CHECKED',upper_bound=checked['upper_bound'],
                classification=checked['classification'],primal_status=checked['primal_status'],
                bundle_sha256=bundle_sha,checker_stdout_sha256=sha(root/'checker.stdout'))
    except TimeoutError as exc:result.update(status='TIMEOUT',error=str(exc))
    except (LimitError,MemoryError) as exc:result.update(status='LIMIT',error=str(exc))
    except Exception as exc:result.update(status='ERROR',error=type(exc).__name__+': '+str(exc))
    result.update(job_id=job['job_id'],statement_sha256=job['statement_sha256'],
                  completed_offset=time.monotonic()-spec['start'],worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # A worker result is a candidate terminal; only the outer publication receipt accepts it.
    save(root/'worker_result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);a=p.parse_args()
    execute(strict_json(a.spec.read_bytes()))
