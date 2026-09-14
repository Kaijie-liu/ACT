import copy
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from act.pipeline.moe.external_pair_comparison import jobs,execute,ARMS,CFG,CFG_HASH,SELECTION,SELECTION_HASH
from act.pipeline.moe.experiment1 import WRITE_ROOT,_sha256
from act.pipeline.moe.review_external_pair_comparison import route_inventory


class ExternalPairComparisonTests(unittest.TestCase):
    def test_frozen_cohort_and_order(self):
        self.assertEqual(_sha256(CFG),CFG_HASH);self.assertEqual(_sha256(SELECTION),SELECTION_HASH)
        s=json.loads(SELECTION.read_text());s['samples']=s['samples'][:10]
        schedule=jobs(s,False);self.assertEqual(len(schedule),60)
        self.assertEqual(len({j['job_id'] for j in schedule}),60)
        for i in range(0,len(schedule),2):
            a,b=schedule[i:i+2];self.assertEqual(a['dataset_index'],b['dataset_index'])
            self.assertEqual(a['model'],b['model']);self.assertEqual({a['method'],b['method']},set(ARMS))
        self.assertEqual(len(jobs(s,True)),6)

    def test_route_inventory_and_unknown(self):
        pairs=list(map(list,itertools.combinations(range(8),2)))
        r={'feasible':pairs[:2],'infeasible':pairs[2:],'unresolved':[],'exact':True,
           'branches':[{'route_set':p,'feasibility':'feasible' if p in pairs[:2] else 'infeasible'} for p in pairs]}
        self.assertTrue(route_inventory(r))
        for change in [lambda x:x['feasible'].pop(),lambda x:x['infeasible'].append([0,1]),
                       lambda x:x['branches'].pop(),lambda x:x['branches'][0].update(feasibility='infeasible')]:
            changed=copy.deepcopy(r);change(changed)
            with self.assertRaises(ValueError):route_inventory(changed)
        r['exact']=False;self.assertFalse(route_inventory(r))
        r['feasible'].remove(pairs[0]);r['unresolved']=[pairs[0]];r['branches'][0]['feasibility']='unknown'
        self.assertFalse(route_inventory(r))

    def test_watchdog_kills_owned_child_group(self):
        with tempfile.TemporaryDirectory(dir=WRITE_ROOT) as directory:
            log=Path(directory)/'worker.log'
            command=[sys.executable,'-c',
                'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"]); print(p.pid,flush=True); time.sleep(10)']
            start=time.monotonic();code,expired=execute(command,log,start,os.environ.copy(),budget=.5)
            self.assertTrue(expired);self.assertLess(code,0);self.assertLess(time.monotonic()-start,3)
            child=int(log.read_text().strip());stat=Path(f'/proc/{child}/stat')
            # An adopted zombie is terminated too; it cannot keep using CPU.
            def stopped():
                try:return stat.read_text().split()[2]=='Z'
                except (FileNotFoundError,ProcessLookupError):return True
            for _ in range(20):
                if stopped():break
                time.sleep(.01)
            self.assertTrue(stopped())


if __name__=='__main__':unittest.main()
