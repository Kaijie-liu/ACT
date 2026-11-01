"""Transport-only controls; no checkpoint, solver, or real model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from architecture_s2_reproduce import catalog
from architecture_s1_execution import digest,finite_queue,Artifacts
from scoped_proof.io import save,sha


class ReproductionControls(unittest.TestCase):
    def test_fixed_roster(self):
        q=catalog()
        self.assertEqual(len(q),102)
        self.assertEqual(len({r['id'] for r in q}),102)
        self.assertEqual({(r['model'],r['index']) for r in q if r['mode']=='complete'},
            {(m,i) for m in ('D-act_seed0','M_seed0') for i in range(50)})
        self.assertTrue(all(r['radius']==2 for r in q))
        self.assertEqual(len(q[:10])+len(q[10:]),102)

    def test_expired_batch_launches_nothing(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d)/'records';Artifacts.create(root)
            manifest=Path(d)/'manifest.json'
            save(manifest,dict(synthetic_control=True,queue_budget_seconds=0))
            r=finite_queue(root,[dict(id='never',mode='primary')],manifest,workers=1)
            self.assertEqual(r['status'],'PREFIX_STOP')
            self.assertEqual(r['launched'],0)
            self.assertFalse((root/'never').exists())

    def test_bad_location_identity_rejected_before_import(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            loc=Path(d)/'locations.json';save(loc,dict(schema='S2A_RELOCATION_V1'))
            code=f"import sys;sys.path.insert(0,{str(ROOT/'scripts')!r});import genbab_cpu_ready"
            for h in ('0'*64,sha(loc)):
                env=dict(os.environ,ACT_REPRODUCTION_LOCATIONS=str(loc),
                    ACT_REPRODUCTION_LOCATIONS_SHA256=h,PYTHONDONTWRITEBYTECODE='1')
                r=subprocess.run([sys.executable,'-B','-S','-c',code],env=env,capture_output=True,timeout=10)
                self.assertNotEqual(r.returncode,0)
                self.assertNotIn(b'torch',r.stderr)


if __name__=='__main__':unittest.main()
