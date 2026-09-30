"""Solver-free observation roster controls on the already frozen tiny sources."""
from fractions import Fraction as F
import json
import os
from pathlib import Path
import unittest
from scoped_proof.io import load,save,sha,ROOT
from scoped_source.endpoint_source_controls import cases
from scoped_source.factored_source import pack
from scripts.h2_capacity_build import build
import time


class ObservationControls(unittest.TestCase):
    def test_complete_roster_and_absent_native_evidence(self):
        root=Path(os.environ['H2_OBSERVATION_ROOT']); root.mkdir(parents=True,exist_ok=False)
        records=[]
        for case,doc,reuse in cases():
            for mode in ('endpoints','mccormick'):
                path=root/(case+'-'+mode); path.mkdir(); trace=[]; proposal_count=0
                def observe(op,**fields): trace.append({'operation':op,**fields})
                def no_candidate(lp,**kwargs):
                    nonlocal proposal_count
                    proposal_count+=1; return None
                source=pack(doc,path/'source',lambda:None,2**20)
                digest=build(path,expected_source_manifest=source,mode=mode,reuse_keys=reuse,
                    deadline=time.monotonic()+30,proposer=no_candidate,observer=observe)
                m=load(path/'manifest.json'); expected=[]; ends=[]; pairs=[]; native_expected=0
                for ref in m['pairs']:
                    p=load(path/ref['file']); pair=p['context']['pair']; pairs.append(pair)
                    for duty in p['duties']:
                        expected.append((pair,duty['competitor'],duty['origin']))
                        if mode=='endpoints':
                            weights=sorted(set(map(F,p['context']['gate'])))
                            ends.extend((pair,duty['competitor'],str(w),duty['origin']) for w in weights)
                            for end in duty['endpoints']:
                                if duty['origin']=='PROPOSED': self.assertIsNone(end['certificate'])
                            native_expected+=len(weights) if duty['origin']=='PROPOSED' else 0
                        else:
                            if duty['origin']=='PROPOSED': self.assertIsNone(duty['certificate'])
                            native_expected+=duty['origin']=='PROPOSED'
                self.assertEqual([e['pair'] for e in trace if e['operation']=='pair_begin'],pairs)
                self.assertEqual([e['pair'] for e in trace if e['operation']=='pair_published'],pairs)
                self.assertEqual([(e['pair'],e['competitor'],e['origin']) for e in trace if e['operation']=='property_begin'],expected)
                self.assertEqual([(e['pair'],e['competitor']) for e in trace if e['operation']=='property_record_complete'],[(p,j) for p,j,_ in expected])
                self.assertEqual([(e['pair'],e['competitor'],e['weight'],e['origin']) for e in trace if e['operation']=='endpoint_begin'],ends)
                self.assertEqual(proposal_count,native_expected)
                self.assertFalse(any(e['operation'].startswith(('native_','candidate_')) for e in trace))
                records.append({'case':case,'mode':mode,'proof_sha256':digest,'properties':len(expected),
                    'proposal_adapter_calls_with_None':proposal_count,'native_calls':0,
                    'trace':trace,'native_candidates_published':0,'native_bounds_checked':0})
        save(root/'result.json',{'schema':'H2_CAPACITY_OBSERVER_CONTROLS_V1','status':'PASS',
            'tests':1,'cases':8,'new_solves':0,'records':records,
            'sources':{n:sha(ROOT/n) for n in ('scripts/test_h2_capacity_observation.py','scripts/h2_capacity_build.py')}})


if __name__=='__main__': unittest.main(verbosity=2)
