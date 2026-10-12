"""Offline archive metadata mutations; no generation or source checking."""
import copy
import unittest
from scripts.audit_source_dag_controls import envelope,receipt_cost,ANCHORS,FILES,PROTOCOL,LIMIT


class AuditControls(unittest.TestCase):
    def inputs(self):
        code={f:'0'*64 for f in FILES}; code['docs/hz_source_dag_design_20261013.md']=PROTOCOL
        e=dict(code=code,source_anchors=dict(ANCHORS),archive_limit=LIMIT,protocol=PROTOCOL,source_members={x:{} for x in ANCHORS})
        t=dict(code=dict(code),source_anchors=dict(ANCHORS),archive_cap=LIMIT,status='PASS_COMPONENT_CONTROLS',
            real_models=0,gpu_calls=0,full_request_certificates=0,outer_before_terminal_seconds=4,
            calls=[dict(name=x,status='COMPLETE',returncode=0,hard_cap_seconds=300,parent_observed_seconds=1)
                   for x in ('unit','small40','dense','relocated')])
        return e,t

    def test_fixed_sources_protocol_capacity_and_code(self):
        e,t=self.inputs(); envelope(e,t)
        mutations=[lambda e,t:e['source_anchors'].update(dense='1'*64),
            lambda e,t:e.update(archive_limit=LIMIT*2),lambda e,t:e.update(code={}),
            lambda e,t:e.update(protocol='1'*64),lambda e,t:t['code'].pop('scoped_source/source_dag.py'),
            lambda e,t:t['calls'].pop(),lambda e,t:t.update(full_request_certificates=1)]
        for mutate in mutations:
            e,t=self.inputs(); mutate(e,t)
            with self.assertRaises(ValueError): envelope(e,t)

    def test_finite_positive_elapsed(self):
        for bad in (-1,float('nan'),float('inf'),301,True):
            e,t=self.inputs(); t['calls'][0]['parent_observed_seconds']=bad
            with self.assertRaises(ValueError): envelope(e,t)

    def test_complete_consistent_counters_and_phase_costs(self):
        w=dict(row_visits=3,coefficient_visits=4,arithmetic=5)
        base=dict(work=w,check=dict(work=dict(w)),worker_seconds=6,phases=dict(
            generation_and_encode_seconds=2,file_write_seconds=1,independent_parse_source_check_seconds=2,
            native_seconds=1,native_coordinates=9))
        receipt_cost(base)
        for mutate in (lambda r:r['work'].update(arithmetic=-1),lambda r:r['work'].update(extra=1),
                       lambda r:r['check']['work'].update(arithmetic=7),lambda r:r.update(worker_seconds=-1),
                       lambda r:r['phases'].update(native_seconds=3),lambda r:r['phases'].update(file_write_seconds=8)):
            r=copy.deepcopy(base); mutate(r)
            with self.assertRaises(ValueError): receipt_cost(r)


if __name__=='__main__': unittest.main()
