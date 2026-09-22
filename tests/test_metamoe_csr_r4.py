import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_csr_r4 import contract, validate
from freeze_metamoe_csr_r4 import rebind
from audit_metamoe_csr import diagnostic_review
from recent_moe_deployment import sha256


class R4Controls(unittest.TestCase):
    def config(self):
        cfg = json.loads(Path('configs/recent_moe/metamoe_csr_smoke_r3.json').read_text())
        cfg.update(protocol='metamoe_csr_spatial_r4_old_input', seconds=90)
        cfg['hybridz']['sparse_resource_policy'] = 'csr_spatial_v2'
        cfg['requests'] = [r for r in cfg['requests'] if r['id'] == 'mnist_0']
        cfg['parent_config_sha256'] = sha256('configs/recent_moe/metamoe_csr_smoke_r3.json')
        return cfg

    def test_registered_contract_and_mutations(self):
        cfg = self.config()
        contract(cfg)
        for key, value in [('seconds', 91), ('epsilon', 4/255), ('group_rss_limit_bytes', 16*2**30), ('margin', 0.)]:
            bad = copy.deepcopy(cfg)
            bad[key] = value
            with self.assertRaises(ValueError):
                contract(bad)
        for field, value in [('index', 1), ('dataset', 'CIFAR10'), ('id', 'mnist_1')]:
            bad = copy.deepcopy(cfg)
            bad['requests'][0][field] = value
            with self.assertRaises(ValueError):
                contract(bad)

    def test_new_tensor_and_environment_not_inherited(self):
        cfg = self.config()
        cfg['requests'][0]['tensor_file'] = 'other.npz'
        with self.assertRaisesRegex(ValueError, 'tensor/request'):
            validate(cfg)
        cfg = self.config()
        cfg['python']['act'] = '/some/new/python'
        with self.assertRaisesRegex(ValueError, 'inherited execution'):
            validate(cfg)

    def test_unapproved_source_rebinding(self):
        with patch('freeze_metamoe_csr_r4.sha256', return_value='new'):
            with self.assertRaisesRegex(ValueError, 'unapproved source'):
                rebind({'files': {'forbidden.py': 'old'}})

    def review(self, mutate):
        with tempfile.TemporaryDirectory(prefix='csr-r4-control-', dir='/data1/Kane/MOE') as temp:
            root = Path(temp)
            cfg = root/'config.json'
            cfg.write_text(json.dumps({'group_rss_limit_bytes': 8*2**30}))
            for name in ('stdout', 'stderr'):
                (root/f'{name}.txt').write_text('')
            receipt = {'status': 'COMPLETED', 'deadline_seconds': 90, 'group_rss_limit_bytes': 8*2**30,
                'execution_including_preflight_seconds': 3., 'peak_sampled_group_rss_bytes': 100,
                'stdout_sha256': sha256(root/'stdout.txt'), 'stderr_sha256': sha256(root/'stderr.txt')}
            phases = ['router', 'guarded_expert_0', 'guarded_expert_1']
            result = {'config_sha256': sha256(cfg), 'status': 'COMPLETE', 'completed_phases': phases,
                'solver_calls': 0, 'events': [{'phase': phase, 'sparse': {}, 'dense_retained': False} for phase in phases]}
            mutate(receipt, result)
            for i, event in enumerate(result['events'], 1):
                (root/f'layer_{i:03d}.json').write_text(json.dumps(event))
            (root/'receipt.json').write_text(json.dumps(receipt))
            (root/'diagnostic.json').write_text(json.dumps(result))
            return diagnostic_review(cfg, root)

    def test_missing_phase_empty_events_solver_dense_rejected(self):
        self.assertTrue(self.review(lambda r, d: None)['passed'])
        for mutation in [lambda r, d: d['completed_phases'].pop(),
                         lambda r, d: d.update(events=[]), lambda r, d: d.update(solver_calls=1),
                         lambda r, d: d['events'][0].update(dense_retained=True)]:
            self.assertFalse(self.review(mutation)['passed'])

    def test_late_rss_outer_cannot_upgrade(self):
        for mutation in [lambda r, d: r.update(status='TIMEOUT'),
                         lambda r, d: r.update(status='RESOURCE_LIMIT'),
                         lambda r, d: r.update(execution_including_preflight_seconds=90.),
                         lambda r, d: r.update(peak_sampled_group_rss_bytes=8*2**30+1)]:
            self.assertFalse(self.review(mutation)['passed'])


if __name__ == '__main__':
    unittest.main()
