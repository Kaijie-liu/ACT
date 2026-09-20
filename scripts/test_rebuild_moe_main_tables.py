"""Accounting regressions; no checkpoint, solver, or raw experiment access."""
import copy
import json
from pathlib import Path
import unittest
import rebuild_moe_main_tables as tables


class MainTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {k: json.loads((tables.ROOT / p).read_text())
                         for k, p in tables.SOURCES.items()}

    def test_frozen_totals(self):
        rows = tables.build(self.documents)
        self.assertEqual(len(rows), 21)
        adaptive = [r for r in rows if r['cohort'] == 'MLP confirmation' and r['arm'] == 'adaptive']
        self.assertEqual(sum(r['positive'] for r in adaptive), 179)
        self.assertEqual(sum(r['positive'] + r['unsafe'] for r in adaptive), 256)
        self.assertEqual(sum(r['positive'] for r in rows if r['cohort'] == 'Conv evidence'), 0)

    def test_bad_inputs_rejected(self):
        def missing(d):
            d['external']['full_requests'].pop()
        def duplicate(d):
            r = d['conv']['terminals']
            r[3] = copy.deepcopy(r[1])
        def grade(d):
            d['conv']['terminals'][2]['status'] = 'SAFE'
        def denominator(d):
            d['confirmation']['full']['audit']['models']['seed0']['methods']['adaptive']['states']['SAFE'] += 1
        def delta(d):
            d['confirmation']['full']['audit']['models']['seed0']['contrasts']['matched']['SAFE']['net'] += 1
        def cost(d):
            d['external']['full_requests'][0]['adaptive_seconds'] = float('nan')
        def failed(d):
            d['evidence']['issues'] = ['failure']
        def terminal(d):
            d['evidence']['terminal_bindings'][0]['status'] = 'UNKNOWN'
        def cohort(d):
            d['conv']['terminals'][0]['dataset_index'] = 99999
        for mutation in (missing, duplicate, grade, denominator, delta, cost, failed, terminal, cohort):
            with self.subTest(mutation=mutation.__name__):
                d = copy.deepcopy(self.documents)
                mutation(d)
                with self.assertRaises(ValueError):
                    tables.build(d)

    def test_committed_rendering(self):
        import hashlib
        hashes = {k: hashlib.sha256((tables.ROOT / p).read_bytes()).hexdigest()
                  for k, p in tables.SOURCES.items()}
        self.assertEqual(tables.render(tables.build(self.documents), hashes),
                         (tables.ROOT / 'paper/results/main_tables.md').read_text())


if __name__ == '__main__':
    unittest.main()
