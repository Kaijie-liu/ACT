import json
import hashlib
import io
from pathlib import Path
import tempfile
import subprocess
import tarfile
import unittest
from unittest.mock import patch

from act.pipeline.moe import review_schedule_confirmation_100 as review


class ReviewTests(unittest.TestCase):
    def test_missing_evidence_is_unknown_not_zero_cost_or_reuse(self):
        value = review.evidence_summary(None)
        self.assertIsNone(value['property_counters'])
        self.assertIsNone(value['exact_pair_count'])
        self.assertEqual(value['reason'], 'OUTER_HARD_DEADLINE')

    def test_available_inventory_does_not_imply_executed_reuse(self):
        e = {'route_coverage': {'route_sets_exact': False, 'feasible_route_sets': [[0, 1]]},
             'verdict': {'decision_tier': 'TIER1_GATE_ELIMINATION', 'reason': 'SAFE_GATE_ELIMINATION'},
             'proof_reuse': {'available_fact_count': 99}, 'tier2': {'invoked': False}}
        with patch.object(review, 'counters', return_value={'reused_pair_properties': 0, 'recorded_weighted_query_rows': 0}):
            value = review.evidence_summary(e)
        self.assertIsNone(value['exact_pair_count'])
        self.assertEqual(value['property_counters']['reused_pair_properties'], 0)
        self.assertEqual(value['available_fact_count'], 99)

    def test_terminal_ledger_mismatch_fails_before_extracting(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root = Path(tmp); (root/'job').mkdir()
            (root/'rows.jsonl').write_text('{"job_id":"job","status":"TIMEOUT"}\n')
            (root/'job/terminal.json').write_text('{"job_id":"job","status":"SAFE"}')
            with self.assertRaisesRegex(ValueError, 'terminal file disagrees'):
                review.derive(root)

    def test_real_supplement_reconstructs_and_denominators_close(self):
        archived = json.loads(review.OUTPUT.read_text())
        rebuilt = review.derive()
        self.assertEqual(rebuilt, archived['descriptive_supplement'])
        self.assertEqual(len(rebuilt['missing_package_terminals']), 161)
        self.assertEqual(sum(t['rows'] for t in rebuilt['totals'].values()), 900)
        primary = rebuilt['safe_discordances']['matched']
        self.assertEqual(len(primary['gained_safe']), 23)
        self.assertEqual(primary['lost_safe'], [])
        self.assertTrue(all(g['adaptive']['exact_pair_count']>1 for g in primary['gained_safe']))
        self.assertEqual(primary['gained_safe_decision_tiers'], {'TIER1_GATE_ELIMINATION': 2, 'TIER2_F0': 21})
        secondary = rebuilt['safe_discordances']['legacy']
        self.assertEqual(len(secondary['gained_safe']), 40)
        self.assertEqual(len(secondary['lost_safe']), 2)
        self.assertTrue(all(g['adaptive']['status']=='UNKNOWN' and g['adaptive']['exact_pair_count']==1
                            for g in secondary['lost_safe']))
        # The regression suite now runs on later implementation revisions.
        # Check the archived execution's Git bytes, not today's working files.
        # The actual archival CLI still requires the frozen current sources.
        data = subprocess.check_output(['git', 'archive', review.EXECUTION_HEAD, 'act'], cwd=review.PROJECT_ROOT)
        digest = hashlib.sha256()
        with tarfile.open(fileobj=io.BytesIO(data)) as source:
            for member in sorted((m for m in source.getmembers() if m.isfile() and m.name.endswith('.py')), key=lambda m: m.name):
                sha = hashlib.sha256(source.extractfile(member).read()).hexdigest()
                digest.update(f'{member.name}:{sha}\n'.encode())
        self.assertEqual(digest.hexdigest(), archived['source_sha256'])

    def test_archival_cli_still_rejects_changed_execution_sources(self):
        with patch.object(review, 'frozen_source_identity', return_value='changed'), patch.object(review, 'audit') as audit:
            with self.assertRaises(ValueError): review.build()
            audit.assert_not_called()

    def test_archival_write_never_overwrites_or_launches_new_queries(self):
        with patch('sys.argv', ['review', '--write']), patch.object(review, 'build') as build:
            with self.assertRaisesRegex(ValueError, 'do not overwrite'): review.main()
            build.assert_not_called()


if __name__ == '__main__': unittest.main()
