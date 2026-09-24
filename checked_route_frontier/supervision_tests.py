"""Synthetic hard-cutoff, partial-evidence and complete-cost controls only."""
import copy
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from checked_route_frontier import study
from scoped_proof.io import load, save


class SupervisorControls(unittest.TestCase):
    def folder(self):
        folder = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        self.addCleanup(folder.cleanup)
        return Path(folder.name)/'run'

    def test_actual_complete_synthetic_all_phases_and_cost(self):
        root = self.folder()
        result = study.run_one(root, 'prunable', 'frontier', budget=30)
        self.assertEqual(result['status'], 'COMPLETED_SYNTHETIC_CHECK')
        cost, terminal, checked = [load(root/name) for name in ('cost.json', 'terminal.json', 'check.json')]
        self.assertTrue(terminal['synthetic_positive'])
        self.assertEqual((checked['required'], checked['positive_bounds'], checked['discharged_by_exclusion']), (252, 9, 243))
        self.assertFalse(terminal['new_real_positive'])
        self.assertEqual([row['phase'] for row in terminal['phases']], ['build', 'check'])
        self.assertAlmostEqual(cost['stage_seconds'] + cost['overhead_seconds'], cost['end_to_end_seconds'])
        self.assertGreaterEqual(result['seconds'], cost['end_to_end_seconds'])
        self.assertGreater(cost['sampled_peak_rss'], 0)
        with self.assertRaises(FileExistsError):
            study.run_one(root, 'prunable', 'frontier')

    def test_deadline_keeps_partial_evidence_not_success(self):
        root = self.folder()
        def command(phase, directory, deadline):
            return [sys.executable, '-S', '-c',
                    "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('partial'); time.sleep(30)",
                    str(directory/'partial.txt')]
        result = study.run_one(root, 'prunable', 'frontier', budget=.3, commands=command)
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertTrue((root/'partial.txt').exists())
        self.assertFalse(load(root/'terminal.json')['synthetic_positive'])
        self.assertFalse((root/'check.json').exists())
        self.assertLess(result['seconds'], 1.5)

    def test_exception_retains_partial_and_never_starts_check(self):
        root = self.folder()
        def command(phase, directory, deadline):
            return [sys.executable, '-S', '-c',
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('partial'); raise RuntimeError('control')",
                    str(directory/'partial.txt')]
        result = study.run_one(root, 'tied', 'frontier', budget=2, commands=command)
        self.assertEqual(result['status'], 'ERROR')
        self.assertTrue((root/'partial.txt').exists())
        self.assertEqual(len(load(root/'terminal.json')['phases']), 1)

    def test_successful_empty_workers_cannot_forge_receipt(self):
        root = self.folder()
        result = study.run_one(root, 'tied', 'frontier', budget=2,
            commands=lambda *args: [sys.executable, '-S', '-c', 'pass'])
        self.assertEqual(result['status'], 'ERROR')
        self.assertFalse(load(root/'terminal.json')['synthetic_positive'])

    def test_late_complete_positive_file_is_not_accepted(self):
        root = self.folder()
        def command(phase, directory, deadline):
            if phase == 'build':
                return [sys.executable, '-S', '-m', 'checked_route_frontier.worker', phase,
                        str(directory), '--deadline', str(deadline)]
            return [sys.executable, '-S', '-c',
                    "import pathlib,sys,time; from checked_route_frontier.worker import work; "
                    "work('check',pathlib.Path(sys.argv[1]),float(sys.argv[2])); time.sleep(30)",
                    str(directory), str(deadline)]
        result = study.run_one(root, 'prunable', 'frontier', budget=8, commands=command)
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertTrue(load(root/'check.json')['complete_output_positive_proof'])
        self.assertFalse(load(root/'terminal.json')['synthetic_positive'])

    def test_final_ledger_overrun_invalidates_completion(self):
        root = self.folder()
        original_time = time.monotonic
        shift = [0.]
        def write(path, value):
            result = save(path, value)
            if Path(path).name == 'cost.json':
                shift[0] = 31.
            return result
        with patch('checked_route_frontier.study.save', side_effect=write), \
             patch('checked_route_frontier.study.time.monotonic', side_effect=lambda: original_time()+shift[0]):
            result = study.run_one(root, 'prunable', 'frontier', budget=30)
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertTrue((root/'publication_timeout.json').exists())


if __name__ == '__main__':
    unittest.main()
