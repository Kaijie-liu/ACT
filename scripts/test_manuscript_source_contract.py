"""Manuscript preservation/accounting checks only; no model or solver imports."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/'docs/manuscript_source_consolidation_20260920.json'


def preserved(text,name,expected):
    start=f'<!-- BEGIN PRESERVED {name} -->\n';end=f'<!-- END PRESERVED {name} -->'
    if text.count(start)!=1 or text.count(end)!=1:raise ValueError('preservation markers')
    a=text.index(start)+len(start);b=text.index(end,a);block=text[a:b]
    if (hashlib.sha256(block.encode()).hexdigest()!=expected['sha256'] or
            len(block.encode())!=expected['bytes'] or block.count('\n')!=expected['lines']):
        raise ValueError('preserved block changed')
    return block


class ManuscriptContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=json.loads(RECEIPT.read_bytes())
        cls.appendix=(ROOT/cls.r['appendix']).read_text()

    def test_exact_history_preserved_not_repeated_in_main(self):
        total=0
        for name,record in self.r['preserved'].items():
            block=preserved(self.appendix,name,record);total+=record['lines']
            self.assertNotIn(block,(ROOT/record['old_path']).read_text())
        self.assertEqual(total,244)

    def test_preservation_rejects_missing_or_mutated_blocks(self):
        for name,record in self.r['preserved'].items():
            block=preserved(self.appendix,name,record)
            for text in (self.appendix.replace(block,block[1:],1),
                         self.appendix.replace(f'<!-- END PRESERVED {name} -->','',1)):
                with self.assertRaises(ValueError):preserved(text,name,record)

    def test_pinned_evidence_and_main_table_unchanged(self):
        for p,h in self.r['evidence_sha256'].items():
            with self.subTest(path=p):self.assertEqual(hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),h)

    def test_old_positive_and_new_source_result_are_not_spliced(self):
        old=json.loads((ROOT/'docs/portable_conv_proof_v1_review.json').read_bytes())
        new=json.loads((ROOT/'docs/full_source_v1_review.json').read_bytes())
        self.assertEqual(old['positive_obligations'],old['required_obligations'])
        self.assertTrue(old['trusted_upstream_lowering']);self.assertFalse(old['deployed_float_SAFE'])
        self.assertTrue(new['complete']);self.assertFalse(new['complete_output_positive_proof'])
        self.assertFalse(new['complete_strict_network_certificate'])
        final=json.loads((ROOT/'docs/property_ranges_v1_analysis.json').read_bytes())
        self.assertEqual(final['complete_positive_requests'],0)
        self.assertEqual([(a['positive'],a['nonpositive'],a['missing']) for a in final['arms']],[(0,9,0),(0,8,1)])

    def test_stop_boundary_not_an_experiment(self):
        stop=json.loads((ROOT/'docs/property_diagnosis_v1_controls.json').read_bytes())
        self.assertEqual(stop['decision'],'STOP_INPUT98_FOLLOWUP')
        self.assertFalse(stop['new_control_proposed'])
        for k in ('new_solver_calls','new_model_forwards','new_source_propagations','new_lower_bounds_proved'):
            self.assertEqual(stop[k],0)
        self.assertFalse(self.r['new_experiment_authorized']);self.assertFalse(self.r['acceptance_changed'])
        self.assertFalse(self.r['complete_source_checked_positive_claim'])

    def test_changed_document_local_links_exist(self):
        for p in self.r['manuscript_files']:
            file=ROOT/p
            for link in re.findall(r'\]\(([^\s)]+)\)',file.read_text()):
                if ':' in link or link.startswith('#'):continue
                target=(file.parent/link.split('#')[0]).resolve()
                with self.subTest(source=p,target=str(target)):self.assertTrue(target.is_file())


if __name__=='__main__':unittest.main()
