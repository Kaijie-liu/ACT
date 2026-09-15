import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.review_conv_sign_lp import journal_check
from scripts.conv_sign_lp_contract import DEFAULT, ROOT, read, sha


class CaptureReviewTests(unittest.TestCase):
    def check_mutation(self, mutate, rehash=False):
        d=DEFAULT/'rank2_monolithic'; events=[json.loads(l) for l in (d/'budget_journal.jsonl').read_text().splitlines()]
        mutate(events)
        if rehash:
            previous='0'*64
            for seq,e in enumerate(events):
                e.pop('sha256',None);e.update(seq=seq,previous=previous)
                previous=hashlib.sha256(json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                e['sha256']=previous
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            p=Path(tmp)/'journal.jsonl'
            p.write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaises(ValueError):journal_check(p,sha(d/'job.json'),read(d/'job.json')['expected_scope'])

    def test_real_capture_is_not_property_solve_or_verdict(self):
        for job in ('rank2_monolithic','rank24_monolithic'):
            d=DEFAULT/job;r=journal_check(d/'budget_journal.jsonl',sha(d/'job.json'),read(d/'job.json')['expected_scope'])
            self.assertEqual(r['property_native_solves'],0)
            self.assertFalse(r['request_verdict_emitted'])

    def test_hash_corruption_rejected(self):
        self.check_mutation(lambda es:es[-1].update(exception='corrupt'))

    def test_rehashed_wrong_capture_exit_rejected(self):
        self.check_mutation(lambda es:es[-1].update(exception='BudgetExhausted'),rehash=True)

    def test_wrong_property_even_with_valid_hash_rejected(self):
        def mutate(es):
            e=next(e for e in es if e['kind']=='PROPERTY_BEGIN')
            e['scope']['pairs']=[[1,2]]
        self.check_mutation(mutate,rehash=True)


if __name__=='__main__':unittest.main()
