from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_rome_batch_evidence import kind


class Interpretation(unittest.TestCase):
    def base(self):
        return {'status':'ATTACK_EVALUATION_COMPLETED','label':7,'clean_prediction':7,
            'adversarial_prediction':4,'empirically_correct_after_attack':False}
    def test_clean_error_not_attack_gain(self):
        self.assertEqual(kind({**self.base(),'clean_prediction':4}),'PREEXISTING_CLEAN_ERROR')
        self.assertEqual(kind(self.base()),'PERTURBATION_BREAK_REPLAYED')
    def test_timeout_no_promotion(self):
        self.assertEqual(kind({'status':'TIMEOUT'}),'INCOMPLETE_NOT_ROBUST')
    def test_mismatch_reject_and_no_break_label(self):
        with self.assertRaises(ValueError):kind({**self.base(),'empirically_correct_after_attack':True})
        self.assertEqual(kind({**self.base(),'adversarial_prediction':7,'empirically_correct_after_attack':True}),
            'NO_BREAK_FOUND_COMPLETED')


if __name__=='__main__':unittest.main()
