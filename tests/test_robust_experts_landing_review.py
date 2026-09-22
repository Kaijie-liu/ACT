import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import review_robust_experts_landing as review


class SavedReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'local/train').mkdir(parents=True)
        (self.root/'local/evaluate').mkdir(parents=True)
        self.recipe = {'trainer':{'max_epochs':2}, 'model':{'optimizer':{'lr':.01}}}
        self.train = {'epochs':2, 'global_step':126, 'state_digest':'state',
                      'observations':{'updates':126,'epoch_batches':{str(e):[640]*62+[320] for e in range(2)}}}
        rows = []
        for e in range(2):
            self.write(f'epoch{e:03d}.json', {'epoch':e, 'global_step':63*(e+1),
                'train_batch_sizes':[640]*62+[320], 'elapsed_seconds':10*(e+1),
                'lr':[.01*(1-(e+1)/2)**.9]})
            rows.append({'epoch':e,'step':63*(e+1)-1,'train/acc':.1,'val/acc':.2,
                         'loss':.5,'main_loss':3.5,'aux_loss':-3.,
                         'val/loss':.4,'val/main_loss':3.4,'val/aux_loss':-3.})
        self.csv('local/train/metrics.csv', rows)
        self.ev = {'full_test_set':True,'empirical_only':True,'formal_SAFE':False,'records':[]}
        metrics_rows = []
        for kind, prefix in zip(('clean','PGD20','APGD20'),('test/','attack/pgd/','attack/apgd/')):
            metrics = {prefix+'acc':.2, prefix+'loss':.5, prefix+'main_loss':3.5, prefix+'aux_loss':-3.}
            record = {'kind':kind, 'examples':10000, 'batches':16 if kind=='clean' else 40,
                      'state_digest':'state','checkpoint_sha256':'weight','metrics':[metrics],'seconds':1.}
            self.ev['records'].append(record)
            self.write('evaluation_'+kind+'.json',record)
            metrics_rows.append(metrics)
        self.csv('local/evaluate/metrics.csv',metrics_rows)

    def write(self, name, obj):
        (self.root/name).write_text(json.dumps(obj))

    def csv(self, name, rows):
        with (self.root/name).open('w', newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            w.writeheader();w.writerows(rows)

    def test_valid_negative_entropy_is_not_missing_loss(self):
        rows = review.trajectory(self.root,self.train,self.recipe)
        self.assertEqual(rows[-1]['main_loss'],3.5)
        self.assertEqual(review.evaluation(self.root,self.train,self.ev,'weight')['clean']['correct_from_aggregate'],2000)

    def test_missing_epoch_rejected(self):
        (self.root/'epoch001.json').unlink()
        with self.assertRaisesRegex(ValueError,'journal roster'):
            review.trajectory(self.root,self.train,self.recipe)

    def test_batch_loss_rejected(self):
        obj = review.read(self.root/'epoch000.json');obj['train_batch_sizes'].pop()
        self.write('epoch000.json',obj)
        with self.assertRaisesRegex(ValueError,'coverage'):
            review.trajectory(self.root,self.train,self.recipe)

    def test_lr_change_rejected(self):
        obj = review.read(self.root/'epoch000.json');obj['lr']=[.1]
        self.write('epoch000.json',obj)
        with self.assertRaisesRegex(ValueError,'LR trajectory'):
            review.trajectory(self.root,self.train,self.recipe)

    def test_csv_epoch_or_loss_mutation_rejected(self):
        for field,value,match in [('main_loss',8.,'loss decomposition'),('epoch',4,'CSV epoch roster')]:
            with self.subTest(field=field):
                original=(self.root/'local/train/metrics.csv').read_text()
                with (self.root/'local/train/metrics.csv').open() as f: rows=list(csv.DictReader(f))
                rows[0][field]=value;self.csv('local/train/metrics.csv',rows)
                with self.assertRaisesRegex(ValueError,match): review.trajectory(self.root,self.train,self.recipe)
                (self.root/'local/train/metrics.csv').write_text(original)

    def test_missing_attack_rejected(self):
        self.ev['records'].pop()
        with self.assertRaisesRegex(ValueError,'test roster'):
            review.evaluation(self.root,self.train,self.ev,'weight')

    def test_wrong_weight_or_denominator_rejected(self):
        for field,value,match in [('checkpoint_sha256','other','state identity'),('examples',9999,'denominator')]:
            with self.subTest(field=field):
                ev=copy.deepcopy(self.ev);ev['records'][0][field]=value
                self.write('evaluation_clean.json',ev['records'][0])
                with self.assertRaisesRegex(ValueError,match): review.evaluation(self.root,self.train,ev,'weight')

    def test_csv_disagreement_rejected(self):
        path=self.root/'local/evaluate/metrics.csv'
        path.write_text(path.read_text().replace('0.2','0.3'))
        with self.assertRaisesRegex(ValueError,'CSV inconsistency'):
            review.evaluation(self.root,self.train,self.ev,'weight')

    def test_state_digest_differs_on_any_tensor_change(self):
        import torch
        state={'a':torch.tensor([1.,2.]),'b':torch.tensor(3)}
        before=review.state_digest(state)
        state['a'][0]=2.
        self.assertNotEqual(before,review.state_digest(state))
        state['a'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):review.state_digest(state)


if __name__=='__main__':
    unittest.main()
