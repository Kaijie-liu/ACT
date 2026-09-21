import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robust_experts_pipeline_r2 import configuration


class ProductionRecipe(unittest.TestCase):
    def test_full_not_short_control(self):
        from omegaconf import OmegaConf
        raw=json.loads(Path('configs/recent_moe/robust_experts_paper_training_recipe_r1.json').read_text())
        for arm,recipe in raw['recipes'].items():
            cfg=configuration(recipe,Path('/data1/Kane/MOE/not-executed'),'/data1/Kane/MOE/data','training','train')
            self.assertEqual(cfg.trainer.max_epochs,200)
            self.assertEqual(cfg.trainer.min_epochs,200)
            self.assertEqual(cfg.model.scheduler.T_max,200)
            self.assertIsNone(cfg.trainer.resume_from_checkpoint)
            for key in ['limit_train_batches','limit_val_batches','limit_test_batches','max_steps']:
                self.assertNotIn(key,cfg.trainer)
            self.assertEqual(cfg.model.optimizer.lr,.01)
            self.assertEqual(cfg.datamodule.batch_size,640)
            self.assertEqual(cfg.datamodule.num_workers,2)
            self.assertEqual(cfg.callbacks.checkpoint.save_top_k,-1)
            self.assertEqual(cfg.logger.csv.version,'train')
            self.assertFalse(cfg.wandb)
            self.assertFalse(cfg.use_clearml)
            if arm=='convmoe':self.assertEqual(cfg.model.model.k,2)
            # Symbolic method resolvers remain serializable without import-time callables.
            json.dumps(OmegaConf.to_container(cfg,resolve=False))


if __name__=='__main__':unittest.main()
