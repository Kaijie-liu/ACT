import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robust_experts_gpu_step_control import configured


class GPURecipe(unittest.TestCase):
    def test_no_batch_or_pgd_reduction(self):
        raw=json.loads(Path('configs/recent_moe/robust_experts_paper_training_recipe_r1.json').read_text())
        for arm,recipe in raw['recipes'].items():
            cfg=configured(recipe,Path('/data1/Kane/MOE/control-not-executed'),Path('/data1/Kane/MOE/data'))
            self.assertEqual(cfg.datamodule.batch_size,640)
            self.assertEqual(cfg.model.attack.steps,7)
            self.assertEqual(cfg.trainer.max_epochs,200)
            self.assertEqual(cfg.trainer.max_steps,1)
            self.assertFalse(cfg.execute_attack)
            if arm=='convmoe':self.assertEqual(cfg.model.model.k,2)
            recipe['datamodule']['batch_size']=2
            with self.assertRaises(ValueError):configured(recipe,Path('/data1/Kane/MOE/test'),Path('/data1/Kane/MOE/data'))


if __name__=='__main__':unittest.main()
