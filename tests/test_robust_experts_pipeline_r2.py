import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robust_experts_pipeline_r2 import configuration
from robust_experts_supervised_pipeline import configuration as old


class StageLogControls(unittest.TestCase):
    def test_only_log_version_changed(self):
        from omegaconf import OmegaConf
        recipes=json.loads(Path('configs/recent_moe/robust_experts_paper_training_recipe_r1.json').read_text())['recipes']
        for recipe in recipes.values():
            ref=OmegaConf.to_container(old(recipe,Path('/data1/Kane/MOE/control'),'/data1/Kane/MOE/data','control'),resolve=False)
            for stage in ['train','evaluate']:
                cfg=OmegaConf.to_container(configuration(recipe,Path('/data1/Kane/MOE/control'),
                    '/data1/Kane/MOE/data','control',stage),resolve=False)
                self.assertEqual(cfg['logger']['csv']['version'],stage)
                cfg['logger']['csv']['version']='pipeline'
                self.assertEqual(cfg,ref)


if __name__=='__main__':unittest.main()
