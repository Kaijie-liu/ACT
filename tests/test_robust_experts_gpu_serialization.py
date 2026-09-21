import json
import unittest
from omegaconf import OmegaConf


class Serialization(unittest.TestCase):
    def test_symbolic_author_method_preserved(self):
        name='gpu_control_test_callable'
        OmegaConf.register_new_resolver(name,lambda: lambda x:x)
        try:
            cfg=OmegaConf.create({'transform':'${'+name+':}'})
            with self.assertRaises(TypeError):json.dumps(OmegaConf.to_container(cfg,resolve=True))
            value=OmegaConf.to_container(cfg,resolve=False)
            self.assertEqual(json.loads(json.dumps(value))['transform'],'${'+name+':}')
        finally:
            OmegaConf.clear_resolver(name)


if __name__=='__main__':unittest.main()
