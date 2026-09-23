import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_assignment_layers_r2 import append_progress


class RecorderTests(unittest.TestCase):
    def test_multiple_layers_distinct_and_duplicate_refused(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as folder:
            p=Path(folder)/'layer_progress.json'
            append_progress(p,{'layers':[{'layer':0}]})
            append_progress(p,{'layers':[{'layer':0},{'layer':1}]})
            self.assertTrue((p.parent/'layer_progress_000.json').exists())
            self.assertTrue((p.parent/'layer_progress_001.json').exists())
            self.assertFalse(p.exists())
            with self.assertRaises(FileExistsError):append_progress(p,{'layers':[{'layer':1}]})
    def test_other_file_unchanged(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as folder:
            p=Path(folder)/'result.json';append_progress(p,{'status':'ERROR'})
            self.assertTrue(p.exists())
            with self.assertRaises(FileExistsError):append_progress(p,{'status':'PASS'})
    def test_invalid_layer(self):
        with self.assertRaises(ValueError):append_progress('/data1/Kane/MOE/layer_progress.json',{'layers':[{'layer':-1}]})


if __name__=='__main__':unittest.main()
