import ast
import tempfile
import unittest
from pathlib import Path
import zipfile

from portable_proof.runtime import Store, compact, digest, strict_json
from scripts.build_portable_conv_proof import ROOT, extract


class PortableTests(unittest.TestCase):
    def test_checker_extraction_excludes_solver(self):
        for name in ('lp_certificate', 'sparse_lp_certificate'):
            text = extract(ROOT / f'act/back_end/solver/{name}.py')
            ast.parse(text)
            self.assertNotIn('def propose(', text)
            self.assertNotIn('scipy', text)
            self.assertNotIn('linprog', text)

    def test_only_requested_aggregation_helpers(self):
        text = extract(ROOT / 'scripts/check_conv_pre_f0_r2.py', {'aggregate', 'order_bounds'})
        self.assertNotIn('validate_job', text)
        self.assertNotIn('check_directory', text)
        ast.parse(text)

    def test_json_ambiguity_rejected(self):
        for raw in ('{"a":1,"a":2}', '[NaN]', '[Infinity]'):
            with self.assertRaises(ValueError):
                strict_json(raw)

    def test_array_roundtrip_and_bad_reference(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/moe/results') as tmp:
            path = Path(tmp) / 'proof.zip'
            arr = [-0.0, 1, '1/2', 0.1] * 10
            raw = compact(arr)
            sha = digest(raw)
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr(sha, raw)
            store = Store(path)
            self.assertEqual(compact(store.decode({'$array': sha})), raw)
            for bad in ('../source', 'z'*64, None):
                with self.assertRaises(ValueError):
                    store.get(bad)
            store.zip.close()

    def test_corrupt_object_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/moe/results') as tmp:
            path = Path(tmp) / 'proof.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('0'*64, '[1]')
            store = Store(path)
            with self.assertRaisesRegex(ValueError, 'content hash'):
                store.get('0'*64)
            store.zip.close()


if __name__ == '__main__':
    unittest.main()
