"""Finite stdlib regression controls for captured tensor-row intake."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.factored_source import Source
from scoped_source.tensor_rows import TensorRows
from scoped_source.tensor_row_controls import fixture, oracle
from source_enclosure.format import compact, identity


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE')
        self.root = Path(self.temp.name)/'source'
        self.sha = fixture(self.root, chunk_bytes=40)

    def tearDown(self):
        self.temp.cleanup()

    def reader(self, sha=None):
        return TensorRows(self.root, self.sha if sha is None else sha, deadline=time.monotonic()+300)

    def manifest(self):
        return json.loads((self.root/'manifest.json').read_text())

    def change(self, fn):
        m = self.manifest()
        fn(m)
        (self.root/'manifest.json').write_bytes(compact(m))
        return identity(m)

    def consume(self, sha=None):
        r = self.reader(sha)
        for _ in r.rows():
            pass
        return r.receipt()

    def test_small_oracle_and_old_path_exact(self):
        for size in (40, 1048576):
            root = Path(self.temp.name)/f'small{size}'
            sha = fixture(root, chunk_bytes=size)
            result = oracle(root, sha)
            self.assertEqual(result['receipt']['statistics']['affine_rows'], 10)
            old = Source(root, sha, lambda: None)
            old_affine = [(key, node) for key, node in old.nodes() if node['kind'] == 'affine']
            new = TensorRows(root, sha, deadline=time.monotonic()+300)
            rows = list(new.rows())
            for row, (key, node) in zip(rows, old_affine, strict=True):
                self.assertEqual(key, f'{row.graph}/layer/{row.layer}/value/{row.row}')
                self.assertEqual(row.bias, node['bias'])
                self.assertEqual([v for v in row.coefficients if v], list(node['terms'].values()))
            self.assertEqual(new.receipt()['statistics']['values'], 45)

    def test_old_materializers_never_called(self):
        with patch.object(Source, 'tensor', side_effect=AssertionError('whole tensor forbidden')), \
                patch.object(Source, 'nodes', side_effect=AssertionError('old nodes forbidden')):
            self.consume()

    def test_partial_close_and_last_row_need_exhaustion(self):
        for count in (1, 10):
            reader = self.reader()
            iterator = reader.rows()
            for _ in range(count):
                next(iterator)
            with self.assertRaises(ValueError):
                reader.receipt()
            iterator.close()
            with self.assertRaises(ValueError):
                reader.receipt()
            with self.assertRaises(ValueError):
                next(reader.rows())

    def test_duplicate_active_iterator_refused(self):
        reader = self.reader()
        first = reader.rows()
        next(first)
        with self.assertRaises(ValueError):
            next(reader.rows())
        list(first)
        reader.receipt()
        with self.assertRaises(ValueError):
            next(reader.rows())

    def test_public_aliases_and_replay(self):
        reader = self.reader()
        reader.request['label'] = 1
        reader.statistics['raw_bytes'] = -1
        iterator = reader.rows()
        row = next(iterator)
        with self.assertRaises(FrozenInstanceError):
            row.bias = 999
        with self.assertRaises(FrozenInstanceError):
            reader.plan[0].kind = 'Invalid'
        list(iterator)
        original = reader.receipt()
        self.assertEqual(reader.request['label'], 0)
        altered = reader.receipt()
        altered['tensor_hashes'].clear()
        self.assertEqual(original, self.consume())

    def test_wrong_external_identity(self):
        with self.assertRaisesRegex(ValueError, 'external'):
            self.reader('0'*64)

    def test_graph_semantic_mutations(self):
        original = (self.root/'manifest.json').read_bytes()
        mutations = [
            lambda d: d['request'].update(gate='CONSTANT'),
            lambda d: d['request'].update(tie_policy='FIRST'),
            lambda d: d['request'].update(training=True),
            lambda d: d['request'].update(label=2),
            lambda d: d['request'].update(top_k=True),
            lambda d: d['networks'][0]['layers'][0].update(dimensions=[0, -1]),
            lambda d: d['networks'][1]['layers'][2].update(inplace=True),
            lambda d: d['networks'][0]['layers'][1].update(index=2),
            lambda d: d['networks'][0]['layers'][1].update(training=True),
            lambda d: d['networks'][0]['layers'][1].update(weight_name='experts.0.1.weight'),
            lambda d: d['networks'][0]['layers'][1].update(weight={'tensor_ref': 'experts.0.1.weight'}),
            lambda d: d['networks'].pop(),
            lambda d: d['center'].update(tensor_ref='router.1.bias'),
            lambda d: d['request']['center'].update(sha256='0'*64),
            lambda d: d['request'].update(margin='-1'),
        ]
        for index, fn in enumerate(mutations):
            with self.subTest(index=index):
                (self.root/'manifest.json').write_bytes(original)
                sha = self.change(lambda m: fn(m['declaration']))
                with self.assertRaises(ValueError):
                    self.consume(sha)

    def test_unused_tensor_and_wrong_shape(self):
        original = (self.root/'manifest.json').read_bytes()
        sha = self.change(lambda m: m['tensors'][-1].update(shape=[3, 2]))
        with self.assertRaises(ValueError):
            self.consume(sha)
        (self.root/'manifest.json').write_bytes(original)
        sha = self.change(lambda m: m['declaration']['networks'][1]['layers'].__delitem__(slice(2, None)))
        with self.assertRaises(ValueError):
            self.consume(sha)

    def test_full_tensor_hash_not_only_chunk_hash(self):
        def mutate(m):
            inv = m['declaration']['state_inventory']
            inv[-1]['sha256'] = '0'*64
            h = hashlib.sha256()
            for v in inv:
                h.update(v['name'].encode())
                h.update(v['sha256'].encode())
            m['declaration']['request']['model_state']['sha256'] = h.hexdigest()
        sha = self.change(mutate)
        r = self.reader(sha)
        with self.assertRaisesRegex(ValueError, 'tensor/state'):
            list(r.rows())
        with self.assertRaises(ValueError):
            r.receipt()

    def test_corrupt_truncate_and_missing_member(self):
        target = self.root/self.manifest()['tensors'][-1]['chunks'][-1]['file']
        original = target.read_bytes()
        for raw in (b'', original[:-8], bytes([original[0] ^ 1])+original[1:], original+b'12345678'):
            target.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.consume()
        target.unlink()
        with self.assertRaises(ValueError):
            self.reader()

    def test_extra_directory_and_symlink(self):
        extra = self.root/'extra'
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, 'extra'):
            self.reader()
        extra.rmdir()
        extra.symlink_to(self.root/'t', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.reader()

    def test_chunk_descriptor_mutations(self):
        original = (self.root/'manifest.json').read_bytes()
        for change in ({'offset': 8}, {'bytes': 7}, {'file': '../outside'}, {'sha256': '0'*64}):
            (self.root/'manifest.json').write_bytes(original)
            sha = self.change(lambda m: m['tensors'][-1]['chunks'][0].update(change))
            with self.assertRaises(ValueError):
                self.consume(sha)

    def test_nonfinite_even_with_correct_chunk_hash(self):
        m = self.manifest()
        chunk = m['tensors'][-1]['chunks'][0]
        target = self.root/chunk['file']
        raw = struct.pack('<d', float('nan'))+target.read_bytes()[8:]
        target.write_bytes(raw)
        chunk['sha256'] = hashlib.sha256(raw).hexdigest()
        (self.root/'manifest.json').write_bytes(compact(m))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.consume(identity(m))

    def test_mutation_during_read_no_completion(self):
        reader = self.reader()
        iterator = reader.rows()
        next(iterator)
        m = self.manifest()
        chunk = next(t for t in m['tensors'] if t['name'] == 'experts.1.1.weight')['chunks'][-1]
        p = self.root/chunk['file']
        p.write_bytes(b'\x00'*chunk['bytes'])
        with self.assertRaises(ValueError):
            list(iterator)
        with self.assertRaises(ValueError):
            reader.receipt()
        with self.assertRaises(ValueError):
            self.consume()

    def test_manifest_changed_after_rows_started(self):
        reader = self.reader()
        iterator = reader.rows()
        next(iterator)
        self.change(lambda m: m['declaration']['request'].update(label=1))
        with self.assertRaises(ValueError):
            list(iterator)
        with self.assertRaises(ValueError):
            reader.receipt()

    def test_deadline_during_consumption(self):
        reader = self.reader()
        iterator = reader.rows()
        next(iterator)
        with patch('scoped_source.tensor_rows.time.monotonic', return_value=reader._deadline+1):
            with self.assertRaises(TimeoutError):
                next(iterator)
        with self.assertRaises(ValueError):
            reader.receipt()

    def test_deadline_in_inventory(self):
        reader = self.reader()
        from scoped_source.tensor_rows import _bounded_inventory
        def expired():
            raise TimeoutError('controlled inventory deadline')
        with self.assertRaises(TimeoutError):
            _bounded_inventory(self.root, reader._files, expired)

    def test_exponent_literal_rejected_before_fraction(self):
        reader = self.reader()
        with patch('scoped_source.tensor_rows.Fraction', side_effect=AssertionError('must reject before parsing')):
            for value in ('1e1000000000', '1/0e999999', '9'*4097, 1 << 8193):
                with self.assertRaisesRegex(ValueError, 'bounded rational'):
                    reader._fraction(value)

    def test_input_limit_precedes_values(self):
        def mutate(m):
            m['declaration']['request']['center']['shape'] = [1, 4097]
            m['tensors'][0]['shape'] = [1, 4097]
            m['tensors'][0]['chunks'][0]['bytes'] = 4097*8
        sha = self.change(mutate)
        with patch.object(TensorRows, '_values', side_effect=AssertionError('input must refuse before decode')):
            with self.assertRaisesRegex(ValueError, 'bounded input'):
                self.reader(sha)

    def test_changed_property_keeps_no_old_identity(self):
        self.change(lambda m: m['declaration']['request'].update(label=1))
        with self.assertRaisesRegex(ValueError, 'external'):
            self.consume()


class ArchiveBoundaryControls(unittest.TestCase):
    def test_terminal_own_bytes_count_before_success(self):
        from scripts.run_tensor_rows_controls import terminal_bytes
        base = {'status': 'PASS', 'x': 1}
        _, raw = terminal_bytes(base, 0, 1000)
        accepted, _ = terminal_bytes(base, 1000-len(raw), 1000)
        self.assertEqual(accepted['status'], 'PASS')
        refused, raw = terminal_bytes(base, 1001-len(raw), 1000)
        self.assertEqual(refused['status'], 'ARCHIVE_LIMIT_REFUSED')
        self.assertEqual(json.loads(raw)['status'], 'ARCHIVE_LIMIT_REFUSED')
        self.assertEqual(base['status'], 'PASS')

    def test_failure_is_not_promoted_by_serialization(self):
        from scripts.run_tensor_rows_controls import terminal_bytes
        summary, raw = terminal_bytes({'status': 'FAILED'}, 20, 1000)
        self.assertEqual(summary['status'], 'FAILED')
        self.assertEqual(json.loads(raw), summary)


if __name__ == '__main__':
    unittest.main()
