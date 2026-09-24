"""Run unchanged publication controls against new writer, plus batch boundaries."""
import json
import time
from unittest.mock import patch
import unittest

from bounded_evidence import tests as previous
from bounded_evidence import stream as original
from batched_evidence import stream
from batched_evidence.worker import POLICY
from scoped_proof.io import PYTHON, load


def command(phase, root, deadline):
    from residual_proof.supervisor import command as old
    if phase == 'construct':
        return [PYTHON, '-S', '-m', 'batched_evidence.worker', str(root), '--deadline', str(deadline)]
    return old(phase, root, deadline, load(root / 'spec.json'))


class SerializationControls(previous.SerializationControls):
    def setUp(self):
        self.binding = patch.object(previous, 'stream', stream); self.binding.start()
        super().setUp()
    def tearDown(self):
        try: super().tearDown()
        finally: self.binding.stop()

    def test_batch_edges_nested_and_fallback_preserve_bytes(self):
        for count in (0, 1, 127, 128, 129, 257):
            self.check({'x': [None, True, -0., '中', -10**100] * count,
                        'mixed': [list(range(count)), {'a': 2}, ('x' * 2049,), '😀' * 3000]}, block=128)

    def test_all_batches_bounded_and_scalar_inventory_kept(self):
        values = [str(i) + '/999' for i in range(4097)]
        stats = {}; path = self.root / 'data'
        stream.save(path, values, deadline=time.monotonic() + 30, metrics=stats)
        self.assertEqual(load(path), values)
        self.assertEqual(stats['batched_scalars'], len(values))
        self.assertEqual(stats['native_batches'], 33)
        self.assertLessEqual(stats['max_batch_bytes'], 16384)
        self.assertLessEqual(stats['max_batch_items'], 128)

    def test_bad_scalar_after_batch_never_publishes(self):
        for i, bad in enumerate((float('nan'), float('inf'), object(), 1 << 12001)):
            path = self.root / str(i)
            with self.assertRaises((ValueError, TypeError)):
                stream.save(path, [0] * 129 + [bad], deadline=time.monotonic() + 30)
            self.assertFalse(path.exists())

    def test_batch_depth_limit_same_as_original(self):
        obj = [1, 2, 3]
        for _ in range(64): obj = [obj]
        for i, writer in enumerate((original, stream)):
            with self.assertRaises(ValueError): writer.save(self.root / str(i), obj, deadline=time.monotonic() + 30)

    def test_fallback_long_strings_same_chunk_cap(self):
        stats = {}; value = ['😀' * 8000, 3, 'z' * 1600, {'v': ['7/9'] * 200}]
        data = b''.join(stream.chunks(value, lambda: None, metrics=stats))
        self.assertEqual(data, json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        self.assertLessEqual(stats['max_fragment_bytes'], 24576)


class PipelineControls(previous.PipelineControls):
    @classmethod
    def setUpClass(cls):
        cls.bindings = [patch.object(previous, 'command', command),
                        patch.object(previous, 'POLICY', POLICY), patch.object(previous, 'stream', stream)]
        for binding in cls.bindings: binding.start()
        try: super().setUpClass()
        except BaseException:
            for binding in reversed(cls.bindings): binding.stop()
            raise
    @classmethod
    def tearDownClass(cls):
        try: super().tearDownClass()
        finally:
            for binding in reversed(cls.bindings): binding.stop()

    def test_watchdog_partial_and_exception_charged_without_acceptance(self):
        # Reuse the unchanged assertions, changing ONLY the writer under fault.
        from residual_proof import supervisor
        original = supervisor.supervise
        def adapt(*args, **kwargs):
            factory = kwargs['command_factory']
            def factory2(phase, root, end):
                return ['batched_evidence.probe' if v == 'bounded_evidence.probe' else v
                        for v in factory(phase, root, end)]
            return original(*args, **dict(kwargs, command_factory=factory2))
        with patch.object(supervisor, 'supervise', adapt):
            super().test_watchdog_partial_and_exception_charged_without_acceptance()


if __name__ == '__main__': unittest.main()
