from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp

import scratch_phase_projection_float64_probe as probe


class TriangularExpansionTests(unittest.TestCase):
    def test_up_changes_include_prior_influences_and_down_change_does_not(self):
        changes = [
            (1, 10, False, True),
            (2, 20, False, True),
            (3, 30, True, False),
        ]
        positions = {1: {10: 0}, 2: {20: 0}, 3: {30: 0}}
        first_pre = {
            1: np.array([[1.0, 2.0]]),
            2: np.array([[3.0, -1.0]]),
            3: np.array([[4.0, 5.0]]),
        }
        delta_pre = {
            1: np.array([[0.0, 0.0, 0.0]]),
            2: np.array([[0.5, 0.0, 0.0]]),
            3: np.array([[7.0, 11.0, 0.0]]),
        }

        actual = probe._triangular_input_expansion(
            changes,
            positions,
            first_pre,
            delta_pre,
            input_width=2,
        )

        expected = np.array(
            [
                [1.0, 2.0],
                [3.5, 0.0],
                [-4.0, -5.0],
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_non_change_is_rejected(self):
        with self.assertRaisesRegex(Exception, "invalid phase change"):
            probe._triangular_input_expansion(
                [(1, 10, True, True)],
                {1: {10: 0}},
                {1: np.array([[1.0]])},
                {1: np.array([[0.0]])},
                input_width=1,
            )


class CsrBoxUpperTests(unittest.TestCase):
    def test_matches_dense_corner_evaluation_with_empty_row(self):
        matrix = sp.csr_matrix(
            np.array(
                [
                    [2.0, -3.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-4.0, 0.0, 5.0],
                ]
            )
        )
        lower = np.array([-1.0, -2.0, -3.0])
        upper = np.array([4.0, 5.0, 6.0])
        actual = probe._csr_box_upper(matrix, lower, upper)
        expected = np.array([14.0, 0.0, 34.0])
        np.testing.assert_array_equal(actual, expected)

    def test_dense_all_nonzero_support_matches_exact_reference(self):
        class Snapshot:
            kind = "DENSE"
            input_size = 3
            output_size = 2

        snapshot = Snapshot()
        for mask in (
            np.array([False, False, False]),
            np.array([False, True, False]),
        ):
            actual = probe._all_nonzero_affine_support_forward(snapshot, mask)
            expected = np.full(2, bool(np.any(mask)))
            np.testing.assert_array_equal(actual, expected)
        output = np.array([False, True])
        np.testing.assert_array_equal(
            probe._all_nonzero_affine_support_backward(snapshot, output),
            np.ones(3, dtype=bool),
        )

    @unittest.skipUnless(probe.torch.cuda.is_available(), "CUDA is required")
    def test_conv_all_nonzero_support_matches_reference_both_directions(self):
        topology = probe._live.get_exact_conv_spatial_topology(
            input_shape=(1, 2, 4, 4),
            output_shape=(1, 4, 2, 2),
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
        )
        snapshot = SimpleNamespace(
            kind="CONV2D",
            input_size=32,
            output_size=16,
            weight=np.ones((4, 2, 3, 3), dtype=np.float64),
            topology=topology,
        )
        source = np.zeros(32, dtype=bool)
        source[[0, 7, 17, 31]] = True
        output = np.zeros(16, dtype=bool)
        output[[0, 6, 15]] = True
        np.testing.assert_array_equal(
            probe._all_nonzero_affine_support_forward(snapshot, source),
            probe._live._affine_support_forward(snapshot, source),
        )
        np.testing.assert_array_equal(
            probe._all_nonzero_affine_support_backward(snapshot, output),
            probe._live._affine_support_backward(snapshot, output),
        )

    @unittest.skipUnless(probe.torch.cuda.is_available(), "CUDA is required")
    def test_gpu_selected_csr_matches_cpu_reference(self):
        topology = probe._live.get_exact_conv_spatial_topology(
            input_shape=(1, 2, 4, 4),
            output_shape=(1, 4, 2, 2),
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
        )
        snapshot = SimpleNamespace(
            kind="CONV2D",
            input_size=32,
            output_size=16,
            weight=np.arange(1, 73, dtype=np.float64).reshape(4, 2, 3, 3),
            topology=topology,
        )
        selected = np.array([0, 3, 6, 9, 15], dtype=np.int64)
        possible = np.ones(32, dtype=bool)
        possible[[1, 8, 17, 30]] = False
        cpu = probe._live._selected_affine_matrix(
            snapshot, selected, possible, name="test.gpu_csr.cpu"
        )
        gpu = probe._gpu_selected_affine_matrix(
            snapshot, selected, possible, name="test.gpu_csr.gpu"
        )
        np.testing.assert_array_equal(
            gpu.indptr.detach().cpu().numpy(), cpu.indptr
        )
        np.testing.assert_array_equal(
            gpu.indices.detach().cpu().numpy(), cpu.indices
        )
        np.testing.assert_array_equal(
            gpu.data.detach().cpu().numpy().view(np.uint64),
            cpu.data.view(np.uint64),
        )

    @unittest.skipUnless(probe.torch.cuda.is_available(), "CUDA is required")
    def test_gpu_selected_csr_matches_grouped_dilated_batched_reference(self):
        topology = probe._live.get_exact_conv_spatial_topology(
            input_shape=(2, 4, 6, 7),
            output_shape=(2, 4, 3, 9),
            kernel=(2, 3),
            stride=(2, 1),
            padding=(1, 2),
            dilation=(2, 1),
            groups=2,
        )
        weight = np.arange(1, 49, dtype=np.float64).reshape(4, 2, 2, 3)
        weight[1::2] *= -1.0
        snapshot = SimpleNamespace(
            kind="CONV2D",
            input_size=2 * 4 * 6 * 7,
            output_size=2 * 4 * 3 * 9,
            weight=weight,
            topology=topology,
        )
        selected = np.array([0, 8, 27, 55, 108, 161, 215], dtype=np.int64)
        possible = np.ones(snapshot.input_size, dtype=bool)
        possible[np.arange(0, snapshot.input_size, 11)] = False
        cpu = probe._live._selected_affine_matrix(
            snapshot, selected, possible, name="test.gpu_grouped.cpu"
        )
        gpu = probe._gpu_selected_affine_matrix(
            snapshot, selected, possible, name="test.gpu_grouped.gpu"
        )
        np.testing.assert_array_equal(gpu.indptr.cpu().numpy(), cpu.indptr)
        np.testing.assert_array_equal(gpu.indices.cpu().numpy(), cpu.indices)
        np.testing.assert_array_equal(
            gpu.data.cpu().numpy().view(np.uint64), cpu.data.view(np.uint64)
        )

    @unittest.skipUnless(probe.torch.cuda.is_available(), "CUDA is required")
    def test_gpu_selected_dense_csr_and_zero_weight_fail_closed(self):
        snapshot = SimpleNamespace(
            kind="DENSE",
            input_size=4,
            output_size=3,
            weight=np.arange(1, 13, dtype=np.float64).reshape(3, 4),
            topology=None,
        )
        selected = np.array([0, 2], dtype=np.int64)
        possible = np.array([True, False, True, True])
        cpu = probe._live._selected_affine_matrix(
            snapshot, selected, possible, name="test.gpu_dense.cpu"
        )
        gpu = probe._gpu_selected_affine_matrix(
            snapshot, selected, possible, name="test.gpu_dense.gpu"
        )
        np.testing.assert_array_equal(gpu.indptr.cpu().numpy(), cpu.indptr)
        np.testing.assert_array_equal(gpu.indices.cpu().numpy(), cpu.indices)
        np.testing.assert_array_equal(
            gpu.data.cpu().numpy().view(np.uint64), cpu.data.view(np.uint64)
        )
        snapshot.weight[0, 0] = 0.0
        with self.assertRaisesRegex(Exception, "all-nonzero"):
            probe._gpu_selected_affine_matrix(
                snapshot, selected, possible, name="test.gpu_dense.zero"
            )


class ScopeTests(unittest.TestCase):
    def test_terminal_proof_calls_and_prohibited_paths_are_explicit(self):
        source = Path(probe.__file__).read_text(encoding="utf-8")
        self.assertIn("_singleton_interval_forward", source)
        self.assertIn("_exact_singleton_margin_lower", source)
        self.assertIn('"input_sampling_used": False', source)
        self.assertIn('"onnx_input_execution_used": False', source)
        self.assertIn('"pgd_used": False', source)
        self.assertIn('"bab_used": False', source)
        self.assertIn('"backward_used": False', source)
        self.assertIn('"dual_tightening_used": False', source)
        self.assertNotIn("import onnxruntime", source)
        self.assertNotIn("np.random", source)
        self.assertNotIn("torch.rand", source)


if __name__ == "__main__":
    unittest.main()
