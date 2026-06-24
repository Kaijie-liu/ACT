# ===- act/back_end/hybridz_tf/__main__.py - HybridZ TF self-tests ------===#
# ACT: Abstract Constraint Transformer
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from .hybridz_tf import _test_sparse_matmul_const_propagation
from .tf_cnn import _test_hz_cnn_exact_maxpool_only
from .tf_mlp import _test_hz_mul_exact_point_and_var_drop


def main() -> None:
    tests = [
        _test_sparse_matmul_const_propagation,
        _test_hz_cnn_exact_maxpool_only,
        _test_hz_mul_exact_point_and_var_drop,
    ]
    passed = 0
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"{passed}/{len(tests)} passed")


if __name__ == "__main__":  # pragma: no cover
    main()
