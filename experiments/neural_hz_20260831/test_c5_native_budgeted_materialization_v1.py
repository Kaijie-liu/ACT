import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr, SparseHZAffineTerm
from experiments.neural_hz_20260831 import c5_ordered_union_contraction_v3 as compiler
from experiments.neural_hz_20260831.c5_native_budgeted_materialization_v1 import BudgetedMaterializer
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


def fixture():
    op = ImplicitConv2DOp(np.ones((2, 2, 1, 1)), (1, 2, 8, 8))
    hz = source(op.shape[1])
    hz.c[:] = 1.
    operators = (op, sp.eye(op.shape[0], format="csr"), op, sp.eye(op.shape[0], format="csr"))
    term = SparseHZAffineTerm(hz, operators)
    return SparseHZAffineExpr((term, term), np.zeros(op.shape[0]), op.shape[0], hz.frame_id), np.arange(op.shape[0])


def test_whole_sequence_counts_both_stages_and_all_branches():
    expr, rows = fixture()
    budget = BudgetedMaterializer(2, whole_cap=32, branch_cap=16)
    original = compiler.contract
    budget.run(expr, rows)
    budget.run(expr, rows)
    assert budget.remaining_whole == 0 and budget.remaining_branches == [0, 0]
    assert len(budget.stage_stats) == 2 and compiler.contract is original


@pytest.mark.parametrize("whole,branch", [(24, 16), (32, 8)])
def test_second_stage_cannot_reset_frozen_budget(whole, branch):
    expr, rows = fixture()
    budget = BudgetedMaterializer(2, whole_cap=whole, branch_cap=branch)
    original = compiler.contract
    budget.run(expr, rows)
    with pytest.raises(MemoryError):
        budget.run(expr, rows)
    assert budget.failed and compiler.contract is original
    with pytest.raises(ValueError, match="failed"):
        budget.run(expr, rows)
