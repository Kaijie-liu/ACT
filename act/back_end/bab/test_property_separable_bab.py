"""Controlled gates for property-separable (one conjunct per tree) BaB."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from act.back_end.bab import bab as bab_module
from act.back_end.bab.bab import (
    _check_input_specs_batched,
    _dispatch_dual_solve,
    _expand_property_forest_root,
    _split_from_decision,
    _select_spec_rows,
    _strictly_certified_slack,
    _validate_property_forest_child_partition,
    _validate_property_forest_receipt,
    verify_bab_batched,
)
from act.back_end.bab.branching.bounding import TopKBounding
from act.back_end.bab.branching.branching import SplitDecision
from act.back_end.bab.property_forest_authority import (
    new_property_forest_run_token,
    source_file_digests,
    validate_bab_safe_capability,
)
from act.back_end.bab.node import (
    SubproblemBatch,
    _infer_spec_axis_size,
    split_input,
)
from act.back_end.config import BaBConfig
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import tf_forward
from act.back_end.solver.solver_base import SolveStatus
from act.back_end.solver.solver_dual import DualSolver
from act.front_end.specs import InKind, OutKind, OutputSpec
from act.util.device_manager import initialize_device
from act.util.stats import VerifyResult, VerifyStatus


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _three_row_affine_net(*, unsafe: bool = False) -> Net:
    kind = OutKind.UNSAFE_LINEAR if unsafe else OutKind.LINEAR_LE
    assertion = OutputSpec(
        kind=kind,
        c=torch.eye(3, dtype=DTYPE),
        d=torch.tensor([0.5, 0.25, 0.75], dtype=DTYPE),
    ).encode_linear(B=1, n_out=3, device=DEVICE, dtype=DTYPE)
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 3), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=[0, 1, 2],
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": -torch.ones(1, 3, dtype=DTYPE),
                "ub": torch.ones(1, 3, dtype=DTYPE),
            },
            in_vars=[0, 1, 2],
            out_vars=[0, 1, 2],
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.eye(3, dtype=DTYPE),
                "bias": torch.zeros(3, dtype=DTYPE),
                "in_features": 3,
                "out_features": 3,
            },
            in_vars=[0, 1, 2],
            out_vars=[3, 4, 5],
        ),
        Layer(
            id=3,
            kind="ASSERT",
            params=assertion,
            in_vars=[3, 4, 5],
            out_vars=[3, 4, 5],
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2]},
        succs={0: [1], 1: [2], 2: [3], 3: []},
    )


def _two_axis_duplicate_relu_net(
    *,
    hard_threshold: float = 0.1,
) -> Net:
    """Safe graph whose two hard conjuncts need different input splits.

    Rows 2 and 5 are respectively

    ``ReLU(x0) - ReLU(x0) <= hard_threshold`` and
    ``ReLU(x1) - ReLU(x1) <= hard_threshold``.

    The exact graph value is zero.  On the root ``[-1, 1]^2`` box, however,
    the two duplicated relaxed ReLUs are independent and each dual upper is
    ``1/2``.  Splitting x0 at zero makes row 2 exact; splitting x1 makes row 5
    exact.  The other four rows are constant zero with threshold 1/2 and are
    strictly certified by root presolve.
    """

    assertion = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.eye(6, dtype=DTYPE),
        d=torch.tensor(
            [
                0.5,
                0.5,
                hard_threshold,
                0.5,
                0.5,
                hard_threshold,
            ],
            dtype=DTYPE,
        ),
    ).encode_linear(B=1, n_out=6, device=DEVICE, dtype=DTYPE)
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 2), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=[0, 1],
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": -torch.ones(1, 2, dtype=DTYPE),
                "ub": torch.ones(1, 2, dtype=DTYPE),
            },
            in_vars=[0, 1],
            out_vars=[0, 1],
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ],
                    dtype=DTYPE,
                ),
                "bias": torch.zeros(4, dtype=DTYPE),
                "in_features": 2,
                "out_features": 4,
            },
            in_vars=[0, 1],
            out_vars=[2, 3, 4, 5],
        ),
        Layer(
            id=3,
            kind="RELU",
            params={},
            in_vars=[2, 3, 4, 5],
            out_vars=[6, 7, 8, 9],
        ),
        Layer(
            id=4,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -1.0],
                    ],
                    dtype=DTYPE,
                ),
                "bias": torch.zeros(6, dtype=DTYPE),
                "in_features": 4,
                "out_features": 6,
            },
            in_vars=[6, 7, 8, 9],
            out_vars=[10, 11, 12, 13, 14, 15],
        ),
        Layer(
            id=5,
            kind="ASSERT",
            params=assertion,
            in_vars=[10, 11, 12, 13, 14, 15],
            out_vars=[10, 11, 12, 13, 14, 15],
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _two_axis_duplicate_relu_residual_net(
    *,
    hard_threshold: float = 0.1,
) -> Net:
    """Fanout/ADD form of the two-row SAFE toy used by the C50 gate.

    The shared duplicate preactivation ``[x0, x0, x1, x1]`` fans out through
    an identity branch and an exact-zero branch before being recombined by
    ADD.  Thus the mathematical graph and the two hard ASSERT rows are
    unchanged, while the shared stem must remain live through two distinct
    consumer nodes and can be reclaimed only after the second one.
    """

    chain = _two_axis_duplicate_relu_net(
        hard_threshold=hard_threshold
    )
    output_params = dict(chain.layers[4].params)
    assert_params = dict(chain.layers[5].params)
    layers = list(chain.layers[:3])
    layers.extend(
        [
            Layer(
                id=3,
                kind="DENSE",
                params={
                    "weight": torch.eye(4, dtype=DTYPE),
                    "bias": torch.zeros(4, dtype=DTYPE),
                    "in_features": 4,
                    "out_features": 4,
                },
                in_vars=[2, 3, 4, 5],
                out_vars=[6, 7, 8, 9],
            ),
            Layer(
                id=4,
                kind="DENSE",
                params={
                    "weight": torch.zeros((4, 4), dtype=DTYPE),
                    "bias": torch.zeros(4, dtype=DTYPE),
                    "in_features": 4,
                    "out_features": 4,
                },
                in_vars=[2, 3, 4, 5],
                out_vars=[10, 11, 12, 13],
            ),
            Layer(
                id=5,
                kind="ADD",
                params={
                    "x_vars": [6, 7, 8, 9],
                    "y_vars": [10, 11, 12, 13],
                },
                in_vars=[6, 7, 8, 9, 10, 11, 12, 13],
                out_vars=[14, 15, 16, 17],
            ),
            Layer(
                id=6,
                kind="RELU",
                params={},
                in_vars=[14, 15, 16, 17],
                out_vars=[18, 19, 20, 21],
            ),
            Layer(
                id=7,
                kind="DENSE",
                params=output_params,
                in_vars=[18, 19, 20, 21],
                out_vars=[22, 23, 24, 25, 26, 27],
            ),
            Layer(
                id=8,
                kind="ASSERT",
                params=assert_params,
                in_vars=[22, 23, 24, 25, 26, 27],
                out_vars=[22, 23, 24, 25, 26, 27],
            ),
        ]
    )
    return Net(
        layers=layers,
        preds={
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [2],
            5: [3, 4],
            6: [5],
            7: [6],
            8: [7],
        },
        succs={
            0: [1],
            1: [2],
            2: [3, 4],
            3: [5],
            4: [5],
            5: [6],
            6: [7],
            7: [8],
            8: [],
        },
    )


class _PropertyRowInputBrancher:
    """Test-only deterministic brancher keyed by immutable ASSERT-row id."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int], list[int]]] = []

    def compute_scores(
        self,
        batch: SubproblemBatch,
        _net: Net,
    ) -> torch.Tensor:
        if batch.spec_row_ids is None:
            raise AssertionError("property brancher received an untagged lane")
        scores = torch.full_like(batch.lb, -torch.inf)
        split_dims: list[int] = []
        for lane, raw_row in enumerate(batch.spec_row_ids.tolist()):
            row = int(raw_row)
            if row == 2:
                split_dim = 0
            elif row == 5:
                split_dim = 1
            else:
                raise AssertionError(f"unexpected property row {row}")
            scores[lane, split_dim] = 1.0
            split_dims.append(split_dim)
        self.calls.append(
            (
                [int(value) for value in batch.spec_row_ids.tolist()],
                split_dims,
                [int(value) for value in batch.depths.tolist()],
            )
        )
        return scores

    @staticmethod
    def select(scores: torch.Tensor) -> torch.Tensor:
        return scores.argmax(dim=-1)


def _exact_duplicate_relu_violation_check(
    _net: Net,
    x_batch: torch.Tensor,
    assert_layer: Layer,
) -> torch.Tensor:
    """Independent concrete oracle for the duplicate-ReLU E2E toy.

    ACTToTorch currently reconstructs a full ``VerifiableModel`` wrapper for
    concrete replay.  Its multi-row LINEAR_LE OutputSpec layer has an unrelated
    shape mismatch, so this focused BaB test supplies the exact graph output
    (all six coordinates are zero) without weakening the abstract solves.
    """

    flat = x_batch.reshape(x_batch.shape[0], -1)
    if flat.shape[1] != 2 or not bool(torch.isfinite(flat).all().item()):
        raise AssertionError("malformed concrete toy input")
    outputs = torch.zeros(
        (flat.shape[0], 6), device=flat.device, dtype=flat.dtype
    )
    C = torch.as_tensor(
        assert_layer.params["C"], device=flat.device, dtype=flat.dtype
    ).reshape(-1, 6)
    thresholds = torch.as_tensor(
        assert_layer.params["thresholds"],
        device=flat.device,
        dtype=flat.dtype,
    ).reshape(-1)
    values = outputs @ C.transpose(0, 1)
    return (values > thresholds.unsqueeze(0) + 1.0e-8).any(dim=1)


def _select_forest_lanes(
    batch: SubproblemBatch,
    indices: torch.Tensor,
) -> SubproblemBatch:
    """Test-only lane selector used to simulate a deleted proof tree."""

    def select_state(state):
        if state is None:
            return None
        return {
            layer_id: tensor.index_select(0, indices.to(tensor.device))
            for layer_id, tensor in state.items()
        }

    def select_optional(value):
        return (
            None
            if value is None
            else value.index_select(0, indices.to(value.device))
        )

    return SubproblemBatch(
        lb=batch.lb.index_select(0, indices.to(batch.lb.device)),
        ub=batch.ub.index_select(0, indices.to(batch.ub.device)),
        depths=batch.depths.index_select(0, indices.to(batch.depths.device)),
        incremental_alpha=select_state(batch.incremental_alpha),
        incremental_eta=select_state(batch.incremental_eta),
        split_signs=select_state(batch.split_signs),
        parent_margins=select_optional(batch.parent_margins),
        lower_bound=select_optional(batch.lower_bound),
        node_id=select_optional(batch.node_id),
        parent_id=select_optional(batch.parent_id),
        spec_row_ids=select_optional(batch.spec_row_ids),
    )


def _run_two_axis_property_forest(
    *,
    hard_threshold: float = 0.1,
    max_depth: int = 1,
    frontier_cap: int = 0,
    net_factory=_two_axis_duplicate_relu_net,
    run_token: str | None = None,
    source_digests: dict[str, str] | None = None,
    return_context: bool = False,
):
    """Run the complete production BaB loop with test-only deterministic IO."""

    net = net_factory(
        hard_threshold=hard_threshold
    )
    config = BaBConfig(
        solver_tier="dual",
        branching_method="width",
        bounding_method="topk",
        property_separable_bab=True,
        provenance_enabled=True,
        max_depth=int(max_depth),
        max_nodes=100,
        frontier_cap=int(frontier_cap),
    )
    brancher = _PropertyRowInputBrancher()
    k_log: list[int] = []
    solver_factory = lambda: DualSolver()
    with (
        mock.patch(
            "act.back_end.bab.bab._build_branching_strategy",
            return_value=brancher,
        ),
        mock.patch(
            "act.back_end.bab.bab.check_violations_batched",
            side_effect=_exact_duplicate_relu_violation_check,
        ),
    ):
        result = verify_bab_batched(
            net,
            solver_factory,
            config,
            max_batch_size=8,
            time_budget_s=10.0,
            _k_log=k_log,
            _property_forest_run_token=run_token,
            _property_forest_source_digests=source_digests,
        )
    if return_context:
        return (
            result,
            brancher,
            k_log,
            net,
            config,
            solver_factory,
        )
    return result, brancher, k_log


def _run_c50_release_gate(*, retain_all: bool):
    """Run one side of the C50 lifecycle equivalence gate.

    ``retain_all`` changes only the internal forward-state release hook.  The
    verifier, solver, property forest, brancher, and concrete replay paths are
    otherwise identical.
    """

    real_dispatch = bab_module._dispatch_dual_solve
    dual_trace: list[dict[str, object]] = []

    def traced_dispatch(*args, **kwargs):
        result = real_dispatch(*args, **kwargs)
        batch = kwargs["batch"]
        dual_trace.append(
            {
                "spec_row_ids": (
                    None
                    if batch.spec_row_ids is None
                    else tuple(
                        int(value)
                        for value in batch.spec_row_ids.tolist()
                    )
                ),
                "statuses": tuple(result.solution.statuses),
                "row_slack": (
                    None
                    if result.row_slack is None
                    else result.row_slack.detach().cpu().contiguous().clone()
                ),
                "row_certified": (
                    None
                    if result.row_certified is None
                    else result.row_certified.detach()
                    .cpu()
                    .contiguous()
                    .clone()
                ),
            }
        )
        return result

    release_events: list[
        tuple[int, tuple[int, ...], frozenset[int]]
    ] = []
    real_release = tf_forward._release_consumed_forward_state

    def tracked_release(
        layer_id,
        preds,
        remaining_consumers,
        box_state,
        lin_state,
        frame_dict,
    ):
        before = set(box_state)
        real_release(
            layer_id,
            preds,
            remaining_consumers,
            box_state,
            lin_state,
            frame_dict,
        )
        release_events.append(
            (
                int(layer_id),
                tuple(int(value) for value in preds),
                frozenset(before - set(box_state)),
            )
        )

    release_patch = (
        mock.patch.object(
            tf_forward,
            "_release_consumed_forward_state",
            return_value=None,
        )
        if retain_all
        else mock.patch.object(
            tf_forward,
            "_release_consumed_forward_state",
            side_effect=tracked_release,
        )
    )
    with (
        mock.patch.object(
            bab_module,
            "_dispatch_dual_solve",
            side_effect=traced_dispatch,
        ),
        release_patch,
    ):
        result, brancher, k_log = _run_two_axis_property_forest(
            net_factory=_two_axis_duplicate_relu_residual_net,
        )
    return result, brancher, k_log, dual_trace, release_events


def _clone_tensor_state(state):
    return (
        None
        if state is None
        else {
            layer_id: tensor.clone()
            for layer_id, tensor in state.items()
        }
    )


def _run_dual_alpha_eta_post_prune_pair(
    *,
    hard_threshold: float,
    nan_row_five: bool,
    split_neuron: int,
):
    """Compare optimized joint rows with their M=1 property-forest lanes."""

    net = _two_axis_duplicate_relu_net(
        hard_threshold=hard_threshold
    )
    assert_layer = net.by_id[5]
    if nan_row_five:
        for field in ("d", "thresholds"):
            updated = assert_layer.params[field].clone()
            updated[..., 5] = float("nan")
            assert_layer.params[field] = updated

    bounds = Bounds(
        -torch.ones(1, 2, dtype=DTYPE),
        torch.ones(1, 2, dtype=DTYPE),
    )
    config = BaBConfig(
        solver_tier="dual_alpha_eta",
        branching_method="random",
        dual_n_iters=8,
        lr_alpha=0.1,
        lr_beta=0.1,
        lr_decay=1.0,
        per_class_alpha=True,
    )

    # This is the same pre-prune representation used by the production root
    # presolve: one input box and all six ASSERT rows on the M axis.
    full_root_batch = SubproblemBatch.from_bounds(bounds)
    full_root = _dispatch_dual_solve(
        net=net,
        assert_layer=assert_layer,
        batched_bounds=bounds,
        k_actual=1,
        batch=full_root_batch,
        config=config,
        optimize=True,
    )
    keep_rows = torch.where(
        (~full_root.row_certified).any(dim=0)
    )[0]
    warm_alpha = _select_spec_rows(
        full_root_batch.incremental_alpha,
        keep_rows,
    )
    if warm_alpha is None:
        raise AssertionError(
            "dual_alpha_eta root did not return incremental alpha state"
        )

    # Add one child phase after root row pruning.  It is deliberately shared
    # by both property rows, matching the property-forest phase invariant,
    # while still forcing dual_alpha_eta to materialize eta state.
    phase = torch.zeros(
        (1, int(keep_rows.numel()), 4), dtype=DTYPE
    )
    phase[:, :, split_neuron] = 1.0
    warm_eta = {3: torch.zeros_like(phase)}
    split_signs = {3: phase}

    def pruned_root() -> SubproblemBatch:
        return SubproblemBatch(
            lb=bounds.lb.clone(),
            ub=bounds.ub.clone(),
            depths=torch.ones(1, dtype=torch.long),
            incremental_alpha=_clone_tensor_state(warm_alpha),
            incremental_eta=_clone_tensor_state(warm_eta),
            split_signs=_clone_tensor_state(split_signs),
        )

    # Joint form keeps the two non-contiguous original rows on M.  Forest form
    # transposes exactly that state to two immutable B lanes with M=1.
    joint_batch = pruned_root()
    forest_batch = _expand_property_forest_root(
        pruned_root(),
        keep_rows,
    )
    forest_bounds = Bounds(
        bounds.lb.repeat(int(keep_rows.numel()), 1),
        bounds.ub.repeat(int(keep_rows.numel()), 1),
    )
    joint = _dispatch_dual_solve(
        net=net,
        assert_layer=assert_layer,
        batched_bounds=bounds,
        k_actual=1,
        batch=joint_batch,
        config=config,
        optimize=True,
        keep_rows=keep_rows,
    )
    forest = _dispatch_dual_solve(
        net=net,
        assert_layer=assert_layer,
        batched_bounds=forest_bounds,
        k_actual=int(keep_rows.numel()),
        batch=forest_batch,
        config=config,
        optimize=True,
    )
    return {
        "keep_rows": keep_rows,
        "full_root": full_root,
        "warm_alpha": warm_alpha,
        "joint_batch": joint_batch,
        "forest_batch": forest_batch,
        "joint": joint,
        "forest": forest,
    }


def _assert_joint_forest_state_bitwise_equal(
    testcase: unittest.TestCase,
    joint_state,
    forest_state,
    *,
    label: str,
) -> None:
    testcase.assertIsNotNone(joint_state)
    testcase.assertIsNotNone(forest_state)
    testcase.assertEqual(set(joint_state), set(forest_state))
    for layer_id in sorted(joint_state):
        expected = joint_state[layer_id].transpose(0, 1).contiguous()
        actual = forest_state[layer_id].contiguous()
        testcase.assertEqual(expected.dtype, DTYPE)
        testcase.assertEqual(actual.dtype, DTYPE)
        testcase.assertEqual(expected.shape, actual.shape)
        testcase.assertTrue(
            torch.equal(
                expected.view(torch.int64),
                actual.view(torch.int64),
            ),
            f"{label} layer {layer_id} changed at the bit level",
        )


class PropertySeparableBaBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_device("cpu", "float64")

    def test_verify_batched_safe_forest_uses_distinct_row_splits(self):
        result, brancher, k_log = _run_two_axis_property_forest()
        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        metadata = result.metadata
        self.assertEqual(metadata["spec_rows_kept"], 2)
        self.assertEqual(
            metadata["property_forest_root_rows"], [2, 5]
        )
        self.assertEqual(metadata["property_forest_root_count"], 2)
        self.assertTrue(
            metadata["property_forest_all_solves_single_row"]
        )
        self.assertTrue(
            metadata["property_forest_coverage_complete"]
        )
        self.assertEqual(
            metadata["property_forest_coverage_by_row"],
            {
                "2": {
                    "processed": 3,
                    "certified_nodes": 2,
                    "covered": True,
                },
                "5": {
                    "processed": 3,
                    "certified_nodes": 2,
                    "covered": True,
                },
            },
        )
        receipt = metadata[
            "property_forest_node_conservation_receipt"
        ]
        self.assertTrue(
            metadata["property_forest_node_conservation_valid"]
        )
        self.assertEqual(
            metadata["property_forest_node_conservation_errors"], []
        )
        self.assertFalse(receipt["proof_authority"])
        self.assertTrue(receipt["complete"])
        self.assertEqual(
            receipt["rows"],
            {
                "2": {
                    "roots": 1,
                    "children_expected": 2,
                    "children_minted": 2,
                    "processed": 3,
                    "certified": 2,
                    "branched": 1,
                    "active_pool": 0,
                    "dropped": {
                        "frontier_cap": 0,
                        "max_depth": 0,
                    },
                    "terminal_reasons": {
                        "certified": 2,
                        "dropped_max_depth": 0,
                        "dropped_frontier_cap": 0,
                        "active_pool": 0,
                    },
                    "integrity_errors": [],
                },
                "5": {
                    "roots": 1,
                    "children_expected": 2,
                    "children_minted": 2,
                    "processed": 3,
                    "certified": 2,
                    "branched": 1,
                    "active_pool": 0,
                    "dropped": {
                        "frontier_cap": 0,
                        "max_depth": 0,
                    },
                    "terminal_reasons": {
                        "certified": 2,
                        "dropped_max_depth": 0,
                        "dropped_frontier_cap": 0,
                        "active_pool": 0,
                    },
                    "integrity_errors": [],
                },
            },
        )
        self.assertEqual(
            receipt["totals"],
            {
                "roots": 2,
                "children_expected": 4,
                "children_minted": 4,
                "processed": 6,
                "certified": 4,
                "branched": 2,
                "active_pool": 0,
                "dropped": {
                    "frontier_cap": 0,
                    "max_depth": 0,
                },
            },
        )
        self.assertEqual(metadata["pool_remaining"], 0)
        self.assertFalse(metadata["any_dropped_frontier_cap"])
        self.assertFalse(metadata["any_dropped_max_depth"])
        # Both non-contiguous original row ids are branched in one root wave,
        # but row 2 selects x0 while row 5 independently selects x1.
        self.assertEqual(
            brancher.calls,
            [([2, 5], [0, 1], [0, 0])],
        )
        self.assertEqual(k_log, [2, 4])

    def test_c50_last_use_release_matches_retain_all_safe_forest_bitwise(
        self,
    ):
        reclaimed = _run_c50_release_gate(retain_all=False)
        retained = _run_c50_release_gate(retain_all=True)
        (
            reclaimed_result,
            reclaimed_brancher,
            reclaimed_k_log,
            reclaimed_trace,
            release_events,
        ) = reclaimed
        (
            retained_result,
            retained_brancher,
            retained_k_log,
            retained_trace,
            retained_release_events,
        ) = retained

        self.assertEqual(reclaimed_result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(retained_result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(
            reclaimed_result.metadata["property_forest_root_rows"],
            [2, 5],
        )
        self.assertEqual(
            retained_result.metadata["property_forest_root_rows"],
            [2, 5],
        )
        self.assertEqual(
            reclaimed_result.metadata[
                "property_forest_coverage_by_row"
            ],
            retained_result.metadata[
                "property_forest_coverage_by_row"
            ],
        )
        self.assertEqual(
            reclaimed_result.metadata[
                "property_forest_coverage_by_row"
            ],
            {
                "2": {
                    "processed": 3,
                    "certified_nodes": 2,
                    "covered": True,
                },
                "5": {
                    "processed": 3,
                    "certified_nodes": 2,
                    "covered": True,
                },
            },
        )
        self.assertEqual(
            reclaimed_brancher.calls,
            retained_brancher.calls,
        )
        self.assertEqual(
            reclaimed_brancher.calls,
            [([2, 5], [0, 1], [0, 0])],
        )
        self.assertEqual(reclaimed_k_log, retained_k_log)
        self.assertEqual(reclaimed_k_log, [2, 4])
        self.assertEqual(len(reclaimed_trace), len(retained_trace))
        self.assertGreater(len(reclaimed_trace), 0)

        for call_index, (released_call, retained_call) in enumerate(
            zip(reclaimed_trace, retained_trace)
        ):
            self.assertEqual(
                released_call["spec_row_ids"],
                retained_call["spec_row_ids"],
            )
            self.assertEqual(
                released_call["statuses"],
                retained_call["statuses"],
            )
            self.assertTrue(
                torch.equal(
                    released_call["row_certified"],
                    retained_call["row_certified"],
                ),
                f"dual call {call_index} changed its row mask",
            )
            released_slack = released_call["row_slack"]
            retained_slack = retained_call["row_slack"]
            self.assertEqual(released_slack.dtype, DTYPE)
            self.assertEqual(retained_slack.dtype, DTYPE)
            self.assertEqual(
                released_slack.shape,
                retained_slack.shape,
            )
            self.assertTrue(
                torch.equal(
                    released_slack.view(torch.int64),
                    retained_slack.view(torch.int64),
                ),
                f"dual call {call_index} changed row slack bits",
            )

        root_mask = reclaimed_trace[0]["row_certified"][0]
        self.assertEqual(
            torch.where(~root_mask)[0].tolist(),
            [2, 5],
        )
        self.assertEqual(retained_release_events, [])
        # Layer 2 is the shared fanout stem.  It must not be released after
        # consumer 3, and is released exactly after consumer 4 completes.
        stem_release_events = [
            (layer_id, preds)
            for layer_id, preds, released_ids in release_events
            if 2 in released_ids
        ]
        self.assertGreater(len(stem_release_events), 0)
        self.assertTrue(
            all(event == (4, (2,)) for event in stem_release_events)
        )

    def test_deleted_property_tree_fails_closed_at_terminal_pool(self):
        real_expand = _expand_property_forest_root

        def delete_second_tree(root, row_ids):
            forest = real_expand(root, row_ids)
            return _select_forest_lanes(
                forest,
                torch.tensor([0], device=forest.lb.device),
            )

        with mock.patch(
            "act.back_end.bab.bab._expand_property_forest_root",
            side_effect=delete_second_tree,
        ):
            result, brancher, _ = _run_two_axis_property_forest()

        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(result.metadata["pool_remaining"], 0)
        self.assertFalse(
            result.metadata["any_dropped_frontier_cap"]
        )
        self.assertFalse(result.metadata["any_dropped_max_depth"])
        self.assertEqual(
            result.metadata["reason"],
            "property_forest_incomplete_coverage",
        )
        self.assertFalse(
            result.metadata["property_forest_coverage_complete"]
        )
        self.assertEqual(
            result.metadata["property_forest_coverage_by_row"]["5"],
            {
                "processed": 0,
                "certified_nodes": 0,
                "covered": False,
            },
        )
        self.assertEqual(brancher.calls, [([2], [0], [0])])

    def test_duplicate_root_and_deleted_or_duplicate_children_fail_closed(
        self,
    ):
        real_expand = _expand_property_forest_root

        def duplicate_first_root(root, row_ids):
            forest = real_expand(root, row_ids)
            return _select_forest_lanes(
                forest,
                torch.tensor([0, 1, 0], device=forest.lb.device),
            )

        with mock.patch(
            "act.back_end.bab.bab._expand_property_forest_root",
            side_effect=duplicate_first_root,
        ):
            duplicate_root, _, _ = _run_two_axis_property_forest()
        self.assertEqual(duplicate_root.status, VerifyStatus.UNKNOWN)
        self.assertEqual(
            duplicate_root.metadata["reason"],
            "property_forest_incomplete_coverage",
        )
        root_receipt = duplicate_root.metadata[
            "property_forest_node_conservation_receipt"
        ]
        self.assertFalse(
            duplicate_root.metadata[
                "property_forest_node_conservation_valid"
            ]
        )
        self.assertEqual(root_receipt["rows"]["2"]["roots"], 2)
        self.assertFalse(root_receipt["complete"])

        real_split = split_input

        def corrupt_children(mode):
            def split(batch, split_dims):
                children, parent_index = real_split(batch, split_dims)
                n = batch.batch_size
                if mode == "delete":
                    indices = torch.arange(
                        n, device=children.lb.device
                    )
                elif mode == "duplicate":
                    indices = torch.cat(
                        [
                            torch.arange(
                                children.batch_size,
                                device=children.lb.device,
                            ),
                            torch.arange(n, device=children.lb.device),
                        ]
                    )
                else:
                    # Keep the advertised child count while replacing every
                    # right half by a duplicate of its left sibling.
                    indices = torch.arange(
                        n, device=children.lb.device
                    ).repeat(2)
                return (
                    _select_forest_lanes(children, indices),
                    parent_index.index_select(
                        0, indices.to(parent_index.device)
                    ),
                )

            return split

        for mode in ("delete", "duplicate", "replace"):
            with self.subTest(corruption=mode), mock.patch(
                "act.back_end.bab.bab.split_input",
                side_effect=corrupt_children(mode),
            ):
                result, _, _ = _run_two_axis_property_forest()
            self.assertEqual(result.status, VerifyStatus.UNKNOWN)
            self.assertEqual(
                result.metadata["reason"],
                "property_forest_incomplete_coverage",
            )
            self.assertFalse(
                result.metadata[
                    "property_forest_node_conservation_valid"
                ]
            )
            receipt = result.metadata[
                "property_forest_node_conservation_receipt"
            ]
            self.assertFalse(receipt["complete"])
            for row_id in ("2", "5"):
                row = receipt["rows"][row_id]
                self.assertEqual(row["children_expected"], 2)
                # Malformed children are audited but never enter the pool.
                self.assertEqual(row["children_minted"], 0)
                self.assertTrue(row["integrity_errors"])

    def test_independent_receipt_validator_rejects_tampering(self):
        result, _, _ = _run_two_axis_property_forest()
        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        original = result.metadata[
            "property_forest_node_conservation_receipt"
        ]
        valid, errors = _validate_property_forest_receipt(
            original,
            expected_row_ids=(2, 5),
            expected_processed=6,
            expected_pool_remaining=0,
        )
        self.assertTrue(valid)
        self.assertEqual(errors, ())

        def mutate_processed(receipt):
            receipt["rows"]["2"]["processed"] -= 1

        def mutate_duplicate_root(receipt):
            receipt["rows"]["2"]["roots"] = 2

        def mutate_row_ids(receipt):
            receipt["root_rows"] = [5, 2]

        def mutate_nonfinite(receipt):
            receipt["rows"]["2"]["certified"] = float("nan")

        def mutate_bool_counter(receipt):
            receipt["rows"]["2"]["branched"] = True

        def mutate_terminal(receipt):
            receipt["rows"]["2"]["terminal_reasons"][
                "certified"
            ] = 3

        def mutate_authority(receipt):
            receipt["proof_authority"] = True

        def mutate_omission(receipt):
            del receipt["rows"]["5"]

        for label, mutation in (
            ("processed", mutate_processed),
            ("duplicate_root", mutate_duplicate_root),
            ("row_ids", mutate_row_ids),
            ("nonfinite", mutate_nonfinite),
            ("bool_counter", mutate_bool_counter),
            ("terminal", mutate_terminal),
            ("authority", mutate_authority),
            ("omission", mutate_omission),
        ):
            with self.subTest(tamper=label):
                tampered = copy.deepcopy(original)
                mutation(tampered)
                valid, errors = _validate_property_forest_receipt(
                    tampered,
                    expected_row_ids=(2, 5),
                    expected_processed=6,
                    expected_pool_remaining=0,
                )
                self.assertFalse(valid)
                self.assertTrue(errors)

    def test_live_safe_capability_binds_complete_forest_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "toy.onnx"
            spec_path = root / "toy.vnnlib"
            model_path.write_bytes(b"exact-two-axis-model-v1")
            spec_path.write_bytes(b"exact-two-axis-property-v1")
            sources = {"onnx": model_path, "vnnlib": spec_path}
            before = source_file_digests(sources)
            token = new_property_forest_run_token()
            (
                result,
                _,
                _,
                net,
                config,
                solver_factory,
            ) = _run_two_axis_property_forest(
                run_token=token,
                source_digests=before,
                return_context=True,
            )
            safe, errors = validate_bab_safe_capability(
                result,
                net=net,
                solver_factory=solver_factory,
                config=config,
                max_batch_size=8,
                time_budget_s=10.0,
                expected_dtype="float64",
                expected_device="cpu",
                run_token=token,
                source_paths=sources,
                source_digests_before_run=before,
            )
            self.assertEqual(errors, ())
            self.assertIsNotNone(safe)
            self.assertTrue(safe["proof_authority"])
            self.assertEqual(
                safe["authority_scope"],
                "this_live_trusted_run_only",
            )
            self.assertFalse(safe["portable_signature"])
            self.assertFalse(
                safe["serialized_receipt_reauthorizes"]
            )
            self.assertEqual(
                safe["live_facts"]["root_certified_rows"],
                [0, 1, 3, 4],
            )
            self.assertEqual(
                safe["live_facts"]["forest_rows"], [2, 5]
            )
            self.assertEqual(
                safe["live_facts"]["processed_nodes"], 6
            )
            # The verifier-held capability is consumed and removed; neither
            # the saved seal nor a second call can re-authorize the result.
            replay, replay_errors = validate_bab_safe_capability(
                result,
                net=net,
                solver_factory=solver_factory,
                config=config,
                max_batch_size=8,
                time_budget_s=10.0,
                expected_dtype="float64",
                expected_device="cpu",
                run_token=token,
                source_paths=sources,
                source_digests_before_run=before,
            )
            self.assertIsNone(replay)
            self.assertIn(
                "missing_stale_or_forged_live_capability",
                replay_errors,
            )

    def test_live_safe_capability_rejects_forged_stale_and_tampered_state(
        self,
    ):
        import act.back_end.bab.property_forest_authority as authority

        self.assertFalse(
            hasattr(authority, "seal_property_forest_live_result")
        )
        with self.assertRaisesRegex(TypeError, "verifier-issued"):
            authority._LiveCapability(object())

        mutations = (
            "forged_status",
            "forged_counterexample",
            "copied_result",
            "missing_capability",
            "node_receipt",
            "live_facts",
            "stale_network",
            "stale_config",
            "stale_source",
            "wrong_source_attribution",
            "stale_token",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model_path = root / "toy.onnx"
                spec_path = root / "toy.vnnlib"
                model_path.write_bytes(b"model-before")
                spec_path.write_bytes(b"spec-before")
                sources = {
                    "onnx": model_path,
                    "vnnlib": spec_path,
                }
                before = source_file_digests(sources)
                token = new_property_forest_run_token()
                (
                    result,
                    _,
                    _,
                    net,
                    config,
                    solver_factory,
                ) = _run_two_axis_property_forest(
                    run_token=token,
                    source_digests=before,
                    return_context=True,
                )
                expected_token = token
                validation_result = result
                validation_sources = sources
                validation_before = before
                if mutation == "forged_status":
                    result.status = VerifyStatus.UNKNOWN
                elif mutation == "forged_counterexample":
                    result.counterexample = torch.zeros(
                        2, dtype=DTYPE
                    )
                elif mutation == "copied_result":
                    validation_result = VerifyResult(
                        VerifyStatus.CERTIFIED,
                        metadata=dict(result.metadata),
                    )
                elif mutation == "missing_capability":
                    result.metadata.pop(
                        "_property_forest_live_capability"
                    )
                elif mutation == "node_receipt":
                    result.metadata[
                        "property_forest_node_conservation_receipt"
                    ]["rows"]["2"]["processed"] -= 1
                elif mutation == "live_facts":
                    result.metadata[
                        "property_forest_live_facts"
                    ]["forest_rows"] = [5, 2]
                elif mutation == "stale_network":
                    net.by_id[2].params["weight"][0, 0] += 0.25
                elif mutation == "stale_config":
                    config.max_nodes += 1
                elif mutation == "stale_source":
                    spec_path.write_bytes(b"spec-after")
                elif mutation == "wrong_source_attribution":
                    wrong_model = root / "wrong.onnx"
                    wrong_spec = root / "wrong.vnnlib"
                    wrong_model.write_bytes(b"unrelated-model")
                    wrong_spec.write_bytes(b"unrelated-spec")
                    validation_sources = {
                        "onnx": wrong_model,
                        "vnnlib": wrong_spec,
                    }
                    validation_before = source_file_digests(
                        validation_sources
                    )
                elif mutation == "stale_token":
                    expected_token = new_property_forest_run_token()
                safe, errors = validate_bab_safe_capability(
                    validation_result,
                    net=net,
                    solver_factory=solver_factory,
                    config=config,
                    max_batch_size=8,
                    time_budget_s=10.0,
                    expected_dtype="float64",
                    expected_device="cpu",
                    run_token=expected_token,
                    source_paths=validation_sources,
                    source_digests_before_run=validation_before,
                )
                self.assertIsNone(safe)
                self.assertTrue(errors)

    def test_live_safe_capability_issues_for_root_presolve_only(self):
        net = _three_row_affine_net()
        assertion = net.by_id[3]
        assertion.params["d"] = torch.full(
            (1, 3), 2.0, dtype=DTYPE
        )
        assertion.params["thresholds"] = torch.full(
            (1, 3), 2.0, dtype=DTYPE
        )
        config = BaBConfig(
            solver_tier="dual",
            branching_method="width",
            bounding_method="topk",
            property_separable_bab=True,
            max_depth=1,
            max_nodes=100,
            frontier_cap=0,
        )
        solver_factory = lambda: DualSolver()
        token = new_property_forest_run_token()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "root.onnx"
            spec_path = root / "root.vnnlib"
            model_path.write_bytes(b"root-model")
            spec_path.write_bytes(b"root-spec")
            sources = {"onnx": model_path, "vnnlib": spec_path}
            before = source_file_digests(sources)
            result = verify_bab_batched(
                net,
                solver_factory,
                config,
                max_batch_size=8,
                time_budget_s=10.0,
                _property_forest_run_token=token,
                _property_forest_source_digests=before,
            )
            self.assertEqual(result.status, VerifyStatus.CERTIFIED)
            self.assertEqual(
                result.metadata["resolved_by"], "root_presolve"
            )
            safe, errors = validate_bab_safe_capability(
                result,
                net=net,
                solver_factory=solver_factory,
                config=config,
                max_batch_size=8,
                time_budget_s=10.0,
                expected_dtype="float64",
                expected_device="cpu",
                run_token=token,
                source_paths=sources,
                source_digests_before_run=before,
            )
            self.assertEqual(errors, ())
            self.assertIsNotNone(safe)
            self.assertEqual(
                safe["live_facts"]["mode"], "root_presolve"
            )
            self.assertEqual(
                safe["live_facts"]["root_certified_rows"],
                [0, 1, 2],
            )

    def test_live_phase_cube_partition_rejects_same_count_replacement(
        self,
    ):
        net = _two_axis_duplicate_relu_net()
        parent = SubproblemBatch(
            lb=-torch.ones(2, 2, dtype=DTYPE),
            ub=torch.ones(2, 2, dtype=DTYPE),
            depths=torch.zeros(2, dtype=torch.long),
            spec_row_ids=torch.tensor([2, 5], dtype=torch.long),
        )
        decision = SplitDecision(
            kind="neuron",
            layer_id=torch.tensor([3, 3], dtype=torch.long),
            neuron_idx=torch.tensor([0, 2], dtype=torch.long),
        )
        children, parent_index = _split_from_decision(
            parent, decision, net
        )
        valid, errors = _validate_property_forest_child_partition(
            parent,
            children,
            parent_index,
            expected_children_per_parent=2,
        )
        self.assertTrue(valid)
        self.assertEqual(errors, ())

        replacement = torch.tensor([0, 1, 0, 1])
        replaced = _select_forest_lanes(children, replacement)
        replaced_parent = parent_index.index_select(0, replacement)
        valid, errors = _validate_property_forest_child_partition(
            parent,
            replaced,
            replaced_parent,
            expected_children_per_parent=2,
        )
        self.assertFalse(valid)
        self.assertTrue(
            any("phase_cube_incomplete" in error for error in errors)
        )

    def test_split_row_id_swap_is_unknown_before_pool_push(self):
        def swapped_split(batch, split_dims):
            children, parent_index = split_input(batch, split_dims)
            if children.spec_row_ids is None:
                raise AssertionError("forest split lost ids before tamper")
            ids = children.spec_row_ids
            children.spec_row_ids = torch.where(
                ids == 2,
                torch.full_like(ids, 5),
                torch.full_like(ids, 2),
            )
            return children, parent_index

        with mock.patch(
            "act.back_end.bab.bab.split_input",
            side_effect=swapped_split,
        ):
            result, _, _ = _run_two_axis_property_forest()
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(
            result.metadata["reason"],
            "property_forest_incomplete_coverage",
        )
        self.assertEqual(result.metadata["pool_remaining"], 0)
        self.assertFalse(
            result.metadata[
                "property_forest_node_conservation_valid"
            ]
        )
        self.assertEqual(
            result.metadata[
                "property_forest_node_conservation_receipt"
            ]["totals"]["children_minted"],
            0,
        )

    def test_frontier_depth_zero_and_nan_gates_fail_closed(self):
        frontier, _, _ = _run_two_axis_property_forest(
            frontier_cap=1
        )
        self.assertEqual(frontier.status, VerifyStatus.UNKNOWN)
        self.assertTrue(
            frontier.metadata["any_dropped_frontier_cap"]
        )
        self.assertFalse(
            frontier.metadata["property_forest_coverage_complete"]
        )
        self.assertFalse(
            frontier.metadata[
                "property_forest_node_conservation_valid"
            ]
        )
        self.assertEqual(
            frontier.metadata[
                "property_forest_node_conservation_receipt"
            ]["totals"]["dropped"]["frontier_cap"],
            2,
        )

        depth, _, _ = _run_two_axis_property_forest(max_depth=0)
        self.assertEqual(depth.status, VerifyStatus.UNKNOWN)
        self.assertTrue(depth.metadata["any_dropped_max_depth"])
        self.assertFalse(
            depth.metadata["property_forest_coverage_complete"]
        )
        self.assertEqual(
            depth.metadata[
                "property_forest_node_conservation_receipt"
            ]["totals"]["dropped"]["max_depth"],
            2,
        )

        for label, threshold in (
            ("zero", 0.0),
            ("nan", float("nan")),
        ):
            with self.subTest(boundary=label):
                result, _, _ = _run_two_axis_property_forest(
                    hard_threshold=threshold,
                    max_depth=1,
                )
                self.assertEqual(result.status, VerifyStatus.UNKNOWN)
                self.assertTrue(
                    result.metadata["any_dropped_max_depth"]
                )
                self.assertFalse(
                    result.metadata[
                        "property_forest_coverage_complete"
                    ]
                )
                self.assertEqual(result.metadata["pool_remaining"], 0)

    def test_root_expansion_binds_original_rows_and_dual_state(self):
        root = SubproblemBatch(
            lb=-torch.ones(1, 3, dtype=DTYPE),
            ub=torch.ones(1, 3, dtype=DTYPE),
            depths=torch.zeros(1, dtype=torch.long),
            incremental_alpha={
                7: torch.tensor(
                    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
                    dtype=DTYPE,
                )
            },
            incremental_eta={
                7: torch.tensor(
                    [[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                    dtype=DTYPE,
                )
            },
        )
        forest = _expand_property_forest_root(
            root, torch.tensor([2, 5, 8])
        )
        self.assertEqual(forest.batch_size, 3)
        self.assertEqual(forest.spec_row_ids.tolist(), [2, 5, 8])
        self.assertEqual(forest.incremental_alpha[7].shape, (3, 1, 2))
        self.assertEqual(
            forest.incremental_alpha[7][:, 0].tolist(),
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        )
        self.assertEqual(
            forest.incremental_eta[7][:, 0].tolist(),
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            _expand_property_forest_root(
                root, torch.tensor([2, 2, 8])
            )

    def test_split_and_fair_pool_preserve_row_identity(self):
        forest = SubproblemBatch(
            lb=-torch.ones(4, 2, dtype=DTYPE),
            ub=torch.ones(4, 2, dtype=DTYPE),
            depths=torch.zeros(4, dtype=torch.long),
            lower_bound=torch.tensor(
                [-10.0, -9.0, -1.0, -0.5], dtype=DTYPE
            ),
            spec_row_ids=torch.tensor([0, 0, 1, 1]),
        )
        children, parent_index = split_input(
            forest, torch.zeros(4, dtype=torch.long)
        )
        self.assertTrue(
            torch.equal(
                children.spec_row_ids,
                forest.spec_row_ids.index_select(0, parent_index),
            )
        )

        pool = TopKBounding()
        pool.push(forest)
        popped = pool.pop(2)
        self.assertEqual(set(popped.spec_row_ids.tolist()), {0, 1})

    def test_nonfinite_priorities_are_deterministic_and_lossless(self):
        pool = TopKBounding()
        pool.push(
            SubproblemBatch(
                lb=torch.zeros(5, 1, dtype=DTYPE),
                ub=torch.ones(5, 1, dtype=DTYPE),
                depths=torch.zeros(5, dtype=torch.long),
                lower_bound=torch.tensor(
                    [
                        float("nan"),
                        float("-inf"),
                        0.0,
                        float("inf"),
                        1.0,
                    ],
                    dtype=DTYPE,
                ),
                node_id=torch.arange(5),
                parent_id=torch.full((5,), -1),
            )
        )

        scores = pool._priority_scores()
        self.assertTrue(bool(torch.isfinite(scores).all().item()))
        self.assertEqual(scores[0].item(), torch.finfo(DTYPE).max)
        self.assertEqual(scores[1].item(), torch.finfo(DTYPE).max)
        self.assertEqual(scores[3].item(), torch.finfo(DTYPE).min)

        popped_ids: list[int] = []
        while not pool.empty:
            popped_ids.extend(pool.pop(2).node_id.tolist())
        self.assertEqual(popped_ids[:2], [0, 1])
        self.assertEqual(popped_ids[-1], 3)
        self.assertEqual(sorted(popped_ids), list(range(5)))

    def test_nonfinite_priorities_keep_fair_row_rotation(self):
        pool = TopKBounding()
        pool.push(
            SubproblemBatch(
                lb=torch.zeros(6, 1, dtype=DTYPE),
                ub=torch.ones(6, 1, dtype=DTYPE),
                depths=torch.zeros(6, dtype=torch.long),
                lower_bound=torch.tensor(
                    [
                        float("nan"),
                        10.0,
                        float("-inf"),
                        10.0,
                        float("inf"),
                        0.0,
                    ],
                    dtype=DTYPE,
                ),
                node_id=torch.arange(6),
                parent_id=torch.full((6,), -1),
                spec_row_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
            )
        )

        first = pool.pop(3)
        self.assertEqual(first.spec_row_ids.tolist(), [0, 1, 2])
        self.assertEqual(first.node_id.tolist(), [0, 2, 5])

        popped_ids = first.node_id.tolist()
        while not pool.empty:
            popped_ids.extend(pool.pop(2).node_id.tolist())
        self.assertEqual(sorted(popped_ids), list(range(6)))

    def test_shared_alpha_neuron_axis_is_not_mistaken_for_specs(self):
        shared_alpha = SubproblemBatch(
            lb=-torch.ones(2, 3, dtype=DTYPE),
            ub=torch.ones(2, 3, dtype=DTYPE),
            depths=torch.zeros(2, dtype=torch.long),
            incremental_alpha={
                7: torch.zeros(2, 19, dtype=DTYPE)
            },
        )
        self.assertEqual(_infer_spec_axis_size(shared_alpha), 1)

        shared_alpha.spec_row_ids = torch.tensor([4, 9])
        self.assertEqual(_infer_spec_axis_size(shared_alpha), 1)

        per_class = SubproblemBatch(
            lb=-torch.ones(2, 3, dtype=DTYPE),
            ub=torch.ones(2, 3, dtype=DTYPE),
            depths=torch.zeros(2, dtype=torch.long),
            incremental_alpha={
                7: torch.zeros(2, 5, 19, dtype=DTYPE)
            },
        )
        self.assertEqual(_infer_spec_axis_size(per_class), 5)

        per_class.split_signs = {
            7: torch.zeros(2, 3, 19, dtype=DTYPE)
        }
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            _infer_spec_axis_size(per_class)

    def test_lane_specific_m1_slacks_equal_joint_m3_slacks(self):
        net = _three_row_affine_net()
        assert_layer = net.by_id[3]
        bounds = Bounds(
            -torch.ones(1, 3, dtype=DTYPE),
            torch.ones(1, 3, dtype=DTYPE),
        )
        config = BaBConfig(
            solver_tier="dual",
            branching_method="random",
        )
        joint_batch = SubproblemBatch.from_bounds(bounds)
        joint = _dispatch_dual_solve(
            net=net,
            assert_layer=assert_layer,
            batched_bounds=bounds,
            k_actual=1,
            batch=joint_batch,
            config=config,
            optimize=False,
        )
        self.assertEqual(joint.row_slack.shape, (1, 3))

        forest_batch = _expand_property_forest_root(
            SubproblemBatch.from_bounds(bounds),
            torch.arange(3),
        )
        forest_bounds = Bounds(
            bounds.lb.repeat(3, 1), bounds.ub.repeat(3, 1)
        )
        forest = _dispatch_dual_solve(
            net=net,
            assert_layer=assert_layer,
            batched_bounds=forest_bounds,
            k_actual=3,
            batch=forest_batch,
            config=config,
            optimize=False,
        )
        self.assertEqual(forest.row_slack.shape, (3, 1))
        self.assertTrue(
            torch.allclose(
                forest.row_slack[:, 0],
                joint.row_slack[0],
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_dual_alpha_eta_post_prune_m1_matches_joint_bitwise(self):
        gate = _run_dual_alpha_eta_post_prune_pair(
            hard_threshold=0.1,
            nan_row_five=False,
            split_neuron=0,
        )
        keep_rows = gate["keep_rows"]
        full_root = gate["full_root"]
        joint = gate["joint"]
        forest = gate["forest"]
        joint_batch = gate["joint_batch"]
        forest_batch = gate["forest_batch"]

        self.assertEqual(keep_rows.tolist(), [2, 5])
        self.assertEqual(
            torch.where(~full_root.row_certified[0])[0].tolist(),
            [2, 5],
        )
        self.assertEqual(joint.row_slack.shape, (1, 2))
        self.assertEqual(forest.row_slack.shape, (2, 1))
        self.assertTrue(
            torch.equal(
                joint.row_slack[0].view(torch.int64),
                forest.row_slack[:, 0].contiguous().view(torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                joint.row_certified[0],
                forest.row_certified[:, 0],
            )
        )
        self.assertEqual(
            joint.row_certified.tolist(),
            [[True, False]],
        )
        self.assertTrue(
            torch.equal(
                joint.row_certified,
                _strictly_certified_slack(joint.row_slack),
            )
        )
        self.assertEqual(
            joint.solution.statuses,
            (SolveStatus.SAT,),
        )
        self.assertEqual(
            forest.solution.statuses,
            (SolveStatus.UNSAT, SolveStatus.SAT),
        )

        for label, joint_state, forest_state in (
            (
                "alpha",
                joint_batch.incremental_alpha,
                forest_batch.incremental_alpha,
            ),
            (
                "eta",
                joint_batch.incremental_eta,
                forest_batch.incremental_eta,
            ),
            (
                "split signs",
                joint_batch.split_signs,
                forest_batch.split_signs,
            ),
        ):
            _assert_joint_forest_state_bitwise_equal(
                self,
                joint_state,
                forest_state,
                label=label,
            )

        warm_alpha = gate["warm_alpha"][3]
        optimized_alpha = joint_batch.incremental_alpha[3]
        self.assertFalse(
            torch.equal(
                warm_alpha.view(torch.int64),
                optimized_alpha.view(torch.int64),
            )
        )
        self.assertTrue(
            bool(torch.isfinite(optimized_alpha).all().item())
        )
        self.assertTrue(
            bool(
                ((optimized_alpha >= 0.0) & (optimized_alpha <= 1.0))
                .all()
                .item()
            )
        )
        optimized_eta = joint_batch.incremental_eta[3]
        self.assertTrue(bool(torch.isfinite(optimized_eta).all().item()))
        self.assertTrue(bool((optimized_eta >= 0.0).all().item()))

    def test_dual_alpha_eta_post_prune_zero_nan_fail_closed(self):
        gate = _run_dual_alpha_eta_post_prune_pair(
            hard_threshold=0.5,
            nan_row_five=True,
            split_neuron=2,
        )
        joint = gate["joint"]
        forest = gate["forest"]
        joint_batch = gate["joint_batch"]
        forest_batch = gate["forest_batch"]

        self.assertEqual(gate["keep_rows"].tolist(), [2, 5])
        self.assertEqual(float(joint.row_slack[0, 0].item()), 0.0)
        self.assertTrue(bool(torch.isnan(joint.row_slack[0, 1]).item()))
        self.assertTrue(
            torch.equal(
                joint.row_slack[0].view(torch.int64),
                forest.row_slack[:, 0].contiguous().view(torch.int64),
            )
        )
        self.assertEqual(
            joint.row_certified.tolist(),
            [[False, False]],
        )
        self.assertEqual(
            forest.row_certified.tolist(),
            [[False], [False]],
        )
        self.assertFalse(
            bool(_strictly_certified_slack(joint.row_slack).any().item())
        )
        self.assertEqual(
            joint.solution.statuses,
            (SolveStatus.SAT,),
        )
        self.assertEqual(
            forest.solution.statuses,
            (SolveStatus.SAT, SolveStatus.SAT),
        )

        for label, joint_state, forest_state in (
            (
                "boundary alpha",
                joint_batch.incremental_alpha,
                forest_batch.incremental_alpha,
            ),
            (
                "boundary eta",
                joint_batch.incremental_eta,
                forest_batch.incremental_eta,
            ),
            (
                "boundary split signs",
                joint_batch.split_signs,
                forest_batch.split_signs,
            ),
        ):
            _assert_joint_forest_state_bitwise_equal(
                self,
                joint_state,
                forest_state,
                label=label,
            )

    def test_zero_and_nonfinite_slack_fail_closed(self):
        mask = _strictly_certified_slack(
            torch.tensor(
                [[-1.0, 0.0, 5e-12, 1.0, float("nan"), float("inf")]],
                dtype=DTYPE,
            )
        )
        self.assertEqual(
            mask.tolist(),
            [[False, False, False, True, False, False]],
        )

        # max(x_0) == 1 on this box.  A row x_0 <= 1 therefore has exactly
        # zero certified slack and must remain unresolved: concrete TOP1 and
        # MARGIN semantics count the analogous output tie as a violation.
        net = _three_row_affine_net()
        net.by_id[3].params["d"] = torch.tensor(
            [[1.0, 2.0, 2.0]], dtype=DTYPE
        )
        bounds = Bounds(
            -torch.ones(1, 3, dtype=DTYPE),
            torch.ones(1, 3, dtype=DTYPE),
        )
        result = _dispatch_dual_solve(
            net=net,
            assert_layer=net.by_id[3],
            batched_bounds=bounds,
            k_actual=1,
            batch=SubproblemBatch.from_bounds(bounds),
            config=BaBConfig(solver_tier="dual"),
            optimize=False,
        )
        self.assertEqual(float(result.row_slack[0, 0].item()), 0.0)
        self.assertNotEqual(result.solution.statuses[0], "UNSAT")

        # The non-BaB dual entry point must use the same strict boundary and
        # non-finite policy; otherwise the ordinary verifier could still
        # certify a tie even though the C48 path fails closed.
        exact_boundary = OutputSpec(
            kind=OutKind.LINEAR_LE,
            c=torch.eye(3, dtype=DTYPE),
            d=torch.tensor([1.0, 2.0, 2.0], dtype=DTYPE),
        )
        evaluated = DualSolver().evaluate_spec(net, exact_boundary)
        self.assertEqual(float(evaluated.slack[0, 0].item()), 0.0)
        self.assertFalse(bool(evaluated.certified[0].item()))

        nonfinite_boundary = OutputSpec(
            kind=OutKind.LINEAR_LE,
            c=torch.eye(3, dtype=DTYPE),
            d=torch.tensor(
                [float("nan"), 2.0, 2.0], dtype=DTYPE
            ),
        )
        evaluated_nonfinite = DualSolver().evaluate_spec(
            net, nonfinite_boundary
        )
        self.assertTrue(
            bool(torch.isnan(evaluated_nonfinite.slack[0, 0]).item())
        )
        self.assertFalse(
            bool(evaluated_nonfinite.certified[0].item())
        )

    def test_invalid_row_and_or_semantics_fail_closed(self):
        net = _three_row_affine_net()
        bounds = Bounds(
            -torch.ones(1, 3, dtype=DTYPE),
            torch.ones(1, 3, dtype=DTYPE),
        )
        config = BaBConfig(solver_tier="dual")
        bad = SubproblemBatch.from_bounds(bounds)
        bad.spec_row_ids = torch.tensor([3])
        with self.assertRaisesRegex(ValueError, "outside"):
            _dispatch_dual_solve(
                net=net,
                assert_layer=net.by_id[3],
                batched_bounds=bounds,
                k_actual=1,
                batch=bad,
                config=config,
                optimize=False,
            )

        unsafe = _three_row_affine_net(unsafe=True)
        tagged = SubproblemBatch.from_bounds(bounds)
        tagged.spec_row_ids = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, "UNSAFE_LINEAR"):
            _dispatch_dual_solve(
                net=unsafe,
                assert_layer=unsafe.by_id[3],
                batched_bounds=bounds,
                k_actual=1,
                batch=tagged,
                config=config,
                optimize=False,
            )

    def test_counterexample_input_replay_checks_every_constraint(self):
        poly = Layer(
            id=10,
            kind="INPUT_SPEC",
            params={
                "kind": InKind.LIN_POLY,
                "A": torch.tensor([[1.0, 1.0]], dtype=DTYPE),
                "b": torch.tensor([0.5], dtype=DTYPE),
            },
            in_vars=[0, 1],
            out_vars=[0, 1],
        )
        points = torch.tensor(
            [[0.2, 0.2], [0.4, 0.4]], dtype=DTYPE
        )
        self.assertEqual(
            _check_input_specs_batched(points, [poly]).tolist(),
            [True, False],
        )

        linf = Layer(
            id=11,
            kind="INPUT_SPEC",
            params={
                "kind": InKind.LINF_BALL,
                "center": torch.zeros(1, 2, dtype=DTYPE),
                "eps": torch.tensor([0.5], dtype=DTYPE),
            },
            in_vars=[0, 1],
            out_vars=[0, 1],
        )
        linf_points = torch.tensor(
            [[0.5, -0.5], [0.5000000001, 0.0]], dtype=DTYPE
        )
        self.assertEqual(
            _check_input_specs_batched(linf_points, [linf]).tolist(),
            [True, False],
        )

        embedding = Layer(
            id=12,
            kind="INPUT_SPEC",
            params={
                "kind": InKind.LP_EMBEDDING,
                "center": torch.zeros(1, 2, 2, dtype=DTYPE),
                "eps": torch.tensor([1.0], dtype=DTYPE),
                "p_norm": torch.tensor(2.0, dtype=DTYPE),
                "perturbed_positions": torch.tensor([True, False]),
            },
            in_vars=[0, 1, 2, 3],
            out_vars=[0, 1, 2, 3],
        )
        embedding_points = torch.tensor(
            [
                [[0.6, 0.8], [0.0, 0.0]],
                [[0.6, 0.8], [0.0, 1e-12]],
            ],
            dtype=DTYPE,
        )
        self.assertEqual(
            _check_input_specs_batched(
                embedding_points, [embedding]
            ).tolist(),
            [True, False],
        )

        missing_box = Layer(
            id=13,
            kind="INPUT_SPEC",
            params={"kind": InKind.BOX},
            in_vars=[0, 1],
            out_vars=[0, 1],
        )
        self.assertEqual(
            _check_input_specs_batched(points, [missing_box]).tolist(),
            [False, False],
        )
        unknown = Layer(
            id=14,
            kind="INPUT_SPEC",
            params={"kind": "UNKNOWN_INPUT_KIND"},
            in_vars=[0, 1],
            out_vars=[0, 1],
        )
        with self.assertRaisesRegex(NotImplementedError, "unsupported"):
            _check_input_specs_batched(points, [unknown])


if __name__ == "__main__":
    unittest.main()
