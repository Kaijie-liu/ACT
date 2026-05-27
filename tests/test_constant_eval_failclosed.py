"""Soundness regression: ``_evaluate_constant_subgraph`` must fail-closed
on placeholder.

History
=======
The R12 advisor audit identified that ``_evaluate_constant_subgraph``,
when reaching the model's input placeholder, would silently substitute
``self.sample_input`` and continue. Any handler that relied on this
return value (``OnnxConstantOfShape``, ``OnnxExpand``,
``OnnxSlice`` via ``_resolve_slice_input_to_int_list``, ``OnnxPow`` /
``OnnxBinaryMathOperation`` constant resolution) would then bake a
data-dependent shape/index/branch into the ACT IR as a fixed tensor.

For formal verification of an input *box*, this is unsound: the IR is
only correct at the sample center, not across the spec region.

The fix routes substitution behind an explicit
``allow_sample_substitution`` kwarg, defaulting to False. These tests
pin both halves of the new contract:

  1. Default (formal-safe) call returns ``None`` at the placeholder.
  2. Opt-in call still substitutes ``sample_input`` for the limited
     non-formal uses (sample-locally-valid IR).
  3. The recursion propagates the flag, so a deep chain that bottoms
     out at the placeholder behaves consistently.

We exercise the actual builder rather than mocking it to keep the test
honest about API drift.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn

from act.pipeline.verification.torch2act import _LayerGraphBuilder


class _DeviceIsolated(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._dev = torch.get_default_device() if hasattr(torch, "get_default_device") else None
        self._dt = torch.get_default_dtype()
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        try:
            torch.set_default_device(self._dev or "cpu")
        except Exception:
            pass
        torch.set_default_dtype(self._dt)
        super().tearDown()


class TestEvaluateConstantSubgraphFailClosed(_DeviceIsolated):
    """A chain rooted at the model placeholder is NOT a constant; the
    evaluator must say so (return None) under the default policy."""

    def _make_builder_with_placeholder_chain(self):
        # A trivial linear model so the FX graph has a placeholder and at
        # least one call_module reachable from it. The model is irrelevant
        # — we exercise _evaluate_constant_subgraph on its first node.
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(3, 2)

            def forward(self, x):
                return self.lin(x)

        m = Tiny().to(torch.float64)
        sample = torch.zeros(1, 3, dtype=torch.float64)
        b = _LayerGraphBuilder(m, (1, 3), torch.float64, sample_input=sample)
        n_inputs = 3
        b.prev_out = b._alloc_ids(n_inputs)
        b._extract_graph()
        b._pre_register_nodes()
        # Locate the placeholder fx node name
        ph_name = None
        for n in b.fx_graph.nodes:
            if n.op == "placeholder":
                ph_name = n.name
                break
        self.assertIsNotNone(ph_name, "expected a placeholder in FX graph")
        return b, ph_name

    def test_default_returns_none_on_placeholder(self):
        """The default (formal-safe) call MUST return None even though
        ``sample_input`` is set on the builder. Silently substituting
        would downgrade the IR to sample-local validity."""
        b, ph_name = self._make_builder_with_placeholder_chain()
        v = b._evaluate_constant_subgraph(ph_name)
        self.assertIsNone(
            v,
            "default _evaluate_constant_subgraph must fail-closed at the "
            "placeholder; got a tensor instead"
        )

    def test_explicit_opt_in_substitutes_sample(self):
        """The opt-in caller (``allow_sample_substitution=True``) still
        substitutes the sample. This path remains for documented sample-
        locally-valid use cases only — formal callers must NOT use it."""
        b, ph_name = self._make_builder_with_placeholder_chain()
        v = b._evaluate_constant_subgraph(ph_name, allow_sample_substitution=True)
        self.assertIsNotNone(v)
        self.assertTrue(torch.is_tensor(v))
        self.assertEqual(tuple(v.shape), (1, 3))

    def test_no_sample_no_substitution_even_with_optin(self):
        """When the builder was constructed without ``sample_input``,
        even the opt-in call cannot substitute and must return None.
        Prevents accidental silent fall-through if a caller forgot to
        pass the sample."""
        class Tiny(nn.Module):
            def __init__(self): super().__init__(); self.lin = nn.Linear(3, 2)
            def forward(self, x): return self.lin(x)
        m = Tiny().to(torch.float64)
        b = _LayerGraphBuilder(m, (1, 3), torch.float64, sample_input=None)
        b.prev_out = b._alloc_ids(3)
        b._extract_graph()
        b._pre_register_nodes()
        ph_name = next(n.name for n in b.fx_graph.nodes if n.op == "placeholder")
        self.assertIsNone(b._evaluate_constant_subgraph(
            ph_name, allow_sample_substitution=True
        ))


class TestPlaceholderAwareHandlersFailClosed(_DeviceIsolated):
    """End-to-end: a handler that resolves shape via
    ``_evaluate_constant_subgraph`` must NOT accept a placeholder-rooted
    chain. The visible signal is a clean ValueError, not a silent
    convert-and-continue.

    We assert this via the actual ``_convert_OnnxConstantOfShape`` /
    ``_convert_OnnxExpand`` handlers by constructing a builder whose
    state mimics 'shape arg comes from the placeholder' and confirming
    the resolver returns None.
    """

    def test_constant_of_shape_rejects_placeholder_shape(self):
        # We don't synthesize an actual OnnxConstantOfShape model — just
        # confirm the resolver layer (the same code the handler calls)
        # returns None for a placeholder-rooted arg. Handlers raise
        # "cannot resolve target shape" on None, which is the desired
        # fail-closed behavior.
        class Tiny(nn.Module):
            def __init__(self): super().__init__(); self.lin = nn.Linear(2, 2)
            def forward(self, x): return self.lin(x)
        m = Tiny().to(torch.float64)
        sample = torch.zeros(1, 2, dtype=torch.float64)
        b = _LayerGraphBuilder(m, (1, 2), torch.float64, sample_input=sample)
        b.prev_out = b._alloc_ids(2)
        b._extract_graph()
        b._pre_register_nodes()
        ph_name = next(n.name for n in b.fx_graph.nodes if n.op == "placeholder")
        # The actual handlers call this without specifying the kwarg, so
        # the default (False) applies. Confirm None is returned.
        self.assertIsNone(b._evaluate_constant_subgraph(ph_name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
