"""Production loader gate: off-path identity, exact HZ semantics, atomic failure."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from act.pipeline.verification import batchnorm_graph
from act.pipeline.verification.torch2act import TorchToACT
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.verifiable_model import (
    InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel,
)
from act.back_end.hybridz_tf.tf_cnn import sparse_conv2d_matrix_from_layer_csr
from act.back_end.solver.solver_hz import (
    SparseHZono, sparse_hz_linear, sparse_hz_add_const, sparse_hz_add_same_frame,
)
from experiments.neural_hz_20260831 import bn_graph_faithfulness_certificate_prototype as oracle
from experiments.neural_hz_20260831.test_bn_graph_faithfulness_certificate_prototype import _bn_chain


class ResidualBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1, bias=True).double()
        self.bn = nn.BatchNorm2d(2, eps=0.0).double()
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[0.5, -0.25], [1., 0.5]]).reshape(2, 2, 1, 1))
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))
            self.bn.weight.copy_(torch.tensor([-0.5, 1.5]))
            self.bn.bias.copy_(torch.tensor([0.25, -0.25]))
            self.bn.running_mean.copy_(torch.tensor([0.5, -1.]))
            self.bn.running_var.copy_(torch.tensor([1., 4.]))
        self.eval()

    def forward(self, x):
        return self.bn(self.conv(x)) + x


def wrap(model):
    x = torch.zeros((1, 2, 2, 2), dtype=torch.float64)
    return VerifiableModel(
        input_layer=InputLayer(labeled_input=LabeledInputTensor(x, torch.tensor([0])),
                               shape=tuple(x.shape), dtype=x.dtype),
        input_spec=InputSpecLayer(InputSpec(kind=InKind.BOX, lb=x - 1, ub=x + 1)),
        model=model,
        output_spec=OutputSpecLayer(OutputSpec(kind=OutKind.RANGE,
                                               lb=torch.full((1, 8), -100., dtype=x.dtype))),
    )


def test_flag_off_never_calls_repair_and_retains_old_graph(monkeypatch):
    wrapped = wrap(ResidualBN())
    reference = TorchToACT(wrapped).run()
    def forbidden(*args):
        raise AssertionError("off path called repair")
    monkeypatch.setattr(batchnorm_graph, "repair_batchnorm_producer_graph", forbidden)
    implicit = TorchToACT(wrapped).run()
    explicit = TorchToACT(wrapped, repair_batchnorm_producer_graph=False).run()
    assert implicit.preds == explicit.preds == reference.preds
    assert implicit.succs == explicit.succs == reference.succs
    assert not oracle.audit_graph_faithfulness(reference.layers, reference.preds, reference.succs).accepted


def test_flag_on_matches_independent_clone_and_is_idempotent():
    wrapped = wrap(ResidualBN())
    old = TorchToACT(wrapped).run()
    plan = oracle.plan_batchnorm_graph_repair(old.layers, old.preds, old.succs)
    clone, certificate = oracle.apply_repair_plan_to_clone(old.layers, old.preds, old.succs, plan)
    candidate = TorchToACT(wrapped, repair_batchnorm_producer_graph=True).run()
    assert certificate.accepted
    assert candidate.preds == clone.predecessor_dict()
    assert candidate.succs == clone.successor_dict()
    assert oracle.audit_graph_faithfulness(candidate.layers, candidate.preds, candidate.succs).accepted
    p, s = batchnorm_graph.repair_batchnorm_producer_graph(candidate.layers, candidate.preds, candidate.succs)
    assert p is candidate.preds and s is candidate.succs


def test_flag_on_non_bn_model_has_identical_graph():
    wrapped = wrap(nn.Identity())
    a = TorchToACT(wrapped).run()
    b = TorchToACT(wrapped, repair_batchnorm_producer_graph=True).run()
    assert a.preds == b.preds and a.succs == b.succs


def test_flag_type_is_explicit():
    with pytest.raises(TypeError):
        TorchToACT(wrap(ResidualBN()), repair_batchnorm_producer_graph="false")


@pytest.mark.parametrize("fault", ["nonfinite", "width", "missing_pair", "unrelated", "asymmetry", "duplicate_producer", "wrong_shape", "nonsibling"])
def test_repair_rejects_inconsistent_graph_without_mutation(fault):
    layers, preds, succs = _bn_chain(sibling=True)
    for lid, name in ((3, "a"), (4, "c")):
        layers[lid].params[name] = torch.tensor(layers[lid].params[name], dtype=torch.float64)
    if fault == "nonfinite":
        layers[3].params["a"][0] = float("nan")
    elif fault == "width":
        layers[4].params["c"] = torch.ones(1)
    elif fault == "missing_pair":
        layers[4].params["paired_with_scale"] = False
    elif fault == "unrelated":
        layers[5].in_vars[0] = 0
    elif fault == "asymmetry":
        succs[2].remove(4)
    elif fault == "duplicate_producer":
        layers[5].out_vars[0] = 0
    elif fault == "wrong_shape":
        layers[3].params["a"] = torch.ones((1, 2))
    elif fault == "nonsibling":
        preds[4] = [1]
        succs[2].remove(4)
        succs[1].append(4)
    before = deepcopy((preds, succs))
    with pytest.raises(ValueError):
        batchnorm_graph.repair_batchnorm_producer_graph(layers, preds, succs)
    assert (preds, succs) == before


def test_repaired_residual_preserves_nonconvex_hz_and_exact_dyadic_witnesses():
    model = ResidualBN()
    net = TorchToACT(wrap(model), repair_batchnorm_producer_graph=True).run()
    source = SparseHZono(
        c=np.arange(8, dtype=np.float64) / 16,
        Gc=sp.csr_matrix(np.arange(16, dtype=np.float64).reshape(8, 2) / 32),
        Gb=sp.csr_matrix(np.array([1., -1.] * 4).reshape(8, 1) / 8),
        Ac=sp.csr_matrix([[1., 0.]]), Ab=sp.csr_matrix([[-1.]]), b=np.array([0.]),
        Auc=sp.csr_matrix([[0., 1.]]), Aub=sp.csr_matrix([[0.5]]), ub=np.array([1.]),
        frame_id=17, exact=True,
    )
    cache = {}
    for layer in net.layers:
        operands = [cache[parent] for parent in net.preds[layer.id]]
        if layer.kind == "INPUT":
            hz = source
        elif layer.kind in {"INPUT_SPEC", "ASSERT"}:
            hz = operands[0]
        elif layer.kind == "CONV2D":
            matrix, bias = sparse_conv2d_matrix_from_layer_csr(layer)
            hz = sparse_hz_linear(operands[0], matrix, bias)
        elif layer.kind == "SCALE":
            hz = sparse_hz_linear(operands[0], sp.diags(layer.params["a"].numpy()))
        elif layer.kind == "BIAS":
            hz = sparse_hz_add_const(operands[0], layer.params["c"])
        elif layer.kind == "ADD":
            hz = sparse_hz_add_same_frame(*operands)
        else:
            raise AssertionError(layer.kind)
        cache[layer.id] = hz
    out = cache[net.layers[-1].id]
    assert (out.n_cont, out.n_bin, out.frame_id, out.exact) == (2, 1, 17, True)
    # Independent closed-form affine map proves equality for ALL latent
    # assignments, beyond the concrete witness checks below.
    channels = np.array([[0.5, -0.25], [1., 0.5]])
    scale = np.array([-0.5, 0.75])
    matrix = np.kron(np.eye(2) + np.diag(scale) @ channels, np.eye(4))
    bias = np.repeat(scale * np.array([0.25, -0.5])
                     + np.array([0.25, -0.25]) - scale * np.array([0.5, -1.]), 4)
    np.testing.assert_array_equal(out.c, matrix @ source.c + bias)
    np.testing.assert_array_equal(out.Gc.toarray(), matrix @ source.Gc.toarray())
    np.testing.assert_array_equal(out.Gb.toarray(), matrix @ source.Gb.toarray())
    for name in ("Ac", "Ab", "Auc", "Aub"):
        np.testing.assert_array_equal(getattr(source, name).toarray(), getattr(out, name).toarray())
    for name in ("b", "ub"):
        np.testing.assert_array_equal(getattr(source, name), getattr(out, name))
    # Equality xi[0] = z retains both binary phases. Every chosen point obeys
    # the inequality; all operations here have exact dyadic arithmetic.
    for z, t in product((-1., 1.), (-1., 0., 0.5)):
        xi, binary = np.array([z, t]), np.array([z])
        assert np.all(source.Auc @ xi + source.Aub @ binary <= source.ub)
        x = source.c + source.Gc @ xi + source.Gb @ binary
        with torch.no_grad():
            concrete = model(torch.from_numpy(x).reshape(1, 2, 2, 2)).numpy().reshape(-1)
        represented = out.c + out.Gc @ xi + out.Gb @ binary
        np.testing.assert_array_equal(represented, concrete)
