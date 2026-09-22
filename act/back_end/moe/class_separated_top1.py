"""Class-separated raw-score top-1 intake (MetaMoE author semantics).

This is a separate entry, NOT an extension of selected-softmax top-2 F0.
It preserves zero-filled unselected classes and proves score/score defined
before accepting the branch reduction. Upstream ACT lowering and the frozen
HZ solver policy remain trusted; no source-complete certificate is claimed.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from pathlib import Path

import torch
from torch import nn

from act.back_end.moe.factory import build_act_moe_program
from act.back_end.moe.model import OutputLevelMoE
from act.back_end.moe.schema import OutputLevelMoESpec


METAMOE_SOURCE_SHA256 = "560ef6cee9000faf0f1d4579d4e91f7a0439d3cdc3a8722e37b69571f3836c34"
MODEL_WRAPPER_SHA256 = "078cd5e856c8679cde28853a9b5d6db74c82d9114a8ed95073611eadd5b41e2c"


class _PaddedExpert(nn.Module):
    def __init__(self, expert, width, offset, total, *, dtype, device):
        super().__init__()
        self.expert = expert
        self.embed = nn.Linear(width, total, bias=False, dtype=dtype, device=device)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[offset:offset + width] = torch.eye(width, dtype=dtype, device=device)
        self.embed.requires_grad_(False)

    def forward(self, x):
        return self.embed(self.expert(x))


class ClassSeparatedTop1(nn.Module):
    """Executable zero-block/score-division contract; optional original model.

    Components must return logits tensors, and must already be in eval mode.
    The author's wrapper can be unwrapped only by the pinned intake below.
    No dtype conversion, normalization, BN folding or clipping is implicit.
    """
    def __init__(self, router, experts, class_counts, *, original=None, source_sha256=None):
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.class_counts = tuple(class_counts)
        if (len(self.experts) != len(self.class_counts) or not self.experts or
                any(type(n) is not int or n < 1 for n in self.class_counts)):
            raise ValueError("invalid class partitions")
        self.offsets = (0,)
        for n in self.class_counts:
            self.offsets += (self.offsets[-1] + n,)
        self.total_classes = self.offsets[-1]
        self.original = original
        self.source_sha256 = source_sha256
        if any(m.training for component in [router, *experts] for m in component.modules()):
            raise ValueError("eval mode is required; do not silently change the model")
        self.eval()

    @classmethod
    def from_metamoe(cls, model):
        """Accept the pinned ORIGINAL author class, not a guessed lookalike."""
        if (type(model).__module__ != "vision_transformer_moe" or type(model).__name__ != "MetaMoE"
                or "forward" in model.__dict__):
            raise ValueError("unsupported or instance-overridden MetaMoE source")
        path = inspect.getsourcefile(type(model))
        if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != METAMOE_SOURCE_SHA256:
            raise ValueError("unreviewed MetaMoE source version")
        if model.meta_top_k != 1 or any(m.training for m in model.modules()):
            raise ValueError("original MetaMoE intake requires eval and top-1")
        counts = tuple(model.num_classes_list)
        offsets = [0]
        for n in counts:
            offsets.append(offsets[-1] + n)
        if (model.num_experts != len(counts) or list(model.class_offsets) != offsets
                or model.total_classes != offsets[-1]):
            raise ValueError("author partition metadata mismatch")
        components = []
        for e in model.experts:
            # The pinned ModelWrapper adds only a zero auxiliary tensor.
            if type(e).__module__ == "model_wrapper" and type(e).__name__ == "ModelWrapper":
                p = Path(inspect.getsourcefile(type(e)))
                if hashlib.sha256(p.read_bytes()).hexdigest() != MODEL_WRAPPER_SHA256:
                    raise ValueError("unreviewed expert wrapper")
                if "forward" in e.__dict__:
                    raise ValueError("overridden expert wrapper")
                components.append(e.model)
            else:
                components.append(e)
        return cls(model.meta_gating_net, components, counts, original=model,
                   source_sha256=METAMOE_SOURCE_SHA256)

    def forward(self, x):
        if self.original is not None:
            return self.original(x)[0]
        scores = self.router(x)
        values, indices = scores.topk(1, dim=1)
        weights = values / values.sum(dim=1, keepdim=True)
        out = x.new_zeros(x.shape[0], self.total_classes)
        for i, expert in enumerate(self.experts):
            samples, slots = torch.where(indices == i)
            if len(samples):
                out[samples, self.offsets[i]:self.offsets[i + 1]] = (
                    expert(x[samples]) * weights[samples, slots, None]).to(x.dtype)
        return out

    def reduced_components(self, sample):
        with torch.no_grad():
            scores = self.router(sample)
            if scores.shape != (1, len(self.experts)) or not torch.isfinite(scores).all():
                raise ValueError("router shape/nonfinite control failed")
            for i, expert in enumerate(self.experts):
                y = expert(sample)
                if not isinstance(y, torch.Tensor) or y.shape != (1, self.class_counts[i]):
                    raise ValueError("expert class width mismatch")
                if not torch.isfinite(y).all():
                    raise ValueError("nonfinite expert")
        experts = [_PaddedExpert(e, n, self.offsets[i], self.total_classes,
                                  dtype=sample.dtype, device=sample.device).eval()
                   for i, (e, n) in enumerate(zip(self.experts, self.class_counts))]
        # A module container also keeps a bare nn.Linear as an FX call_module.
        return OutputLevelMoE(nn.Sequential(self.router), experts, OutputLevelMoESpec(len(experts))).eval()


def classification_rows(total_classes, label, *, dtype=torch.float64):
    if not 0 <= label < total_classes or total_classes < 2:
        raise ValueError("invalid global class label")
    q = torch.zeros(total_classes - 1, total_classes, dtype=dtype, device="cpu")
    for row, other in enumerate(i for i in range(total_classes) if i != label):
        q[row, label], q[row, other] = 1, -1
    return q


def validate_replay(model, point, lower, upper, rows, thresholds):
    """Original complete-model witness only; NaNs and out-of-box points fail."""
    if point is None or point.numel() != lower.numel():
        return False
    x = point.detach().reshape_as(lower).to(lower)
    if not torch.isfinite(x).all() or (x < lower).any() or (x > upper).any():
        return False
    with torch.no_grad():
        out = model(x)
    return bool(torch.isfinite(out).all() and ((out @ rows.T) < thresholds).any())


def verify_class_separated_box(model, *, center, lower, upper, rows, thresholds,
                               total_seconds=300.0, hybridz_config=None):
    """Tier-1 HZ-policy result for q @ output >= threshold on ALL legal top-1s.

    This in-process routine rejects late results and spends remaining time.
    Native solver calls MUST additionally be enclosed by the process supervisor
    for a hard deadline (the deployment runner does so). No new solver gate.
    """
    from act.back_end.moe.route_a import RouteAEngine
    from act.back_end.solver.solver_hz import hz_support_bounds
    from act.front_end.specs import OutKind, OutputSpec
    from act.util.stats import VerifyStatus
    from act.util.device_manager import get_default_device
    started = time.monotonic()
    if not math.isfinite(total_seconds) or total_seconds <= 0:
        raise ValueError("positive finite request budget required")
    deadline = started + total_seconds
    if get_default_device().type != "cpu":
        raise ValueError("initialize ACT with initialize_device('cpu', 'float64') before this CPU entry")
    tensors = [center, lower, upper, rows, thresholds]
    if (any(t.device.type != "cpu" or t.dtype != torch.float64 or not torch.isfinite(t).all() for t in tensors)
            or center.shape != lower.shape or center.shape != upper.shape or center.ndim < 2
            or center.shape[0] != 1 or (lower > center).any() or (center > upper).any()
            or rows.ndim != 2 or rows.shape[1] != model.total_classes or rows.shape[0] < 1
            or thresholds.shape != (rows.shape[0],)):
        raise ValueError("explicit CPU/float64 box and complete property rows required")
    if any(m.training for m in model.modules()):
        raise ValueError("eval required")
    if any(p.device.type != "cpu" or p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError("no implicit model dtype/device conversion")
    result = {"schema": 1, "status": "UNKNOWN", "evidence_grade": "NONE",
              "semantics": "class_separated_raw_top1_zero_fill_any_legal_ties",
              "class_counts": list(model.class_counts), "source_sha256": model.source_sha256,
              "property_rows": len(rows), "nonzero_obligations": [],
              "trusted": ["network_to_HZ", "input_and_guard_lowering", "HZ_solver_numerical_policy"],
              "source_complete": False}
    def finish(status, reason, grade="NONE"):
        result.update(status=status, reason=reason, evidence_grade=grade,
                      seconds=time.monotonic() - started)
        if time.monotonic() >= deadline:
            result.update(status="TIMEOUT", reason="request_deadline", evidence_grade="NONE")
        return result
    if validate_replay(model, center, lower, upper, rows, thresholds):
        result["witness"] = center.tolist()
        return finish("UNSAFE_REPLAYED", "original_model_center", "FULL_MODEL_REPLAY")
    surrogate = model.reduced_components(center)
    program = build_act_moe_program(surrogate, center=center, lower=lower, upper=upper,
                                    output_spec=OutputSpec(kind=OutKind.LINEAR_LE, c=-rows, d=-thresholds))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return finish("TIMEOUT", "lowering_deadline")
    from act.back_end.hybridz_tf.sparse_budget import SparseResourceLimit
    engine = RouteAEngine(program, expert_models=surrogate.experts, hybridz_config=hybridz_config,
                         time_limit_per_route=min(30.0, remaining / (3 * len(model.experts))))
    try:
        report = engine.run_tier1()
    except SparseResourceLimit as exc:
        result['sparse_resource_failure'] = exc.event
        result['sparse_resource_events'] = engine.sparse_resource_events
        return finish('UNKNOWN', 'sparse_representation_resource_limit')
    result['sparse_resource_events'] = engine.sparse_resource_events
    candidates = report.router.candidates
    result.update(candidates=list(candidates.candidates), excluded=list(candidates.infeasible),
                  unresolved=list(candidates.unresolved), candidate_minimal=candidates.minimal,
                  expert_statuses={str(i): r.status.value for i, r in report.expert_results})
    # Never infer completeness from an empty or partial set of statuses.
    if (not candidates.candidates or candidates.unresolved or
            set(candidates.candidates) & set(candidates.infeasible) or
            set(candidates.candidates) | set(candidates.infeasible) != set(range(len(model.experts)))):
        return finish("UNKNOWN", "incomplete_route_coverage")
    for i, r in report.expert_results:
        if validate_replay(model, r.counterexample, lower, upper, rows, thresholds):
            result["witness"] = r.counterexample.tolist()
            return finish("UNSAFE_REPLAYED", "original_model_expert_witness", "FULL_MODEL_REPLAY")
    if time.monotonic() >= deadline:
        return finish("TIMEOUT", "tier1_deadline")
    by_expert = {b.expert: b for b in candidates.branches}
    for pos, i in enumerate(candidates.candidates):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return finish("TIMEOUT", "nonzero_deadline")
        support = hz_support_bounds(by_expert[i].conditioned_router, [i],
                    time_limit=min(30.0, remaining / (len(candidates.candidates) - pos)), relax_binaries=False)
        lo, hi = float(support.bounds.lb.item()), float(support.bounds.ub.item())
        proven = math.isfinite(lo) and math.isfinite(hi) and lo <= hi and (lo > 0 or hi < 0)
        result["nonzero_obligations"].append({"expert": i, "lower": lo, "upper": hi,
            "lower_status": list(support.lower_status), "upper_status": list(support.upper_status),
            "accepted": proven})
    statuses = dict(report.expert_results)
    if (all(r["accepted"] for r in result["nonzero_obligations"])
            and set(statuses) == set(candidates.candidates)
            and all(r.status == VerifyStatus.CERTIFIED for r in statuses.values())):
        return finish("POSITIVE", "all_guards_defined_and_global_properties_accepted", "HZ_POLICY_ACCEPTED")
    return finish("UNKNOWN", "nonzero_or_global_output_obligation_unproved")
