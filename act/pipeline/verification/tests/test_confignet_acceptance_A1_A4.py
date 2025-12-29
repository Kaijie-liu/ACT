# act/pipeline/verification/tests/test_confignet_acceptance_A1_A4.py
"""
Acceptance tests for ConfigNet refactor: A1–A4

A1: import-light build_parser + canonical_hash single source of truth
A2: sampler: deterministic + no hard-coded template names
A3: overrides: applied correctly + ModelFactory strict unique layer checks
A4: torch reference is derived from overridden ACT net (eps/y_true reflected)
"""

from __future__ import annotations

import copy
import inspect
import sys
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest
pytest.importorskip("torch")


# -----------------------------------------------------------------------------
# A1: import-light / canonical_hash
# -----------------------------------------------------------------------------


def _no_new_heavy_imports(before: set, after: set) -> None:
    new = after - before
    assert not any(m == "torchvision" or m.startswith("torchvision.") for m in new), f"New heavy import: {new}"
    assert not any(m == "torch" or m.startswith("torch.") for m in new), f"New heavy import: {new}"
    assert not any(m == "transformers" or m.startswith("transformers.") for m in new), f"New heavy import: {new}"


warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    module="torchvision.io.image",
)


def test_A1_build_parser_import_light_and_confignet_default_pool() -> None:
    """
    Must remain lightweight:
      - build_parser() must not import torch/torchvision/transformers
      - --validate-verifier default confignet_source must be pool
    """
    from act.pipeline.cli import build_parser

    before = set(sys.modules)
    parser = build_parser()
    after = set(sys.modules)
    _no_new_heavy_imports(before, after)

    args = parser.parse_args(["--validate-verifier"])
    assert args.confignet_source == "pool"

    args2 = parser.parse_args(["--validate-verifier", "--confignet-source", "runtime"])
    assert args2.confignet_source == "runtime"


def test_A1_canonical_hash_is_single_source_of_truth() -> None:
    """
    Enforce: canonical_hash must have ONE definition (single source of truth),
    and jsonl must import/reuse it rather than re-defining.

    Acceptance condition:
      - act.pipeline.verification.confignet.jsonl.canonical_hash is
        act.pipeline.verification.confignet.configs.canonical_hash
    """
    from act.pipeline.verification.confignet import configs
    from act.pipeline.verification.confignet import jsonl

    assert (
        jsonl.canonical_hash is configs.canonical_hash
    ), "canonical_hash must be defined once (configs.py) and reused in jsonl.py"

    # sanity: order-invariant hashing on dict keys
    h1 = configs.canonical_hash({"b": 1, "a": 2})
    h2 = configs.canonical_hash({"a": 2, "b": 1})
    assert h1 == h2


# -----------------------------------------------------------------------------
# A2: sampler no hard-coding + deterministic
# -----------------------------------------------------------------------------


def test_A2_sampler_is_deterministic_and_has_no_hardcoded_templates() -> None:
    """
    Enforce sampler refactor:
      - no hard-coded template names like "mnist_robust_easy" / "cifar_margin_moderate"
      - no TEMPLATE_BY_ARCH dict at module scope
      - sampling must be deterministic for same seed
    """
    from act.pipeline.verification.confignet import sampler as sampler_mod
    from act.pipeline.verification.confignet.sampler import sample_configs

    src = inspect.getsource(sampler_mod)

    forbidden_literals = ["mnist_robust_easy", "cifar_margin_moderate"]
    for lit in forbidden_literals:
        assert lit not in src, f"Hard-coded template '{lit}' found in sampler.py; must sample from ModelFactory pool."

    assert "TEMPLATE_BY_ARCH" not in src, "Do not keep TEMPLATE_BY_ARCH hard-coding. Derive templates dynamically."

    # Determinism: same seed => same sequence
    out1 = sample_configs(seed=0, n=10)
    out2 = sample_configs(seed=0, n=10)

    d1 = [(m.to_dict(), s.to_dict()) for (m, s) in out1]
    d2 = [(m.to_dict(), s.to_dict()) for (m, s) in out2]
    assert d1 == d2, "Sampling must be deterministic for the same seed."


# -----------------------------------------------------------------------------
# A3: overrides applied + strict unique checks (INPUT_SPEC / ASSERT)
# -----------------------------------------------------------------------------


def _pick_network_with_layers(factory: Any) -> str:
    """
    Pick a network name that contains both INPUT_SPEC and ASSERT.
    """
    for name in sorted(factory.list_networks()):
        net = factory.get_act_net(name)
        kinds = [L.kind for L in net.layers]
        if "INPUT_SPEC" in kinds and "ASSERT" in kinds:
            return name
    raise AssertionError("No network in examples pool has both INPUT_SPEC and ASSERT.")


def _dup_first_layer(net: Any, kind: str) -> Any:
    """
    Return a deepcopied net with the first layer of given kind duplicated.
    """
    new_net = copy.deepcopy(net)
    for i, L in enumerate(new_net.layers):
        if L.kind == kind:
            dup = copy.deepcopy(L)
            new_net.layers.insert(i + 1, dup)
            # rebuild by_id if present / expected
            new_net.by_id = {layer.id: layer for layer in new_net.layers}
            return new_net
    raise AssertionError(f"Layer kind {kind} not found to duplicate.")


def test_A3_overrides_applied_and_factory_strict_unique_layers() -> None:
    """
    A3 acceptance:
      1) build_act_net must apply overrides correctly (eps/assert_kind/y_true)
      2) ModelFactory.get_act_net must enforce strict uniqueness:
           - exactly ONE INPUT_SPEC and exactly ONE ASSERT
         and raise ValueError if duplicates exist (even in cached nets)
    """
    from act.pipeline.verification.confignet.builders import build_act_net
    from act.pipeline.verification.confignet.configs import ModelConfig, SpecConfig
    from act.pipeline.verification.confignet.utils import extract_effective_spec
    from act.pipeline.verification.model_factory import ModelFactory

    # (1) overrides applied
    m = ModelConfig(
        arch="mlp",
        template_name="mnist_robust_easy",
        input_shape=[1, 784],
        num_classes=10,
        seed=0,
    )
    s = SpecConfig(
        eps=0.02,
        norm="linf",
        targeted=False,
        true_label=3,
        assert_kind="TOP1_ROBUST",
    )
    net = build_act_net(m, s, name="caseA3")
    eff = extract_effective_spec(net)

    assert eff["eps"] == pytest.approx(0.02)
    assert eff["assert_kind"] == "TOP1_ROBUST"
    assert eff["y_true"] == 3

    # (2) strict uniqueness checks
    factory = ModelFactory()
    name = _pick_network_with_layers(factory)

    # backup original cached net
    original = factory.nets[name]

    try:
        # Make cached net invalid (duplicate INPUT_SPEC)
        bad_net = _dup_first_layer(original, "INPUT_SPEC")
        factory.nets[name] = bad_net

        # Any override call must now raise (strict unique)
        with pytest.raises(ValueError):
            _ = factory.get_act_net(name, spec_overrides={"eps": 0.01})

        # Also test duplicate ASSERT
        factory.nets[name] = _dup_first_layer(original, "ASSERT")
        with pytest.raises(ValueError):
            _ = factory.get_act_net(name, spec_overrides={"y_true": 1, "assert_kind": "TOP1_ROBUST"})

    finally:
        factory.nets[name] = original


# -----------------------------------------------------------------------------
# A4: torch reference follows overrides (built from overridden act_net)
# -----------------------------------------------------------------------------


def _extract_eps_from_torch_model(torch_model: Any) -> Optional[float]:
    """
    Best-effort extraction: scan modules for something that looks like an InputSpec layer.
    We accept multiple implementation styles:
      - module class name contains 'InputSpec'
      - module has attribute 'eps'
      - module has meta dict with 'eps'
    """
    # Avoid importing torch at module scope
    import torch  # noqa: F401

    eps_candidates: List[float] = []

    for mod in torch_model.modules():
        cls_name = mod.__class__.__name__
        if "InputSpec" in cls_name or "Spec" in cls_name:
            if hasattr(mod, "eps"):
                try:
                    eps_candidates.append(float(getattr(mod, "eps")))
                except Exception:
                    pass
            if hasattr(mod, "meta") and isinstance(getattr(mod, "meta"), dict):
                meta = getattr(mod, "meta")
                if "eps" in meta:
                    try:
                        eps_candidates.append(float(meta["eps"]))
                    except Exception:
                        pass

    if not eps_candidates:
        return None
    # Prefer the first; it should be unique in clean implementations
    return eps_candidates[0]


def test_A4_build_torch_model_accepts_act_net_and_reflects_overrides() -> None:
    """
    A4 acceptance:
      - build_torch_model MUST be built from the overridden ACT net (not rebuilt from template_name).
      - We verify this by checking that the torch model reflects overridden eps (InputSpec) if present.
      - Also requires API: build_torch_model(act_net, ...) must work.
    """
    torch = pytest.importorskip("torch")

    from act.pipeline.verification.confignet.builders import build_act_net, build_torch_model
    from act.pipeline.verification.confignet.configs import ModelConfig, SpecConfig
    from act.pipeline.verification.confignet.utils import extract_effective_spec

    # Build overridden ACT net
    m = ModelConfig(
        arch="mlp",
        template_name="mnist_robust_easy",
        input_shape=[1, 784],
        num_classes=10,
        seed=0,
    )
    s = SpecConfig(eps=0.02, norm="linf", targeted=False, true_label=3, assert_kind="TOP1_ROBUST")
    act_net = build_act_net(m, s, name="caseA4")

    eff = extract_effective_spec(act_net)
    assert eff["eps"] == pytest.approx(0.02)

    # Must support build_torch_model(act_net, ...)
    try:
        torch_model = build_torch_model(act_net)
    except TypeError as e:
        raise AssertionError(
            "build_torch_model must accept an overridden act_net as input (e.g., build_torch_model(act_net, device, dtype)). "
            "Do NOT rebuild from template_name."
        ) from e

    # Best-effort check: eps should be visible in torch model's input spec wrapper (if present)
    eps_in_torch = _extract_eps_from_torch_model(torch_model)
    assert eps_in_torch is not None, (
        "Torch model does not expose InputSpec eps; "
        "please ensure ACTToTorch preserves spec wrappers or exposes eps in InputSpec layer."
    )
    assert eps_in_torch == pytest.approx(0.02), (
        "Torch model eps does not match overridden act_net eps. "
        "This strongly suggests torch model was rebuilt from template_name rather than converted from act_net."
    )
