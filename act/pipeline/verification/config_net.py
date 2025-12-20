# config_net.py
#!/usr/bin/env python3
"""
ConfigNet sampler wrapper over existing example pool.

Week-1 MVP: deterministically sample an existing ACT example network and
its specifications using ModelFactory without generating new architectures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional

from act.pipeline.verification.model_factory import ModelFactory
from act.back_end.verifier import gather_input_spec_layers, get_assert_layer


@dataclass
class ConfigSample:
    """Bundled sample containing ACT net, PyTorch model, and specs."""

    name: str
    act_net: Any
    torch_model: Any
    input_specs: List[Any]
    assert_layer: Any
    metadata: dict


class ConfigNet:
    """
    Deterministic sampler for existing example networks.

    Uses ModelFactory to load ACT nets and PyTorch models from
    act/back_end/examples/examples_config.yaml.
    """

    def __init__(self, factory: Optional[ModelFactory] = None):
        self.factory = factory or ModelFactory()
        self._available = self.factory.list_networks()

    def sample(self, seed: Optional[int] = None, name: Optional[str] = None) -> ConfigSample:
        """
        Sample a configuration deterministically or fetch by name.

        Args:
            seed: Optional integer seed for deterministic sampling.
            name: Optional network name; if provided, overrides sampling.

        Returns:
            ConfigSample with ACT net, PyTorch model, specs, and metadata.
        """
        if name is not None:
            if name not in self._available:
                raise ValueError(f"Requested network '{name}' not found in available: {self._available}")
            chosen = name
        else:
            rng = random.Random(seed)
            # sort to keep deterministic ordering independent of YAML dict ordering
            ordered = sorted(self._available)
            if not ordered:
                raise ValueError("No networks available for sampling.")
            chosen = rng.choice(ordered)

        act_net = self.factory.get_act_net(chosen)
        torch_model = self.factory.create_model(chosen, load_weights=True)
        input_specs = gather_input_spec_layers(act_net)
        assert_layer = get_assert_layer(act_net)

        metadata = {
            "seed": seed,
            "source": "examples_config.yaml",
            "available_count": len(self._available),
            "chosen": chosen,
        }

        return ConfigSample(
            name=chosen,
            act_net=act_net,
            torch_model=torch_model,
            input_specs=input_specs,
            assert_layer=assert_layer,
            metadata=metadata,
        )
