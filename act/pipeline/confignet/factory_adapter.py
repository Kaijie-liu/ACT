#!/usr/bin/env python3
#===- act/pipeline/confignet/factory_adapter.py - Adapter ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Adapter utilities for using ConfigNet instances in verification pipelines.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from act.pipeline.verification.torch2act import TorchToACT
from act.pipeline.confignet.builders import build_generated_instance
from act.pipeline.confignet.input_sampling import sample_feasible_inputs
from act.pipeline.confignet.schema import InstanceSpec
from act.pipeline.confignet.seeds import derive_seed


class ConfignetFactoryAdapter:
    """
    Adapter that builds per-instance cases and can emulate ModelFactory APIs.
    """

    def __init__(self, *, device: str, dtype: torch.dtype):
        self._device = device
        self._dtype = dtype
        self._instances: Dict[str, InstanceSpec] = {}
        self._generated: Dict[str, Any] = {}
        self._act_nets: Dict[str, Any] = {}
        self._call_counts: Dict[str, int] = {}
        self._seed_inputs_by_name: Dict[str, int] = {}
        self._strict_input = True

    def build_case(
        self,
        instance_cfg: InstanceSpec,
        *,
        seed_inputs: int,
        num_samples: int = 1,
        deterministic_algos: bool = False,
        strict_input: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a single case for L1/L2 orchestration.
        """
        if not isinstance(instance_cfg, InstanceSpec):
            raise TypeError(f"instance_cfg must be InstanceSpec, got {type(instance_cfg)}")

        generated = build_generated_instance(
            instance_cfg,
            device=self._device,
            dtype=self._dtype,
            deterministic_algos=deterministic_algos,
        )
        act_net = None
        act_build_error = None
        try:
            act_net = TorchToACT(generated.wrapped_model).run()
        except Exception as e:
            act_build_error = str(e)

        xs = sample_feasible_inputs(
            instance_cfg,
            num_samples=int(num_samples),
            seed=int(seed_inputs),
            device=self._device,
            dtype=self._dtype,
            strict_input=bool(strict_input),
        )

        instance_meta = {
            "net_family": str(getattr(instance_cfg.family, "value", instance_cfg.family)),
            "arch": instance_cfg.model_cfg.to_dict(),
            "spec": {
                "input_spec": instance_cfg.input_spec.to_dict(),
                "output_spec": instance_cfg.output_spec.to_dict(),
            },
            "input_shape": list(instance_cfg.model_cfg.input_shape),
            "eps": float(instance_cfg.input_spec.eps or 0.0),
        }

        return {
            "generated": generated,
            "torch_model": generated.wrapped_model,
            "act_model": act_net,
            "act_build_error": act_build_error,
            "inputs": xs,
            "spec": {
                "input_spec": instance_cfg.input_spec.to_dict(),
                "output_spec": instance_cfg.output_spec.to_dict(),
            },
            "instance_meta": instance_meta,
        }

    def configure_for_validation(
        self,
        instances: List[InstanceSpec],
        generated: List[Any],
        *,
        seed_inputs_by_name: Optional[Dict[str, int]] = None,
        act_nets_by_name: Optional[Dict[str, Any]] = None,
        strict_input: bool = True,
    ) -> None:
        self._instances = {inst.instance_id: inst for inst in instances}
        self._generated = {gi.instance_spec.instance_id: gi for gi in generated}
        self._act_nets = {}
        self._call_counts = {}
        self._seed_inputs_by_name = seed_inputs_by_name or {}
        self._strict_input = bool(strict_input)

        if act_nets_by_name:
            self._act_nets = dict(act_nets_by_name)
            for name in self._generated.keys():
                self._call_counts[name] = 0
            return

        for name, gen in self._generated.items():
            converter = TorchToACT(gen.wrapped_model)
            self._act_nets[name] = converter.run()
            self._call_counts[name] = 0

    def list_networks(self) -> List[str]:
        return list(self._instances.keys())

    def get_act_net(self, name: str):
        return self._act_nets[name]

    def create_model(self, name: str, load_weights: bool = True):
        return self._generated[name].wrapped_model

    def generate_test_input(self, name: str, test_case: str = "random") -> torch.Tensor:
        inst = self._instances[name]
        count = self._call_counts[name]
        self._call_counts[name] = count + 1
        base_seed = self._seed_inputs_by_name.get(name, int(inst.seed))
        seed = derive_seed(int(base_seed), count, "inputs|level2")
        xs = sample_feasible_inputs(
            inst,
            num_samples=1,
            seed=seed,
            device=self._device,
            dtype=self._dtype,
            strict_input=bool(self._strict_input),
        )
        return xs[0]
