"""Semantics-preserving adapters for the pinned AdvMoE CIFAR architecture."""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from act.pipeline.moe.advmoe_architecture_audit import _attach_router, _external_models


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _initialize_main_model_like_released_source(model: nn.Module) -> None:
    """Replay the pinned source's main-model initialization before router draw."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                module.weight.normal_(0, 0.01)
                module.bias.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.fill_(1)
                module.bias.zero_()


def construct_official_init(seed: int = 1234) -> tuple[nn.Module, nn.Module, type]:
    """Construct the released README configuration in its RNG consumption order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    with _external_models() as (resnet, router_module, moe_layer):
        model = resnet.resnet18_cifar_moe(
            num_classes=10, n_expert=2, ratio=0.5
        )
        _initialize_main_model_like_released_source(model)
        router = router_module.build_router(num_experts=2)
        _attach_router(model, router)
        moe_type = moe_layer.MoEConv
    return model, router, moe_type


def _static_subsample_shortcut(block: nn.Module) -> nn.Conv2d:
    in_channels = int(block.conv1.in_channels)
    out_channels = int(block.conv1.out_channels)
    if out_channels < in_channels or block.conv1.stride != (2, 2):
        raise ValueError("unsupported AdvMoE router shortcut")
    shortcut = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=2, bias=False
    ).to(device=block.conv1.weight.device, dtype=block.conv1.weight.dtype)
    offset = (out_channels - in_channels) // 2
    with torch.no_grad():
        shortcut.weight.zero_()
        for channel in range(in_channels):
            shortcut.weight[offset + channel, channel, 0, 0] = 1
    shortcut.weight.requires_grad_(False)
    return shortcut


class CrownCompatibleAdvMoeRouter(nn.Module):
    """Fixed-shape lowering of the official 32x32 router.

    It replaces two source constructs that current auto_LiRPA cannot consume:
    strided tensor slices in projection shortcuts and dynamically sized global
    average pooling.  Both replacements are exact for `[B,3,32,32]` inputs.
    """

    def __init__(self, router: nn.Module):
        super().__init__()
        self.conv1 = copy.deepcopy(router.conv1)
        self.bn1 = copy.deepcopy(router.bn1)
        self.layer1 = copy.deepcopy(router.layer1)
        self.layer2 = copy.deepcopy(router.layer2)
        self.layer3 = copy.deepcopy(router.layer3)
        self.pool = nn.AvgPool2d(8)
        self.fc = copy.deepcopy(router.fc)
        for stage in (self.layer1, self.layer2, self.layer3):
            for block in stage:
                if block.conv1.stride == (2, 2):
                    block.shortcut = _static_subsample_shortcut(block)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return self.fc(out.reshape(-1, 64))

    @staticmethod
    def validate_input_shape(inputs: torch.Tensor) -> None:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 32, 32):
            raise ValueError("AdvMoE router adapter requires [B,3,32,32]")


def _static_expert_convolution(module: nn.Module, route: int) -> nn.Conv2d:
    if route < 0 or route >= int(module.n_expert):
        raise ValueError("route index outside expert range")
    width = int(module.expert_width)
    static = nn.Conv2d(
        in_channels=int(module.in_channels),
        out_channels=width,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
    ).to(device=module.weight.device, dtype=module.weight.dtype)
    start = route * width
    end = start + width
    with torch.no_grad():
        static.weight.copy_(module.weight[start:end])
        if static.bias is not None:
            static.bias.copy_(module.bias[start:end])
    return static


def _replace_moe_convolutions(
    parent: nn.Module, *, route: int, moe_type: type
) -> int:
    replaced = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, moe_type):
            setattr(parent, name, _static_expert_convolution(child, route))
            replaced += 1
        else:
            replaced += _replace_moe_convolutions(
                child, route=route, moe_type=moe_type
            )
    return replaced


def specialize_advmoe_path(
    model: nn.Module, route: int, moe_type: type
) -> tuple[nn.Module, int]:
    """Remove dynamic dispatch and specialize every hidden MoE conv to one route."""
    specialized = copy.deepcopy(model)
    replaced = _replace_moe_convolutions(
        specialized, route=int(route), moe_type=moe_type
    )
    specialized.router = None
    if replaced != 16:
        raise RuntimeError(f"expected 16 routed convolutions, replaced {replaced}")
    return specialized, replaced


class CrownCompatibleAdvMoePath(nn.Module):
    """Fixed-shape lowering of one already specialized CIFAR-10 path."""

    def __init__(self, specialized: nn.Module):
        super().__init__()
        if specialized.router is not None:
            raise ValueError("AdvMoE path must be specialized before lowering")
        self.conv1 = copy.deepcopy(specialized.conv1)
        self.bn1 = copy.deepcopy(specialized.bn1)
        self.layer1 = copy.deepcopy(specialized.layer1)
        self.layer2 = copy.deepcopy(specialized.layer2)
        self.layer3 = copy.deepcopy(specialized.layer3)
        self.layer4 = copy.deepcopy(specialized.layer4)
        self.pool = nn.AvgPool2d(4)
        self.linear = copy.deepcopy(specialized.linear)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 32, 32):
            raise ValueError("AdvMoE path adapter requires [B,3,32,32]")
        out = torch.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        return self.linear(out.reshape(out.shape[0], -1))


def path_adapter_equivalence(
    specialized: nn.Module,
    inputs: torch.Tensor,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    """Compare the literal static path with its fixed-pooling CROWN lowering."""

    specialized = specialized.eval()
    adapted = CrownCompatibleAdvMoePath(specialized).eval()
    with torch.no_grad():
        original = specialized(inputs)
        lowered = adapted(inputs)
    difference = (original - lowered).abs()
    return {
        "outputs_equal": bool(torch.equal(original, lowered)),
        "outputs_close": bool(torch.allclose(original, lowered, atol=atol, rtol=rtol)),
        "predictions_equal": bool(
            torch.equal(original.argmax(dim=1), lowered.argmax(dim=1))
        ),
        "max_abs_error": float(difference.max().item()),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def adapter_equivalence(
    router: nn.Module, inputs: torch.Tensor
) -> dict[str, Any]:
    router = router.eval()
    adapted = CrownCompatibleAdvMoeRouter(router).eval()
    adapted.validate_input_shape(inputs)
    with torch.no_grad():
        original = router(inputs)
        lowered = adapted(inputs)
    difference = (original - lowered).abs()
    return {
        "outputs_equal": bool(torch.equal(original, lowered)),
        "max_abs_error": float(difference.max().item()),
        "routes_equal": bool(
            torch.equal(original.argmax(dim=1), lowered.argmax(dim=1))
        ),
    }
