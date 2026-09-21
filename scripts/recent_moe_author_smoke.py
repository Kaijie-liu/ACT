"""Exercise unmodified author architectures, NOT pretrained accuracy/certification."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["dual_rs", "robust_experts", "rome", "metamoe"])
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "code" if args.project == "dual_rs" else repo))
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    spec = {"project": args.project, "python": sys.version, "torch": torch.__version__,
            "seed": 20260921, "device": "cpu", "dtype": "float32",
            "trained_checkpoint": False, "dataset_loaded": False,
            "grade": "AUTHOR_MODEL_INIT_SMOKE"}
    if args.project == "dual_rs":
        from archs.cifar_resnet import resnet
        model = resnet(depth=110, num_classes=3)
        classes = 3
        spec["architecture"] = "author cifar_resnet110 variance estimator; normalization wrapper not included"
    elif args.project == "robust_experts":
        from src.models.nn.moe.resnet_conv_moe import ResNetConvMoE
        from src.models.nn.moe.routing import GlobalAvgLinearRoutingNetwork
        model = ResNetConvMoE(layers=18, num_channels=3, small_inputs=True,
                             num_classes=100, num_experts=4, k=2,
                             routing_layer_type=GlobalAvgLinearRoutingNetwork,
                             expert_capacity=None, balancing_loss_type="entropy",
                             balancing_loss=0.5, moe_layer_prefix="layer4")
        classes = 100
        spec["architecture"] = "author ResNet18 ConvMoE, layer4, E=4 k=2 GAP-FC"
    elif args.project == "metamoe":
        sys.path.insert(0, str(repo / "src/Vision_Transformer_Pytorch"))
        from small_expert import UltraVerifiableCNN
        model = UltraVerifiableCNN(num_classes=10)
        classes = 10
        spec["architecture"] = "author UltraVerifiableCNN expert only; not full MetaMoE"
    else:
        from models import build_model
        model = build_model("cifar10", pretrained=False, s=4.0, b=6.0)
        classes = 10
        spec["architecture"] = "author full CIFAR10 ViT RoME; code-default b=6 (paper b=2 unresolved)"
    model.eval()
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    assert list(logits.shape) == [2, classes]
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    spec.update({"parameters": sum(p.numel() for p in model.parameters()),
                 "logits_shape": list(logits.shape), "finite_forward": True,
                 "finite_input_gradient": True,
                 "input_gradient_l1": float(x.grad.abs().sum().detach())})
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
