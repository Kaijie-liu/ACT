"""Create an isolated copy of public benchmark data; no model/solver calls."""
import argparse
import json
from pathlib import Path
import shutil
import time

from recent_moe_deployment import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--existing-cifar", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    a.root.mkdir(parents=True, exist_ok=False)
    source = a.existing_cifar.resolve() / "cifar-10-batches-py"
    before = {str(x.relative_to(source)): sha256(x) for x in source.rglob("*") if x.is_file()}
    if not before:
        raise ValueError("existing CIFAR source missing")
    shutil.copytree(source, a.root / "cifar-10-batches-py")
    import torchvision
    cifar = torchvision.datasets.CIFAR10(str(a.root), train=False, download=False)
    mnist = torchvision.datasets.MNIST(str(a.root), train=False, download=True)
    if len(cifar) != 10000 or len(mnist) != 10000:
        raise ValueError("unexpected test sizes")
    if before != {str(x.relative_to(source)): sha256(x) for x in source.rglob("*") if x.is_file()}:
        raise ValueError("existing source changed")
    manifest = {"cifar_test_size": len(cifar), "mnist_test_size": len(mnist),
                "cifar_source": str(source), "cifar_source_unchanged": True,
                "torchvision": torchvision.__version__, "dataset_validation": "author library integrity checks",
                "files": {str(x.relative_to(a.root)): {"sha256": sha256(x), "bytes": x.stat().st_size}
                          for x in sorted(a.root.rglob("*")) if x.is_file()},
                "elapsed_seconds": time.monotonic() - start,
                "scope": "public data installation only; no sample selected by verification outcomes"}
    with (a.root / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
