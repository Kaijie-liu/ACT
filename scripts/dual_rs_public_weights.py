"""Fetch only the two public CIFAR base models, no execution/training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen


VIT = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
DIFFUSION = "https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt"


def fetch(url, target, expected=None):
    start = time.monotonic()
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists() or partial.exists():
        raise ValueError("download target exists; no silent reuse or replacement")
    h, size = hashlib.sha256(), 0
    with urlopen(url, timeout=60) as response, partial.open("xb") as stream:
        while data := response.read(1024 * 1024):
            size += len(data)
            if size > 1024 ** 3:
                raise ValueError("per-file 1GiB ceiling exceeded")
            stream.write(data)
            h.update(data)
    digest = h.hexdigest()
    if expected and digest != expected:
        raise ValueError("published LFS hash mismatch")
    partial.rename(target)
    return {"url": url, "filename": target.name, "bytes": size,
            "sha256": digest, "published_sha256": expected,
            "elapsed_seconds": time.monotonic() - start}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with urlopen(f"https://huggingface.co/api/models/{VIT}?blobs=true", timeout=60) as r:
        metadata = json.load(r)
    revision = metadata["sha"]
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("invalid model revision")
    (a.output / "vit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = {r["rfilename"]: r for r in metadata["siblings"]}
    weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    if weight not in files:
        raise ValueError("no known weight format")
    target = a.output / "vit"
    target.mkdir()
    downloaded = [fetch(DIFFUSION, a.output / "cifar10_uncond_50M_500K.pt")]
    for name in ["config.json", "preprocessor_config.json", weight]:
        expected = files[name].get("lfs", {}).get("sha256")
        row = fetch(f"https://huggingface.co/{VIT}/resolve/{revision}/{name}", target / name, expected)
        row["filename"] = "vit/" + name
        downloaded.append(row)
    result = {"stage": "public_cifar_base_models_only", "vit_repository": VIT,
              "vit_revision": revision, "files": downloaded,
              "sigma_estimator_checkpoint_available": False,
              "elapsed_seconds": time.monotonic() - start,
              "scope": "download identities; no model loading, accuracy or fresh certification"}
    with (a.output / "manifest.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
