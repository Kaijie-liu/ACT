import sys
import warnings

from act.pipeline.cli import build_parser


def _no_new_heavy_imports(before, after):
    new = after - before
    assert not any(m == "torchvision" or m.startswith("torchvision.") for m in new)
    assert not any(m == "torch" or m.startswith("torch.") for m in new)
    assert not any(m == "transformers" or m.startswith("transformers.") for m in new)


warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    module="torchvision.io.image",
)


def test_confignet_source_default_pool():
    before = set(sys.modules)
    parser = build_parser()
    after = set(sys.modules)
    _no_new_heavy_imports(before, after)
    args = parser.parse_args(["--validate-verifier"])
    assert args.confignet_source == "pool"


def test_confignet_source_runtime_flag():
    before = set(sys.modules)
    parser = build_parser()
    after = set(sys.modules)
    _no_new_heavy_imports(before, after)
    args = parser.parse_args(["--validate-verifier", "--confignet-source", "runtime"])
    assert args.confignet_source == "runtime"
