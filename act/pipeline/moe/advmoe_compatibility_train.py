"""Run the released AdvMoE CLI under an explicit numerical compatibility bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from typing import Any

from act.pipeline.moe.advmoe_softmax_underflow_bridge import (
    SoftmaxUnderflowGradientBridge,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _router_kl_target_softmax_callsite(script: Path) -> tuple[Path, int]:
    """Bind the bridge to the released router-KL target and nothing else."""

    needle = "F.log_softmax(adv_scores, dim=1), F.softmax(scores, dim=1)"
    matches = [
        line_number
        for line_number, line in enumerate(
            script.read_text(encoding="utf-8").splitlines(), 1
        )
        if needle in line
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "released router-KL softmax callsite is absent or ambiguous"
        )
    return script.resolve(), matches[0]


def run(
    workspace: Path,
    official_source: Path,
    summary: Path,
    official_arguments: list[str],
) -> None:
    workspace = workspace.resolve()
    official_source = official_source.resolve()
    summary = summary.resolve()
    official_source.relative_to(workspace)
    summary.relative_to(workspace)
    if summary.exists():
        raise FileExistsError(summary)
    if _git(official_source, "status", "--porcelain=v1"):
        raise RuntimeError("official source clone is dirty")
    script = official_source / "train_moe.py"
    if not script.is_file():
        raise RuntimeError("released train_moe.py is missing")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(official_source))
    sys.argv = [str(script), *official_arguments]
    target_callsite = _router_kl_target_softmax_callsite(script)
    bridge = SoftmaxUnderflowGradientBridge(
        allowed_callsites=(target_callsite,)
    )
    started = time.time()
    status = "RUNNING"
    error = None
    try:
        with bridge:
            runpy.run_path(str(script), run_name="__main__")
        status = "PASSED"
    except BaseException as caught:
        status = "FAILED"
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        _atomic_json(
            summary,
            {
                "schema_version": 2,
                "status": status,
                "label": "official-code numerical-compatibility variant; callsite-scoped softmax-underflow gradient bridge",
                "official_source": {
                    "commit": _git(official_source, "rev-parse", "HEAD"),
                    "tree": _git(official_source, "rev-parse", "HEAD^{tree}"),
                    "clean_after": not bool(
                        _git(official_source, "status", "--porcelain=v1")
                    ),
                },
                "bridge": bridge.summary(),
                "runtime_seconds": time.time() - started,
                "error": error,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("official_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    official_arguments = arguments.official_arguments
    if official_arguments[:1] == ["--"]:
        official_arguments = official_arguments[1:]
    run(
        arguments.workspace,
        arguments.official_source,
        arguments.summary,
        official_arguments,
    )


if __name__ == "__main__":
    main()
