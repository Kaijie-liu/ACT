"""Bounded, non-overwriting author-artifact deployment receipts (not certification).

No package installation, checkpoint download, remote logging or training grid.
Run the supervisor with act-py312. Child interpreters are explicitly recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_identity(repo):
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    files = git("ls-files", "-z").split("\0")
    hashes = {name: sha256(repo / name) for name in files if name and (repo / name).is_file()}
    gitlinks = [line for line in git("ls-files", "--stage").splitlines()
                if line.startswith("160000 ")]
    return {"path": str(repo), "head": git("rev-parse", "HEAD"),
            "origin": git("remote", "get-url", "origin"),
            "status": git("status", "--porcelain"), "tracked_sha256": hashes,
            "submodule_gitlinks": gitlinks}


def supervise(command, cwd, output, seconds, grade, repo=None, cpu_only=True):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite deadline required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before = git_identity(repo) if repo else None
    if before and before["status"]:
        raise ValueError("author checkout must be clean")
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "MPLBACKEND": "Agg"})
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TMPDIR"):
        target = output / key.lower()
        target.mkdir()
        env[key] = str(target)
    state, error, exit_code = "ERROR", None, None
    with (output / "stdout.txt").open("w") as out, (output / "stderr.txt").open("w") as err:
        try:
            if time.monotonic() - start >= seconds:
                raise subprocess.TimeoutExpired(command, seconds)
            p = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
                                 start_new_session=True)
            try:
                exit_code = p.wait(timeout=max(0.001, seconds - (time.monotonic() - start)))
                state = "COMPLETED" if exit_code == 0 else "ERROR"
            except subprocess.TimeoutExpired:
                # Only this newly-created process group, never another user's job.
                os.killpg(p.pid, signal.SIGKILL)
                exit_code = p.wait()
                state = "TIMEOUT"
        except subprocess.TimeoutExpired:
            state = "TIMEOUT"
            error = "preflight exhausted deadline; child not started"
        except Exception as exc:
            error = repr(exc)
    execution_seconds = time.monotonic() - start
    after = git_identity(repo) if repo else None
    identity_match = before == after
    if not identity_match:
        state = "SOURCE_CHANGED"
    receipt = {"schema": 1, "command": command, "cwd": str(Path(cwd).resolve()),
               "status": state, "exit_code": exit_code, "error": error,
               "evidence_grade": grade, "deadline_seconds": seconds,
               "execution_including_preflight_seconds": execution_seconds,
               "total_with_postflight_seconds": time.monotonic() - start,
               "postflight_in_execution_budget": False,
               "cpu_only": cpu_only, "threads": 2,
               "source_before": before, "source_unchanged": identity_match,
               "command_file_sha256": {
                   str((Path(cwd) / v).resolve()): sha256(Path(cwd) / v)
                   for v in command if (Path(cwd) / v).is_file()},
               "stdout_sha256": sha256(output / "stdout.txt"),
               "stderr_sha256": sha256(output / "stderr.txt"),
               "claim": "Deployment receipt only; not a reproduced accuracy or a proof."}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cwd", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--grade", required=True,
                   choices=["AUTHOR_RESULT_REPLAY", "AUTHOR_MODEL_INIT_SMOKE",
                            "AUTHOR_CHECKPOINT_SMOKE", "ENTRYPOINT_PROBE",
                            "DEPENDENCY_SETUP", "DATA_PREPARATION", "AUTHOR_COMPONENT_CONTROL"])
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("missing child command")
    result = supervise(command, args.cwd, args.output, args.seconds, args.grade, args.repo)
    print(json.dumps({k: v for k, v in result.items() if k != "source_before"}, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
