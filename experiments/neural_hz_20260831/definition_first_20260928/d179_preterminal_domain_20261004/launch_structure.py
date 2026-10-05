"""One-shot external 60-second supervisor; preserves an independent receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent / "results/d179_preterminal_domain_20261004_v1"
PYTHON = Path("/data1/Kane/miniconda3/bin/python")


def main():
    if sys.argv[1:] != ["--enabled"] or RUN.exists():
        raise ValueError("explicit --enabled and an unconsumed output version required")
    freeze = json.loads((HERE / "freeze.json").read_text())
    if freeze.get("schema") != "d179_preterminal_domain_v1":
        raise ValueError("freeze schema")
    for raw_path, sha in freeze["source_sha256"].items():
        path = Path(raw_path)
        if path.is_symlink() or path.parent != HERE or not path.is_file():
            raise ValueError("unexpected frozen source")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("source mismatch before launch: " + str(path))
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0")
    command = [str(PYTHON), "-B", str(HERE / "run_structure.py"), "--enabled"]
    started = time.monotonic()
    timed_out = False
    try:
        child = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=60, check=False)
        code, output, errors = child.returncode, child.stdout, child.stderr
    except subprocess.TimeoutExpired as error:
        # subprocess.run has killed and waited for this exact child. Never restart.
        code, output, errors, timed_out = 124, error.stdout or b"", error.stderr or b"", True
    receipt = dict(schema="d179_external_receipt_v1", command=command, external_timeout_s=60,
                   timed_out=timed_out, child_exit_code=code, supervisor_wall_s=time.monotonic() - started,
                   stdout_tail=output[-8192:].decode("utf-8", errors="replace"),
                   stderr_tail=errors[-8192:].decode("utf-8", errors="replace"),
                   output_truncated=len(output) > 8192 or len(errors) > 8192, formal_gain=0)
    if RUN.is_symlink() or not RUN.is_dir():
        print(json.dumps(receipt, sort_keys=True))
        raise RuntimeError("child did not create its exclusive result directory; receipt in tool output")
    receipt["internal_exit_present"] = (RUN / "exit.json").is_file()
    receipt["qualification_claimed_by_supervisor"] = False
    with (RUN / "EXTERNAL_RECEIPT.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True))
    return code if code >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
