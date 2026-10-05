from __future__ import annotations

import hashlib
import json

import pytest

from seal_trial9_after_exit_v1 import (
    EXPECTED_PROVENANCE,
    EXPECTED_RESULT,
    seal_after_exit,
)


PID = 251298
START_TICKS = 698719726


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _paths(tmp_path):
    root = tmp_path / "experiment"
    results = root / "results"
    results.mkdir(parents=True)
    proc = tmp_path / "proc"
    proc.mkdir()
    return (
        root,
        results / "trial9.json",
        results / "trial9.log",
        root / "trial9_exit_seal_v1.json",
        proc,
    )


def _complete_result() -> dict[str, object]:
    return {
        **EXPECTED_RESULT,
        "provenance": EXPECTED_PROVENANCE,
        "census_stop_reached": True,
        "verdict": "UNKNOWN",
    }


def _seal(root, result, log, sidecar, proc):
    return seal_after_exit(
        experiment_root=root,
        result_path=result,
        log_path=log,
        sidecar_path=sidecar,
        proc_root=proc,
        pid=PID,
        start_ticks=START_TICKS,
        poll_seconds=0.001,
    )


def _write_proc_stat(proc, pid, start_ticks):
    process = proc / str(pid)
    process.mkdir()
    fields = ["R"] + ["0"] * 18 + [str(start_ticks)] + ["0"] * 30
    (process / "stat").write_text(f"{pid} (python worker) {' '.join(fields)}\n")


def test_normal_result_is_validated_hashed_and_exclusively_published(tmp_path):
    root, result, log, sidecar, proc = _paths(tmp_path)
    result_bytes = (json.dumps(_complete_result()) + "\n").encode()
    log_bytes = b"worker log\n"
    result.write_bytes(result_bytes)
    log.write_bytes(log_bytes)
    record = _seal(root, result, log, sidecar, proc)
    assert record["classification"] == "COMPLETE_JSON_VALIDATED"
    assert record["verdict"] == "UNKNOWN"
    assert record["formal_gain"] == 0
    assert record["exit_status"] == "UNRECOVERABLE"
    assert record["process_binding"]["observation"] == "ORIGINAL_PID_EXITED"
    assert record["result_artifact"]["sha256"] == _sha256(result_bytes)
    assert record["log_artifact"]["sha256"] == _sha256(log_bytes)
    published = json.loads(sidecar.read_text())
    claimed = published.pop("seal_payload_sha256")
    canonical = json.dumps(
        published, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    assert claimed == _sha256(canonical)


def test_truncated_result_is_hashed_without_guessing(tmp_path):
    root, result, log, sidecar, proc = _paths(tmp_path)
    raw = b'{"schema":"neural_hz_shadow_v1"'
    result.write_bytes(raw)
    log.write_text("log\n")
    record = _seal(root, result, log, sidecar, proc)
    assert record["classification"] == "TRUNCATED_OR_CORRUPT"
    assert record["result_artifact"]["sha256"] == _sha256(raw)
    assert record["verdict"] is None
    assert record["exit_status"] == "UNRECOVERABLE"
    assert record["formal_gain"] == 0


def test_missing_result_is_recorded_without_guessing(tmp_path):
    root, result, log, sidecar, proc = _paths(tmp_path)
    log.write_text("log only\n")
    record = _seal(root, result, log, sidecar, proc)
    assert record["classification"] == "NO_RESULT_JSON"
    assert record["result_artifact"]["exists"] is False
    assert record["verdict"] is None
    assert record["exit_status"] == "UNRECOVERABLE"
    assert record["formal_gain"] == 0


def test_pid_reuse_is_not_mistaken_for_original_process(tmp_path):
    root, result, log, sidecar, proc = _paths(tmp_path)
    result.write_text(json.dumps(_complete_result()))
    log.write_text("log\n")
    _write_proc_stat(proc, PID, START_TICKS + 1)
    record = _seal(root, result, log, sidecar, proc)
    binding = record["process_binding"]
    assert binding["observation"] == "PID_REUSED"
    assert binding["observed_start_ticks"] == START_TICKS + 1


def test_existing_sidecar_is_rejected_without_overwrite(tmp_path):
    root, result, log, sidecar, proc = _paths(tmp_path)
    result.write_text(json.dumps(_complete_result()))
    log.write_text("log\n")
    sidecar.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _seal(root, result, log, sidecar, proc)
    assert sidecar.read_bytes() == b"existing"


@pytest.mark.parametrize("escaped", ["result", "log", "sidecar"])
def test_path_escape_is_rejected(tmp_path, escaped):
    root, result, log, sidecar, proc = _paths(tmp_path)
    result.write_text(json.dumps(_complete_result()))
    log.write_text("log\n")
    outside = tmp_path / f"outside-{escaped}"
    values = {"result": result, "log": log, "sidecar": sidecar}
    values[escaped] = outside
    with pytest.raises(ValueError, match="escapes experiment root"):
        _seal(root, values["result"], values["log"], values["sidecar"], proc)
