"""Offline saved-record verification, never an online acceptance repair."""
import time
ENTRY = time.monotonic()
import json
from pathlib import Path
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
RUN = Path("/data1/Kane/MOE/results/margin_support_20261021_r1")


def audit():
    sys.path.append(str(ROOT))
    from scoped_proof.io import load, sha, save
    from scoped_source.hz_request_store import Work, compact, require
    from scoped_source import check_margin_support as check
    from fractions import Fraction as F
    freeze_path = ROOT/"docs/margin_support_freeze_20261021.json"
    freeze_hash = sha(freeze_path)
    freeze = load(freeze_path, freeze_hash, 131072)
    terminal_hash = sha(RUN/"result.json") if (RUN/"result.json").is_file() else None
    terminal = (load(RUN/"result.json", terminal_hash, 2**20) if terminal_hash else
                dict(status="NO_TERMINAL", freeze_sha256=freeze_hash))
    require(terminal["freeze_sha256"] == freeze_hash, "terminal freeze")
    for path, digest in freeze["files"].items():
        require(sha(ROOT/path) == digest, "frozen source drift")
    worker_hash = sha(RUN/"worker.json") if (RUN/"worker.json").is_file() else None
    worker = load(RUN/"worker.json", worker_hash, 2**20) if worker_hash else None
    if worker:
        require(worker["freeze_sha256"] == freeze_hash, "worker freeze")
    if "worker" in terminal:
        require(terminal["worker"]["sha256"] == worker_hash, "terminal candidate binding")
    work = Work(ENTRY+300)
    receipts = []
    for item in worker["certificates"] if worker else []:
        frozen = freeze["sources"][item["label"]]
        require(item["source_sha256"] == frozen["sha256"], "saved source identity")
        receipt = check.check(compact(item["source"]), item["certificate"],
                              expected_source=frozen["sha256"], expected_mode=frozen["mode"],
                              expected_cells=frozen["cells"], work=work)
        require(receipt == item["receipt"], "saved exact receipt")
        receipts.append(dict(label=item["label"], **receipt))
    slice_checked = False
    for row in worker["cases"] if worker else []:
        if row["name"] == "nonzero_dual_constraints":
            detail = row["detail"]
            raw = compact(detail["slice_source"])
            source = check.inspect(raw, freeze["sources"]["slice_support"]["sha256"], work)
            require(check.checked_support(source, F(1, 4), F(0), detail["slice_support"], work)
                    == F(1, 16), "saved slice lower")
            slice_checked = True
    complete = (worker is not None and worker["status"] == "PASS"
                and [r["name"] for r in worker["cases"]] == freeze["cases"]
                and all(r["status"] == "PASS" for r in worker["cases"])
                and [r["label"] for r in receipts] == freeze["certificates"]
                and slice_checked)
    if terminal["status"] == "PASS":
        require(complete and terminal["online_reception"]["receipts"] == receipts,
                "complete accepted evidence")
        session = terminal["session"]
        require(session["status"] == "COMPLETED" and session["returncode"] == 0
                and session["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
                and session["execution_error"] is None and session["cleanup_error"] is None,
                "online session acceptance")
        require(terminal["deadline"] == terminal["entry"]+300
                and worker["deadline"] == terminal["deadline"]
                and worker["worker_deadline"] == terminal["entry"]+240
                and session["run_deadline"] == terminal["entry"]+250
                and session["cleanup_deadline"] == terminal["entry"]+270
                and session["finished_at"] < session["cleanup_deadline"], "original deadline chain")
        spent, final = worker["work_counts"], terminal["online_reception"]["work_counts"]
        require(set(spent) == set(final) == {"row_visits", "coefficient_visits", "arithmetic"}
                and all(type(v) is int and v >= 0 for v in [*spent.values(), *final.values()])
                and all(final[k] >= spent[k] for k in spent)
                and sum(final.values()) == terminal["online_reception"]["total_work"]
                and sum(final.values())-sum(spent.values()) == terminal["online_reception"]["reception_work"],
                "continued Work audit")
        require(terminal["caps"] == freeze["caps"]
                and worker["sampled_worker_rss"] <= freeze["caps"]["rss_bytes"]
                and session["sampled_session_plus_caller_rss"] <= freeze["caps"]["rss_bytes"]
                and all(0 < r["bytes"] <= freeze["caps"]["rss_bytes"]
                        for r in terminal["controller_rss_samples"])
                and terminal["bytes_before_terminal"]+2*(RUN/"result.json").stat().st_size+32768
                    <= freeze["caps"]["artifact_bytes"], "frozen resource audit")
        require(terminal["online_reception"]["total_work"] <= freeze["caps"]["work"]
                and terminal["final_fds"] == terminal["initial_fds"]
                and terminal["elapsed_before_terminal_export"] < 300, "online costs/FD")
    result = dict(status="PASS_SAVED_EVIDENCE", online_status_unchanged=terminal["status"],
                  complete_roster=complete, offline_only=True, receipts=receipts,
                  slice_checked=slice_checked, offline_work=sum(work.counts.values()),
                  seconds_before_archive=time.monotonic()-ENTRY,
                  result_sha256=terminal_hash, worker_sha256=worker_hash,
                  freeze_sha256=freeze_hash, no_network_or_solver=True,
                  controller_exit_requires_external_command_receipt=True)
    require(time.monotonic() < ENTRY+300, "offline deadline")
    bundle = dict(terminal=terminal, worker=worker, offline_audit=result)
    raw = compact(bundle)
    require(len(raw) <= 2**20, "compact archive")
    saved = save(ROOT/"docs/margin_support_records_20261021.json", bundle)
    print(json.dumps(dict(archive=saved, audit=result), sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] != ["--archive-saved-only"]:
        raise SystemExit("saved-only audit opt in required")
    audit()
