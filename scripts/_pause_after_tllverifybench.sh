#!/usr/bin/env bash
# Watcher: waits for "[tllverifybench_2023] DONE" in the abcrown runner's
# _run.log, then kills the whole process tree (PID 983111 and descendants)
# to free the GPU. Resume later with the same runner — it will skip
# completed instances (which have non-empty .result files).
set -u
LOG=/data1/Kane/ACT/audit_results/abcrown_nopgd_20260525/_run.log
PID=983111
STAMP=/data1/Kane/ACT/audit_results/abcrown_nopgd_20260525/_paused_after_tllverifybench.flag
echo "[watcher] waiting for tllverifybench_2023 DONE in $LOG" >&2
while kill -0 "$PID" 2>/dev/null; do
  if grep -q '\[tllverifybench_2023\] DONE' "$LOG" 2>/dev/null; then
    echo "[watcher] caught DONE — killing tree of PID=$PID" >&2
    # Kill descendants first, then the bash itself.
    pkill -TERM -P "$PID" 2>/dev/null
    sleep 2
    pkill -KILL -P "$PID" 2>/dev/null
    kill -TERM "$PID" 2>/dev/null
    sleep 2
    kill -KILL "$PID" 2>/dev/null
    date -Iseconds >"$STAMP"
    echo "[watcher] paused at $(cat $STAMP)" >&2
    exit 0
  fi
  sleep 10
done
echo "[watcher] PID $PID already gone, nothing to do" >&2
