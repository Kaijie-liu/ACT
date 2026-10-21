"""Relocated structural reception only, with no ACT or numerical imports."""
import time
ENTRY = time.monotonic()
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from scoped_source.development_work import DevelopmentWork, POLICY
    from scoped_source.native_capture import require, canonical
    from b1_native_capture_cases import replay
    deadline = float(sys.argv[3])
    require(ENTRY < deadline and deadline-ENTRY <= 300, "portable same deadline")
    require(resource.getrlimit(resource.RLIMIT_AS) == (8*2**30, 8*2**30), "portable inherited AS")
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124))
    signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())
    def read(name, digest):
        path = root/name
        require(not path.is_symlink() and path.stat().st_size <= 2**20, "bounded portable metadata")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "portable identity")
        return json.loads(raw)
    value = read("worker.json", sys.argv[1])
    counts = read("continued.json", sys.argv[2])
    work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(counts)
    receipts = replay(root, value, work)
    require(not any(k in sys.modules for k in ("act", "numpy", "scipy", "torch")), "stdlib only")
    result = dict(status="PASS", deadline=deadline, work_counts=work.counts, receipts=receipts,
                  elapsed_before_export=time.monotonic()-ENTRY, checked_feasibility=False,
                  complete_request=False)
    raw = canonical(result)
    require(len(raw) <= 2**18, "reserved portable output")
    with (root/"receipt.json").open("xb") as stream:
        stream.write(raw)
    work.tick()
    print(json.dumps(dict(status="PASS", elapsed=time.monotonic()-ENTRY)), flush=True)


if __name__ == "__main__":
    main()
