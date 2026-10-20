"""Portable A2 original-coefficient reception, standard library only."""
import time
ENTRY = time.monotonic()
from pathlib import Path
import resource
import signal
import sys


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from scoped_proof.io import load, save
    from scoped_source.development_work import DevelopmentWork, POLICY
    from scoped_source.hz_request_store import require
    import margin_a2_cases as cases
    deadline = float(sys.argv[3])
    require(0 < deadline-time.monotonic() <= 300, "same portable deadline")
    require(resource.getrlimit(resource.RLIMIT_AS) == (8*2**30, 8*2**30), "inherited AS")
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124))
    signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())
    value = load(root/"worker.json", sys.argv[1], 2**18)
    counts = load(root/"continued.json", sys.argv[2], 4096)
    work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(counts)
    before = sum(work.counts.values())
    result = cases.receive(root, value, work)
    require(not any(k in sys.modules for k in ("act", "torch", "numpy", "scipy")), "stdlib reception")
    result.update(status="PASS", deadline=deadline, entry=ENTRY,
                  reception_work=sum(work.counts.values())-before,
                  elapsed_before_export=cases.duration(ENTRY, time.monotonic(), deadline, "portable"))
    from scoped_source.hz_request_store import compact
    require(len(compact(result)) <= 2**18, "portable bounded receipt")
    save(root/"receipt.json", result)
    work.tick()
    print("A2_PORTABLE_PASS", flush=True)
    work.tick()


if __name__ == "__main__":
    main()
