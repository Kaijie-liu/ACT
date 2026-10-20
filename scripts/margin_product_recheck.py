"""Relocated stdlib slice replay; no ACT/checkpoint/native optimizer access."""
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
    from scoped_source.hz_request_store import require, decode
    import margin_product_cases as cases
    deadline = float(sys.argv[3])
    require(time.monotonic() < deadline and deadline-ENTRY <= 300, "portable same deadline")
    require(resource.getrlimit(resource.RLIMIT_AS) == (8*2**30, 8*2**30), "portable inherited AS")
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124))
    signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())
    raw = (root/"candidate.json").read_bytes()
    require(len(raw) <= 4*2**20 and hashlib.sha256(raw).hexdigest() == sys.argv[1], "portable candidate anchor")
    encoded_counts = (root/"continued.json").read_bytes()
    require(hashlib.sha256(encoded_counts).hexdigest() == sys.argv[2], "portable Work anchor")
    value = decode(raw)
    require(value["deadline"] == deadline, "portable request deadline binding")
    work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(decode(encoded_counts))
    before = sum(work.counts.values())
    receipts = cases.receive(value, work)
    require(not any(n in sys.modules for n in ("act", "torch", "numpy", "scipy")), "stdlib product reception")
    result = dict(status="PASS", receipts=receipts, work_counts=work.counts,
                  reception_work=sum(work.counts.values())-before, deadline=deadline,
                  complete_request=False, scalar_aggregate_checked=True,
                  elapsed_before_export=time.monotonic()-ENTRY)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(len(encoded) <= 2**20, "portable reserved output capacity")
    with (root/"receipt.json").open("xb") as stream:
        stream.write(encoded)
    work.tick()
    print(json.dumps(dict(status="PASS", seconds=time.monotonic()-ENTRY)), flush=True)
    work.tick()


if __name__ == "__main__":
    main()
