"""Two-second stdlib-only reproduction; no project/native extension imports."""
import faulthandler
from fractions import Fraction
import sys
import time
import tracemalloc

faulthandler.enable(file=sys.stderr,all_threads=True)
tracemalloc.start(1)
faulthandler.dump_traceback_later(.1,repeat=True,file=sys.stderr)
started=time.monotonic();count=0
while time.monotonic()-started<2:
    value=Fraction(0)
    for _ in range(100):value=Fraction(1,3)*value+Fraction(0)
    count+=100
faulthandler.cancel_dump_traceback_later()
print(dict(completed=True,iterations=count,elapsed_s=time.monotonic()-started,
    traced=tracemalloc.get_traced_memory()),flush=True)
tracemalloc.stop()
