"""Fixed compiler-preparation probe, no test calls or Neural-HZ operations."""
import hashlib
import json
from pathlib import Path
import statistics
import time
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import code_image,code_tree,code_key,compiled
from experiments.neural_hz_20260831.c50_cleanup_compile_v1 import compile_cleanup
from experiments.neural_hz_20260831.c51_source_custody_v1 import custody


def probe(freeze,exp,*,enabled=False):
    if not enabled:return None
    exp=Path(exp).resolve();selected=[(exp/n).resolve() for n in freeze['tests']]
    sources={(exp/n).resolve():sha for n,sha in freeze['source_sha256'].items()}
    times=[[],[]];digests=[];counts=[]
    # Fixed three alternating pairs, no discarded rounds or outcome tuning.
    for iteration in range(3):
        for arm in ((0,1) if iteration%2==0 else (1,0)):
            start=time.perf_counter();reg=custody(selected,sources,exp.parents[1],enabled=True)
            for path in reg.specs:reg.prepare(path)
            if arm==0:
                for path,receipt in reg._receipts.items():
                    source=path.read_bytes()
                    code=(compile_cleanup(source,str(path),enabled=True)[0] if receipt.policy=='cleanup'
                        else compiled(source,str(path),rewritten=True))
                    if tuple((code_key(c),code_image(c)) for c in code_tree(code))!=receipt.code_images:
                        raise ValueError('before-execution custody differs from independent fresh compilation')
            digest=hashlib.sha256(repr(tuple((str(p),r) for p,r in reg._receipts.items())).encode()).hexdigest()
            times[arm].append(time.perf_counter()-start);digests.append(digest);counts.append(reg.compile_count)
    if len(set(digests))!=1 or len(set(counts))!=1:raise ValueError('compiler source population or output changed')
    medians=[statistics.median(v) for v in times]
    return dict(schema='c51_compile_reuse_probe_v1',paired_rounds=3,definition_sources=counts[0],
        reference_compilations=2*counts[0],candidate_compilations=counts[0],
        reference_s=times[0],candidate_s=times[1],reference_median_s=medians[0],candidate_median_s=medians[1],
        median_saved_s=medians[0]-medians[1],all_independent_compiler_images_identical=True,
        complete_compiler_receipts_sha256=digests[0],tests_executed=0,HZ_operations_executed=0,
        full_suite_timing_gate_proved=False,formal_gain=0)
