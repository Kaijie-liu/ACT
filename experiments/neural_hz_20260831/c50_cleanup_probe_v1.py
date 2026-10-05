"""Bounded compiler-only probe; never a substitute for the complete test gate."""
import statistics
import time
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import compiled
from experiments.neural_hz_20260831.c50_cleanup_compile_v1 import compile_cleanup,exact_canaries

SOURCE=b'''
def run(n):
    total = 0
    for i in range(n):
        x = i & 7
        y = x + 1
        assert y > x
        assert x == 0 or y < 10
        total += x
    return total
'''


def probe(*,enabled=False):
    if not enabled:return None
    canary=exact_canaries(enabled=True)
    old={};new={}
    exec(compiled(SOURCE,'<c50-fixed-cleanup-probe>',rewritten=True),old)
    code,stats=compile_cleanup(SOURCE,'<c50-fixed-cleanup-probe>',enabled=True);exec(code,new)
    durations=[[],[]];checks=[]
    # Fixed complete workloads,7 alternating paired rounds; no tuning,
    # discarded warmups or early-stop-on-a-favorable-sample rule.
    for round_index in range(7):
        for arm in ((0,1) if round_index%2==0 else (1,0)):
            start=time.perf_counter();result=(old,new)[arm]['run'](100000)
            durations[arm].append(time.perf_counter()-start)
            if result!=350000:raise ValueError('compiler probe changed loop/condition execution')
            checks.append(result)
    a,b=map(statistics.median,durations)
    return dict(schema='c50_fixed_private_cleanup_probe_v1',canaries=canary,compile=stats,
        rounds=7,loop_iterations_per_arm_per_round=100000,all_checksums_equal=True,
        reference_s=durations[0],candidate_s=durations[1],reference_median_s=a,candidate_median_s=b,
        median_speed_ratio=a/b,strict_probe_time_payment=b<a,
        full_test_suite_or_generator_payment_proved=False,formal_gain=0)
