"""Same complete C72 phase and C70 proof, with guarded journal queries."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831 import c70_native_worker_v1 as core
from experiments.neural_hz_20260831 import c72_native_worker_v2 as phase
from experiments.neural_hz_20260831.c73_outer_query_v1 import compile_journal
RUN=Path(__file__).resolve().parent/'results/c73_outer_query_20260913_v1'

if __name__=='__main__':
    phase.RUN=RUN
    core.RUN=RUN
    core.phase=phase.recovered_phase
    core.compile_journal=compile_journal
    core.main()
