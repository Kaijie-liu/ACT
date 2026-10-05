"""Same frozen physical implementation, exclusively routed to the v2 attempt."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831 import c62_physical_boundary_worker_v1 as worker

worker.DIRECTORY=Path(__file__).resolve().parent/'results/c62_physical_boundary_20260913_v2'
if __name__=='__main__':worker.main()

