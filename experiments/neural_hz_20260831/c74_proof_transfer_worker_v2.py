"""Separate v2 output; complete original transfer/restore logic is unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831 import c74_proof_transfer_worker_v1 as core
core.RUN=Path(__file__).resolve().parent/'results/c74_native_binding_20260913_v2'
if __name__=='__main__':core.main()
