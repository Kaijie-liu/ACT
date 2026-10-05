"""Neural-HZ rigorous engine n019 = n017 + AveragePool / MaxPool from n010 (LOG N040, N089).

AveragePool is affine; MaxPool is computed exactly as m_{j+1} = m_j + ReLU(v_{j+1} - m_j) with the
projection-aligned ReLU, so it keeps binary phases.  The two operators are taken unchanged from
nhz_sound_v10.py; everything else is n017."""
from nhz_sound_v10 import SoundEngineV10
from nhz_sound_v17 import SoundEngineV17

V19_VERSION = "n019.0"


class SoundEngineV19(SoundEngineV17):
    op_AveragePool = SoundEngineV10.op_AveragePool
    op_MaxPool = SoundEngineV10.op_MaxPool
