"""Neural-HZ rigorous engine n017 = n014 (difference-aware softmax bounds) + n016 (recorded
smooth units for sparse smooth rows in terminal plans).  Both are additive to n011.2 and
touch disjoint operators (Softmax/state products vs Sigmoid/Tanh); MRO: n016, n014, n011.2."""
from nhz_sound_v14 import SoundEngineV14
from nhz_sound_v16 import SoundEngineV16

V17_VERSION = "n017.0"


class SoundEngineV17(SoundEngineV16, SoundEngineV14):
    pass
