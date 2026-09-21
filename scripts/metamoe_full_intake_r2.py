"""R2 launch repair: explicit ACT import root and isolated CPU author stack."""
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
os.environ['PYTHONPATH'] = str(root)  # inherited by R1's independently supervised worker
sys.path.insert(0, str(root))

if __name__ == '__main__':
    from recent_moe_deployment import sha256
    from recent_moe_env_inventory import inventory
    import metamoe_full_intake as frozen_r1
    config = Path(sys.argv[sys.argv.index('--config') + 1])
    cfg = json.loads(config.read_text())
    env = Path(cfg['environment_inventory'])
    if (sha256(env) != cfg['environment_inventory_sha256'] or
            inventory([cfg['python']]) != json.loads(env.read_text())):
        raise ValueError('isolated environment changed')
    frozen_r1.main()
