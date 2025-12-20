# act/pipeline/verification/run_confignet_sample.py
#!/usr/bin/env python3
"""
Sample ConfigNet cases and emit JSONL logs.

Usage:
  python -m act.pipeline.verification.run_confignet_sample --seed 0 --n 10 --out logs/confignet.jsonl
"""

from __future__ import annotations

import argparse

from act.pipeline.verification.confignet import sample_configs, set_global_seeds
from act.pipeline.verification.confignet.jsonl import write_jsonl_records, make_record


def parse_args():
    p = argparse.ArgumentParser(description="Sample ConfigNet configs and log JSONL.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", type=str, default="logs/confignet_samples.jsonl")
    p.add_argument("--build", action="store_true", help="Build ACT net to record effective spec (CPU-only).")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    configs = sample_configs(seed=args.seed, n=args.n)
    records = []
    for idx, (mcfg, scfg) in enumerate(configs):
        effective_spec = None
        if args.build:
            from act.pipeline.verification.confignet.builders import build_act_net
            from act.pipeline.verification.confignet.utils import extract_effective_spec

            act_net = build_act_net(mcfg, scfg, name=f"confignet_{idx}")
            effective_spec = extract_effective_spec(act_net)

        records.append(
            make_record(
                args.seed,
                idx,
                mcfg.to_dict(),
                scfg.to_dict(),
                overrides={
                    "template_name": mcfg.template_name,
                    "spec": scfg.to_dict(),
                    "effective_spec": effective_spec,
                },
            )
        )
    write_jsonl_records(args.out, records)
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
