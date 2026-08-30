# AdvMoE dependency and Blackwell audit

## Result

The read-only dependency audit passes with zero issues. No package was installed
and no environment was created.

The official repository does **not** define an exact author environment:

- Python, PyTorch, torchvision, and CUDA are unpinned;
- the only versioned requirement is `scipy==1.6.0`;
- the README asks users to install `requirement.txt`, while the repository file
  is named `requirements.txt`.

Consequently, this line cannot use the RT-ER label “exact author-pin
reproduction.” Its future environment must be labelled
`official-code, Blackwell-compatible dependency reproduction` and must record
every resolved version.

## Existing environments

The official router source executes a finite `[2,2]` CUDA forward on the
available `sm_120` Blackwell GPU under the unchanged `act-py312` environment
(`torch 2.9.1+cu128`). This establishes model-only kernel compatibility, unlike
the RT-ER exact-pin failure. It does not establish that the full trainer runs.

The released training entry point fails before its argument parser in
`act-py312`; the first missing import is `h5py`. Static package inspection also
finds absent tensorboard, easydict, lmdb, and datasets. The isolated CROWN
environment lacks the same training packages and is not repurposed as a
training environment.

Tracked audit result:
`act/pipeline/moe/results/advmoe_dependency_audit_20260830.json`.

## CROWN backend feasibility correction

The 269K-parameter router is a conventional CNN mathematically, but the current
backend cannot consume or bound it as cheaply as a bare architecture count
suggests:

1. the literal source graph is rejected by auto_LiRPA at strided tensor-slice
   shortcuts;
2. the ACT adapter replaces those shortcuts with fixed identity 1x1 stride-2
   convolutions and replaces dynamic global pooling by fixed `AvgPool2d(8)`;
3. these replacements are bit-exact on the registered 32x32 domain in the
   current tests;
4. after lowering, batch-20 IBP completes quickly, but default patches-mode
   CROWN takes more than 90 seconds on one CPU thread and a single-sample GPU
   probe exhausts a 95 GiB Blackwell after the worker holds more than 62 GiB.

The CROWN time/OOM observations are engineering feasibility probes, not paper
coverage measurements. No further full CROWN probe runs concurrently with B1.
The harness keeps the backend pluggable and records bound errors rather than
silently falling back.

## Next environment gate

A future isolated training environment may be created only after B1/B3 resource
gates permit it. It must:

- stay below `/data1/Kane/MOE`;
- leave `act-py312` and the CROWN environment unchanged;
- start from a frozen resolved dependency manifest;
- run import, one-batch forward/backward, router-update, optimizer-state, and
  checkpoint-resume smoke tests before seed-0 training;
- retain the no-license policy: no external source is copied into ACT and
  checkpoint redistribution remains subject to legal review.
