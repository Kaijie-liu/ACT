#!/bin/bash
# Rerun the 4 VGG mini-atlas iids that failed to produce snapshots in the
# main 2026-06-03 run:
#   iid 2: watchdog killed at RSS 42.67 GB (cap was 40 GB)  → cap = 60 GB
#   iid 15, 16, 17: wall timeout at 307 s                  → wall = 600 s
#
# Per advisor 2026-06-04: the 4 missing iids do NOT change the §6b gate
# decision (PROCEED), but completing the table is good hygiene. Wait for
# the clean canonical sweep to finish so we don't compete for GPU.
set -u

ACT_ROOT=/data1/Kane/ACT
PY=/data1/Kane/miniconda3/envs/act-py312/bin/python

# iid 2 — bump RSS cap
$PY $ACT_ROOT/research/vgg_mini_atlas_driver.py \
  --iids 2 \
  --wall-s 600 --rss-gb 60 \
  --out "$ACT_ROOT/audit_results/vgg_mini_atlas_missing_rerun_iid2_$(date -u +%Y%m%dT%H%M%SZ)"

# iids 15-17 — bump wall budget
$PY $ACT_ROOT/research/vgg_mini_atlas_driver.py \
  --iids 15,16,17 \
  --wall-s 600 --rss-gb 50 \
  --out "$ACT_ROOT/audit_results/vgg_mini_atlas_missing_rerun_15to17_$(date -u +%Y%m%dT%H%M%SZ)"
