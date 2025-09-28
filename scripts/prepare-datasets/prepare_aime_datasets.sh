#!/usr/bin/env bash
set -uxo pipefail

mkdir -p data/datasets
# BeyondAIME
huggingface-cli download ByteDance-Seed/BeyondAIME --repo-type dataset --local-dir data/datasets/hf/ByteDance-Seed--BeyondAIME
