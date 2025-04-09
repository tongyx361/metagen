#!/usr/bin/env bash
set -uxo pipefail

mkdir -p data/datasets
# MATH-500
wget https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/math_splits/test.jsonl -O data/datasets/math-500.jsonl
# TheoremQA
huggingface-cli download TIGER-Lab/TheoremQA --repo-type dataset --local-dir data/datasets/hf/TIGER-Lab--TheoremQA
# MMLU-Pro-1k
jupyter execute notebooks/downsample-mmlu-pro.ipynb
