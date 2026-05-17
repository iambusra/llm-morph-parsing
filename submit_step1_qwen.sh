#!/bin/bash
#SBATCH --job-name=step1_qwen
#SBATCH --output=step1_qwen_%j.out
#SBATCH --error=step1_qwen_%j.err
#SBATCH --partition=jag-hi
#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

source /nlp/scr/busra/miniconda3/etc/profile.d/conda.sh
conda activate experiment2

export TOKENIZERS_PARALLELISM=false

python3 -u step1_align.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --output-dir qwen_step1