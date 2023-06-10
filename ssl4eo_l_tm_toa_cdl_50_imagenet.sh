#!/usr/bin/env bash

#SBATCH --time=3-00:00:00
#SBATCH --job-name=tm-toa-50-cdl-magenet
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

python3 ~/torchgeo/experiments/ssl4eo/run_ssl4eol_tm_toa_cdl_50_imagenet.py
