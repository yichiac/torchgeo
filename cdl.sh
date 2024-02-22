#!/usr/bin/env bash

#SBATCH --time=03:00:00
#SBATCH --mem=80G
#SBATCH --job-name=cdlsentinel2
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

cd ~/torchgeo
python3 test_cdlsentinel2.py
