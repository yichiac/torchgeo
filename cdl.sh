#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --job-name=cdl-A100-single
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --account=bcnh-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mail-type=END
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out

. /projects/bcnh/spack/share/spack/setup-env.sh
spack env activate dali

cd ~/torchgeo
python3 -m torchgeo fit --config experiments/torchgeo/conf/cdlsentinel2.yaml
