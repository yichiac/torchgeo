#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --job-name=cdls2
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

cd ~/torchgeo
python3 -m torchgeo fit --config experiments/torchgeo/conf/cdlsentinel2.yaml
