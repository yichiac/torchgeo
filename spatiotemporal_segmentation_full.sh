#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --job-name=pastis_full
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu

#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out

cd ~/torchgeo

python3 spatiotemporal_segmentation_full.py
