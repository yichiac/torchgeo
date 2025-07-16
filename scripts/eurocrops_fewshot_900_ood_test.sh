#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=eurocrops-fewshot-test
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out

python3 -m torchgeo test --config experiments/fewshot/eurocrops_900_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path fewshot/ym2ga7l0/checkpoints/epoch=49-step=1750.ckpt
