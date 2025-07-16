#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=eurocrops-fewshot
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out

python3 -m torchgeo fit --config experiments/fewshot/eurocrops_10_ood_ssl4eo.yaml --seed_everything 0
python3 -m torchgeo fit --config experiments/fewshot/eurocrops_100_ood_ssl4eo.yaml --seed_everything 0
python3 -m torchgeo fit --config experiments/fewshot/eurocrops_900_ood_ssl4eo.yaml --seed_everything 0
