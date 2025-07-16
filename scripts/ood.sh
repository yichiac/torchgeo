#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=ood
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out


python3 -m torchgeo fit --config experiments/ood/cdl_ood_ssl4eo.yaml --seed_everything 0
python3 -m torchgeo fit --config experiments/ood/eurocrops_ood_ssl4eo.yaml --seed_everything 0
python3 -m torchgeo fit --config experiments/ood/nccm_ood_ssl4eo.yaml --seed_everything 0
python3 -m torchgeo fit --config experiments/ood/sact_ood_ssl4eo.yaml --seed_everything 0
python3 -m torchgeo fit --config experiments/ood/sas_ood_ssl4eo.yaml --seed_everything 0
