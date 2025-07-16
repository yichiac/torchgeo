#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=ood-test
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out

python3 -m torchgeo test --config experiments/ood/cdl_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path OOD/urq6mes5/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/ood/eurocrops_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path OOD/jk3q59t5/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/ood/nccm_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path OOD/taxeouvm/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/ood/sact_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path OOD/nonfvwq1/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/ood/sas_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path OOD/b9qclw7e/checkpoints/epoch=49-step=1450.ckpt
