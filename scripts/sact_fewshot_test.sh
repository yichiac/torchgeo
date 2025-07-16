#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=sact-fewshot-test
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out


python3 -m torchgeo test --config experiments/fewshot/sact_10_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path fewshot/1mf8fwoh/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/fewshot/sact_100_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path fewshot/1maezvl2/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/fewshot/sact_900_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path fewshot/2oj34fke/checkpoints/epoch=49-step=1750.ckpt
