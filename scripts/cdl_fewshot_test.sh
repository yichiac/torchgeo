#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=cdl-fewshot-test
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out

python3 -m torchgeo test --config experiments/fewshot/cdl_10_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path fewshot/7qhm4d7f/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/fewshot/cdl_100_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path fewshot/4fkrih48/checkpoints/epoch=49-step=1450.ckpt
python3 -m torchgeo test --config experiments/fewshot/cdl_900_ood_ssl4eo.yaml --seed_everything 0 --ckpt_path fewshot/jjh5w3yo/checkpoints/epoch=49-step=1750.ckpt
