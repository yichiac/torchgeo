#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=sact-ssl4eo-test
#SBATCH --output=experiment-logs/%x-%j-o.out
#SBATCH --error=experiment-logs/%x-%j-e.out


python3 -m torchgeo test --config experiments/sact/sact_resnet50_ssl4eo_frozen_10ID.yaml --seed_everything 0 --ckpt_path ID/j167fh5f/checkpoints/epoch=99-step=100.ckpt
python3 -m torchgeo test --config experiments/sact/sact_resnet50_ssl4eo_frozen_100ID.yaml --seed_everything 0 --ckpt_path ID/ua3ky8xh/checkpoints/epoch=99-step=100.ckpt
