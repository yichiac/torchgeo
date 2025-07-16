#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=sas-ssl4eo-test
#SBATCH --output=experiment-logs/%x-%j-o.out
#SBATCH --error=experiment-logs/%x-%j-e.out


python3 -m torchgeo test --config experiments/sas/sentinel2_sas_resnet50_ssl4eo_frozen_10ID.yaml --seed_everything 0 --ckpt_path ID/mhdv2shn/checkpoints/epoch=99-step=100.ckpt
python3 -m torchgeo test --config experiments/sas/sentinel2_sas_resnet50_ssl4eo_frozen_100ID.yaml --seed_everything 0 --ckpt_path ID/r4jnhmd4/checkpoints/epoch=99-step=100.ckpt
