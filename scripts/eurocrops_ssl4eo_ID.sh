#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=euro-ssl4eo
#SBATCH --output=experiment-logs/%x-%j-o.out
#SBATCH --error=experiment-logs/%x-%j-e.out

python3 -m torchgeo fit --config experiments/eurocrops/sentinel2_eurocrops_resnet50_ssl4eo_rasterized_frozen_10ID.yaml --seed_everything 0
python3 -m torchgeo fit --config experiments/eurocrops/sentinel2_eurocrops_resnet50_ssl4eo_rasterized_frozen_100ID.yaml --seed_everything 0
