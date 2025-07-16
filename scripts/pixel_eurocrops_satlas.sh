#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=euro-satlas
#SBATCH --output=experiment-logs/%x-%j-o.out
#SBATCH --error=experiment-logs/%x-%j-e.out


python3 -m torchgeo fit --config experiments/eurocrops/sentinel2_eurocrops_resnet50_satlas_rasterized_frozen.yaml --seed_everything 0 --model.lr 0.01
python3 -m torchgeo fit --config experiments/eurocrops/sentinel2_eurocrops_resnet50_satlas_rasterized_frozen.yaml --seed_everything 0 --model.lr 0.001
python3 -m torchgeo fit --config experiments/eurocrops/sentinel2_eurocrops_resnet50_satlas_rasterized_frozen.yaml --seed_everything 0 --model.lr 0.0001
