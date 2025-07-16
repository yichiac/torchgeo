#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=cdl-imagenet
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out


python3 -m torchgeo fit --config experiments/cdl/sentinel2_cdl_resnet50_imagenet_frozen.yaml --seed_everything 0 --model.lr 0.01
python3 -m torchgeo fit --config experiments/cdl/sentinel2_cdl_resnet50_imagenet_frozen.yaml --seed_everything 0 --model.lr 0.001
python3 -m torchgeo fit --config experiments/cdl/sentinel2_cdl_resnet50_imagenet_frozen.yaml --seed_everything 0 --model.lr 0.0001
