#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=id-test
#SBATCH --output=experiment-logs/o%x-%j.out
#SBATCH --error=experiment-logs/e%x-%j.out

python3 -m torchgeo test --config experiments/cdl/sentinel2_cdl_resnet50_ssl4eo_frozen.yaml --seed_everything 0 --ckpt_path ID/j2hh10y7/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/cdl/sentinel2_cdl_resnet50_satlas_frozen.yaml --seed_everything 0 --ckpt_path ID/5nxnhjze/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/cdl/sentinel2_cdl_resnet50_imagenet_frozen.yaml --seed_everything 0 --ckpt_path ID/rxbrx49g/checkpoints/epoch=99-step=700.ckpt

python3 -m torchgeo test --config experiments/eurocrops/sentinel2_eurocrops_resnet50_ssl4eo_rasterized_frozen.yaml --seed_everything 0 --ckpt_path ID/5dcinxpv/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/eurocrops/sentinel2_eurocrops_resnet50_satlas_rasterized_frozen.yaml --seed_everything 0 --ckpt_path ID/v50fr9zx/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/eurocrops/sentinel2_eurocrops_resnet50_imagenet_rasterized_frozen.yaml --seed_everything 0 --ckpt_path ID/jajkuc8s/checkpoints/epoch=99-step=700.ckpt

python3 -m torchgeo test --config experiments/nccm/sentinel2_nccm_resnet50_ssl4eo_frozen.yaml --seed_everything 0 --ckpt_path ID/rythp5r4/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/nccm/sentinel2_nccm_resnet50_satlas_frozen.yaml --seed_everything 0 --ckpt_path ID/ukpbi65e/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/nccm/sentinel2_nccm_resnet50_imagenet_frozen.yaml --seed_everything 0 --ckpt_path ID/acur6xdc/checkpoints/epoch=99-step=700.ckpt

python3 -m torchgeo test --config experiments/sas/sentinel2_sas_resnet50_ssl4eo_frozen.yaml --seed_everything 0 --ckpt_path ID/y3efi369/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/sas/sentinel2_sas_resnet50_satlas_frozen.yaml --seed_everything 0 --ckpt_path ID/lr4y2pxn/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/sas/sentinel2_sas_resnet50_imagenet_frozen.yaml --seed_everything 0 --ckpt_path ID/m702t268/checkpoints/epoch=99-step=700.ckpt

python3 -m torchgeo test --config experiments/sact/sact_resnet50_ssl4eo_frozen.yaml --seed_everything 0 --ckpt_path ID/xg4ikze7/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/sact/sact_resnet50_satlas_frozen.yaml --seed_everything 0 --ckpt_path ID/79bt1kre/checkpoints/epoch=99-step=700.ckpt
python3 -m torchgeo test --config experiments/sact/sact_resnet50_imagenet_frozen.yaml --seed_everything 0 --ckpt_path ID/3u06xj6z/checkpoints/epoch=99-step=700.ckpt
