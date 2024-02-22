#!/usr/bin/env bash

#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --job-name=s2_download_small
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=END
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

wget -i ~/urls.txt -P /projects/dali/data/sentinel2/
