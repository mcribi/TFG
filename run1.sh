#!/bin/bash
#SBATCH -J segme_lung
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o ./slurm_outputs/%A.out
#SBATCH -e ./slurm_outputs/%A.err
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
python ./segmentacion/segmentar_pulmones.py