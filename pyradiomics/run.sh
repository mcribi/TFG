#!/bin/bash
#SBATCH -J 3dlung
#SBATCH -p dios
#SBATCH --gres=gpu:0
#SBATCH -c 8
#SBATCH -o ./slurm_outputs/%A.out
#SBATCH -e ./slurm_outputs/%A.err
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/pyradiomics_env
python ./segmentacion/segmentar_nodulos.py