#!/bin/bash
#SBATCH --job-name=mini-project
#SBATCH --output=logs/mini-project_%j.log
#SBATCH --error=logs/mini-project_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi

mkdir -p logs

module purge
module load Python/3.10.8-GCCcore-12.2.0
source ~/venv/bin/activate
cd ~/mini-project-snowpoles
nvidia-smi
python main.py