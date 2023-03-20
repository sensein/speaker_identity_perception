#!/bin/bash
#SBATCH --job-name=train_ray
#SBATCH --output=/om2/user/gelbanna/job_logs/job_%j.out
#SBATCH --error=/om2/user/gelbanna/job_logs/job_%j.err
#SBATCH --mem=90Gb
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=gablab
#SBATCH --mail-user=gelbanna@mit.edu
#SBATCH --mail-type=ALL

module add openmind/miniconda/3.9.1

source activate /om2/user/gelbanna/miniconda3/envs/ASpD38

srun python train.py