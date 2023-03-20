#!/bin/bash
#SBATCH --job-name=tera_vctk
#SBATCH --output=/om2/user/gelbanna/job_logs/job_%j.out
#SBATCH --error=/om2/user/gelbanna/job_logs/job_%j.err
#SBATCH --mem=90Gb
#SBATCH --gres=gpu:1
#SBATCH -N 1                 # one node
#SBATCH -n 1
#SBATCH --time=48:00:00
#SBATCH --partition=gablab
#SBATCH --mail-user=gelbanna@mit.edu

module add openmind/miniconda/3.9.1

source activate /om2/user/gelbanna/miniconda3/envs/ASpD38

python precompute_encoder_features.py