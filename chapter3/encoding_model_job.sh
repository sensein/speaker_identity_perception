#!/bin/bash
#SBATCH --job-name=encoding_model
#SBATCH --output=/om2/scratch/Tue/gelbanna/aspd/job_logs/job_%j.out
#SBATCH --error=/om2/scratch/Tue/gelbanna/aspd/job_logs/job_%j.err
#SBATCH --mem=20Gb
#SBATCH -N 1                 # one node
#SBATCH -n 1                # two CPU (hyperthreaded) cores
#SBATCH --time=90:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gablab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gelbanna@mit.edu

module add openmind/miniconda/3.9.1

source activate /om2/user/gelbanna/miniconda3/envs/fmri38

module load openmind/hcp-workbench/1.2.3

python encoding_models.py