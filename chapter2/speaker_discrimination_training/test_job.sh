#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/om2/user/gelbanna/job_logs/job_%j.out
#SBATCH --error=/om2/user/gelbanna/job_logs/job_%j.err
#SBATCH --mem=20Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --partition=gablab
#SBATCH -x node[112,113]

module add openmind/miniconda/3.9.1

source activate /om2/user/gelbanna/miniconda3/envs/ASpD38

srun python test.py
# python ./decoders/lstm.py