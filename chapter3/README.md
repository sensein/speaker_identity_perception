# Chapter 3: From Models to Brains

This directory includes:

1. *studyforrest_analysis.ipynb*: Preprocessing CIFTI files and stimuli audios, extracting models' embeddings from each layer, and segmenting narrator's voice from audio movie.
2. *encoding_models.py*: Implement and Fit a Ridge Regression model to predict grayordinate values from models' audio embeddings.
3. *utils.py*: Helper functions for running the encoding models.
4. *encoding_model_job.sh*: Bash file to run encoding model jobs on slurm.