# Chapter 2: Speaker Discrimination in Models and Humans

This directory includes:

1. *stimuli_selection.ipynb*: Pipeline for compiling stimuli used in the behavioral experiment for **Study 1**.
2. *behavioral_exp_results.ipynb*: Analyzing human behavioral data on speaker discrimination task for **Study 1**.
3. *timit_bootstrap.py*: Running ASpR on HuBERT best-performing layer with bootstrapping for **Study 2**.
4. *compute_distance.py*: Compute distance mertics across speakers utterances for **Study 1 & 2**.
5. *speaker_discrimination_training* directory: This directory includes scripts for taining models on an automatic speaker discrimination task (ASpD) for **Study 3**. 
   "Training Script is adapted and based on this [repo](https://github.com/nttcslab/byol-a)."
   1. *decoders*:
      1. *mlp.py*: Multi-layer Perceptron Architecture Class
      2. *lstm.py*: LSTM Architecture Class
   2. *encoder.py*: Encoder Class (Pre-trained SSMs)
   3. *dataset.py*: Dataset Class
   4. *aggregator.py*: Aggregating embeddings into *Same* and *Different* pairs
   5. *common.py*: Helper functions for training
   6. *learner.py*: Trainer Class using LightningModule
   7. *train.py*: Main training script including ray tuning code
   8. *train_job.sh*: Bash file to run a training job on slurm
   9. *test.py*: Test performance of a predefined checkpoint
   10. *test_job.sh*: Bash file to run a testing job on slurm
   11. *precompute_encoder_features.py*: Extract speaker embeddings using pretrained encoders
   12. *precompute_features_helper.py*: Helper functions for extracting embeddings
   13. *precompute_features_job.sh*: Bash file to run a embeddings extraction job on slurm
   14. *config.yaml*: Preset parameters for training
6. *aspd38.yml*: Conda environment used for this chapter in the project.
