1. *deciphering_enigma* package: A package developed by the author to process and analyze audio utterances
   1. *utils.py*: Pre-processing audio files.
   2. *feature_extractor.py*: Extract audio embeddings using pre-trained self-supervised models.
   3. *linear_encoding.py*: Fit multiple classifiers with hyperparameter tuning.
   4. *reducer_tuner.py*: Run multiple dimensionality reduction algorithms with hyperparameter tuning.
   5. *representation_similarity.py*: compute CKA scores across all representations.
   6. *settings.py*: Preset hyperparameters for models, classifiers, and reducers.
   
2. *config.yaml*: Preset parameters for experiments.

3. *Benchmark*: code used to benchmark SSMs and handcrafted representations on an automatic speaker recognition (ASpR) task (VoxCeleb/TIMIT)
   1. *XX_benchmark.py*: Benchmarking performance of models on dataset *XX* (VoxCeleb/TIMIT).
   2. *XX_layer_analysis.py*: Benchmarking performance of layers within models on dataset *XX* (VoxCeleb/TIMIT).
   3. *voxceleb_subset_analysis.py*: Benchmarking performance of models on subsets of speakers of VoxCeleb.
   4. *voxceleb_utils.py*: Helper fuctions for VoxCeleb Benchmark.

4. All speech experiments were executed using *deciphering enigma* package. Please find all speech experiements in this [Jupyter Book](https://gasserelbanna.github.io/vocal-identity-jupyter-book/5_chapter1/datasets_experiments.html).