'''code for loading data and linear evaluation is mainly adapted from https://github.com/nttcslab/byol-a'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import soundfile as sf
import deciphering_enigma
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold

#define the experiment config file path
path_to_config = './config.yaml'

#read the experiment config file
exp_config = deciphering_enigma.load_yaml_config(path_to_config)
dataset_path = exp_config.dataset_path

#register experiment directory and read wav files' paths
audio_files = deciphering_enigma.build_experiment(exp_config)
audio_files = [audio for audio in audio_files if audio.endswith('_normloud.wav')]
print(f'Dataset has {len(audio_files)} samples')

#extract metadata from file name convention
metadata_df, audio_format = deciphering_enigma.extract_metadata(exp_config, audio_files)
metadata_df['AudioPath'] = audio_files
metadata_df['ID'] = np.array(list(map(lambda x: x.split('/')[-2][1:], audio_files)))
metadata_df['Gender'] = np.array(list(map(lambda x: x.split('/')[-2][0], audio_files)))

#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = deciphering_enigma.load_dataset(list(metadata_df.AudioPath), cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format, 
                                                    norm_loudness=exp_config.norm_loudness, target_loudness=exp_config.target_loudness)

#generate speech embeddings
feature_extractor = deciphering_enigma.FeatureExtractor()
embeddings_dict = feature_extractor.extract(audio_tensor_list, exp_config)

#split data to train (7 scentences) and test (3 scentences)
seed = 42
np.random.seed(seed=42)
metadata_df['Train'] = 1
for speaker in metadata_df['ID'].unique():
    speaker_df = metadata_df.loc[metadata_df.ID == speaker]
    labels = np.random.choice(speaker_df.Label, 3, replace=False)
    metadata_df['Train'] = metadata_df.apply(lambda row: 0 if row.ID==speaker 
                                                      and row.Label in labels else row.Train, axis=1)

parameter_space = {'clf__hidden_layer_sizes': [(10,), (50,), (100,)], 
                   'clf__learning_rate_init': np.logspace(-2, -4, num=3)}
for model_name, embeddings in embeddings_dict.items():
    print(f'Testing {model_name}')
    results = defaultdict(list)
    model_dir = f'../{exp_config.dataset_name}/{model_name}'
    
    #Create df for train and test
    df_embeddings = pd.DataFrame(embeddings).add_prefix('Embeddings_')
    columns = list(df_embeddings.columns)
    df = pd.concat([metadata_df, df_embeddings], axis=1)
    X_train_df = df[columns].loc[df.Train == 1]
    Y_train_df = df['ID'].loc[df.Train == 1]
    X_test_df = df[columns].loc[df.Train == 0]
    Y_test_df = df['ID'].loc[df.Train == 0]
    
    #Build pipeline and define k-fold CV
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    pipeline = deciphering_enigma.build_pipeline(parameter_space, cv, (), 200, True, 
                                                 True, 'recall_macro')
    
    #Run training and testing
    for _ in range(10):
        score = deciphering_enigma.linear_eval(X_train_df, Y_train_df, X_test_df, Y_test_df, pipeline)
        results['Model'].append(model_name)
        results['Score'].append(score)
    
    #Save results
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(f'{model_dir}/{model_name}_{seed}_results.csv')