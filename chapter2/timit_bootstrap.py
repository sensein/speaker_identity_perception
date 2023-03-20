import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import soundfile as sf
import deciphering_enigma
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn import model_selection
from sklearn.model_selection import RepeatedStratifiedKFold

def linear_eval(X, y, X_test, y_test, pipeline):
    """Perform a single run of linear evaluation."""
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)
    return score, y_pred

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


parameter_space = {'clf__hidden_layer_sizes': [(10,), (50,), (100,)], 
                   'clf__learning_rate_init': np.logspace(-2, -4, num=3)}
predictions = {'ID':[], 'Pred':[]}

for model_name, embeddings in embeddings_dict.items():
    print(f'Testing {model_name}')
    results = defaultdict(list)
    model_dir = f'../{exp_config.dataset_name}/{model_name}'
    
    for _ in range(10000):

        #Split data to train (7 scentences) and test (3 scentences)
        y_train, y_test = model_selection.train_test_split(metadata_df['AudioPath'], stratify=metadata_df['ID'], test_size=0.3)
        test_list = list(y_test)
        metadata_df['Train'] = 1
        metadata_df['Train'] = metadata_df.apply(lambda row: 0 if row.AudioPath in test_list else row.Train, axis=1)

        #Create df for train and test
        df_embeddings = pd.DataFrame(embeddings).add_prefix('Embeddings_')
        columns = list(df_embeddings.columns)
        df = pd.concat([metadata_df, df_embeddings], axis=1)
        X_train_df = df[columns].loc[df.Train == 1]
        Y_train_df = df['ID'].loc[df.Train == 1]
        X_test_df = df[columns].loc[df.Train == 0]
        Y_test_df = df['ID'].loc[df.Train == 0]
        
        #Build pipeline and define k-fold CV
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        pipeline = deciphering_enigma.build_pipeline(parameter_space, cv, (), 200, True, 
                                                    True, 'recall_macro')
        
        #Run training and testing
        score, y_pred = linear_eval(X_train_df, Y_train_df, X_test_df, Y_test_df, pipeline)
        predictions['ID'].append(list(Y_test_df))
        predictions['Pred'].append(list(y_pred))
        results['Model'].append(model_name)
        results['Score'].append(score)
    
    #Save results
    predictions_df = pd.DataFrame.from_dict(predictions)
    predictions_df.to_csv(f'{model_dir}/{model_name}_bootstrap_predictions.csv')
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(f'{model_dir}/{model_name}_bootstrap_results.csv')