import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import deciphering_enigma
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torchaudio
from torch.utils.data import Dataset
from deciphering_enigma.settings import _MODELS_WEIGHTS
from sklearn.model_selection import RepeatedStratifiedKFold
from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel

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

#split data to train (7 scentences) and test (3 scentences)
seed = 42
np.random.seed(seed=42)
metadata_df['Train'] = 1
for speaker in metadata_df['ID'].unique():
    speaker_df = metadata_df.loc[metadata_df.ID == speaker]
    labels = np.random.choice(speaker_df.Label, 3, replace=False)
    metadata_df['Train'] = metadata_df.apply(lambda row: 0 if row.ID==speaker 
                                                      and row.Label in labels else row.Train, axis=1)

#load audio files as torch tensors to get ready for feature extraction
audio_tensor_list = deciphering_enigma.load_dataset(list(metadata_df.AudioPath), cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format, 
                                                    norm_loudness=exp_config.norm_loudness, target_loudness=exp_config.target_loudness)

model_components = ['feature_extractor.conv_layers', 'encoder.layers']
activation = {}

def get_activation(name):
    def hook(model, input, output):
        if 'encoder' in name:
            activation[name] = output[0].detach()
        else:
            activation[name] = output.detach()
    return hook

def get_model(model_name, weight_file, exp_config):
    feature_extractor = deciphering_enigma.FeatureExtractor()
    model = feature_extractor.load_model(model_name, weight_file, exp_config)
    return model

def get_embeddings(audio_tensor_list, model, component, layer, layer_name):
    embeddings = []
    for audio in tqdm(audio_tensor_list, desc=f'Generating {model_name} Embeddings...'):
        model.eval()
        with torch.no_grad():
            getattr(getattr(model, component.split('.')[0]), component.split('.')[1])[layer].register_forward_hook(get_activation(layer_name))
            _ = model(audio.to('cuda').unsqueeze(0))
            if 'encoder' in layer_name:
                embedding = activation[layer_name]
            else:
                embedding = activation[layer_name].permute(0,2,1)
            embedding = embedding.mean(1) + embedding.amax(1)
            embedding = np.squeeze(embedding.cpu().detach().numpy())
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    return embeddings

for model_name, weights_file in _MODELS_WEIGHTS.items():
    print(f'Evaluating {model_name}')
    model_dir = f'../{exp_config.dataset_name}/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    model = get_model(model_name, weights_file, exp_config)
    for component in model_components:
        num_layers = len(dict(model.named_modules())[component])
        for layer in range(num_layers):
            print(f'  getting embeddings for {component}{layer}...')
            emb_file_name = f'{model_dir}/{component}{layer}_embeddings.npy'
            layer_name = f'{component}_{layer}'
            if os.path.isfile(emb_file_name):
                print(f'{model_name} embeddings are already saved for {exp_config.dataset_name} for layer {layer_name}')
                continue
            else:
                embeddings = get_embeddings(audio_tensor_list, model, component, layer, layer_name)
                print(embeddings.shape)
                np.save(emb_file_name, embeddings)


parameter_space = {'clf__hidden_layer_sizes': [(10,), (50,), (100,)],
                   'clf__learning_rate_init': np.logspace(-2, -4, num=3)}

for model_name, weights_file in _MODELS_WEIGHTS.items():
    model_dir = f'../{exp_config.dataset_name}/{model_name}'
    print(f'Testing {model_name}')
    model = get_model(model_name, weights_file, exp_config)
    for component in model_components:
        num_layers = len(dict(model.named_modules())[component])
        for layer in range(num_layers):
            results = defaultdict(list)
            layer_name = f'{component}_{layer}'
            print(f'    Testing {component}{layer}')

            #Load layer embeddings
            emb_file_name = f'{model_dir}/{component}{layer}_embeddings.npy'
            embeddings = np.load(emb_file_name)

            #Create df for train and test
            df_embeddings = pd.DataFrame(embeddings).add_prefix('Embeddings_')
            columns = list(df_embeddings.columns)
            df = pd.concat([metadata_df, df_embeddings], axis=1)
            X_train_df = df[columns].loc[df.Train == 1]
            Y_train_df = df['ID'].loc[df.Train == 1]
            X_test_df = df[columns].loc[df.Train == 0]
            Y_test_df = df['ID'].loc[df.Train == 0]
            
            #Build pipeline and define k-fold CV
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
            pipeline = deciphering_enigma.build_pipeline(parameter_space, cv, (), 200, True, 
                                                        True, 'recall_macro')
            
            #Run training and testing
            for _ in range(10):
                score = deciphering_enigma.linear_eval(X_train_df, Y_train_df, X_test_df, Y_test_df, pipeline)
                results['Model'].append(model_name)
                results['Layer'].append(layer_name)
                results['Score'].append(score)
            
            #Save results
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(f'{model_dir}/{model_name}_{layer_name}_{seed}_results.csv')