import re
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import yaml
from pathlib import Path
from easydict import EasyDict

import seaborn as sns
import plotly.express as px

import torch
import torchaudio

import librosa
import serab_byols
import soundfile as sf

import pacmap
from umap import UMAP

import scipy
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from settings import *

##################################### Config file processing ########################################

def split_string(string, delimiters):
    """Splits a string by a list of delimiters.
    adapted from https://datagy.io/python-split-string-multiple-delimiters/
    Args:
        string (str): string to be split
        delimiters (list): list of delimiters
    Returns:
        list: list of split strings
    """
    pattern = r'|'.join(delimiters)
    return re.split(pattern, string)

def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file()
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg

##################################### Data Loading and Audio Preprocessing Functions ########################################

def stereo_to_mono_audio(audio):
    """Convert stereo audio (2 channels) to mono audio (one channel) by averaging the two channels"""
    return np.mean(audio, axis=1)

def int_to_float_audio(audio):
    """Convert audio signal with integer values to float values"""
    if np.issubdtype(audio.dtype, np.integer):
        float_audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    elif audio.dtype == np.float64:
        float_audio = np.float32(audio)
    else:
        float_audio = audio
    return float_audio

def resample_audio(audio, orig_sr, req_sr):
    """Resample audio signal from original sampling rate (orig_sr) to required sampling rate (req_sr) using librosa package"""
    return librosa.core.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=req_sr,
                res_type='kaiser_best')

def chunking_audio(audio, file, chunk_dur, sr, save_path, speaker_id, audio_format):
    audio_tensor_list = []
    audio_len = audio.shape[0]
    noSections = int(np.floor(audio_len/(chunk_dur*sr)))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/{speaker_id}', exist_ok=True)
    try:
        for i in range(0, noSections):
            temp = audio[i*chunk_dur*sr:i*chunk_dur*sr + chunk_dur*sr]
            sf.write(f'{save_path}/{speaker_id}/{os.path.basename(file)}_{str(i).zfill(3)}.{audio_format}', temp, sr)
            audio_tensor_list.append(torch.from_numpy(temp))
        return audio_tensor_list
    except:
        print(f'The number of seconds in audio {file} is smaller than the splitting criteria ({chunk_dur} sec)')

def preprocess_audio_files(files_paths, speaker_ids, chunk_dur=3, resampling_rate=16000, save_path='./preprocessed_audios', audio_format='wav'):
    """Preprocess audio files by convert stereo audio to mono, resample, break audio to smaller chunks and convert audio array to list of torch tesnors.

    Parameters
    ----------
    files_paths : str
        Path for the audio files
    speaker_ids : list
        List of speakers id
    chunk_dur : int
        duration of audio samples after chunking
    resampling rate : int
        Sample rate required for resampling
    save_path : str
        Path for saving the preprocessed audio chunks
    Returns
    -------
    audio_tensor_list : list
        list of torch tensors of preprocessed audios
    """
    audio_tensor_list = []
    save_path = f'{save_path}_dur{chunk_dur}sec'
    for idx, file in enumerate(tqdm(files_paths, desc=f'Preprocessing Audio Files...', total=len(files_paths))):
        # Read audio file
        file_name = os.path.basename(file).split('.')[0]
        audio, orig_sr = sf.read(file)
        
        # Convert to mono if needed
        if audio.ndim == 2:
            audio = stereo_to_mono_audio(audio)
        
        # Convert to float if necessary
        float_audio = int_to_float_audio(audio)
        
        # Resample if needed
        if orig_sr != resampling_rate:
            float_audio = resample_audio(audio, orig_sr, resampling_rate)
        
        # Split audio into chunks
        audio_tensor_list += chunking_audio(float_audio, file, chunk_dur, resampling_rate, save_path, speaker_ids[idx], audio_format)
    return audio_tensor_list

def load_dataset(files, preprocessing=False, speaker_ids=[], chunk_dur=3, resampling_rate=16000, save_path='./preprocessed_audios', audio_format='wav'):
    """Load audio dataset and read the audio files as list of torch tensors.

    Parameters
    ----------
    files_paths : str
        Path for the audio files
    preprocessing : bool
        Apply preprocessing to audio files or not
    speaker_ids : list
        List of speakers id
    chunk_dur : int
        duration of audio samples after chunking
    resampling rate : int
        Sample rate required for resampling
    save_path : str
        Path for saving the preprocessed audio chunks
    Returns
    -------
    audio_tensor_list : list
        list of torch tensors of preprocessed audios
    """
    audio_tensor_list = []
    if preprocessing:
        return preprocess_audio_files(files, speaker_ids, chunk_dur, resampling_rate, save_path, audio_format)
    else:
        for file in tqdm(files, desc=f'Loading Audio Files...', total=len(files)):
            # Read audio file
            audio, orig_sr = sf.read(file)
            # Convert to mono if needed
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            audio_len = audio.shape[0]

            # Convert to float if necessary
            if np.issubdtype(audio.dtype, np.integer):
                float_audio = audio.astype(np.float32) / np.iinfo(np.int16).max
            elif audio.dtype == np.float64:
                float_audio = np.float32(audio)
            else:
                float_audio = audio

            # Resample if needed
            if orig_sr != resampling_rate:
                float_audio = librosa.core.resample(
                    float_audio,
                    orig_sr=orig_sr,
                    target_sr=resampling_rate,
                    res_type='kaiser_best'
                )
            audio_tensor_list.append(torch.from_numpy(float_audio))
    return audio_tensor_list

##################################### Dimensionality Reduction Functions ########################################

def get_reducers_params():
    params_pca = {
        'reducer__svd_solver': ['auto']
    }
    
    params_tsne = {
        'reducer__perplexity': np.arange(5,155,10),
        'reducer__init': ['random', 'pca']
    }
    
    params_umap = {
        'reducer__n_neighbors': np.arange(10,100,20),
        'reducer__min_dist': np.logspace(0, -5, num=6)
    }
    
    params_pacmap = {
        'reducer__n_neighbors': np.arange(10,100,20),
        'reducer__MN_ratio': [0.1, 0.5, 1],
        'reducer__FP_ratio': [1, 2, 5]
    }

    log_list = ['PCA', 'tSNE', 'UMAP', 'PaCMAP']
    param_list = [params_pca, params_tsne, params_umap, params_pacmap]
    return log_list, param_list

def embedding_quality(X, Z, knn=10, subsetsize=1000):
    nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
    ind1 = nbrs1.kneighbors(return_distance=False)

    nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X.shape[0] / knn

    subset = np.random.choice(X.shape[0], size=subsetsize, replace=True)
    d1 = pdist(X[subset,:])
    d2 = pdist(Z[subset,:])
    rho = scipy.stats.spearmanr(d1[:,None],d2[:,None]).correlation
    return (mnn, rho)

def grid_search_dim_red(name, params, embeddings, knn=10, subsetsize=1000):
    mnns=[]; rhos=[]; red_embeds_list = []; hyperparams=[]
    if name == 'tSNE':
        for perplexity in params['reducer__perplexity']:
            for init in params['reducer__init']:
                print(f'Params: {perplexity} and {init}')
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, init=init)
                red_embeddings = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
                mnn, rho = embedding_quality(embeddings, red_embeddings, knn=knn, subsetsize=subsetsize)
                mnns.append(mnn); rhos.append(rho); red_embeds_list.append(red_embeddings); hyperparams.append([perplexity, init])
    elif name == 'UMAP':
        for n_neighbors in params['reducer__n_neighbors']:
            for min_dist in params['reducer__min_dist']:
                print(f'Params: {n_neighbors} and {min_dist}')
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
                red_embeddings = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
                mnn, rho = embedding_quality(embeddings, red_embeddings, knn=knn, subsetsize=subsetsize)
                mnns.append(mnn); rhos.append(rho); red_embeds_list.append(red_embeddings); hyperparams.append([n_neighbors, min_dist])
    elif name == 'PaCMAP':
        for n_neighbors in params['reducer__n_neighbors']:
            for MN_ratio in params['reducer__MN_ratio']:
                for FP_ratio in params['reducer__FP_ratio']:
                    print(f'Params: {n_neighbors}, {MN_ratio} and {FP_ratio}')
                    reducer = pacmap.PaCMAP(n_dims=2, random_state=42, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)
                    red_embeddings = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
                    mnn, rho = embedding_quality(embeddings, red_embeddings, knn=knn, subsetsize=subsetsize)
                    mnns.append(mnn); rhos.append(rho); red_embeds_list.append(red_embeddings); hyperparams.append([n_neighbors, MN_ratio, FP_ratio])
    else:
        reducer = PCA(n_components=2, random_state=42)
        red_embeddings = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
        mnn, rho = embedding_quality(embeddings, red_embeddings, knn=knn, subsetsize=subsetsize)
        mnns.append(mnn); rhos.append(rho); red_embeds_list.append(red_embeddings)
    return red_embeds_list, mnns, rhos, hyperparams

def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', plot_type='sns', red_name='PCA', row=1, col=1, hovertext='', label='spon'):
    if plot_type == 'sns':
        sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name
                        , style=label_name, palette='deep', ax=axis)
        axis.set(xlabel=None, ylabel=None)
        axis.get_legend().remove()
        if len(metrics) != 0:
            axis.set_title(f'{red_name}: KNN={metrics[0]:0.2f}, CPD={metrics[1]:0.2f}', fontsize=20)
        else:
            axis.set_title(f'{red_name}', fontsize=20)
    elif plot_type == 'plotly':
        traces = px.scatter(x=df[red_name, opt_structure, 'Dim1'], y=df[red_name, opt_structure, 'Dim2'], color=df[label_name].astype(str), hover_name=hovertext)
        traces.layout.update(showlegend=False)
        axis.add_traces(
            list(traces.select_traces()),
            rows=row, cols=col
        )
    else:
        points = axis.scatter(df[red_name, opt_structure, 'Dim1'], df[red_name, opt_structure, 'Dim2'],
                     c=df[label_name], s=20, cmap="Spectral")
        return points

##################################### Embeddings Generation Functions ########################################

def load_model(model_name, device):
    if model_name == 'byols_cnn':
        model = serab_byols.load_model(checkpoint_path, model_name)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        # Disable parameter tuning
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = hub.load(_MODELS_WEIGHTS[model_name])
    return model
        
def generate_speech_embeddings(audio_tensor_list, model, model_name):
    # Generate speech embeddings
    if model_name == 'BYOL-S':
        embeddings = serab_byols.get_scene_embeddings(audio_tensor_list, model)
        # Convert torch tensor to numpy
        embeddings = embeddings.cpu().detach().numpy()
    else:
        for audio in tqdm(audio_tensor_list):
            if model_name == 'trill':
                embedding = np.mean(model(audio, sample_rate=_REQUIRED_SAMPLE_RATE)['layer19'], axis=0, keepdims=True)

            elif model_name == 'vggish':
                embedding = np.mean(model(audio), axis=0, keepdims=True)

            elif model_name == 'yamnet':
                _, embedding, _ = model(audio)
                embedding = np.mean(embedding, axis=0, keepdims=True)
        
    return embeddings

##################################### DF Processing and Stats Functions ########################################
    
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
    
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

##################################### Sklearn ML Functions ########################################

def get_sklearn_models(seed=42):
    """
    Load a family of standard ML classifiers as a dictionary.

    Current models: Logistic Regression, Random Forests and SVC

    Returns
    ----------
    dict
        A dictionary of ML classifiers
        key = model name, value = model (a scikit-learn Estimator)
    """
    
    lr = LogisticRegression(max_iter=1e4, random_state=seed)
    params_lr = {
        'estimator__C': np.logspace(5, -5, num=11),
        'estimator__class_weight': ['balanced', None],
    }

    rf = RandomForestClassifier(random_state=seed)
    params_rf = {
        'estimator__max_depth': range(5, 30, 5),
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__class_weight': ['balanced', None],
    }

    svc = SVC(max_iter=1e4, random_state=seed)
    params_svc = {
        'estimator__kernel': ['linear'],
        'estimator__C': np.logspace(5, -5, num=11),
        'estimator__class_weight': ['balanced', None]
    }

    estimator_list = [lr, rf, svc]
    log_list = ['LR', 'RF', 'SVC']
    param_list = [params_lr, params_rf, params_svc]

    return log_list, estimator_list, param_list

def eval_features_importance(clf_name, estimator):
    """
    Extract the features ordered based on their importance to the classifier
    
    Returns
    ----------
    Features_importances: Pandas DataFrame
    """
    print("Extract important features from {} model:".format(clf_name))
    if clf_name == 'RF':
        feature_importances = pd.DataFrame(np.abs(estimator.best_estimator_.named_steps["estimator"].feature_importances_),
                                                columns=['importance']).sort_values('importance', ascending=False)
    else:
        feature_importances = pd.DataFrame(np.abs(estimator.best_estimator_.named_steps["estimator"].coef_[0]),
                                            columns=['importance']).sort_values('importance', ascending=False)
    return feature_importances