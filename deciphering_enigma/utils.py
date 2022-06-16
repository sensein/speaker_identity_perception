import re
import os
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt

import yaml
from pathlib import Path
from easydict import EasyDict

import torch
import tensorflow_hub as hub

from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel

import librosa
import serab_byols
import parselmouth
import soundfile as sf
import pyloudnorm as pyln
from parselmouth.praat import call

from scipy.spatial.distance import pdist, squareform, directed_hausdorff

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .settings import _MODELS_WEIGHTS

import warnings
warnings.filterwarnings('ignore')

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

def build_experiment(exp_config):
    """Create a directory to store experiment's results"""
    #create a directory with the experiment name (datasetname_modelname)
    os.makedirs(f'../{exp_config.dataset_name}', exist_ok=True)
    return sorted(glob(exp_config.dataset_path))

def extract_metadata(exp_config, audio_files):
    metadata = {}
    metadata['AudioNames'] = np.array(list(map(lambda x: os.path.basename(x), audio_files)))
    labels_list = split_string(exp_config.name_convention, ['_', '\.'])
    for idx, label in enumerate(labels_list):
        if label != labels_list[-1]:
            metadata[label] = np.array(list(map(lambda x: split_string(os.path.basename(x), ['_', '\.'])[idx], audio_files)))
        else:
            audio_format = label
    return pd.DataFrame(metadata), audio_format

##################################### Data Loading and Preprocessing Functions ########################################

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
            sf.write(f'{save_path}/{speaker_id}/{Path(file).stem}_{str(i).zfill(3)}.{audio_format}', temp, sr)
    except:
        print(f'The number of seconds in audio {file} is smaller than the splitting criteria ({chunk_dur} sec)')

def balance_data(dataset_path):
    files = []
    wav_dirs = glob(dataset_path)
    for wav_dir in wav_dirs:
        wav_files = np.array(sorted(glob(f'{wav_dir}/*.wav')))
        ids = np.array(list(map(lambda x: os.path.basename(x).split('_')[0], wav_files)))
        labels = np.array(list(map(lambda x: os.path.basename(x).split('_')[3], wav_files)))
        min_label = min(Counter(labels).values())
        script_files = [file for file in wav_files if os.path.basename(file).split('_')[3] == 'script'][:min_label]
        spon_files = [file for file in wav_files if os.path.basename(file).split('_')[3] == 'spon'][:min_label]
        files += spon_files + script_files
    return files

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
    audio_format : str
        The audio extention (e.g. wav/flac/mp3)
    Returns
    -------
    audio_path : str
        path to audio files
    """
    save_path = f'{save_path}_dur{chunk_dur}sec'
    for idx, file in enumerate(tqdm(files_paths, desc=f'Preprocessing Audio Files...', total=len(files_paths))):
        # Read audio file
        file_name = os.path.basename(file).split('.')[0]
        audio, orig_sr = sf.read(file)
        
        # Convert to mono if needed
        if audio.ndim == 2:
            audio = stereo_to_mono_audio(audio)
        
        # Convert to float if needed
        float_audio = int_to_float_audio(audio)
        
        # Resample if needed
        if orig_sr != resampling_rate:
            float_audio = resample_audio(audio, orig_sr, resampling_rate)
        
        # Split audio into chunks
        chunking_audio(float_audio, file, chunk_dur, resampling_rate, save_path, speaker_ids[idx], audio_format)
    audio_path = f'{save_path}/*/*.{audio_format}'
    return audio_path

def load_dataset(files, cfg, speaker_ids=[], audio_format='wav', device='cuda'):
    """Load audio dataset and read the audio files as list of torch tensors.

    Parameters
    ----------
    files : str
        Path for the audio files
    speaker_ids : list
        List of speakers id
    resampling rate : int
        Sample rate required for resampling
    audio_format : str
        Name of audio extension
    Returns
    -------
    audio_tensor_list : list
        list of torch tensors of preprocessed audios
    files : list
        list of audios paths
    """
    audio_tensor_file = f'../{cfg.dataset_name}/audios_tensor_list.pkl'
    if os.path.isfile(audio_tensor_file):
        print(f'Audio Tensors are already saved for {cfg.dataset_name}')
        return pickle.load(open(audio_tensor_file, "rb"))
    else:
        audio_tensor_list = []
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
            if orig_sr != cfg.resampling_rate:
                float_audio = librosa.core.resample(
                    float_audio,
                    orig_sr=orig_sr,
                    target_sr=cfg.resampling_rate,
                    res_type='kaiser_best'
                )
            audio_tensor_list.append(torch.from_numpy(float_audio).to(torch.device(device)))
        with open(audio_tensor_file, 'wb') as f:
            pickle.dump(audio_tensor_list, f)
        return audio_tensor_list

##################################### Embeddings Generation Functions ########################################

def load_model(model_name, weights_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'BYOL-' in model_name:
        model = serab_byols.load_model(weights_file, model_name.split('_')[-1]).to(device)
    elif model_name == 'HuBERT':
        model = HubertModel.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
    elif model_name == 'Wav2Vec2':
        model = Wav2Vec2Model.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
    elif model_name == 'Data2Vec':
        model = Data2VecAudioModel.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
    elif model_name == 'TRILLsson':
        model = hub.KerasLayer('https://tfhub.dev/google/trillsson5/1')
    else:
        model = hub.load(weights_file)
    return model

def generate_speech_embeddings(audio_tensor_list, model, model_name, cfg):
    model_dir = f'../{cfg.dataset_name}/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    emb_file_name = f'{model_dir}/embeddings.npy'
    if os.path.isfile(emb_file_name):
        print(f'{model_name} embeddings are already saved for {cfg.dataset_name}')
        return np.load(emb_file_name)
    else:
        # Generate speech embeddings
        if 'BYOL-' in model_name:
            embeddings = serab_byols.get_scene_embeddings(audio_tensor_list, model)
            # Convert torch tensor to numpy
            embeddings = embeddings.cpu().detach().numpy()
        else:
            embeddings = []
            for audio in tqdm(audio_tensor_list, desc=f'Generating {model_name} Embeddings...'):
                if model_name == 'TRILL':
                    embedding = np.mean(model(audio.cpu().detach().numpy(), sample_rate=cfg.resampling_rate)['layer19'], axis=0)

                elif model_name == 'VGGish':
                    embedding = np.mean(model(audio.cpu().detach().numpy()), axis=0)

                elif model_name == 'YAMNET':
                    _, embedding, _ = model(audio.cpu().detach().numpy())
                    embedding = np.mean(embedding, axis=0)
                    
                elif model_name == 'TRILLsson':
                    embedding = model(audio.unsqueeze(0).cpu().detach().numpy())['embedding']
                    embedding = np.squeeze(embedding)
            
                else:
                    model.eval()
                    with torch.no_grad():
                        embedding = model(audio.unsqueeze(0)).last_hidden_state.mean(1)
                        embedding = np.squeeze(embedding.cpu().detach().numpy())
                embeddings.append(embedding)
            embeddings = np.array(embeddings)
        np.save(emb_file_name, embeddings)
        return embeddings

def standardize_embeddings(embeddings_dict):
    stand_embeddings_dict = {}
    for model_name, embeddings in embeddings_dict.items():
        stand_embeddings_dict[model_name] = StandardScaler().fit_transform(embeddings)
    return stand_embeddings_dict

def extract_models(audio_tensor_list, cfg):
    embeddings = {}
    for model_name, weights_file in _MODELS_WEIGHTS.items():
        print(f'Load {model_name} Model')
        model_dir = f'../{cfg.dataset_name}/{model_name}'
        emb_file_name = f'{model_dir}/embeddings.npy'
        if os.path.isfile(emb_file_name):
            print(f'{model_name} embeddings are already saved for {cfg.dataset_name}')
            embeddings[model_name] = np.load(emb_file_name)
        else:
            model = load_model(model_name, weights_file)
            embeddings[model_name] = generate_speech_embeddings(audio_tensor_list, model, model_name, cfg)
        print(embeddings[model_name].shape)
    return embeddings

##################################### High Dimension Analysis Functions ######################################## 
 
def compute_hausdorff_dist(u,v):
    return directed_hausdorff(np.expand_dims(u, axis=0),np.expand_dims(v, axis=0))

def process_distances(long_form, dataset_name):
    norm_dist_df_dict = {}
    #remove distances computed between different speakers and different labels
    long_form = long_form.loc[(long_form['Label_1']==long_form['Label_2']) & (long_form['ID_1']==long_form['ID_2'])]
    #remove duplicate distances
    long_form = long_form.drop_duplicates(subset=['Distance'])
    #standardize distances within speaker
    long_form['Distance'] = long_form.groupby(['ID_1'])['Distance'].transform(lambda x: (x - x.mean()) / x.std())
    #remove distances above 99% percentile
    long_form = long_form[long_form.Distance < np.percentile(long_form.Distance,99)]
    #standardize distances to be comparable with other models
    # long_form['Distance'] = (long_form['Distance']-long_form['Distance'].mean())/long_form['Distance'].std()
    return long_form

def compute_distances(metadata_df, embeddings_dict, dataset_name, dist_metric, columns_list):
    distance_df_dict = {}
    if dist_metric == 'hausdorff':
        dist_metric = compute_hausdorff_dist
    for model_name, embeddings in embeddings_dict.items():
        long_form_df_path = f'../{dataset_name}/{model_name}/longform_{dist_metric}_norm_distance_perspeakerlabel.csv'
        if os.path.isfile(long_form_df_path):
            print(f'DF for the {dist_metric} distances using {model_name} already exist!')
            distance_df_dict[model_name] = pd.read_csv(long_form_df_path)
        else:
            print(f'Computing {dist_metric} distances between {model_name} embeddings...')
            #add embeddings to metadata dataframe
            df_embeddings = pd.DataFrame(embeddings)
            df_embeddings = df_embeddings.add_prefix('Embeddings_')
            df = pd.concat([metadata_df, df_embeddings], axis=1)
            df.to_csv(f'../{dataset_name}/{model_name}/df_embeddings.csv')

            #create distance-based dataframe between all data samples in a square form
            pairwise = pd.DataFrame(
                squareform(pdist(df_embeddings, dist_metric)),
                columns = df[columns_list],
                index = df[columns_list]
            )
            pairwise.to_csv(f'../{dataset_name}/{model_name}/pairwise_{dist_metric}_distance.csv')

            #move from square form DF to long form DF
            long_form = pairwise.unstack()
            #rename columns and turn into a dataframe
            long_form.index.rename(['Sample_1', 'Sample_2'], inplace=True)
            long_form = long_form.to_frame('Distance').reset_index()

            #expand Sample 1 & 2 to meta data columns
            rename_dict = dict(zip(list(range(len(list(metadata_df.columns)))), list(metadata_df.columns)))
            sample1_df = pd.DataFrame(long_form['Sample_1'].tolist(),index=long_form.index).rename(columns=rename_dict).add_suffix('_1')
            sample2_df = pd.DataFrame(long_form['Sample_2'].tolist(),index=long_form.index).rename(columns=rename_dict).add_suffix('_2')
            long_form = pd.concat([sample1_df, sample2_df, long_form['Distance']], axis=1)

            #remove the distances computed between same samples (distance = 0)
            long_form = long_form.loc[long_form['AudioNames_1'] != long_form['AudioNames_2']]
            long_form['Model'] = model_name
            long_form = process_distances(long_form, dataset_name)
            long_form.to_csv(long_form_df_path)
            distance_df_dict[model_name] = long_form
    df_all = pd.concat(distance_df_dict.values(), ignore_index=True)
    df_all.to_csv(f'../{dataset_name}/allmodels_{dist_metric}_distances.csv')
    return df_all

def visualize_violin_dist(df_all):
    fig, ax = plt.subplots(1, 1, figsize=(30, 10))
    violin = sns.violinplot(data=df_all, x='Model', y='Distance', inner='quartile', hue='Label_1', split=True, ax=ax)
    ax.set(xlabel=None, ylabel=None)
    ax.set_xticklabels(ax.get_xticklabels(), size = 15)
    ax.set_yticklabels(ax.get_yticks(), size = 15)
    ax.set_ylabel('Standardized Cosine Distances', fontsize=20)
    ax.set_xlabel('Models', fontsize=20)

    # statistical annotation
    y, h, col = df_all['Distance'].max() + df_all['Distance'].max()*0.05, df_all['Distance'].max()*0.01, 'k'
    for i, model_name in enumerate(df_all['Model'].unique()):
        d=cohend(df_all['Distance'].loc[(df_all.Label_1=='spon') & (df_all.Model==model_name)], df_all['Distance'].loc[(df_all.Label_1=='script') & (df_all.Model==model_name)])
        x1, x2 = -0.25+i, 0.25+i
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+(h*1.5), f'cohen d={d:.2}', ha='center', va='bottom', color=col, fontsize=15)
    violin.legend(fontsize = 15, \
                   bbox_to_anchor= (1, 1), \
                   title="Labels", \
                   title_fontsize = 18, \
                   shadow = True, \
                   facecolor = 'white');
    plt.tight_layout()

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

def visualize_embeddings(df, label_name, metrics=[], axis=[], acoustic_param={}, opt_structure='Local', plot_type='sns', red_name='PCA', row=1, col=1, hovertext='', label='spon'):
    if plot_type == 'sns':
        if label_name == 'Gender':
            sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name, palette='deep', ax=axis)
        else:
            sns.scatterplot(data=df, x=(red_name, opt_structure, 'Dim1'), y=(red_name, opt_structure, 'Dim2'), hue=label_name
                            , style=label_name, palette='deep', ax=axis)
        axis.set(xlabel=None, ylabel=None)
        axis.get_legend().remove()
    elif plot_type == 'plotly':
        traces = px.scatter(x=df[red_name, opt_structure, 'Dim1'], y=df[red_name, opt_structure, 'Dim2'], color=df[label_name].astype(str), hover_name=hovertext)
        traces.layout.update(showlegend=False)
        axis.add_traces(
            list(traces.select_traces()),
            rows=row, cols=col
        )
    else:
        points = axis.scatter(df[red_name, opt_structure, 'Dim1'], df[red_name, opt_structure, 'Dim2'],
                     c=df[label_name], s=10, cmap="Spectral")
        return points

def add_os_features(embeddings_dict, dataset_name, audio_files_orig, audio_files):
    os_features = np.load(f'../{dataset_name}/os_egemaps_embeddings_script_spon_5sec.npy')
    _,_,idx=np.intersect1d(audio_files_orig, audio_files, return_indices=True)
    os_features = os_features[idx, :]

    freq_indices = list(np.arange(0,10,1))+list(np.arange(30,32,1))+list(np.arange(40,58,1))
    spectral_indices = list(np.arange(20,30,1))+list(np.arange(36,40,1))+list(np.arange(58,81,1))
    energy_indices = list(np.arange(10,20,1))+list(np.arange(32,36,1))+list(np.arange(81,88,1))

    embeddings_dict['OS_Frequency'] = os_features[:, freq_indices]
    embeddings_dict['OS_Spectral'] = os_features[:, spectral_indices]
    embeddings_dict['OS_Energy'] = os_features[:, energy_indices]
    return embeddings_dict
    
def run_cka(cka_class, embeddings_dict):
    num_models = len(embeddings_dict.keys())
    cka_ = np.zeros((num_models, num_models))
    print(cka_.shape)
    for i, (_, model_1) in enumerate(tqdm(embeddings_dict.items())):
        for j, (_, model_2) in enumerate(embeddings_dict.items()):
            cka_[i,j] = cka_class.compute(model_1, model_2)
    return cka_

##################################### Sklearn ML Functions ########################################

def eval_features_importance(clf_name, estimator):
    """
    Extract the features ordered based on their importance to the classifier
​
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

def compute_acoustic_features(audio_files, save_path='.', feature='f0', mfcc_num=1):
    feature_values = []
    for file in tqdm(audio_files):
        audio, orig_sr = librosa.load(file)
        if feature == 'f0':
            #measure the median fundamental frequency
            f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C1'),
                                    fmax=librosa.note_to_hz('C5'), sr=orig_sr)
            feature_values.append(np.nanmedian(f0))
        elif feature == 'rms':
            feature_values.append(np.nanmedian(librosa.feature.rms(audio)))
        elif feature == 'mfcc':
            #measure the first mfcc
            mfccs = librosa.feature.mfcc(audio, sr=orig_sr)
            feature_values.append(np.nanmedian(mfccs[mfcc_num-1,:]))
        elif feature == 'num_syl':
            feature_values.append(speech_rate(file)['nsyll'])
    feature_values = np.array(feature_values)
    with open(f"{save_path}/{feature}.npy", "wb") as output_file:
        np.save(output_file, feature_values)
    return feature_values

def speech_rate(filename):
    silencedb = -25
    mindip = 2
    minpause = 0.3

    # print a single header line with column names and units
    # cols = ['soundname', 'nsyll', 'npause', 'dur(s)', 'phonationtime(s)', 'speechrate(nsyll / dur)', 'articulation '
    #        'rate(nsyll / phonationtime)', 'ASD(speakingtime / nsyll)']
    # df = pd.DataFrame(columns = cols)

    sound = parselmouth.Sound(filename)
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    # asd = speakingtot / voicedcount
    speechrate_dictionary = {'soundname':filename,
                             'nsyll':voicedcount,
                             'npause': npause,
                             'dur(s)':originaldur,
                             'phonationtime(s)':intensity_duration,
                             'speechrate(nsyll / dur)': speakingrate,
                             "articulation rate(nsyll / phonationtime)":articulationrate,
                             # "ASD(speakingtime / nsyll)":asd
                            }
    return speechrate_dictionary