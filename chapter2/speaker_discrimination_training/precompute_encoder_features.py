import os
from glob import glob
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.transforms as T

from encoder import Encoder
from precompute_features_helper import *

#define the config file path
path_to_config = './config.yaml'

#read the config file and get paths for audio files
config = load_yaml_config(path_to_config)
config.encoder_name = 'HuBERT_best'
config.encoder_weights = 'facebook/hubert-xlarge-ll60k'
audio_files = glob(f'{config.audio_dir}train*/*/*/*.flac')
# audio_files = glob('/om2/user/gelbanna/pilot_stimuli/*.flac')
print(f'Dataset has {len(audio_files)} samples')

#define encoder class
encoder = Encoder(config.encoder_name, config.encoder_weights, config.device)

#extract embeddings from audio files and save as torch tensors
path_to_save_pt = f'/om2/scratch/Tue/gelbanna/{config.encoder_name}/train'
# path_to_save_pt = f'/om2/user/gelbanna/pilot_stimuli'
os.makedirs(path_to_save_pt, exist_ok=True)
for audio in tqdm(audio_files, desc=f'Generating {config.encoder_name} Embeddings...'):
    audio_name = os.path.basename(audio).split('.')[0]
    audio_tensor, sr = torchaudio.load(audio)
    # print(sr)
    resampler = T.Resample(sr, 16000)
    audio_tensor = resampler(audio_tensor)
    embedding = encoder(audio_tensor.to(config.device))
    torch.save(embedding.detach().cpu(), f'{path_to_save_pt}/{audio_name}.pt')