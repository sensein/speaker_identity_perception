import os
from glob import glob
from tqdm import tqdm

import torch
import torchaudio

from encoder import Encoder
from precompute_features_helper import *

#define the config file path
path_to_config = './config.yaml'

#read the config file and get paths for audio files
config = load_yaml_config(path_to_config)
audio_files = glob(f'{config.audio_dir}test*/*/*/*.flac')
print(f'Dataset has {len(audio_files)} samples')

#define encoder class
encoder = Encoder(config.encoder_name, config.encoder_weights, config.device)

#extract embeddings from audio files and save as torch tensors
path_to_save_pt = f'/om2/scratch/Wed/gelbanna/{config.encoder_name}/test'
os.makedirs(path_to_save_pt, exist_ok=True)
for audio in tqdm(audio_files, desc=f'Generating {config.encoder_name} Embeddings...'):
    audio_name = os.path.basename(audio).split('.')[0]
    audio_tensor, sr = torchaudio.load(audio)
    assert sr == 16000
    embedding = encoder(audio_tensor.to(config.device))
    torch.save(embedding.detach().cpu(), f'{path_to_save_pt}/{audio_name}.pt')