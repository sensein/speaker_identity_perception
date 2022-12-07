import os
from glob import glob
from tqdm import tqdm

#define the config file path
path_to_config = './config.yaml'

#read the config file and get paths for audio files
config = load_yaml_config(path_to_config)
audio_files = glob(f'{cfg.audio_dir}train*/*/*/*.flac')
print(f'Dataset has {len(audio_files)} samples')

