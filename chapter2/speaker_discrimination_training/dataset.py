"""Dataset Class definition"""

import os
import random
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset

class Pooling(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
    def forward(self, x):
        if self.method == 'mean':
            return torch.mean(x, dim=1)
        elif self.method == 'max':
            (x, _) = torch.max(x, dim=1)
            return x
        elif self.method == 'mean+max':
            x_mean = torch.mean(x, dim=1)
            (x_max, _) = torch.max(x, dim=1)
            return x_mean+x_max

class DFInEmbeddingsOutDataset(Dataset):
    """DataFrame in, Encoder Embeddings out, dataset class.

    Args:
        df: pandas dataframe containing audio paths and speaker metadata
        encoder_name: name of the encoder used for feature extraction
        pooling_method: the temporal pooling method applied on the extracted embeddings 
        num_trials: the number of samples generated per epoch
    """

    def __init__(self, df, encoder_name, encoder_features_path, pooling_method, dataset='train', num_trials=None, device=torch.device('cpu')):
        super().__init__()
        self.df = df
        self.encoder_name = encoder_name
        self.encoder_features_path = encoder_features_path
        self.pooling_method = pooling_method
        self.num_trials = num_trials
        self.device = device
        self.dataset = dataset

    def __len__(self):
        if self.num_trials is None:
            return len(self.df)
        return self.num_trials

    def __getitem__(self, _):
        # Randomly select a speaker row from df
        idx = random.randint(0, len(self.df)-1)
        speaker = self.df.iloc[idx]
        # Randomly sample a same and different speaker rows from the first speaker
        same_sp = self.df[(self.df.ID == speaker.ID) & (self.df.AudioPath != speaker.AudioPath)].sample(1)
        diff_sp = self.df[(self.df.SEX == speaker.SEX) & (self.df.ID != speaker.ID)].sample(1)
        # Sanity Check
        assert (speaker.AudioPath != same_sp.AudioPath.values[0]) & (speaker.ID == same_sp.ID.values[0]) & (speaker.ID != diff_sp.ID.values[0])
        # Set temporal pooling method

        pooling = Pooling(self.pooling_method)
        # Load precomputed embeddings from encoder
        files_path = f'{self.encoder_features_path}/{self.encoder_name}/{self.dataset}'
        orig_filename = os.path.basename(speaker.AudioPath).split('.')[0]
        same_filename = os.path.basename(same_sp.AudioPath.values[0]).split('.')[0]
        diff_filename = os.path.basename(diff_sp.AudioPath.values[0]).split('.')[0]
        embedd_orig = torch.load(f'{files_path}/{orig_filename}.pt')
        embedd_same = torch.load(f'{files_path}/{same_filename}.pt')
        embedd_diff = torch.load(f'{files_path}/{diff_filename}.pt')

        return [pooling(embedd_orig), pooling(embedd_same), pooling(embedd_diff)]