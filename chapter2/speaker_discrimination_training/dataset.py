"""Dataset Class definition"""

import os
import random
import torch
import torchaudio
import torch.nn as nn
from encoder import Encoder
import torchaudio.transforms as T
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

class TestDataset(Dataset):
    def __init__(self, df, labels, encoder_name, encoder_weights, pooling_method, device=torch.device('cpu')):
        self.df = df
        self.labels = torch.FloatTensor(labels)
        self.device = device
        self.pooling_method = pooling_method
        # define encoder object
        self.encoder = Encoder(encoder_name, encoder_weights, device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select a pair
        pair_df = self.df.iloc[idx]
        labels = self.labels[idx]
        # Read audio files of the selected pair
        path_1 = f'/om2/user/gelbanna/pilot_stimuli/{pair_df.FileName_1}'
        path_2 = f'/om2/user/gelbanna/pilot_stimuli/{pair_df.FileName_2}'
        sp1_audio, sr = torchaudio.load(path_1)
        sp2_audio, sr = torchaudio.load(path_2)
        resampler = T.Resample(sr, 16000)
        sp1_audio = resampler(sp1_audio)
        sp2_audio = resampler(sp2_audio)
        # Extract the embeddings of the selected pair
        sp1_embeddings = self.encoder(sp1_audio.to(self.device))
        sp2_embeddings = self.encoder(sp2_audio.to(self.device))
        # Set temporal pooling method
        if self.pooling_method:
            pooling = Pooling(self.pooling_method)
            sp1_embeddings = pooling(sp1_embeddings)
            sp2_embeddings = pooling(sp2_embeddings)
        return [sp1_embeddings, sp2_embeddings, labels]



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
        # Randomly sample a same and different speaker rows relative to the first speaker
        # if self.df.loc[self.df.ID == speaker.ID].Scentence.unique().shape[0] > 1:
        #     same_sp = self.df[(self.df.ID == speaker.ID) & (self.df.AudioPath != speaker.AudioPath) & (self.df.Scentence != speaker.Scentence)].sample(1)
        # else:
        same_sp = self.df[(self.df.ID == speaker.ID) & (self.df.AudioPath != speaker.AudioPath)].sample(1)
        diff_sp = self.df[(self.df.ID != speaker.ID) & (self.df.SEX == speaker.SEX)].sample(1)
        # & (self.df.SEX == speaker.SEX)
        # Sanity Check
        # assert (speaker.AudioPath != same_sp.AudioPath.values[0]) & (speaker.ID == same_sp.ID.values[0]) & (speaker.ID != diff_sp.ID.values[0]) 
        # assert (speaker.BkG != same_sp.BkG.values[0]) & (same_sp.BkG.values[0] == diff_sp.BkG.values[0])
        
        # Load precomputed embeddings from encoder
        files_path = f'{self.encoder_features_path}/{self.encoder_name}/{self.dataset}'
        orig_filename = os.path.basename(speaker.AudioPath).split('.')[0]
        same_filename = os.path.basename(same_sp.AudioPath.values[0]).split('.')[0]
        diff_filename = os.path.basename(diff_sp.AudioPath.values[0]).split('.')[0]
        embedd_orig = torch.load(f'{files_path}/{orig_filename}.pt')
        embedd_same = torch.load(f'{files_path}/{same_filename}.pt')
        embedd_diff = torch.load(f'{files_path}/{diff_filename}.pt')

        # Set temporal pooling method
        if self.pooling_method:
            pooling = Pooling(self.pooling_method)
            embedd_orig = pooling(embedd_orig)
            embedd_same = pooling(embedd_same)
            embedd_diff = pooling(embedd_diff)

        return [embedd_orig, embedd_same, embedd_diff]