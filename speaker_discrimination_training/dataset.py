"""Dataset Class definition"""

import random
import torch
import torchaudio
from torch.utils.data import Dataset

from encoder import Encoder


class DFInEmbeddingsOutDataset(Dataset):
    """Data Frame in, Encoder embeddings out, dataset class.

    Args:
        df: pandas dataframe containing audio paths and speaker metadata
        encoder_name: name of the encoder used for feature extraction
        encoder_weights: the pre-trained weights path for the encoder
        num_trials: the number of samples generated per epoch
    """

    def __init__(self, df, encoder_name, encoder_weights, num_trials=None, device=torch.device('cpu')):
        super().__init__()
        self.df = df
        self.encoder = Encoder(encoder_name, encoder_weights, device)
        self.num_trials = num_trials
        self.device = device

    def __len__(self):
        if self.num_trials is None:
            return len(self.df)
        return self.num_trials

    def __getitem__(self, _):
        # Randomly select a speaker row from df
        idx = random.randint(0, len(self.df))
        speaker = self.df.iloc[idx]
        # Randomly sample a same and different speaker rows from the first speaker
        same_sp = self.df[(self.df.ID == speaker.ID) & (self.df.AudioPath != speaker.AudioPath)].sample(1)
        diff_sp = self.df[(self.df.SEX == speaker.SEX) & (self.df.ID != speaker.ID)].sample(1)
        # Sanity Check
        assert (speaker.AudioPath != same_sp.AudioPath.values[0]) & (speaker.ID == same_sp.ID.values[0]) & (speaker.ID != diff_sp.ID.values[0])
        # Load audios as torch tensors
        wav_orig, _ = torchaudio.load(speaker.AudioPath)
        wav_same, _ = torchaudio.load(same_sp.AudioPath.values[0])
        wav_diff, _ = torchaudio.load(diff_sp.AudioPath.values[0])
        # Extract embeddings from encoder
        embedd_orig = self.encoder(wav_orig.to(self.device))
        embedd_same = self.encoder(wav_same.to(self.device))
        embedd_diff = self.encoder(wav_diff.to(self.device))
        # assert embedd_orig.dim() == embedd_same.dim() == embedd_diff.dim() == 2

        return embedd_orig, embedd_same, embedd_diff