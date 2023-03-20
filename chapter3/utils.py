#Import packages
import os
import pickle
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import moviepy
import soundfile as sf

from pyannote.audio import Pipeline

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel

from scipy.stats import pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler

from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV

import nilearn
import nibabel as nib
import hcp_utils as hcp
from nilearn import image as nimg
from nilearn import plotting as nplot
from bids.layout import BIDSLayout

import cortex
from cortex import fmriprep

from nipype.interfaces.workbench.base import WBCommand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

activation = {}

def get_activation(name):
    def hook(model, input, output):
        if 'encoder' in name:
            activation[name] = output[0].detach()
        else:
            activation[name] = output.detach()
    return hook

def load_model(model_name, device=torch.device('cpu')):
        if 'HuBERT' in model_name:
            model = HubertModel.from_pretrained('facebook/hubert-xlarge-ll60k', 
                                                cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif 'Wav2Vec2' in model_name:
            model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-960h-lv60-self', 
                                                  cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif 'Data2Vec' in model_name:
            model = Data2VecAudioModel.from_pretrained('facebook/data2vec-audio-large-960h', 
                                                       cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        return model

class WavInEmbeddOutDataset(Dataset):

    def __init__(self, audio_files, model, model_name, layer_name):
        super().__init__()

        # initializations
        self.files = audio_files
        self.model = model
        self.model_name = model_name
        self.layer_name = layer_name

    def __len__(self):
        return len(self.files)
    
    def generate_speech_embeddings(self, wav):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(wav.to('cuda'))
            if 'encoder' in self.layer_name:
                embedding = activation[self.layer_name]
            else:
                embedding = activation[self.layer_name].permute(0,2,1)
            embedding = embedding.mean(1) + embedding.amax(1)
            embedding = np.squeeze(embedding.cpu().detach().numpy())
        return embedding

    def __getitem__(self, idx):
        # load single channel .wav audio
        wav, sr = torchaudio.load(self.files[idx])
        return self.generate_speech_embeddings(wav)

def get_embeddings(files, model, model_name, layer_name):
    ds = WavInEmbeddOutDataset(files, model, model_name, layer_name)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, pin_memory=False, shuffle=False, drop_last=False)
    embs = []
    for X in tqdm(dl):
        embs.extend(X.numpy())
    return np.array(embs)

def get_voxel_labels(ciftis_path, mask):
    Y_data = defaultdict(list)
    run_onsets = []
    run_size = 0
    for idx in range(8):
        #read the preprocessed cifti file as an array
        func_img = nimg.load_img(ciftis_path[idx])
        func_signal = np.array(func_img.dataobj)
        func_signal = func_signal[:, mask]
        if idx == 7:
            func_signal = func_signal[:-1,:]
        if idx != 6:
            Y_data['Train'].append(func_signal)
            run_onsets.append(run_size)
            run_size += func_signal.shape[0]
        else:
            Y_data['Test'].append(func_signal)
    Y_train = np.concatenate(Y_data['Train'])
    Y_test = Y_data['Test'][0]
    return Y_train, Y_test, run_onsets

def split_data(model_name, layer_name, embeddings_path):
    X_data = defaultdict(list)
    for idx in range(8):
        #read model embeddings for this segment/run
        embeddings = np.load(f'{embeddings_path}/seg{idx}/{model_name}/{layer_name}_embeddings.npy')
        if idx != 6:
            X_data['Train'].append(embeddings)
        else:
            embeddings = embeddings[:-1,:]
            X_data['Test'].append(embeddings)
    X_train = np.concatenate(X_data['Train'])
    X_test = X_data['Test'][0]
    return X_train, X_test

def avg_pearsonr(pred, truth):
    corrs=[]
    for i in range(pred.shape[0]):
        corr, _ = pearsonr(pred[i,:], truth[i,:])
        corrs.append(corr)
    return np.mean(corrs)



"""This code is adapted from https://github.com/gallantlab/voxelwise_tutorials/"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_random_state


def generate_leave_one_run_out(n_samples, run_onsets, random_state=None,
                               n_runs_out=1):
    """Generate a leave-one-run-out split for cross-validation.
    Generates as many splits as there are runs.
    Parameters
    ----------
    n_samples : int
        Total number of samples in the training set.
    run_onsets : array of int of shape (n_runs, )
        Indices of the run onsets.
    random_state : None | int | instance of RandomState
        Random state for the shuffling operation.
    n_runs_out : int
        Number of runs to leave out in the validation set. Default to one.
    Yields
    ------
    train : array of int of shape (n_samples_train, )
        Training set indices.
    val : array of int of shape (n_samples_val, )
        Validation set indices.
    """
    random_state = check_random_state(random_state)

    n_runs = len(run_onsets)
    # With permutations, we are sure that all runs are used as validation runs.
    # However here for n_runs_out > 1, a run can be chosen twice as validation
    # in the same split.
    all_val_runs = np.array(
        [random_state.permutation(n_runs) for _ in range(n_runs_out)])

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs have no samples. Check that run_onsets "
                         "does not include any repeated index, nor the last "
                         "index.")

    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack([runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val

        
class Delayer(BaseEstimator, TransformerMixin):
    """Scikit-learn Transformer to add delays to features.
    This assumes that the samples are ordered in time.
    Adding a delay of 0 corresponds to leaving the features unchanged.
    Adding a delay of 1 corresponds to using features from the previous sample.
    Adding multiple delays can be used to take into account the slow
    hemodynamic response, with for example `delays=[1, 2, 3, 4]`.
    Parameters
    ----------
    delays : array-like or None
        Indices of the delays applied to each feature. If multiple values are
        given, each feature is duplicated for each delay.
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during the fit.
    Example
    -------
    >>> from sklearn.pipeline import make_pipeline
    >>> from voxelwise_tutorials.delayer import Delayer
    >>> from himalaya.kernel_ridge import KernelRidgeCV
    >>> pipeline = make_pipeline(Delayer(delays=[1, 2, 3, 4]), KernelRidgeCV())
    """

    def __init__(self, delays=None):
        self.delays = delays

    def fit(self, X, y=None):
        """Fit the delayer.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.
        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values. Ignored.
        Returns
        -------
        self : returns an instance of self.
        """
        X = self._validate_data(X, dtype='numeric')
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Transform the input data X, copying features with different delays.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data.
        Returns
        -------
        Xt : array of shape (n_samples, n_features * n_delays)
            Transformed data.
        """
        check_is_fitted(self)
        X = check_array(X, copy=True)

        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        if self.delays is None:
            return X

        X_delayed = np.zeros((n_samples, n_features * len(self.delays)),
                             dtype=X.dtype)
        for idx, delay in enumerate(self.delays):
            beg, end = idx * n_features, (idx + 1) * n_features
            if delay == 0:
                X_delayed[:, beg:end] = X
            elif delay > 0:
                X_delayed[delay:, beg:end] = X[:-delay]
            elif delay < 0:
                X_delayed[:-abs(delay), beg:end] = X[abs(delay):]

        return X_delayed

    def reshape_by_delays(self, Xt, axis=1):
        """Reshape an array, splitting and stacking across delays.
        Parameters
        ----------
        Xt : array of shape (n_samples, n_features * n_delays)
            Transformed array.
        axis : int, default=1
            Axis to split.
        Returns
        -------
        Xt_split :array of shape (n_delays, n_samples, n_features)
            Reshaped array, splitting across delays.
        """
        delays = self.delays or [0]  # deals with None
        return np.stack(np.split(Xt, len(delays), axis=axis))