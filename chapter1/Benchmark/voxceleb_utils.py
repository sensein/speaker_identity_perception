import os
import sys
import shutil
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import copy
import re
import logging
from collections import defaultdict
from itertools import chain
import subprocess

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
from torch.utils.data import Dataset

from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from .feature_extractor import FeatureExtractor

def get_data(base_path, subset='all'):
    audio_files = glob(base_path)
    labels = list(map(lambda x: x.split('/')[-3], audio_files))
    df_audios = pd.DataFrame({'audio_file': audio_files, 'label': labels})
    df_audios['audio'] = df_audios.audio_file.apply(lambda x: '/'.join(x.split('/')[-3:]))
    # replace all str label with int label
    df_audios['label_numeric'] = df_audios.label.map({l: i for i, l in enumerate(df_audios.label.unique())})

    df_split = pd.read_csv("/om2/user/gelbanna/datasets/VoxCeleb1/iden_split.txt", sep=" ", header=None)
    df_split.rename(columns={0: 'split', 1: 'audio'}, inplace=True)
    df_split['label'] = df_split.audio.apply(lambda x: x.split('/')[0])

    df_metadata = pd.read_csv("/om2/user/gelbanna/datasets/VoxCeleb1/vox1_meta.csv", sep='\t')
    df_metadata.rename(columns={'VoxCeleb1 ID': 'label'}, inplace=True)
    df_metadata['label_numeric'] = df_metadata.label.map({l: i for i, l in enumerate(df_metadata.label.unique())})
    df_metadata.drop(columns=['Set'], inplace=True)

    df = pd.merge(df_audios, df_split, how='left', on='audio')
    df.drop(columns=['audio', 'label_x', 'label_y'], inplace=True)
    df = pd.merge(df, df_metadata, how='left', on='label_numeric')
    df.rename(columns={'VGGFace1 ID': 'ID'}, inplace=True)
    
    if subset != 'all':
        np.random.seed(seed=42)
        male_ids = np.random.choice(df.loc[df.Gender == 'm'].ID.unique(), subset//2)
        female_ids = np.random.choice(df.loc[df.Gender == 'f'].ID.unique(), subset//2)
        ids = np.concatenate((male_ids, female_ids))
        df = df.loc[df.ID.isin(ids)].reset_index(drop=True)
    
    idxs = [None, None, None]
    idxs[0] = df[df.split == 1].index.values
    idxs[1] = df[df.split == 2].index.values
    idxs[2] = df[df.split == 3].index.values
    return df, idxs
    

def load_metadata(base_path, subset):
    df, fold_idxs = get_data(base_path, subset)
    return df, fold_idxs


class BaseDataSource:
    """Data source base class, see TaskDataSource for the detail."""

    def __init__(self, df, fold_idxes):
        self.df, self.fold_idxes = df, fold_idxes
        self.get_idxes = None

    @property
    def labels(self):
        if self.get_idxes is not None:
            return self.df.label_numeric.values[self.get_idxes]
        return self.df.label_numeric.values

    def __len__(self):
        if self.get_idxes is None:
            return len(self.df)
        return len(self.get_idxes)

    def index_of_folds(self, folds):
        idxes = []
        for fold in folds:
            idxes.extend(self.fold_idxes[fold])
        return idxes

    def subset_by_idxes(self, idxes):
        dup = copy.copy(self)
        dup.get_idxes = idxes
        return dup

    def subset(self, folds):
        """Returns a subset data source for the fold indexes.
        folds: List of fold indexes.
        """
        return self.subset_by_idxes(self.index_of_folds(folds))

    @property
    def n_folds(self):
        return len(self.fold_idxes)

    @property
    def n_classes(self):
        return len(set(self.df.label.values))

    def real_index(self, index):
        if self.get_idxes is not None:
            index = self.get_idxes[index]
        return index


class TaskDataSource(BaseDataSource):
    """Downstream task data source class.

    This class provides files and metadata in the dataset,
    as well as methods to manage data splits/folds.

    Properties:
        files: Audio sample pathnames.
        labels: List of int, class indexes of samples.
        classes: List of int, possible class indexes. [0, 1, 2, ] for example of 3 classes.
        n_classes: Number of classes
        n_folds: Number of folds.

    Methods:
        subset: Makes subset data source access.
    """

    def __init__(self, base_path, subset):
        super().__init__(*load_metadata(base_path, subset))
        self.audio_paths = base_path

    def file_name(self, index):
        index = self.real_index(index)
        return self.df.audio_file.values[index]

    @property
    def files(self):
        return [self.file_name(i) for i in range(len(self))]

    
class WavInEmbeddOutDataset(Dataset, FeatureExtractor):

    def __init__(self, audio_files, model, model_name, cfg):
        super().__init__()

        # initializations
        self.files = audio_files
        self.model = model
        self.model_name = model_name
        self.cfg = cfg

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load single channel .wav audio
        wav, sr = torchaudio.load(self.files[idx])
        return np.squeeze(self.generate_speech_embeddings(wav, self.model, self.model_name, self.cfg, False, True), axis=0)


def get_embeddings(files, model_name, weight_file, exp_config):
    feature_extractor = FeatureExtractor()
    model = feature_extractor.load_model(model_name, weight_file, exp_config)
    ds = WavInEmbeddOutDataset(files, model, model_name, exp_config)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, pin_memory=False, shuffle=False, drop_last=False)
    embs = []
    for X in tqdm(dl):
        embs.extend(X.numpy())
    return np.array(embs)


def speaker_normalization(embeddings, speaker_ids):
    """Normalize embedding features by per-speaker statistics."""
    all_speaker_ids = np.unique(speaker_ids)
    for speaker in all_speaker_ids:
        speaker_features = speaker_ids == speaker

        # Normalize feature mean.
        embeddings[speaker_features] -= embeddings[speaker_features].mean(axis=0)

        # Normalize feature variance.
        stds = embeddings[speaker_features].std(axis=0)
        stds[stds == 0] = 1
        embeddings[speaker_features] /= stds
    return embeddings


def get_splits(x, y):
    split_indices = np.repeat([-1, 0], [x.shape[0], y.shape[0]])
    splits = PredefinedSplit(split_indices)
    return splits


def build_pipeline(clf_params, splits, hidden_sizes, epochs, early_stopping, verbose, scoring):
    clf = MLPClassifier(hidden_layer_sizes=hidden_sizes, max_iter=epochs,
              early_stopping=early_stopping, verbose=verbose)
    pipeline = Pipeline([('transformer', StandardScaler()), ('clf', clf)])
    grid_pipeline = GridSearchCV(pipeline, param_grid=clf_params, n_jobs=-1, cv=splits, scoring=scoring, verbose=3)
    return grid_pipeline


def linear_eval(X, y, X_test, y_test, pipeline):
    """Perform a single run of linear evaluation."""
    pipeline.fit(X, y)
    score = pipeline.score(X_test, y_test)
    return score