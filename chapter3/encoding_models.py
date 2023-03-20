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

from utils import *

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


#base directory for fmriprep output
fmriprep_dir = './forrestgump_project/merge_ds/fmriprep-22.1.0'
layout = BIDSLayout(fmriprep_dir, validate=False, config=['bids','derivatives'])

model_components = ['feature_extractor.conv_layers', 'encoder.layers']

#define ROIs using MMP regions (42 regions)
assoc_aud_regions = [107,123,125,128,129,130,175,176,287,303,305,308,309,310,355,356]
early_aud_regions = [24,104,124,173,174,204,284,304,353,354]
ifc_regions = [74,75,76,79,80,81,82,171,254,255,256,259,260,261,262,351]
regions = assoc_aud_regions + ear_aud_regions + ifc_regions
mask = np.logical_or.reduce([hcp.mmp.map_all == value for value in regions])

#query list of subjects
subjects = layout.get_subjects()
subjects_results = defaultdict(list)
#query list of subjects
subjects = layout.get_subjects()
subjects_results = defaultdict(list)
for sub in tqdm(subjects):
    if not os.path.exists(f'sub-{sub}_aud_results.pkl'):
        if sub != '10':
            print('Subject: ',sub)
            #path for preprocessed cifti files
            ciftis_path = glob(f'forrestgump_project/merge_ds/fmriprep-22.1.0/sub-{sub}/ses-forrestgump/cleaned/*')
            #path for model's embeddings
            embeddings_path = f'stimuli_videos/chunked_stimuli'

            Y_train, Y_test, run_onsets = get_voxel_labels(ciftis_path, mask)
            n_samples_train = Y_train.shape[0]
            cv = generate_leave_one_run_out(n_samples_train, run_onsets)
            cv = check_cv(cv)

            #use GPU for training
            backend = set_backend("torch_cuda", on_error="warn")
            #hyperparameters for ridge regression
            alphas = np.logspace(1,20,20)
            delays = [list(range(i)) for i in range(3, 7)]
            #standardize features N~(0,1)
            scaler = StandardScaler()

            for model_name in ['Wav2Vec2','HuBERT','Data2Vec']:
                model = load_model(model_name)
                for component in model_components:
                    num_layers = len(dict(model.named_modules())[component])
                    for layer in range(num_layers):
                        print(f'Evaluating embeddings for {model_name}-{component}{layer}...')
                        layer_name = f'{component}{layer}'
                        X_train, X_test = split_data(model_name, layer_name, embeddings_path)
                        for delay in delays:
                            delayer = Delayer(delay)
                            pipeline = make_pipeline(
                                scaler,
                                delayer,
                                KernelRidgeCV(
                                    alphas=alphas, cv=cv,
                                    solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                                                    n_targets_batch_refit=100)),
                            )
                            pipeline.fit(X_train, Y_train)
                            Y_pred = pipeline.predict(X_test)
                            Y_pred = backend.to_numpy(Y_pred)
                            scores = pipeline.score(X_test,Y_test)
                            scores = backend.to_numpy(scores)
                            avg_corr = avg_pearsonr(Y_pred, Y_test)
                            subjects_results['Subject'].append(sub)
                            subjects_results['Model'].append(model_name)
                            subjects_results['Layer'].append(layer_name)
                            subjects_results['Avg_Corr'].append(avg_corr)
                            subjects_results['R2_Scores'].append(scores)
            df = pd.DataFrame(subjects_results)
            df.to_pickle(f'sub-{sub}_aud_results.pkl')