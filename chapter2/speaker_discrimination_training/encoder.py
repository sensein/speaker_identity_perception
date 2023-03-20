"""Encoder Definitions
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import s3prl.hub as s3hub
import librosa
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T

import serab_byols

from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel


class Encoder(nn.Module):
    def __init__(self, model_name, model_weights, device):
        super().__init__()
        with torch.no_grad():
            self.encoder_name = model_name
            self.encoder = self.load_model(model_name, model_weights, device)

    def load_model(self, model_name, weights_file, device):
        if 'BYOL-' in model_name:
            model = serab_byols.load_model(weights_file, model_name.split('_')[-1]).to(device)
        elif 'HuBERT' in model_name:
            model = HubertModel.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif 'Wav2Vec2' in model_name:
            model = Wav2Vec2Model.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif 'Data2Vec' in model_name:
            model = Data2VecAudioModel.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif model_name == 'APC':
            model = getattr(s3hub, 'apc')().to(device)
        elif model_name == 'TERA':
            model = getattr(s3hub, 'tera')().to(device)
        elif model_name == 'TRILLsson':
            model = hub.KerasLayer('https://tfhub.dev/google/trillsson5/1')
        elif model_name == 'Log-Mel-Spectrogram':
            model = mel_spec_fn = T.MelSpectrogram(
                        sample_rate=16000,
                        n_fft=2048,
                        win_length=None,
                        hop_length=512,
                        n_mels=128,
                        f_min=5,
                        f_max=20000,
                        power=2,
                        ).to(device)
        elif model_name == 'Cochleagram':
            model = cgram.human_cochleagram
        else:
            raise ValueError('Encoder not found.')
        return model

    def generate_speech_embeddings(self, audio_tensor, model, model_name):
        # Generate speech embeddings
        if 'BYOL-' in model_name:
            embedding = serab_byols.get_scene_embeddings(audio_tensor, model)

        elif model_name == 'TERA' or model_name == 'APC':
            with torch.no_grad():
                embedding = model(audio_tensor)["last_hidden_state"]

        elif model_name == 'Log-Mel-Spectrogram':
            mel_spec = model(audio_tensor)
            embedding = (mel_spec + torch.finfo().eps).log().squeeze(0)
        
        elif model_name == 'Cochleagram':
            coch = model(audio_tensor.cpu().detach().numpy(), cfg.resampling_rate, strict=False, n=40)
            embedding = (torch.from_numpy(coch) + torch.finfo().eps).log().squeeze(0)
            
        else:
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    if 'encoder' in name:
                        activation[name] = output[0].detach()
                    else:
                        activation[name] = output.detach()
                return hook
            model.eval()
            with torch.no_grad():
                if 'latent' in model_name:
                    if 'HuBERT' in model_name:
                        model.feature_extractor.register_forward_hook(get_activation('latent_features'))
                        _ = model(audio_tensor)
                        embedding = activation['latent_features'].permute(0,2,1)
                    else:
                        embedding = model(audio_tensor).extract_features
                elif 'best' in model_name:
                    if 'HuBERT' in model_name:
                        num = 6
                    elif 'Wav2Vec2' in model_name:
                        num = 0
                    elif 'Data2Vec' in model_name:
                        num = 3
                    model.encoder.layers[num].register_forward_hook(get_activation(f'{model_name}_encoder'))
                    _ = model(audio_tensor)
                    embedding = activation[f'{model_name}_encoder']
                else:
                    embedding = model(audio_tensor).last_hidden_state
        return embedding

    def forward(self, audio_tensor):
        embeddings = self.generate_speech_embeddings(audio_tensor, self.encoder, self.encoder_name)
        return embeddings