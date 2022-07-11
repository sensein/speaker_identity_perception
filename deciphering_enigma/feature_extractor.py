import os
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import s3prl.hub as s3hub
import tensorflow_hub as hub

import librosa
from pycochleagram.pycochleagram import cgram
from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel

import serab_byols
from .settings import _MODELS_WEIGHTS

class FeatureExtractor():
    def __init__(self):
        self.models_weights = _MODELS_WEIGHTS

    def load_model(self, model_name, weights_file):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'BYOL-' in model_name:
            model = serab_byols.load_model(weights_file, model_name.split('_')[-1]).to(device)
        elif model_name == 'HuBERT':
            model = HubertModel.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif model_name == 'Wav2Vec2':
            model = Wav2Vec2Model.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif model_name == 'Data2Vec':
            model = Data2VecAudioModel.from_pretrained(weights_file, cache_dir='/om2/user/gelbanna/huggingface/').to(device)
        elif model_name == 'APC':
            model = getattr(s3hub, 'apc')().to(device)
        elif model_name == 'TERA':
            model = getattr(s3hub, 'tera')().to(device)
        elif model_name == 'TRILLsson':
            model = hub.KerasLayer('https://tfhub.dev/google/trillsson5/1')
        elif model_name == 'Log-Mel-Spectrogram':
            model = librosa.feature.melspctrogram
        elif model_name == 'Cochleagram':
            model = cgram.human_cochleagram
        else:
            model = hub.load(weights_file)
        return model

    def generate_speech_embeddings(self, audio_tensor_list, model, model_name, cfg):
        model_dir = f'../{cfg.dataset_name}/{model_name}'
        os.makedirs(model_dir, exist_ok=True)
        emb_file_name = f'{model_dir}/embeddings.npy'
        if os.path.isfile(emb_file_name):
            print(f'{model_name} embeddings are already saved for {cfg.dataset_name}')
            return np.load(emb_file_name)
        else:
            # Generate speech embeddings
            if 'BYOL-' in model_name:
                embeddings = serab_byols.get_scene_embeddings(audio_tensor_list, model)
                # Convert torch tensor to numpy
                embeddings = embeddings.cpu().detach().numpy()
            else:
                embeddings = []
                for audio in tqdm(audio_tensor_list, desc=f'Generating {model_name} Embeddings...'):
                    if model_name == 'TRILL':
                        embedding = np.mean(model(audio.cpu().detach().numpy(), sample_rate=cfg.resampling_rate)['layer19'], axis=0)

                    elif model_name == 'VGGish':
                        embedding = np.mean(model(audio.cpu().detach().numpy()), axis=0)

                    elif model_name == 'YAMNET':
                        _, embedding, _ = model(audio.cpu().detach().numpy())
                        embedding = np.mean(embedding, axis=0)
                        
                    elif model_name == 'TRILLsson':
                        embedding = model(audio.unsqueeze(0).cpu().detach().numpy())['embedding']
                        embedding = np.squeeze(embedding)

                    elif model_name == 'TERA' or model_name == 'APC':
                        model.eval()
                        with torch.no_grad():
                            embedding = model(audio.to('cuda').unsqueeze(0))["hidden_states"]

                    elif model_name == 'Log-Mel-Spectrogram' or model_name == 'Cochleagram':
                        embedding = model(audio.cpu().detach().numpy(), cfg.resampling_rate)

                    else:
                        model.eval()
                        with torch.no_grad():
                            embedding = model(audio.to('cuda').unsqueeze(0)).last_hidden_state
                            embedding = embedding.mean(1) + embedding.amax(1)
                            embedding = np.squeeze(embedding.cpu().detach().numpy())
                            
                    embeddings.append(embedding)
                embeddings = np.array(embeddings)
            np.save(emb_file_name, embeddings)
            return embeddings

    def extract(self, audio_tensor_list, cfg):
        embeddings = {}
        for model_name, weights_file in self.models_weights.items():
            print(f'Load {model_name} Model')
            model_dir = f'../{cfg.dataset_name}/{model_name}'
            emb_file_name = f'{model_dir}/embeddings.npy'
            if os.path.isfile(emb_file_name):
                print(f'{model_name} embeddings are already saved for {cfg.dataset_name}')
                embeddings[model_name] = np.load(emb_file_name)
            else:
                model = self.load_model(model_name, weights_file)
                embeddings[model_name] = self.generate_speech_embeddings(audio_tensor_list, model, model_name, cfg)
            print(embeddings[model_name].shape)
        return embeddings