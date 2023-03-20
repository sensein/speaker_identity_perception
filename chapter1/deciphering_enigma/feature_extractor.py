import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import s3prl.hub as s3hub
import tensorflow_hub as hub
import librosa
import torchaudio
import torchaudio.transforms as T
import pycochleagram.cochleagram as cgram
from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel

import serab_byols
from .settings import _MODELS_WEIGHTS


class FeatureExtractor():
    def __init__(self):
        self.models_weights = _MODELS_WEIGHTS

    def load_model(self, model_name, weights_file, cfg=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'BYOL-' in model_name:
            model = serab_byols.load_model(weights_file, model_name.split('_')[-1]).to(device)
        elif model_name == 'Pyannote':
            model = Inference(weights_file, window="whole")
        elif model_name == 'SpeechBrain':
            model = EncoderClassifier.from_hparams(source=weights_file).to(device)
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
                        sample_rate=cfg.resampling_rate,
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
            model = hub.load(weights_file)
        return model

    def generate_speech_embeddings(self, audio_tensor_list, model, model_name, cfg, save=True, tqdm_disable=False):
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
                for audio in tqdm(audio_tensor_list, desc=f'Generating {model_name} Embeddings...', disable=tqdm_disable):
                    if model_name == 'TRILL':
                        embedding = np.mean(model(audio.cpu().detach().numpy(), sample_rate=cfg.resampling_rate)['layer19'], axis=0)

                    elif model_name == 'SpeechBrain':
                        embeddings = classifier.encode_batch(audio.to('cuda'))
                        print(embeddings.shape)

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
                            embedding = model(audio.to('cuda').unsqueeze(0))["last_hidden_state"]
                            embedding = embedding.mean(1) + embedding.amax(1)
                            embedding = np.squeeze(embedding.cpu().detach().numpy())

                    elif model_name == 'Log-Mel-Spectrogram':
                        mel_spec = model(audio.to('cuda'))
                        embedding = (mel_spec + torch.finfo().eps).log().squeeze(0)
                        embedding = embedding.mean(1) + embedding.amax(1)
                        embedding = np.squeeze(embedding.cpu().detach().numpy())
                    
                    elif model_name == 'Cochleagram':
                        coch = model(audio.cpu().detach().numpy(), cfg.resampling_rate, strict=False, n=40)
                        embedding = (torch.from_numpy(coch) + torch.finfo().eps).log().squeeze(0)
                        embedding = embedding.mean(1) + embedding.amax(1)
                        embedding = np.squeeze(embedding.cpu().detach().numpy())
                        
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
                                    _ = model(audio.to('cuda').unsqueeze(0))
                                    embedding = activation['latent_features'].permute(0,2,1)
                                else:
                                    embedding = model(audio.to('cuda').unsqueeze(0)).extract_features
                            elif 'best' in model_name:
                                if 'HuBERT' in model_name:
                                    num = 6
                                elif 'Wav2Vec2' in model_name:
                                    num = 0
                                elif 'Data2Vec' in model_name:
                                    num = 3
                                model.encoder.layers[num].register_forward_hook(get_activation(f'{model_name}_encoder'))
                                _ = model(audio.to('cuda').unsqueeze(0))
                                embedding = activation[f'{model_name}_encoder']
                                
                            else:
                                embedding = model(audio.to('cuda').unsqueeze(0)).last_hidden_state
                            embedding = embedding.mean(1) + embedding.amax(1)
                            embedding = np.squeeze(embedding.cpu().detach().numpy())

                    embeddings.append(embedding)
                embeddings = np.array(embeddings)
            if save:
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
                model = self.load_model(model_name, weights_file, cfg)
                embeddings[model_name] = self.generate_speech_embeddings(audio_tensor_list, model, model_name, cfg)
            print(embeddings[model_name].shape)
        return embeddings