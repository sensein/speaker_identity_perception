import os
import numpy as np
import pandas as pd
import seaborn as sns
import deciphering_enigma
import matplotlib.pyplot as plt
from collections import defaultdict
from deciphering_enigma.settings import _MODELS_WEIGHTS

path_to_config = './config.yaml'
exp_config = deciphering_enigma.load_yaml_config(path_to_config)

data = deciphering_enigma.TaskDataSource(exp_config.dataset_path, subset='all')

folds = [defaultdict(list) for _ in range(data.n_folds)]
parameter_space = {'clf__learning_rate_init': np.logspace(-2, -4, num=3)
                  }

model_components = ['feature_extractor.conv_layers', 'encoder.layers']
activation = {}

def get_activation(name):
    def hook(model, input, output):
        if 'encoder' in name:
            activation[name] = output[0].detach()
        else:
            activation[name] = output.detach()
    return hook

def get_model(model_name, weight_file, exp_config):
    feature_extractor = deciphering_enigma.FeatureExtractor()
    model = feature_extractor.load_model(model_name, weight_file, exp_config)
    return model

def get_embeddings(audio_tensor_list, model, component, layer, layer_name):
    embeddings = []
    for audio in tqdm(audio_tensor_list, desc=f'Generating {model_name} Embeddings...'):
        model.eval()
        with torch.no_grad():
            getattr(getattr(model, component.split('.')[0]), component.split('.')[1])[layer].register_forward_hook(get_activation(layer_name))
            _ = model(audio.to('cuda').unsqueeze(0))
            if 'encoder' in layer_name:
                embedding = activation[layer_name]
            else:
                embedding = activation[layer_name].permute(0,2,1)
            embedding = embedding.mean(1) + embedding.amax(1)
            embedding = np.squeeze(embedding.cpu().detach().numpy())
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    return embeddings

for model_name, weights_file in _MODELS_WEIGHTS.items():
    results = defaultdict(list)
    model_dir = f'../{exp_config.dataset_name}/{model_name}'
    print(f'Testing {model_name}')
    model = get_model(model_name, weights_file, exp_config)
    for component in model_components:
        num_layers = len(dict(model.named_modules())[component])
        for layer in range(num_layers):
            layer_name = f'{component}_{layer}'
            print(f'Testing {component}{layer}')
            for i in range(3):
                label_file_name = f'{model_dir}/{i}_labels.npy'
                emb_file_name = f'{model_dir}/{i}_{layer_name}_embeddings.npy'
                folds[i]['X'] = np.load(emb_file_name)
                if folds[i]['X'].ndim > 2:
                    folds[i]['X'] = np.squeeze(folds[i]['X'], axis=1)
                print(folds[i]['X'].shape)
                folds[i]['y'] = np.load(label_file_name)
            splits = deciphering_enigma.get_splits(folds[0]['y'], folds[1]['y'])
            train_X = np.concatenate((folds[0]['X'], folds[1]['X']), axis=0)
            train_y = np.concatenate((folds[0]['y'], folds[1]['y']), axis=0)
            pipeline = deciphering_enigma.build_pipeline(parameter_space, splits, (), 200, True, True, 'recall_macro')
            score = deciphering_enigma.linear_eval(train_X, train_y, folds[2]['X'], folds[2]['y'], pipeline)
            results['Layer'].append(component+str(layer))
            results['Score'].append(score)

            #Save results
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(f'{model_dir}/{model_name}_{layer_name}_results.csv')