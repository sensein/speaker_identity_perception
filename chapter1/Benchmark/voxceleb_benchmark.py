'''code for loading data and linear evaluation is mainly adapted from https://github.com/nttcslab/byol-a'''

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
                  
for model_name, weights_file in _MODELS_WEIGHTS.items():
    print(f'Testing {model_name}')
    results = defaultdict(list)
    model_dir = f'../{exp_config.dataset_name}/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    
    for i in range(data.n_folds):
        fold_data = data.subset([i])
        print(f'Number of Speakers = {np.unique(fold_data.labels).shape}')
        emb_file_name = f'{model_dir}/{i}_embeddings.npy'
        if os.path.isfile(emb_file_name):
            print(f'{model_name} embeddings are already saved for {exp_config.dataset_name}')
            folds[i]['X'] = np.load(emb_file_name)
            if folds[i]['X'].ndim > 2:
                folds[i]['X'] = np.squeeze(folds[i]['X'], axis=1)
            print(folds[i]['X'].shape)
        else:
            print(f'getting embeddings for fold #{i} ({len(fold_data)} samples)...')
            folds[i]['X'] = deciphering_enigma.get_embeddings(fold_data.files, model_name, weights_file, exp_config)
            print(folds[i]['X'].shape)
            np.save(f'../{exp_config.dataset_name}/{model_name}/{i}_embeddings.npy', folds[i]['X'])
        folds[i]['y'] = fold_data.labels
        
    splits = deciphering_enigma.get_splits(folds[0]['y'], folds[1]['y'])
    train_X = np.concatenate((folds[0]['X'], folds[1]['X']), axis=0)
    train_y = np.concatenate((folds[0]['y'], folds[1]['y']), axis=0)
    pipeline = deciphering_enigma.build_pipeline(parameter_space, splits, (), 200, True, True, 'recall_macro')
    for _ in range(3):
        score = deciphering_enigma.linear_eval(train_X, train_y, folds[2]['X'], folds[2]['y'], pipeline)
        results['Model'].append(model_name)
        results['Score'].append(score)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f'{model_dir}/results.csv')