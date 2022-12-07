import os
import itertools
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, PredefinedSplit

from deciphering_enigma.settings import _hyperparams_grid_models, _split_train

class ML_Encoder():
    def __init__(self, random_seed=42):
        self.seed = random_seed
        self.split_train = _split_train
        self.model_params_grid = _hyperparams_grid_models

    def get_sklearn_model(self, name):
        if name == 'LR':
            return LogisticRegression
        elif name == 'RF':
            return RandomForestClassifier
        elif name == 'SVM':
            return SVC
        elif name == 'MLP':
            return MLPClassifier
        else:
            raise AttributeError(f'This reducer {name} is not included...')
    
    def create_df(self, features, labels, ids):
        data_df = pd.DataFrame(features).add_prefix('feature_')
        data_df['label'] = labels.tolist()
        data_df['ID'] = ids.tolist()
        return data_df
    
    def split_data(self, df, train_label):
        np.random.seed(self.seed)
        train_set = []; test_set = []
        for speaker in df['ID'].unique():
            speaker_orig_df = df.loc[(df.ID == speaker) & (df.label == train_label)]
            msk = np.random.rand(len(speaker_orig_df)) < self.split_train
            train_set.append(speaker_orig_df[msk])
            for label in df['label'].unique():
                speaker_label_df = df.loc[(df.ID == speaker) & (df.label == label)]
                test_set.append(speaker_label_df[~msk])
        train_set = pd.concat(train_set)
        test_set = pd.concat(test_set)
        return train_set, test_set
    
    def build_pipeline(self, clf, clf_params, splits):
        pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
        grid_pipeline = GridSearchCV(pipeline, param_grid=clf_params, n_jobs=-1, cv=splits, scoring='recall_macro')
        return grid_pipeline
    
    def get_dev_scores(self, grid_result):
        scores_list = [ v.tolist() for k,v in grid_result.cv_results_.items() if 'split' in k]
        return list(itertools.chain.from_iterable(scores_list))
        
    def run(self, model_name, features, labels, ids, dataset_name, train_label):
        # create DF with features, labels and ids
        dev_results = {'Model':[], 'Label':[], 'Clf':[], 'Score':[]}
        data_df = self.create_df(features, labels, ids)
        if os.path.isfile(f'../{dataset_name}/{model_name}/linear_encoding_scores.csv'):
            print(f'Linear Encoding Scores are already saved for {model_name} model!')
        else:
            print(f'{model_name}:')
            train_set, test_set = self.split_data(data_df, train_label)
            train_features = train_set.loc[:,train_set.columns.str.startswith("feature")]
            train_ids = train_set['ID']
            for label in data_df['label'].unique():
                print(f'    Testing {label} samples...')   
                test_label_set = test_set.loc[test_set['label'] == label]
                test_label_features = test_label_set.loc[:,test_label_set.columns.str.startswith("feature")]
                test_label_ids = test_label_set['ID']

                train_features_all = pd.concat([train_features, test_label_features])
                train_ids_all = pd.concat([train_ids, test_label_ids])

                split_indices = np.repeat([-1, 0], [len(train_features), len(test_label_features)])
                split = PredefinedSplit(split_indices)
                for i, (clf_name, clf_params) in enumerate(self.model_params_grid.items()):
                    print(f'     Step {i+1}/{len(self.model_params_grid.keys())}: {clf_name}...')    
                    clf_object = self.get_sklearn_model(clf_name)
                    if clf_name == 'RF' or clf_name == 'MLP':
                        clf = clf_object(random_state = self.seed)
                    else:
                        clf = clf_object(random_state = self.seed, max_iter=1e4)

                    grid_pipeline = self.build_pipeline(clf, clf_params, split)
                    grid_result = grid_pipeline.fit(train_features_all, train_ids_all)

                    dev_results['Model'].append(model_name)
                    dev_results['Label'].append(label)
                    dev_results['Clf'].append(clf_name)
                    dev_results['Score'].append(self.get_dev_scores(grid_result))
                    print(f'        Best {clf_name} UAR for training: {grid_result.best_score_*100: .2f} using {grid_result.best_params_}')
                df = pd.DataFrame(dev_results)
                df = df.explode('Score')
                df.to_csv(f'../{dataset_name}/{model_name}/linear_encoding_scores.csv', index=False)