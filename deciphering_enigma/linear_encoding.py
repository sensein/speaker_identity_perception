import itertools
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

from .settings import _hyperparams_grid_models, _split_train

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
        else:
            raise AttributeError(f'This reducer {name} is not included...')
    
    def create_df(self, features, labels, ids):
        data_df = pd.DataFrame(features).add_prefix('feature_')
        data_df['label'] = labels.tolist()
        data_df['ID'] = ids.tolist()
        return data_df
    
    def split_data(self, df):
        np.random.seed(self.seed)
        train_set = []; test_set = []
        for speaker in df['ID'].unique():
            speaker_df = df.loc[df.ID == speaker]
            msk = np.random.rand(len(speaker_df)) < self.split_train
            train_set.append(speaker_df[msk])
            test_set.append(speaker_df[~msk])
        train_set = pd.concat(train_set)
        test_set = pd.concat(test_set)
        return train_set.loc[:,train_set.columns.str.startswith("feature")], test_set.loc[:,test_set.columns.str.startswith("feature")], train_set['ID'], test_set['ID']
    
    def build_pipeline(self, clf, clf_params):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.seed)
        pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
        grid_pipeline = GridSearchCV(pipeline, param_grid=clf_params, n_jobs=-1, cv=cv, scoring='recall_macro')
        return grid_pipeline
    
    def get_dev_scores(self, grid_result):
        scores_list = [ v.tolist() for k,v in grid_result.cv_results_.items() if 'split' in k]
        return list(itertools.chain.from_iterable(scores_list))
        
    def run(self, model_name, features, labels, ids, save_path='./'):
        # create DF with features, labels and ids
        dev_results = {}
        data_df = self.create_df(features, labels, ids)
        for label in data_df['label'].unique():
            print(f'{model_name} with {label} samples:')
            data_label = data_df.loc[data_df['label'] == label]
            train_features, test_features, train_ids, test_ids = self.split_data(data_label)
            dev_results[label] = {}
            for i, (clf_name, clf_params) in enumerate(self.model_params_grid.items()):
                print(f'    Step {i+1}/{len(self.model_params_grid.keys())}: {clf_name}...')    
                clf_object = self.get_sklearn_model(clf_name)
                if clf_name == 'RF':
                    clf = clf_object(random_state = self.seed)
                else:
                    clf = clf_object(random_state = self.seed, max_iter=1e4)
                grid_pipeline = self.build_pipeline(clf, clf_params)
                grid_result = grid_pipeline.fit(train_features, train_ids)
                test_result = grid_result.score(test_features, test_ids)
                dev_results[label][clf_name] = self.get_dev_scores(grid_result)
                # print(f'        Best {clf_name} UAR for training: {grid_result.best_score_*100: .2f} using {grid_result.best_params_}')
                print(f'        Test Data UAR: {test_result*100: .2f}')
        return dev_results