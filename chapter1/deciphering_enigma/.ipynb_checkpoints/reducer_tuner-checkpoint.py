import os
import numpy as np
import pandas as pd

import scipy
from scipy.spatial.distance import pdist

from umap import UMAP
from pacmap import PaCMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from .reducer_tuner_settings import _hyperparams_grid, _optimize_function, _knn, _subsetsize

class ReducerTuner():
    """Tuner for dimensionality reduction methods.

    Implements grid-search across hyperparameters for each dimensionality reduction method preset in the settings script.
    NOTE: any method added in the settings script should follow sklearn implementation.
    Tunes reduced dimensions by optimizing local and global structure metrics.
    Saves tuned results for each method as a pandas dataframe.
    """

    def __init__(self):
        self.reducer_params_grid = _hyperparams_grid
        self.optimize_func = _optimize_function
        self.knn = _knn; self.subsetsize = _subsetsize

    def embedding_quality(self, X, Z, knn=10, subsetsize=1000):
        nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
        ind1 = nbrs1.kneighbors(return_distance=False)

        nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
        ind2 = nbrs2.kneighbors(return_distance=False)

        intersections = 0.0
        for i in range(X.shape[0]):
            intersections += len(set(ind1[i]) & set(ind2[i]))
        mnn = intersections / X.shape[0] / knn

        subset = np.random.choice(X.shape[0], size=subsetsize, replace=True)
        d1 = pdist(X[subset,:])
        d2 = pdist(Z[subset,:])
        rho = scipy.stats.spearmanr(d1[:,None],d2[:,None]).correlation
        return (mnn, rho)
    
    def get_reducer(self, name):
        if name == 'PCA':
            return PCA
        elif name == 'tSNE':
            return TSNE
        elif name == 'UMAP':
            return UMAP
        elif name == 'PaCMAP':
            return PaCMAP
        else:
            raise AttributeError(f'This reducer {name} is not included...')

    def fit_eval(self, embeddings, reducer):
        reduced_embeddings = reducer.fit_transform(StandardScaler().fit_transform(embeddings))
        local_val, global_val = self.embedding_quality(embeddings, reduced_embeddings, knn=self.knn, subsetsize=self.subsetsize)
        return reduced_embeddings, local_val, global_val
    
    def save_results_pandas(self, reducers_embeddings_dict, metadata=None, save_path='./'):
        combined_column_obj = pd.MultiIndex.from_product([reducers_embeddings_dict.keys(),['Local', 'Global'], ['Dim1', 'Dim2']], names=["Method", "Optimized Metric", "Dim"])
        df = pd.DataFrame(data=[], columns=combined_column_obj)
        for j, name in enumerate(reducers_embeddings_dict.keys()):
            global_embeddings = reducers_embeddings_dict[name]['Global']
            local_embeddings = reducers_embeddings_dict[name]['Local']
            df.loc[:, (name, 'Local', 'Dim1')] = local_embeddings[:,0]
            df.loc[:, (name, 'Local', 'Dim2')] = local_embeddings[:,1]
            df.loc[:, (name, 'Global', 'Dim1')] = global_embeddings[:,0]
            df.loc[:, (name, 'Global', 'Dim2')] = global_embeddings[:,1]
        temp_df = metadata_df.copy()
        temp_df.columns = pd.MultiIndex.from_tuples(map(lambda x: (x, '', ''), temp_df.columns))
        df = pd.concat([df, temp_df], axis=1)
        df.to_csv(save_path)

    def tune_reducer(self, embeddings, metadata=None, dataset_name=None, model_name=None, save_results = True, save_path='./'):
        reducers_embeddings_dict = {}
        metrics_dict = {}
        df_path = f'../{dataset_name}/{model_name}/dim_reduction.csv'
        if os.path.isfile(df_path):
            print(f'Tuned Reduced Embeddings already saved for {model_name} model!')
        else:
            for i, (reducer_name, reducer_params) in enumerate(self.reducer_params_grid.items()):
                print(f'Reducer {i+1}/{len(self.reducer_params_grid.keys())}: {reducer_name}...')
                reducers_embeddings_dict[reducer_name] = {}
                reducer_object = self.get_reducer(reducer_name)
                params_iterator = list(ParameterGrid(reducer_params))
                all_embeddings = []; local_metrics = []; global_metrics = []
                for params in params_iterator:
                    reducer = reducer_object(n_components=2, random_state=42, **params)
                    reduced_embeddings, local_metric, global_metric = self.fit_eval(embeddings, reducer)
                    all_embeddings.append(reduced_embeddings); local_metrics.append(local_metric); global_metrics.append(global_metric)
                max_local_idx = np.argmax(local_metrics)
                max_global_idx = np.argmax(global_metrics)
                metrics_dict[reducer_name] = {'Local': np.max(local_metrics), 'Global': np.max(global_metrics)}
                reducers_embeddings_dict[reducer_name]['Local'] = all_embeddings[max_local_idx]
                reducers_embeddings_dict[reducer_name]['Global'] = all_embeddings[max_global_idx]
            if save_results:
                self.save_results_pandas(reducers_embeddings_dict, metadata, save_path)
