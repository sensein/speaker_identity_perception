#TODO: add spectral and diffusion map embeddings
import numpy as np

_hyperparams_grid = {
                        'PCA' : {
                            'svd_solver': ['auto'] 
                            },
        
                        'tSNE' : {
                            'perplexity': np.arange(5,155,10),
                            'init': ['random', 'pca']
                            },
                        
                        'UMAP' : {
                            'n_neighbors': np.arange(10,100,20),
                            'min_dist': np.logspace(0, -5, num=6)
                            },
                        
                        'PaCMAP' : {
                            'n_neighbors': np.arange(10,100,20),
                            'MN_ratio': [0.1, 0.5, 1],
                            'FP_ratio': [1, 2, 5]
                            }
                    }

_optimize_function = ['Local', 'Global']
_knn = 195; _subsetsize = 1000