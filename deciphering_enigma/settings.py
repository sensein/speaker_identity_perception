import numpy as np

_MODELS_WEIGHTS = {'BYOL-A_default': '../byola_checkpoints/byola/AudioNTT2020-BYOLA-64x96d2048.pth',
                   'BYOL-S_default': '../serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth',
                   'Hybrid_BYOL-S_default': '../byola_checkpoints/hybrid_byols/default2048_BYOLAs64x96-osandbyolaloss6373-e100-bs128-lr0003-rs42.pth',
                   'BYOL-S_cvt': '../byola_checkpoints/byols_encoders/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-2107301623-e100-bs256-lr0003-rs42.pth',
                   'Hybrid_BYOL-S_cvt': '../serab-byols/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth',
                   # 'TERA': '',
                   'TRILLsson': 'https://tfhub.dev/google/trillsson2/1',
                   # 'YAMNET': 'https://tfhub.dev/google/yamnet/1',
                   # 'TRILL': 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3',
                   'Wav2Vec2': 'facebook/wav2vec2-large-960h-lv60-self',
                   'HuBERT': 'facebook/hubert-xlarge-ll60k',
                   'Data2Vec': 'facebook/data2vec-audio-large-960h'}


#TODO: add spectral and diffusion map embeddings

_hyperparams_grid_reducers = {
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

# ML encoding variables
_split_train = 0.7 # 70% percent
_hyperparams_grid_models = {
                        'LR': {
                            'estimator__C': np.logspace(5, -4, num=11),
                            'estimator__class_weight': ['balanced', None],
                        },
                        'RF': {
                            'estimator__max_depth': range(5, 30, 5),
                            'estimator__class_weight': ['balanced', None],
                        },
                        'SVM': {
                            'estimator__C': np.logspace(5, -4, num=11),
                            'estimator__class_weight': ['balanced', None]
                        }
                    }
