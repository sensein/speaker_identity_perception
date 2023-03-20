''' This code is adapted from 
https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=45qb6zdSsHj6'''

import os
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class CKA():

    def __init__(self, unbiased=True, kernel='linear', rbf_threshold=1.0):
        self.unbiased = unbiased
        self.bias_str = 'unbiased' if unbiased else 'biased'
        self.kernel = kernel
        self.threshold = rbf_threshold

    def gram_linear(self, x):
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
            x: A num_examples x num_features matrix of features.

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """
        return x.dot(x.T)

    def gram_rbf(self, x):
        """Compute Gram (kernel) matrix for an RBF kernel.

        Args:
        x: A num_examples x num_features matrix of features.
        threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

        Returns:
        A num_examples x num_examples Gram matrix of examples.
        """
        dot_products = x.dot(x.T)
        sq_norms = np.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = np.median(sq_distances)
        return np.exp(-sq_distances / (2 * self.threshold ** 2 * sq_median_distance))

    def center_gram(self, gram):
        """Center a symmetric Gram matrix.

        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
        gram: A num_examples x num_examples symmetric matrix.

        Returns:
        A symmetric matrix with centered columns and rows.
        """
        if not np.allclose(gram, gram.T):
            # raise ValueError('Input must be a symmetric matrix.')
            gram = gram.copy()

        if self.unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]
        return gram

    def compute(self, X, Y):
        """Compute CKA.

        Args:
        gram_x: A num_examples x num_samples.
        gram_y: A num_examples x num_samples.

        Returns:
        The value of CKA between X and Y.
        """

        if self.kernel == 'linear':
            gram_x = self.gram_linear(X)
            gram_y = self.gram_linear(Y)
        else:
            gram_x = self.gram_rbf(X)
            gram_y = self.gram_rbf(Y)
        gram_x = self.center_gram(gram_x)
        gram_y = self.center_gram(gram_y)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        """Compute CKA with a linear kernel, in feature space.

        This is typically faster than computing the Gram matrix when there are fewer
        features than examples.

        Args:
            features_x: A num_examples x num_features matrix of features.
            features_y: A num_examples x num_features matrix of features.
            debiased: Use unbiased estimator of dot product similarity. CKA may still be
            biased. Note that this estimator may be negative.

        Returns:
            The value of CKA between X and Y.
        """
        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
            sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = _debiased_dot_product_similarity_helper(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)
         
    def plot_heatmap(self, matrix, features_name, save_path='./', save_fig=True):
        fig, ax = plt.subplots(1, 1, figsize=(35, 30))
        ax = sns.heatmap(matrix, ax=ax, yticklabels=features_name, annot=True, annot_kws={"fontsize":35}, xticklabels=features_name)
        ax.figure.axes[-1].set_ylabel('CKA Similarity', size=40)
        ax.figure.axes[-1].set_yticklabels(ax.figure.axes[-1].get_yticklabels(), size = 40)
        ax.set_xticklabels(ax.get_xticklabels(), size = 40)
        ax.set_yticklabels(ax.get_yticklabels(), size = 40)
        ax.set_xlabel('Models', fontsize=40)
        ax.set_ylabel('Models', fontsize=40)
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{save_path}CKA_models_{self.kernel}_{self.bias_str}.png')

    def run_cka(self, embeddings_dict, compute_from='examples', save_path='./', save_array=True):
        cka_file = f'{save_path}cka.npy'
        if os.path.isfile(cka_file):
            print(f'CKA is already saved')
            cka_ = np.load(cka_file)
        else:
            num_models = len(embeddings_dict.keys())
            cka_ = np.zeros((num_models, num_models))
            scaler = StandardScaler()
            for i, (_, model_1) in enumerate(tqdm(embeddings_dict.items())):
                for j, (_, model_2) in enumerate(embeddings_dict.items()):
                    model_1 = scaler.fit_transform(model_1)
                    model_2 = scaler.fit_transform(model_2)
                    if compute_from == 'examples':
                        cka_[i,j] = self.compute(model_1, model_2)
                    else:
                        cka_[i,j] = self.feature_space_linear_cka(model_1, model_2)
            if save_array:
                np.save(cka_file, cka_)
        return cka_