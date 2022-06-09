''' This code is adapted from 
https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=45qb6zdSsHj6'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
            raise ValueError('Input must be a symmetric matrix.')
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

    def plot_heatmap(self, matrix, features_name, save_path='./', save_fig=True):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.heatmap(matrix, ax=ax, yticklabels=features_name, annot=True, xticklabels=features_name)
        ax.set_xlabel('Models', fontsize=15)
        ax.set_ylabel('Models', fontsize=15)
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{save_path}CKA_models_{self.kernel}_{self.bias_str}.png')
