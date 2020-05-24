"""Base class for Gaussian Mixture Model."""

# Author: Ziqi Yuan <1564123490@qq.com>

import numpy as np


def construct_dataset(data_path="../data/cluster.dat"):
    dataset = []
    with open(data_path) as f:
        for line in f.readlines():
            point_tmp = line.strip().split(' ')
            dataset.append([float(point_tmp[0]), float(point_tmp[1])])
    return np.array(dataset)

class GaussianMixture:
    """Gaussian Mixture.

        Representation of a Gaussian mixture model probability distribution.
        This class allows to estimate the parameters of a Gaussian mixture
        distribution.

        Parameters
        ----------
        n_components : int, defaults to 1.
            The number of mixture components.

        tol : float, defaults to 1e-3.
            The convergence threshold. EM iterations will stop when the
            lower bound average gain is below this threshold.

        max_iter : int, defaults to 100.
            The number of EM iterations to perform.

        random_state : int, RandomState instance or None, optional (default=None)
            Controls the random seed given to the method chosen to initialize the
            parameters (see `init_params`).
            In addition, it controls the generation of random samples from the
            fitted distribution (see the method `sample`).
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        verbose : boolean, default to false.
            Enable verbose output. If true then it prints the likelihood each iteration step.

        Attributes
        ----------
        weights_ : array-like, shape (n_components,)
            The weights of each mixture components.

        means_ : array-like, shape (n_components, n_features)
            The mean of each mixture component.

        covariances_ : array-like
            The covariance of each mixture component.
            shape (n_components, n_features, n_features) if 'full'

        """
    def __init__(self, train_set, n_components=1, threshold=0.1, max_iter=100, random_state=3333, verbose=False):

        self.train_set = train_set
        self.n_components = n_components
        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose

        self.default_var = 0.1
        np.random.seed(random_state)
        self.n_samples, self.n_features = train_set.shape

        self._initialize(self.n_samples, self.n_features, self.n_components)

    def _initialize(self, n_samples, n_features, n_components):
        """Initialization of the module parameters.

        Parameters
        ----------
        n_samples : int, the number of samples in trainset

        n_features : int, the dim of each sample.

        n_components : int, the number of mixture components.

        """

        self.weights_ = np.random.uniform(1 / n_components, 1 / n_components, n_components)
        self.means_ = np.empty((n_components, n_features), dtype=float)
        selected_sample = np.random.choice(n_samples, n_components)
        for i, sample in enumerate(selected_sample):
            self.means_[i] = self.train_set[sample]
        self.covariances_ = np.empty((n_components, n_features, n_features), dtype=float)

        for i in range(n_components):
            self.covariances_[i] = np.eye(n_features) * self.default_var

        self.posterior_matrix = np.empty((n_samples, n_components), dtype=float)

    def _estimate_post_prob(self, train_set, weight, means, covariances):
        probability_density_matrix = self._estimate_pdf(train_set, means, covariances)

        numerator = probability_density_matrix * weight.reshape(1, -1)
        Denominator = np.sum(numerator, axis=1).reshape(-1, 1)
        return numerator / Denominator


    def _estimate_pdf(self, train_set, means, covariances):
        """
        Parameters
        ----------
        train_set : array-like, shape (n_samples, n_features)

        means : array-like, shape (n_components, n_features)

        covariances: array-like, shape (n_components, n_features, n_features)

        Returns
        -------
        probability_density_matrix : array, shape (n_samples, n_component)
        """
        probability_density_matrix = np.zeros((self.n_samples, self.n_components), dtype=float)
        coef_pi = np.power((2*np.pi), self.n_features/2)
        for index in range(self.n_components):
            coef_det = np.power(np.linalg.det(covariances[index]), 0.5)
            covInv = np.linalg.inv(covariances[index])
            shift_input = train_set-means[index, :]
            probability_density_matrix[:, index] = \
                (1/(coef_pi*coef_det)) * np.exp(np.sum(-0.5*shift_input.dot(covInv)*shift_input, axis=1))
        return probability_density_matrix


if __name__ == "__main__":

    PointSet = construct_dataset()
    print(f"PointSet has {len(PointSet)} data")
    cluster = GaussianMixture(PointSet, n_components=2)
    print(cluster._estimate_post_prob(PointSet, cluster.weights_ ,cluster.means_, cluster.covariances_)[84])