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
    def __init__(self, train_set, n_components=1, threshold=0.1, max_iter=10, random_state=3333, verbose=False):

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

    def train(self):
        prelikelihood = -np.inf

        for epochs in range(self.max_iter):
            post_prob = self._estimate_post_prob(self.train_set, self.weights_, self.means_, self.covariances_)
            nk = self._cal_nk(post_prob)
            self.weights_ = nk / self.n_samples
            self.means_ = self._update_means(self.means_, nk, post_prob, self.train_set)
            self.covariances_ = self._update_covariances(self.means_, self.covariances_, nk, post_prob, self.train_set)

            if epochs % 3 == 2:

                print(f"weights for each cluster: \n{self.weights_}")
                print(f"means for each cluster: \n{self.means_}")
                print(f"covariances for each cluster:\n {self.covariances_}")
                print("***************************************************")




    def _estimate_post_prob(self, train_set, weight, means, covariances):
        """
        Parameters
        ----------
        train_set : array-like, shape (n_samples, n_features)

        weight : array-like, shape (n_components, )

        means : array-like, shape (n_components, n_features)

        covariances: array-like, shape (n_components, n_features, n_features)

        Returns
        -------
         post_prob(gamma) : array, shape (n_samples, n_component)
        """
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

    def _cal_nk(self, post_prob_matrix):
        nk = np.sum(post_prob_matrix, axis=0)
        return nk.reshape(1, self.n_components)

    def _update_means(self, means, nk, post_prob, train_set):
        means_new = np.zeros_like(means)
        for index in np.arange(0, self.n_components):
            sumd = np.sum((post_prob[:, index].reshape(self.n_samples, 1)) * train_set, axis=0)
            means_new[index, :] = sumd.reshape(1, self.n_features) / nk[:, index]
        return means_new

    def _update_covariances(self, means, covariances, nk, post_prob, train_set):
        covariances_new = np.zeros_like(covariances)
        for index in np.arange(0, self.n_components):
            shift = train_set - means[index, :]
            shift_tmp = post_prob[:, index].reshape(self.n_samples, 1) * shift
            covariances_new[index] = shift_tmp.T.dot(shift) / nk[:, index]
        return covariances_new




if __name__ == "__main__":

    PointSet = construct_dataset()

    cluster = GaussianMixture(PointSet, n_components=2)
    cluster.train()
