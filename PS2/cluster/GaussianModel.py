"""Base class for Gaussian Mixture Model."""

# Author: Ziqi Yuan <1564123490@qq.com>

import numpy as np
import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt


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
        weights_ : array-like, shape (1, n_components)
            The weights of each mixture components.

        means_ : array-like, shape (n_components, n_features)
            The mean of each mixture component.

        covariances_ : array-like
            The covariance of each mixture component.
            shape (n_components, n_features, n_features) if 'full'

        """
    def __init__(self,
                 train_set,
                 has_test=False,
                 test_set=None,
                 n_components=1,
                 threshold=1e-6,
                 max_iter=200,
                 random_state=111,
                 type="full",
                 verbose=False):

        self.train_set = train_set
        self.test_set = test_set
        self.has_test = has_test

        self.n_components = n_components
        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose

        self.default_var = 0.1
        self.type = type
        self.likelihood = None
        self.record_train = []
        self.record_test = []
        self.color_iter = ['r', 'g', 'b', 'c', 'm']
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
            probability_density_matrix = self._estimate_pdf(self.train_set, self.means_, self.covariances_)
            # print(self._cal_likelihood(self.weights_, probability_density_matrix))
            post_prob = self._estimate_post_prob(self.weights_, probability_density_matrix)
            nk = self._cal_nk(post_prob)
            self.weights_ = nk / self.n_samples
            self.means_ = self._update_means(self.means_, nk, post_prob, self.train_set)

            # Note : the only difference between full & diag is due to the non-linear limit of the cov matrix.
            # so only the update function is different, ohters should be the same.
            if self.type == "full":
                self.covariances_ = self._update_covariances(self.means_,
                                                             self.covariances_,
                                                             nk, post_prob, self.train_set)
            elif self.type == "diag":
                self.covariances_ = self._update_covariances_diag(self.means_,
                                                             self.covariances_,
                                                             nk, post_prob, self.train_set)

            if epochs % 20 == 19:

                print(f"weights for each cluster: \n{self.weights_}")
                print(f"means for each cluster: \n{self.means_}")
                print(f"covariances for each cluster:\n {self.covariances_}")
                print("***************************************************")

                # self.visualize(self.train_set, result, self.color_iter)
                # print(result)
                # plt.scatter(self.train_set[:, 0], self.train_set[:, 1], .8, c=result)
                # plt.show()

            current_likelihood = GaussianMixture._cal_likelihood(self.weights_, probability_density_matrix)
            self.record_train.append(current_likelihood)
            if self.has_test:
                current_likelihood_test = GaussianMixture._cal_likelihood(self.weights_,
                                                                      self._estimate_pdf(self.test_set, self.means_, self.covariances_))
                self.record_test.append(current_likelihood_test)
            '''when likelihood value doesnot change so much -> early stop.'''
            if np.abs(current_likelihood - prelikelihood) < self.threshold:
                print(f"early stop at epochs: {epochs}")
                self.likelihood = current_likelihood
                break

            prelikelihood = current_likelihood

    def predict(self, dataset):
        result = []
        probability_density_matrix = self._estimate_pdf(dataset, self.means_, self.covariances_)
        for row in probability_density_matrix:
            result.append(np.argmax(row))
        return np.array(result).reshape(-1)

    def visualize(self, X, y, color_iter):
        """ draw the graph.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        y : array-like, shape (n_samples)

        color_iter: array-like, shape (5)
        """
        plt.title(f"log likelihood: {self.likelihood}")
        splot = plt.subplot(1, 1, 1)
        for i, (mean, covar, color) in enumerate(zip(
                self.means_, self.covariances_, color_iter)):
            v, w = np.linalg.eigh(covar)
            u = w[0] / np.linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(y == i):
                continue
            plt.scatter(X[y == i, 0], X[y == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        plt.show()

    def _estimate_post_prob(self, weight, probability_density_matrix):
        """
        Parameters
        ----------
        weight : array-like, shape (n_components, )

        probability_density_matrix : array-like, shape (n_samples, n_component)

        Returns
        -------
         post_prob(gamma) : array, shape (n_samples, n_component)
        """
        # probability_density_matrix = self._estimate_pdf(train_set, means, covariances)

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
        probability_density_matrix = np.zeros((train_set.shape[0], self.n_components), dtype=float)
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
        """Estimate the means vectors.

            Parameters
            ----------
            means : array-like, shape (n_components, n_features)

            nk : array-like, shape (1, n_components)

            post_prob : array-like, shape (n_samples, n_components)

            train_set : array-like, shape (n_samples, n_features)

            Returns
            -------
            means : array, shape (n_components, n_features)
                The means vector of the current components.
         """
        means_new = np.zeros_like(means)
        for index in np.arange(0, self.n_components):
            sumd = np.sum((post_prob[:, index].reshape(self.n_samples, 1)) * train_set, axis=0)
            means_new[index, :] = sumd.reshape(1, self.n_features) / nk[:, index]
        return means_new

    def _update_covariances(self, means, covariances, nk, post_prob, train_set):
        """Estimate the full covariance vectors.

            Parameters
            ----------
            means : array-like, shape (n_components, n_features)

            covariances : array-like, shape (n_components, n_feature, n_feature)

            nk : array-like, shape (1, n_components)

            post_prob : array-like, shape (n_samples, n_components)

            train_set : array-like, shape (n_samples, n_features)

            Returns
            -------
            covariances : array, shape (n_components, n_features, n_features)
                The covariance vector of the current components.
         """
        covariances_new = np.zeros_like(covariances)
        for index in np.arange(0, self.n_components):
            shift = train_set - means[index, :]
            shift_tmp = post_prob[:, index].reshape(self.n_samples, 1) * shift
            covariances_new[index] = shift_tmp.T.dot(shift) / nk[:, index]
        return covariances_new

    def _update_covariances_diag(self, means, covariances, nk, post_prob, train_set):
        """Estimate the diagonal covariance vectors.

            Parameters
            ----------
            means : array-like, shape (n_components, n_features)

            covariances : array-like, shape (n_components, n_feature, n_feature)

            nk : array-like, shape (1, n_components)

            post_prob : array-like, shape (n_samples, n_components)

            train_set : array-like, shape (n_samples, n_features)

            Returns
            -------
            covariances : array, shape (n_components, n_features, n_features)
                The covariance vector of the current components.
         """
        covariance_new = np.zeros_like(covariances)
        avg_X2 = np.dot(post_prob.T, train_set * train_set) / nk.reshape(-1, 1)
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(post_prob.T, train_set) / nk.reshape(-1, 1)
        covariance_tmp = avg_X2 - 2 * avg_X_means + avg_means2
        for index in range(self.n_components):
            for column in range(self.n_features):
                covariance_new[index, column, column] = covariance_tmp[index, column]
        return covariance_new

    @staticmethod
    def _cal_likelihood(weight, probability_density_matrix):
        """ Calculate the log likelihood function.（计算对数似然函数）
        Parameters
        ----------
        weight : array-like, shape (1, n_components)
            The weights of each mixture components.

        probability_density_matrix : array_like, shape (n_samples, n_components)

        Returns
        -------
        log likelihood_function : float, the value of the log likelihood func.
        """
        sub = np.sum(probability_density_matrix*weight, axis=1)
        logsub = np.log(sub)
        return np.sum(logsub)



if __name__ == "__main__":

    PointSet = construct_dataset()
    # test the diag GMM module.
    cluster = GaussianMixture(PointSet, n_components=2, type="diag")
    cluster.train()
    result = cluster.predict(cluster.train_set)
    cluster.visualize(cluster.train_set, result, cluster.color_iter)

    # test the full GMM module.
    cluster = GaussianMixture(PointSet, n_components=2, type="full")
    cluster.train()
    result = cluster.predict(cluster.train_set)
    cluster.visualize(cluster.train_set, result, cluster.color_iter)

    # test the different K values.
    cluster = GaussianMixture(PointSet, n_components=3, type="full")
    cluster.train()
    result = cluster.predict(cluster.train_set)
    cluster.visualize(cluster.train_set, result, cluster.color_iter)

    cluster = GaussianMixture(PointSet, n_components=4, type="full")
    cluster.train()
    result = cluster.predict(cluster.train_set)
    cluster.visualize(cluster.train_set, result, cluster.color_iter)

    cluster = GaussianMixture(PointSet, n_components=5, type="full")
    cluster.train()
    result = cluster.predict(cluster.train_set)
    # print(cluster.train_set.shape)
    cluster.visualize(cluster.train_set, result, cluster.color_iter)

    # advance part train test split. & record the likelihood on train & test set.
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test = train_test_split(PointSet, test_size=0.2, random_state=120)
    #
    # cluster = GaussianMixture(X_train, test_set=X_test, n_components=2, type="full")
    # cluster.train()
    # result_train = cluster.predict(cluster.train_set)
    # cluster.visualize(cluster.train_set, result_train, cluster.color_iter)
    #
    # result_test = cluster.predict(cluster.test_set)
    # cluster.visualize(cluster.test_set, result_test, cluster.color_iter)




