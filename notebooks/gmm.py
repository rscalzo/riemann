import numpy as np
import scipy.stats as ss


class GaussianParameters:
    def __init__(self, mean, covariance):
        self.dim = mean.size
        self.mean = mean.ravel()
        self.covariance = covariance.reshape(self.dim, self.dim)

    @staticmethod
    def serialise(means, covariances):
        return np.concatenate((means, covariances.reshape(means.shape[0], -1)), axis=1)

    def serialised(self):
        return self.serialise(self.mean, self.covariance)

    @staticmethod
    def deserialise(v):
        assert v.dim() == 2
        dim = int(((1 + 4 * v.size(-1)) ** 0.5 - 1) / 2)
        n = v.size(0)
        return v[:, :dim], v[:, dim:].reshape(n, dim, dim)

    @staticmethod
    def dimension(v):
        return int(((1 + 4 * v.size(-1)) ** 0.5 - 1) / 2)


class GMMParameters:
    @staticmethod
    def serialise(weights, means, covariances):
        """
        Serialise GMM parameters into a single array

        Parameters:
            weights (torch.Tensor): an array of N scalars
            means (torch.Tensor): an N-by-D matrix of mean locations
            covariances (torch.Tensor): a batch of N D-by-D covariance matrices

        Returns:
            torch.Tensor: a N+ND+ND^2 array with the serialised parameters.
        """
        params_mat = GaussianParameters.serialise(means, covariances)
        params = np.concatenate((weights.reshape(-1, 1), params_mat), axis=1).ravel()
        return params

    @staticmethod
    def deserialise(params, n_components):
        assert params.size / n_components >= 3, "Parameters must encode weights, means and covariances."
        params_mat = params.reshape(n_components, -1)
        weights = params_mat[:, 0].reshape(-1, 1)
        means, covariances = GaussianParameters.deserialise(params_mat[:, 1:])
        return weights, means, covariances

    @staticmethod
    def serialise_no_weights(means, covariances):
        params_mat = GaussianParameters.serialise(means, covariances)
        return params_mat.reshape(-1)

    @staticmethod
    def deserialise_no_weights(params, n_components):
        params_mat = params.reshape(n_components, -1)
        means, covariances = GaussianParameters.deserialise(params_mat)
        return means, covariances


class GMM:
    """
    Gaussian Mixture Model
    """

    def __init__(self, weights, means, covariances):
        """
        Constructor.
        :param weights: an array of N GMM weights
        :param means: a N-by-D tensor of GMM mean D-dimensional locations
        :param covariances: a N-by-D-by-D tensor consisting of N D-by-D covariance matrices
        :param dtype: torch data type to enforce
        :param device: torch device to enforce
        """
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.chol_covs = np.linalg.cholesky(self.covariances)

    @property
    def parameters(self):
        return GMMParameters.serialise(self.weights, self.means, self.covariances)

    @staticmethod
    def from_parameters(params, n_components, no_weights=False):
        """
        Constructs a new GMM object from the given parameters.

        :param params: GMM parameters array, following param_serial's serialisation format
        :param n_components: number of Gaussian components
        :param no_weights: if set, assumes no weights are encoded by the parameters, i.e. equally weighted components.
        :return: a GMM object.
        """
        if no_weights:
            means, covariances = GMMParameters.deserialise_no_weights(params, n_components)
            weights = np.ones(n_components) / n_components
        else:
            weights, means, covariances = GMMParameters.deserialise(params, n_components)

        return GMM(weights, means, covariances)

    def sample(self, n_samples):
        """
        Samples points from the GMM

        :param n_samples: number of samples
        :return: a matrix with samples on each row
        :rtype: torch.Tensor
        """
        n_comp, d = self.means.shape
        z = np.random.randn(n_samples, d)
        sampled_cs = np.random.choice(n_comp, p=self.weights.ravel(), size=n_samples)
        samples = np.empty((n_samples, d))
        for i in range(n_samples):
            c = sampled_cs[i]
            samples[i] = self.means[c] + self.chol_covs[c] @ z[i]
        return samples

    def density(self, x):
        """
        Computes probability density at the given points in x.

        :param x: a matrix with points on each row, or a single row vector
        :return: a scalar with the probability density value
        :rtype torch.Tensor
        """
        nc = self.weights.shape[0]
        pdf_x = np.zeros(x.shape[0])
        for c in range(nc):
            mvn_pdf_x = ss.multivariate_normal.pdf(x, self.means[c], self.covariances[c])
            pdf_x += self.weights[c] * mvn_pdf_x
        return pdf_x
