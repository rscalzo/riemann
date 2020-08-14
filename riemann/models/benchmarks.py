#!/usr/env/python

"""
Benchmark distributions for test suite or common examples
"""

# 1-D unit Gaussian
benchmark_gauss1d = MultiGaussianDist(0.0, 1.0)

# 2-D isotropic unit Gaussian
benchmark_gauss2d_iso = MultiGaussianDist(np.zeros(2), np.eye(2))

# 2-D unit Gaussian with strong correlations
benchmark_gauss2d_corr = MultiGaussianDist(
    np.zeros(2), 0.1*np.eye(2) + 0.9*np.ones((2,2)))

# 100-D isotropic unit Gaussian
benchmark_gauss100d_iso = MultiGaussianDist(np.zeros(100), np.eye(100))

# 100-D unit Gaussian with strong correlations
benchmark_gauss100d_corr = MultiGaussianDist(
    np.zeros(100), 0.1*np.eye(100) + 0.9*np.ones((100,100)))

# preliminary infrastructure for defining mixture distributions; in these
# model instances theta always just represents a point in parameter space
# and so the constituent models all have to have the same calling signature
class MixtureDist(Model):

    def __init__(self, components):
        """
        :param components: list of (weight, dist) tuples where weight is a
            float and dist is a Model instance describing a distribution
        """
        self._mixweights, self._model_list = zip(components)
        self._mixweights = np.array(self._mixweights)/np.sum(self._mixweights)
        self._cumulwts = np.cumsum(self._mixweights)

    def log_likelihood(self, theta):
        """
        Log likelihood of the Model.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return logL:  log likelihood
        """
        mix_logLs = np.array(
            [m.log_likelihood(theta) for m in self._model_list])
        return np.log(np.sum(self._mixweights * np.exp(mix_logLs)))

    def draw(self, Ndraws=1):
        """
        Draw from the mixture distribution.
        :param Ndraws:
        :return: np.array with shape (Ndraws, Ndim)
        """
        idx = np.searchsorted(self._cumulwts, np.random.uniform(size=Ndraws))
        draws = [ ]
        for i in idx:
            draws.append(self._model_list[i].draw(1))
        return np.array(draws)

# mixture of two 1-D Gaussians; re-express in terms of MixtureDistribution
# or MixtureModel infrastructure once we have that infrastructure sorted
benchmark_gauss1d_mix2 = MixtureDist(
    [[0.3, MultiGaussianDist(-5.0, 1.0)],
     [0.7, MultiGaussianDist(+5.0, 1.0)]],
)

# mixture of two 2-D gaussians with strong correlations
benchmark_gauss2d_mix2_cross = MixtureDist(
    [[0.3, MultiGaussianDist([0, 0], [[1.0,  0.9], [ 0.9, 1.0]])],
     [0.7, MultiGaussianDist([0, 0], [[1.0, -0.9], [-0.9, 1.0]])]],
)