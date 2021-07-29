#!/usr/env/python

"""
RS 2018/03/15:  Models for Riemann
"""

import numpy as np
from riemann import Model, ParameterError


def logsumexp(x, axis=None):
    """
    Numerically stable log(sum(exp(x))); ganked from autograd docs:
        https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
    Extended to accept an axis keyword if necessary.
    """
    max_x = np.max(x, axis=axis)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis))


class MultiGaussianDist(Model):
    """
    A multivariate Gaussian distribution with known mean and covariance.
    Probably a little over-engineered for a one-dimensional Gaussian...
    """

    def __init__(self, mu, C):
        """
        :param mu: mean; np.array of shape (Ndim,)
        :param C: covariance; np.array of shape (Ndim, Ndim)
        """
        # Sanity checks for input
        mu = np.atleast_1d(mu)
        C = np.atleast_2d(C)
        if C.shape[0] != C.shape[1]:
            raise ParameterError("C has non-square shape {}".format(C.shape))
        if C.shape[1] != mu.shape[0]:
            raise ParameterError("mu and C have incompatible shapes {}, {}"
                                 .format(mu.shape, C.shape))
        # Cache Cholesky factor and log det for later use
        self.mu = mu
        self.L = np.linalg.cholesky(C)
        self.logdetC = 2*np.sum(np.log(np.diag(self.L)))
        self.Ndim = len(mu)

    def log_prior(self, theta):
        return 0.0

    def log_likelihood(self, theta):
        y = theta - self.mu
        u = np.linalg.solve(self.L, y)
        return -0.5*(np.dot(u, u) + len(y)*np.log(2*np.pi) + self.logdetC)

    def draw(self, Ndraws=1):
        """
        Draw from the Gaussian with these parameters.
        :return: np.array with shape (Ndraws, Ndim)
        """
        epsilon = np.random.normal(size=(Ndraws, self.mu.shape[0]))
        return np.dot(epsilon, self.L.T) + self.mu


class MultiGaussianModel(Model):
    """
    A multivariate Gaussian distribution with unknown mean + covariance
    meant to be deployed in constructions like mixture models.
    """

    def __init__(self, mu, C, data):
        """
        :param mu: init guess mean; np.array of shape (Ndim,)
        :param C: init guess covariance; np.array of shape (Ndim, Ndim)
        """
        # Sanity checks for input
        mu = np.atleast_1d(mu)
        C = np.atleast_2d(C)
        self.Ndim = N = mu.shape[0]
        if C.shape[0] != C.shape[1]:
            raise ParameterError("C has non-square shape {}".format(C.shape))
        if C.shape[1] != N:
            raise ParameterError("mu and C have incompatible shapes {}, {}"
                                 .format(mu.shape, C.shape))
        if data.shape[1] != self.Ndim:
            raise ParameterError("data and mu have incompatible shapes {}, {}"
                                 .format(data.shape, self.mu.shape))
        
        self.Npars = N + N*(N+1)/2      # total number of parameters
        self._theta_cached = None       # for lazy evaluation
        self.data = data

    def pack(self):
        # TODO this is horribly wrong given unpack() below
        theta = np.array(self.mu)
        for i, Crow in enumerate(self.C):
            theta = np.concatenate([theta, Crow[i:]])
        return theta

    def unpack(self, theta):
        # This will work, technically, but autograd probably won't like it
        if np.all(theta == self._theta_cached):
            return
        if len(theta) != self.Ndim + self.Ndim*(self.Ndim+1)/2:
            raise ParameterError("theta, mu and C have incompatible shapes")
        # Represent covariance directly as lower-triangular Cholesky factor,
        # in the hopes of improving numerical stability
        self.mu = theta[:self.Ndim]
        k = self.Ndim
        self.L = np.zeros((k, k))
        for i, Lrow in enumerate(self.L):
            # self.C[i,i:] = self.C[i:,i] = theta[k:k+(self.Ndim-i)]
            self.L[i:,i] = theta[k:k+(self.Ndim-i)]
            k += self.Ndim - i
        self.C = np.dot(self.L, self.L.T)
        self.logdetC = 2*np.sum(np.log(np.diag(self.L)))
        self.logNd2pi = self.Ndim*np.log(2*np.pi)
        self._theta_cached = theta

    def log_likelihood_pointwise(self, theta):
        self.unpack(theta)
        # Pointwise log likelihood should have shape (Ndata, Npars)
        y = self.data - self.mu[np.newaxis,:]
        # np.linalg.solve(A, b) solves Ax = b
        # so A.shape = (Npars, Npars), b.shape = (Npars, Ndata)
        # meaning x.shape = (Npars, Ndata)
        u = np.linalg.solve(self.L, y.T).T
        logL_ptwise = -0.5*(np.array([np.dot(ui, ui) for ui in u]) +
                            self.logNd2pi + self.logdetC)
        return logL_ptwise

    def log_likelihood(self, theta):
        return np.sum(self.log_likelihood_pointwise(theta))

    def log_prior(self, theta):
        # Put some Wishart prior stuff in here later, when I have a brain
        self.unpack(theta)
        if np.any(np.diag(self.L) <= 0):
            logP = -np.inf
        else:
            logP = 0.0
        return logP


class MixtureDist(Model):
    """
    A mixture distribution with fixed mixture weights.
    """

    def __init__(self, model_list, weights):
        self._models = model_list
        self.weights = weights
        self._thetas = [ ]

    def log_likelihood(self, theta):
        return logsumexp([np.log(self.weights[i]) +
                          self._models[i].log_likelihood(theta)
                          for i in range(len(self._models))])
                
    def log_prior(self, theta):
        return 0.0
    

class MixtureModel(Model):
    """
    A mixture model of one or more distributions.  The latent component
    memberships are marginalized out to make things easier.
    """
    
    def __init__(self, model_list):
        self.model_list = model_list

    def pack(self):
        theta = np.array(self.mu)
        for i, Crow in enumerate(self.C):
            theta = np.concatenate([theta, Crow[i:]])
        return theta

    def log_likelihood(self, theta):
        logLij = np.array([m.logL(theta) for m in self.model_list])
        return logsumexp(logLij, axis=0)

    def log_prior(self, theta):
        return np.sum([m.logP(theta) for m in self.model_list])
