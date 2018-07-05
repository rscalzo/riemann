#!/usr/env/python

"""
RS 2018/03/15:  Models for Riemann
"""

import numpy as np
from ..riemann import Model, ParameterError


def logsumexp(x, axis=None):
    """
    Numerically stable log(sum(exp(x))); ganked from autograd docs:
        https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
    Extended to accept an axis keyword if necessary.
    """
    max_x = np.max(x, axis=axis)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis))


class UniGaussian(Model):
    """
    An easier model than the supposedly easy model below.
    """

    def __init__(self, mu, sigma):
        self.Npars = 2
        self.mu = mu
        self.sigma = sigma
        self.theta_cached = None

    def load_data(self, data):
        if len(data.shape) != 1:
            raise ParameterError("data needs to be 1-dimensional")
        self.data = data

    def pack(self):
        return np.array([self.mu, self.sigma])

    def unpack(self, theta):
        if theta == self.theta_cached:
            return
        if theta.shape != (2,):
            raise ParameterError("theta should have shape (2,)")
        self.mu, self.sigma = theta
        self.theta_cached = theta

    def log_likelihood(self, theta, pointwise=False):
        self.unpack(theta)
        y = self.data - self.mu
        logL_ptwise = -0.5*(y**2 + np.log(2*np.pi*self.sigma))
        return logL_ptwise if pointwise else np.sum(logL_ptwise)

    def log_prior(self, theta):
        self.unpack(theta)
        return -0.5*self.mu**2 + 1.0/self.sigma**2


class MultiGaussian(Model):
    """
    A multivariate Gaussian, because we start with the easy things.
    """

    def __init__(self, mu, C):
        if C.shape[0] != C.shape[1]:
            raise ParameterError("C has non-square shape {}".format(C.shape))
        if C.shape[1] != mu.shape[0]:
            raise ParameterError("mu and C have incompatible shapes {}, {}"
                                 .format(mu.shape, C.shape))
        self.Ndim = N
        self.Npars = N + N*(N+1)/2
        self.mu = np.zeros(size=(N,))
        self.C = np.eye(size=(N,N))
        self.L = np.sqrt(self.C)
        self.theta_cached = None

    def load_data(self, data):
        if data.shape[0] != self.Ndim:
            raise ParameterError("data and mu have incompatible shapes {}, {}"
                                 .format(data.shape, mu.shape))
        self.data = data

    def pack(self):
        theta = np.array(self.mu)
        for i, Crow in enumerate(self.C):
            theta = np.concatenate([theta, Crow[i:]])
        return theta

    def unpack(self, theta):
        # This will work, technically, but autograd won't like it
        if theta == self.theta_cached:
            return
        if theta.shape != self.Ndim + self.Ndim*(self.Ndim+1)/2:
            raise ParameterError("theta, mu and C have incompatible shapes")
        self.mu = theta[:self.Ndim]
        k = self.Ndim
        for i, Crow in enumerate(self.C):
            self.C[i,i:] = self.C[i:,i] = theta[k:k+(Ndim-i)]
            k += Ndim - i
        eps = 1e-10*np.eye(np.median(np.diag(C)))
        self.L = np.linalg.cholesky(C + eps)
        self.theta_cached = theta

    def log_likelihood(self, theta, pointwise=False):
        self.unpack(theta)
        # Pointwise log likelihood should have shape (Ndata, Npars)
        y = self.data - self.mu[np.newaxis,:]
        u = np.linalg.solve(self.L, y)
        logdetC = 2*np.sum(np.log(np.diag(self.L)))
        logL_ptwise = -0.5*(np.array([np.dot(ui, ui) for ui in u]) +
                            np.log(2*np.pi*logdetC))
        return logL_ptwise if pointwise else np.sum(logL_ptwise)

    def log_prior(self, theta):
        # Put some Wishart prior stuff in here later, when I have a brain
        self.unpack(theta)
        return 1.0


class MixtureModel(Model):
    """
    A mixture model of one or more distributions.  The latent component
    memberships are marginalized out to make things easier.
    """
    
    def __init__(self, model_list):
        self.model_list = model_list

    def load_data(self, data):
        for m in model_list:
            m.load_data(data)

    def pack(self):
        theta = np.array(self.mu)
        for i, Crow in enumerate(self.C):
            theta = np.concatenate([theta, Crow[i:]])
        return theta

    def log_likelihood(self, theta, pointwise=False):
        logLij = np.array([m.logL_ptwise(theta) for m in model_list])
        logL_ptwise = logsumexp(logLij, axis=0)
        return logL_ptwise if pointwise else np.sum(logL_ptwise)

    def log_prior(self, theta, pointwise=False):
        return np.sum([m.logP(theta) for m in model_list])


class SimpleGaussian(Model):
    """
    Just samples a Gaussian without trying to fit anything to data yet.
    """

    def load_data(self, data):
        pass

    def log_posterior(self, theta):
        mu, sigma = 1.0, 0.5
        return -0.5*np.sum((theta - mu)**2/sigma**2)


class SqueezedMultiGaussian(Model):
    """
    Just samples a Gaussian without trying to fit anything to data yet.
    """

    def __init__(self, M, rho=0.0):
        self.Ndim = M
        self.mu = np.zeros(M)
        self.C = (1.0-rho)*np.eye(M) + rho*np.ones((M,M))
        self.L = np.linalg.cholesky(self.C)
        self.theta_cached = None

    def load_data(self, data):
        pass

    def log_posterior(self, theta):
        y = theta - self.mu
        u = np.linalg.solve(self.L, y)
        logdetC = 2*np.sum(np.log(np.diag(self.L)))
        return -0.5*(np.dot(u, u) + len(y)*np.log(2*np.pi) + logdetC)
