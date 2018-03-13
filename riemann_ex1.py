#!/usr/env/python

"""
RS 2018/03/06:  Geometric Monte Carlo, Exercise 1

Exercise 1:  implement and test pCN.
Exercise 1:  implement and test infinity-MALA.
"""

import numpy as np
import matplotlib.pyplot as plt
from riemann import Sampler, Proposal, Model
from riemann import ParameterError


def logsumexp(x, axis=None):
    """
    Numerically stable log(sum(exp(x))); ganked from autograd docs:
        https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
    Extended to accept an axis keyword if necessary.
    """
    max_x = np.max(x, axis=axis)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis))


class MetropolisRandomWalk(Proposal):
    """
    A Metropolis random walk proposal.
    """

    def __init__(self, C):
        self.L = np.random.cholesky(C)

    def propose(self, theta):
        if self.L.shape[1] != theta.shape[0]:
            raise ParameterError("theta and L have incompatible shapes")
        xi = np.random.normal(size=theta.shape)
        return theta + np.dot(self.L, xi), 1.0


class pCN(Proposal):
    """
    A preconditioned Crank-Nicholson proposal.
    """

    def __init__(self, C, rho):
        self.L = np.random.cholesky(C)
        self.rho = rho

    def propose(self, theta):
        # proposal theta'
        xi = np.random.normal(size=theta.shape)
        theta_p = self.rho*theta + np.sqrt(1-self.rho**2)*np.dot(self.L, xi)
        # proposal density ratio q(theta'|theta)/q(theta|theta')
        dtheta_fwd = theta_p - self.rho*theta
        dtheta_rev = theta - self.rho*theta_p
        u_fwd = np.linalg.solve(self.L, dtheta_fwd)
        u_rev = np.linalg.solve(self.L, dtheta_rev)
        logqratio = -0.5*(np.dot(u_fwd, u_fwd) - np.dot(u_rev, u_rev))

        return theta, logqratio


class UniGaussian(Model):
    """
    An easier model than the supposedly easy model below.
    """

    def __init__(self, mu, sigma):
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
        self.mu, self.sig ma = theta
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

    def log_likelihood(self, theta, pointwise=True):
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

    def log_likelihood(self, theta, pointwise=True):
        Li = np.array([m.logL_ptwise(theta) for m in model_list])
        return logsumexp(Li, axis=0)

    def log_prior(self, theta, pointwise=True):
        return np.sum([m.logP(theta) for m in model_list])
