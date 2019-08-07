#!/usr/bin/env python

"""
RS 2019/07/19:  Changepoint Regression Model

This is a model ideally meant to be sampled by trans-dimensional MCMC
schemes such as reversible jump.  It's a proof of concept for writing
a generalizable interface for trans-dimensional problems.
"""

import copy
import autograd.numpy as np
from autograd import jacobian
from scipy.special import gammaln
from ..riemann import Model


def log_gamma_dist(alpha, beta, x):
    return alpha*np.log(beta) + (alpha-1)*np.log(x) - beta*x - gammaln(alpha)


class ChangepointParams(object):
    """
    A parameter object that fully specifies the state of a
    ChangepointRegression1D instance as defined below.
    """
    
    def __init__(self, cpx, cpv, sig):
        """
        :param cpx:  np.array of shape (N,) with changepoint locations
        :param cpv:  np.array of shape (N+1,) with model values between
            changepoints, i.e the constant pieces of the model
        :param sig:  hierarchical Gaussian noise standard deviation
        """
        if len(cpv) != len(cpx) + 1:
            raise ValueError("number of constant pieces ({}) must be "
                             "1 more than number of changepoints ({})"
                             .format(len(cpv), len(cpx)))
        self.cpx = np.array(cpx)
        self.cpv = np.array(cpv)
        self.sig = np.array(sig)

    def __str__(self):
        return ("ChangepointParams instance with sig = {}, "
                "cpx = {}, cpv = {}".format(self.sig, self.cpx, self.cpv))


def add_changepoint_mapping(p):
    """
    Deterministic mapping (h, u) -> (h1, h2) for adding a changepoint.
    :param p: np.array containing (h, u)
    :return: np.array containing (h1, h2)
    """
    # u[0] = h (height of previous step)
    # u[1] = u (log height delta, between 0 and 1)
    # Return two new heights
    h, u = p
    ufac = np.sqrt((1-u)/u)
    return np.array([h/ufac, h*ufac])

def add_changepoint_mapping_inv(p):
    """
    Inverse mapping (h, u) <- (h1, h2) for removing a changepoint.
    :param p: np.array containing (h1, h2)
    :return: np.array containing (h, u)
    """
    h1, h2 = p
    h = np.sqrt(h1*h2)
    u = 1.0/(1 + h2/h1)
    return np.array([h, u])

def add_changepoint_logjac(p):
    jac = jacobian(add_changepoint_mapping)(p)
    return np.log(np.abs(np.linalg.det(jac)))

def add_changepoint_logjac_inv(p):
    jac = jacobian(add_changepoint_mapping_inv)(p)
    return np.log(np.abs(np.linalg.det(jac)))


class ChangepointRegression1D(Model):
    """
    1-D piecewise constant changepoint regression model
    """

    def __init__(self, x, y, xmin, xmax, lamb, kmax, alpha, beta, debug=False):
        """
        :param x: np.array of shape (M,) with 1-D predictors
        :param y: np.array of shape (M,) with 1-D responses
        """
        if len(x) != len(y):
            raise ValueError("length of predictor array ({}) must be"
                             "same as length of response array ({})"
                             .format(len(x), len(y)))
        self.x = np.array(x)
        self.y = np.array(y)
        # Fixed hyperparameters
        self.xmin = xmin
        self.xmax = xmax
        self.kmax = kmax
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.debug = debug

    def log_likelihood(self, theta):
        """
        :param theta:  ChangepointParams instance
        """
        # Evaluate the model:  a piecewise constant function.
        # WATCH OUT:  this way of doing it won't be auto-differentiable
        # with respect to the changepoint vector -- see if we can find
        # a cleverer way to do it that will in principle allow us to use
        # HMC or other gradient methods.  Not a big deal for now as it
        # shouldn't affect us implementing RJMCMC.
        ypred = self.predict(theta, self.x)
        # Evaluate the log likelihood
        logdetC = len(self.x)*np.log(theta.sig**2)
        logNd2pi = len(self.x)*np.log(2*np.pi)
        logL = -0.5*(np.sum((ypred-self.y)**2 / theta.sig**2)
                     + logdetC + logNd2pi)
        if self.debug:
            print("ChangepointRegression1D:  logL = {:.3f}".format(float(logL)))
        if np.isnan(logL):
            logL = -np.inf
        return logL

    def log_prior(self, theta):
        """
        :param theta:  ChangepointParams instance
        """
        # Number of steps:  Poisson distributed
        k = len(theta.cpv)
        logP_k = k*np.log(self.lamb) - gammaln(k) - self.lamb
        # Heights of steps:  gamma(alpha, beta) distributed
        logP_v = np.sum(log_gamma_dist(self.alpha, self.beta, theta.cpv))
        # Changepoint locations:  order statistic distributed
        # Peter Green uses even order statistics only, to avoid stupidly
        # infinitesimal no-data intervals.  The thing written down below
        # is the log joint prior over these intervals.
        L = self.xmax - self.xmin
        s = np.concatenate([[self.xmin], theta.cpx, [self.xmax]])
        logP_s = gammaln(2*k+1) + np.sum(np.log(s[1:] - s[:-1])) - k*np.log(L)
        # Use an improper Jeffery's prior for the noise variance
        logP_sig2 = np.log(1.0/theta.sig**2)
        if theta.sig < 0:
            logP_sig2 = np.nan
        # Diagnostics printing
        if self.debug:
            print("ChangepointRegression1D: theta =", theta)
            print("ChangepointRegression1D: "
                  "logP_k = {:.3f}, logP_v = {:.3f}, "
                  "logP_s = {:.3f}, logP_sig2 = {:.3f}"
                  .format(float(logP_k), float(logP_v),
                          float(logP_s), float(logP_sig2)))
        logP = logP_k + logP_v + logP_s + logP_sig2
        # Return result, or -np.inf if we walked off the edge of the priors
        if np.isnan(logP):
            logP = -np.inf
        return logP

    def generate_synthetic_data(self, theta, Ndata):
        """
        Generate a synthetic dataset from parameters theta.
        :param theta:  ChangepointParams instance
        :param Ndata:  number of data points
        :return:  (x, y) pairs corresponding to synthetic dataset
        """
        L = self.xmax - self.xmin
        x = np.sort(self.xmin + L*np.random.uniform(size=(Ndata,)))
        epsilon = np.random.normal(size=x.shape)
        y = self.predict(theta, x) + theta.sig*epsilon
        return x, y

    def predict(self, theta, x):
        """
        Evaluate the model at the given set of x-values.
        :param theta:  ChangepointParams instance
        :param x:  np.array
        """
        return theta.cpv[np.searchsorted(theta.cpx, x)]

    def plot(self, theta, x, y):
        """
        Plot the data with values shown.
        """
        cpredx = np.concatenate(
                [[self.xmin], np.repeat(theta.cpx, 2), [self.xmax]])
        cpredy = np.concatenate(theta.cpv, 2)
        plt.errorbar(x, y, yerr=theta.sig, ls='None', marker='o')
        plt.plot(cpredx, cpredy, ls='-.')

    def add_changepoint(self, theta, s, u):
        """
        :param theta:  ChangepointParams instance
        :param s:  new changepoint location, 0 < s < L
        :param u:  new height delta parameter, 0 < u < 1
        """
        # Sanity check on parameters
        if not (self.xmin < s < self.xmax):
            raise ValueError("ChangepointRegression1D.add_changepoint: "
                             "require 0 < s < L for new changepoint")
        if not (0 < u < 1):
            raise ValueError("ChangepointRegression1D.add_changepoint: "
                             "require 0 < u < 1 for new changepoint")
        # Split the segment in which the new changepoint s appears
        n = np.searchsorted(theta.cpx, s)
        h = theta.cpv[n]
        p = np.array([h, u])
        h1, h2 = add_changepoint_mapping(p)
        logjac = add_changepoint_logjac(p)
        # Modify the existing ChangepointParams instance
        cpx = np.concatenate([theta.cpx[:n], [s], theta.cpx[n:]])
        cpv = np.concatenate([theta.cpv[:n], [h1, h2], theta.cpv[n+1:]])
        sig = theta.sig
        theta_p = ChangepointParams(cpx, cpv, sig)
        # Return theta-prime and the log Jacobian
        return theta_p, logjac

    def subtract_changepoint(self, theta, n):
        """
        :param theta:  ChangepointParams instance
        :param n:  index of changepoint to remove
        """
        # Sanity check on parameters
        if n not in range(len(theta.cpx)):
            raise ValueError("ChangepointRegression1D.add_changepoint: "
                             "n has to be a valid changepoint index")
        # Remove changepoint n and deterministically remap the model
        h1, h2 = theta.cpv[n], theta.cpv[n+1]
        p = np.array([h1, h2])
        h, u = add_changepoint_mapping_inv(p)
        logjac = add_changepoint_logjac_inv(p)
        # Modify the existing ChangepointParams instance
        cpx = np.concatenate([theta.cpx[:n], theta.cpx[n+1:]])
        cpv = np.concatenate([theta.cpv[:n], [h], theta.cpv[n+2:]])
        sig = theta.sig
        theta_p = ChangepointParams(cpx, cpv, sig)
        # Return theta-prime and the log Jacobian
        return theta_p, logjac
