#!/usr/env/python

"""
RS 2021/07/29:  Proposals for Riemann -- adaptive mix-in methods
"""

import numpy as np
from riemann import Proposal


class AdaptScaleProposal(Proposal):
    """
    A proposal that adapts an overall scale variable in the adapt stage
    to reach a target acceptance rate (e.g. 0.25 for MHRW or 0.75 for HMC).
    The adapted step size is available as the self.scale attribute.
    """

    def __init__(self, target_accept_rate):
        self.Nsamples = 0
        self.Naccepts = 0
        self.last_theta = None
        self.accept_rate = 0.0
        self.target_accept_rate = target_accept_rate
        self.scale = 1.0

    def adapt(self, theta):
        self.Nsamples += 1
        self.Naccepts += np.any(theta != self.last_theta)
        self.accept_rate = self.Naccepts / float(self.Nsamples)
        rescale_ratio = np.exp(1.0 / float(self.Nsamples))
        if self.accept_rate > self.target_accept_rate:
            self.scale *= rescale_ratio
        else:
            self.scale /= rescale_ratio
        self.last_theta = theta


class AdaptCovProposal(Proposal):
    """
    A proposal that adapts a global covariance matrix scaling similar to
    the Adaptive Metropolis Random Walk proposal of Haario et al. (2001),
    based on the covariance of the chain history.  The adapted covariance
    and its Cholesky factor are available as self.C and self.L.
    """

    def __init__(self, C0, t_adapt=1, marginalize=False, smooth_adapt=False):
        """
        :param C0: initial covariance; np.array of shape (Npars, Npars)
        :param t_adapt: prior weight (in samples) to give covariance
            (or, analogously, a timescale for adaptation)
        :param marginalize: experimental option to adapt step sizes in each
            dimension but not covariances; meant to improve performance on
            posteriors with some local curvature
        :param smooth_adapt: experimental option to smoothly ramp adaptation
            to zero rather than insisting on a sharp cutoff between adaptive
            and non-adaptive phases; meant to avert catastrophic failure if
            the acceptance rate is very low for t < t_adapt
        """
        self.C0 = C0
        self.C = C0
        self.L = np.linalg.cholesky(C0)
        self.t_adapt = t_adapt
        self.marginalize = marginalize
        self.smooth_adapt = smooth_adapt
        self._S = 0.0
        self._SX = 0.0
        self._SX2 = 0.0

    def adapt(self, theta):
        """
        Adapt based on whether the last proposal was accepted or not.
        """
        # Accumulate the accepted theta into the sample covariance
        X = np.atleast_1d(theta)
        self._S += 1
        self._SX += X
        self._SX2 += X[:,None] * X[None,:]
        # Recalculate the covariance if needed.  I suggest doing it on
        # a schedule with intervals increasing like t**2 since this is
        # the schedule on which signal-to-noise accumulates.  We have to
        # take a Cholesky factor again every time we do this...
        n = np.float(self._S)
        if np.sqrt(n)**2 == n and n > 2:
            Cs = (self._SX2 - self._SX[:,None]*self._SX[None,:]/n)/(n-1)
            if self.smooth_adapt:
                # experimental version
                self.C = (n * Cs + self.t_adapt * self.C0) / (n + self.t_adapt)
            else:
                # strict Haario+ 2001 setting
                if n < self.t_adapt:
                    self._cov = self.C0
                else:
                    Creg = np.mean(np.diag(self.C0))*np.eye(Cs.shape[0])
                    self.C = Cs + 1e-12 * Creg
            # if the 'marginalize' option has been set, tune the step sizes
            # in each direction but not the covariances
            if self.marginalize:
                self.C = np.diag(np.diag(self.C))
            # scaling exponent with dimension tuned to give reasonable
            # acceptance rates for multi-D Gaussians in numbers of dimensions
            # for which random walk doesn't totally suck
            self.C /= (self.C.shape[0]) ** 0.4
            self.L = np.linalg.cholesky(self.C)/(self.C.shape[0])**0.2
