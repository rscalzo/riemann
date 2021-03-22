#!/usr/env/python

"""
RS 2018/03/15:  Proposals for Riemann
"""

import numpy as np
from riemann import Proposal, ParameterError


class MetropolisRandomWalk(Proposal):
    """
    A Metropolis random walk proposal.
    """

    def __init__(self, C):
        self._L = np.linalg.cholesky(np.atleast_2d(C))

    def propose(self, theta):
        theta = np.atleast_1d(theta)
        if self._L.shape[1] != theta.shape[0]:
            raise ParameterError("theta and L have incompatible shapes")
        xi = np.random.normal(size=theta.shape)
        return theta + np.dot(self._L, xi), 0.0


class AdaptiveMetropolisRandomWalk(MetropolisRandomWalk):
    """
    An adaptive Metropolis random walk similar to Haario et al. (2001)
    that updates the proposal shape based on the covariance of the
    chain history.
    """

    def __init__(self, C0, t_adapt=1, marginalize=False, smooth_adapt=False):
        """
        :param C0: initial covariance; np.array of shape (Npars, Npars)
        :param t_adapt: prior weight (in samples) to give covariance
            (or, analogously, a timescale for adaptation)
        :param marginalize: option to adapt step sizes in each dimension
            but not covariances
        """
        self.C0 = C0
        self.t_adapt = t_adapt
        self.marginalize = marginalize
        self.smooth_adapt = smooth_adapt
        # Internal state variables associated with adaptation
        self._L = np.linalg.cholesky(C0)
        self._S = 0.0
        self._SX = 0.0
        self._SX2 = 0.0

    def adapt(self, accepted_theta):
        """
        Adapt based on whether the last proposal was accepted or not.
        """
        # Accumulate the accepted theta into the sample covariance
        X = np.atleast_1d(accepted_theta)
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
                # experimental version that hopefully averts catastrophic
                # failure if the acceptance rate is very low for t < t_adapt
                C = (n*Cs + self.t_adapt*self.C0)/(n + self.t_adapt)
            else:
                # strict Haario+ 2001 setting
                if n < self.t_adapt:
                    C = self.C0
                else:
                    Creg = np.mean(np.diag(self.C0))*np.eye(Cs.shape[0])
                    C = Cs + 1e-12*Creg
            # if the 'marginalize' option has been set, tune the step sizes
            # in each direction but not the covariances -- this is useful
            # for tuning random walks in posteriors with some curvature
            if self.marginalize:
                C = np.diag(np.diag(C))
            # scaling exponent with dimension tuned to give reasonable
            # acceptance rates for multi-D Gaussians in numbers of dimensions
            # for which random walk doesn't totally suck
            self._L = np.linalg.cholesky(C)/(C.shape[0])**0.2


class pCN(Proposal):
    """
    A preconditioned Crank-Nicholson proposal.
    """

    def __init__(self, C, rho):
        self.rho = rho
        self.rho_c = np.sqrt(1-self.rho**2)
        self._L = np.linalg.cholesky(C)

    def propose(self, theta):
        # proposal theta'
        theta = np.atleast_1d(theta)
        xi = np.random.normal(size=theta.shape)
        theta_p = self.rho*theta + self.rho_c*np.dot(self._L, xi)
        # proposal density ratio q(theta'|theta)/q(theta|theta')
        dtheta_fwd = theta_p - self.rho*theta
        dtheta_rev = theta - self.rho*theta_p
        u_fwd = np.linalg.solve(self._L*self.rho_c, dtheta_fwd)
        u_rev = np.linalg.solve(self._L*self.rho_c, dtheta_rev)
        logqratio = -0.5*((np.dot(u_fwd, u_fwd) -
                           np.dot(u_rev, u_rev)))

        return theta_p, logqratio
