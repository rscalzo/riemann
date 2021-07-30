#!/usr/env/python

"""
RS 2018/03/15:  Proposals for Riemann
"""

import numpy as np
from riemann import Proposal, ParameterError
from .adaptive import AdaptScaleProposal, AdaptCovProposal


class MetropolisRandomWalk(Proposal):
    """
    A Metropolis random walk proposal.
    """

    def __init__(self, C):
        self.scale = 1.0
        self.L = np.linalg.cholesky(np.atleast_2d(C))

    def propose(self, theta):
        theta = np.atleast_1d(theta)
        if self.L.shape[1] != theta.shape[0]:
            raise ParameterError("theta and L have incompatible shapes")
        xi = np.random.normal(size=theta.shape)
        return theta + self.scale * np.dot(self.L, xi), 0.0


class AdaptScaleRandomWalk(AdaptScaleProposal, MetropolisRandomWalk):
    """
    A Metropolis random walk that adapts only the overall scale, maintaining
    the covariance structure (relative step sizes) provided by the user.
    """

    def __init__(self, C):
        AdaptScaleProposal.__init__(self, 0.25)
        MetropolisRandomWalk.__init__(self, C)


class AdaptCovRandomWalk(AdaptCovProposal, MetropolisRandomWalk):
    """
    An adaptive Metropolis random walk similar to Haario et al. (2001)
    that updates the proposal shape based on the chain covariance.
    The default behavior reproduces Haario et al. (2001) but there are
    other experimental options.
    """

    def __init__(self, C0, t_adapt=1, marginalize=False, smooth_adapt=False):
        MetropolisRandomWalk.__init__(self, C0)
        AdaptCovProposal.__init__(self, C0,
                                  t_adapt=t_adapt,
                                  marginalize=marginalize,
                                  smooth_adapt=smooth_adapt)


# Some aliases for backward compatibility
AdaptiveMetropolisRandomWalk = HaarioRandomWalk = AdaptCovRandomWalk


class AdaptScaleCovRandomWalk(AdaptScaleRandomWalk, AdaptCovRandomWalk):
    """
    A Metropolis random walk that adapts both scale and covariance.
    Experimental and not obviously correct.
    """

    def __init__(self, C0, t_adapt=1, marginalize=False, smooth_adapt=False):
        AdaptScaleRandomWalk.__init__(self, 0.25)
        AdaptCovRandomWalk.__init__(self, C0,
                                    t_adapt=t_adapt,
                                    marginalize=marginalize,
                                    smooth_adapt=smooth_adapt)

    def adapt(self, theta):
        AdaptScaleRandomWalk.adapt(self, theta)
        AdaptCovRandomWalk.adapt(self, theta)


class pCN(Proposal):
    """
    A preconditioned Crank-Nicholson proposal.
    """

    def __init__(self, C, rho):
        self.scale = 1.0
        self.rho = rho
        self.rho_c = np.sqrt(1-self.rho**2)
        self.L = np.linalg.cholesky(C)

    def propose(self, theta):
        # proposal theta'
        theta = np.atleast_1d(theta)
        xi = np.random.normal(size=theta.shape)
        theta_p = self.rho*theta + self.rho_c*np.dot(self.L, xi)
        # proposal density ratio q(theta'|theta)/q(theta|theta')
        dtheta_fwd = theta_p - self.rho*theta
        dtheta_rev = theta - self.rho*theta_p
        u_fwd = np.linalg.solve(self.L * self.rho_c, dtheta_fwd)
        u_rev = np.linalg.solve(self.L * self.rho_c, dtheta_rev)
        logqratio = -0.5*((np.dot(u_fwd, u_fwd) -
                           np.dot(u_rev, u_rev)))
        return theta_p, logqratio


class AdaptScalepCN(AdaptScaleProposal, pCN):
    """
    A preconditioned Crank-Nicholson proposal with an adaptive scaling
    parameter targeting a fixed acceptance rate of 0.25.
    """

    def __init__(self, C, rho):
        AdaptScaleProposal.__init__(self, 0.25)
        pCN.__init__(self, C, rho)
        self.rho0 = self.rho

    def propose(self, theta):
        # Rescale the rho parameter to adapt the acceptance rate,
        # given that 0 < rho < 1 for valid proposals
        self.rho = np.tanh(self.rho/self.scale)
        return pCN.propose(self, theta)