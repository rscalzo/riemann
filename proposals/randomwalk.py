#!/usr/env/python

"""
RS 2018/03/15:  Proposals for Riemann
"""

import numpy as np
from ..riemann import Proposal, ParameterError


class MetropolisRandomWalk(Proposal):
    """
    A Metropolis random walk proposal.
    """

    def __init__(self, C):
        print "Using MetropolisRandomWalk from inside package"
        self.L = np.linalg.cholesky(C)

    def propose(self, theta):
        theta = np.atleast_1d(theta)
        if self.L.shape[1] != theta.shape[0]:
            raise ParameterError("theta and L have incompatible shapes")
        xi = np.random.normal(size=theta.shape)
        return theta + np.dot(self.L, xi), 0.0


class pCN(Proposal):
    """
    A preconditioned Crank-Nicholson proposal.
    """

    def __init__(self, C, rho):
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
        u_fwd = np.linalg.solve(self.L*self.rho_c, dtheta_fwd)
        u_rev = np.linalg.solve(self.L*self.rho_c, dtheta_rev)
        logqratio = -0.5*((np.dot(u_fwd, u_fwd) -
                           np.dot(u_rev, u_rev)))

        return theta_p, logqratio
