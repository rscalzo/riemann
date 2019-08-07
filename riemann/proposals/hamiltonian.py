#!/usr/env/python

"""
RS 2018/03/15:  Proposals for Riemann
"""

import numpy as np
from ..riemann import Proposal, ParameterError


def leapfrog(p0, q0, N, epsilon, mgradU):
    """
    Leapfrog integration step
    :param p0: initial momentum variables
    :param q0: initial coordinate variables (thetas)
    :param N: number of leapfrog steps to take
    :paran eps: step size for HMC steps
    :param mgradU: callable giving grad(log(posterior)) w/rt theta
    """
    # Initial half-step in momentum
    p = p0 + 0.5*epsilon * mgradU(q0)
    # Full step in coordinates
    q = q0 + epsilon * p
    # If more than one leapfrog step desired, alternate additional steps
    # in p and q until desired trajectory length reached
    for i in range(N-1):
        p = p + epsilon * mgradU(q)
        q = q + epsilon * p
    # Final half-step in momentum, with reflection (for reversibility)
    p = p + 0.5*epsilon * mgradU(q)
    # Return
    return p, q


class VanillaHMC(Proposal):
    """
    A vanilla Hamiltonian Monte Carlo proposal.
    """

    def __init__(self, eps, M, gradlogpost):
        """
        :param eps: value of epsilon for leapfrog integration
        :param M: number of leapfrog steps (integer >= 1)
        :param gradlogpost: callable giving grad(log(posterior)) w/rt theta
        """
        self.M = M
        self.eps = eps
        self._mgradU = gradlogpost

    def propose(self, theta):
        # Generate momenta
        theta = np.atleast_1d(theta)
        p_theta = np.random.normal(size=theta.shape)
        # Integrate along the trajectory
        p_theta_new, theta_new = leapfrog(
                p_theta, theta, self.M, self.eps, self._mgradU)
        # Calculate difference in kinetic energy
        logqratio = 0.5*(p_theta_new**2 - p_theta**2)
        # Return!
        return theta_new, logqratio
