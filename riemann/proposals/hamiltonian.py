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

class LookAheadHMC(Proposal):
    """
    Implementation of the Look-Ahead HMC proposal without detailed balance
    (Sohl-Dickstein, Mudigonda & DeWeese 2014, arXiv:1409.5191)
    """

    def __init__(self, eps, M, logpost, gradlogpost, beta=1.0):
        """
        :param eps: value of epsilon for leapfrog integration
        :param M: maximum number of leapfrog steps (integer >= 1)
        :param beta: momentum corruption parameter, float in interval [0, 1]
            with 0 = deterministic integration, 1 = fully randomized momenta
        :param logpost: callable giving log(posterior) w/rt theta
        :param gradlogpost: callable giving grad(log(posterior)) w/rt theta
        """
        self.M = M
        self.eps = eps
        self._mU = logpost
        self._mgradU = gradlogpost
        self._p_theta = None

    def propose(self, theta):
        # Generate momenta
        theta = np.atleast_1d(theta)
        p_theta = self._p_theta
        Dp_theta = np.random.normal(size=theta.shape)
        if self._p_theta is None:
            p_theta = Dp_theta
        else:
            p_theta = np.sqrt(beta)*Dp_theta + np.sqrt(1-beta)*p_theta
        # Integrate and store M leapfrog points along the trajectory
        logP0 = self._mU(theta) - 0.5*p_theta**2
        plist, qlist = [p_theta], [theta]
        logPlist = np.array([logP0])
        pi_list = np.array([0.0])
        for i in range(self.M):
            p_theta_i, theta_i = leapfrog(
                p_theta, theta, 1, self.eps, self._mgradU)
            logP = self._mU(theta_i) - 0.5*p_theta_i**2
            plist.append(p_theta_i)
            qlist.append(theta_i)
            pi_i = np.min(1.0 - np.sum(pi_list),
                          np.exp(logP - logP0)*(1.0 - np.sum(pi_list*np.exp(logP0 - logP))))
            pi_list = np.concatenate([pi_list, [pi_i]])
        pi_list[0] = 1.0 - np.sum(pi_list)
        # Pick a state!
        idx = np.searchsorted(np.cumsum(pi_list), np.random.uniform())
        theta = theta_list[idx]
        # There is no reject step so logqratio = infinity for auto-accept
        # if we're using a Metropolis-Hastings sampler to wrap this thing
        logqratio = -np.inf
        # Return!
        return theta_new, logqratio
