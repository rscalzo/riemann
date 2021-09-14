#!/usr/env/python

"""
RS 2018/03/15:  Proposals for Riemann
"""

import numpy as np
import matplotlib.pyplot as plt
from riemann import Proposal, ParameterError
from .adaptive import AdaptScaleProposal, AdaptCovProposal


def leapfrog(p0, q0, Nsteps, epsilon, mgradU, M=None, debug=False):
    """
    Leapfrog integration step
    :param p0: initial momentum variables; np.array of shape (d, )
    :param q0: initial coordinate variables (thetas) of shape (d, )
    :param Nsteps: number (int) of leapfrog steps to take
    :param eps: step size (float) for HMC steps
    :param mgradU: callable giving grad(log(posterior)) w/rt theta
    :param M: optional mass matrix, of shape (d, d)
    """
    qlist = [q0]
    # Mass matrix accounting
    scale = (M is not None)
    # Initial half-step in momentum
    p = p0 + 0.5*epsilon * mgradU(q0)
    # Full step in coordinates
    psc = np.linalg.solve(M, p) if scale else p
    q = q0 + epsilon * psc
    qlist.append(q)
    # If more than one leapfrog step desired, alternate additional steps
    # in p and q until desired trajectory length reached
    for i in range(Nsteps - 1):
        p = p + epsilon * mgradU(q)
        psc = np.linalg.solve(M, p) if scale else p
        q = q + epsilon * psc
        qlist.append(q)
    # Final half-step in momentum, with reflection (for reversibility)
    p = p + 0.5*epsilon * mgradU(q)
    # Visuals for debugging
    if debug:
        qlist = np.array(qlist)
        plt.plot(qlist[:,0], qlist[:,1], marker='o', ls='-', color='gray')
        plt.plot(qlist[0,0], qlist[0,1], marker='o', ls='-', color='green')
        plt.plot(qlist[-1,0], qlist[-1,1], marker='o', ls='-', color='red')
        ax = plt.gca()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        plt.show()
    # Return
    return p, q


class VanillaHMC(Proposal):
    """
    A vanilla Hamiltonian Monte Carlo proposal.
    """

    def __init__(self, eps, Nsteps, gradlogpost, M=None):
        """
        :param d: dimension of parameter space
        :param eps: value of epsilon for leapfrog integration
        :param Nsteps: number of leapfrog steps (integer >= 1)
        :param gradlogpost: callable giving grad(log(posterior)) w/rt theta
        :param M: optional mass matrix of shape (d, d)
        """
        super().__init__()
        self.Nsteps = Nsteps
        self.eps = eps
        self._mgradU = gradlogpost
        if M is None:
            self.M, self.chM = None, None
        else:
            self.M, self.chM = M, np.linalg.cholesky(M)

    def propose(self, theta):
        # Generate momenta
        theta = np.atleast_1d(theta)
        p_theta = np.random.normal(size=theta.shape)
        if self.chM is not None:
            p_theta = np.dot(self.chM, p_theta)
        # Integrate along the trajectory
        p_theta_new, theta_new = leapfrog(
                p_theta, theta, self.Nsteps, self.eps, self._mgradU, self.M)
        if self.chM is not None:
            p_theta = np.linalg.solve(self.chM, p_theta)
            p_theta_new = np.linalg.solve(self.chM, p_theta_new)
        # Calculate difference in kinetic energy
        logqratio = 0.5*(np.sum(p_theta_new**2) - np.sum(p_theta**2))
        # Return!
        return theta_new, logqratio


class AdaptScaleHMC(AdaptScaleProposal, VanillaHMC):

    def __init__(self, eps, Nsteps, gradlogpost, M=None):
        AdaptScaleProposal.__init__(self, 0.75)
        VanillaHMC.__init__(self, eps, Nsteps, gradlogpost, M=M)
        self.eps0 = self.eps

    def propose(self, theta):
        self.eps = self.scale * self.eps0
        return VanillaHMC.propose(self, theta)


class AdaptCovHMC(AdaptCovProposal, VanillaHMC):

    def __init__(self, eps, Nsteps, gradlogpost, M0,
                 t_adapt=1, marginalize=False, smooth_adapt=False):
        VanillaHMC.__init__(self, eps, Nsteps, gradlogpost, M=M0)
        AdaptCovProposal.__init__(self, M0,
                                  t_adapt=t_adapt,
                                  marginalize=marginalize,
                                  smooth_adapt=smooth_adapt)

    def propose(self, theta):
        self.M, self.chM = self.C, self.L
        return VanillaHMC.propose(self, theta)


class AdaptScaleCovHMC(AdaptScaleHMC, AdaptCovHMC):

    def __init__(self, eps, Nsteps, gradlogpost, M0,
                 t_adapt=1, marginalize=False, smooth_adapt=False):
        AdaptScaleHMC.__init__(self, eps, Nsteps, gradlogpost)
        AdaptCovHMC.__init__(self, eps, Nsteps, gradlogpost, M0,
                             t_adapt=t_adapt,
                             marginalize=marginalize,
                             smooth_adapt=smooth_adapt)

    def propose(self, theta):
        self.eps = self.scale * self.eps0
        self.M, self.chM = self.C, self.L
        return VanillaHMC.propose(self, theta)


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
        super().__init__()
        self.M = M
        self.eps = eps
        self._mU = logpost
        self._mgradU = gradlogpost
        self._p_theta = None
        self._beta = beta
        self._move_idx_count = np.zeros(M+1)

    def propose(self, theta):
        # Generate momenta
        theta = np.atleast_1d(theta)
        p_theta = self._p_theta
        Dp_theta = np.random.normal(size=theta.shape)
        if self._p_theta is None:
            p_theta = Dp_theta
        else:
            p_theta = (np.sqrt(self._beta)*Dp_theta +
                       np.sqrt(1-self._beta)*p_theta)
        # Integrate and store M leapfrog points along the trajectory
        logP0 = self._mU(theta) - 0.5*p_theta**2
        p_theta_list, theta_list = [p_theta], [theta]
        logPlist = np.array([logP0])
        pi_La_list = [0.0]
        pi_FLa_list = [0.0]
        for i in range(self.M):
            # Leapfrog integration
            p_theta_i, theta_i = leapfrog(
                p_theta_list[-1], theta_list[-1], 1, self.eps, self._mgradU)
            logP = self._mU(theta_i) - 0.5*p_theta_i**2
            # Form probability ratio of forward and backward hops
            # and calculate transition probability for this step
            R = np.exp(logP - logP0)
            pi_La = np.min([1.0 - np.sum(pi_La_list),
                           (1.0 - np.sum(pi_FLa_list))*R])
            pi_FLa = np.min([1.0 - np.sum(pi_FLa_list),
                            (1.0 - np.sum(pi_La_list))/R])
            # print("theta, R, 1/R =", theta_i, R, 1.0/R)
            # Append all quantities to chain
            pi_La_list.append(pi_La)
            pi_FLa_list.append(pi_FLa)
            p_theta_list.append(p_theta_i)
            theta_list.append(theta_i)
        # Normalize
        pi_La_list[0] = 1.0 - np.sum(pi_La_list)
        p_theta_list[0] *= -1
        # print("pi_La_list =", pi_La_list)
        # print("pi_FLa_list =", pi_FLa_list)
        # Pick a state!
        idx = np.searchsorted(np.cumsum(pi_La_list), np.random.uniform())
        self._p_theta, theta_new = p_theta_list[idx], theta_list[idx]
        self._move_idx_count[idx] += 1.0
        # There is no reject step so logqratio = infinity for auto-accept
        # if we're using a Metropolis-Hastings sampler to wrap this thing
        logqratio = -np.inf
        # Return!
        return theta_new, logqratio
