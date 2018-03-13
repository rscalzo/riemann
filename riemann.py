#!/usr/bin/env python

"""
RS 2018/03/11:  Riemann -- a geometric MCMC sampler

This package supports CTDS's investigations into geometric MCMC, as in
"information geometry", using a connection associated with the Fisher metric.
"""

import numpy as np
import matplotlib.pyplot as plt


class Sampler(object):
    """
    A class associated with sampling.  Given a Model and a Proposal,
    sample the Model using the Proposal.  Supported methods:
        sample:  sample the model given a burn-in and thinning ratio
        acor:  compute auto-correlation time of a chain
        R:  compute Gelman-Rubin statistic for a set of chains
    """

    pass


class Proposal(object):
    """
    A class associated with MCMC proposals.  Given a Model and the state
    of the chain, propose the next state.  Supports Models with and/or
    without derivative information.  Supported methods:
        propose:  draw from proposal distribution
        mhratio:  compute asymmetry of proposal density btwn two states
            (taking possible proposal asymmetries into account)
    """

    def __init__(self):
        pass

    def propose(self, theta, dtheta=None):
        """
        Given a state theta, compute a new state theta'.
        :param theta:  parameter vector specifying Model's current state
        :param dtheta:  derivatives of Model around current theta value
        :return theta_p:  proposed new parameter vector q(theta'|theta)
        :return mhratio:  log(q(theta'|theta)/q(theta|theta')) 
        """
        return theta, 1.0

    def mhratio(self, theta1, theta2):
        """
        Computes log ratio of proposal density function between the two
        states theta1 and theta2.
        :param theta1:  parameter vector of state 1
        :param theta2:  parameter vector of state 2
        :return logqr:  log(q(theta2|theta1)/q(theta1|theta2))
        """
        pass


class Model(object):
    """
    A class associated with statistical models.  Encapsulates the data
    (perhaps in a plug-in way) and the form of the prior and likelihood.
    Supported methods:
        pack:  compute Model's internal state from a parameter vector
        unpack:  compute a parameter vector from Model's internal state
        eval:  evaluates the Model, with options to compute derivatives
    """

    def __init__(self):
        pass

    def load_data(self, data):
        """
        Initializer for a dataset.
        """
        pass

    def pack(self):
        """
        Optional:  Compute parameter vector from Model's internal state.
        """
        pass

    def unpack(self, theta):
        """
        Optional:  Compute Model's internal state from parameter vector.
        """
        pass

    def log_likelihood(self, theta):
        """
        Log likelihood of the Model given a parameter vector theta.
        Assumes self.load_data() has been called.
        """
        pass

    def log_prior(self, theta):
        """
        Log prior of the Model given a parameter vector theta.
        """
        pass

    def log_posterior(self, theta):
        """
        (Unnormalized) log posterior given a parameter vector theta.
        """
        pass

    def logL(self, theta):
        return self.log_likelihood(theta)

    def logP(self, theta):
        return self.log_prior(theta)

    def __call__(self, theta):
        return self.log_posterior(theta)
