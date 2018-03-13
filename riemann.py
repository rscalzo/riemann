#!/usr/bin/env python

"""
RS 2018/03/11:  Riemann -- a geometric MCMC sampler

This package supports CTDS's investigations into geometric MCMC, as in
"information geometry", using a connection associated with the Fisher metric.
"""

import numpy as np
import matplotlib.pyplot as plt


class RiemannBaseError(Exception):
    """
    Simple base class for Riemann exceptions.
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.msg)


class ParameterError(RiemannBaseError):
    """
    Exception for faulty parameters.
    """
    pass


class Sampler(object):
    """
    A class associated with sampling.  Given a Model and a Proposal,
    sample the Model using the Proposal.
    """

    def run(self, model, proposal, Nburn=0, Nthin=1):
        """
        Run the sampler.
        """
        pass

    def sample(self, model, proposal):
        """
        Draw a single sample from the MCMC chain.
        """
        pass

    def acor(self):
        """
        Computes autocorrelation of the MCMC chain.
        """
        pass


class Proposal(object):
    """
    A class associated with MCMC proposals.  Given a Model and the state
    of the chain, propose the next state.  Supports Models with and/or
    without derivative information.  Supported methods:
        propose:  draw from proposal distribution
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
        raise NotImplementedError("Non-overloaded abstract method!")


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
        raise NotImplementedError("Non-overloaded abstract method!")

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
        raise NotImplementedError("Non-overloaded abstract method!")

    def log_prior(self, theta):
        """
        Log prior of the Model given a parameter vector theta.
        """
        raise NotImplementedError("Non-overloaded abstract method!")

    def log_posterior(self, theta):
        """
        (Unnormalized) log posterior given a parameter vector theta.
        """
        return self.log_prior(theta) + self.log_posterior(theta)

    def logL(self, theta):
        return self.log_likelihood(theta)

    def logP(self, theta):
        return self.log_prior(theta)

    def __call__(self, theta):
        return self.log_posterior(theta)
