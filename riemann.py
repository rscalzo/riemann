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

    def __init__(self, model, proposal, data, theta0):
        """
        Initialize a Sampler with a model, a proposal, data, and a guess
        at some reasonable starting parameters.
        """
        self.model = model
        self.proposal = proposal
        self.model.load_data(data)
        self._chain_thetas = [ theta0 ]
        self._chain_logPs = [ model.log_posterior(theta) ]

    def run(self, model, proposal, Nsamples, Nburn=0, Nthin=1):
        """
        Run the Sampler.
        """
        # Burn in chain; throw away samples, we don't care about them
        self._chain_thetas = self._chain_thetas[-1:]
        self._chain_logPs = self._chain_logPs[-1:]
        for i in range(Nburn):
            theta, logpost = self.sample()
        # Reset and sample chain, keeping every Nthin-th sample
        self._chain_thetas = self._chain_thetas[-1:]
        self._chain_logPs = self._chain_logPs[-1:]
        for i in range(Nsamples):
            theta, logpost = self.sample()
            if i % Nthin = 0:
                self._chain_thetas.append(theta)
                self._chain_logPs.append(logpost)

    def sample(self):
        """
        Draw a single sample from the MCMC chain, and accept or reject
        using the Metropolis-Hastings criterion.
        """
        theta_old = self._chain_thetas[-1]
        theta_prop, logqratio = self.proposal.propose(theta_old)
        logpost = self.model.log_posterior(theta_prop)
        mhratio = min(1, np.exp(logpost + logqratio))
        if np.random.uniform() < mhratio:
            return theta_prop, logpost
        else:
            return theta_new, logpost

    def acor(self):
        """
        Computes autocorrelation of the MCMC chain.
        """
        pass


class Proposal(object):
    """
    A class associated with MCMC proposals.  Given a Model and the state
    of the chain, propose the next state.  Supports Models with and/or
    without derivative information.
    """

    def __init__(self):
        pass

    def propose(self, theta, **kwargs):
        """
        Given a state theta, compute a new state theta'.
        :param theta:  parameter vector specifying Model's current state
        :param kwargs:  other settings to be used by derived classes
        :return theta_p:  proposed new parameter vector q(theta'|theta)
        :return logqratio:  log(q(theta'|theta)/q(theta|theta')) 
        """
        raise NotImplementedError("Non-overloaded abstract method!")


class Model(object):
    """
    A class associated with statistical models.  Encapsulates the data
    (perhaps in a plug-in way) and the form of the prior and likelihood.
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
