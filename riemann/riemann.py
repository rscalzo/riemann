#!/usr/bin/env python

"""
RS 2018/03/11:  Riemann -- a geometric MCMC sampler

This package supports CTDS's investigations into geometric MCMC, as in
"information geometry", using a connection associated with the Fisher metric.
"""

import numpy as np


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

    def __init__(self, model, proposal, theta0):
        """
        Initialize a Sampler with a model, a proposal, and a guess
        at some reasonable starting parameters.
        """
        self.model = model
        self.proposal = proposal
        self._chain_thetas = [ theta0 ]
        self._chain_logpost = [ model.log_posterior(theta0) ]

    def run(self, Nsamples, Nburn=0, Nthin=1):
        """
        Run the Sampler.
        """
        # Burn in chain; throw away samples, we don't care about them
        self._chain_thetas = self._chain_thetas[-1:]
        self._chain_logpost = self._chain_logpost[-1:]
        for i in range(Nsamples):
            self.sample()
        self._chain_thetas = self._chain_thetas[Nburn::Nthin]
        self._chain_logpost = self._chain_logpost[Nburn::Nthin]

    def current_state(self):
        """
        Returns a (theta, log_posterior) tuple for the current state.
        """
        return self._chain_thetas[-1], self._chain_logpost[-1]

    def _add_state(self, theta, logpost):
        """
        Adds an explicitly given state to the chain.  Internal use only!
        Useful as a hook for meta-methods like parallel tempering.
        :param theta: parameter vector to add to the chain history
        :param logpost: log posterior value to add to the chain history
        """
        self._chain_thetas.append(theta)
        self._chain_logpost.append(logpost)

    def sample(self):
        """
        Draw a single sample from the MCMC chain, accept or reject via
        the Metropolis-Hastings criterion, and add to the chain history.
        :return theta: new parameter vector added to chain history
        :return logpost: log posterior value added to chain history
        """
        # Retrieve the last state
        theta_old, logpost_old = self.current_state()
        theta_prop, logqratio = self.proposal.propose(theta_old)
        logpost_prop = self.model.log_posterior(theta_prop)
        mhratio = min(0, logpost_prop - logpost_old - logqratio)
        if np.log(np.random.uniform()) < mhratio:
            theta, logpost = theta_prop, logpost_prop
        else:
            theta, logpost = theta_old, logpost_old
        self.proposal.adapt(theta)
        self._add_state(theta, logpost)
        return theta, logpost


class Proposal(object):
    """
    A class associated with Metropolis-Hastings proposals.
    Given a Model and the state of the chain, propose the next state.
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

    def adapt(self, accepted_theta):
        """
        A hook for Adaptive Proposals to accept feedback from a Sampler.
        Won't be used for most Proposals.  This prevents us from having
        to implement a separate AdaptiveSampler class.
        :param accepted_theta: last accepted parameter vector
        """
        pass


class Model(object):
    """
    A class associated with statistical models.  Encapsulates the data
    (perhaps in a plug-in way) and the form of the prior and likelihood.
    """

    def __init__(self):
        pass

    def pack(self):
        """
        Optional:  Compute parameter vector from Model's internal state.
        :return theta:  parameter vector as np.array of shape (Npars, )
        """
        pass

    def unpack(self, theta):
        """
        Optional:  Compute Model's internal state from parameter vector.
        :param theta:  parameter vector as np.array of shape (Npars, )
        """
        pass

    def log_likelihood(self, theta):
        """
        Log likelihood of the Model.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return logL:  log likelihood
        """
        raise NotImplementedError("Non-overloaded abstract method!")

    def log_prior(self, theta):
        """
        Log prior of the Model.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return logL:  log prior
        """
        raise NotImplementedError("Non-overloaded abstract method!")

    def log_posterior(self, theta):
        """
        (Unnormalized) log posterior of the Model.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return logpost:  log posterior
        """
        logP, logL = self.log_prior(theta), self.log_likelihood(theta)
        if np.any([np.isinf(logP), np.isnan(logP),
                   np.isinf(logL), np.isnan(logL)]):
            logpost = -np.inf
        else:
            logpost = logP + logL
        return logpost

    def logL(self, theta):
        return self.log_likelihood(theta)

    def logP(self, theta):
        return self.log_prior(theta)

    def __call__(self, theta):
        return self.log_posterior(theta)
