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
        self._chain_accept = [ True ]
        self._chain_thetas = [ theta0 ]
        self._chain_logPs = [ model.log_posterior(theta0) ]

    def run(self, Nsamples, Nburn=0, Nthin=1):
        """
        Run the Sampler.
        """
        # Burn in chain; throw away samples, we don't care about them
        self._chain_thetas = self._chain_thetas[-1:]
        self._chain_logPs = self._chain_logPs[-1:]
        for i in range(Nsamples):
            theta, logpost = self.sample()
            self._chain_accept.append(theta != self._chain_thetas[-1])
            self._chain_thetas.append(theta)
            self._chain_logPs.append(logpost)
        self._chain_accept = self._chain_accept[Nburn::Nthin]
        self._chain_thetas = self._chain_thetas[Nburn::Nthin]
        self._chain_logPs = self._chain_thetas[Nburn::Nthin]

    def sample(self):
        """
        Draw a single sample from the MCMC chain, and accept or reject
        using the Metropolis-Hastings criterion.
        """
        theta_old = self._chain_thetas[-1]
        logpost_old = self._chain_logPs[-1]
        theta_prop, logqratio = self.proposal.propose(theta_old)
        logpost = self.model.log_posterior(theta_prop)
        mhratio = min(1, np.exp(logpost - logpost_old - logqratio))
        if np.random.uniform() < mhratio:
            return theta_prop, logpost
        else:
            return theta_old, logpost_old

    def acor(self):
        """
        Computes autocorrelation of the MCMC chain.
        """
        pass

    def print_chain_stats(self):
        """
        Displays useful quantities like the acceptance probability and
        autocorrelation time.
        """
        print "Acceptance probability of chain:  {:.3g}".format(
                np.sum(self._chain_accept)/float(len(self._chain_accept)))
        print "Standard deviation of chain:  {:.3g}".format(
                np.std(self._chain_thetas))


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


class Model(object):
    """
    A class associated with statistical models.  Encapsulates the data
    (perhaps in a plug-in way) and the form of the prior and likelihood.
    """

    def __init__(self):
        pass

    def load_data(self, data):
        """
        Initializer for a dataset.  Should include any checks that the
        data are properly formatted, complete, etc. for this Model.
        :param data:  arbitrary object instance containing data
        """
        raise NotImplementedError("Non-overloaded abstract method!")

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
        Log likelihood of the Model; assumes load_data() has been called
        (if this Model needs data).
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
        return self.log_prior(theta) + self.log_likelihood(theta)

    def logL(self, theta):
        return self.log_likelihood(theta)

    def logP(self, theta):
        return self.log_prior(theta)

    def __call__(self, theta):
        return self.log_posterior(theta)

    def grad_log_posterior(self, theta):
        """
        (Unnormalized) gradient of the log posterior.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return dlogL:  gradient of the log posterior, shape (Npars, )
        """
        return np.sum(self.score_matrix(theta), axis=0)

    def score_matrix(self, theta):
        """
        Score matrix of the Model:  a matrix with the contribution of
        each data point to the gradient of the log posterior.
        S.sum(axis=0) is the gradient needed for Hamiltonian dynamics;
        np.dot(S, S.T) provides the Fisher matrix for geometric methods.
        In practice this will probably involve auto-differentiation.
        :param theta:  parameter vector
        :return S:  np.array of shape (Ndata, Npars)
        """
        raise NotImplementedError("Non-overloaded abstract method!")
