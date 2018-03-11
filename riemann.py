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
        accept:  evaluate Metropolis-Hastings criterion
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

    pass
