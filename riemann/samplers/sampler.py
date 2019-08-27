#!/usr/env/python
# Copyright 2019 The Centre for Translational Data Science (CTDS) 
# at the University of Sydney. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Base code for all MH-based samplers.
2019-08-27
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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