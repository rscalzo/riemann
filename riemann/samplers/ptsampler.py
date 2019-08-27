#!/usr/env/python

"""
RS 2019/05/10:  Samplersfor Riemann
"""

import numpy as np
# from ..sampling_errors import ParameterError
from riemann import Model, Sampler, ParameterError


class TemperedModel(Model):
    """
    A wrapper for a generic Model to assign it a temperature.  Might not
    play well with planned GradientModel as implemented, or with other
    classes that have special methods; would like to find nicer way for
    a TemperedModel to inherit from some other model -- we really want
    something that generates a new class.
    """

    def __init__(self, base_model, beta):
        """
        :param base_model: Model class instance to wrap
        :param beta: initial beta for this model
        """
        # Sanity checks
        if not (beta >= 0 and beta <= 1):
            raise ParameterError("beta = {} must be a number between 0 and 1".format(beta))
        else:
            self.base_model = base_model
            self.beta = beta

    def set_beta(self, beta):
        self.beta = beta

    def log_likelihood(self, theta):
        return (self.base_model.log_likelihood(theta)) * self.beta

    def log_prior(self, theta):
        return self.base_model.log_prior(theta)


class PTSampler(Sampler):
    """
    A parallel-tempered Sampler, which is basically a meta-Sampler class
    that runs a bunch of different mini-Samplers in parallel.
    """

    def __init__(self, model, proposal, theta0, betas=None, Pswap=0.1):
        """
        :param model: Model class instance to sample
        :param proposal: Proposal class instance to use within chains
        :param theta0: initial parameter guess
        :param betas: optional np.array of shape (Ntemps,)
        :param Pswap: probability per unit time of proposing a swap
        """

        # Initialize betas
        # If the user doesn't provide a temperature ladder, initialize a
        # default ladder.
        if betas is None:
            self.betas = 0.5**np.arange(5)
        elif isinstance(betas, np.array) and len(betas.shape) == 1:
            if betas.shape[0] > 20:
                print ("PTSampler: warning -- more than 20 temperatures"
                       " -- watch out, this could be *really* slow")
            sorted_betas = np.array(sorted(betas))[::-1]
            print ("Initializing temperature ladder with betas =", betas)
            self.betas = betas

        # Other sanity checks
        if not (Pswap > 0 and Pswap < 1):
            raise ParameterError("Pswap must be a number between 0 and 1")
        else:
            self.Pswap = Pswap

        # Set up a ladder of Samplers with different Models
        print("PTSampler: betas =", self.betas)
        self.samplers = [ ]
        for beta in self.betas:
            submodel = TemperedModel(model, beta)
            self.samplers.append(Sampler(submodel, proposal, theta0))

    def run(self, Nsamples, Nburn=0, Nthin=1):
        """
        Run the sampler.
        """
        for i in range(Nsamples):
            self.sample()
        self._chain_thetas = self.samplers[0]._chain_thetas
        self._chain_logpost = self.samplers[0]._chain_logpost

    def sample(self):
        """
        Draw a single sample for each MCMC chain, either by the regular
        Metropolis-Hastings within-chain proposals or through swaps
        :return theta: new parameter vector to add to chain history
        :return logpost: log posterior value to add to chain history
        """
        swapped = [ ]
        Nchains = len(self.samplers)
        for i in range(Nchains):
            u = np.random.uniform()
            if i in swapped:
                # already swapped this chain with preceding chain
                continue
            elif (u > self.Pswap or i == Nchains-1):
                # don't initiate swap; regular within-chain proposal
                self.samplers[i].sample()
            else:
                # swap proposal with next chain down the stack 
                j = i + 1
                theta_i, logpost_ii = self.samplers[i].current_state()
                theta_j, logpost_jj = self.samplers[j].current_state()
                logpost_ij = self.samplers[i].model.log_posterior(theta_j)
                logpost_ji = self.samplers[j].model.log_posterior(theta_i)
                # Metropolis ratio = difference in probability
                mhratio = min(1, np.exp((logpost_ji + logpost_ij) -
                                        (logpost_ii + logpost_jj)))
                if np.random.uniform() < mhratio:
                    self.samplers[i]._add_state(theta_j, logpost_ij)
                    self.samplers[j]._add_state(theta_i, logpost_ji)
                else:
                    self.samplers[i]._add_state(theta_i, logpost_ii)
                    self.samplers[j]._add_state(theta_j, logpost_jj)
                swapped.extend([i, j])
        # return values in case someone wants them
        return np.array(zip([s.current_state() for s in self.samplers]))
