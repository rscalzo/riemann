# Adding tests for test suite

import numpy as np
import matplotlib.pyplot as plt
import emcee

from riemann import Sampler
from riemann.proposals.randomwalk import AdaptCovRandomWalk
from riemann.proposals.randomwalk import AdaptScaleRandomWalk
from riemann.proposals.randomwalk import AdaptScaleCovRandomWalk
from riemann.proposals.randomwalk import pCN, AdaptScalepCN
from riemann.proposals.hamiltonian import VanillaHMC,\
    AdaptScaleHMC, AdaptCovHMC, AdaptScaleCovHMC
from riemann.models.benchmarks import benchmark_gauss2d_corr
from riemann.models.benchmarks import benchmark_gauss2d_mix2_cross


def testRandomWalk():
    """
    quick run of adaptive Metropolis through its paces
    :return: nothing
    """
    C0 = 0.0001*np.eye(2)
    proposal = AdaptCovRandomWalk(C0, t_adapt=100, smooth_adapt=False)
    proposal = AdaptScaleRandomWalk(C0)
    proposal = AdaptScaleCovRandomWalk(C0, t_adapt=0, smooth_adapt=False)
    """
    C0 = np.eye(2)
    proposal = pCN(C0, 0.5)
    proposal = AdaptScalepCN(C0, 0.5)
    """
    proposal = VanillaHMC(0.1, 5, benchmark_gauss2d_corr.grad_log_likelihood)
    proposal = AdaptScaleHMC(0.1, 5, benchmark_gauss2d_corr.grad_log_likelihood)
    proposal = AdaptCovHMC(0.1, 5, benchmark_gauss2d_corr.grad_log_likelihood,
                           np.eye(2), t_adapt=100, smooth_adapt=True)
    proposal = AdaptScaleCovHMC(0.1, 5, benchmark_gauss2d_corr.grad_log_likelihood,
                           np.eye(2), t_adapt=100, smooth_adapt=True)
    proposal.scale = 1.0
    sampler = Sampler(benchmark_gauss2d_corr, proposal, np.ones(2))
    sampler.run(10000, 1000, 1)
    chain = np.array(sampler._chain_thetas)
    tau = emcee.autocorr.integrated_time(chain)
    print("chain =", chain)
    print("proposal.scale =", proposal.scale)
    print("chain.acceptrate =", np.mean(chain[:-1] != chain[1:]))
    print("tau =", tau)
    # let's see how we did
    plt.figure(figsize=(6,8))
    plt.subplot(2,1,1)
    plt.plot(chain[:,0], chain[:,1], marker='o', ls='None')
    plt.subplot(2,1,2)
    plt.plot(chain)
    plt.show()
    return chain