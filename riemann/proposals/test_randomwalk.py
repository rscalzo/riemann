# Adding tests for test suite

import numpy as np
import matplotlib.pyplot as plt

from riemann import Sampler
from riemann.proposals.randomwalk import AdaptCovRandomWalk
from riemann.proposals.randomwalk import AdaptScaleRandomWalk
from riemann.proposals.randomwalk import AdaptScaleCovRandomWalk
from riemann.proposals.randomwalk import pCN, AdaptScalepCN
from riemann.models.benchmarks import benchmark_gauss2d_corr


def testRandomWalk():
    """
    quick run of adaptive Metropolis through its paces
    :return: nothing
    """
    C0 = 0.0001*np.eye(2)
    proposal = AdaptCovRandomWalk(C0, t_adapt=0, smooth_adapt=False)
    """
    proposal = AdaptScaleRandomWalk(C0)
    proposal = AdaptScaleCovRandomWalk(C0, t_adapt=0, smooth_adapt=False)
    C0 = np.eye(2)
    proposal = pCN(C0, 0.5)
    proposal = AdaptScalepCN(C0, 0.5)
    """
    sampler = Sampler(benchmark_gauss2d_corr, proposal, np.ones(2))
    sampler.run(10000, 1000, 1)
    chain = np.array(sampler._chain_thetas)
    print("chain =", chain)
    print("proposal.scale =", proposal.scale)
    print("chain.acceptrate =", np.mean(chain[:-1] != chain[1:]))
    # let's see how we did
    plt.figure(figsize=(6,8))
    plt.subplot(2,1,1)
    plt.plot(chain[:,0], chain[:,1], marker='o', ls='None')
    plt.subplot(2,1,2)
    plt.plot(chain)
    plt.show()
    return chain