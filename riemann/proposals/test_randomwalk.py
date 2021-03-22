# Adding tests for test suite

import numpy as np
import matplotlib.pyplot as plt

from riemann import Sampler
from riemann.proposals.randomwalk import AdaptiveMetropolisRandomWalk
from riemann.models.benchmarks import benchmark_gauss2d_corr


def testAdaptiveMetropolisRandomWalk():
    """
    quick run of adaptive Metropolis through its paces
    :return: nothing
    """
    proposal = AdaptiveMetropolisRandomWalk(0.1*np.eye(2), t_adapt=0,
                                            smooth_adapt=False)
    sampler = Sampler(benchmark_gauss2d_corr, proposal, np.ones(2))
    sampler.run(10000, 1000, 1)
    chain = np.array(sampler._chain_thetas)
    print("chain =", chain)
    # let's see how we did
    plt.plot(chain[:,0], chain[:,1], marker='o', ls='None')
    plt.show()
    return chain