#!/usr/env/python

"""
RS 2021/03/31:  Tests for HMC
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from riemann import Sampler
from riemann.proposals.hamiltonian import VanillaHMC
from riemann.proposals.geometric import findiffgrad
from riemann.models.benchmarks import benchmark_gaussNd_corr

def profile_timer(f, *args, **kwargs):
    """
    Times a function call f() and prints how long it took in seconds
    (to the nearest millisecond).
    :param func:  the function f to call
    :return:  same return values as f
    """
    t0 = time.time()
    result = f(*args, **kwargs)
    t1 = time.time()
    print ("time to run {}: {:.3f} sec".format(f.__name__, t1-t0))
    return result

def testVanillaHMC():
    """
    Quick test of regular vanilla HMC proposal
    :return: nothing (if successful)
    """
    # Start with a highly correlated 2-D Gaussian
    N = 10000
    model = benchmark_gaussNd_corr(2, 0.99)
    def glp(q):
        # grad_A = -np.linalg.solve(model.L.T, np.linalg.solve(model.L, q))
        # grad_B = findiffgrad(model, q, 1e-6*np.ones(len(q)))
        return -np.linalg.solve(model.L.T, np.linalg.solve(model.L, q))
        # return findiffgrad(model, q, 1e-6*np.ones(len(q)))

    proposal = VanillaHMC(0.05, 5, glp, M=np.array([[1.0,0.9],[0.9,1.0]]))
    sampler = Sampler(model, proposal, np.array([1,1]))
    profile_timer(sampler.run, N)
    chain = np.array(sampler._chain_thetas)
    acceptrate = 1.0*np.sum(chain[:-1,0] != chain[1:,0])/len(chain)
    print("acceptance rate =", acceptrate)

    # 2-D scatter plot
    plt.plot(chain[:,0], chain[:,1], ls='None', marker='o')
    plt.show()
    # Marginals
    Nbins = 101
    x = np.linspace(-5, 5, 101)
    dx = np.mean(x[1:]-x[:-1])
    xnorm = N*dx/np.sqrt(2*np.pi)
    plt.subplot(2,1,1)
    plt.hist(chain[:,0], range=(-5,5), bins=Nbins)
    plt.plot(x, xnorm*np.exp(-0.5*x**2), lw=2, ls='--')
    plt.subplot(2,1,2)
    plt.hist(chain[:,1], range=(-5,5), bins=Nbins)
    plt.plot(x, xnorm*np.exp(-0.5*x**2), lw=2, ls='--')
    plt.show()
    # Traces
    plt.subplot(2,1,1)
    plt.plot(chain[:,0])
    plt.subplot(2,1,2)
    plt.plot(chain[:,1])
    plt.show()