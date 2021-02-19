try:
    import Riemann
except ImportError:
    print("Fudging sys.path to support in-place execution in local sandbox")
    import sys
    sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from riemann import Model, Sampler, Proposal
from riemann import ParameterError
from riemann.models.gaussian import MultiGaussianDist
from riemann.samplers.localemulator import MHLocalEmulatorSampler
from riemann.proposals.hamiltonian import VanillaHMC as HMC
from riemann.proposals.hamiltonian import EmulatorHMC
from sklearn.neighbors import KernelDensity

def lp_test(p):
    return -1/10*p[0]**4 - 1/2*(2*p[1] - p[0]**2)**2

def lp_test_grad(p):
    return np.array([-4/10*p[0]**3 + 2*p[0]*(2*p[1] - p[0]**2),  - 2*(2*p[1] - p[0]**2)])

class lp_gauss(object):
    def __init__(self, covariance):
        self.inv_covariance = np.linalg.inv(covariance)
    def __call__(self, p):
        return float(-np.dot(p.T, np.dot(self.inv_covariance, p)))

class lp_gauss_grad(object):
    def __init__(self, covariance):
        self.inv_covariance = np.linalg.inv(covariance)
    def __call__(self, p):
        return -2 * np.dot(p.T, self.inv_covariance.T).flatten()

covariance = np.eye(2)
proposal = EmulatorHMC(0.1, 10)
model = MultiGaussianDist(np.zeros(2), covariance)
sampler = MHLocalEmulatorSampler(model, proposal, np.zeros(2), optimise_refinement = False, R2_refinement = True, R2_threshold = 0.9, approximation_degree = 1)
sampler.run(400)

plt.plot(np.array(sampler._chain_thetas)[:, 0])
plt.show()
plt.plot(np.array(sampler._chain_thetas)[:, 1])
plt.show()
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(sampler._chain_thetas))
kde.score_samples(np.array(sampler._chain_thetas))
plt.tricontourf(np.array(sampler._chain_thetas)[:, 0], np.array(sampler._chain_thetas)[:, 1], kde.score_samples(np.array(sampler._chain_thetas)))
plt.show()
plt.scatter(np.array(sampler._chain_thetas)[:, 0], np.array(sampler._chain_thetas)[:, 1])
plt.show()
