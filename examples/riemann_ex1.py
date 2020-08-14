#!/usr/env/python

"""
RS 2018/03/06:  Geometric Monte Carlo, Exercise 1

Exercise 1:  implement and test pCN.
Exercise 1:  implement and test infinity-MALA.
"""

try:
    import Riemann
except ImportError:
    print("Fudging sys.path to support in-place execution in local sandbox")
    import sys
    sys.path.append("../")

from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
from riemann import Model, Sampler, Proposal
from riemann import ParameterError
from riemann.models.gaussian import MultiGaussianDist
from riemann.proposals.hamiltonian import VanillaHMC as HMC
from riemann.proposals.hamiltonian import LookAheadHMC


def logsumexp(x, axis=None):
    """
    Numerically stable log(sum(exp(x))); ganked from autograd docs:
        https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
    Extended to accept an axis keyword if necessary.
    """
    max_x = np.max(x, axis=axis)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis))


class MetropolisRandomWalk(Proposal):
    """
    A Metropolis random walk proposal.
    """

    def __init__(self, C):
        self.L = np.linalg.cholesky(C)

    def propose(self, theta):
        theta = np.atleast_1d(theta)
        if self.L.shape[1] != theta.shape[0]:
            raise ParameterError("theta and L have incompatible shapes")
        xi = np.random.normal(size=theta.shape)
        return theta + np.dot(self.L, xi), 0.0


class pCN(Proposal):
    """
    A preconditioned Crank-Nicholson proposal.
    """

    def __init__(self, C, rho):
        self.rho = rho
        self.rho_c = np.sqrt(1-self.rho**2)
        self.L = np.linalg.cholesky(C)

    def propose(self, theta):
        # proposal theta'
        theta = np.atleast_1d(theta)
        xi = np.random.normal(size=theta.shape)
        theta_p = self.rho*theta + self.rho_c*np.dot(self.L, xi)
        # proposal density ratio q(theta'|theta)/q(theta|theta')
        dtheta_fwd = theta_p - self.rho*theta
        dtheta_rev = theta - self.rho*theta_p
        u_fwd = np.linalg.solve(self.L*self.rho_c, dtheta_fwd)
        u_rev = np.linalg.solve(self.L*self.rho_c, dtheta_rev)
        logqratio = -0.5*((np.dot(u_fwd, u_fwd) -
                           np.dot(u_rev, u_rev))) # / self.rho_c**2
        """
        print "theta      = {}".format(theta)
        print "theta_p    = {}".format(theta_p)
        print "dtheta_fwd = {}".format(dtheta_fwd)
        print "dtheta_rev = {}".format(dtheta_rev)
        print "u_fwd      = {}".format(u_fwd)
        print "u_rev      = {}".format(u_rev)
        print "logqratio  = {}".format(logqratio)
        """

        return theta_p, logqratio


class UniGaussian(Model):
    """
    An easier model than the supposedly easy model below.
    """

    def __init__(self, mu, sigma):
        self.Npars = 2
        self.mu = mu
        self.sigma = sigma
        self.theta_cached = None

    def load_data(self, data):
        if len(data.shape) != 1:
            raise ParameterError("data needs to be 1-dimensional")
        self.data = data

    def pack(self):
        return np.array([self.mu, self.sigma])

    def unpack(self, theta):
        if theta == self.theta_cached:
            return
        if theta.shape != (2,):
            raise ParameterError("theta should have shape (2,)")
        self.mu, self.sigma = theta
        self.theta_cached = theta

    def log_likelihood(self, theta, pointwise=False):
        self.unpack(theta)
        y = self.data - self.mu
        logL_ptwise = -0.5*(y**2 + np.log(2*np.pi*self.sigma))
        return logL_ptwise if pointwise else np.sum(logL_ptwise)

    def log_prior(self, theta):
        self.unpack(theta)
        return -0.5*self.mu**2 + 1.0/self.sigma**2


class MultiGaussian(Model):
    """
    A multivariate Gaussian, because we start with the easy things.
    """

    def __init__(self, mu, C):
        if C.shape[0] != C.shape[1]:
            raise ParameterError("C has non-square shape {}".format(C.shape))
        if C.shape[1] != mu.shape[0]:
            raise ParameterError("mu and C have incompatible shapes {}, {}"
                                 .format(mu.shape, C.shape))
        N = len(mu)
        self.Ndim = N
        self.Npars = N + N*(N+1)/2
        self.mu = np.zeros((N,))
        self.C = np.eye(N)
        self.L = np.sqrt(self.C)
        self.theta_cached = None

    def load_data(self, data):
        if data.shape[0] != self.Ndim:
            raise ParameterError("data and mu have incompatible shapes {}, {}"
                                 .format(data.shape, mu.shape))
        self.data = data

    def pack(self):
        theta = np.array(self.mu)
        for i, Crow in enumerate(self.C):
            theta = np.concatenate([theta, Crow[i:]])
        return theta

    def unpack(self, theta):
        # This will work, technically, but autograd won't like it
        if theta == self.theta_cached:
            return
        if theta.shape != self.Ndim + self.Ndim*(self.Ndim+1)/2:
            raise ParameterError("theta, mu and C have incompatible shapes")
        self.mu = theta[:self.Ndim]
        k = self.Ndim
        for i, Crow in enumerate(self.C):
            self.C[i,i:] = self.C[i:,i] = theta[k:k+(Ndim-i)]
            k += Ndim - i
        eps = 1e-10*np.eye(np.median(np.diag(C)))
        self.L = np.linalg.cholesky(C + eps)
        self.theta_cached = theta

    def log_likelihood(self, theta, pointwise=False):
        self.unpack(theta)
        # Pointwise log likelihood should have shape (Ndata, Npars)
        y = self.data - self.mu[np.newaxis,:]
        u = np.linalg.solve(self.L, y)
        logdetC = 2*np.sum(np.log(np.diag(self.L)))
        logL_ptwise = -0.5*(np.array([np.dot(ui, ui) for ui in u]) +
                            np.log(2*np.pi*logdetC))
        return logL_ptwise if pointwise else np.sum(logL_ptwise)

    def log_prior(self, theta):
        # Put some Wishart prior stuff in here later, when I have a brain
        self.unpack(theta)
        return 1.0


class MixtureModel(Model):
    """
    A mixture model of one or more distributions.  The latent component
    memberships are marginalized out to make things easier.
    """
    
    def __init__(self, model_list):
        self.model_list = model_list

    def load_data(self, data):
        for m in model_list:
            m.load_data(data)

    def pack(self):
        theta = np.array(self.mu)
        for i, Crow in enumerate(self.C):
            theta = np.concatenate([theta, Crow[i:]])
        return theta

    def log_likelihood(self, theta, pointwise=False):
        logLij = np.array([m.logL_ptwise(theta) for m in model_list])
        logL_ptwise = logsumexp(logLij, axis=0)
        return logL_ptwise if pointwise else np.sum(logL_ptwise)

    def log_prior(self, theta, pointwise=False):
        return np.sum([m.logP(theta) for m in model_list])


class SimpleGaussian(Model):
    """
    Just samples a Gaussian without trying to fit anything to data yet.
    """

    def load_data(self, data):
        pass

    def log_posterior(self, theta):
        return -0.5*np.sum(theta**2)


def test_sampling_gauss1d():
    """
    Sample a distribution by various methods.
    """
    # Let's just do a univariate Gaussian for now.
    N, d = 1000, 2
    theta0 = np.random.normal((d,))
    model = SimpleGaussian()
    model = MultiGaussianDist(np.zeros(d), np.eye(d))
    logpost = model.log_posterior
    gradlogpost = grad(model.log_posterior)
    # proposal = MetropolisRandomWalk([[1]])
    # proposal = pCN([[1]], 0.1)
    proposal = HMC(1.5, 3, gradlogpost)
    # proposal = LookAheadHMC(1.5, 3, logpost, gradlogpost, beta=0.1)
    sampler = Sampler(model, proposal, theta0)
    sampler.run(N)
    #sampler.print_chain_stats()
    # Plot the samples
    plt.subplot(2, 1, 1)
    x = np.linspace(-5, 5, 26)
    dx = np.mean(x[1:] - x[:-1])
    Xd = np.array(sampler._chain_thetas)
    if len(Xd.shape) > 1:  Xd = Xd[:,0]
    plt.hist(Xd, bins=26, range=(-5,5), color='b')
    plt.plot(x, N*dx*np.exp(-0.5*x**2)/np.sqrt(2*np.pi),
             color='r', ls='--', lw=2)
    plt.subplot(2, 1, 2)
    plt.plot(sampler._chain_thetas)
    plt.show()


if __name__ == "__main__":
    test_sampling_gauss1d()
