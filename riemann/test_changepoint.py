#!/usr/bin/env python

"""
RS 2019/08/04:  Tests for changepoint regression classes
"""

import sys
sys.path.append("../")

import copy
import numpy as np
import matplotlib.pyplot as plt
from riemann import Sampler, Proposal, Model
from riemann.models import ChangepointParams, ChangepointRegression1D
from riemann.proposals import MetropolisRandomWalk


class ChangepointRegression1DProp(Proposal):
    """
    Proposal for changepoint regression problems with an assumed fixed
    number of changepoints.
    """

    def __init__(self, model, hscale):
        self.model = model
        self.Ndata = len(model.x)
        self.hscale = hscale
        self.P_cumprop_cpx = 0.20
        self.P_cumprop_cpv = 0.40
        self.P_cumprop_sig = 0.60
        self.P_cumprop_dim = 1.00
        self.k = None

    def _reset_proposal_covariances(self, k):
        xmin, xmax = self.model.xmin, self.model.xmax
        C_cpx = 0.01*(xmax-xmin)/(k+1) * np.eye(k)
        C_cpv = 0.01*self.hscale**2/self.Ndata * np.eye(k+1)
        C_sig = np.atleast_2d(0.01*self.hscale)
        self.cpx_prop = MetropolisRandomWalk(C_cpx)
        self.cpv_prop = MetropolisRandomWalk(C_cpv)
        self.cps_prop = MetropolisRandomWalk(C_sig)
        self.k = k
    
    def propose(self, theta):
        theta_new = copy.deepcopy(theta)
        if len(theta.cpx) != self.k:
            self._reset_proposal_covariances(len(theta.cpx))
        if np.random.uniform() < self.P_cumprop_cpx:
            # Propose change to changepoints
            theta_new.cpx, logqratio = self.cpx_prop.propose(theta.cpx)
        elif np.random.uniform() < self.P_cumprop_cpv:
            # Propose change to step function values
            theta_new.cpv, logqratio = self.cpv_prop.propose(theta.cpv)
        elif np.random.uniform() < self.P_cumprop_sig:
            # Propose change to hierarchical noise variance
            theta_new.sig, logqratio = self.cps_prop.propose(theta.sig)
        else:
            # Transdimensional proposal
            if len(theta.cpx) == 0 or np.random.uniform() > 0.5:
                s = np.random.uniform(self.model.xmin, self.model.xmax)
                u = 0.5 + np.random.uniform(-0.1, 0.1)/np.sqrt(self.Ndata)
                theta_new, logqratio = \
                    self.model.add_changepoint(theta, s, u)
                print ("-- proposed:  add s, u =", s, u,
                       "; logqratio =", logqratio)
            else:
                n = np.random.randint(len(theta.cpx))
                theta_new, logqratio = \
                    self.model.subtract_changepoint(theta, n)
                print ("-- proposed:  subtract n =", n,
                       "; logqratio =", logqratio)

        return theta_new, logqratio


def run_changepoint_model_test(model, proposal, theta_true, theta0):
    """
    Various tests for ChangepointRegression1D models
    """

    sampler = Sampler(model, proposal, theta0)
    sampler.run(20000)
    xpred = np.linspace(model.xmin, model.xmax, 100)
    ypred = [ ]
    kpred = [ ]
    theta_pars = { }
    for theta_i in sampler._chain_thetas[10000:]:
        Ncpx = len(theta_i.cpx)
        if Ncpx not in theta_pars:
            theta_pars[Ncpx] = { 'cpx': [ ], 'cpv': [ ], 'sig': [ ] }
        ypred.append(model.predict(theta_i, xpred))
        kpred.append(len(theta_i.cpx))
        theta_pars[Ncpx]['cpx'].append(theta_i.cpx)
        theta_pars[Ncpx]['cpv'].append(theta_i.cpv)
        theta_pars[Ncpx]['sig'].append(theta_i.sig)
    ypred = np.array(ypred)
    ypred_mean = np.mean(ypred, axis=0)
    ypred_std = np.std(ypred, axis=0)

    # Bunch of plots
    fig = plt.figure(figsize=(6,8))
    # Panel 1:  fit to the data showing posterior predictive
    # and locations of true changepoints
    plt.subplot(3, 1, 1)
    plt.errorbar(model.x, model.y, yerr=theta_true.sig,
                 marker='o', ls='None', c='k', label='Data')
    plt.fill_between(xpred, ypred_mean-ypred_std, ypred_mean+ypred_std,
                     color='b', alpha=0.5, label='68% Credibility')
    plt.plot(xpred, ypred_mean,
             color='b', ls='--', lw=2, label='Posterior Mean')
    for cpx_i in theta_true.cpx:
        plt.axvline(cpx_i, ls='-.', c='k')
    plt.suptitle('ChangepointRegression1D Test')
    plt.legend(loc='best')
    # Panel 2:  trace plots for cpx
    for iplt, attr in enumerate(['cpx', 'cpv', 'sig']):
        plt.subplot(3, 3, iplt + 4)
        N0 = 0
        for k in sorted(theta_pars.keys()):
            ypar = theta_pars[k][attr]
            # print("ypar[{}][{}] = {}".format(k, attr, ypar))
            if k > 0:
                plt.plot(range(N0, N0 + len(ypar)), ypar)
            N0 += len(ypar)
        plt.xlabel(attr)
    # Panel 3: histogram on number of components
    plt.subplot(3, 2, 5)
    plt.hist(kpred, range=(-0.5,10.5), bins=11)
    plt.xlabel("# of Changepoints")
    plt.subplot(3, 2, 6)
    plt.plot(kpred)
    plt.xlabel("# of Changepoints")
    # Show 'em all
    plt.subplots_adjust(top=0.92, wspace=0.40, hspace=0.35)
    plt.show()

def test_changepoint_model_01(Ncpx, Ndata, xmin, xmax, hmin, hmax, sig):
    """
    Formulate and run changepoint model test 1:  a single changepoint,
    no trans-dimensional proposal yet.
    """
    theta_true = ChangepointParams(
            np.sort(np.random.uniform(xmin, xmax, size=(Ncpx,))),
            np.random.uniform(hmin, hmax, size=(Ncpx+1,)), sig)
    model = ChangepointRegression1D(
            [ ], [ ], xmin, xmax, 1.0*Ncpx, 2*Ncpx, 1, 1, debug=False)
    model.x, model.y = model.generate_synthetic_data(theta_true, Ndata)
    proposal = ChangepointRegression1DProp(model, hmax-hmin)
    theta0 = ChangepointParams([0.5*(xmin+xmax)], [hmin, hmax], 0.1)
    run_changepoint_model_test(model, proposal, theta_true, theta0)


if __name__ == "__main__":
    """
    # Generic changepoint model on unit interval
    test_changepoint_model_01(1, 25, 0.0, 1.0, 0.0, 1.0)
    # No longer on unit interval!
    test_changepoint_model_01(1, 25, 1.0, 3.0, 0.0, 1.0)
    # No longer unit-scaled
    test_changepoint_model_01(1, 25, 0.0, 1.0, 1.0, 3.0)
    # No longer on unit interval or unit scaled
    test_changepoint_model_01(1, 25, 1.0, 3.0, 1.0, 3.0)
    # Two known changepoints instead of one
    test_changepoint_model_01(2, 25, 1.0, 3.0, 1.0, 3.0)
    """
    # Indeterminate number of changepoints
    test_changepoint_model_01(5, 100, 1.0, 3.0, 1.0, 3.0, 0.1)
