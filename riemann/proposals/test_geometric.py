#!/usr/env/python

"""
RS 2021/03/31:  Tests for geometric elements incl finite differencing
"""

import numpy as np
import matplotlib.pyplot as plt
from riemann.proposals.geometric import findiffdir, findiffgrad, findiffG
from riemann.models.gaussian import MultiGaussianDist, MultiGaussianModel

f = lambda x: np.sum(x ** 2)
gradf = lambda x: 2 * x

def test_findiffdir():
    x = np.array([1, 1])
    n = x/np.sqrt(np.sum(x**2))
    dfdx_findiff = findiffdir(f, x, 1e-6*x)
    dfdx_exact = np.dot(gradf(x), n)
    print("dfdx_findiff =", dfdx_findiff)
    print("dfdx_exact   =", dfdx_exact)
    np.testing.assert_allclose(dfdx_findiff, dfdx_exact)

def test_findiffgrad():
    x = np.array([1, 1])
    grad_findiff = findiffgrad(f, x, 1e-6*x)
    grad_exact = gradf(x)
    print("grad_findiff =", grad_findiff)
    print("grad_exact   =", grad_exact)
    np.testing.assert_allclose(grad_findiff, grad_exact)

def test_findiffG():
    mu_0, C_0 = 0.0, 3.0
    ptrue = MultiGaussianDist(mu_0, C_0)
    data = ptrue.draw(100)
    plt.hist(data, range=(-10,10), bins=25)
    plt.show()
    model = MultiGaussianModel(mu_0, C_0, data)
    model.parscales = 1e-6*np.ones(2)
    mu_range = np.arange(-3.0, 3.1, 0.5)
    sig_range = np.arange(0.5, 10.0, 0.5)
    mu_grid, sig_grid = np.meshgrid(mu_range, sig_range)
    lpvals, arrows = [ ], [ ]
    for mu_i, sig_i in zip(mu_grid.ravel(), sig_grid.ravel()):
        theta = np.array([mu_i, sig_i])
        G = findiffG(model, theta)
        print("theta =", theta)
        print("G =", G)
        l, lv = np.linalg.eig(G)
        lpvals.append(model(theta))
        arrow1 = np.concatenate([[mu_i, sig_i], 1/np.sqrt(l[0])*lv[0]])
        arrow2 = np.concatenate([[mu_i, sig_i], 1/np.sqrt(l[1])*lv[1]])
        print(">> {}".format(arrow1))
        print(">> {}".format(arrow2))
        arrows.append(arrow1)
        arrows.append(arrow2)
    plt.plot(mu_grid, sig_grid, ls='None', marker='o')
    for a in arrows:
        plt.arrow(a[0], a[1], a[2], a[3])
    plt.show()
