"""
Created by Created by Hadi Afshar 
2019-08-21
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.
import scipy.stats as stats
from tqdm import tqdm
import contribs.mice.mice_data

import matplotlib.pyplot as plt

from sklearn import preprocessing


# from sklearn.gaussian_process import GaussianProcessRegressor
# GaussianProcessRegressor.predict()

def calcK(mX1, mX2, sig2_f, l):
    n, q = mX1.shape
    m, q2 = mX2.shape
    assert q2 == q
    mK = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            deltaX = mX1[i] - mX2[j]
            mK[i, j] = sig2_f * np.exp(-deltaX.dot(deltaX) / (2 * l))
    return mK


def create_fake_data(n, q, theta, lower_bound, upper_bound):
    mX = np.random.uniform(low=lower_bound, high=upper_bound, size=(n, q))

    mK = calcK(mX1=mX, mX2=mX, sig2_f=theta.sigma2_f, l=theta.l)
    print('mK:\n', mK, '\n --------\n')
    vy = np.random.multivariate_normal(mean=np.zeros(n), cov=mK + theta.sigma2_e * np.eye(n))
    return mX, vy


def create_fake_data_linear(n, q, theta, lower_bound, upper_bound):
    mX = np.random.uniform(low=lower_bound, high=upper_bound, size=(n, q))
    vw = np.ones(q) * theta.sigma2_f  # np.random.uniform(low=0, high=theta.sigma2_f, size=q)
    vy = mX.dot(vw) + theta.l * np.random.normal(loc=np.zeros(n), scale=np.power(theta.sigma2_e, 0.5))
    return mX, vy


def do_plot(title_name, xs, ys):
    # assert len(X_names) == len(values)
    # y_pos = range(len(ys))
    plt.plot(xs, ys, 'r.')
    # plt.xticks(y_pos, X_names)
    # plt.ylabel(y_name)
    plt.title(title_name)
    plt.show()


def pdf_y_given_params(vy, theta, mX):
    n = len(vy)
    try:
        k = calcK(mX1=mX, mX2=mX, sig2_f=theta.sigma2_f, l=theta.l)
        pdf = stats.multivariate_normal(mean=np.zeros(n), cov=k +
                                                              theta.sigma2_e * np.eye(n)).pdf(vy)
    except ValueError as e:  # todo what should I do with ValueError: the input matrix must be positive semidefinite
        print(e)
        pdf = 0

    # print('pdf: ', pdf)
    return pdf


class Theta:
    def __init__(self, sigma2_f, l, sigma2_e):
        self.sigma2_f = sigma2_f
        self.l = l
        self.sigma2_e = sigma2_e

    def propose(self, eps_f, eps_l, eps_e):
        s2f, l, s2e = 0.0, 0.0, 0.0
        while s2f <= 0.0:
            s2f = np.random.normal(loc=self.sigma2_f, scale=np.sqrt(eps_f))
        while l <= 0.0:
            l = np.random.normal(loc=self.l, scale=np.sqrt(eps_l))
        while s2e <= 0.0:
            s2e = np.random.normal(loc=self.sigma2_e, scale=np.sqrt(eps_e))

        return Theta(sigma2_f=s2f,
                     l=l,
                     sigma2_e=s2e)

    def addTo(self, other_theta):
        self.sigma2_f += other_theta.sigma2_f
        self.l += other_theta.l
        self.sigma2_e += other_theta.sigma2_e

    def mult(self, scalar):
        return Theta(self.sigma2_f * scalar, self.l * scalar, self.sigma2_e * scalar)

    def copy(self):
        return Theta(self.sigma2_f, self.l, self.sigma2_e)

    def __str__(self):
        return "[sigma2_f: {0:9.3f},\tl: {1:9.3f},\tsigma2_e: {2:9.3f}]".format(self.sigma2_f, self.l, self.sigma2_e)


def deprecated_old_calc_f_star_mean_cov(mX_test, vy, mX, theta):
    """
    :param mX_test: test feature matrix (X*)
    :param vy: training output vector (of size n)
    :param mX: training feature matrix (n x q)
    :param theta: parameters
    :return: (mean(f*), cov(f*)) where
            mean(f*) := E[f* | X, y, X, theta] =
                         K(X*, X; theta) . [K(X, X; theta) + sigma_e^2 . I_n]^{-1} . y
            cov(f*) := K(X*, X*; theta) - K(X*, X; theta) [K(X, X; theta)]^{-1} K(X, X*; theta)
    """

    # K(X, X*)
    mK_train_test = calcK(mX1=mX, mX2=mX_test, sig2_f=theta.sigma2_f, l=theta.l)

    # K(X*, X)
    # mK_test_train = calcK(mX1=mX_test, mX2=mX, sig2_f=theta.sigma2_f, l=theta.l)
    mK_test_train = mK_train_test.transpose()

    # K(X,X)
    mK_train_train = calcK(mX1=mX, mX2=mX, sig2_f=theta.sigma2_f, l=theta.l)

    # K(X,X) + sigma2_e . I_n
    noisyKXX = mK_train_train + (theta.sigma2_e + 0.01) * np.eye(N=mX.shape[0])  # note: 0.01 is jitter for instability

    noisyKXX_inverse = np.linalg.inv(noisyKXX)

    mean_f = mK_test_train.dot((noisyKXX_inverse).dot(vy))

    # K(X*, X*)
    mK_test_test = calcK(mX1=mX_test, mX2=mX_test, sig2_f=theta.sigma2_f, l=theta.l)

    cov_f = mK_test_test - mK_test_train.dot(noisyKXX_inverse).dot(mK_train_test)

    return mean_f, cov_f


def calc_f_star_mean_cov(mX_test, vy, mX, theta):
    """
    :param mX_test: test feature matrix (X*)
    :param vy: training output vector (of size n)
    :param mX: training feature matrix (n x q)
    :param theta: parameters
    :return: (mean(f*), cov(f*)) where
            mean(f*) := E[f* | X, y, X, theta] =
                         K(X*, X; theta) . [K(X, X; theta) + sigma_e^2 . I_n]^{-1} . y
            cov(f*) := K(X*, X*; theta) - K(X*, X; theta) [K(X, X; theta)]^{-1} K(X, X*; theta)
    """

    # K(X, X*) a.k.a K*
    mK_train_test = calcK(mX1=mX, mX2=mX_test, sig2_f=theta.sigma2_f, l=theta.l)

    # K(X*, X)
    # mK_test_train = calcK(mX1=mX_test, mX2=mX, sig2_f=theta.sigma2_f, l=theta.l)
    mK_test_train = mK_train_test.transpose()

    # K(X,X)
    mK_train_train = calcK(mX1=mX, mX2=mX, sig2_f=theta.sigma2_f, l=theta.l)

    # K(X,X) + sigma2_e . I_n
    noisyKXX = mK_train_train + (theta.sigma2_e + 0.01) * np.eye(N=mX.shape[0])  # note: 0.01 is jitter for instability

    # noisyKXX_inverse = np.linalg.inv(noisyKXX)
    L = np.linalg.cholesky(noisyKXX)
    # alpha = L.T \ (L \ y)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, vy))  # todo linalg.solve or linalg.lstsq ?
    # mean_f = mK_test_train.dot((noisyKXX_inverse).dot(vy))
    mean_f = mK_test_train.dot(alpha)

    # v = L \ K(X, X*)
    v = np.linalg.solve(L, mK_train_test)
    # K(X*, X*)
    mK_test_test = calcK(mX1=mX_test, mX2=mX_test, sig2_f=theta.sigma2_f, l=theta.l)

    # cov_f = mK_test_test - mK_test_train.dot(noisyKXX_inverse).dot(mK_train_test)
    # cov_f = K(X*, X*) - v.T v
    cov_f = mK_test_test - v.T.dot(v)

    return mean_f, cov_f


def sort_based_on_dim(mX, dim_to_sort_on, vy=None):
    """
    :param mX: an (n x q) matrix
    :param vy: an n dimensional vector
    :param dim_to_sort_on: a column index 0 <= dim_to_sort_on < q
    :return: sort rows of mX and their corresponding y such that dim_to_sort_on'th column of X is sorted
    Example:
    mX = np.array([[1, 10], [5, 4], [3, 30]])
    vy = np.array([100, 200, 300])
    """
    indexes = np.argsort(mX[:, dim_to_sort_on])
    sorted_mX = mX[indexes]
    sorted_vy = None if vy is None else vy[indexes]
    return sorted_mX, sorted_vy


def main():
    pass


if __name__ == "__main__":
    main()
