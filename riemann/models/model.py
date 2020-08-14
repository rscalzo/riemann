import numpy as np


class Model(object):
    """
    A class associated with statistical models.  Encapsulates the data
    (perhaps in a plug-in way) and the form of the prior and likelihood.
    """

    def __init__(self):
        pass

    def pack(self):
        """
        Optional:  Compute parameter vector from Model's internal state.
        :return theta:  parameter vector as np.array of shape (Npars, )
        """
        pass

    def unpack(self, theta):
        """
        Optional:  Compute Model's internal state from parameter vector.
        :param theta:  parameter vector as np.array of shape (Npars, )
        """
        pass

    def log_likelihood(self, theta):
        """
        Log likelihood of the Model.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return logL:  log likelihood
        """
        raise NotImplementedError("Non-overloaded abstract method!")

    def log_prior(self, theta):
        """
        Log prior of the Model.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return logL:  log prior
        """
        raise NotImplementedError("Non-overloaded abstract method!")

    def log_posterior(self, theta):
        """
        (Unnormalized) log posterior of the Model.
        :param theta:  parameter vector as np.array of shape (Npars, )
        :return logpost:  log posterior
        """
        logP, logL = self.log_prior(theta), self.log_likelihood(theta)
        if np.any([np.isinf(logP), np.isnan(logP),
                   np.isinf(logL), np.isnan(logL)]):
            logpost = -np.inf
        else:
            logpost = logP + logL
        return logpost

    def logL(self, theta):
        return self.log_likelihood(theta)

    def logP(self, theta):
        return self.log_prior(theta)

    def __call__(self, theta):
        return self.log_posterior(theta)