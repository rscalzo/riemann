#!/usr/bin/env python

"""
RS 2018/03/11:  Riemann -- a geometric MCMC sampler

This package supports CTDS's investigations into geometric MCMC, as in
"information geometry", using a connection associated with the Fisher metric.
"""

import numpy as np
import matplotlib.pyplot as plt


class RiemannBaseError(Exception):
    """
    Simple base class for Riemann exceptions.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.msg)


class ParameterError(RiemannBaseError):
    """
    Exception for faulty parameters.
    """
    pass


