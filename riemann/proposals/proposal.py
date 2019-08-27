class Proposal(object):
    """
    A class associated with Metropolis-Hastings proposals.
    Given a Model and the state of the chain, propose the next state.
    """

    def __init__(self):
        pass

    def propose(self, theta, **kwargs):
        """
        Given a state theta, compute a new state theta'.
        :param theta:  parameter vector specifying Model's current state
        :param kwargs:  other settings to be used by derived classes
        :return theta_p:  proposed new parameter vector q(theta'|theta)
        :return logqratio:  log(q(theta'|theta)/q(theta|theta'))
        """
        raise NotImplementedError("Non-overloaded abstract method!")

    def adapt(self, accepted_theta):
        """
        A hook for Adaptive Proposals to accept feedback from a Sampler.
        Won't be used for most Proposals.  This prevents us from having
        to implement a separate AdaptiveSampler class.
        :param accepted_theta: last accepted parameter vector
        """
        pass