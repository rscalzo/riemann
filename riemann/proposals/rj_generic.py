import numpy as np
from .rj_proposals import RJState, RJMapping, RJMatching


class GenericMatchingProp(RJMatching):
    def __init__(self, dim):
        super().__init__()
        if isinstance(dim, int):
            self._base_dim = dim
            self._matching_fn = self._match_base
        if isinstance(dim, list):
            self.dims = dim
            self._matching_fn = self._match_list

    def _match_base(self, state, new_k):
        if new_k > state.idx:
            return np.random.randn(abs(new_k - state.idx) * self._base_dim)

    def _match_list(self, state, new_k):
        dim_diff = self.dims[new_k-1] - self.dims[state.idx-1]
        if dim_diff > 0:
            return np.random.randn(dim_diff)

    def match(self, state, new_k):
        return self._matching_fn(state, new_k)

    def logp(self, k, new_k, u):
        return -0.5 * (u * u).sum() - 0.5 * abs(new_k - k) * np.log(2 * np.pi)


class MoveProp:
    def __init__(self, max_k, no_jump_ratio=1):
        self.max_k = max_k
        self.move_p = np.array([1. / max_k] * max_k)
        self.no_jump_ratio = no_jump_ratio

    def move(self, state):
        p = np.ones((self.max_k,))
        p[state.idx - 1] *= self.no_jump_ratio
        self.move_p = p / p.sum()
        return np.random.choice(self.max_k, p=self.move_p) + 1

    def logp(self, k, new_k):
        return np.log(self.move_p[new_k - 1])


class GenericMapping(RJMapping):
    def __init__(self, means, cov_factors):
        super().__init__()
        self.means = means
        self.cov_factors = cov_factors

    def forward(self, state, u, k_high):
        k_low = state.idx - 1
        k_high = k_high - 1
        new_param = self.means[k_high] + np.dot(self.cov_factors[k_high],
                                                np.vstack([np.linalg.solve(self.cov_factors[k_low],
                                                                           (state.param - self.means[k_low])[:, None]),
                                                           u[:, None]])
                                                ).ravel()
        return RJState(new_param, k_high + 1)

    def backward(self, state, k_low):
        k_high = state.idx - 1
        k_low = k_low - 1
        new_dim = self.means[k_low].shape[-1]
        diff = np.linalg.solve(self.cov_factors[k_high], (state.param - self.means[k_high])[:, None])
        new_param = self.means[k_low] + np.dot(self.cov_factors[k_low], diff[:new_dim]).ravel()
        u = diff[new_dim:]
        return RJState(new_param, k_low + 1), u

    def log_det(self, state, u, k_high):
        k_low = state.idx - 1
        return np.log(np.linalg.det(self.cov_factors[k_high - 1])) - np.log(np.linalg.det(self.cov_factors[k_low]))


class RandomWalk:
    def __init__(self, proposal_sd=1.):
        self.proposal_sd = proposal_sd

    def propose(self, state):
        new_param = state.param + np.random.randn(*state.param.shape) * self.proposal_sd
        log_q_ratio = 0
        return RJState(new_param, state.idx), log_q_ratio
