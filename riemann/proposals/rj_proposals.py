from abc import ABC, abstractmethod
from typing import List

from riemann import Proposal


class RJState:
    def __init__(self, param, idx):
        self.param = param
        assert idx > 0
        self.idx = idx


class RJMapping(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, state, u, k_high):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, state, k_low):
        raise NotImplementedError()

    @abstractmethod
    def log_det(self, state, u, k_high):
        raise NotImplementedError()


class RJMatching(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def match(self, state, new_k):
        raise NotImplementedError()

    @abstractmethod
    def logp(self, k, new_k, u):
        raise NotImplementedError()


class JumpProposal:
    def __init__(self, mapping: RJMapping, matching_prop: RJMatching):
        super().__init__()
        self.mapping = mapping
        self.matching_prop = matching_prop

    def propose(self, state, new_k):  # TODO: Allow for jumping between models with same dimensionality.
        k = state.idx
        u = self.matching_prop.match(state, new_k)
        if u is not None:  # moving to higher dimension
            new_state = self.mapping.forward(state, u, new_k)
            logqratio = self.mapping.log_det(state, u, new_k) - self.matching_prop.logp(k, new_k, u)
        else:  # moving to lower dimension
            new_state, u = self.mapping.backward(state, new_k)
            logqratio = self.matching_prop.logp(k, new_k, u) - self.mapping.log_det(new_state, u, k)
        return new_state, logqratio


class RJProposal(Proposal):
    def __init__(self, move_prop, jump_prop: JumpProposal, within_props: List[Proposal]):
        super().__init__()
        self.move_prop = move_prop
        self.within_props = within_props
        self.jump_prop = jump_prop

    def propose(self, state, **kwargs):
        new_k = self.move_prop.move(state)
        k = state.idx
        if new_k == k:
            new_state, logqratio = self.within_props[k-1].propose(state)
        else:
            new_state, logqratio = self.jump_prop.propose(state, new_k)
        return new_state, logqratio + self.move_prop.logp(k, new_k)
