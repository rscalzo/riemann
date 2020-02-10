from .. import riemann


class RJState:
    def __init__(self, param, idx):
        self.param = param
        assert idx > 0
        self.idx = idx


class RJProposal(riemann.Proposal):
    def __init__(self, move_prop, jump_prop, within_prop):
        super().__init__()
        self.move_prop = move_prop
        self.within_prop = within_prop
        self.jump_prop = jump_prop

    def propose(self, state, **kwargs):
        new_k = self.move_prop.move(state)
        k = state.idx
        if new_k == k:
            new_state, logqratio = self.within_prop.propose(state)
        else:
            new_state, logqratio = self.jump_prop.propose(state, new_k)
        return new_state, logqratio + self.move_prop.logp(k, new_k)


class JumpProposal:
    def __init__(self, mapping, matching_prop):
        super().__init__()
        self.mapping = mapping
        self.matching_prop = matching_prop

    def propose(self, state, new_k):
        k = state.idx
        u = self.matching_prop.match(state, new_k)
        if u is not None:  # moving to higher dimension
            new_state = self.mapping.forward(state, u, new_k)
            logqratio = self.mapping.log_det(state, u, new_k) \
                        - self.matching_prop.logp(k, new_k, u)
        else:  # moving to lower dimension
            new_state, u = self.mapping.backward(state, new_k)
            logqratio = self.matching_prop.logp(k, new_k, u) \
                        - self.mapping.log_det(new_state, u, k)
        return new_state, logqratio
