from riemann import Model, Sampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
import numpy as np

def distance(a, b):
    return np.linalg.norm(a-b)

def min1exp(x): # returns min(1, exp(x))
    if x > 0:
        return 1
    else:
        return np.exp(x)

class MHLocalEmulatorSampler(Sampler):
    def __init__(self, model, proposal, theta0, beta = 0.1, gamma = 2, optimise_refinement = False, R2_refinement = False, R2_threshold = 0.7, kfold_refinement = False, kfold_k = 3, refine_exactly = False, approximation_degree = 1):
        """
        Initialize a Sampler with a model, a proposal, data, and a guess
        at some reasonable starting parameters.
        :param model: callable accepting a np.array parameter vector
            of shape matching the initial guess theta0, and returning
            a probability (such as a posterior probability)
        :param proposal: callable accepting a np.array parameter vector
            of shape matching the initial guess theta0, and returning
            a proposal of the same shape, as well as the log ratio
                log (q(theta'|theta)/q(theta|theta'))
        :param theta0: np.array of shape (Npars,)
        """
        self.model = model
        self.proposal = proposal
        self._chain_thetas = [ theta0 ]
        self._chain_logpost = [ model.log_posterior(theta0) ]
        self.sample_points = []
        self.sample_points_output = []
        self.beta = beta # As mentioned in the paper, rate of random refinement
        self.gamma = gamma # As mentioned in the paper, the criteria for error-based refinement
        self.dimension = len(theta0)
        self.optimise_refinement = optimise_refinement # Whether to optimise the refinement process or randomly create a new point
        self.approximation_degree = approximation_degree # Set this to 1 for linear regression, 2 for quadratic regression
        self.R2_refinement = R2_refinement # Whether we use R^2 refinement criteria
        self.R2_threshold = R2_threshold
        self.kfold_refinement = kfold_refinement # Whether we use k-fold refinement criteria
        self.kfold_k = kfold_k

        self.refine_exactly = refine_exactly # Whether we refine a new point at the exact value of theta

        try:
            def grad_function(theta_plus, theta_minus):
                return self.weightedRegressionGradient(theta_plus, theta_minus)
            self.proposal.set_gradlogpost(grad_function)
        except:
            pass #proposal doesn't need a gradient function

        number_starting_sample_points = self.approximationPoints()
        for i in range(int(number_starting_sample_points)):
            self.sample_points.append(np.random.normal(size = self.dimension) + theta0)
            self.sample_points_output.append(self.model.log_posterior(self.sample_points[i]))

        self.neighbors_query = NearestNeighbors(n_neighbors=self.approximationPoints())
        self.neighbors_query.fit(np.array(self.sample_points))


    def minApproximationPoints(self):
        if self.approximation_degree == 1:
            return self.dimension + 1
        else:
            return (self.dimension + 1) * (self.dimension + 2) / 2

    def approximationPoints(self):
        return int(self.minApproximationPoints() * np.ceil(np.sqrt(self.dimension)))

    def weightedRegressionOutput(self, theta_center, leave_out = -1): # WARNING: THIS WILL NOT NOT NOT AUTOMATICALLY CHECK AND REFINE POINTS

        poly = PolynomialFeatures(self.approximation_degree)
        reg = self.weightedLinearRegressionModel(theta_center, leave_out)
        def prediction_func(point):
            polynomial_input = poly.fit_transform(point)
            return reg.predict(polynomial_input)

        return prediction_func

    def weightedRegressionGradient(self, theta_center, theta_minus, leave_out = -1): # WARNING: THIS WILL AUTOMATICALLY CHECK AND REFINE POINTS

        while True:
            if not self.checkRefinementNear(theta_center, theta_minus):
                break

        poly = PolynomialFeatures(self.approximation_degree)
        reg = self.weightedLinearRegressionModel(theta_center, leave_out)

        return reg.coef_[1:self.dimension + 1]

    def weightedLinearRegressionModel(self, theta_center, leave_out = -1, return_R2 = False):

        # distances = [distance(self.sample_points[j], theta_center) for j in range(len(self.sample_points))]
        # point_distance_order = np.argsort(distances).flatten()

        # Faster way of generating point_distance_order using sklearn NearestNeighbors
        point_distance_order = self.neighbors_query.kneighbors(theta_center.reshape(1, -1), return_distance=False)[0]

        weights = np.zeros(len(self.sample_points))
        R_def = distance(self.sample_points[point_distance_order[self.minApproximationPoints() - 1]], theta_center)
        R = distance(self.sample_points[point_distance_order[self.approximationPoints() - 1]], theta_center)

        # Sometimes NearestNeighbors skips an index, so we make up for it here
        missing = -1
        for j in range(self.approximationPoints()):
            if not j in point_distance_order:
                missing = j
                break
        if missing != -1:
            for j in range(self.approximationPoints()):
                if (point_distance_order[j] > missing):
                    point_distance_order[j] -= 1

        for i in range(self.approximationPoints()):
            if i < self.minApproximationPoints():
                weights[point_distance_order[i]] = 1
            else:
                if (R - R_def == 0):
                    pass # we're on the boundary of the region, just set leave weight as 0 for ease of use
                else:
                    weights[point_distance_order[i]] = max(0, (1 - ((distance(self.sample_points[point_distance_order[i]], theta_center) - R_def) / (R - R_def))**3)**3)

        if type(leave_out) == int:
            if leave_out != -1:
                weights[point_distance_order[leave_out]] = 0 # leave this point out by setting its weight to 0
        else:
            for i in leave_out:
                weights[point_distance_order[i]] = 0


        reg = linear_model.LinearRegression(fit_intercept = False)

        X = np.array(self.sample_points)
        poly = PolynomialFeatures(self.approximation_degree)
        polynomial_input = poly.fit_transform(X)

        reg.fit(polynomial_input, np.array(self.sample_points_output), weights)

        if not return_R2:
            return reg
        else:
            R2 = reg.score(polynomial_input, np.array(self.sample_points_output), weights)
            return reg, R2

    def checkRefinementNear(self, theta_plus, theta_minus, refine = True):

        if self.R2_refinement: # Here we check if the R2 from the regression is high enough
            model_plus, R2 = self.weightedLinearRegressionModel(theta_plus, return_R2 = True)
            if R2 < self.R2_threshold:
                if refine:
                    self.refineNear(theta_plus)
                return True

            model_minus, R2 = self.weightedLinearRegressionModel(theta_minus, return_R2 = True)
            if R2 < self.R2_threshold:
                if refine:
                    self.refineNear(theta_minus)
                return True

            return False

        if self.kfold_refinement:

            plus_approximation = self.weightedRegressionOutput(theta_plus)
            minus_approximation = self.weightedRegressionOutput(theta_minus)

            plus_approximations_leave_outs = []
            minus_approximations_leave_outs = []

            k = self.kfold_k

            for i in range(k):
                leave_out_indices = range(i * self.approximationPoints() // k, min(self.approximationPoints(), (i + 1) * self.approximationPoints() // k))
                plus_approximations_leave_outs.append(self.weightedRegressionOutput(theta_plus, leave_out = leave_out_indices))
                minus_approximations_leave_outs.append(self.weightedRegressionOutput(theta_minus, leave_out = leave_out_indices))

            #the following three are min(1, zeta)
            min1_zeta = min1exp(plus_approximation([theta_plus]) - minus_approximation([theta_minus]))
            min1_zeta_j_pluses = [min1exp(plus_approximations_leave_outs[j]([theta_plus]) - minus_approximation([theta_minus])) for j in range(k)]
            min1_zeta_j_minuses = [min1exp(plus_approximation([theta_plus]) - minus_approximations_leave_outs[j]([theta_minus])) for j in range(k)]

            #the following three are min(1, 1 / zeta)
            min1_zeta_inv = min1exp(-(plus_approximation([theta_plus]) - minus_approximation([theta_minus])))
            min1_zeta_j_pluses_inv = [min1exp(-(plus_approximations_leave_outs[j]([theta_plus]) - minus_approximation([theta_minus]))) for j in range(k)]
            min1_zeta_j_minuses_inv = [min1exp(-(plus_approximation([theta_plus]) - minus_approximations_leave_outs[j]([theta_minus]))) for j in range(k)]

            epsilon_plus = 0
            epsilon_minus = 0

            for j in range(k):
                possible_epsilon_plus = np.abs(min1_zeta - min1_zeta_j_pluses[j]) + np.abs(min1_zeta_inv - min1_zeta_j_pluses_inv[j])
                if possible_epsilon_plus > epsilon_plus:
                    epsilon_plus = possible_epsilon_plus
                possible_epsilon_minus = np.abs(min1_zeta - min1_zeta_j_minuses[j]) + np.abs(min1_zeta_inv - min1_zeta_j_minuses_inv[j])
                if possible_epsilon_minus > epsilon_minus:
                    epsilon_minus = possible_epsilon_minus

            if epsilon_plus >= epsilon_minus and epsilon_plus >= self.gamma:
                if refine:
                    self.refineNear(theta_plus)
                return True
            elif epsilon_minus >= epsilon_plus and epsilon_minus >= self.gamma:
                if refine:
                    self.refineNear(theta_minus)
                return True

            return False

        # Otherwise we do leave-one-out cross-validation

        plus_approximation = self.weightedRegressionOutput(theta_plus)
        minus_approximation = self.weightedRegressionOutput(theta_minus)

        plus_approximations_leave_outs = []
        minus_approximations_leave_outs = []
        for j in range(self.approximationPoints()):
            plus_approximations_leave_outs.append(self.weightedRegressionOutput(theta_plus, leave_out = j))
            minus_approximations_leave_outs.append(self.weightedRegressionOutput(theta_minus, leave_out = j))

        #the following three are min(1, zeta)
        min1_zeta = min1exp(plus_approximation([theta_plus]) - minus_approximation([theta_minus]))
        min1_zeta_j_pluses = [min1exp(plus_approximations_leave_outs[j]([theta_plus]) - minus_approximation([theta_minus])) for j in range(self.approximationPoints())]
        min1_zeta_j_minuses = [min1exp(plus_approximation([theta_plus]) - minus_approximations_leave_outs[j]([theta_minus])) for j in range(self.approximationPoints())]

        #the following three are min(1, 1 / zeta)
        min1_zeta_inv = min1exp(-(plus_approximation([theta_plus]) - minus_approximation([theta_minus])))
        min1_zeta_j_pluses_inv = [min1exp(-(plus_approximations_leave_outs[j]([theta_plus]) - minus_approximation([theta_minus]))) for j in range(self.approximationPoints())]
        min1_zeta_j_minuses_inv = [min1exp(-(plus_approximation([theta_plus]) - minus_approximations_leave_outs[j]([theta_minus]))) for j in range(self.approximationPoints())]

        epsilon_plus = 0
        epsilon_minus = 0

        for j in range(self.approximationPoints()):
            possible_epsilon_plus = np.abs(min1_zeta - min1_zeta_j_pluses[j]) + np.abs(min1_zeta_inv - min1_zeta_j_pluses_inv[j])
            if possible_epsilon_plus > epsilon_plus:
                epsilon_plus = possible_epsilon_plus
            possible_epsilon_minus = np.abs(min1_zeta - min1_zeta_j_minuses[j]) + np.abs(min1_zeta_inv - min1_zeta_j_minuses_inv[j])
            if possible_epsilon_minus > epsilon_minus:
                epsilon_minus = possible_epsilon_minus

        if epsilon_plus >= epsilon_minus and epsilon_plus >= self.gamma:
            if refine:
                self.refineNear(theta_plus)
            return True
        elif epsilon_minus >= epsilon_plus and epsilon_minus >= self.gamma:
            if refine:
                self.refineNear(theta_minus)
            return True

    def refineNear(self, theta):

        point_distance_order = self.neighbors_query.kneighbors(theta.reshape(1, -1), return_distance=False)[0]

        R = distance(self.sample_points[point_distance_order[self.approximationPoints() - 1]], theta)

        if not self.optimise_refinement:
            point = np.random.normal(size = self.dimension)
            point = point / np.linalg.norm(point)
            point = point * np.random.uniform() * R
            point = point + theta

            self.sample_points.append(point)
            self.sample_points_output.append(self.model.log_posterior(point))
        else:
            def minimise_objective(p):
                min_distance = distance(p, self.sample_points[point_distance_order[0]])
                for j in range(1, len(point_distance_order)):
                    new_distance = distance(p, self.sample_points[point_distance_order[j]])
                    min_distance = min(min_distance, new_distance)
                    if distance(theta, self.sample_points[point_distance_order[j]]) > 3 * R: # paper says that any points >3R away from theta are irrelevant
                        break
                return -min_distance #returning the negative as we want to maximise this min distance

            def constraint_func(p):
                sum = 0
                for i in range(len(p)):
                    sum += (p[i] - theta[i])**2
                return [sum]

            def constraint_jacob(x):
                jacobian = [0] * len(x)
                for i in range(len(x)):
                    jacobian[i] = 2 * (x[i] - theta[i])
                return [jacobian]

            constraint = NonlinearConstraint(constraint_func, -np.inf, R**2, jac = constraint_jacob)#, hess = constraint_hess)

            tolerance = 1
            result = minimize(minimise_objective, theta, constraints = [constraint])

            self.sample_points.append(result.x)
            self.sample_points_output.append(self.model.log_posterior(result.x))
        self.neighbors_query.fit(np.array(self.sample_points))

    def sampleStandard(self): # The standard way of sampling using the emulator method described in the paper
        theta_old, logpost_old = self.current_state()
        theta_prop, logqratio = self.proposal.propose(theta_old)

        did_refine = True # this keeps the model running until we didn't refine the sample set that pass

        while did_refine == True:
            did_refine = False

            new_approximation = self.weightedRegressionOutput(theta_prop)
            old_approximation = self.weightedRegressionOutput(theta_old)

            mhratio = np.exp(min(0, new_approximation([theta_prop]) - old_approximation([theta_old]) - logqratio))

            if np.random.uniform() < self.beta:
                did_refine = True
                if np.random.uniform() < 0.5:
                    self.refineNear(theta_prop)
                else:
                    self.refineNear(theta_old)
            else:
                did_refine = self.checkRefinementNear(theta_prop, theta_old)


        logpost = new_approximation([theta_prop])

        if np.random.uniform() < mhratio:
            self._add_state(theta_prop, logpost)
            return theta_prop, logpost
        else:
            self._add_state(theta_old, logpost_old)
            return theta_old, logpost_old

    def sampleExactly(self):
        theta_old, logpost_old  = self.current_state()
        theta_prop, logqratio = self.proposal.propose(theta_old)

        should_refine_at_theta = False

        if np.random.uniform() < self.beta:
            if np.random.uniform() < 0.5:
                self.refineNear(theta_prop)
            else:
                self.refineNear(theta_old)

             # make this the second last sample point, so that the possible theta_old is still there for checking
            temp = self.sample_points[-1]
            self.sample_points[-1] = self.sample_points[-2]
            self.sample_points[-2] = temp

            temp = self.sample_points_output[-1]
            self.sample_points_output[-1] = self.sample_points_output[-2]
            self.sample_points_output[-2] = temp

        should_refine_at_theta = self.checkRefinementNear(theta_prop, theta_old, refine = False)

        mhratio = 0
        logpost = 0

        if should_refine_at_theta:

            density_old = None

            if not np.array_equal(self.sample_points[-1], theta_old):
                density_old = self.model(theta_old)
                self.sample_points.append(theta_old)
                self.sample_points_output.append(density_old)
            else:
                density_old = self.sample_points_output[-1]

            logpost = self.model(theta_prop)

            try:
                mhratio = np.exp(min(0, logpost - density_old - logqratio))
            except:
                print(logpost - density_old - logqratio)
            if np.random.uniform() < mhratio:
                self.sample_points.append(theta_prop)
                self.sample_points_output.append(logpost)
                self.neighbors_query.fit(np.array(self.sample_points))
                self._add_state(theta_prop, logpost)
                return theta_prop, logpost
            else:
                self.neighbors_query.fit(np.array(self.sample_points))
                self._add_state(theta_old, logpost_old)
                return theta_old, logpost_old

        else:

            new_approximation = self.weightedRegressionOutput(theta_prop)
            old_approximation = self.weightedRegressionOutput(theta_old)
            logpost = new_approximation([theta_prop])

            try:
                mhratio = np.exp(min(0, new_approximation([theta_prop]) - old_approximation([theta_old]) - logqratio))
            except:
                print(new_approximation([theta_prop]) - old_approximation([theta_old]) - logqratio)


            if np.random.uniform() < mhratio:
                self._add_state(theta_prop, logpost)
                return theta_prop, logpost
            else:
                self._add_state(theta_old, logpost_old)
                return theta_old, logpost_old

    def sample(self):
        if self.refine_exactly:
            return self.sampleExactly()
        else:
            return self.sampleStandard()
