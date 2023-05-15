from abc import ABCMeta, abstractmethod
from ContinuousGreedy import UniformMatroidSolver, PartitionMatroidSolver, SamplerEstimator, PolynomialEstimator, \
    ContinuousGreedy, StochasticGradientEstimator, generate_samples
from networkx import Graph, DiGraph
from networkx.algorithms import bipartite
from oco_tools import ThresholdObjective
from time import time
from wdnf import WDNF, Taylor
import argparse
import logging
import math
import networkx as nx
import numpy as np
import sys
import random


def mul_series(degree):
    """
    :param degree:
    :return:
    """
    if degree <= 2:
        return 1
    else:
        degree * mul_series(degree - 2)


def derive(function_type, x, degree):
    """Helper function to create derivatives list of Taylor objects. Given the
    degree and the center of the Taylor expansion with the type of the functions
    returns the value of the function's derivative at the given center point.
    """
    if function_type == np.log1p:
        if degree == 0:
            return np.log1p(x)  # log1p(x) is ln(x+1)
        else:
            return (((-1.0) ** (degree - 1)) * math.factorial(degree - 1)) / ((1.0 + x) ** degree)

    if function_type == np.sqrt:
        if degree == 0:
            return np.sqrt(x)
        else:
            return (((-1.0) ** (degree - 1)) * mul_series(2 * degree - 3) * ((1 + x) ** (0.5 - degree))) / (2 ** degree)

    if function_type == 'log':
        if degree == 0:
            return 0  # np.log(x)
        else:
            return (((-1.0) ** (degree - 1)) * math.factorial(degree - 1)) / (x ** degree)

    if function_type == qs:
        if degree == 0:
            return 0  # qs(x)
        else:
            return math.factorial(degree) / ((1.0 - x) ** (degree + 1))

    if function_type == id:  # idle delete later
        if degree == 0:
            return 0  # x
        elif degree == 1:
            return 1
        else:
            return 0


def find_derivatives(function_type, center, degree):
    """Type is either 'log1p' or 'queue_size', helper function to create the
    derivatives list of Taylor objects.
    """
    derivatives = [derive(function_type, center, i) for i in range(degree + 1)]
    return derivatives


def wdnf_to_threshold(wdnf):
    """
    Given a WDNF object, returns a ThresholdObjective object.
    :param wdnf:
    :return:
    """
    pass


class Problem(object):
    """Abstract class to parent classes of different problem instances.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """
        """
        self.problemSize = 0
        self.groundSet = set()
        self.wdnf_dict = dict()

    def utility_function(self, y):
        pass

    def get_solver(self):
        """
        """
        pass

    def func(self, x):
        """
        """
        pass

    def get_sampler_estimator(self, num_of_samples, dependencies={}):
        """
        """
        # return SamplerEstimator(self.utility_function, num_of_samples, dependencies)
        pass

    def get_polynomial_estimator(self, center, degree):
        """
        """
        pass

    def get_stochastic_sampler_estimator(self, num_of_samples, dependencies={}):
        """
        """
        # return SamplerEstimator(self.utility_function, num_of_samples, dependencies)
        pass

    def get_stochastic_polynomial_estimator(self, center, degree):
        """
        """
        pass

    def get_initial_point(self):
        """
        """
        pass

    def get_continuous_greedy(self, estimator, iterations, need_restart=False, backup_file=None):
        """
        """
        logging.info('Creating the ContinuousGreedy object...')
        new_cg = ContinuousGreedy(self.get_solver(), estimator, self.get_initial_point())
        logging.info('done.')
        return new_cg.fw(iterations, True, need_restart, backup_file)

    def translate(self):
        """ takes self.wdnf_dict and returns the params of a ThresholdObjective
        """
        n = self.problemSize
        # sys.stderr.write('n: ' + str(n) + '\n')
        threshold_objectives = []
        C = range(len(self.wdnf_dict))
        # sign = self.wdnf_dict[0].sign
        F = WDNF(dict(), -1)
        for graph in self.wdnf_dict:
            params = dict()
            params['n'] = n
            wdnf = self.wdnf_dict[graph].coefficients.copy()
            wdnf.pop((), None)
            # sys.stderr.write('wdnfs: ' + str(wdnfs) + '\n')
            C = range(len(wdnf))
            # sys.stderr.write('C: ' + str(C) + '\n')
            params['C'] = C
            # sys.stderr.write('b: ' + str(np.ones(len(C))) + '\n')
            params['b'] = np.ones(len(C))
            # sys.stderr.write('w: ' + str(np.ones(n)) + '\n')
            params['w'] = np.ones((len(C), n))
            # print(list(self.wdnf_dict.keys()))
            # sys.stderr.write('S: ' + str(list(wdnfs.keys())) + '\n')
            params['S'] = [list(key) for key in list(wdnf.keys())]
            # sys.stderr.write('c: ' + str(list(wdnfs.values())) + '\n')
            params['c'] = [-1 * value for value in list(wdnf.values())]
            # extensive tests
            # y = dict.fromkeys(self.groundSet, 0.0)
            # for _ in range(100):
            #     x = dict(zip(y.keys(), np.random.randint(2, size=len(y)).tolist()))
            #     x1 = np.array(list(x.values()))
            #     print(f"x is {x} and wdnf(x) is {self.wdnf_dict[graph](x)} while ThresholdObjective(x) is "
            #           f"{ThresholdObjective(params).eval(x1)}")
            #     assert math.isclose(self.wdnf_dict[graph](x), ThresholdObjective(params).eval(x1), rel_tol=1e-5), \
            #         f"TRANSLATION IS INCORRECT!"
            F += (1.0 / self.instancesSize) * self.wdnf_dict[graph]
            threshold_objectives.append(ThresholdObjective(params))
        F_params = dict()
        # print(f"F is {F.coefficients}")
        F_wdnf = F.coefficients.copy()
        F_wdnf.pop((), None)
        F_params['n'] = n
        C = range(len(F_wdnf))
        F_params['C'] = C
        F_params['b'] = np.ones(len(C))
        F_params['w'] = np.ones((len(C), n))
        F_params['S'] = [list(key) for key in list(F_wdnf.keys())]
        F_params['c'] = [-1 * value for value in list(F_wdnf.values())]
        # print(f"F_params: {F_params}")
        # y = dict.fromkeys(self.groundSet, 0.0)
        # for _ in range(100):
        #     x = dict(zip(y.keys(), np.random.randint(2, size=len(y)).tolist()))
        #     x1 = np.array(list(x.values()))
        #     print(f"x is {x} and wdnf(x) is {F(x)} while ThresholdObjective(x) is "
        #           f"{ThresholdObjective(F_params).eval(x1)}")
        #     assert math.isclose(F(x), ThresholdObjective(F_params).eval(x1), rel_tol=1e-5), \
        #         f"TRANSLATION IS INCORRECT!"
        F = ThresholdObjective(F_params)
        return threshold_objectives, F


class InfluenceMaximization(Problem):
    """
    """

    def __init__(self, graphs, constraints, target_partitions=None):
        """
        graphs is a list of DiGraph objects from networkx. If given, target_partitions is a dictionary with
        {node : type} pairs and converts the problem to an Influence Maximization over partition matroids, constraints
        is an integer denoting the number of seeds if constraints is over uniform matroid or a dictionary with
        {type : int} pairs if over partition matroids.
        """
        super(InfluenceMaximization, self).__init__()
        logging.info('Initializing the ground set...')
        self.groundSet = set(graphs[0].nodes())  # all graphs share the same set of nodes
        logging.info('... done.')
        self.problemSize = graphs[0].number_of_nodes()  # |V|
        self.instancesSize = len(graphs)  # |G|
        logging.info('\nProblem size is %d and instances size is %d.' % (self.problemSize, self.instancesSize))
        self.constraints = constraints  # number of seeds aka k
        self.target_partitions = target_partitions
        logging.info('Initializing the WDNF dictionary and dependencies...')
        wdnf_dict = dict()
        dependencies = dict()
        wdnf_lengths = []
        for i in range(self.instancesSize):
            paths = nx.algorithms.dag.transitive_closure(graphs[i])
            # sys.stderr.write("paths of cascade " + str(i) + " are:" + str(graphs[i].edges()) + '\n')
            wdnf_list = [WDNF({tuple(sorted([node] + list(paths.predecessors(node)))): -1.0 / self.problemSize}, -1)
                         for node in self.groundSet]
            p_v = [len(key) for wdnf in wdnf_list for key in wdnf.coefficients]
            avg_p_v = (sum(p_v) * 1.0) / len(p_v)
            min_p_v = min(p_v)
            max_p_v = max(p_v)
            std_dev_p_v = np.sqrt((sum([(path_len - avg_p_v) ** 2 for path_len in p_v]) * 1.0) / len(p_v))
            logging.info("\nAverage P_v size is %s, maximum P_v size is %d, minimum P_v size is %d, and standard "
                         "deviation of P_v's is %s" % (avg_p_v, max_p_v, min_p_v, std_dev_p_v))
            resulting_wdnf = sum(wdnf_list) + WDNF({(): 1.0}, -1)
            # sys.stderr.write("wdnf is " + str(resulting_wdnf.coefficients) + '\n')
            wdnf_lengths.append(len(resulting_wdnf.coefficients))
            dependencies.update(resulting_wdnf.find_dependencies())
            wdnf_dict[i] = resulting_wdnf  # prod(1 - x_u) for all u in P_v
            logging.info(f"Cascade {str(i)} WDNFs are generated.\n")
        self.avg_wdnf_len = (sum(wdnf_lengths) * 1.0) / len(wdnf_lengths)
        self.max_wdnf_len = max(wdnf_lengths)
        logging.info('\nAverage WDNF size is %d and maximum WDNF size is %d.' % (self.avg_wdnf_len, self.max_wdnf_len))
        self.wdnf_dict = wdnf_dict
        self.utility_function = np.log1p
        self.dependencies = dependencies
        # sys.stderr.write("wdnf_dict is " + str(wdnf_dict) + '\n')
        logging.info('... done. An instance of a influence maximization problem has been created.')

    def get_solver(self):
        """
        """
        logging.info('Getting solver...')
        if self.target_partitions is None:
            solver = UniformMatroidSolver(self.groundSet, self.constraints)
        else:
            solver = PartitionMatroidSolver(self.target_partitions, self.constraints)
        logging.info('...done.')
        return solver

    def objective_func(self, x):
        """
        :param x:
        :return:
        """
        objective = [self.utility_function(self.wdnf_dict[g](x)) * (1.0 / self.instancesSize) for g in
                     self.wdnf_dict]
        return sum(objective)

    def get_sampler_estimator(self, num_of_samples, dependencies={}):
        """
        """
        return SamplerEstimator(self.objective_func, num_of_samples, dependencies)

    def get_polynomial_estimator(self, center, degree):
        """
        """
        logging.info('Getting polynomial estimator...')
        derivatives = find_derivatives(self.utility_function, center, degree)  # log
        my_taylor = Taylor(degree, derivatives, center)
        final_wdnf = [(1.0 / self.instancesSize) * (my_taylor.compose(self.wdnf_dict[graph]))
                      for graph in range(self.instancesSize)]
        final_wdnf = sum(final_wdnf)
        logging.info('...done.')
        logging.info('\nNumber of terms appear in the resulting WDNF is %d.' % len(final_wdnf.coefficients))
        return PolynomialEstimator(final_wdnf)

    def get_stochastic_sampler_estimator(self, num_of_samples, dependencies={}):
        """
        :param num_of_samples:
        :param dependencies:
        :return:
        """
        logging.info('Getting sampler estimators...')
        sampler_estimator_dict = dict()
        for graph in self.wdnf_dict:  # change to inline (maybe)
            new_objective_func = lambda x: self.utility_function(self.wdnf_dict[graph](x))  # this can be new_objective
            sampler_estimator_dict[graph] = SamplerEstimator(new_objective_func, num_of_samples, dependencies)
        # sampler_estimator_dict = {graph: SamplerEstimator(self.wdnf_dict[graph], self.utility_function,
        # num_of_samples, dependencies) for graph in self.wdnf_dict}
        logging.info('...done.')
        return StochasticGradientEstimator(sampler_estimator_dict)

    def get_stochastic_polynomial_estimator(self, center, degree):
        """
        """
        logging.info('Getting polynomial estimator...')
        derivatives = find_derivatives(self.utility_function, center, degree)  # log
        my_taylor = Taylor(degree, derivatives, center)
        polynomial_estimator_dict = {graph: PolynomialEstimator(my_taylor.compose(self.wdnf_dict[graph])) for graph in
                                     self.wdnf_dict}

        logging.info('...done.')
        return StochasticGradientEstimator(polynomial_estimator_dict)

    def get_initial_point(self):
        """
        """
        return dict.fromkeys(self.groundSet, 0.0)


class FacilityLocation(Problem):
    """
    """

    def __init__(self, bipartite_graph, constraints, target_partitions=None):
        """
        bipartite_graph is a weighted bipartite graph, constraints is an integer which denotes the maximum
        number of facilities/movies to be chosen.
        """
        super(FacilityLocation, self).__init__()
        X = {n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 0}  # facilities, movies
        self.X = set(map(int, X))  # facilities, movies
        print(f"movies are: {X}")
        self.Y = set(bipartite_graph) - X  # customers, users
        print(f"users are: {self.Y}")
        self.constraints = constraints
        print(f"constraints are: {constraints}")
        # self.partitioned_set = dict.fromkeys(self.Y, self.X)  # ???
        self.target_partitions = target_partitions
        print(f"target partitions are: {target_partitions}")
        self.problemSize = len(self.X)  # number of facilities, movies
        print(f"# movies is: {self.problemSize}")
        self.instancesSize = len(self.Y)  # number of customers, users
        print(f"# users is: {self.instancesSize}")
        wdnf_dict = dict()
        dependencies = dict()
        wdnf_lengths = []
        for y in self.Y:
            weights = dict()
            # weights = {facility: bipartite_graph.get_edge_data(facility, y)['weight'] for facility in self.X}
            for facility in self.X:
                weights[facility] = bipartite_graph.get_edge_data(facility, y)['weight'] \
                    if bipartite_graph.has_edge(facility, y) else 0.0
            weights[len(list(self.X)) + 1] = 0.0
            print(f"ratings of user {y} are: {weights}")
            descending_weights = sorted(weights.values(), reverse=True)
            print(f"sorted ratings of user {y} are: {descending_weights}")
            indices = sorted(range(len(weights.values())), key=lambda k: list(weights.values())[k], reverse=True)
            wdnf_so_far = WDNF(dict(), -1)
            for i in range(len(list(self.X))):
                index = tuple(int(list(weights.keys())[index]) for index in indices[:(i + 1)])
                wdnf_so_far += (descending_weights[i] - descending_weights[i + 1]) * \
                               (WDNF({(): 1.0}, -1) + (-1.0) * WDNF({index: 1.0}, -1))
                if descending_weights[i + 1] == 0:
                    break
            wdnf_lengths.append(len(wdnf_so_far.coefficients))
            print(str(y) + ": " + str(wdnf_so_far.coefficients))
            wdnf_dict[y] = wdnf_so_far
            dependencies.update(wdnf_so_far.find_dependencies())
        self.avg_wdnf_len = (sum(wdnf_lengths) * 1.0) / len(wdnf_lengths)
        self.max_wdnf_len = max(wdnf_lengths)
        logging.info('\nAverage WDNF size is %d and maximum WDNF size is %d.' % (self.avg_wdnf_len, self.max_wdnf_len))
        self.wdnf_dict = wdnf_dict
        self.dependencies = dependencies

    def utility_function(self, y):
        objective = [np.log1p(self.wdnf_dict[customer](y)) * (1.0 / self.size) for customer in self.wdnf_dict]
        return sum(objective)

    def get_solver(self):
        """
        """
        logging.info('Getting solver...')
        if self.target_partitions is None:
            solver = UniformMatroidSolver(self.X, self.constraints)
        else:
            solver = PartitionMatroidSolver(self.target_partitions, self.constraints)
        logging.info('...done.')
        return solver

    def get_polynomial_estimator(self, center, degree):
        """
        """
        logging.info('Getting polynomial estimator...')
        # derivatives = find_derivatives(np.log1p, center, degree) # log
        derivatives = find_derivatives(np.sqrt, center, degree)  # sqrt
        my_taylor = Taylor(degree, derivatives, center)
        final_wdnf = [(1.0 / self.size) * (my_taylor.compose(self.wdnf_dict[customer]))
                      for customer in self.wdnf_dict]
        final_wdnf = sum(final_wdnf)
        logging.info('...done.')
        logging.info('\nNumber of terms appear in the resulting WDNF is %d.' % len(final_wdnf.coefficients))
        return PolynomialEstimator(final_wdnf)

    def get_initial_point(self):
        """
        """
        return dict.fromkeys(self.X, 0.0)  # MAP TO INTEGER!

class TeamFormation(Problem):
    '''
        Quadratic set functions (QSF): f(x) = h^T x + 1/2 x^T H x
    '''
    def __init__(self, functions, constraints, target_partitions=None):
        self.problemSize = len(functions[0][0]) # n = number of individuals = len(h)
        self.instancesSize = len(functions)
        self.constraints = constraints
        self.target_partitions = target_partitions
        self.groundSet = set(range(self.problemSize))
        self.wdnfs = [self.convert_to_WDNF(f) for f in functions]
        self.wdnf_dict = {}
        for t in range(len(self.wdnfs)):
            self.wdnf_dict[t] = self.wdnfs[t]
        
        self.thresholds = [self.convert_to_Thresholds(f) for f in functions]
        self.frac_opt = 400 if len(target_partitions) > 1 else 200 # optimal team of 4 (2) achieves reward 400 (200) per iteration
        
    def convert_to_WDNF(self, f):
        h, H = f
        n = self.problemSize
        coefficients = {}
        # linear terms
        for i in range(n):
            coefficients[(i,)] = h[i]
            
        # quadratic terms H_{i,j} x_i*x_j
        for i in range(n-1):
            for j in range(i+1, n):
                coefficients[(i, j)] = H[i][j]
        
        return WDNF(coefficients, sign=1)

    def convert_to_Thresholds(self, f):
        h, H = f
        n = self.problemSize
        C = int(1 + n * (n-1) / 2)
        w = [np.array(h) + np.dot(H, np.ones(n))] + [[1] * n] * (C-1)
        b = [1e+20] + [1] * (C-1)
        S = [list(range(n))] + [[i,j] for j in range(n) for i in range(n) if i < j]
        c = [1] + [-H[i][j] for j in range(n) for i in range(n) if i < j]
        params = {}
        params['n'] = n
        params['C'] = list(range(C))
        params['w'] = w
        params['b'] = b
        params['S'] = S
        params['c'] = c

        threshold_obj = ThresholdObjective(params)
        assert threshold_obj.eval(np.zeros(n)) == 0, "translation not correct"
        return threshold_obj

if __name__ == "__main__":
    B = 5
