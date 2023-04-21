from abc import ABCMeta, abstractmethod
from ContinuousGreedy import UniformMatroidSolver, PartitionMatroidSolver, SamplerEstimator, PolynomialEstimator, \
    ContinuousGreedy, StochasticGradientEstimator
from networkx import Graph, DiGraph
from networkx.algorithms import bipartite
from time import time
from wdnf import WDNF, Taylor
import argparse
import logging
import math
import networkx as nx
import numpy as np
import sys
import random


def qs(x):
    """Given rho returns rho / (1 - rho)
    """
    return x / (1.0 - x)


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
    
    def translate()


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
            sys.stderr.write("paths of cascade " + str(i) + " are:" + str(graphs[i].edges()) + '\n')
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
            sys.stderr.write("wdnf is " + str(resulting_wdnf.coefficients) + '\n')
            wdnf_lengths.append(len(resulting_wdnf.coefficients))
            dependencies.update(resulting_wdnf.find_dependencies())
            wdnf_dict[i] = resulting_wdnf  # prod(1 - x_u) for all u in P_v
        self.avg_wdnf_len = (sum(wdnf_lengths) * 1.0) / len(wdnf_lengths)
        self.max_wdnf_len = max(wdnf_lengths)
        logging.info('\nAverage WDNF size is %d and maximum WDNF size is %d.' % (self.avg_wdnf_len, self.max_wdnf_len))
        self.wdnf_dict = wdnf_dict
        self.utility_function = np.log1p
        self.dependencies = dependencies
        sys.stderr.write("dependencies are " + str(dependencies) + '\n')
        logging.info('... done. An instance of a influence maximization problem has been created.')
        
    def translate():
        
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
        for graph in self.wdnf_dict:  #change to inline (maybe)
            new_objective_func = lambda x: self.utility_function(self.wdnf_dict[graph](x)) #this can be new_objective
            sampler_estimator_dict[graph] = SamplerEstimator(new_objective_func, num_of_samples, dependencies)
        # sampler_estimator_dict = {graph: SamplerEstimator(self.wdnf_dict[graph], self.utility_function, num_of_samples,
        #                                                   dependencies) for graph in self.wdnf_dict}
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
        self.X = map(int, X)  # facilities, movies
        self.Y = set(bipartite_graph) - X  # customers, users
        self.constraints = constraints
        # self.partitioned_set = dict.fromkeys(self.Y, self.X)  # ???
        self.target_partitions = target_partitions
        self.size = len(self.Y)  # number of customers, users
        wdnf_dict = dict()
        dependencies = dict()
        wdnf_lengths = []
        for y in self.Y:
            weights = dict()
            # weights = {facility: bipartite_graph.get_edge_data(facility, y)['weight'] for facility in self.X}
            for facility in self.X:
                try:
                    weights[facility] = bipartite_graph.get_edge_data(str(facility), y)['weight']
                except TypeError:
                    weights[facility] = 0.0
            weights[len(self.X) + 1] = 0.0
            descending_weights = sorted(weights.values(), reverse=True)
            indices = sorted(range(len(weights.values())), key=lambda k: weights.values()[k], reverse=True)
            wdnf_so_far = WDNF(dict(), -1)
            for i in range(len(self.X)):
                index = tuple(int(weights.keys()[index]) for index in indices[:(i + 1)])
                wdnf_so_far += (descending_weights[i] - descending_weights[i + 1]) * \
                               (WDNF({(): 1.0}, -1) + (-1.0) * WDNF({index: 1.0}, -1))
                if descending_weights[i + 1] == 0:
                    break
            wdnf_lengths.append(len(wdnf_so_far.coefficients))
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


if __name__ == "__main__":
        B = 5

