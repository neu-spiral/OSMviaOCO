from ContinuousGreedy import multilinear_relaxation
from helpers import save, load
from ProblemInstances import InfluenceMaximization, FacilityLocation, derive
from time import time
import argparse
import logging
import numpy as np
import os
import pickle
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Module for ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problem', type=str, help='If the problem instance is created before, provide it here to save'
                                                    ' time instead of recreating it.')
    parser.add_argument('--problemType', default='DR', type=str, help='Type of the problem instance',
                        choices=['DR', 'QS', 'FL', 'IM'])
    parser.add_argument('--input', default='datasets/epinions_20', type=str,
                        help='Data input for the InfluenceMaximization problem')
    parser.add_argument('--partitions', default=None,
                        help='Partitions input for the InfluenceMaximization problem')
    parser.add_argument('--testMode', default=False, type=bool, help='Tests the quality of the estimations from '
                        'different aspects')
    parser.add_argument('--rewardsInput', default="datasets/DR_rewards0", help='Input file that stores rewards')
    parser.add_argument('--partitionsInput', default="datasets/DR_givenPartitions0", help='Input file that stores partitions')
    parser.add_argument('--typesInput', default="datasets/DR_types0",
                        help='Input file that stores targeted partitions of the ground set')
    parser.add_argument('--constraints', default=3, type=int,
                        help='Number of constraints for each partition')
    parser.add_argument('--estimator', default='sampler', type=str, help='Type of the estimator',
                        choices=['polynomial', 'sampler', 'samplerWithDependencies'])
    parser.add_argument('--stochasticORdeterministic', default='stochastic', type=str, help='Type of the function',
                        choices=['stochastic', 'deterministic'])
    parser.add_argument('--iterations', default=100, type=int,
                        help='Number of iterations used in the Frank-Wolfe algorithm')
    parser.add_argument('--degree', default=1, type=int, help='Degree of the polynomial estimator')
    parser.add_argument('--center', default=0.5, type=float,
                        help='The point around which Taylor approximation is calculated')
    parser.add_argument('--samples', default=500, type=int,
                        help='Number of samples used to calculate the sampler estimator')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.problem is not None:
        newProblem = load(args.problem)
        args.problemType = args.problem.split("_")[0].split("/")[-1]
        args.input = args.problem.split("_")[1] + "_" + args.problem.split("_")[2] + "_" + args.problem.split("_")[3]
        args.constraints = int(args.problem.split("_")[-1])

    else:
        if args.problemType == 'DR':
            rewards = load(args.rewardsInput)
            givenPartitions = load(args.partitionsInput)
            types = load(args.typesInput)
            k_list = load(args.constraints)
            newProblem = DiversityReward(rewards, givenPartitions, types, k_list)

        elif args.problemType == 'QS':
            pass

        elif args.problemType == 'FL':
            logging.info('Loading movie ratings...')
            bipartite_graph = load(args.input)
            target_partitions = load(args.partitions)
            k_list = dict.fromkeys(target_partitions.keys(), 0)
            k_list['Drama'] = args.constraints
            k_list['Comedy'] = args.constraints
            logging.info('...done. Defining a FacilityLocation Problem...')
            newProblem = FacilityLocation(bipartite_graph, k_list, target_partitions)
            logging.info('...done. %d seeds will be selected from each partition.' % args.constraints)

        elif args.problemType == 'IM':
            logging.info('Loading cascades...')
            graphs = load(args.input)
            if args.partitions is not None:
                target_partitions = load(args.partitions)
                k_list = dict.fromkeys(target_partitions.keys(), args.constraints)
            else:
                target_partitions = None
                k_list = args.constraints
            logging.info('...done. Just loaded %d cascades.' % (len(graphs)))
            logging.info('Defining an InfluenceMaximization problem...')
            newProblem = InfluenceMaximization(graphs, k_list, target_partitions)
            logging.info('...done. %d seeds will be selected from each partition.' % args.constraints)

        elif args.problemType == 'ALTIM':
            logging.info('Loading cascades...')
            graphs = load(args.input)
            if args.partitions is not None:
                target_partitions = load(args.partitions)
                k_list = dict.fromkeys(target_partitions.keys(), args.constraints)
            else:
                target_partitions = None
                k_list = args.constraints
            logging.info('...done. Just loaded %d cascades.' % (len(graphs)))
            logging.info('Defining an AltInfluenceMaximization problem...')
            newProblem = AltInfluenceMaximization(graphs, k_list, target_partitions)
            logging.info('...done. %d seeds will be selected from each partition.' % args.constraints)

        problem_dir = "problems/"
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)
        save(problem_dir + args.problemType + "_" + args.input.split("/")[-1] + "_k_" + str(args.constraints),
             newProblem)

    directory_output = "results/continuous_greedy/" + args.problemType + "/" + args.input.split("/")[-1] + "/k_" \
                       + str(args.constraints) + "_" + str(args.iterations) + "_FW" + "/"

    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
    logging.info('...output directory is created...')
    output = directory_output + args.stochasticORdeterministic + args.estimator

    if args.testMode is True:
        sys.stderr.write("\nConvergence checker is activated.")
        fixed_point = 0.5
        directory_output = "results/convergence_test/" + args.problem + "/y" + str(fixed_point).strip('.') + "/"
        if not os.path.exists(directory_output):
            os.makedirs(directory_output)
        y = dict.fromkeys(newProblem.groundSet, fixed_point)
        # if os.path.exists("random_y"):
        #     y = load("random_y")
        # else:
        #     y = dict(zip(newProblem.groundSet, np.random.rand(newProblem.problemSize).tolist()))
        #     print(y)
        #     save("random_y", y)
        # out = multilinear_relaxation(newProblem.utility_function, y)
        # sys.stderr.write("multilinear relaxation is: " + str() + '\n')
        if args.estimator == 'polynomial':
            output = directory_output + args.estimator + "_" + str(args.degree)
            start = time()
            poly_grad, poly_estimation = newProblem.get_polynomial_estimator(args.center, args.degree).estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated grad is: " + str(poly_grad) + '\n')
            sys.stderr.write("estimated value of the function is: " + str(poly_estimation) + '\n')
            # if os.path.exists(output):
            #     poly_results = load(output)
            #     poly_results.append((elapsed_time, args.degree, poly_estimation, out))
            # else:
            #     poly_results = [(elapsed_time, args.degree, poly_estimation, out)]
            poly_results = [elapsed_time, args.degree, poly_estimation]
            save(output, poly_results)

        else:  # args.estimator == 'sampler':
            output = directory_output + args.estimator + "_" + str(args.samples)
            start = time()
            sampler_grad, sampler_estimation = newProblem.get_sampler_estimator(args.samples).estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated value of the function is: " + str(sampler_estimation) + '\n')
            # if os.path.exists(output):
            #     sampler_results = load(output)
            #     sampler_results.append((elapsed_time, args.samples, sampler_estimation, out))
            # else:
            #     sampler_results = [(elapsed_time, args.samples, sampler_estimation, out)]
            sampler_results = [elapsed_time, args.samples, sampler_estimation]
            save(output, sampler_results)

        # if args.estimator == 'samplerWithDependencies':
        #     sampler_output = directory_output + args.estimator + '_1_graph_y_rand2' + '_samp_with_dep_estimation'
        #     start = time()
        #     sampler_grad, sampler_estimation = newProblem.get_sampler_estimator(args.samples, newProblem.dependencies)\
        #                                                  .estimate(y)
        #     elapsed_time = time() - start
        #     sys.stderr.write("estimated value of the function is: " + str(sampler_estimation) + '\n')
        #     if os.path.exists(sampler_output):
        #         sampler_results = load(sampler_output)
        #         sampler_results.append((elapsed_time, args.samples, sampler_estimation, out))
        #     else:
        #         sampler_results = [(elapsed_time, args.samples, sampler_estimation, out)]
        #     save(sampler_output, sampler_results)

    else:
        if args.estimator == 'polynomial' and args.stochasticORdeterministic == 'deterministic':
            logging.info('Initiating the Continuous Greedy algorithm using Polynomial Estimator...')
            sys.stderr.write('output directory is:' + output)
            output += "_degree_" + str(args.degree) + "_around_" + str(args.center).replace(".", "")
            output_backup = output + '_backup'
            estimator = newProblem.get_polynomial_estimator(args.center, args.degree)
            y, track, bases = newProblem.get_continuous_greedy(estimator, int(args.iterations),
                                                               backup_file=output_backup)
            print(output)
            results = []
            for key in track:
                # results.append((key, track[key][0], track[key][1],
                #                 multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
                #                 args.degree, args.center))
                results.append((key, track[key][0], track[key][1],
                                newProblem.objective_func(track[key][1]), args.estimator,
                                args.degree, args.center, bases))
            # results = [track, newProblem.utility_function, args.estimator, args.degree, args.center]

        if args.estimator == 'polynomial' and args.stochasticORdeterministic == 'stochastic':
            logging.info('Initiating the Continuous Greedy algorithm using Polynomial Estimator...')
            sys.stderr.write('output directory is:' + output)
            output += "_degree_" + str(args.degree) + "_around_" + str(args.center).replace(".", "")
            output_backup = output + '_backup'
            estimator = newProblem.get_stochastic_polynomial_estimator(args.center, args.degree)
            y, track, bases = newProblem.get_continuous_greedy(estimator, int(args.iterations),
                                                               backup_file=output_backup)
            print(output)
            results = []
            for key in track:
                # results.append((key, track[key][0], track[key][1],
                #                 multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
                #                 args.degree, args.center))
                results.append((key, track[key][0], track[key][1],
                                newProblem.objective_func(track[key][1]), args.estimator,
                                args.degree, args.center, bases))
            # results = [track, newProblem.utility_function, args.estimator, args.degree, args.center]

        if args.estimator == 'sampler' and args.stochasticORdeterministic == 'deterministic':
            logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator...')
            output += "_" + str(args.samples) + "_samples"
            output_backup = output + '_backup'
            estimator = newProblem.get_sampler_estimator(args.samples)
            y, track, bases = newProblem.get_continuous_greedy(estimator, args.iterations,
                                                                   backup_file=output_backup)
            print(output)
            results = []
            for key in track:
                results.append((key, track[key][0], track[key][1],
                                newProblem.objective_func(track[key][1]), args.estimator,
                                args.samples, bases))

        if args.estimator == 'sampler' and args.stochasticORdeterministic == 'stochastic':
            logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator...')
            output += "_" + str(args.samples) + "_samples"
            output_backup = output + '_backup'
            estimator = newProblem.get_stochastic_sampler_estimator(args.samples)
            y, track, bases = newProblem.get_continuous_greedy(estimator, args.iterations,
                                                                   backup_file=output_backup)
            print(output)
            results = []
            for key in track:
                results.append((key, track[key][0], track[key][1],
                                newProblem.objective_func(track[key][1]), args.estimator,
                                args.samples, bases))


        if args.estimator == 'samplerWithDependencies':
            logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator with Dependencies...')
            output += "_" + str(args.samples) + "_samples"
            output_backup = output + '_backup'
            y, track, bases = newProblem.sampler_continuous_greedy(args.samples, args.iterations,
                                                                   dependencies=newProblem.dependencies,
                                                                   backup_file=output_backup)
            # sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')

            print(output)
            results = []
            for key in track:
                # results.append((key, track[key][0], track[key][1],
                #                 multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
                #                 args.samples))
                results.append((key, track[key][0], track[key][1],
                                newProblem.utility_function(track[key][1]), args.estimator,
                                args.samples, bases))

        save(output, results)
