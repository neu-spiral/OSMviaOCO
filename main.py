from ContinuousGreedy import multilinear_relaxation
from helpers import save, load
from ProblemInstances import InfluenceMaximization, FacilityLocation, derive
from oco_tools import ThresholdObjective, ZeroOneDecisionSet, RelaxedPartitionMatroid, OCOPolicy
from time import time
import args
import logging
import numpy as np
import os
import pickle
import sys


if __name__ == "__main__":
    
    parser = args.create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.problem is not None:
        newProblem = load(args.problem)
        args.problemType = args.problem.split("_")[0].split("/")[-1]
        args.input = args.problem.split("_")[1] + "_" + args.problem.split("_")[2] + "_" + args.problem.split("_")[3]
        args.constraints = int(args.problem.split("_")[-1])

    else:
        if args.problemType == 'QS':
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

        if args.estimator == 'polynomial':
            output = directory_output + args.estimator + "_" + str(args.degree)
            start = time()
            poly_grad, poly_estimation = newProblem.get_polynomial_estimator(args.center, args.degree).estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated grad is: " + str(poly_grad) + '\n')
            sys.stderr.write("estimated value of the function is: " + str(poly_estimation) + '\n')
            poly_results = [elapsed_time, args.degree, poly_estimation]
            save(output, poly_results)

        else:  # args.estimator == 'sampler':
            output = directory_output + args.estimator + "_" + str(args.samples)
            start = time()
            sampler_grad, sampler_estimation = newProblem.get_sampler_estimator(args.samples).estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated value of the function is: " + str(sampler_estimation) + '\n')
            sampler_results = [elapsed_time, args.samples, sampler_estimation]
            save(output, sampler_results)
            

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
                results.append((key, track[key][0], track[key][1],
                                newProblem.utility_function(track[key][1]), args.estimator,
                                args.samples, bases))

        save(output, results)
