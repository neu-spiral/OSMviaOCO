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
        if args.problemType == 'FL':
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

    directory_output = "results/" + "/" + args.policy + "/" + args.problemType + "/" + args.input.split("/")[-1] + "/k_" \
                       + str(args.constraints) + "_" + str(args.iterations) + "_FW" + "/"

    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
        logging.info('...output directory is created...')
    output = directory_output


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
                results.append((key, track[key][0], track[key][1],
                                newProblem.objective_func(track[key][1]), args.estimator,
                                args.degree, args.center, bases))


        save(output, results)
