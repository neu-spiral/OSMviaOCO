from helpers import save, load
from ProblemInstances import InfluenceMaximization, FacilityLocation
from oco_tools import ThresholdObjective, ZeroOneDecisionSet, RelaxedPartitionMatroid, OCOPolicy, OGA, \
    ShiftedNegativeEntropyOMD, OptimisticPolicy
from KKL.translate import Translator
from KKL.mapping import WDNFMapping
from KKL.offline_alg import ApproxGreedy
from KKL.online_alg import KKL
from KKL.game import Game

from time import time
import args
import logging
import numpy as np
import math
import os
import pickle
import sys

if __name__ == "__main__":

    parser = args.create_parser()
    args = parser.parse_args()

    eta = args.eta
    logging.basicConfig(level=logging.INFO)

    ### LOAD OR GENERATE THE PROBLEM INSTANCE
    # use here if the problem instance has already been created
    if args.problem is not None:
        newProblem = load(args.problem)
        args.problemType = args.problem.split("_")[0].split("/")[-1]
        args.input = args.problem.split("_")[1] + "_" + args.problem.split("_")[2] + "_" + args.problem.split("_")[3]
        args.constraints = int(args.problem.split("_")[-1])

    # use here to generate a new problem instance given the input and the partitions file
    else:
        if args.problemType == 'FL':
            logging.info('Loading movie ratings...')
            bipartite_graph = load(args.input)  # this needs one single bipartite graph
            bipartite_graph = bipartite_graph[0]
            target_partitions = load(args.partitions)
            k_list = dict.fromkeys(target_partitions.keys(), args.k)
            # k_list['Drama'] = args.constraints
            # k_list['Comedy'] = args.constraints
            logging.info('...done. Defining a FacilityLocation Problem...')
            newProblem = FacilityLocation(bipartite_graph, k_list, target_partitions)
            cardinalities_k = list(k_list.values())
            sets_S = list(target_partitions.values())
            sets_S = [list(sets_S[i]) for i in range(len(sets_S))]
            new_decision_set = RelaxedPartitionMatroid(newProblem.problemSize, cardinalities_k,
                                                       sets_S)  # make sure the k_list and target_partitions
            # formats fit to cardinalities_k, sets_S format
            logging.info('...done. %d seeds will be selected from each partition.' % args.k)

        elif args.problemType == 'IM':
            print("IM problem")
            logging.info('Loading cascades...')
            graphs = load(args.input)  # this needs a list of graphs
            if args.partitions is not None:
                target_partitions = load(args.partitions)
                k_list = dict.fromkeys(target_partitions.keys(), args.k)
            else:
                target_partitions = None
                k_list = args.k
            logging.info('...done. Just loaded %d cascades.' % (len(graphs)))
            logging.info('Defining an InfluenceMaximization problem...')
            newProblem = InfluenceMaximization(graphs, k_list, target_partitions)
            if args.policy != 'KKL':
                cardinalities_k = list(k_list.values())
                sets_S = list(target_partitions.values())
                sets_S = [list(sets_S[i]) for i in range(len(sets_S))]
                new_decision_set = RelaxedPartitionMatroid(newProblem.problemSize, cardinalities_k, sets_S)
            logging.info('...done. %d seeds will be selected from each partition.' % args.k)

        # generate a file for problems if it does not already exists
        problem_dir = "problems/"
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)

        # save the newly generated problem instance
        save(problem_dir + args.problemType + "_" + args.input.split("/")[-1] + "_k_" + str(args.k),
             newProblem)

    ### GENERATE THE ThresholdObjective OBJECT
    new_objective = newProblem.translate()  ## TODO test what new_objective returns
    # print(new_objective[0].params)

    ### CREATE THE OUTPUT DIRECTORY TO SAVE THE RESULTS IF NOT ALREADY EXISTS
    output_dir = "results/" + args.policy + "/" + args.problemType + "/" + args.input.split("/")[-1] + "/k_" \
                 + str(args.k) + "_" + str(args.T) + "_iter" + "/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('...output directory is created...')
    sys.stderr.write('output directory is:' + output_dir)

    frac_output = output_dir + 'fractional'
    int_output = output_dir + 'integral'

    # GENERATE THE OCOPolicy OBJECT
    if args.policy == 'OGD':
        newPolicy = OCOPolicy(new_decision_set, eta, new_objective[0])
        logging.info("An Online Gradient Descent policy is generated.")

    elif args.policy == 'OGA':
        newPolicy = OGA(new_decision_set, eta, new_objective)
        logging.info("An OGA policy is generated.")

    elif args.policy == 'OMD':
        newPolicy = ShiftedNegativeEntropyOMD(new_decision_set, eta, new_objective)
        logging.info("A Shifted Negative Entropy Online Mirror Descent policy is generated.")

    elif args.policy == 'Optimistic':
        newPolicy = OptimisticPolicy(new_decision_set, eta, new_objective)
        logging.info("An Optimistic policy is generated.")

    elif args.policy == 'KKL':
        logging.info("A KKL policy is generated.")
        translator = Translator(newProblem)
        ws = translator.ws
        n = translator.n  # dimension of action s
        m = translator.m  # dimension of Phi(s)
        T = translator.T  # horizon
        sign = translator.sign  # sign used in the wdnf functions
        index_to_set = translator.index_to_set
        ground_set = newProblem.groundSet

        mapping = WDNFMapping(n, index_to_set, sign)
        linear_solver = newProblem.get_solver()
        initial_point = newProblem.get_initial_point()
        approx_alg = ApproxGreedy(linear_solver, mapping, initial_point, n)

        # set constants
        W = max([np.linalg.norm(w) for w in ws])  # ||w|| <= W
        R = np.sqrt(m)  # ||Phi(s)|| <= R
        alpha = math.e
        delta = (alpha + 1) * R ** 2 / T
        eta = (alpha + 1) * R / (W * np.sqrt(T))

        # initialize online algorithm
        alg = KKL(approx_alg, mapping, alpha, delta, eta, R, n)

        # initialize game
        game = Game(alg, mapping, ws, n, T)

    if args.policy in ['OGD', 'OGA', 'OMD', 'Optimistic']:
        ## RUN THE OCOPolicy
        while newPolicy.current_iteration < args.T:
            # TODO design backups if the algorithm is interrupted
            logging.info("Running iteration #{x}...\n".format(x=newPolicy.current_iteration))  ## TODO format string
            newPolicy.step()
        logging.info("The algorithm is finished.")

        # SAVE THE RESULTS OF THE OCOPolicy
        final_frac_rewards = newPolicy.frac_rewards
        final_int_rewards = newPolicy.int_rewards
        print(f"frac rewards: {final_frac_rewards}")
        print(f"int rewards: {final_int_rewards}")

        def get_cum_avg_reward(rewards: np.ndarray) -> np.ndarray:
            return np.cumsum(rewards) / (np.arange(args.T) + 1)

        cum_frac_rewards = get_cum_avg_reward(final_frac_rewards)
        cum_int_rewards = get_cum_avg_reward(final_int_rewards)
        print(f"cumulative averaged fractional  rewards: {cum_frac_rewards}")
        print(f"cumulative averaged integral rewards: {cum_int_rewards}")

        save(frac_output, cum_frac_rewards)
        save(int_output, cum_int_rewards)
        logging.info("The rewards are saved to: " + output_dir + ".")

    if args.policy == 'KKL':
        game.play()
        cum_avg_reward = game.get_cum_avg_reward()
        print(f"cum_avg_reward: {cum_avg_reward}")
        save(frac_output, cum_avg_reward)
        logging.info("The rewards are saved to: " + output_dir + ".")
