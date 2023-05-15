from helpers import save, load
from ProblemInstances import InfluenceMaximization, FacilityLocation, TeamFormation
from oco_tools import ThresholdObjective, ZeroOneDecisionSet, RelaxedPartitionMatroid, OGA, OCOPolicy, \
    ShiftedNegativeEntropyOMD, MetaPolicy, OptimisticPolicy, FixedShare, FSF, OnlineTBG
from KKL.translate import Translator
from KKL.mapping import WDNFMapping, IdentityMapping
from KKL.offline_alg import ApproxGreedy, ApproxAlgInterval
from KKL.online_alg import KKL
from KKL.game import Game

from time import time
import args
import logging
import numpy as np
import math
import os
import pickle
import random
import sys

import networkx as nx

if __name__ == "__main__":

    parser = args.create_parser()
    args = parser.parse_args()

    eta = args.eta
    gamma = args.gamma
    n_colors = args.n_colors
    seed = args.seed
    T = args.T
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
            if args.partitions is not None:
                target_partitions = load(args.partitions)
                # print(f"target partitions are: {target_partitions}")
                k_list = dict.fromkeys(target_partitions.keys(), args.k)
                constraints = 'partition_matroid'
            else:
                target_partitions = None
                k_list = [args.k]
                constraints = 'cardinality'
            logging.info('...done. Defining a FacilityLocation Problem...')
            newProblem = FacilityLocation(bipartite_graph, k_list, target_partitions)
            cardinalities_k = list(k_list.values())
            print(f"constraints are: {cardinalities_k}")
            sets_S = list(target_partitions.values())
            sets_S = [list(sets_S[i]) for i in range(len(sets_S))]
            print(f"sets are: {sets_S}")
            new_decision_set = RelaxedPartitionMatroid(newProblem.problemSize, cardinalities_k, sets_S)
            # make sure the k_list and target_partitions
            # formats fit to cardinalities_k, sets_S format
            logging.info('...done. %d seeds will be selected from each partition.' % args.k)

        elif args.problemType == 'IM':
            print("IM problem")
            logging.info('Loading cascades...')
            graphs = load(args.input)  # this needs a list of graphs

            assert len(graphs) == T, "Number of f_t's do not match T!"

            if args.partitions is not None:
                target_partitions = load(args.partitions)
                # print(target_partitions)
                k_list = dict.fromkeys(target_partitions.keys(), args.k)
                constraints = 'partition_matroid'
            else:
                target_partitions = None
                k_list = [args.k]
                constraints = 'cardinality'
            logging.info('...done. Just loaded %d cascades.' % (len(graphs)))
            logging.info('Defining an InfluenceMaximization problem...')
            newProblem = InfluenceMaximization(graphs, k_list, target_partitions)
            cardinalities_k = list(k_list.values())
            sets_S = list(target_partitions.values())
            sets_S = [list(sets_S[i]) for i in range(len(sets_S))]
            new_decision_set = RelaxedPartitionMatroid(newProblem.problemSize, cardinalities_k, sets_S)
            logging.info('...done. %d seeds will be selected from each partition.' % args.k)

        elif args.problemType == 'TF':
            logging.info('Loading team formation dataset...')
            functions = load(args.input)
            print(f'h_0 = {functions[0][0]}')
            target_partitions = load(args.partitions)
            print(f"target partitions are: {target_partitions}")
            k_list = dict.fromkeys(target_partitions.keys(), args.k)
            logging.info('...done. Defining a TeamFormation Problem...')
            newProblem = TeamFormation(functions, k_list, target_partitions)
            cardinalities_k = list(k_list.values())
            print(f"constraints are: {cardinalities_k}")
            sets_S = list(target_partitions.values())
            sets_S = [list(sets_S[i]) for i in range(len(sets_S))]
            print(f"sets are: {sets_S}")
            new_decision_set = RelaxedPartitionMatroid(newProblem.problemSize, cardinalities_k, sets_S)

        # generate a file for problems if it does not already exist
        problem_dir = "problems/"
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)

        # save the newly generated problem instance
        save(problem_dir + args.problemType + "_" + args.input.split("/")[-1] + "_k_" + str(args.k),
             newProblem)

    ### GENERATE THE ThresholdObjective OBJECTS
    if args.problemType != 'TF':
        new_objectives, F = newProblem.translate()  # it should return a list of ThresholdObjective objects
    if args.problemType == 'TF':
        new_objectives = newProblem.thresholds
    
    num_objectives = len(new_objectives)
    logging.info("ThresholdObjectives are generated.")
    # print(f"Threshold Objective: {new_objective.params}")


    # if args.traceType == 'sequential':
    #     if num_objectives < T:
    #         trace = list(range(num_objectives)) * math.ceil((1.0*T) / num_objectives)
    #     else:
    #         trace = list(range(num_objectives))
    #     trace = trace[:T]
    # elif args.traceType == 'random':
    #     trace = random.sample(range(T), T)
    # elif args.traceType == 'custom':
    #     trace = load(args.trace)
    # print(f"trace is: {trace}")

    # GENERATE THE OCOPolicy OBJECT
    if args.policy == 'OGA':
        newPolicy = OGA(new_decision_set, new_objectives[0], eta)
        output_dir = f"results/{args.policy}/{args.problemType}/{args.input.split('/')[-1]}/{constraints}/" \
                     f"k_{args.k}_{args.T}_iter/eta_{str(eta).replace('.', 'p')}/"
        logging.info("An OGA policy is generated.")

    elif args.policy == 'OMD':
        newPolicy = ShiftedNegativeEntropyOMD(new_decision_set, new_objectives[0], eta, gamma)
        output_dir = f"results/{args.policy}/{args.problemType}/{args.input.split('/')[-1]}/{constraints}/" \
                     f"k_{args.k}_{args.T}_iter/eta_{str(eta).replace('.', 'p')}_gamma_{str(gamma).replace('.', 'p')}/"
        logging.info("A Shifted Negative Entropy Online Mirror Descent policy is generated.")

    elif args.policy == 'Meta':
        newPolicy = MetaPolicy(new_decision_set, new_objectives[0], eta)
        output_dir = f"results/{args.policy}/{args.problemType}/{args.input.split('/')[-1]}/{constraints}/" \
                     f"k_{args.k}_{args.T}_iter/eta_{str(eta).replace('.', 'p')}/"
        logging.info("A Meta policy is generated.")

    elif args.policy == 'Optimistic':
        newPolicy = OptimisticPolicy(new_decision_set, new_objectives[0], eta, type(OCOPolicy))
        output_dir = f"results/{args.policy}/{args.problemType}/{args.input.split('/')[-1]}/{constraints}/" \
                     f"k_{args.k}_{args.T}_iter/eta_{str(eta).replace('.', 'p')}/"
        logging.info("An Optimistic policy is generated.")

    elif args.policy == 'Fixed':
        newPolicy = FixedShare(new_decision_set, new_objectives[0], eta, type(OCOPolicy))
        output_dir = f"results/{args.policy}/{args.problemType}/{args.input.split('/')[-1]}/{constraints}/" \
                     f"k_{args.k}_{args.T}_iter/eta_{str(eta).replace('.', 'p')}/"
        logging.info("A fixed share policy is generated.")

    elif args.policy == 'FSF':
        newPolicy = FSF(new_decision_set, new_objectives[0], eta, gamma)
        output_dir = f"results/{args.policy}/{args.problemType}/{args.input.split('/')[-1]}/{constraints}/" \
                     f"k_{args.k}_{args.T}_iter/eta_{str(eta).replace('.', 'p')}_gamma_{str(gamma).replace('.', 'p')}/"
        logging.info("An FSF policy is generated.")

    elif args.policy == 'OnlineTBG':
        n = new_objectives[0].params['n']
        newPolicy = OnlineTBG(new_decision_set, new_objectives[0], n=n, eta=eta, n_colors=n_colors)
        output_dir = f"results/{args.policy}/{args.problemType}/{args.input.split('/')[-1]}/{constraints}/" \
                     f"k_{args.k}_{args.T}_iter/eta_{str(eta).replace('.', 'p')}_n_colors_{n_colors}/"

        logging.info("An online TBG policy is generated.")
    
    elif args.policy == 'Random':
        newPolicy = OCOPolicy(new_decision_set, new_objectives[0], eta=eta)
        logging.info("A Random policy (OCOPolicy) is generated.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('...output directory is created...')
    sys.stderr.write('output directory is:' + output_dir)

    output = output_dir + f"seed_{seed}"


    ## RUN THE OCOPolicy
    np.random.seed(seed)
    running_time = []
    start = time()
    while newPolicy.current_iteration < args.T:
        # TODO design backups if the algorithm is interrupted
        i = newPolicy.current_iteration
        logging.info(f"Running iteration #{i}...\n")  ## TODO format string
        newPolicy.objective = new_objectives[i]
        newPolicy.step()
        running_time.append(time() - start)
    logging.info("The algorithm is finished.")


    newPolicy.objective = F  # new policy
    for _ in range(100):
        newPolicy.step()


    opt_frac_reward = newPolicy.frac_rewards.pop()
    opt_int_reward = newPolicy.int_rewards.pop()

    # SAVE THE RESULTS OF THE OCOPolicy
    final_frac_rewards = newPolicy.frac_rewards[:T-1]
    final_int_rewards = newPolicy.int_rewards[:T-1]
    print(f"frac rewards: {final_frac_rewards}")
    print(f"int rewards: {final_int_rewards}")
    running_time = newPolicy.running_time

    def get_cum_avg_reward(rewards: np.ndarray) -> np.ndarray:
        return np.cumsum(rewards) / (np.arange(len(rewards)) + 1)


    cum_frac_rewards = get_cum_avg_reward(final_frac_rewards)
    cum_int_rewards = get_cum_avg_reward(final_int_rewards)
    # print(f"cumulative averaged fractional rewards: {cum_frac_rewards}")
    
    save(output, {'cum_frac_rewards': cum_frac_rewards, 'cum_int_rewards': cum_int_rewards,
                    'running_time': running_time, 'opt_frac_reward': opt_frac_reward,
                    'opt_int_reward': opt_int_reward})
    logging.info(f"The rewards are saved to: {output}.")

