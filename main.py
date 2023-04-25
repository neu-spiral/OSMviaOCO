from helpers import save, load
from ProblemInstances import InfluenceMaximization, FacilityLocation
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

    eta = args.eta
    logging.basicConfig(level=logging.INFO)

    ### LOAD OR GENERATE THE PROBLEM INSTANCE
    #use here if the problem instance has already been created
    if args.problem is not None:
        newProblem = load(args.problem)
        args.problemType = args.problem.split("_")[0].split("/")[-1]
        args.input = args.problem.split("_")[1] + "_" + args.problem.split("_")[2] + "_" + args.problem.split("_")[3]
        args.constraints = int(args.problem.split("_")[-1])

    #use here to generate a new problem instance given the input and the partitions file
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
            cardinalities_k = list(k_list.values())
            sets_S = list(target_partitions.values())
            sets_S = [list(sets_S[i]) for i in range(len(sets_S))]
            new_decision_set = RelaxedPartitionMatroid(newProblem.X, k_list, target_partitions) ##make sure the k_list and target_partiotions formats fit to cardinalities_k, sets_S format
            logging.info('...done. %d seeds will be selected from each partition.' % args.constraints)

        elif args.problemType == 'IM':
            logging.info('Loading cascades...')
            graphs = load(args.input)
            if args.partitions is not None:
                target_partitions = load(args.partitions)
                k_list = dict.fromkeys(target_partitions.keys(), args.k)
            else:
                target_partitions = None
                k_list = args.k
            logging.info('...done. Just loaded %d cascades.' %(len(graphs)))
            logging.info('Defining an InfluenceMaximization problem...')
            newProblem = InfluenceMaximization(graphs, k_list, target_partitions)
            cardinalities_k = list(k_list.values())
            sets_S = list(target_partitions.values())
            sets_S = [list(sets_S[i]) for i in range(len(sets_S))]
            new_decision_set = RelaxedPartitionMatroid(newProblem.problemSize, cardinalities_k, sets_S) 
            logging.info('...done. %d seeds will be selected from each partition.' % args.k)

        #generate a file for problems if it does not already exists
        problem_dir = "problems/"
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)
        
        #save the newly generated problem instance
        save(problem_dir + args.problemType + "_" + args.input.split("/")[-1] + "_k_" + str(args.k),
             newProblem)

    
    ### GENERATE THE ThresholdObjective OBJECT
    new_objective = newProblem.translate() ## TODO test what new_objective returns
    
    
    ### CREATE THE OUTPUT DIRECTORY TO SAVE THE RESULTS IF NOT ALREADY EXISTS
    output_dir = "results/" + args.policy + "/" + args.problemType + "/" + args.input.split("/")[-1] + "/k_" \
                       + str(args.k) + "_" + str(args.T) + "_iter" + "/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('...output directory is created...')
    sys.stderr.write('output directory is:' + output_dir)
    
    frac_output = output_dir + 'fractional'
    int_output = output_dir + 'integral'
    

    ### GENERATE THE OCOPolicy OBJECT
    if args.policy == 'OGD':
        newPolicy = OGDPolicy(new_decision_set, eta, new_objective) ## TODO this class is not defined yet
        logging.info("An Online Gradient Descent policy is generated.")
        
    elif args.policy == 'BanditOGD':
        newPolicy = BanditOGDPolicy(new_decision_set, eta, new_objective) ## TODO this class is not defined yet
        logging.info("A Bandit Online Gradient Descent policy is generated.")
        pass
    
    elif args.policy == 'whatever': ## TODO change with policy names
        newPolicy = OCOPolicy(new_decision_set, eta, new_objective) 
        logging.info("An Online Convex Optimization policy is generated.")
        pass

    
    #RUN THE OCOPolicy
    while newPolicy.current_iteration < args.T:
        # TODO design backups if the algorithm is interrupted
        logging.info("Running iteration #{x}...\n".format(x=newPolicy.current_iteration)) ## TODO format string
        newPolicy.step()
    logging.info("The algorithm is finished.")
    
    #SAVE THE RESULTS OF THE OCOPolicy
    final_frac_rewards = newPolicy.frac_rewards
    final_int_rewards = newPolicy.int_rewards
    
    save(frac_output, final_frac_rewards)
    save(int_output, final_int_rewards)
    logging.info("The rewards are saved to: " + output_dir + ".")
