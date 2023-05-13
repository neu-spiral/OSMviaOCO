import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Main Module for ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problem', type=str, help='If the problem instance is created before, provide it here to save'
                                                    ' time instead of recreating it.')
    parser.add_argument('--problemType', default='IM', type=str, help='Type of the problem instance',
                        choices=['FL', 'IM', 'TF'])
    parser.add_argument('--input', default='', type=str,
                        help='Input file to generate problem instances from')
    parser.add_argument('--partitions', default=None,
                        help='Partitions file to generate constraints from')
    parser.add_argument('--policy', default='OGA', type=str, help='Type of the algorithm',
                        choices=['OGA', 'OMD', 'Optimistic', 'KKL'])  # policies
    parser.add_argument('--eta', default=0.1, type=float, help='eta of the policy')
    parser.add_argument('--k', default=1, type=int, help='cardinality k for each partition')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed to control the randomness of the algorithm')
    parser.add_argument('--T', default=100, type=int,
                        help='Number of iterations used in the maximization algorithm')
    parser.add_argument('--traceType', default='sequential', type=str, help='Construction type of the trace list',
                        choices=['sequential', 'random', 'custom'])
    parser.add_argument('--KKLalg', default='continuous-greedy', type=str,
                        help='Type of offline approximation algorithm for KKL',
                        choices=['continuous-greedy', 'greedy'])
    parser.add_argument('--setting', default='full-information', help='Type of feedback to the online player',
                        choices=['full-information', 'bandit'])

    subparsers = parser.add_subparsers(dest='subcommand')
    subparsers.required = False
    parser_custom = subparsers.add_parser('custom')
    parser_custom.add_argument('--trace', default=None, type=str, help='file that contains the trace of the policy')
    return parser
