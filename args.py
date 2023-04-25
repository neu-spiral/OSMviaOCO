import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Main Module for ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problem', type=str, help='If the problem instance is created before, provide it here to save'
                                                    ' time instead of recreating it.')
    parser.add_argument('--problemType', default='IM', type=str, help='Type of the problem instance',
                        choices=['FL', 'IM'])
    parser.add_argument('--input', default='', type=str,
                        help='Input file to generate problem instances from')
    parser.add_argument('--partitions', default=None,
                        help='Partitions file to generate constraints from')
    parser.add_argument('--policy', default='OGD', type=str, help='Type of the algorithm',
                        choices=['OGD', 'BanditOGD', 'whatever']) #policies?
    parser.add_argument('--eta', default=0.1, type=float, help='eta of the policy') #default eta?
    parser.add_argument('--k', default=1, type=int, help='cardinality k for each partition')
    parser.add_argument('--T', default=100, type=int,
                        help='Number of iterations used in the maximization algorithm')
    return parser