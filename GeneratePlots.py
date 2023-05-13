from ContinuousGreedy import multilinear_relaxation
from helpers import load
from ProblemInstances import InfluenceMaximization, FacilityLocation, derive
import argparse
import datetime
import numpy as np
import os
import sys
import matplotlib

matplotlib.use('Agg')
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotter for results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default='./results/OGA', type=str,
                        help='Input directory for the plots')
    parser.add_argument('--type', default='ITERvsREWARDS', type=str, help='Type of the plot',
                        choices=['TIMEvsREWARDS', 'ITERvsREWARDS'])
    parser.add_argument('--font_name', type=str, help='Font of the title and axes')
    parser.add_argument('--font_size', default=10, type=int, help='Font size of the title and axes')
    parser.add_argument('--marker_size', default=8, type=int, help='Font size of the markers')
    args = parser.parse_args()
    n = 10  # plot every nth element from a list
    font_name = args.font_name
    font_size = args.font_size
    marker_size = args.marker_size
    plt.rcParams.update({'font.size': font_size})

    path = args.input  # "./results/continuous_greedy"
    files = dict()

    for root, dir_names, f_names in os.walk(path):
        for f in f_names:
            if root in files:
                files[root].append(f)
            else:
                files[root] = [f]

    if args.type == 'ITERvsREWARDS':
        for root in files:
            frac_opts = []
            jet = plt.get_cmap('tab10')
            colors = iter(jet(np.linspace(0, 1, 10)))
            for f in files[root]:
                result = load(os.path.join(root, f))
                eta = eval(f.split('_')[-1].replace('p', '.'))
                my_color = next(colors)
                frac_opts.append(result['opt_frac_reward'])
                plt.plot(range(len(result['cum_frac_rewards'])), result['cum_frac_rewards'],
                         label=f"eta={eta}, fractional", linestyle='dashed', color=my_color)
                plt.plot(range(len(result['cum_int_rewards'])), result['cum_int_rewards'],
                         label=f"eta={eta}, integral", linestyle='solid', color=my_color)
                # plt.plot(range(len(result['cum_frac_rewards'])), result['cum_frac_rewards'], marker='o',
                #          markersize=marker_size, label=f"eta={eta}, fractional", linestyle='dashed', color=my_color)
                # plt.plot(range(len(result['cum_int_rewards'])), result['cum_int_rewards'], marker='^',
                #          markersize=marker_size, label=f"eta={eta}, integral", linestyle='dotted', color=my_color)

            plt.axhline(y=max(frac_opts), color='r', linestyle='-')
            plt.ylabel("rewards")
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda l: l[0]))
            ax.legend(handles, labels)
            plt.show()
            output_dir = root.replace("results", "plots")  # .replace(f"eta_{str(eta).replace('.', 'p')}", '')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.xlabel("T")
            plt.savefig(output_dir + '/rewards.pdf', bbox_inches="tight")
            plt.savefig(output_dir + '/rewards.png', bbox_inches="tight")
            print(output_dir)
            plt.close()
