from ContinuousGreedy import multilinear_relaxation
from helpers import load
from ProblemInstances import InfluenceMaximization, FacilityLocation, derive
from tabulate import tabulate
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
    parser.add_argument('--input', default='./results', type=str,
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
        plots = dict()
        for root in files:
            eta = eval(root.split('/')[-1].split('_')[1].replace('p', '.'))
            gamma = eval(root.split('/')[-1].split('_')[3].replace('p', '.')) if 'gamma' in root else None
            frac_rewards = []
            int_rewards = []
            running_time = []
            frac_opts = []
            int_opts = []
            for f in files[root]:
                result = load(os.path.join(root, f))

                frac_rewards.append(result['cum_frac_rewards'])
                int_rewards.append(result['cum_int_rewards'])
                running_time.append(np.array(result['running_time']))
                frac_opts.append(result['opt_frac_reward'])

            dir_name = root.split('/')[-1]
            output_dir = root.replace('results', 'plots').replace(f"/{dir_name}", '')

            avg_frac_rewards = np.average(np.array(frac_rewards), axis=0)
            avg_int_rewards = np.average(np.array(int_rewards), axis=0)
            avg_running_time = np.average(np.array(running_time), axis=0)
            avg_frac_opt = np.average(np.array(frac_opts), axis=0)
            avg_int_opt = np.average(np.array(int_opts), axis=0)

            std_frac_rewards = np.std(np.array(frac_rewards), axis=0)
            std_int_rewards = np.std(np.array(int_rewards), axis=0)
            std_running_time = np.std(np.array(running_time), axis=0)
            std_frac_opt = np.std(np.array(frac_opts), axis=0)
            std_int_opt = np.std(np.array(int_opts), axis=0)

            if output_dir not in plots:
                plots[output_dir] = {(eta, gamma): {'avg_frac_rewards': avg_frac_rewards,
                                                    'avg_int_rewards': avg_int_rewards,
                                                    'avg_running_time': avg_running_time, 'avg_frac_opt': avg_frac_opt,
                                                    'avg_int_opt': avg_int_opt, 'std_frac_rewards': std_frac_rewards,
                                                    'std_int_rewards': std_int_rewards,
                                                    'std_running_time': std_running_time,
                                                    'std_frac_opt': std_frac_opt,
                                                    'std_int_opt': std_int_opt}}
            else:
                plots[output_dir][(eta, gamma)] = {'avg_frac_rewards': avg_frac_rewards,
                                                   'avg_int_rewards': avg_int_rewards,
                                                   'avg_running_time': avg_running_time, 'avg_frac_opt': avg_frac_opt,
                                                   'avg_int_opt': avg_int_opt, 'std_frac_rewards': std_frac_rewards,
                                                   'std_int_rewards': std_int_rewards,
                                                   'std_running_time': std_running_time,
                                                   'std_frac_opt': std_frac_opt,
                                                   'std_int_opt': std_int_opt}

        # print(f"plots are {plots}")
        for output_dir in plots:
            print(output_dir)
            jet = plt.get_cmap('tab20')
            colors = iter(jet(np.linspace(0, 1, 40)))
            frac_opts_per_eta = []
            for pair in plots[output_dir]:
                # print(f"eta is {eta}")
                eta = pair[0]
                gamma = pair[1]
                my_color = next(colors)
                result = plots[output_dir][(eta, gamma)]
                frac_opts_per_eta.append(result['avg_frac_opt'])
                T = len(result['avg_frac_rewards'])
                t1 = int(T / 3)
                t2 = int(2 * T / 3)
                t3 = T - 1
                headers = ['t', 'f_avg', 'f_std_dev', 'eta', 'gamma']
                table = [[t1, result['avg_frac_rewards'][t1], result['std_frac_rewards'][t1], eta, gamma],
                         [t2, result['avg_frac_rewards'][t2], result['std_frac_rewards'][t2], eta, gamma],
                         [t3, result['avg_frac_rewards'][t3], result['std_frac_rewards'][t3], eta, gamma]
                         ]
                print(tabulate(table, headers=headers))  # , floatfmt=".3f"))
                my_label = f"eta={eta}, gamma={gamma}, fractional" if gamma is not None else f"eta={eta}, fractional"
                plt.plot(range(T), result['avg_frac_rewards'],
                         label=my_label, linestyle='dashed', color=my_color)
                # plt.plot(range(len(result['avg_int_rewards'])), result['avg_int_rewards'],
                #          label=f"eta={eta}, gamma={gamma}, integral", linestyle='solid', color=my_color)
            my_color = next(colors)
            print(f"F_opt is {max(frac_opts_per_eta):.3f}")
            plt.axhline(y=max(frac_opts_per_eta), color=my_color, linestyle='-', label=f"fractional optimum")

            plt.ylabel("rewards")
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda l: l[0]))
            ax.legend(handles, labels)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.xlabel("T")
            plt.savefig(output_dir + '/rewards.pdf', bbox_inches="tight")
            plt.savefig(output_dir + '/rewards.png', bbox_inches="tight")
            plt.close()
