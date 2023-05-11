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
    parser.add_argument('--input', default='./results/OGD', type=str,
                        help='Input directory for the plots')
    parser.add_argument('--type', default='TIMEvsREWARDS', type=str, help='Type of the plot',
                        choices=['TIMEvsREWARDS', 'LOGTIMEvsUTILITY', 'ITERATIONSvsUTILITY', 'SEEDSvsUTILITY',
                                 'PARETO', 'PARETOLOG', 'CONV_TEST'])
    parser.add_argument('--font_name', type=str, help='Font of the title and axes')
    parser.add_argument('--font_size', default=14, type=int, help='Font size of the title and axes')
    parser.add_argument('--marker_size', default=8, type=int, help='Font size of the markers')
    args = parser.parse_args()
    n = 10  # plot every nth element from a list
    font_name = args.font_name
    font_size = args.font_size
    marker_size = args.marker_size
    plt.rcParams.update({'font.size': font_size})
    poly_marker, samp_marker, swd_marker = 'X', '^', 's'
    poly_line, samp_line, swd_line = 'dashed', 'dashdot', 'dotted'
    poly_label, samp_label, swd_label = 'POLY', 'SAMP', 'SWD'

    path = args.input  # "./results/continuous_greedy"
    files = dict()

    for root, dir_names, f_names in os.walk(path):
        for f in f_names:
            if root in files:
                files[root].append(f)
            else:
                files[root] = [f]

    if args.type == 'TIMEvsREWARDS':
        root = 'results/OGD/IM/ZKC_10_01/k_3_100_iter/'
        # 'results/OGD/IM/RB1powerlaw_10_10_21/k_1_100_iter/'
        frac_file = 'fractional'
        int_file = 'integral'
        frac_results = load(os.path.join(root, frac_file))
        int_results = load(os.path.join(root, int_file))

        # range(len(frac_results[0]))
        plt.plot(range(len(frac_results[0])), frac_results[0], marker='o', markersize=marker_size,
                 label='fractional', linestyle='dashed')
        plt.plot(range(len(int_results[0])), int_results[0], marker='^', markersize=marker_size,
                 label='integral', linestyle='dotted')

        plt.ylabel("rewards")
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda l: l[0]))
        ax.legend(handles, labels)
        plt.show()
        output_dir = root.replace("results", "plots")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.xlabel("T")
        plt.savefig(output_dir + '/rewards.pdf', bbox_inches="tight")
        plt.savefig(output_dir + '/rewards.png', bbox_inches="tight")
        print(output_dir)
        plt.close()

    elif args.type == 'SEEDSvsUTILITY':
        seeds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        utility1 = []
        utility2 = []
        for seed in seeds:
            path1 = 'results/continuous_greedy/IM/epinions_100_10cascades/k_' + str(seed) \
                    + '_100_FW/polynomial_degree_1_around_05'
            path2 = 'results/continuous_greedy/IM/epinions_100_10cascades/k_' + str(
                seed) + '_100_FW/polynomial_degree_2_around_05'
            result1 = load(path1)
            result2 = load(path2)
            utility1.append(result1[-1][3])
            utility2.append(result2[-1][3])
        plt.figure()
        plt.plot(seeds, utility1, 's', label='Polynomial Estimator degree 1')
        plt.plot(seeds, utility2, 's', label='Polynomial Estimator degree 2')
        plt.title("Number of seeds vs utility")
        plt.xlabel("Constraints")
        plt.ylabel("f^(y)")
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda l: l[0]))
        ax.legend(handles, labels)
        plt.show()
        plt.savefig('seeds.pdf', bbox_inches="tight")

    elif args.type == 'PARETO':
        plt.figure()
        poly_dict = dict()
        samp_dict = dict()
        swd_dict = dict()
        for file in files:
            if "backup" not in file:
                result = load(path + '/' + file)  # result is a file with a list with lines in the form
                # (key, track[key][0], track[key][1], multilinear_relaxation(newProblem.utility_function,
                # track[key][1]), args.estimator, args.samples)
                if result[-1][4] == 'polynomial':
                    # poly_time.append(result[-1][1])
                    # #print('\n' + str(poly_time))
                    # poly_utility.append(result[-1][3])
                    poly_dict[result[-1][1]] = result[-1][3]
                    poly_label = 'POLY'
                    poly_marker = 'x'
                elif result[-1][4] == 'sampler':
                    # print('\n' + str(result[-1][4]))
                    # samp_time.append(result[-1][1])
                    # samp_utility.append(result[-1][3])
                    samp_dict[result[-1][1]] = result[-1][3]
                    samp_label = 'SAMP'
                    samp_marker = '^'
                else:
                    # print('\n' + str(result[-1][4]))
                    # swd_time.append(result[-1][1])
                    # swd_utility.append(result[-1][3])
                    swd_dict[result[-1][1]] = result[-1][3]
                    swd_label = 'SWD'
                    swd_marker = 'o'
            else:
                problem_file = path.replace('/', '_') \
                    .replace('_100_FW', '') \
                    .replace('results_continuous_greedy_', 'problems/')
                problem = load(problem_file)
                result = load(path + '/' + file)  # result is a list in the format [y, track, bases]
                file_name = file.split("_")
                track = result[1]
                max_key = max(track.iterkeys())
                t = []  # time it took to compute the fractional vector y
                # FW_iterations = []
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                if file_name[0] == 'polynomial':
                    poly_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                elif file_name[0] == 'sampler':
                    samp_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                else:
                    swd_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
        poly_lists = sorted(poly_dict.items())
        try:
            poly_time, poly_utility = zip(*poly_lists)
            plt.plot(poly_time, poly_utility, marker=poly_marker, markersize=marker_size, label=poly_label,
                     linestyle=poly_line)
        except ValueError:
            pass
        samp_lists = sorted(samp_dict.items())
        try:
            samp_time, samp_utility = zip(*samp_lists)
            plt.plot(samp_time, samp_utility, marker=samp_marker, markersize=marker_size, label=samp_label,
                     linestyle=samp_line)
        except ValueError:
            pass
        swd_lists = sorted(swd_dict.items())
        try:
            swd_time, swd_utility = zip(*swd_lists)
            plt.plot(swd_time, swd_utility, marker=swd_marker, markersize=marker_size,
                     label=swd_label, linestyle=swd_line)
        except ValueError:
            pass
        plt.title("Comparison of estimators")
        plt.xlabel("time spent (seconds)")
        plt.ylabel(r'$f (\mathbf{y})$')  # , fontsize=12)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda l: l[0]))
        ax.legend(handles, labels)
        plt.show()
        output_dir = 'results/plots/' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_pareto.pdf', bbox_inches="tight")

    elif args.type == 'PARETOLOG':
        for root in files:
            if "backup" not in root:
                poly_dict = dict()
                samp_dict = dict()
                swd_dict = dict()
                y_dict = dict()
                # print root
                try:
                    k = int(root.split('_')[-3])
                except:
                    pass
                plt.figure()
                for f in files[root]:
                    if "backup" not in f:
                        result = load(os.path.join(root, f))  # result is a file with a list with lines in the form
                        # (key, track[key][0], track[key][1], multilinear_relaxation(newProblem.utility_function,
                        # track[key][1]), args.estimator, args.samples)
                        # track[key][1] is y
                        if result[-1][4] == 'polynomial':
                            # result[-1][1] is running time
                            # result[-1][3] is utility aka f(y)
                            # time.append(datetime.timedelta(seconds=item[1]))  # datetime.datetime.(item[1])
                            poly_dict[result[-1][1]] = result[-1][3]  # / np.log(2)) * 100  # / np.sqrt(201)) * 100
                            # / np.log(2)) * 100
                            y = result[-1][2]
                            sys.stderr.write("\nPOLY of degree " + str(result[-1][5]) + " results in \ny = " + str(y))
                            selection = sorted(y.values(), reverse=True)
                            indices = sorted(range(1, len(y.values()) + 1), key=lambda i: list(y.values())[i - 1],
                                             reverse=True)
                            try:
                                selection = set(indices[:k])
                                sys.stderr.write("\nThis y selects " + str(selection))
                                y_dict[(result[-1][4], result[-1][5])] = (y, selection)  # y
                            except:
                                pass
                        elif result[-1][4] == 'sampler':
                            # result[-1][1] is running time
                            # result[-1][3] is utility aka f(y)
                            samp_dict[result[-1][1]] = result[-1][3]  # / np.log(2)) * 100
                            # / np.sqrt(201)) * 100  # / np.log(2)) * 100
                            y = result[-1][2]
                            sys.stderr.write("\nSAMP with " + str(result[-1][5]) + " samples results in \ny = " + str(y))
                            selection = sorted(y.values(), reverse=True)
                            indices = sorted(range(1, len(y.values()) + 1), key=lambda i: list(y.values())[i - 1],
                                             reverse=True)
                            try:
                                selection = set(indices[:k])
                                sys.stderr.write("\nThis y selects " + str(selection))
                                y_dict[(result[-1][4], result[-1][5])] = (y, selection)  # y
                            except:
                                pass
                        else:
                            # result[-1][1] is running time
                            # result[-1][3] is utility aka f(y)
                            swd_dict[result[-1][1]] = result[-1][3]  # / np.log(2)) * 100 # / np.sqrt(201)) * 100 #
                            y = result[-1][2]
                            sys.stderr.write("\nSWD with " + str(result[-1][5]) + " samples results in \ny = " + str(y))
                            selection = sorted(y.values(), reverse=True)
                            indices = sorted(range(1, len(y.values()) + 1), key=lambda i: y.values()[i - 1],
                                             reverse=True)
                            try:
                                selection = set(indices[:k])
                                sys.stderr.write("\nThis y selects " + str(selection))
                                y_dict[(result[-1][4], result[-1][5])] = (y, selection)  # y
                            except:
                                pass
                # sys.stderr.write("\ny_dict = " + str(y_dict))
                for est in y_dict:
                    if est != ('polynomial', 1):
                        try:
                            common = y_dict[est][1].intersection(y_dict[('polynomial', 1)][1])
                            different = y_dict[est][1].difference(y_dict[('polynomial', 1)][1])
                            sys.stderr.write("\n" + str(est[0]) + " estimator with " + str(est[1]) + " chooses the following "
                                             "elements in common with the polynomial estimator of degree 1 \n" + str(common) +
                                             "\nand the following elements differently \n" + str(different))
                        except:
                            pass
                    else:
                        pass
                poly_lists = sorted(poly_dict.items())
                try:
                    poly_time, poly_utility = zip(*poly_lists)
                    print(poly_time, poly_utility, 'POLY')
                    plt.semilogx(poly_time, poly_utility, marker=poly_marker, markersize=marker_size, label=poly_label,
                                 linestyle=poly_line)
                except ValueError:
                    pass
                samp_lists = sorted(samp_dict.items())
                try:
                    samp_time, samp_utility = zip(*samp_lists)
                    print(samp_time, samp_utility, 'SAMP')
                    plt.semilogx(samp_time, samp_utility, marker=samp_marker, markersize=marker_size, label=samp_label,
                                 linestyle=samp_line)
                except ValueError:
                    pass
                swd_lists = sorted(swd_dict.items())
                try:
                    swd_time, swd_utility = zip(*swd_lists)
                    print(swd_time, swd_utility, 'SWD')
                    plt.semilogx(swd_time, swd_utility, marker=swd_marker, markersize=marker_size, label=swd_label,
                                 linestyle=swd_line)
                except ValueError:
                    pass
                plt.title("Comparison of estimators")
                plt.xlabel("time spent (seconds)")
                plt.ylabel(r"$f(\mathbf{y})$")  # , fontsize=12)
                ax = plt.gca()
                handles, labels = ax.get_legend_handles_labels()
                # sort both labels and handles by labels
                try:
                    labels, handles = zip(*sorted(zip(labels, handles), key=lambda l: l[0]))
                except:
                    continue
                ax.legend(handles, labels)
                plt.show()
                output_dir = root.replace("results/continuous_greedy", "plots")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(output_dir + '/paretolog.pdf', bbox_inches="tight")
                plt.savefig(output_dir + '/paretolog.png', bbox_inches="tight")
                print(output_dir)
                plt.close()

    elif args.type == 'CONV_TEST':
        plt.figure()
        poly_dict = dict()
        samp_dict = dict()
        swd_dict = dict()
        for file in files:
            result = load(path + '/' + file)  # result is a list in the format [elapsed_time, args.degree,
            # poly_estimation]
            file_name = file.split("_")
            # track = result[1]
            # max_key = max(track.iterkeys())
            t = []  # time it took to compute the fractional vector y
            # FW_iterations = []
            estimates = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
            if file_name[0] == 'polynomial':
                poly_dict[result[0]] = result[2]
                poly_label = 'POLY'
            elif file_name[0] == 'sampler':
                samp_dict[result[0]] = result[2]
                samp_label = 'SAMP'
            else:
                swd_dict[result[0]] = result[2]
                swd_label = 'SWD'
        poly_lists = sorted(poly_dict.items())
        try:
            poly_time, poly_utility = zip(*poly_lists)
            plt.semilogx(poly_time, poly_utility, marker=poly_marker, markersize=marker_size, label=poly_label,
                         linestyle='dashed')
        except ValueError:
            pass
        samp_lists = sorted(samp_dict.items())
        try:
            samp_time, samp_utility = zip(*samp_lists)
            plt.semilogx(samp_time, samp_utility, marker=samp_marker, markersize=marker_size, label=samp_label,
                         linestyle='dashdot')
        except ValueError:
            pass
        swd_lists = sorted(swd_dict.items())
        try:
            swd_time, swd_utility = zip(*swd_lists)
            plt.semilogx(swd_time, swd_utility, marker=swd_marker, markersize=marker_size, label=swd_label,
                         linestyle='dotted')
        except ValueError:
            pass
        plt.title("Comparison of estimators")
        plt.xlabel("time spent (seconds)")
        plt.ylabel(r'$\hat{G} (\mathbf{y})$')  # , fontsize=12)
        plt.legend()  # fontsize='large')
        plt.show()
        output_dir = 'results/plots/tests/conv_tests/' + path.replace("results/convergence_test/problems", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_conv_test.pdf', bbox_inches="tight")

    else:
        for root in files: # keys of the files dict, keys are directory names
            if "backup" not in root:
                print(root)
                plt.figure()
                for f in files[root]:
                    t = []  # time it took to compute the fractional vector y
                    FW_iterations = []
                    objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the
                    # best estimator
                    if "backup" in f:
                        new_f = f
                        new_f = new_f.replace('_backup', '')
                        if new_f not in files[root]:
                            problem_file = root.split('_')[:-2]
                            problem_file = "_".join(problem_file) \
                                .replace('/', '_') \
                                .replace('._results_continuous_greedy_', './problems/')
                            problem = load(problem_file)
                            result = load(os.path.join(root, f))  # result is a list in the format [y, track, bases]
                            track = result[1]
                            for key in track:
                                t.append(track[key][0])
                                FW_iterations.append(key)
                                objectives.append(problem.objective_func(track[key][1]))

                            if 'polynomial' in f:
                                my_marker = poly_marker
                                my_label = poly_label + str(f.split('_')[-4])
                                my_line = poly_line
                            elif 'samplerWith' in f:
                                my_marker = swd_marker
                                my_label = swd_label + str(f.split('_')[-3])
                                my_line = swd_line
                            else:
                                my_marker = samp_marker
                                my_label = samp_label + str(f.split('_')[-3])
                                my_line = samp_line
                        else:
                            continue

                    else:
                        result = load(os.path.join(root, f)) # result is a file with a list with lines in the form
                        # (key, track[key][0], track[key][1], multilinear_relaxation(newProblem.utility_function,
                        # track[key][1]), args.estimator, args.samples)
                        for item in result:
                            t.append(item[1])  # datetime.datetime.(item[1])
                            FW_iterations.append(item[0])
                            objectives.append(item[3])
                        if 'polynomial' in f:
                            my_marker = poly_marker
                            my_label = poly_label + str(f.split('_')[-3])
                            my_line = poly_line
                        elif 'samplerWith' in f:
                            my_marker = swd_marker
                            my_label = swd_label + str(f.split('_')[-2])
                            my_line = swd_line
                        else:
                            my_marker = samp_marker
                            my_label = samp_label + str(f.split('_')[-2])
                            my_line = samp_line

                    if len(objectives) < 25:
                        n = 1
                    elif (len(objectives) >= 25) and (len(objectives) < 51):
                        n = 5
                    else:
                        pass

                    if args.type == 'TIMEvsUTILITY':
                        plt.plot(t[0::n], objectives[0::n], marker=my_marker, markersize=marker_size,
                                 label=my_label, linestyle=my_line)
                    elif args.type == 'LOGTIMEvsUTILITY':
                        plt.semilogx(t[0::n], objectives[0::n], marker=my_marker, markersize=marker_size,
                                     label=my_label, linestyle=my_line)
                    elif args.type == 'ITERATIONSvsUTILITY':
                        plt.plot(FW_iterations[0::n], objectives[0::n], marker=my_marker, markersize=marker_size,
                                 label=my_label, linestyle=my_line)
                    else:
                        pass

                plt.ylabel(r"$f(\mathbf{y})$")
                ax = plt.gca()
                handles, labels = ax.get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda l: l[0]))
                ax.legend(handles, labels)
                plt.show()
                output_dir = root.replace("results/continuous_greedy", "plots")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if args.type == 'TIMEvsUTILITY':
                    plt.xlabel("time (seconds)")
                    plt.savefig(output_dir + '/time.pdf', bbox_inches="tight")
                    plt.savefig(output_dir + '/time.png', bbox_inches="tight")
                elif args.type == 'LOGTIMEvsUTILITY':
                    plt.xlabel("time (seconds)")
                    plt.savefig(output_dir + '/logtime.pdf', bbox_inches="tight")
                    plt.savefig(output_dir + '/logtime.png', bbox_inches="tight")
                elif args.type == 'ITERATIONSvsUTILITY':
                    plt.xlabel("iterations")
                    plt.savefig(output_dir + '/iters.pdf', bbox_inches="tight")
                    plt.savefig(output_dir + '/iters.png', bbox_inches="tight")
                print(output_dir)
                plt.close()
