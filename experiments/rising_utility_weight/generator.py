# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes the optimality graphs
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import parser  # noqa: 402
import os  # noqa: 402

import evaluator as m  # noqa: 402
import numpy as np  # noqa: 402
import privacy_utils  # noqa: 402
import utils  # noqa: 402
import csv


def write_noise(filename, delta, utility):
    """Writes noise and x coordinates to a csv file

    Args:
        filename (String): Filename to write to
        noise (nd.array): Generated Noise
        x_coords (nd.array): X-axis discretization
    """
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["delta", "utility"])
        writer.writerows(list(zip(delta, utility)))


def generate(method, epochs, loss_function, plot_dir, args, delta_func):
    """Produces the results for the thesis

    Args:
        method (String): Privacy Accountant method
        epochs (Integer): Epochs to train
        loss_function (String): Utility Loss function
        plot_dir (String): Plotting directory
        args (Arguments): Program arguments
        delta_func (function): Function that computes delta(eps) from the noise
    """
    args.method = method
    args.epochs = epochs
    args.utility_loss_function = loss_function

    args.plot_dir = plot_dir
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    epses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    utility_weights = [0.001, 0.003, 0.005, 0.008, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0]
    for eps in epses:
        args.eps = eps
        args.number_of_compositions = 128
        args.fastmode = True
        args.no_tensorboard = True
        deltas, utilities = [], []
        # New directory for Number of Compositions
        args.plot_dir = plot_dir + "/eps_" + str(eps)
        comps_dir = args.plot_dir
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

        for uw in utility_weights:
            args.plot_dir = comps_dir
            args.utility_weight = uw
            args.fastmode = True
            args.no_tensorboard = True
            noise_dir = plot_dir + "/eps_" + str(eps) + "_uw_" + str(uw)
            if not os.path.exists(noise_dir):
                os.makedirs(noise_dir)
            try:
                print("EXECUTING", args)
                predicted, _, _, delta, utilities = m.execute_single_run(args)
                p = predicted.cpu().detach().numpy()
                x_coords = np.linspace(-args.range_begin, args.range_begin, args.element_size, endpoint=True) + 10 ** -5
                utils.write_noise(noise_dir + "/noise.csv", p, x_coords)
                A, B = privacy_utils.calculate_a_b(p, args.range_begin, args.noise_class, args.mixture_q)
                deltas.append(delta_func(A, B, args.range_begin, args.eps))

                if loss_function == "l2":
                    utilities.append(np.sqrt(np.dot(x_coords ** 2, p)))
                elif loss_function == "l1":
                    utilities.append(np.dot(np.abs(x_coords), p))
            except Exception as e:
                print("ERROR IN COMPUTATION")
                with open(noise_dir + "/error.txt", mode="a") as f:
                    f.write("Error in " + method + " " + str(eps) + " " + loss_function + " " + str(uw))
                    f.write(str(e))

            write_noise(noise_dir + "/delta_utilities.csv", delta, utilities)
        utils.write_delta_utility(plot_dir + "/eps_" + str(eps) + "delta_utilities.csv", utility_weights, deltas, utilities)


def main():
    args = parser.parse_arguments()
    np.set_printoptions(threshold=sys.maxsize)

    # Standard values
    args.random_init = True
    args.optim = "Adam"
    args.learning_rate = 0.001
    args.learning_rate_decay = 0.99995
    args.noise_model = "SigmoidModel"
    args.utility_weight_decay = True
    args.utility_weight_halving_epochs = 2500
    args.utility_weight_minimum = 0.0000001
    args.utility_weight = 0.1
    args.create_thesis_plots = True

    # Values set for the run
    args.sig_num = 10000
    args.element_size = 60000
    args.range_begin = 400
    args.eps = 0.3
    args.buckets_half = 500
    args.pb_buckets_half = 500
    args.factor = 1.000001

    for uwd in [False]:
        # Create plot directory
        args.plot_dir = f"./plots_uwd_{uwd}"
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)
        plot_dir = args.plot_dir

        # What we want to generate
        #  args.utility_weight_halving_epochs = 16666
        # generate("renyi_markov", 100000, "l1", plot_dir + "/renyi_markov_l1", args, privacy_utils.calculate_pdp_delta)
        # generate("renyi_markov", 100000, "l2", plot_dir + "/renyi_markov_l2", args, privacy_utils.calculate_pdp_delta)
        args.utility_weight_decay = False
        args.utility_weight_halving_epochs = 10000
        # generate("pb_ADP", 15000, "l1", plot_dir + "/pb_adp_l1", args, privacy_utils.calculate_adp_delta)
        generate("pb_ADP", 10000, "l2", plot_dir + "/pb_adp_l2", args, privacy_utils.calculate_adp_delta)
        # generate("pb_PDP", 15000, "l1", plot_dir + "/pb_pdp_l1", args, privacy_utils.calculate_pdp_delta)
        # generate("pb_PDP", 15000, "l2", plot_dir + "/pb_pdp_l2", args, privacy_utils.calculate_pdp_delta)


if __name__ == "__main__":
    main()
