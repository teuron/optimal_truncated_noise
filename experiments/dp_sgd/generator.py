# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes Noise for DP-SGD using different compositions
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os  # noqa: 402
import parser  # noqa: 402

import evaluator as m  # noqa: 402
import numpy as np  # noqa: 402
import privacy_utils  # noqa: 402
import utils  # noqa: 402


def generate(method, epochs, loss_function, plot_dir, args):
    """Generates DP-SGD results for the thesis

    Args:
        method (String): Privacy Accountant method
        epochs (Integer): Epochs to train
        loss_function (String): Utility Loss function
        plot_dir (String): Plotting directory
        args (Arguments): Program arguments
    """
    args.method = method
    args.epochs = epochs
    args.utility_loss_function = loss_function

    # Create Path if it does not exists
    args.plot_dir = plot_dir
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # Compositions and utility weights to train with
    ncs = [1, 2, 4, 8, 16, 32, 64, 128]
    utility_weights = [0.1]

    for nc in ncs:
        args.number_of_compositions = nc
        args.fastmode = True
        args.no_tensorboard = True
        deltas, utilities = [], []

        # New directory for Number of Compositions
        args.plot_dir = plot_dir + "/comps_" + str(nc)
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

        for uw in utility_weights:
            args.utility_weight = uw
            args.fastmode = True
            args.no_tensorboard = True

            # Create new folder consisting of compositions and utility_weight
            noise_dir = plot_dir + "/comps_" + str(nc) + "_uw_" + str(uw)
            if not os.path.exists(noise_dir):
                os.makedirs(noise_dir)

            try:
                print("EXECUTING", args)
                predicted, _, _, _ = m.execute_single_run(args)
                p = predicted.cpu().detach().numpy()
                x_coords = np.linspace(-args.range_begin, args.range_begin, args.element_size, endpoint=True) + 10 ** -5
                utils.write_noise(noise_dir + "/noise.csv", p, x_coords)
                A, B = privacy_utils.calculate_a_b(p, args.range_begin, args.noise_class, args.mixture_q)
                deltas.append(privacy_utils.calculate_exact_upper_lower_delta_with_pb(A, B, args.range_begin, args.eps, args.number_of_compositions))

                if loss_function == "l2":
                    utilities.append(np.sqrt(np.dot(x_coords ** 2, p)))
                elif loss_function == "l1":
                    utilities.append(np.dot(np.abs(x_coords), p))
            except Exception as e:
                print("ERROR IN COMPUTATION")
                with open(noise_dir + "/error.txt", mode="a") as f:
                    f.write("Error in " + method + " " + str(nc) + " " + loss_function + " " + str(uw))
                    f.write(str(e))

        utils.write_delta_utility(plot_dir + "/comps_" + str(nc) + "delta_utilities.csv", utility_weights, deltas, utilities)


def main():
    args = parser.parse_arguments()
    np.set_printoptions(threshold=sys.maxsize)

    # Create plot directory
    args.plot_dir = "./plots"
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    plot_dir = args.plot_dir

    # Standard values
    args.random_init = True
    args.optim = "Adam"
    args.learning_rate = 0.001
    args.learning_rate_decay = 0.99995
    args.noise_model = "SigmoidModel"
    args.mixture_q = 0.1
    args.noise_class = "MixtureNoise"
    args.utility_weight_decay = True
    args.utility_weight_halving_epochs = 2500
    args.utility_weight_minimum = 0.0000001
    args.utility_weight = 0.01
    args.create_thesis_plots = True
    args.dp_sgd = True

    # Values set for the run
    args.sig_num = 10000
    args.element_size = 60000
    args.range_begin = 500
    args.eps = 0.3
    args.buckets_half = 500
    args.pb_buckets_half = 500
    args.factor = 1.000001

    args.plot_dir = "./plots_a_b"
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    plot_dir = args.plot_dir

    # What we want to generate
    args.min_of_two_delta = True
    generate("pb_ADP", 40000, "l2", plot_dir + "/pb_adp_l1_40k", args)
    generate("pb_PDP", 40000, "l2", plot_dir + "/pb_pdp_l2_40k", args)
    generate("renyi_markov", 200000, "l2", plot_dir + "/renyi_markov_l2_200k", args)

    args.plot_dir = "./plots_b_a"
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    plot_dir = args.plot_dir

    # What we want to generate
    args.min_of_two_delta = False
    generate("pb_ADP", 40000, "l2", plot_dir + "/pb_adp_l1_40k", args)
    generate("pb_PDP", 40000, "l2", plot_dir + "/pb_pdp_l2_40k", args)
    generate("renyi_markov", 200000, "l2", plot_dir + "/renyi_markov_l2_200k", args)


if __name__ == "__main__":
    main()
