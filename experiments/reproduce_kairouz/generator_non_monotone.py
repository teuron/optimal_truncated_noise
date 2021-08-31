# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Shows the effects of a non monotone model
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import parser  # noqa: 402
import os  # noqa: 402

import evaluator as m  # noqa: 402
import numpy as np  # noqa: 402
import privacy_utils  # noqa: 402
import utils  # noqa: 402


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

    for range in [args.range_begin]:
        args.range_begin = range
        args.fastmode = True
        args.no_tensorboard = True

        # New directory for Number of Compositions
        args.plot_dir = plot_dir + "/range_" + str(range)
        noise_dir = args.plot_dir
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

        try:
            print("EXECUTING", args)
            predicted, _, _, _ = m.execute_single_run(args)
            p = predicted.cpu().detach().numpy()
            x_coords = np.linspace(-args.range_begin, args.range_begin, args.element_size, endpoint=True) + 10 ** -5
            utils.write_noise(f"{noise_dir}/noise.csv", p, x_coords)

            A, B = privacy_utils.calculate_a_b(p, args.range_begin, args.noise_class, args.mixture_q)
            delta = delta_func(A, B, args.range_begin, args.eps)
            dist_events = np.sum(A[B == 0])
            utils.write_delta(f"{noise_dir}/noise_delta_dist_events.csv", [delta], [dist_events])

            # Save optimal noise
            optimal_noise = privacy_utils.get_theoretical_optimal_noise_x(abs(x_coords), args.eps, args.number_of_compositions)
            utils.write_noise(f"{noise_dir}/optimal_noise.csv", optimal_noise, x_coords)
            A, B = privacy_utils.calculate_a_b(optimal_noise, args.range_begin, args.noise_class, args.mixture_q)
            delta = delta_func(A, B, args.range_begin, args.eps)
            dist_events = np.sum(A[B == 0])
            utils.write_delta(f"{noise_dir}/optimal_delta_dist_events.csv", [delta], [dist_events])

        except Exception as e:
            print("ERROR IN COMPUTATION")
            with open(f"{noise_dir}/error.txt", mode="a") as f:
                f.write("Error in " + method + " " + str(range) + " " + loss_function)
                f.write(str(e))


def main():
    args = parser.parse_arguments()
    np.set_printoptions(threshold=sys.maxsize)

    # Create plot directory
    args.plot_dir = "./plots_non_monotone"
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    plot_dir = args.plot_dir

    # Standard values
    args.random_init = True
    args.optim = "Adam"
    args.learning_rate = 0.01
    args.learning_rate_decay = 1.0
    args.noise_model = "CNNModel"
    args.utility_weight_decay = True
    args.utility_weight_halving_epochs = 2500
    args.utility_weight_minimum = 0.00001
    args.utility_weight = 0.5
    args.create_thesis_plots = True

    # Values set for the run
    args.sig_num = 10000
    args.element_size = 6000
    args.eps = 0.3
    args.buckets_half = 500
    args.factor = 1.00001
    args.cnns = 10
    args.number_of_compositions = 1
    args.slope = 500
    args.range_begin = 5
    args.monotone = False

    # What we want to generate
    generate("pb_ADP", 1400, "l1", plot_dir + "/pb_adp_cnn", args, privacy_utils.calculate_adp_delta)


if __name__ == "__main__":
    main()
