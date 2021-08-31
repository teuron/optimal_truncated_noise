# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes the std when trained with different seeds
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import parser  # noqa: 402
import os  # noqa: 402
import random  # noqa: 402

import evaluator as m  # noqa: 402
import numpy as np  # noqa: 402
import privacy_utils  # noqa: 402
import utils  # noqa: 402


def generate(method, epochs, loss_function, plot_dir, args, seed):
    """Generates training standard deviation results for the thesis

    Args:
        method (String): Privacy Accountant method
        epochs (Integer): Epochs to train
        loss_function (String): Utility Loss function
        plot_dir (String): Plotting directory
        args (Arguments): Program arguments
        seed (Integer): Seed to use for the run
    """
    args.method = method
    args.epochs = epochs
    args.utility_loss_function = loss_function

    # Create Path if it does not exists
    args.plot_dir = plot_dir
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    random.seed(1234)
    seeds = [int(random.random() * 100000) for _ in range(10)]
    print(seeds)

    for seed in seeds:
        args.pt_seed = seed
        args.fastmode = True
        args.no_tensorboard = True

        # New directory for Number of Compositions
        args.plot_dir = plot_dir + "/seed_" + str(seed)
        seed_dir = args.plot_dir
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

        try:
            print("EXECUTING", args)

            predicted, _, _, deltas = m.execute_single_run(args)
            p = predicted.cpu().detach().numpy()
            x_coords = np.linspace(-args.range_begin, args.range_begin, args.element_size, endpoint=True) + 10 ** -5
            utils.write_noise(f"{seed_dir}/noise.csv", p, x_coords)
            utils.write_delta_adp_pdp(f"{seed_dir}/noise_deltas.csv", list(range(args.epochs)), deltas["ADP"], deltas["PDP"])
        except Exception as e:
            print("ERROR IN COMPUTATION")
            with open(f"{seed_dir}/error.txt", mode="a") as f:
                f.write("Error in " + method + " " + str(range) + " " + loss_function)
                f.write(str(e))


def main():
    args = parser.parse_arguments()
    np.set_printoptions(threshold=sys.maxsize)
    seed = 1234

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
    args.utility_weight_decay = True
    args.utility_weight_halving_epochs = 2500
    args.utility_weight_minimum = 0.0000001
    args.create_thesis_plots = True

    # Values set for the run
    args.sig_num = 10000
    args.element_size = 60000
    args.eps = 0.3
    args.buckets_half = 500
    args.pb_buckets_half = 500
    args.factor = 1.000001
    args.number_of_compositions = 1

    args.range_begin = 5

    # What we want to generate
    args.utility_weight = 0.1
    generate("renyi_markov", 100000, "l1", plot_dir + "/renyi_markov", args, seed)
    args.utility_weight = 0.5
    generate("pb_ADP", 40000, "l1", plot_dir + "/pb_adp", args, seed)
    args.utility_weight = 0.1
    generate("pb_PDP", 40000, "l1", plot_dir + "/pb_pdp", args, seed)


if __name__ == "__main__":
    main()
