# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Used to estimate the computational efficiency
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import parser  # noqa: 402

import evaluator as m  # noqa: 402
import numpy as np  # noqa: 402


def generate(method, epochs, args):
    """Used to estimate the computational efficiency

    Args:
        method (String): Privacy Accountant method
        epochs (Integer): Epochs to train
        args (Arguments): Program arguments
    """
    args.method = method
    args.epochs = epochs

    # Composition
    ncs = [1, 128]
    for nc in ncs:
        args.number_of_compositions = nc
        args.fastmode = True
        args.no_tensorboard = True

        try:
            print("EXECUTING", args)
            _, _, _, _ = m.execute_single_run(args)
        except KeyboardInterrupt as e:
            print(str(e))


def main():
    args = parser.parse_arguments()
    np.set_printoptions(threshold=sys.maxsize)

    # Standard values
    args.random_init = True
    args.optim = "Adam"
    args.learning_rate = 0.01
    args.learning_rate_decay = 0.99995
    args.noise_model = "SigmoidModel"
    args.mixture_q = 0.1
    args.utility_weight_decay = True
    args.utility_weight_halving_epochs = 2500
    args.utility_weight_minimum = 0.0000001
    args.utility_weight = 0.01
    args.create_thesis_plots = True

    # Values set for the run
    args.sig_num = 10000
    args.element_size = 60000
    args.range_begin = 500
    args.eps = 0.3
    args.buckets_half = 500
    args.factor = 1.000001

    args.noise_class = "SymmetricNoise"
    generate("renyi_markov", 100000, args)
    generate("pb_ADP", 15000, args)
    generate("pb_PDP", 15000, args)

    args.noise_class = "MixtureNoise"
    generate("renyi_markov", 100000, args)
    generate("pb_ADP", 15000, args)
    generate("pb_PDP", 15000, args)


if __name__ == "__main__":
    main()
