""" Contains parsing functions"""
# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)

import argparse
import numpy as np


def parse_arguments():
    """Setting up and parsing command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument("--create_utility_delta_plot", action="store_true", default=False, help="create a utility delta plot")
    parser.add_argument("--eps", type=float, default=0.3, help="the epsilon value the programm to achieve")
    parser.add_argument("--range_begin", type=float, default=15, help="half the width of the symmetric the range of the noise.")
    parser.add_argument("--element_size", type=int, default=60000, help="number of events that occure as noise (resolution of noise)")
    parser.add_argument("--number_of_compositions", type=int, default=1, help="Number_of_compositions")
    parser.add_argument("--noise_model", type=str, default="noise_model_Dirac_delta", help="Noise model")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "Adagrad", "RMSProp"], help="Choose optimizer")
    parser.add_argument("--noise_class", type=str, default="SymmetricNoise", choices=["SymmetricNoise", "MixtureNoise"], help="Choose noise class")
    parser.add_argument("--add_gradient_noise", dest="gradient_noise", action="store_true", default=False, help="Add noise to gradient")
    parser.add_argument(
        "--alternate_lamda_delta_update", dest="alternate_lamda_delta_update", action="store_true", default=False, help="Alternate the update of LAM and Delta in Renyi Markov"
    )

    parser.add_argument("--random_init", dest="random_init", action="store_true", default=False, help="Use random initialization for Sigmoid Model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="what log level is desired.")

    parser.add_argument(
        "--debug", nargs="+", default="slicing", choices=["slicing", "array_content", "gradients"], help="what debug output is desired. Enter as space speareted list."
    )

    parser.add_argument(
        "--method", nargs="?", default="renyi_markov", choices=["pb_ADP", "pb_PDP", "renyi_markov", "fft"], help="what method for composition and delta computation to use"
    )

    parser.add_argument("--utility_loss_function", default="l1", choices=["l1", "l2"], help="What function is used to weight the distance from the expectated result.")
    parser.add_argument("--utility_weight", type=float, default=0.001, help="The weight of the utility_loss compared to the noise delta")
    parser.add_argument("--utility_weight_decay", action="store_true", default=False, help="Whether we use weight_decay on utility_weight or not")
    parser.add_argument("--utility_weight_halving_epochs", type=float, default=5000, help="How many epochs are needed to half the utility_weight")
    parser.add_argument("--utility_weight_minimum", type=float, default=0.0001, help="Minimum utility weight")
    parser.add_argument("--learning_rate_decay", type=float, default=1.0, help="Exponential learning rate decay")
    parser.add_argument("--epochs", type=int, default=15000, help="number of noise optimization iterations.")
    parser.add_argument("--pt_seed", type=int, default=5678, help="seed for pytorch backend")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="directory to place plots")
    parser.add_argument("--dump_data", action="store_true", default=False, help="whether to dump data")
    parser.add_argument("--data_dir", type=str, default="./data_out", help="directory to place data")
    parser.add_argument("--monotone", dest="monotone", action="store_true", default=True, help="use monotonnicity constraint")
    parser.add_argument("--non_monotone", dest="monotone", action="store_false", help="no monotonnicity constraint")
    parser.add_argument("--sig_num", type=int, default=10000, help="number of sigmoids to use for the approximation")
    parser.add_argument("--scale_start", type=int, default=1, help="To what should the scale of all sigmoids be initialized")
    parser.add_argument("--slope", type=int, default=500, help="To what should the slope of all sigmoids be initialized")
    parser.add_argument("--bias", type=int, default=10, help="To what should the bias of all models be initialized")
    parser.add_argument("--cnns", type=int, default=5, help="number of cnns to use for the approximation")
    parser.add_argument("--lambda_exponent", type=float, default=2, help="exponent used to learn lambda in even epochs")
    parser.add_argument("--mixture_q", type=float, default=0.1, help="for MixtureNoise how much A in B")
    parser.add_argument("--GPU", action="store_true", default=False, help="execute on GPU")
    parser.add_argument("--fastmode", action="store_true", default=False, help="execute as fast as possible (less logging, no tensorboard, etc.)")
    parser.add_argument("--no_tensorboard", action="store_true", default=False, help="whether to write data to tensorboard or not")
    parser.add_argument("--create_thesis_plots", action="store_true", default=False, help="whether to create thesis plots or not")
    parser.add_argument("--log_space", action="store_true", default=False, help="whether to learn delta in log_space or not")
    parser.add_argument("--min_of_two_delta", action="store_true", default=False, help="whether to add both deltas in criterion or not")
    parser.add_argument("--dp_sgd", action="store_true", default=False, help="whether we learn dp_sgd or not")
    parser = add_pb_parser_arguments(parser)

    args = parser.parse_args()

    return args


def add_pb_parser_arguments(parser):
    """ adds arguments used in bucketing to the parser """
    parser.add_argument("--buckets_half", type=int, default=500, help="half the number of buckets. Shall be devisible by 2")

    parser.add_argument("--factor", type=int, default=np.float64(1.00001), help="factor for pb")

    return parser
