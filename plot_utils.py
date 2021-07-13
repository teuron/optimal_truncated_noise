""" Contains utility functions for plotting"""
# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)

import os

import numpy as np
import scipy
from matplotlib import pyplot as plt

import privacy_utils
import utils
from privacybuckets import PrivacyBuckets


def plot_log_and_normal(arrays_to_plot, util_delta_plot_dir, name, args, persist=True):
    """Plots a given array as log and normal

    Args:
        arrays_to_plot ([(String, [double])]): Arrays we want to plot
        util_delta_plot_dir (String): Plot directory
        name (String): Name of the plot
        args (Arguments): Program arguments
        persist (Boolean): If it shall be persisted to disc or not
    """
    log_figure = save_arrays_as_log_plot(arrays_to_plot, os.path.join(util_delta_plot_dir, f"{name}_log.png"), args.element_size, args.range_begin, persist=persist)
    log_figure.clf()
    figure = save_arrays_as_plot(arrays_to_plot, os.path.join(util_delta_plot_dir, f"{name}.png"), args.element_size, args.range_begin, persist=persist)
    figure.clf()


def save_arrays_as_log_plot(p, name, element_size, range_begin, persist=False):
    """Saves a numpy array as a log plot using the plot function

    Args:
        p ([type]): [description]
        name ([type]): [description]
        element_size ([type]): [description]
        range_begin ([type]): [description]
        plot_func ([type], optional): [description]. Defaults to plt.semilogy.
        adapt_indices (tuple, optional): [description]. Defaults to (0, 1).
    """
    return save_arrays_as_plot(p, name, element_size, range_begin, plot_func=plt.semilogy, adapt_indices=(0, 1), persist=persist)


def save_arrays_as_plot(p, name, element_size, range_begin, plot_func=plt.plot, adapt_indices=(0, 1), persist=False):
    """Saves a numpy array as a plot using the plot function

    Args:
        p ([type]): [description]
        name ([type]): [description]
        element_size ([type]): [description]
        range_begin ([type]): [description]
        plot_func ([type], optional): [description]. Defaults to plt.plot.
        adapt_indices (tuple, optional): [description]. Defaults to (0, 1).
    """
    plt.clf()
    figure = plt.figure()
    plt.title("pixels {} | range {}".format(element_size, (-range_begin, range_begin)))

    x = np.arange(element_size) / element_size * (2 * range_begin) - range_begin

    y_min = 1
    y_max = 0
    for i, (value, label) in enumerate(p):
        if i in adapt_indices:
            y_min = min(y_min, np.min(value))
            y_max = max(y_max, np.max(value))
        plot_func(x, value, alpha=0.5, label=label)

    plt.ylim((y_min, y_max * 1.05))
    plt.legend()

    if persist:
        plt.savefig(name)
    return figure


def create_arrays_to_plot(predicted, x_coords, optimal_noise, markov_pdp_delta, args):
    """Creates an array of all values we want to plot

    Args:
        predicted (torch.DoubleTensor): Predicted Noise
        x_coords (torch.DoubleTensor): Input coordinates
        optimal_noise ([double]): Optimal Noise
        markov_pdp_delta (torch.DoubleTensor): Not the combined delta!
        args (Arguments): Program arguments

    Returns:
        [([double], String)]: Array of (Value, Name) tuples -> Predicted, Optimal Noise, Fitted Gauss and Priv. Gauss
    """
    # fit a Gaussian:
    fitted_gauss, _, _ = privacy_utils.fit_gaussian(predicted.cpu().detach(), x_coords)

    A, B = privacy_utils.calculate_a_b(predicted.detach().cpu().numpy(), args.range_begin, args.noise_class, args.mixture_q)

    # calculate optimal Gauss
    # assert args.noise_model == "noise_model_Dirac_delta", "this implementation does work only for Diract delta noise"
    mu = 1
    priv_sig = privacy_utils.optimal_sigma_gauss_PDP(mu=mu, n=args.number_of_compositions, eps=args.eps, PDP_delta=markov_pdp_delta.cpu().detach().numpy())

    priv_gauss = scipy.stats.norm.pdf(x_coords, loc=0, scale=priv_sig)
    priv_gauss /= sum(priv_gauss)

    priv_sig = privacy_utils.optimal_sigma_gauss_PDP(
        mu=mu, n=args.number_of_compositions, eps=args.eps, PDP_delta=privacy_utils.calculate_pdp_delta(A, B, args.range_begin, args.eps)
    )
    gauss = scipy.stats.norm.pdf(x_coords, loc=0, scale=priv_sig)
    gauss /= sum(gauss)

    return [[predicted.cpu().detach().numpy(), "p"], [optimal_noise, "optimal_noise"], [fitted_gauss, "fitted_gauss"], [priv_gauss, "priv_gauss"], [gauss, "Gaussian(PDP(p))"]]


def plot_markov_pdp(privacy_loss, log_A, number_of_compositions, eps, current_lam, epoch):
    """Plots the current and perfect LAM in range 0-500 for Renyi Markov

    Args:
        privacy_loss (torch.DoubleTensor): Privacy loss
        log_A (torch.DoubleTensor): torch.log(slice of p_A)
        number_of_compositions (int): Number of compositions
        eps (double): Epsilon privacy
        current_lam (torch.DoubleTensor): current LAM in Renyi Markov
        epoch (int): Current epoch

    Returns:
        (plt.figure, double): Created plot and minimum LAM for Renyi Markov
    """
    lam = np.linspace(0.0, 500.0, 10000)
    renyi_div_times_lam = scipy.special.logsumexp(np.add(np.multiply(privacy_loss, lam[:, np.newaxis]), log_A), 1)
    Y = np.exp((renyi_div_times_lam * number_of_compositions) - (lam * eps))
    plt.clf()

    figure = plt.figure()
    plt.title("Perfect LAM for Epoch {}".format(epoch))
    plt.plot(lam, Y, alpha=0.5, label="Markov PDP")
    plt.axvline(x=current_lam, color="red", label="Current LAM")
    plt.axvline(x=lam[np.argmin(Y)], color="green", label="Best LAM")
    plt.legend()

    print(current_lam, lam[np.argmin(Y)], epoch)

    return figure, lam[np.argmin(Y)]


def plot_thesis_plots(pdp_delta, args, name):
    """Plots and saves pdp_deltas per epoch

    Args:
        pdp_delta ([float]): A list of pdp deltas to plot
        args (Arguments): Program arguments
        name (String): Name of the plot to save to
    """
    plt.clf()
    figure = plt.figure()
    plt.title("Delta plot in range {}".format((-args.range_begin, args.range_begin)))
    plt.semilogy(pdp_delta)
    plt.ylim((0, max(pdp_delta) * 1.05))
    plt.savefig(name, bbox_inches="tight")
    return figure


def plot_pld(A, B, args, name, factor=1.000001, buckets=10000, persist=False, plt_func=plt.plot):
    """Plots the PLD for two given input distributions in normal and log

    Args:
        A (nd.array): Distribution
        B (nd.array): Shifted Distribution
        args (Arguments): Program arguments
        name (str): Filename

    Returns:
        [Figure]: The generated figure
    """
    privacybuckets = PrivacyBuckets(
        number_of_buckets=buckets,
        factor=utils.get_good_factor(factor, args.eps, buckets // 2),
        dist1_array=A,  # distribution A
        dist2_array=B,  # distribution B
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
    )
    privacybuckets_composed = privacybuckets.compose(args.number_of_compositions)
    x_coords = np.linspace(-buckets // 2, buckets // 2 + 1, buckets + 2, endpoint=True) * privacybuckets_composed.log_factor
    pld = np.concatenate((privacybuckets_composed.bucket_distribution, privacybuckets_composed.infty_bucket + privacybuckets_composed.distinguishing_events), axis=None)
    print(privacybuckets_composed.infty_bucket, privacybuckets_composed.distinguishing_events)
    plt.clf()
    figure = plt.figure()
    plt.title("Privacy Loss Distribution")
    plt_func(x_coords, pld, label="PLD")
    plt.vlines(args.eps, 0, max(pld) * 1.05, label="Eps", colors="g")
    plt.ylim((0, max(pld) * 1.05))
    plt.legend()
    if persist:
        plt.savefig(name, bbox_inches="tight")
    return figure
