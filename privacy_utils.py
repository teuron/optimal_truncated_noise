""" Contains utility functions for privacy computations"""
# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)
import ctypes
import numpy as np
import scipy.special as special
import scipy.stats as stats
import torch

import utils

from privacybuckets import PrivacyBuckets

# Old version to compute the optimal noise
"""
def get_theoretical_optimal_noise(element_size, step, eps):
    step = int(1/step)
    gamma = 1 / ( 1 + np.exp(eps/2) )
    y = np.zeros(element_size)
    mid = element_size // 2  # improper
    width = int(step*gamma)
    for i in range( (element_size // step ) // 2 ):
        y[mid + i * step:mid + i * step + width] = np.exp(-i*eps)
        y[mid + i * step + width:mid + (i+1) * step] = np.exp(-(i+1)*eps)
        y[mid - i * step - width: mid - i * step] = np.exp(-i*eps)
        y[mid - (i+1) * step: mid - i * step - width] = np.exp(-(i+1)*eps)

    y /= np.sum(y)

    return y
"""


def get_theoretical_optimal_noise_x(x, eps, number_of_compositions):
    """Calculates theoretical optimal noise

    Args:
        x ([double]): input values
        eps (double): Epsilon for privacy
        number_of_compositions (int): Number of compositions

    Returns:
        [double]: Theoretical optimal noise for the input
    """
    gamma = 1 / (1 + np.exp((eps / number_of_compositions) / 2))
    k = np.array(abs(x), dtype=np.int)
    g = np.array((abs(x) - k) > gamma, dtype=np.int)
    y = np.exp(-(k + g) * (eps / number_of_compositions))
    y /= np.sum(y)
    return y


def calculate_normal_space_adp_priv_gauss_multiplier_in_normal_space(mu, sigma, eps):
    """Calculates the second part of the equation in Lemma 12
    from https://eprint.iacr.org/2018/820.pdf Privacy Loss Classes: The Central Limit Theorem in Differential Privacy
    in normal space
    Args:
        mu (double): Mu
        sigma (double): Sigma
        eps (double): Epsilon

    Returns:
        double: Second part of the equation in normal space
    """
    return special.erfc((eps - mu) / (np.sqrt(2) * sigma)) - np.exp(eps - mu + (sigma * sigma) / 2) * special.erfc((eps - mu + sigma * sigma) / (np.sqrt(2) * sigma))


def calculate_normal_space_adp_priv_gauss_multiplier_in_log_space(mu, sigma, eps):
    """Calculates the second part of the equation in Lemma 12
    from https://eprint.iacr.org/2018/820.pdf Privacy Loss Classes: The Central Limit Theorem in Differential Privacy
    in log space
    Args:
        mu (double): Mu
        sigma (double): Sigma
        eps (double): Epsilon

    Returns:
        double: Second part of the equation in normal space
    """
    try:
        gls = ctypes.CDLL(utils.GSL_PATH)
        gls.gsl_sf_log_erfc.restype = ctypes.c_double
        answer = gls.gsl_sf_log_erfc(ctypes.c_double((eps - mu + sigma * sigma) / (np.sqrt(2) * sigma)))
        return special.erfc((eps - mu) / (np.sqrt(2) * sigma)) - np.exp(answer + eps + (sigma * sigma) / 2 - mu)
    except:  # noqa E722
        print("WARNING: Could not calculate ADP of the priv. Gauss in Logspace")
        print("WARNING: Fallback to normal space computation")
    return calculate_normal_space_adp_priv_gauss_multiplier_in_normal_space(mu, sigma, eps)


def calculate_composed_pb(A, B, range_begin, number_of_compositions, eps, buckets=250000, factor=1.000001):
    """Calculates the composed PrivacyBuckets PLD

    Args:
        A, B ([double]): Input distributions
        range_begin (int): Range begin of [-range_begin, range_begin]
        eps (double): Target epsilon
        number_of_compositions (int): Number of compositions => needs to be a power of two
        buckets (int, optional): How many buckets shall be used. Defaults to 250000.
        factor (float, optional): Factor for Probability Buckets. Defaults to 1.000001.

    Returns:
        (PrivacyBuckets): Composed PrivacyBuckets
    """
    privacybuckets = PrivacyBuckets(
        number_of_buckets=buckets,
        factor=utils.get_good_factor(factor, eps, buckets // 2),
        dist1_array=A,  # distribution A
        dist2_array=B,  # distribution B
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
    )
    return privacybuckets.compose(number_of_compositions)


def calculate_exact_upper_lower_delta_with_pb(A, B, range_begin, eps, number_of_compositions, buckets=250000, factor=1.000001):
    """Calculates exact upper and lower bounds for delta using Probability Buckets

    Args:
        A, B ([double]): Input distributions
        range_begin (int): Range begin of [-range_begin, range_begin]
        eps (double): Target epsilon
        number_of_compositions (int): Number of compositions => needs to be a power of two
        buckets (int, optional): How many buckets shall be used. Defaults to 250000.
        factor (float, optional): Factor for Probability Buckets. Defaults to 1.000001.

    Returns:
        (double, double): lower <= delta <= upper
    """

    privacybuckets_composed = calculate_composed_pb(A, B, range_begin, number_of_compositions, eps, buckets, factor)

    # Print status summary
    privacybuckets_composed.print_state()

    upper_bound = privacybuckets_composed.delta_ADP_upper_bound(eps)
    lower_bound = privacybuckets_composed.delta_ADP_lower_bound(eps)
    return lower_bound, upper_bound


def calculate_exact_upper_lower_delta_with_pb_pdp(A, B, range_begin, eps, number_of_compositions, buckets=250000, factor=1.000001):
    """Calculates exact upper and lower bounds for delta using Probability Buckets PDP

    Args:
        A, B ([double]): Input distributions
        range_begin (int): Range begin of [-range_begin, range_begin]
        eps (double): Target epsilon
        number_of_compositions (int): Number of compositions => needs to be a power of two
        buckets (int, optional): How many buckets shall be used. Defaults to 250000.
        factor (float, optional): Factor for Probability Buckets. Defaults to 1.000001.

    Returns:
        (double, double): lower <= delta <= upper
    """

    privacybuckets = PrivacyBuckets(
        number_of_buckets=buckets,
        factor=utils.get_good_factor(factor, eps, buckets // 2),
        dist1_array=A,  # distribution A
        dist2_array=B,  # distribution B
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
    )
    privacybuckets_composed = privacybuckets.compose(number_of_compositions)

    # Print status summary
    privacybuckets_composed.print_state()

    upper_bound = privacybuckets_composed.delta_PDP_upper_bound(eps)
    lower_bound = privacybuckets_composed.delta_PDP_lower_bound(eps)
    return lower_bound, upper_bound


def calculate_a_b(noise, range_begin, noise_class, q, sensitivity=1):
    """Calculates the A and B for a noise distribution

    Args:
        noise ([double]): Noise
        range_begin (int): Begin of range of the noise

    Returns:
        (A, B): A and B of the noise distribution
    """
    # Calculate the entries we have to shift for "1"
    delta = int(len(noise) / (2 * range_begin)) * sensitivity
    if noise_class == "SymmetricNoise":
        A = np.concatenate((noise, np.zeros(delta)))
        B = np.concatenate((np.zeros(delta), noise))
    elif noise_class == "MixtureNoise":
        A = np.concatenate((noise, np.zeros(delta)))
        B = np.concatenate((noise, np.zeros(delta)))

        B = (1.0 - q) * B
        B[-len(noise) :] += q * noise
        B /= np.sum(B)
    else:
        raise Exception("Noise Class not supported")

    return A, B


def calculate_adp_priv_gauss(markov_pdp_delta, preparer, x_coords, range_begin, args):
    """Calculates the ADP of a Priv. Gauss distribution

    Args:
        markov_pdp_delta (torch.DoubleTensor): delta + dist_events
        x_coords ([double]): Input X-coordinates
        range_begin (int): Begin of range of the x_coords
        args (Arguments): Program arguments

    Returns:
        double: ADP of the priv. Gauss distribution
    """
    # Calculate Gauss distribution
    mu = 1
    sigma = optimal_sigma_gauss_PDP(mu, args.number_of_compositions, args.eps, markov_pdp_delta.cpu().detach().numpy())
    gauss = stats.norm.pdf(x_coords, loc=0, scale=sigma)
    gauss /= sum(gauss)

    # Calculate distinguishing events from gaussian distribution
    _, _, dist_events, dist_events_dual, _, _ = preparer(torch.tensor(gauss))
    dist_events = dist_events + dist_events_dual
    dist_events = dist_events.detach().cpu().numpy()

    # Apply Lemma 12 from https://eprint.iacr.org/2018/820.pdf Privacy Loss Classes: The Central Limit Theorem in Differential Privacy
    return dist_events + ((1 - dist_events) / 2) * calculate_normal_space_adp_priv_gauss_multiplier_in_log_space(mu, sigma, args.eps)


# ADP -> delta(eps)
def calculate_adp_delta(A, B, range_begin, eps):
    """Calculates the ADP-Delta for a noise distribution

    Args:
        noise ([double]): Noise
        range_begin (int): Begin of range of the noise
        eps (double): Epsilon privacy we want to reach

    Returns:
        double: ADP delta of the input noise
    """
    # Calculate the entries we have to shift for "1"

    A_neq_0 = A[np.logical_and(A != 0, B != 0)]
    B_neq_0 = B[np.logical_and(A != 0, B != 0)]

    A_B_eq_0 = A[B == 0]

    return np.sum(np.maximum(A_neq_0 - np.exp(eps) * B_neq_0, 0)) + np.sum(A_B_eq_0)


def calculate_plc(A, B):
    """Calculates the privacy loss class (mu, sigma2, infty bucket) as defined by Definition 10 in https://eprint.iacr.org/2018/820.pdf

    Args:
        A ([double]): Input (has zeros appended on the right side)
        B ([double]): Shifted input by 1 in real space (has zeros appended on the left side)

    Returns:
        [(double, double, double)]: The privacy loss class
    """
    valid_mask = np.all(np.vstack((A != 0, B != 0)), axis=0)
    assert len(valid_mask) == len(A)

    A_valid = A[valid_mask]
    B_valid = B[valid_mask]

    inner_A = A_valid / np.sum(A_valid)  # normalize

    infty_bucket = np.sum(A[np.logical_not(valid_mask)])

    log_A = np.log(A_valid)
    log_B = np.log(B_valid)

    privacy_loss = log_A - log_B

    mean = np.sum(inner_A * privacy_loss)
    var = np.sum(inner_A * ((privacy_loss - mean) ** 2))

    return mean, var, infty_bucket


def get_sigma_of_similar_gaussian(A, B, mu=1):
    """Calculates the sigma of the (infinite) Gaussian that converges against the
    same privacy loss distribution as AB does under composition,

    !!!!! IMPORTANT ignoring dist-events

    Args:
        A ([double]): Input (has zeros appended on the right side)
        B ([double]): Shifted input by 1 in real space (has zeros appended on the left side)
        mu (int, optional): mu. Defaults to 1.

    Returns:
        [double]: Sigma of the (infinite) Gaussian
    """

    mu_pld, var_pld, infty_bucket = calculate_plc(A, B)

    print(f"Warning: mu_pld / 2 sigma_pld**2 = {2 * mu_pld / var_pld} [ should be 1 ]")

    sigma2 = mu ** 2 / (2 * mu_pld)

    return np.sqrt(sigma2)


# ADP <= PDP <= Markov
def calculate_pdp_delta(A, B, range_begin, eps):
    """Calculates the PDP-Delta for a noise distribution

    Args:
        noise ([double]): Noise
        range_begin (int): Begin of range of the noise
        eps (double): Epsilon privacy we want to reach

    Returns:
        double: PDP delta of the input noise
    """
    # Calculate the entries we have to shift for "1"
    return np.sum(A[A > B * np.exp(eps)]) + np.sum(A[B == 0])


def optimal_sigma_gauss_PDP(mu, n, eps, PDP_delta):
    """
    For a given (eps, delta)-PDP guarantees, this function computes the optimal sigma for
    Gaussian additive noise with mean-difference mu, after n composiitons.

    Args:
        mu (double): Mean
        n (double): Number of compositions
        eps (double): Epsilon
        PDP_delta (double): Delta

    Returns:
        [double]: Optimal sigma for a gaussian distribution
    """

    res = mu * np.sqrt(n) / (np.sqrt(2) * eps)

    mult = special.erfcinv(2 * PDP_delta) + np.sqrt(np.square(special.erfcinv(2 * PDP_delta)) + eps)
    res = res * mult
    return res


def fit_gaussian(predicted, x_coords):
    """Calculates a fitted gaussian distribution

    Args:
        predicted ([double]): Predicted noise
        x_coords ([double]): Coordinates

    Returns:
        ([double], double, double): Returns a fitted gaussian distribution together with its mean and variance
    """
    p = predicted
    mean = np.dot(p, x_coords)
    var = np.dot(p, (x_coords - mean) ** 2)

    fitted_gaussian = stats.norm.pdf(x_coords, loc=mean, scale=np.sqrt(var))
    fitted_gaussian /= np.sum(fitted_gaussian)

    return fitted_gaussian, mean, var
