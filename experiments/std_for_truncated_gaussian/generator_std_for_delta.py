# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes sigma of a truncated gaussian for a given delta and range
import sys


sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os  # noqa:E402
import logging  # noqa:E402

import numpy as np  # noqa: E402
import privacy_utils  # noqa: E402
import utils  # noqa: E402
import scipy.stats as stats  # noqa: E402
from privacybuckets import PrivacyBuckets  # noqa: E402
from scipy import optimize  # noqa: E402


def to_optimize(sigma, delta_we_want, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps):
    """Optimization function

    Args:
        sigma (Float): Sigma of the truncated gaussian
        delta_we_want (Float): Delta we want to archieve
        x (nd.array): X-axis discretization
        range (Integer): Range of the noise
        number_of_compositions (Integer): Number of Compositions
        sensitivity (Integer): Sensitivity of the query
        factor (Float): Privacy Bucket factor
        number_of_buckets (Integer): Number of buckets for Privacy Buckets
        eps (Float): Target epsilon

    Returns:
        Float: delta-delta_we_want
    """
    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
    pdf /= np.sum(pdf)

    A, B = privacy_utils.calculate_a_b(pdf, range, "MixtureNoise", 0.1, sensitivity)
    privacybuckets = PrivacyBuckets(
        number_of_buckets=number_of_buckets,
        factor=utils.get_good_factor(factor, eps, number_of_buckets // 2),
        logging_level=logging.CRITICAL,
        dist1_array=B,  # distribution A
        dist2_array=A,  # distribution B
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        error_correction=True,  # error correction. See publication for details
    )
    privacybuckets_composed = privacybuckets.compose(number_of_compositions)

    delta = privacybuckets_composed.delta_ADP_upper_bound(eps)
    delta_l = privacybuckets_composed.delta_ADP_lower_bound(eps)
    print(f"RUN B/A sigma {sigma}, delta {delta}, lower {delta_l}")
    return delta - delta_we_want


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python3 generator_std_for_delta.py {range} {delta} {compositions}")
        exit(1)

    # Parameters to change
    results_dir = "./results"
    tolerance = 10 ** -10
    sample = 60000
    range = int(sys.argv[1])
    x = np.linspace(-range, range, num=sample) + 10 ** -5
    sensitivity = 1
    factor = 1.00001
    number_of_buckets = 250000
    eps = 0.3
    compositions = int(sys.argv[3])
    delta_we_want = float(sys.argv[2])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    try:
        std = optimize.bisect(f=to_optimize, a=5, b=12, xtol=tolerance, args=(delta_we_want, x, range, compositions, sensitivity, factor, number_of_buckets, eps))
    except Exception as e:
        print(str(e))
        exit(1)

    print(f"Sigma {std}")
    pdf = stats.norm.pdf(x, loc=0, scale=std)
    pdf /= np.sum(pdf)

    print(f"Utility L1 {np.dot(np.abs(x), pdf)}")
    print(f"Utility L2 {np.sqrt(np.dot(x ** 2, pdf))}")

    A, B = privacy_utils.calculate_a_b(pdf, range, "MixtureNoise", 0.1, sensitivity)
    privacybuckets = PrivacyBuckets(
        number_of_buckets=number_of_buckets,
        factor=utils.get_good_factor(factor, eps, number_of_buckets // 2),
        logging_level=logging.CRITICAL,
        dist1_array=A,  # distribution A
        dist2_array=B,  # distribution B
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        error_correction=True,  # error correction. See publication for details
    )
    privacybuckets_composed = privacybuckets.compose(compositions)

    delta = privacybuckets_composed.delta_ADP_upper_bound(eps)
    delta_l = privacybuckets_composed.delta_ADP_lower_bound(eps)
    print(f"RUN A/B comps {compositions}, delta {delta}, lower {delta_l}")
