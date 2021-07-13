# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes delta of a truncated gaussian for a given sigma
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os  # noqa:E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import privacy_utils  # noqa: E402
import scipy.stats as stats  # noqa: E402
from privacybuckets import PrivacyBuckets  # noqa: E402


def compute_delta(sigma, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps):
    """Computes delta for a truncated gaussian

    Args:
        sigma (Float): Sigma of the truncated gaussian
        x (nd.array): X-axis discretization
        range (Integer): Range of the noise
        number_of_compositions (Integer): Number of Compositions
        sensitivity (Integer): Sensitivity of the query
        factor (Float): Privacy Bucket factor
        number_of_buckets (Integer): Number of buckets for Privacy Buckets
        eps (Float): Target epsilon

    Returns:
        Floats: upper and lower bounds
    """
    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
    pdf /= np.sum(pdf)

    A, B = privacy_utils.calculate_a_b(pdf, range, "MixtureNoise", 0.1, sensitivity)

    # B/A
    privacybuckets = PrivacyBuckets(
        number_of_buckets=number_of_buckets,
        factor=factor,
        dist1_array=B,  # distribution A
        dist2_array=A,  # distribution B
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        error_correction=True,  # error correction. See publication for details
    )
    privacybuckets_composed = privacybuckets.compose(number_of_compositions)

    delta_upper_b_a = privacybuckets_composed.delta_ADP_upper_bound(eps)
    delta_lower_b_a = privacybuckets_composed.delta_ADP_lower_bound(eps)

    # A/B
    privacybuckets = PrivacyBuckets(
        number_of_buckets=number_of_buckets,
        factor=factor,
        dist1_array=A,  # distribution A
        dist2_array=B,  # distribution B
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        error_correction=True,  # error correction. See publication for details
    )
    privacybuckets_composed = privacybuckets.compose(number_of_compositions)

    delta_upper_a_b = privacybuckets_composed.delta_ADP_upper_bound(eps)
    delta_lower_a_b = privacybuckets_composed.delta_ADP_lower_bound(eps)
    return delta_lower_b_a, delta_upper_b_a, delta_lower_a_b, delta_upper_a_b


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please run as: python3 generator_delta_for_std_csv.py {folder}")
        exit(1)

    # Parameters to change
    sample = 60000
    range = 500
    x = np.linspace(-range, range, num=sample) + 10 ** -5
    sensitivity = 1
    factor = 1.000001
    number_of_buckets = 250000
    eps = 0.3
    folder = sys.argv[1]

    for root, directory, files in os.walk(folder, topdown=False):
        for name in [f for f in files if ".csv" in f]:
            sigmas = pd.read_csv(os.path.join(root, name))
            for idx, row in sigmas.iterrows():
                l_b_a, u_b_a, l_a_b, u_a_b = compute_delta(float(row["Sigma"]), x, range, int(row["Compositions"]), sensitivity, factor, number_of_buckets, eps)
                sigmas.at[idx, "Delta_Lower_B_A"] = l_b_a
                sigmas.at[idx, "Delta_Upper_B_A"] = u_b_a
                sigmas.at[idx, "Delta_Lower_A_B"] = l_a_b
                sigmas.at[idx, "Delta_Upper_A_B"] = u_a_b

                # Add Utility of found std
                pdf = stats.norm.pdf(x, loc=0, scale=float(row["Sigma"]))
                pdf /= np.sum(pdf)
                sigmas.at[idx, "Utility_L2"] = np.sqrt(np.dot(x ** 2, pdf))
                sigmas.at[idx, "Utility_L1"] = np.dot(np.abs(x), pdf)

            sigmas.to_csv(os.path.join(root, name))
