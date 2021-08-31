# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes delta of a truncated laplace for a given scale
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import csv  # noqa:E402
import os  # noqa:E402

import numpy as np  # noqa: E402
import privacy_utils  # noqa: E402
import scipy.stats as stats  # noqa: E402


def write_delta_utility(filename, std, deltas_lower, deltas_upper, utility_l1, utility_l2):
    """Writes sigma, delta and utilities to a CSV file

    Args:
        filename (String): Filename to write to
        std ([Float]): List of sigmas of the truncated gaussian
        delta ([Float]): List of corresponding deltas
        utility_l1 ([Float]): List of corresponding L1 utilities
        utility_l2 ([Float]): List of corresponding L2 utilities
    """
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Std", "Lower", "Upper", "Utility_L1", "Utility_L2"])
        writer.writerows(list(zip(std, deltas_lower, deltas_upper, utility_l1, utility_l2)))


def compute_delta(sigma, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps):
    """Computes delta for a truncated laplace

    Args:
        sigma (Float): Sigma of the truncated laplace
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
    pdf = stats.laplace.pdf(x, loc=0, scale=sigma)
    pdf /= np.sum(pdf)

    A, B = privacy_utils.calculate_a_b(pdf, range, "SymmetricNoise", 0, sensitivity)
    lower, upper = privacy_utils.calculate_exact_upper_lower_delta_with_pb(A, B, range, eps, number_of_compositions)
    return lower, upper


if __name__ == "__main__":
    # Parameters to change
    results_dir = "./results_delta_for_stddev"
    tolerance = 10 ** -10
    sample = 10 ** 6
    range = 400
    x = np.linspace(-range, range, num=sample)
    sensitivity = 1
    factor = 1.00001
    number_of_buckets = 250000
    eps = 0.3
    compositions = [1, 2, 4, 8, 16, 32, 64, 128]
    stddevs = np.logspace(np.log(0.001), np.log(1000), 150, base=np.e)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # For all compositions
    for nb in compositions:
        number_of_compositions = nb
        deltas_lower, deltas_upper, stds, utilities_l1, utilities_l2 = [], [], [], [], []
        # Find the corresponding std for delta
        for sigma in stddevs:
            lower, upper = compute_delta(sigma, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps)
            deltas_lower.append(lower)
            deltas_upper.append(upper)
            stds.append(sigma)

            # Add Utility of found std
            pdf = stats.laplace.pdf(x, loc=0, scale=sigma)
            pdf /= np.sum(pdf)
            utilities_l1.append(np.dot(np.abs(x), pdf))
            utilities_l2.append(np.sqrt(np.dot(x ** 2, pdf)))

        # Save the result
        write_delta_utility(os.path.join(results_dir, f"std_delta_for_number_compositions_{nb}.csv"), stds, deltas_lower, deltas_upper, utilities_l1, utilities_l2)
