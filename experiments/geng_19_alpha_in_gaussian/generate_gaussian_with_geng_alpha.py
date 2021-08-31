# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes delta of a truncated gaussian for a given sigma
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import csv  # noqa:E402
import os  # noqa:E402

import numpy as np  # noqa: E402
import privacy_utils  # noqa: E402
import scipy.stats as stats  # noqa: E402
from scipy import optimize  # noqa: E402


def write_delta_utility(
    filename,
    compositions,
    deltas_lower,
    deltas_upper,
    utility_l1,
    utility_l2,
    dist_events,
    dist_events_dual,
    sigma_l1,
    deltas_lower_l1,
    deltas_upper_l1,
    utilities_l1_l1,
    utilities_l2_l1,
    dist_events_l1,
    dist_events_dual_l1,
    sigma_l2,
    deltas_lower_l2,
    deltas_upper_l2,
    utilities_l1_l2,
    utilities_l2_l2,
    dist_events_l2,
    dist_events_dual_l2,
):
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "Compositions",
                "Lower",
                "Upper",
                "Utility_L1",
                "Utility_L2",
                "Dist_events(A/B)",
                "Dist_events_dual(B/A)",
                "Sigma_L1",
                "Lower_L1",
                "Upper_L1",
                "Utility_L1ofL1",
                "Utility_L2ofL1",
                "Dist_events_L1",
                "Dist_events_dual_L1",
                "Sigma_L2",
                "Lower_L2",
                "Upper_L2",
                "Utility_L1ofL2",
                "Utility_L2ofL2",
                "Dist_events_L2",
                "Dist_events_dual_L2",
            ]
        )

        writer.writerows(
            list(
                zip(
                    compositions,
                    deltas_lower,
                    deltas_upper,
                    utility_l1,
                    utility_l2,
                    dist_events,
                    dist_events_dual,
                    sigma_l1,
                    deltas_lower_l1,
                    deltas_upper_l1,
                    utilities_l1_l1,
                    utilities_l2_l1,
                    dist_events_l1,
                    dist_events_dual_l1,
                    sigma_l2,
                    deltas_lower_l2,
                    deltas_upper_l2,
                    utilities_l1_l2,
                    utilities_l2_l2,
                    dist_events_l2,
                    dist_events_dual_l2,
                )
            )
        )


def to_optimize(sigma, utility_we_want, x, uf):
    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
    pdf /= np.sum(pdf)

    if uf == "l1":
        return np.dot(np.abs(x), pdf) - utility_we_want
    else:
        return np.sqrt(np.dot(x ** 2, pdf)) - utility_we_want


def compute_delta(sigma, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps, alpha):
    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
    pdf /= np.sum(pdf)
    pdf *= 1.0 - alpha
    pdf[30000] += alpha

    A, B = privacy_utils.calculate_a_b(pdf, range, "SymmetricNoise", 0, sensitivity)

    dist_events = np.sum(pdf[len(pdf) - int(len(pdf) / (2 * range)) :])
    dist_events_dual = np.sum(pdf[: (int(len(x) / (2 * range)))])

    lower, upper = privacy_utils.calculate_exact_upper_lower_delta_with_pb(A, B, range, eps, number_of_compositions)
    dist_events_comp = 1.0 - np.power((1.0 - dist_events), number_of_compositions)
    dist_events_comp_dual = 1.0 - np.power((1.0 - dist_events_dual), number_of_compositions)

    return lower, upper, dist_events_comp, dist_events_comp_dual, np.dot(np.abs(x), pdf), np.sqrt(np.dot(x ** 2, pdf))


if __name__ == "__main__":
    # Parameters to change
    tolerance = 10 ** -10
    results_dir = "./results_delta_for_different_q"
    sample = 60001
    range = 400
    x = np.linspace(-range, range, num=sample)
    sensitivity = 1
    factor = 1.00001
    number_of_buckets = 250000
    epses = [0.1, 0.3, 1.0, 2.0]
    compositions = [1, 2, 4, 8, 16, 32, 64, 128]
    alphas = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 0.5]
    stddevs = np.logspace(np.log(10), np.log(100), 10, base=np.e)
    print(alphas, stddevs)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for eps in epses:
        for sigma in stddevs:
            for alpha in alphas:
                deltas_lower, deltas_upper, nbs, utilities_l1, utilities_l2, dist_events, dist_events_dual = [], [], [], [], [], [], []
                sigma_l1, deltas_lower_l1, deltas_upper_l1, utilities_l1_l1, utilities_l2_l1, dist_events_l1, dist_events_dual_l1 = [], [], [], [], [], [], []
                sigma_l2, deltas_lower_l2, deltas_upper_l2, utilities_l1_l2, utilities_l2_l2, dist_events_l2, dist_events_dual_l2 = [], [], [], [], [], [], []
                for nb in compositions:
                    number_of_compositions = nb
                    nbs.append(nb)

                    # Gaussian with Geng alpha added
                    lower, upper, dist_events_comp, dist_events_comp_dual, ul1, ul2 = compute_delta(
                        sigma, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps, alpha
                    )
                    deltas_lower.append(lower)
                    deltas_upper.append(upper)
                    dist_events.append(dist_events_comp)
                    dist_events_dual.append(dist_events_comp_dual)
                    utilities_l1.append(ul1)
                    utilities_l2.append(ul2)

                    # Gaussian with same L1 Utility
                    try:
                        std = optimize.bisect(f=to_optimize, a=5, b=150, xtol=tolerance, args=(ul1, x, "ul1"))
                    except Exception as e:
                        print(ul1)
                        print(str(e))
                        std = 0
                    sigma_l1.append(std)
                    lower, upper, dist_events_comp, dist_events_comp_dual, ul1_, ul2_ = compute_delta(
                        std, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps, 0.0
                    )
                    deltas_lower_l1.append(lower)
                    deltas_upper_l1.append(upper)
                    dist_events_l1.append(dist_events_comp)
                    dist_events_dual_l1.append(dist_events_comp_dual)
                    utilities_l1_l1.append(ul1_)
                    utilities_l2_l1.append(ul2_)

                    # Gaussian with same L2 Utility
                    try:
                        std = optimize.bisect(f=to_optimize, a=5, b=150, xtol=tolerance, args=(ul2, x, "ul2"))
                    except Exception as e:
                        print(ul2)
                        print(str(e))
                        std = 0.0
                    sigma_l2.append(std)
                    lower, upper, dist_events_comp, dist_events_comp_dual, ul1, ul2 = compute_delta(
                        std, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps, 0.0
                    )
                    deltas_lower_l2.append(lower)
                    deltas_upper_l2.append(upper)
                    dist_events_l2.append(dist_events_comp)
                    dist_events_dual_l2.append(dist_events_comp_dual)
                    utilities_l1_l2.append(ul1)
                    utilities_l2_l2.append(ul2)

                # Save the result
                write_delta_utility(
                    os.path.join(results_dir, f"delta_for_alpha_{alpha}_sigma_{sigma}_eps_{eps}.csv"),
                    nbs,
                    deltas_lower,
                    deltas_upper,
                    utilities_l1,
                    utilities_l2,
                    dist_events,
                    dist_events_dual,
                    sigma_l1,
                    deltas_lower_l1,
                    deltas_upper_l1,
                    utilities_l1_l1,
                    utilities_l2_l1,
                    dist_events_l1,
                    dist_events_dual_l1,
                    sigma_l2,
                    deltas_lower_l2,
                    deltas_upper_l2,
                    utilities_l1_l2,
                    utilities_l2_l2,
                    dist_events_l2,
                    dist_events_dual_l2,
                )
