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


def write_delta_utility(filename, compositions, deltas_lower, deltas_upper, utility_l1, utility_l2, dist_events, dist_events_dual):
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Compositions", "Lower", "Upper", "Utility_L1", "Utility_L2", "Dist_events(A/B)", "Dist_events_dual(B/A)"])
        writer.writerows(list(zip(compositions, deltas_lower, deltas_upper, utility_l1, utility_l2, dist_events, dist_events_dual)))


def compute_delta(sigma, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps, q):
    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
    pdf /= np.sum(pdf)

    A, B = privacy_utils.calculate_a_b(pdf, range, "MixtureNoise", q, sensitivity)
    
    dist_events = 0
    dist_events_dual = np.sum(B[-(int(len(x) / (2 * range))):])

    lower, upper = privacy_utils.calculate_exact_upper_lower_delta_with_pb(A, B, range, eps, number_of_compositions)
    dist_events_comp = 1.0 - np.power((1.0 - dist_events), number_of_compositions)
    dist_events_comp_dual = 1.0 - np.power((1.0 - dist_events_dual), number_of_compositions)
    
    return lower, upper, dist_events_comp, dist_events_comp_dual


if __name__ == "__main__":
    # Parameters to change
    results_dir = "./results_delta_for_different_q"
    sample = 10 ** 6
    range = 400
    x = np.linspace(-range, range, num=sample)
    sensitivity = 1
    factor = 1.00001
    number_of_buckets = 250000
    epses = [0.1, 0.3, 1.0, 2.0]
    compositions = [1, 2, 4, 8, 16, 32, 64, 128]
    qs = np.logspace(np.log(10**-6), np.log(0.1), 6, base=np.e)
    stddevs = np.logspace(np.log(10), np.log(100), 10, base=np.e)
    print(qs, stddevs)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for eps in epses:
        for sigma in stddevs:
            for q in qs:
                deltas_lower, deltas_upper, nbs, utilities_l1, utilities_l2, dist_events, dist_events_dual = [], [], [], [], [], [], []
                for nb in compositions:
                    number_of_compositions = nb
                    nbs.append(nb)
                    lower, upper, dist_events_comp, dist_events_comp_dual = compute_delta(sigma, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps, q)
                    deltas_lower.append(lower)
                    deltas_upper.append(upper)
                    dist_events.append(dist_events_comp)
                    dist_events_dual.append(dist_events_comp_dual)
                    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
                    pdf /= np.sum(pdf)
                    utilities_l1.append(np.dot(np.abs(x), pdf))
                    utilities_l2.append(np.sqrt(np.dot(x ** 2, pdf)))

                # Save the result
                write_delta_utility(os.path.join(results_dir, f"delta_for_q_{q}_sigma_{sigma}_eps_{eps}.csv"), nbs, deltas_lower, deltas_upper, utilities_l1, utilities_l2, dist_events, dist_events_dual)
