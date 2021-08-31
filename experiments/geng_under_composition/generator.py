# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes delta of a truncated gaussian for a given sigma
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import csv  # noqa:E402
import os  # noqa:E402
import logging  # noqa:E402

import numpy as np  # noqa: E402
import privacy_utils  # noqa: E402
import scipy.stats as stats  # noqa: E402
from scipy import optimize  # noqa: E402
import utils  # noqa:E402


def write_delta_utility(
    filename,
    compositions,
    deltas_lower,
    deltas_upper,
    utilities_l1,
    utilities_l2,
    dist_events,
    dist_events_dual,
    infty_buckets,
    sigma_l1,
    deltas_lower_l1,
    deltas_upper_l1,
    utilities_l1_l1,
    utilities_l2_l1,
    dist_events_l1,
    dist_events_dual_l1,
    infty_buckets_l1,
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
                "InftyBucket",
                "Sigma_L1",
                "Lower_L1",
                "Upper_L1",
                "Utility_L1ofL1",
                "Utility_L2ofL1",
                "Dist_events_L1",
                "Dist_events_dual_L1",
                "InftyBucket_L1",
            ]
        )

        writer.writerows(
            list(
                zip(
                    compositions,
                    deltas_lower,
                    deltas_upper,
                    utilities_l1,
                    utilities_l2,
                    dist_events,
                    dist_events_dual,
                    infty_buckets,
                    sigma_l1,
                    deltas_lower_l1,
                    deltas_upper_l1,
                    utilities_l1_l1,
                    utilities_l2_l1,
                    dist_events_l1,
                    dist_events_dual_l1,
                    infty_buckets_l1,
                )
            )
        )


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


def compute_delta(pdf, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps):
    A, B = privacy_utils.calculate_a_b(pdf, range, "SymmetricNoise", 0, sensitivity)

    dist_events = np.sum(pdf[len(pdf) - int(len(pdf) / (2 * range)) :])
    dist_events_dual = np.sum(pdf[: (int(len(x) / (2 * range)))])

    privacybuckets_composed = privacy_utils.calculate_composed_pb(A, B, range, number_of_compositions, eps, number_of_buckets, factor)
    privacybuckets_composed.print_state()

    upper = privacybuckets_composed.delta_ADP_upper_bound(eps)
    lower = privacybuckets_composed.delta_ADP_lower_bound(eps)

    dist_events_comp = 1.0 - np.power((1.0 - dist_events), number_of_compositions)
    dist_events_comp_dual = 1.0 - np.power((1.0 - dist_events_dual), number_of_compositions)

    return lower, upper, dist_events_comp, dist_events_comp_dual, np.dot(np.abs(x), pdf), np.sqrt(np.dot(x ** 2, pdf)), privacybuckets_composed.infty_bucket


def geng_noise(x, a, delta, sensitivity):
    d = np.zeros_like(x)
    density = (delta - a) / sensitivity
    distance = ((1 - a) * sensitivity) / (2 * (delta - a))
    d[(x >= -distance) & (x <= distance)] = density
    d = d / np.sum(d)
    return d, int(distance)


if __name__ == "__main__":
    # Parameters to change
    tolerance = 10 ** -10
    results_dir = "./results_delta_for_different_q"
    sample = 60001
    x = np.linspace(-500, 500, num=sample)
    sensitivity = 1
    factor = 1.00001
    number_of_buckets = 250000
    epses = [0.1, 0.3, 1.0, 2.0]
    compositions = [1, 2, 4, 8, 16, 32, 64, 128]
    deltas = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    p_norms = [1, 2]
    sensitivity = 1

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for delta in deltas:
        for p in p_norms:
            a = (p + 1) * delta - p if delta > p / (p + 1) else 0
            # Geng with alpha = 0
            noise, distance = geng_noise(x, a, delta, sensitivity)

            # Gaussian with same delta
            try:
                sigma_gauss = optimize.bisect(f=to_optimize, a=5, b=12, xtol=tolerance, args=(delta, x, distance, 1, sensitivity, factor, number_of_buckets, 0))
            except Exception as e:
                print(f"{delta}, {p}, ")
                print(str(e))
                sigma_gauss = 0
            pdf = stats.norm.pdf(x, loc=0, scale=sigma_gauss)
            pdf /= np.sum(pdf)

            for eps in epses:
                deltas_lower, deltas_upper, nbs, utilities_l1, utilities_l2, dist_events, dist_events_dual, infty_buckets = [], [], [], [], [], [], [], []
                sigma_l1, deltas_lower_l1, deltas_upper_l1, utilities_l1_l1, utilities_l2_l1, dist_events_l1, dist_events_dual_l1, infty_buckets_l1 = [], [], [], [], [], [], [], []

                for nb in compositions:
                    number_of_compositions = nb
                    nbs.append(nb)

                    lower, upper, dist_events_comp, dist_events_comp_dual, ul1, ul2, infty_bucket = compute_delta(
                        noise, x, distance, number_of_compositions, sensitivity, factor, number_of_buckets, eps
                    )
                    deltas_lower.append(lower)
                    deltas_upper.append(upper)
                    dist_events.append(dist_events_comp)
                    dist_events_dual.append(dist_events_comp_dual)
                    utilities_l1.append(ul1)
                    utilities_l2.append(ul2)
                    infty_buckets.append(infty_bucket)

                    sigma_l1.append(sigma_gauss)
                    lower, upper, dist_events_comp, dist_events_comp_dual, ul1_, ul2_, infty_bucket = compute_delta(
                        pdf, x, distance, number_of_compositions, sensitivity, factor, number_of_buckets, eps
                    )
                    deltas_lower_l1.append(lower)
                    deltas_upper_l1.append(upper)
                    dist_events_l1.append(dist_events_comp)
                    dist_events_dual_l1.append(dist_events_comp_dual)
                    utilities_l1_l1.append(ul1_)
                    utilities_l2_l1.append(ul2_)
                    infty_buckets_l1.append(infty_bucket)

                # Save the result
                write_delta_utility(
                    os.path.join(results_dir, f"delta_for_delta_{delta}_eps_{eps}_p_norm_{p}.csv"),
                    nbs,
                    deltas_lower,
                    deltas_upper,
                    utilities_l1,
                    utilities_l2,
                    dist_events,
                    dist_events_dual,
                    infty_buckets,
                    sigma_l1,
                    deltas_lower_l1,
                    deltas_upper_l1,
                    utilities_l1_l1,
                    utilities_l2_l1,
                    dist_events_l1,
                    dist_events_dual_l1,
                    infty_buckets_l1,
                )
