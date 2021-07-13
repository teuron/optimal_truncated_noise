# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes epsilon-delta graph from sigma
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os  # noqa: 402

import numpy as np  # noqa: 402
import pandas as pd  # noqa: 402
import privacy_utils  # noqa: 402
import scipy.stats as stats  # noqa: E402
import utils  # noqa: 402
from privacybuckets import PrivacyBuckets  # noqa: 402


def generate(file, noise, epses, range_begin, number_of_compositions):
    """Generates deltas for different epsilons

    Args:
        file (String): Filename to write result to
        noise (nd.array): Input noise
        epses ([Float]): Input epsilons to compute delta for
        range_begin (Integer): Range of the noise
    """
    deltas_u, deltas_l, dist_events = [], [], []
    A, B = privacy_utils.calculate_a_b(noise, range_begin, "MixtureNoise", 0.1)
    privacybuckets_composed = privacy_utils.calculate_composed_pb(A, B, range_begin, number_of_compositions, 0.3)

    for eps in epses:
        deltas_l.append(privacybuckets_composed.delta_ADP_lower_bound(eps))
        deltas_u.append(privacybuckets_composed.delta_ADP_upper_bound(eps))
        dist_events.append(1.0 - np.power((1.0 - np.sum(A[B == 0])), number_of_compositions))
    utils.write_epsilon_delta(file + "a_b.csv", epses, deltas_u, deltas_l, dist_events)

    deltas_u, deltas_l, dist_events = [], [], []
    privacybuckets_composed = privacy_utils.calculate_composed_pb(B, A, range_begin, number_of_compositions, 0.3)

    for eps in epses:
        deltas_l.append(privacybuckets_composed.delta_ADP_lower_bound(eps))
        deltas_u.append(privacybuckets_composed.delta_ADP_upper_bound(eps))
        dist_events.append(1.0 - np.power((1.0 - np.sum(B[A == 0])), number_of_compositions))
    utils.write_epsilon_delta(file + "b_a.csv", epses, deltas_u, deltas_l, dist_events)


def main():
    np.set_printoptions(threshold=sys.maxsize)

    if len(sys.argv) != 5:
        print("python3 epsilon_delta_graph_from_sigma.py range sigma output.csv compositions")
        exit(1)

    # Create plot directory
    results_dir = "./results_epsilon_delta"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set the right range for the input!
    range_begin = int(sys.argv[1])
    sigma = float(sys.argv[2])
    x = np.linspace(-range_begin, range_begin, num=60000) + 10 ** -5

    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
    noise = pdf / np.sum(pdf)

    epses = np.linspace(0.1, 2, 3000)
    generate(results_dir + "/" + sys.argv[3], noise, epses, range_begin, int(sys.argv[4]))


if __name__ == "__main__":
    main()
