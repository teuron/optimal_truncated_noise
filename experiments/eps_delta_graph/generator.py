# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes epsilon-delta graph from input noise
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import csv  # noqa: 402
import os  # noqa: 402

import numpy as np  # noqa: 402
import pandas as pd  # noqa: 402
import privacy_utils  # noqa: 402
import utils  # noqa: 402


def generate(file, noise, delta_func, epses, range_begin):
    """Generates deltas for different epsilons

    Args:
        file (String): Filename to write result to
        noise (nd.array): Input noise
        delta_func (Function): Function that computes delta(eps) from the noise
        epses ([Float]): Input epsilons to compute delta for
        range_begin (Integer): Range of the noise

    Returns:
        [([Float], [Float])]: Tuple of resulting deltas and dist events
    """
    deltas, dist_events = [], []
    A, B = privacy_utils.calculate_a_b(noise, range_begin, "SymmetricNoise", 0)
    for eps in epses:
        deltas.append(delta_func(A, B, range_begin, eps))
        dist_events.append(np.sum(A[B == 0]))
    utils.write_epsilon_delta(file, epses, deltas, deltas, dist_events)
    return deltas, dist_events


def main():
    if len(sys.argv != 3):
        print("python3 generator.py {range} {file.csv}")
        exit(1)

    np.set_printoptions(threshold=sys.maxsize)

    # Create plot directory
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set the right range for the input!
    range_begin = sys.argv[1]

    # Load Noise and calculate optimal noise from it
    noise = pd.read_csv(sys.argv[2])["Y"].to_numpy()
    x_coords = pd.read_csv(sys.argv[2])["X"].to_numpy()

    optimal_noise = privacy_utils.get_theoretical_optimal_noise_x(x=abs(x_coords), eps=0.3, number_of_compositions=1)

    epses = np.linspace(0.0, 0.5, 3000)
    deltas1, dist_events1 = generate(results_dir + "/epsilon_delta_noise.csv", noise, privacy_utils.calculate_adp_delta, epses, range_begin)
    deltas2, dist_events2 = generate(results_dir + "/epsilon_delta_optimal_noise.csv", optimal_noise, privacy_utils.calculate_adp_delta, epses, range_begin)

    diff = [d2 - d1 for d1, d2 in zip(deltas1, deltas2)]
    dists = [d2 - d1 for d1, d2 in zip(dist_events1, dist_events2)]
    utils.write_epsilon_delta(results_dir + "/epsilon_delta_difference.csv", epses, diff, diff, dists)


if __name__ == "__main__":
    main()
