# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes sigma of a truncated gaussian for a given delta
import sys

import numpy as np
import scipy.stats as stats
from scipy import optimize


def to_optimize(sigma, utility_we_want, x, uf):
    """Scipy optimization function. Optimizes for utility

    Args:
        sigma (float): Current sigma
        utility_we_want (float): utility we want to achieve
        x (nd.array): x coordinates
        uf (str): Either "l1" or "l2" utility loss

    """
    pdf = stats.norm.pdf(x, loc=0, scale=sigma)
    pdf /= np.sum(pdf)

    if uf == "l1":
        return np.dot(np.abs(x), pdf) - utility_we_want
    else:
        return np.sqrt(np.dot(x ** 2, pdf)) - utility_we_want


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("python3 generator_std_for_utility.py {compositions} {desired_utility} {range_to_search_start} {range_to_search_end} {utility_function}")
        exit(1)

    # Parameters to change
    tolerance = 10 ** -10
    sample = 60000
    range = 400
    compositions = int(sys.argv[1])

    utility_we_want = float(sys.argv[2])
    x = np.linspace(-range, range, num=sample) + 10 ** -5
    sensitivity = 1
    factor = 1.000001
    number_of_buckets = 250000
    eps = 0.3

    try:
        std = optimize.bisect(f=to_optimize, a=int(sys.argv[3]), b=int(sys.argv[4]), xtol=tolerance, args=(utility_we_want, x, sys.argv[5]))
    except Exception as e:
        print(str(e))
        std = float("Inf")

    pdf = stats.norm.pdf(x, loc=0, scale=std)
    pdf /= np.sum(pdf)
    print(f"Std: {std}")
    print(f"L Utility want: {utility_we_want}")
    print(f"L1 Utility found: {np.dot(np.abs(x), pdf)}")
    print(f"L2 Utility found: {np.sqrt(np.dot(x ** 2, pdf))}")
