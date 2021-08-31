# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes upper and lower bounds for delta_{A/B, B/A} and the utility loss
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")


import os  # noqa: 402
import re  # noqa: 402

import numpy as np  # noqa: 402
import pandas as pd  # noqa: 402
import privacy_utils  # noqa: E402


def compute_delta(noise, x, range, number_of_compositions, sensitivity, factor, number_of_buckets, eps):
    A, B = privacy_utils.calculate_a_b(noise, range, "MixtureNoise", 0.1, sensitivity)

    delta_lower_b_a, delta_upper_b_a = privacy_utils.calculate_exact_upper_lower_delta_with_pb(B, A, range, eps, number_of_compositions, number_of_buckets, factor)
    delta_lower_a_b, delta_upper_a_b = privacy_utils.calculate_exact_upper_lower_delta_with_pb(A, B, range, eps, number_of_compositions, number_of_buckets, factor)
    return delta_lower_b_a, delta_upper_b_a, delta_lower_a_b, delta_upper_a_b


if len(sys.argv) != 2:
    print("Please specifiy input folder")
    exit(1)


# Parameters to change
range = 500
sensitivity = 1
factor = 1.00001
number_of_buckets = 250000
eps = 0.3

idx = 0
for root, _, files in os.walk(sys.argv[1], topdown=False):
    result = pd.DataFrame()
    for name in [f for f in files if ".csv" in f and "result" not in f and "cutted" not in f]:
        fqdn = os.path.join(root, name)
        data = pd.read_csv(fqdn)
        x = data["X"].to_numpy()
        noise = data["Y"].to_numpy()

        compositions = int(re.search(r"^comps_(\d{1,3})_", name).group(1))

        l_b_a, u_b_a, l_a_b, u_a_b = compute_delta(noise, x, range, compositions, sensitivity, factor, number_of_buckets, eps)
        result.at[idx, "Compositions"] = compositions
        result.at[idx, "Delta_Lower_B_A"] = l_b_a
        result.at[idx, "Delta_Upper_B_A"] = u_b_a
        result.at[idx, "Delta_Lower_A_B"] = l_a_b
        result.at[idx, "Delta_Upper_A_B"] = u_a_b
        result.at[idx, "Utility_L1"] = np.dot(np.abs(x), noise)
        result.at[idx, "Utility_L2"] = np.sqrt(np.dot(x ** 2, noise))

        idx += 1

    result.to_csv(os.path.join(root, "result.csv"))
