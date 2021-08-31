# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# Computes deltas for optimality graphs
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import os  # noqa: 402
import re  # noqa: 402

import numpy as np  # noqa: 402
import privacy_utils  # noqa: 402
import pandas as pd  # noqa: 402


def main(folder, range=400, eps=0.3, element_size=60000, noise_class="SymmetricNoise"):
    x_coords = np.linspace(-range, range, element_size, endpoint=True) + 10 ** -5
    data = pd.DataFrame(columns=["Compositions", "Utility_Weights", "Delta", "Utility"])
    for root, directory, files in os.walk(folder, topdown=False):
        for name in [f for f in files if "noise.csv" in f]:
            noise = pd.read_csv(os.path.join(root, name))
            noise = noise["Y"].to_numpy()

            A, B = privacy_utils.calculate_a_b(noise, range, noise_class, 0.0)

            compositions = re.search(r"^[A-Za-z0-9_/]*comps_(\d{1,3})_", root).group(1)
            utility_weight = re.search(r"^[A-Za-z0-9_/]*uw_(\d.\d{1,10})", root).group(1)

            if "pb_adp" in root or "renyi" in root:
                delta = privacy_utils.calculate_adp_delta(A, B, range, eps)
                lower, upper = privacy_utils.calculate_exact_upper_lower_delta_with_pb(A, B, range, eps, int(compositions))
            elif "pb_pdp" in root:
                delta = privacy_utils.calculate_pdp_delta(A, B, range, eps)
                upper, lower = privacy_utils.calculate_exact_upper_lower_delta_with_pb_pdp(A, B, range, eps, int(compositions))

            if "l2" in root:
                utility = np.sqrt(np.dot(x_coords ** 2, noise))
            elif "l1" in root:
                utility = np.dot(np.abs(x_coords), noise)

            data = data.append({"Compositions": compositions, "Utility_Weights": utility_weight, "Delta": delta, "Utility": utility, "Upper": upper, "Lower": lower}, ignore_index=True)

    for comps in data["Compositions"].unique():
        data[data["Compositions"] == comps].to_csv(f"{folder}/comps_{comps}delta_utilities.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Args has to contain the folder")
    main(sys.argv[1])
