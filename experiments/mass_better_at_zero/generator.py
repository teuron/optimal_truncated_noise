import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import csv

ws = np.exp(np.linspace(np.log(10 ** -20), np.log(1), 50))
stds = np.exp(np.linspace(np.log(1), np.log(100), 10))
x_0 = 0
x_1 = 1
mean = 0
utilities = ["l1", "l2"]
tolerance = 10 ** -20
eps = 0.3


def write_to_csv(filename, ws, tipping_points):
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["W", "Tipping_Point"])
        writer.writerows(list(zip(ws, tipping_points)))


def calculate_utility_loss(x, utility_loss_function):
    if utility_loss_function == "l2":
        return x * x
    elif utility_loss_function == "l1":
        return x


def privacy_loss(x_0, x_1, std, x):
    return ((x_1 - x_0) * (x_0 + x_1 - 2 * x)) / (2 * std * std)


def to_optimize(x, w_std, eps, utility, x_0, x_1, std):
    utility = calculate_utility_loss(x, utility)
    res = (w_std * utility) - np.minimum(1, np.exp(eps - privacy_loss(x_0, x_1, std, x)))
    return res


for utility_functions in utilities:
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for std in stds:
        points, w_list = [], []
        for w in ws:
            w_std = w * std
            try:
                tipping_point = optimize.bisect(f=to_optimize, a=0, b=10000, xtol=tolerance, args=(w_std, eps, utility_functions, x_0, x_1, std))
                points.append(tipping_point)
                w_list.append(w)
            except:  # noqa: E722
                tipping_point = -1
            print("Result", tipping_point, w, std, utility_functions)
        ax.plot(w_list, points)
        write_to_csv(f"result_std_{std}_uti_{utility_functions}.csv", w_list, points)
    plt.show()
