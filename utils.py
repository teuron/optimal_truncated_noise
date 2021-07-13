""" Contains utility functions"""
# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)

from collections.abc import MutableMapping
import numpy as np
import torch
import csv

# GNU Scientific Library shared object path
GSL_PATH = "/usr/lib/x86_64-linux-gnu/libgsl.so"


def get_good_factor(initial_factor, eps, buckets_half):
    """Calculates good factor for PrivacyBuckets

    Args:
        initial_factor (float]): Initial factor
        eps (float): desired eps
        buckets_half (int): Number of buckets / 2
    """
    f = np.float64(initial_factor)

    i_eps = 0
    while not (f ** i_eps >= np.exp(eps) > f ** (i_eps - 1) and i_eps <= buckets_half / 4):
        i_eps += 1
        if i_eps > buckets_half / 4:
            i_eps = 0
            f = f ** 2

    print(f"initial f\t= {initial_factor}")
    print(f"f \t\t= {f}")
    print(f"i_eps \t\t= {i_eps},\t\t\t n/4 \t= {buckets_half/4},\t\t n \t= {buckets_half}")
    print(f"f^i_eps \t= {f**i_eps},\t f^n/4 \t= {f**(buckets_half/4)},\t f^n \t= {f**buckets_half}")
    print(f"e^eps \t\t= {np.exp(eps)}")

    return f


def calculate_utility_loss(x_coords, p, utility_loss_function):
    """Calculates the utility loss for a given input

    Args:
        x_coords ([torch.DoubleTensor]): Input coordinates
        p ([torch.DoubleTensor]): Predicted coordinates
        utility_loss_function (String): Either L1 or L2 loss

    Raises:
        ValueError: If utility_loss_function is not known

    Returns:
        [torch.DoubleTensor]: Utility loss
    """
    if utility_loss_function == "l2":  # 2nd absolute moment
        return torch.sqrt(torch.dot(x_coords ** 2, p))
    elif utility_loss_function == "l1":
        return torch.dot(torch.abs(x_coords), p)
    else:
        raise ValueError(f"Utility loss function '{utility_loss_function}' not known")


class TensorRetrieveDict(MutableMapping):
    """A dictionary that allows to retreive the values of any tensors stored in them.

    Args:
        MutableMapping (MutableMapping): Class
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        tensor_item = self.store[self.__keytransform__(key)]
        item = tensor_item.clone().cpu().data.numpy()
        return item

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key


def write_epsilon_delta(filename, epsilons, upper, lower, dist_events):
    """Writes epsilon, deltas and distevents to a CSV file

    Args:
        filename (String): Filename to write to
        epsilons ([Float]): List of epsilons
        deltas ([Float]): List of deltas
        dist_events ([Float]): List of distinguishing events
    """
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Epsilon", "Upper", "Lower", "Dist_events"])
        writer.writerows(list(zip(epsilons, upper, lower, dist_events)))


def write_noise(filename, noise, x_coords):
    """Writes noise and x coordinates to a csv file

    Args:
        filename (String): Filename to write to
        noise (nd.array): Generated Noise
        x_coords (nd.array): X-axis discretization
    """
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["X", "Y"])
        writer.writerows(list(zip(x_coords.tolist(), noise.tolist())))


def write_delta_utility(filename, utility_weights, delta, utility):
    """Writes the resulting delta and utility from a starting utility weight to a file

    Args:
        filename (String): Filename to write to
        utility_weights ([Integer]): List of utility_weights
        delta ([Float]): List of resulting deltas
        utility ([Float]): List of resulting utility
    """
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Utility_Weights", "Delta", "Utility"])
        writer.writerows(list(zip(utility_weights, delta, utility)))


def write_delta(filename, delta, dist_events):
    """Writes delta and distinguishing events to a csv file

    Args:
        filename (String): Filename to write to
        delta (nd.array): Generated delta
        dist_events (nd.array): Corresponding distinguishing events
    """
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Delta", "Dist-events"])
        writer.writerows(list(zip(delta, dist_events)))


def write_delta_adp_pdp(filename, epochs, adp, pdp):
    """Writes delta ADP/PDP and epochs to a csv file

    Args:
        filename (String): Filename to write to
        epochs (nd.array): Corresponding epochs
        adp (nd.array): Generated delta ADP
        pdp (nd.array): Generated delta PDP
    """
    with open(filename, mode="w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Epochs", "ADP", "PDP"])
        writer.writerows(list(zip(epochs, adp, pdp)))
