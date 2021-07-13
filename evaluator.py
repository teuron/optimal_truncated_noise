# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)

import logging
import os
import parser
import sys
import time
import utils

import models
import numpy as np
import progressbar
import torch
from torch.utils.tensorboard import SummaryWriter

import logging_utils
import plot_utils
import privacy_utils
import torch_utils
import dp_preparer
from criterion import ComputeError
from privacybuckets_pt import ComputePrivacyBucketsDelta
from renyi_markov_delta import ComputeRenyiMarkovDelta
from fft_delta import ComputeFFTDelta

# Define logger
logger = logging.getLogger(__name__)

# Define device to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reset_state(seed):
    """Resets the state of the ML framework

    Args:
        pt_seed (int): Seed for the framework
    """
    # assert False, "How to reset al tensors?. reset_parameters() ? "
    # assert False, "remove all tensors from memory!"
    # I guess what you have missed here is torch.cuda.empty_cache 163. After del Tensor, call torch.cuda.empty_cache() and see whether GPU memory usage changes.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.set_deterministic(True)


def create_name(args, tensorboard=True):
    """Creates a unique name based on the programs argument

    Args:
        args (Arguments): Program Arguments
        tensorboard (bool, optional): Print "runs/" infront of the name. Defaults to True.

    Returns:
        String: Unique name
    """

    if args.noise_model == "CNNModel":
        name = f"{args.cnns}-bias-{args.bias}"
    else:
        name = f"{args.sig_num}-s-{args.scale_start}-l-{args.slope}-b-{args.bias}"

    if tensorboard:
        prefix = "runs/experiment"
    else:
        prefix = "experiment"

    return f"{prefix}-{args.noise_model}-{args.method}-{args.noise_class}-{args.mixture_q}-ut-{args.utility_weight}-lr-{args.learning_rate}-rb-{args.range_begin}-nb-{args.number_of_compositions}-{name}-ri-{args.random_init}-op-{args.optimizer}-gn-{args.gradient_noise}-\
ep-{args.eps}-es-{args.element_size}-ulf-{args.utility_loss_function}-ep-{args.epochs}-sd-{args.pt_seed}-aldu-{args.alternate_lamda_delta_update}-uwd-{args.utility_weight_decay}-uwhe-{args.utility_weight_halving_epochs}-\
uwm-{args.utility_weight_minimum}-log-{args.log_space}-lrd-{args.learning_rate_decay}"


def create_writer(args):
    """Creates a writer for the summary. If "no_tensorboard" is selected in the arguments, it creates a NullWriter

    Args:
        args (Arguments): Arguments of the Program

    Returns:
        SummaryWriter: Writer
    """
    if args.no_tensorboard:
        return logging_utils.NullWriter()

    return SummaryWriter(create_name(args))


def get_method(args, device):
    """Creates an instance of a privacy computing method

    Args:
        args (Arguments): Program Arguments; Have to contain "method", "eps", "number_of_compositions", "fastmode" and "lambda_exponent"
        device (torch.Device): Device where the computation will run on

    Raises:
        ValueError: Is raised when the provided "method" in "args" is not known.

    Returns:
        torch.nn.Module: Chosen method
    """
    if args.method == "renyi_markov":
        method_fn = ComputeRenyiMarkovDelta(args, device)
    elif args.method == "pb_ADP":
        method_fn = ComputePrivacyBucketsDelta(args, "ADP", device)
    elif args.method == "pb_PDP":
        method_fn = ComputePrivacyBucketsDelta(args, "PDP", device)
    elif args.method == "fft":
        method_fn = ComputeFFTDelta(args, device)
    else:
        raise ValueError(f"Method '{args.method}' unknown.")
    return method_fn


def execute_single_run(args):
    """Executes a single run

    Args:
        args (Arguments): Program arguments

    Returns:
        torch.tensor: predicted output
        torch.tensor: privacy loss
        [(String, np.array)]: Arrays to plot
    """

    logger.info(f"Running on Device {device}")
    if not os.path.exists(os.path.join(args.plot_dir, create_name(args, False))):
        os.makedirs(os.path.join(args.plot_dir, create_name(args, False)))
    args.plot_dir = os.path.join(args.plot_dir, create_name(args, False))

    reset_state(args.pt_seed)
    persist = False

    writer = create_writer(args)
    range_begin = args.range_begin
    x_coords = np.linspace(-range_begin, range_begin, args.element_size, endpoint=True) + 10 ** -5

    # what is the dot product of x*optimal_noise?
    optimal_noise = privacy_utils.get_theoretical_optimal_noise_x(abs(x_coords), args.eps, args.number_of_compositions)
    A, B = privacy_utils.calculate_a_b(optimal_noise, args.range_begin, args.noise_class, args.mixture_q)
    optimal_dot_product = np.dot(abs(x_coords), optimal_noise)
    logger.info("Optimal dot product x*optimal_noise = {}".format(optimal_dot_product))

    noise_model = models.__dict__[args.noise_model]
    model = noise_model(args.element_size, range_begin, args, device)
    model = model.double()
    model = model.to(device)

    # Create A/B splitter and Method
    preparer = dp_preparer.__dict__[args.noise_class]
    preparer = preparer(args.element_size, range_begin, args, device)
    preparer = preparer.double()
    preparer = preparer.to(device)

    method_fn = get_method(args, device)

    # Initialize Criterion
    criterion = ComputeError(model, preparer, args.element_size, range_begin, device, method_fn, args)
    logger.debug(criterion)
    criterion = criterion.double()
    criterion = criterion.to(device)

    lr_decay, optimizer = torch_utils.create_optimizer(args, criterion.parameters())

    # Geng et al. noise ADP PDP
    adp_delta_opt = privacy_utils.calculate_adp_delta(A, B, range_begin, args.eps)
    pdp_delta_opt = privacy_utils.calculate_pdp_delta(A, B, range_begin, args.eps)

    # Run training
    input = torch.tensor(x_coords).double()
    input = input.to(device)

    ds = {"ADP": [], "PDP": []}
    deltas = []
    utilities = []
    for step in progressbar.progressbar(range(args.epochs)):
        # Disable Fastmode for last iteration to save statistics
        if step == args.epochs - 1:
            args.fastmode = False
            criterion.fastmode = False
            method_fn.fastmode = False
            persist = True
            writer = logging_utils.FileWriter(writer, create_name(args, False), args.plot_dir)

        predicted = model(input)
        loss, privacy_loss, pdp_delta, debug_tensors, A, B = criterion(predicted, input, step)
        optimizer.zero_grad()
        loss.backward()

        # Add noise to gradient
        torch_utils.add_gradient_noise(criterion.named_parameters(), args, step)
        # Alternate lambda delta update
        torch_utils.alternate_lamda_delta_update(criterion.named_parameters(), args, step)

        optimizer.step()
        lr_decay.step()

        # Add to deltas
        ds.setdefault("ADP", []).append(privacy_utils.calculate_adp_delta(A.detach().cpu().numpy(), B.detach().cpu().numpy(), args.range_begin, args.eps))
        ds.setdefault("PDP", []).append(privacy_utils.calculate_pdp_delta(A.detach().cpu().numpy(), B.detach().cpu().numpy(), args.range_begin, args.eps))
        ds.setdefault("PredictedDelta", []).append(pdp_delta.detach().cpu().numpy())
        utilities.append(utils.calculate_utility_loss(input, predicted, "l2").detach().cpu().numpy())

        # Only log if fastmode is False
        if args.fastmode is False:
            arrays_to_plot = plot_utils.create_arrays_to_plot(predicted, x_coords, optimal_noise, pdp_delta, args)

            try:
                logging_utils.write_parameters_to_tensorboard(writer, criterion, args, step, logger)
                logging_utils.log_debug_tensor(writer, debug_tensors, step, args, predicted, adp_delta_opt, pdp_delta_opt, x_coords, pdp_delta, preparer)
                logging_utils.write_plots_to_tensorboard(writer, debug_tensors, step, args, create_name(args, False), persist, predicted, x_coords, optimal_noise, pdp_delta, A, B)
            except Exception as e:
                print(f"Could not write to tensorboard {str(e)}")

        if args.create_thesis_plots:
            deltas.append(pdp_delta.detach().cpu().numpy())

    if args.create_thesis_plots:
        plot_utils.plot_thesis_plots(deltas, args, f"{os.path.join(args.plot_dir, create_name(args, False))}_pdp.png")
        np.save(os.path.join(args.plot_dir, create_name(args, False)), {"pdp_deltas": deltas, "args": args, "optimal_noise": optimal_noise})

    lower, upper = privacy_utils.calculate_exact_upper_lower_delta_with_pb(
        A.cpu().detach().numpy(), B.cpu().detach().numpy(), args.range_begin, args.eps, args.number_of_compositions
    )
    writer.add_scalar("Probability Buckets Lower Bound for Delta A/B", lower, 0)
    writer.add_scalar("Probability Buckets Upper Bound for Delta A/B", upper, 0)

    lower, upper = privacy_utils.calculate_exact_upper_lower_delta_with_pb(
        B.cpu().detach().numpy(), A.cpu().detach().numpy(), args.range_begin, args.eps, args.number_of_compositions
    )
    writer.add_scalar("Probability Buckets Lower Bound for Delta B/A", lower, 0)
    writer.add_scalar("Probability Buckets Upper Bound for Delta B/A", upper, 0)
    return predicted, privacy_loss, arrays_to_plot, ds, utilities


def main():
    args = parser.parse_arguments()

    # Initialize logger
    logging.basicConfig()
    logger.setLevel(logging.getLevelName(args.log_level.upper()))
    logger.info("Invoked with arguments %s", str(args))

    np.set_printoptions(threshold=sys.maxsize)

    # Create plot directory
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # Perform execution
    now = time.time()

    execute_single_run(args)

    logger.info(f"Elapsed time {now-time.time()}")


if __name__ == "__main__":
    main()
