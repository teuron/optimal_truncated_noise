""" Contains utility functions for logging"""
# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)

import numpy as np
import torch
from matplotlib import pyplot as plt
import utils
import gc
import os
import plot_utils
import privacy_utils


def log_debug_tensor(writer, debug_tensors, step, args, predicted, adp_delta, pdp_delta, x_coords, markov_pdp_delta, preparer):
    """Writes debug tensor to Tensorboard if args.fastmode is False

    Args:
        writer (SummaryWriter): Tensorboard instance
        debug_tensors (dict(nn.array)): tensors to write to tensorboard
        step (int): epoch we are in
        args (Arguments): Program arguments
        predicted (torch.DoubleTensor): Predicted Noise
        adp_delta (double): ADP-Delta of Optimal Noise
        pdp_delta (double): PDP-Delta of Optimal Noise
        x_coords ([double]): Input x-coordinates
    """
    if not args.fastmode:
        d = utils.TensorRetrieveDict(debug_tensors)

        writer.add_scalar("Loss/Combined", d["utility_loss_plus_err"], step)
        writer.add_scalar("Loss/Utility", d["utility_loss"], step)
        writer.add_scalar("Loss/Privacy", d["err_final"], step)
        writer.add_scalar("Loss/Privacy_LOGSPACE", d["privacy_error_log"], step)
        writer.add_scalar("Loss/ADP-Delta", adp_delta, step)
        writer.add_scalar("Loss/PDP-Delta", pdp_delta, step)
        writer.add_scalar(f"Delta/A/B({args.eps})", d["delta_final"], step)
        writer.add_scalar(f"Delta/B/A({args.eps})", d["delta_final_dual"], step)
        writer.add_scalar(f"Delta/A/B({args.eps}) + B/A({args.eps})", d["err_final"], step)
        writer.add_scalar("Dist. events", d["dist_events_comp"], step)
        writer.add_scalar("Dist. events dual", d["dist_events_comp_dual"], step)
        writer.add_scalar("Dist. events/Before Comp", d["dist_events"], step)
        writer.add_scalar("Dist. events dual/Before Comp", d["dist_events_dual"], step)
        writer.add_histogram("Result", predicted.clone().cpu().data.numpy(), step)
        writer.add_scalar(f"Delta/Priv_Gauss{args.eps}", privacy_utils.calculate_adp_priv_gauss(markov_pdp_delta, preparer, x_coords, args.range_begin, args), step)

        if args.method == "renyi_markov":
            alpha = d["lam"] - 1
            alpha_dual = d["lam_dual"] - 1
            renyi_div = d["renyi_div_times_lam"] / (alpha + 1)
            renyi_div_dual = d["renyi_div_times_lam_dual"] / (alpha_dual + 1)
            writer.add_scalar("Loss/Utility_Weight", d["utw"], step)
            writer.add_scalar("Renyi/A/B/Alpha", alpha, step)
            writer.add_scalar("Renyi/A/B/Renyi", renyi_div, step)
            writer.add_scalar("Renyi/B/A/Alpha", alpha_dual, step)
            writer.add_scalar("Renyi/B/A/Renyi", renyi_div_dual, step)
            writer.add_scalar("Renyi/LAM", d["lam"], step)
            writer.add_scalar("Renyi/LAM_DUAL", d["lam_dual"], step)
            writer.add_scalar("Renyi/DIV/TIMES/LAM/Number_Compositions", d["renyi_div_times_lam"] * args.number_of_compositions, step)
            writer.add_scalar("Renyi/DIV/TIMES/LAM/Number_CompositionsMINUSLAMEPS", d["renyi_div_times_lam"] * args.number_of_compositions - d["lam"] * args.eps, step)
            writer.add_scalar("Renyi/DIV/TIMES/LAM/Number_Compositions", d["renyi_div_times_lam"] * args.number_of_compositions, step)
            writer.add_scalar("Renyi/DIV/TIMES/LAM_DUAL/Number_CompositionsMINUSLAMEPS", d["renyi_div_times_lam_dual"] * args.number_of_compositions - d["lam"] * args.eps, step)


def write_parameters_to_tensorboard(writer, criterion, args, step, logger):
    """Writes model parameters to tensorboard

    Args:
        writer (SummaryWriter): Tensorboard instance
        criterion (torch.nn.Module): Loss instance
        args (Arguments): Program arguments
        step (int): epoch we are in
        logger (logging.logger): Error logger
    """
    if not args.fastmode:
        for n, p in criterion.named_parameters():
            try:
                if p.shape == torch.Size([]):
                    writer.add_scalar(n + "/Value", p.clone().cpu().data.numpy(), step)
                else:
                    writer.add_histogram(n + "/Value", p.clone().cpu().data.numpy(), step)

                if p.grad is not None:
                    try:
                        writer.add_histogram(n + "/Gradient", p.grad.clone().cpu().data.numpy(), step)
                        writer.add_scalar(n + "/Gradient Norm", torch.norm(p.grad.data).item(), step)
                    except ValueError as e:
                        logger.error(e)
                        writer.flush()
            except:  # noqa 722
                print("Could not write to tensorboard")


def write_plots_to_tensorboard(writer, debug_tensors, step, args, name, persist, predicted, x_coords, optimal_noise, pdp_delta, A, B):
    """Writes plots to Tensorboard if args.fastmode is False

    Args:
        writer (SummaryWriter): Tensorboard instance
        debug_tensors (dict(nn.array)): tensors to write to tensorboard
        step (int): epoch we are in
        args (Arguments): Program arguments
        name (String): Name of the plot
        persist (Boolean): Save to disc
        predicted (torch.DoubleTensor): Predicted noise
        x_coords ([double]): Input coordinates
        optimal_noise ([double]): Optimal noise
        privacy_loss ([torch.DoubleTensor]): Privacy loss
    """
    if not args.fastmode and (step % 100 == 0 or step == args.epochs - 1) or args.dump_data:
        d = utils.TensorRetrieveDict(debug_tensors)
        persist = persist or args.dump_data

        if args.method == "renyi_markov":
            try:
                # figure, best_lam = plot_utils.plot_markov_pdp(d["privacy_loss"], d["log_A"], args.number_of_compositions, args.eps, d["lam"], step)
                # writer.add_figure("Renyi/MARKOV_PDP", figure, global_step=step)
                # writer.add_scalars("Renyi/MARKOV_DELTA_LAM", {"delta": d["err_final"], "current_lam": d["lam"], "best_lam": best_lam}, step)
                print("Skipping DELTA LAM")
            except:  # noqa: E722
                print("Error in optimal LAM calculation")

        arrays_to_plot = plot_utils.create_arrays_to_plot(predicted, x_coords, optimal_noise, pdp_delta, args)

        figure = plot_utils.save_arrays_as_plot(arrays_to_plot, os.path.join(args.plot_dir, f"{name}.png"), args.element_size, args.range_begin, persist=persist)
        writer.add_figure("Plot/Normal", figure, global_step=step)
        figure.clf()

        log_figure = plot_utils.save_arrays_as_log_plot(arrays_to_plot, os.path.join(args.plot_dir, f"{name}_log.png"), args.element_size, args.range_begin, persist=persist)
        writer.add_figure("Plot/Logarithm", log_figure, global_step=step)
        log_figure.clf()

        arrays_to_plot = [[np.concatenate((d["privacy_loss"], [0] * (args.element_size - len(d["privacy_loss"])))), "privacy loss"]]
        figure = plot_utils.save_arrays_as_plot(arrays_to_plot, os.path.join(args.plot_dir, f"{name}_privacy_loss.png"), args.element_size, args.range_begin, persist=persist)
        writer.add_figure("Renyi/PrivacyLoss", figure, global_step=step)
        figure.clf()

        arrays_to_plot = [[np.concatenate(([0] * (args.element_size - len(d["privacy_loss"])), d["privacy_loss_dual"])), "privacy loss_dual"]]
        figure = plot_utils.save_arrays_as_plot(arrays_to_plot, os.path.join(args.plot_dir, f"{name}_privacy_loss_dual.png"), args.element_size, args.range_begin, persist=persist)
        writer.add_figure("Renyi/PrivacyLossDual", figure, global_step=step)
        figure.clf()

        try:
            A = A.detach().cpu().numpy()
            B = B.detach().cpu().numpy()
            figure = plot_utils.plot_pld(B, A, args, os.path.join(args.plot_dir, f"{name}_pld_B_A.png"), persist=persist)
            writer.add_figure("PLD/PrivacyLossDistribution_B_A", figure, global_step=step)
            figure.clf()

            figure = plot_utils.plot_pld(B, A, args, os.path.join(args.plot_dir, f"{name}_pld_B_A_log.png"), persist=persist, plt_func=plt.semilogy)
            writer.add_figure("PLD/PrivacyLossDistribution_B_A_Log", figure, global_step=step)
            figure.clf()
            writer.flush()

            figure = plot_utils.plot_pld(A, B, args, os.path.join(args.plot_dir, f"{name}_pld_A_B.png"), persist=persist)
            writer.add_figure("PLD/PrivacyLossDistribution_A_B", figure, global_step=step)
            figure.clf()

            figure = plot_utils.plot_pld(A, B, args, os.path.join(args.plot_dir, f"{name}_pld_A_B_log.png"), persist=persist, plt_func=plt.semilogy)
            writer.add_figure("PLD/PrivacyLossDistribution_A_B_Log", figure, global_step=step)
            figure.clf()
        except:  # noqa: E722
            print("Error in PLD")

        writer.flush()

        plt.close("all")
        gc.collect()


class NullWriter:
    """A tensorboard writer implementation, which does nothing"""

    def flush(self):
        pass

    def add_scalar(self, name, value, step):
        pass

    def add_scalars(self, name, value, step):
        pass

    def add_histogram(self, name, value, step):
        pass

    def add_figure(self, name, value, global_step):
        pass


class FileWriter:
    """A File writer implementation"""

    def __init__(self, writer, name, plot_dir):
        self.writer = writer
        self.name = name
        self.filedir = plot_dir
        self.filename = os.path.join(plot_dir, "summary.txt")

    def flush(self):
        self.writer.flush()
        pass

    def add_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)
        with open(self.filename, "a+") as f:
            f.write(name)
            f.write("\n")
            f.write(str(value))
            f.write("\n")
        name = name.replace("/", "")
        np.save(os.path.join(self.filedir, name + ".npy"), value)

    def add_scalars(self, name, value, step):
        self.writer.add_scalars(name, value, step)
        with open(self.filename, "a+") as f:
            f.write(name)
            f.write("\n")
            f.write(str(value))
            f.write("\n")
        name = name.replace("/", "")
        np.savez(os.path.join(self.filedir, name + ".npy"), value)

    def add_histogram(self, name, value, step):
        self.writer.add_histogram(name, value, step)
        with open(self.filename, "a+") as f:
            f.write(name)
            f.write("\n")
            f.write(str(value))
            f.write("\n")
        name = name.replace("/", "")
        np.save(os.path.join(self.filedir, name + ".npy"), value)

    def add_figure(self, name, value, global_step):
        self.writer.add_figure(name, value, global_step)
