# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)
import torch

from utils import calculate_utility_loss
import torch_utils


class ComputeError(torch.nn.Module):
    """Computes the loss based on the utility and privacy"""

    def __init__(self, model, preparer, element_size, range_begin, device, method_fn, args):
        super().__init__()
        self.fastmode = args.fastmode
        self.log_space = args.log_space
        self.model = model
        self.element_size = element_size
        self.Delta = int(element_size / (2 * range_begin))
        self.method_fn = method_fn
        self.utility_weight = torch.tensor(args.utility_weight, device=device)
        # call register buffer for utility_weight?
        self.utility_loss_function = args.utility_loss_function
        self.utility_weight_decay = args.utility_weight_decay
        self.utility_weight_halving_epochs = torch.tensor(args.utility_weight_halving_epochs, device=device)
        self.utility_weight_minimum = torch.tensor(args.utility_weight_minimum, device=device)
        self.base = torch.tensor(2.0, device=device)
        self.device = device
        self.preparer = preparer
        self.min_of_two_delta = args.min_of_two_delta
        self.dp_sgd = args.dp_sgd

    def forward(self, p, x_coords, step):
        # Calculate A, B
        p_A_slice, p_B_slice, dist_events, dist_events_dual, A, B = self.preparer(p)

        # Calculate delta
        delta, delta_dual, dist_events_comp, dist_events_comp_dual, delta_final_right, debug_method = self.method_fn(p_A_slice, p_B_slice, dist_events, dist_events_dual, step)

        # Calculate utility loss
        utility_loss = calculate_utility_loss(x_coords, p, self.utility_loss_function)

        # Calculate utility weight
        if self.utility_weight_decay:
            ut = self.utility_weight / torch.pow(self.base, torch.tensor(step, device=self.device) / self.utility_weight_halving_epochs)
            ut = torch.max(ut, self.utility_weight_minimum)
        else:
            ut = self.utility_weight

        # Add all together in logspace or normalspace
        if self.log_space:
            delta_final = torch_utils.log_add(delta, torch.log(dist_events_comp))
            delta_final_dual = torch_utils.log_add(delta_dual, torch.log(dist_events_comp_dual))

            if self.min_of_two_delta:
                privacy_error = torch.minimum(delta_final, delta_final_dual)
            else:
                privacy_error = torch_utils.log_add(delta_final, delta_final_dual)

            loss = torch_utils.log_add(torch.log(utility_loss) + torch.log(ut), privacy_error)
        else:
            delta_final = torch.add(delta, dist_events_comp)
            delta_final_dual = torch.add(delta_dual, dist_events_comp_dual)
            if self.dp_sgd:
                if self.min_of_two_delta:
                    privacy_error = delta_final
                else:
                    privacy_error = delta_final_dual
            else:
                privacy_error = torch.add(delta_final, delta_final_dual)
            loss = utility_loss * ut + privacy_error

        if self.fastmode:
            debug_tensors = {}
        else:
            debug_tensors = {
                "p": p,
                "p_A_slice": p_A_slice,
                "p_B_slice": p_B_slice,
                "dist_events": dist_events,
                "dist_events_dual": dist_events_dual,
                "delta": delta,
                "delta_dual": delta_dual,
                "dist_events_comp": dist_events_comp,
                "dist_events_comp_dual": dist_events_comp_dual,
                "delta_final": delta_final,
                "delta_final_dual": delta_final_dual,
                "err_final": privacy_error,
                "utility_loss": utility_loss,
                "utility_loss_plus_err": loss,
                "privacy_error_log": privacy_error,
                "utw": ut,
            }

            debug_tensors.update(debug_method)

        return loss, privacy_error, delta_final_right, debug_tensors, A, B
