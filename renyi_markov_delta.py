# written by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reusing code written by David Sommer (ETH Zurich), Esfandiar Mohammadi (University of Lubeck) and Sheila Zingg (ETH Zurich)
import torch

ALPHA_FIRST_GUESS = 2.0
BIAS = 10 ** -4


class ComputeRenyiMarkovDelta(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.fastmode = args.fastmode
        self.alternate_lamda_delta_update = args.alternate_lamda_delta_update
        self.log_space = args.log_space

        self.eps = torch.tensor(args.eps, device=device)
        self.number_of_compositions = torch.tensor(args.number_of_compositions, device=device)
        self.lambda_exponent = torch.tensor(args.lambda_exponent, device=device)

        # Lambdas we learn
        self.lam_2 = torch.nn.Parameter(torch.tensor(ALPHA_FIRST_GUESS + 1), requires_grad=True)
        self.lam_dual_2 = torch.nn.Parameter(torch.tensor(ALPHA_FIRST_GUESS + 1), requires_grad=True)

    def forward(self, p_A_slice, p_B_slice, dist_events, dist_events_dual, step):
        # Make lam and lam_dual positive and add small bias
        lam = self.lam_2 * self.lam_2 + BIAS
        lam_dual = self.lam_dual_2 * self.lam_dual_2 + BIAS

        # if p_A_slice contains 0.0 in it, then the log is inf, which then creates nans in the backward pass
        log_A = torch.log(p_A_slice)
        log_B = torch.log(p_B_slice)

        privacy_loss = log_A - log_B
        privacy_loss_dual = -privacy_loss

        # compute the renyi-divergence for a given alpha = lam + 1:
        #   renyi_div = D_alpha(A||B)
        #             =  1/ (alpha + 1) log sum_o B(o) * exp(privacy_loss(o) * alpha)
        #             =  1/ lam log sum_o A(o) * exp(privacy_loss(o) * lam)
        #             =  1/ lam log sum_o  exp(log_A(o) + privacy_loss(o) * lam)
        # finally, we need to omit the preceding 1 / lam factor to apply the markov inequality.
        renyi_div_times_lam = torch.logsumexp(torch.add(torch.mul(privacy_loss, lam), log_A), 0)

        renyi_div_times_lam_dual = torch.logsumexp(torch.add(torch.mul(privacy_loss_dual, lam_dual), log_B), 0)

        if step % 2 == 0 and self.alternate_lamda_delta_update:
            # extend ot the required number of compositions
            markov_PDP_delta = torch.pow((renyi_div_times_lam * self.number_of_compositions) - (lam * self.eps), self.lambda_exponent)
            markov_PDP_delta_dual = torch.pow((renyi_div_times_lam_dual * self.number_of_compositions) - (lam_dual * self.eps), self.lambda_exponent)

            # Different lambda delta updates we tried
            # (D_lam x n)^k - (lam x eps))^k
            # markov_PDP_delta = torch.pow(renyi_div_times_lam * self.number_of_compositions, self.lambda_exponent) - torch.pow(lam * self.eps, self.lambda_exponent)
            # markov_PDP_delta_dual = torch.pow(renyi_div_times_lam_dual * self.number_of_compositions, self.lambda_exponent) - torch.pow(lam_dual * self.eps, self.lambda_exponent)

            # (d_lam*n)^k - log(lam*eps)*k
            # markov_PDP_delta = torch.pow(renyi_div_times_lam * self.number_of_compositions, 2 * self.lambda_exponent) - torch.log(lam * self.eps) * self.lambda_exponent
            # markov_PDP_delta_dual = (
            #    torch.pow(renyi_div_times_lam_dual * self.number_of_compositions, 2 * self.lambda_exponent) - torch.log(lam_dual * self.eps) * self.lambda_exponent
            # )

            # (d_lam*n)^k +1/(lam*eps)^k
            # markov_PDP_delta = torch.pow(renyi_div_times_lam * self.number_of_compositions, self.lambda_exponent) + 1 / torch.pow(lam * self.eps, self.lambda_exponent)
            # markov_PDP_delta_dual = torch.pow(renyi_div_times_lam_dual * self.number_of_compositions, self.lambda_exponent) + 1 / torch.pow(
            #    lam_dual * self.eps, self.lambda_exponent
            # )

            # k*(log(D_lam x n) - log(lam x eps))
            # markov_PDP_delta = self.lambda_exponent * (torch.log(renyi_div_times_lam * self.number_of_compositions) - torch.log(lam * self.eps))
            # markov_PDP_delta_dual = self.lambda_exponent * (torch.log(renyi_div_times_lam_dual * self.number_of_compositions) - torch.log(lam_dual * self.eps))
        else:
            markov_PDP_delta = (renyi_div_times_lam * self.number_of_compositions) - (lam * self.eps)
            markov_PDP_delta_dual = (renyi_div_times_lam_dual * self.number_of_compositions) - (lam_dual * self.eps)
            if not self.log_space:
                markov_PDP_delta = torch.exp(markov_PDP_delta)
                markov_PDP_delta_dual = torch.exp(markov_PDP_delta_dual)

        dist_events_comp = 1.0 - torch.pow((1.0 - dist_events), self.number_of_compositions)
        dist_events_comp_dual = 1.0 - torch.pow((1.0 - dist_events_dual), self.number_of_compositions)

        if self.fastmode:
            debug_tensors = {}
            delta = torch.exp((renyi_div_times_lam * self.number_of_compositions) - (lam * self.eps))
            delta_dual = torch.exp((renyi_div_times_lam_dual * self.number_of_compositions) - (lam_dual * self.eps))
            delta_final_right = torch.add(delta, dist_events_comp)
        else:
            # Set the right privacy error computation for tensorboard
            delta = torch.exp((renyi_div_times_lam * self.number_of_compositions) - (lam * self.eps))
            delta_dual = torch.exp((renyi_div_times_lam_dual * self.number_of_compositions) - (lam_dual * self.eps))
            delta_final_right = torch.add(delta, dist_events_comp)
            delta_final_dual = torch.add(delta_dual, dist_events_comp_dual)
            privacy_error = torch.add(delta_final_right, delta_final_dual)

            debug_tensors = {
                "lam": lam,
                "lam_dual": lam_dual,
                "renyi_div_times_lam": renyi_div_times_lam,
                "renyi_div_times_lam_dual": renyi_div_times_lam_dual,
                "privacy_loss": privacy_loss,
                "privacy_loss_dual": privacy_loss_dual,
                "log_A": log_A,
                "err_final": privacy_error,
                "delta": delta,
                "delta_dual": delta_dual,
                "delta_final": delta_final_right,
                "delta_final_dual": delta_final_dual,
            }

        return markov_PDP_delta, markov_PDP_delta_dual, dist_events_comp, dist_events_comp_dual, delta_final_right, debug_tensors
